"""
Fine-tune CodeBERT for AI code detection (SemEval Task 13A)
============================================================
Optimized for Kaggle P100 GPU with ~3h time budget.

Strategy:
  - CodeBERT (125M params) — fast to fine-tune, good at code understanding
  - Train on subsample (50K-100K) for speed, full data if time allows
  - max_length=256 (balance speed vs context)
  - gradient accumulation to simulate larger batch
  - fp16 mixed precision
  - Macro F1 as metric (matches competition)

Usage on Kaggle:
  python finetune_codebert.py \
    --train_data /kaggle/input/competitions/.../train.parquet \
    --val_data /kaggle/input/competitions/.../validation.parquet \
    --test_data /kaggle/input/competitions/.../test.parquet \
    --output_dir /kaggle/working/codebert_model \
    --submission_out /kaggle/working/submission.csv \
    --max_samples 100000 --epochs 3 --batch_size 32
"""
import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="task_A/train.parquet")
    parser.add_argument("--val_data", default="task_A/validation.parquet")
    parser.add_argument("--test_data", default=None)
    parser.add_argument("--output_dir", default="./codebert_model")
    parser.add_argument("--submission_out", default="submission_codebert.csv")
    # Model
    parser.add_argument("--model_name", default="microsoft/codebert-base",
                        help="Pretrained model (codebert-base, graphcodebert, codet5-small...)")
    parser.add_argument("--max_length", type=int, default=256)
    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Subsample train data for speed (e.g. 50000, 100000)")
    parser.add_argument("--val_samples", type=int, default=10000,
                        help="Subsample val data for faster eval")
    # Loss
    parser.add_argument("--loss_type", default="ce",
                        choices=["ce", "focal", "label_smooth", "focal_smooth"],
                        help="ce=CrossEntropy, focal=FocalLoss(gamma=2), "
                             "label_smooth=CE+smoothing, focal_smooth=both")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    # Inference
    parser.add_argument("--infer_batch_size", type=int, default=128)
    parser.add_argument("--model_path", default=None,
                        help="Load pre-trained model for inference only")
    # Active Learning
    parser.add_argument("--active_learning", action="store_true",
                        help="Enable active learning: train on seed, select uncertain, retrain")
    parser.add_argument("--al_seed_size", type=int, default=50000,
                        help="Initial seed set size for active learning")
    parser.add_argument("--al_query_size", type=int, default=50000,
                        help="Number of uncertain samples to add each round")
    parser.add_argument("--al_rounds", type=int, default=3,
                        help="Number of active learning rounds")
    parser.add_argument("--al_seed_epochs", type=int, default=1,
                        help="Epochs for seed round (fast)")
    return parser.parse_args()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    macro_f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {'macro_f1': macro_f1, 'accuracy': acc}


class FocalLossTrainer:
    """
    Mixin providing focal loss compute_loss.

    Focal Loss: FL = -alpha * (1-p_t)^gamma * log(p_t)
    - Easy samples (p_t ~1) → weight ~0, bỏ qua
    - Hard samples (p_t ~0.5) → weight ~1, focus vào
    - gamma=0 → standard CE, gamma=2 → default focal

    Tại sao giúp distribution shift:
      Model confident đúng trên Python (easy) → weight giảm
      Model uncertain trên non-Python-like patterns (hard) → weight tăng
      → Buộc model học features tổng quát hơn, không chỉ Python-specific
    """
    focal_gamma = 2.0
    label_smoothing = 0.0
    use_focal = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.use_focal:
            ce = torch.nn.functional.cross_entropy(
                logits, labels, reduction='none',
                label_smoothing=self.label_smoothing)
            pt = torch.exp(-ce)  # probability of correct class
            focal_weight = (1 - pt) ** self.focal_gamma
            loss = (focal_weight * ce).mean()
        else:
            loss = torch.nn.functional.cross_entropy(
                logits, labels,
                label_smoothing=self.label_smoothing)

        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()

    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding,
        EarlyStoppingCallback,
    )
    from datasets import Dataset

    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # Inference-only mode: load saved model and predict test
    # =========================================================================
    if args.model_path:
        logger.info(f"Loading saved model from {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        if args.test_data:
            logger.info(f"Running inference on {args.test_data}...")
            test_df = pd.read_parquet(args.test_data)
            test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.RangeIndex(len(test_df))
            test_ds = Dataset.from_pandas(test_df[['code']].reset_index(drop=True))

            def tokenize_fn(examples):
                return tokenizer(examples['code'], truncation=True, max_length=args.max_length)

            test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=['code'])

            training_args = TrainingArguments(
                output_dir=args.output_dir, per_device_eval_batch_size=args.infer_batch_size,
                fp16=torch.cuda.is_available(), report_to="none",
            )
            trainer = Trainer(model=model, args=training_args,
                              processing_class=tokenizer,
                              data_collator=DataCollatorWithPadding(tokenizer=tokenizer))
            preds = trainer.predict(test_ds)
            y_pred = np.argmax(preds.predictions, axis=1)

            print(f"\nTest prediction distribution:")
            print(f"  Human: {(y_pred==0).sum()} ({(y_pred==0).mean()*100:.1f}%)")
            print(f"  AI:    {(y_pred==1).sum()} ({(y_pred==1).mean()*100:.1f}%)")

            sub = pd.DataFrame({"ID": test_ids, "label": y_pred})
            sub.to_csv(args.submission_out, index=False)
            logger.info(f"Submission saved to {args.submission_out} ({len(sub)} rows)")

        if args.val_data and os.path.exists(args.val_data):
            logger.info(f"Evaluating on {args.val_data}...")
            val_df = pd.read_parquet(args.val_data)[['code', 'label']].dropna()
            val_df['label'] = val_df['label'].astype(int)
            val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

            def tokenize_fn_val(examples):
                return tokenizer(examples['code'], truncation=True, max_length=args.max_length)

            val_ds = val_ds.map(tokenize_fn_val, batched=True, remove_columns=['code'])
            val_ds = val_ds.rename_column('label', 'labels')

            training_args = TrainingArguments(
                output_dir=args.output_dir, per_device_eval_batch_size=args.infer_batch_size,
                fp16=torch.cuda.is_available(), report_to="none",
            )
            trainer = Trainer(model=model, args=training_args,
                              processing_class=tokenizer,
                              data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                              compute_metrics=compute_metrics)
            results = trainer.evaluate(val_ds)
            print(f"\nVal Macro F1: {results['eval_macro_f1']:.4f}")
            print(f"Val Accuracy: {results['eval_accuracy']:.4f}")

        return

    # =========================================================================
    # 1. Load data
    # =========================================================================
    logger.info(f"Loading train: {args.train_data}")
    train_df = pd.read_parquet(args.train_data)
    train_df = train_df[['code', 'label']].dropna()
    train_df['label'] = train_df['label'].astype(int)

    if args.max_samples and args.max_samples < len(train_df):
        logger.info(f"Subsampling train to {args.max_samples} (stratified)...")
        train_df, _ = train_test_split(
            train_df, train_size=args.max_samples,
            stratify=train_df['label'], random_state=42
        )

    logger.info(f"Loading val: {args.val_data}")
    val_df = pd.read_parquet(args.val_data)
    val_df = val_df[['code', 'label']].dropna()
    val_df['label'] = val_df['label'].astype(int)

    if args.val_samples and args.val_samples < len(val_df):
        val_df, _ = train_test_split(
            val_df, train_size=args.val_samples,
            stratify=val_df['label'], random_state=42
        )

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
    logger.info(f"Train labels: {train_df['label'].value_counts().to_dict()}")

    # =========================================================================
    # 2. Tokenizer + Model
    # =========================================================================
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples['code'], truncation=True, max_length=args.max_length
        )

    # =========================================================================
    # 3. Prepare HF datasets
    # =========================================================================
    logger.info("Tokenizing datasets...")
    train_ds = Dataset.from_pandas(train_df[['code', 'label']].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[['code', 'label']].reset_index(drop=True))

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=['code'])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=['code'])

    train_ds = train_ds.rename_column('label', 'labels')
    val_ds = val_ds.rename_column('label', 'labels')

    # =========================================================================
    # 4. Training
    # =========================================================================
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(train_ds) // effective_batch
    eval_steps = max(steps_per_epoch // 3, 100)  # eval ~3 times per epoch
    save_steps = eval_steps

    logger.info(f"Effective batch size: {effective_batch}")
    logger.info(f"Steps per epoch: {steps_per_epoch}, eval every {eval_steps} steps")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        dataloader_num_workers=2,
        report_to="none",
    )

    # Build trainer with appropriate loss
    use_focal = args.loss_type in ("focal", "focal_smooth")
    use_smooth = args.loss_type in ("label_smooth", "focal_smooth")

    if use_focal or use_smooth:
        # Dynamic subclass: Trainer + FocalLossTrainer mixin
        CustomTrainer = type("CustomTrainer", (FocalLossTrainer, Trainer), {})
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer.use_focal = use_focal
        trainer.focal_gamma = args.focal_gamma
        trainer.label_smoothing = args.label_smoothing if use_smooth else 0.0
        logger.info(f"Loss: focal={use_focal}(gamma={trainer.focal_gamma}), "
                     f"smoothing={trainer.label_smoothing}")
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        logger.info("Loss: standard CrossEntropy")

    if args.active_learning:
        # =================================================================
        # ACTIVE LEARNING LOOP
        # =================================================================
        full_train_df = train_df.copy().reset_index(drop=True)

        # Seed: random subset
        seed_idx = full_train_df.sample(
            n=min(args.al_seed_size, len(full_train_df)),
            random_state=42, replace=False
        ).index.tolist()
        selected_idx = set(seed_idx)
        pool_idx = set(range(len(full_train_df))) - selected_idx

        for al_round in range(args.al_rounds):
            round_df = full_train_df.loc[list(selected_idx)].reset_index(drop=True)
            logger.info(f"\n{'='*60}")
            logger.info(f"ACTIVE LEARNING ROUND {al_round+1}/{args.al_rounds}")
            logger.info(f"Training set: {len(round_df)} samples")
            logger.info(f"Pool remaining: {len(pool_idx)} samples")
            logger.info(f"{'='*60}")

            # Prepare dataset for this round
            round_ds = Dataset.from_pandas(round_df[['code', 'label']].reset_index(drop=True))
            round_ds = round_ds.map(tokenize_fn, batched=True, remove_columns=['code'])
            round_ds = round_ds.rename_column('label', 'labels')

            # Reset model for each round (or continue from last)
            if al_round > 0:
                model = AutoModelForSequenceClassification.from_pretrained(
                    args.output_dir, num_labels=2
                )

            # Adjust epochs: seed round uses fewer epochs
            round_epochs = args.al_seed_epochs if al_round == 0 else args.epochs

            round_args = TrainingArguments(
                output_dir=args.output_dir,
                num_train_epochs=round_epochs,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size * 2,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=args.lr,
                warmup_ratio=args.warmup_ratio,
                weight_decay=0.01,
                fp16=torch.cuda.is_available(),
                logging_steps=50,
                eval_strategy="steps",
                eval_steps=max(len(round_ds) // (args.batch_size * args.grad_accum * 3), 100),
                save_strategy="steps",
                save_steps=max(len(round_ds) // (args.batch_size * args.grad_accum * 3), 100),
                load_best_model_at_end=True,
                metric_for_best_model="macro_f1",
                greater_is_better=True,
                save_total_limit=2,
                dataloader_num_workers=2,
                report_to="none",
            )

            if use_focal or use_smooth:
                CustomTrainer = type("CustomTrainer", (FocalLossTrainer, Trainer), {})
                round_trainer = CustomTrainer(
                    model=model, args=round_args,
                    train_dataset=round_ds, eval_dataset=val_ds,
                    processing_class=tokenizer,
                    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                    compute_metrics=compute_metrics,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                )
                round_trainer.use_focal = use_focal
                round_trainer.focal_gamma = args.focal_gamma
                round_trainer.label_smoothing = args.label_smoothing if use_smooth else 0.0
            else:
                round_trainer = Trainer(
                    model=model, args=round_args,
                    train_dataset=round_ds, eval_dataset=val_ds,
                    processing_class=tokenizer,
                    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                    compute_metrics=compute_metrics,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                )

            round_trainer.train()
            round_trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Query pool for uncertain samples
            if al_round < args.al_rounds - 1 and len(pool_idx) > 0:
                logger.info("Querying pool for uncertain samples...")
                pool_list = sorted(pool_idx)
                pool_df = full_train_df.loc[pool_list].reset_index(drop=True)
                pool_ds = Dataset.from_pandas(pool_df[['code']].reset_index(drop=True))
                pool_ds = pool_ds.map(tokenize_fn, batched=True, remove_columns=['code'])

                pool_preds = round_trainer.predict(pool_ds)
                from scipy.special import softmax as sp_softmax
                pool_proba = sp_softmax(pool_preds.predictions, axis=1)[:, 1]

                # Uncertainty = closeness to 0.5
                uncertainty = np.abs(pool_proba - 0.5)
                # Select most uncertain (smallest distance from 0.5)
                n_query = min(args.al_query_size, len(pool_list))
                uncertain_order = np.argsort(uncertainty)[:n_query]
                new_idx = [pool_list[i] for i in uncertain_order]

                selected_idx.update(new_idx)
                pool_idx -= set(new_idx)

                logger.info(f"Added {len(new_idx)} uncertain samples "
                           f"(uncertainty range: {uncertainty[uncertain_order[0]]:.4f} - "
                           f"{uncertainty[uncertain_order[-1]]:.4f})")
                logger.info(f"Proba distribution of selected: "
                           f"mean={pool_proba[uncertain_order].mean():.3f}, "
                           f"std={pool_proba[uncertain_order].std():.3f}")

        # Use last round's trainer for inference
        trainer = round_trainer
        logger.info(f"Active learning complete. Final training set: {len(selected_idx)} samples")
    else:
        logger.info("Starting training...")
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")

    # =========================================================================
    # 5. Full validation evaluation
    # =========================================================================
    logger.info("Evaluating on full validation set...")
    full_val_df = pd.read_parquet(args.val_data)[['code', 'label']].dropna()
    full_val_df['label'] = full_val_df['label'].astype(int)
    full_val_ds = Dataset.from_pandas(full_val_df.reset_index(drop=True))
    full_val_ds = full_val_ds.map(tokenize_fn, batched=True, remove_columns=['code'])
    full_val_ds = full_val_ds.rename_column('label', 'labels')

    preds = trainer.predict(full_val_ds)
    from scipy.special import softmax
    val_proba = softmax(preds.predictions, axis=1)[:, 1]
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = preds.label_ids

    # Save val probabilities for post-processing
    val_proba_path = os.path.join(args.output_dir, "val_proba.npy")
    np.save(val_proba_path, val_proba)
    logger.info(f"Val probabilities saved to {val_proba_path}")

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS (CodeBERT)")
    print("=" * 60)
    print(f"Macro F1:    {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred))

    # =========================================================================
    # 6. Test inference
    # =========================================================================
    if args.test_data:
        logger.info(f"Running inference on {args.test_data}...")
        test_df = pd.read_parquet(args.test_data)
        test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.RangeIndex(len(test_df))

        test_ds = Dataset.from_pandas(test_df[['code']].reset_index(drop=True))
        test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=['code'])

        # Predict in batches
        preds = trainer.predict(test_ds)
        test_proba = softmax(preds.predictions, axis=1)[:, 1]
        y_pred_test = np.argmax(preds.predictions, axis=1)

        # Save test probabilities for post-processing
        test_proba_path = os.path.join(args.output_dir, "test_proba.npy")
        np.save(test_proba_path, test_proba)
        logger.info(f"Test probabilities saved to {test_proba_path}")

        print(f"\nTest prediction distribution (threshold=0.5):")
        print(f"  Human: {(y_pred_test==0).sum()} ({(y_pred_test==0).mean()*100:.1f}%)")
        print(f"  AI:    {(y_pred_test==1).sum()} ({(y_pred_test==1).mean()*100:.1f}%)")

        # Also show distribution at different thresholds
        print(f"\nDistribution at various thresholds:")
        for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            y_t = (test_proba >= t).astype(int)
            print(f"  t={t:.1f}: Human={( y_t==0).sum()} ({(y_t==0).mean()*100:.1f}%), "
                  f"AI={(y_t==1).sum()} ({(y_t==1).mean()*100:.1f}%)")

        sub = pd.DataFrame({"ID": test_ids, "label": y_pred_test})
        sub.to_csv(args.submission_out, index=False)
        logger.info(f"Submission saved to {args.submission_out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
