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
    # Inference
    parser.add_argument("--infer_batch_size", type=int, default=128)
    parser.add_argument("--model_path", default=None,
                        help="Load pre-trained model for inference only")
    return parser.parse_args()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    macro_f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {'macro_f1': macro_f1, 'accuracy': acc}


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
    # 1. Load data
    # =========================================================================
    logger.info(f"Loading train: {args.train_data}")
    train_df = pd.read_parquet(args.train_data)
    train_df = train_df[['code', 'label']].dropna()
    train_df['label'] = train_df['label'].astype(int)

    if args.max_samples and args.max_samples < len(train_df):
        logger.info(f"Subsampling train to {args.max_samples} (stratified)...")
        from sklearn.model_selection import train_test_split
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
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = preds.label_ids

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
        y_pred_test = np.argmax(preds.predictions, axis=1)

        print(f"\nTest prediction distribution:")
        print(f"  Human: {(y_pred_test==0).sum()} ({(y_pred_test==0).mean()*100:.1f}%)")
        print(f"  AI:    {(y_pred_test==1).sum()} ({(y_pred_test==1).mean()*100:.1f}%)")

        sub = pd.DataFrame({"ID": test_ids, "label": y_pred_test})
        sub.to_csv(args.submission_out, index=False)
        logger.info(f"Submission saved to {args.submission_out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
