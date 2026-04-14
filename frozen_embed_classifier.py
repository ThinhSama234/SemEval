"""
Frozen CodeBERT/UniXcoder Embedding + Style Features → Classifier
=================================================================
Strategy:
  - Extract [CLS] embeddings from frozen pretrained model (no fine-tuning)
  - Concatenate with 20 language-agnostic style features
  - Train lightweight classifier (LogisticRegression or MLP)
  - Multi-language knowledge preserved → better generalization

Usage on Kaggle:
  python frozen_embed_classifier.py \
    --train_data /kaggle/input/.../train.parquet \
    --val_data /kaggle/input/.../validation.parquet \
    --test_data /kaggle/input/.../test.parquet \
    --train_style train_style_features.parquet \
    --val_style val_style_features.parquet \
    --test_style test_style_features.parquet \
    --model_name microsoft/codebert-base \
    --classifier mlp \
    --max_samples 100000 --batch_size 128
"""
import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--train_data", default="task_A/train.parquet")
    parser.add_argument("--val_data", default="task_A/validation.parquet")
    parser.add_argument("--test_data", default=None)
    parser.add_argument("--train_style", default=None,
                        help="Pre-extracted style features for train")
    parser.add_argument("--val_style", default=None,
                        help="Pre-extracted style features for val")
    parser.add_argument("--test_style", default=None,
                        help="Pre-extracted style features for test")
    # Model
    parser.add_argument("--model_name", default="microsoft/codebert-base",
                        help="Pretrained model for embedding extraction")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=768,
                        help="Embedding dimension of the model")
    # Extraction
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--val_samples", type=int, default=None)
    # Classifier
    parser.add_argument("--classifier", default="mlp",
                        choices=["logreg", "mlp", "xgb"],
                        help="Classifier type on top of embeddings")
    # Output
    parser.add_argument("--output_dir", default="./frozen_model")
    parser.add_argument("--submission_out", default="submission_frozen.csv")
    # Embedding cache
    parser.add_argument("--save_embeddings", action="store_true",
                        help="Save extracted embeddings to disk")
    parser.add_argument("--load_train_emb", default=None,
                        help="Load pre-extracted train embeddings (.npy)")
    parser.add_argument("--load_val_emb", default=None,
                        help="Load pre-extracted val embeddings (.npy)")
    parser.add_argument("--load_test_emb", default=None,
                        help="Load pre-extracted test embeddings (.npy)")
    return parser.parse_args()


class CodeDataset(Dataset):
    """Simple dataset for batched embedding extraction."""
    def __init__(self, codes, tokenizer, max_length):
        self.codes = codes
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        return self.codes[idx]

    def collate_fn(self, batch):
        return self.tokenizer(
            batch, truncation=True, max_length=self.max_length,
            padding=True, return_tensors='pt'
        )


@torch.no_grad()
def extract_embeddings(model, dataloader, device, desc="Extracting"):
    """Extract [CLS] embeddings from frozen model."""
    model.eval()
    all_embeddings = []
    total = len(dataloader)

    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        # [CLS] token embedding = first token of last hidden state
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_dim)
        all_embeddings.append(cls_emb.cpu().numpy())

        if (i + 1) % 100 == 0 or (i + 1) == total:
            logger.info(f"  {desc}: {i+1}/{total} batches")

    return np.concatenate(all_embeddings, axis=0)


def train_mlp_classifier(X_train, y_train, X_val, y_val, input_dim):
    """Train a simple MLP classifier with PyTorch."""
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader as TDL

    class MLP(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            return self.net(x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )
    train_dl = TDL(train_ds, batch_size=2048, shuffle=True)
    val_dl = TDL(val_ds, batch_size=4096, shuffle=False)

    best_f1 = 0
    best_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(50):
        model.train()
        total_loss = 0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss = nn.functional.cross_entropy(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Eval
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b = X_b.to(device)
                preds = model(X_b).argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(y_b.numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        f1 = f1_score(all_labels, all_preds, average='macro')

        logger.info(f"  Epoch {epoch+1}: loss={total_loss/len(train_dl):.4f}, "
                     f"val_macro_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    logger.info(f"  Best val Macro F1: {best_f1:.4f}")
    return model, device


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoTokenizer, AutoModel

    # =========================================================================
    # 1. Load data
    # =========================================================================
    logger.info(f"Loading train: {args.train_data}")
    train_df = pd.read_parquet(args.train_data)
    train_df = train_df[['code', 'label']].dropna()
    train_df['label'] = train_df['label'].astype(int)

    if args.max_samples and args.max_samples < len(train_df):
        logger.info(f"Subsampling train to {args.max_samples}...")
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

    # =========================================================================
    # 2. Extract embeddings (frozen model)
    # =========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.load_train_emb and args.load_val_emb:
        logger.info("Loading pre-extracted embeddings...")
        train_emb = np.load(args.load_train_emb)
        val_emb = np.load(args.load_val_emb)
    else:
        logger.info(f"Loading frozen model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name).to(device)
        model.eval()
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        logger.info("Extracting train embeddings...")
        train_dataset = CodeDataset(
            train_df['code'].tolist(), tokenizer, args.max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            collate_fn=train_dataset.collate_fn,
            num_workers=2, pin_memory=True
        )
        train_emb = extract_embeddings(model, train_loader, device, "Train")

        logger.info("Extracting val embeddings...")
        val_dataset = CodeDataset(
            val_df['code'].tolist(), tokenizer, args.max_length
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            collate_fn=val_dataset.collate_fn,
            num_workers=2, pin_memory=True
        )
        val_emb = extract_embeddings(model, val_loader, device, "Val")

        if args.save_embeddings:
            np.save(os.path.join(args.output_dir, "train_emb.npy"), train_emb)
            np.save(os.path.join(args.output_dir, "val_emb.npy"), val_emb)
            logger.info("Embeddings saved.")

    logger.info(f"Train embeddings: {train_emb.shape}")
    logger.info(f"Val embeddings: {val_emb.shape}")

    # =========================================================================
    # 3. Load & merge style features (optional)
    # =========================================================================
    STYLE_COLS = [
        'avg_line_length', 'max_line_length', 'blank_line_ratio',
        'comment_ratio', 'avg_indent_level', 'indent_consistency',
        'punctuation_entropy', 'unique_token_ratio', 'avg_identifier_len',
        'nested_depth_max', 'comment_completeness', 'blank_per_function',
        'comment_per_function', 'inline_comment_ratio', 'naming_uniformity',
        'keyword_density', 'avg_block_length', 'cyclomatic_proxy',
        'comment_word_count_avg', 'line_len_burstiness',
    ]

    train_style = None
    val_style = None
    test_style = None
    scaler = None

    if args.train_style:
        logger.info(f"Loading style features: {args.train_style}")
        train_sf = pd.read_parquet(args.train_style)
        val_sf = pd.read_parquet(args.val_style) if args.val_style else None

        # Align indices if subsampled
        if args.max_samples and args.max_samples < len(train_sf):
            train_sf = train_sf.iloc[train_df.index].reset_index(drop=True)

        # Use available style columns
        avail_cols = [c for c in STYLE_COLS if c in train_sf.columns]
        logger.info(f"Style features available: {len(avail_cols)}")

        train_style = train_sf[avail_cols].fillna(0).values
        if val_sf is not None:
            val_style = val_sf[avail_cols].fillna(0).values

        # Scale style features
        scaler = QuantileTransformer(
            n_quantiles=1000, output_distribution='normal', random_state=42
        )
        train_style = scaler.fit_transform(train_style)
        if val_style is not None:
            val_style = scaler.transform(val_style)

    # =========================================================================
    # 4. Combine features
    # =========================================================================
    if train_style is not None:
        # Match dimensions
        min_len = min(len(train_emb), len(train_style))
        X_train = np.hstack([train_emb[:min_len], train_style[:min_len]])
        y_train = train_df['label'].values[:min_len]

        if val_style is not None:
            min_len_v = min(len(val_emb), len(val_style))
            X_val = np.hstack([val_emb[:min_len_v], val_style[:min_len_v]])
            y_val = val_df['label'].values[:min_len_v]
        else:
            X_val = val_emb
            y_val = val_df['label'].values
    else:
        X_train = train_emb
        y_train = train_df['label'].values
        X_val = val_emb
        y_val = val_df['label'].values

    logger.info(f"Feature dims: {X_train.shape[1]} "
                f"(embed={train_emb.shape[1]}"
                f"{f'+style={train_style.shape[1]}' if train_style is not None else ''})")

    # =========================================================================
    # 5. Train classifier
    # =========================================================================
    if args.classifier == "logreg":
        from sklearn.linear_model import LogisticRegression
        logger.info("Training LogisticRegression...")
        clf = LogisticRegression(
            C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1, random_state=42
        )
        clf.fit(X_train, y_train)
        val_pred = clf.predict(X_val)
        val_proba = clf.predict_proba(X_val)[:, 1]

    elif args.classifier == "mlp":
        logger.info("Training MLP classifier...")
        mlp_model, mlp_device = train_mlp_classifier(
            X_train, y_train, X_val, y_val, X_train.shape[1]
        )
        # Get predictions
        mlp_model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(mlp_device)
            # Predict in chunks to avoid OOM
            val_logits = []
            for i in range(0, len(X_val_t), 4096):
                chunk = X_val_t[i:i+4096]
                val_logits.append(mlp_model(chunk).cpu())
            val_logits = torch.cat(val_logits, dim=0)
            val_proba = torch.softmax(val_logits, dim=1)[:, 1].numpy()
            val_pred = val_logits.argmax(dim=1).numpy()

    elif args.classifier == "xgb":
        import xgboost as xgb
        logger.info("Training XGBoost on embeddings...")
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss',
            tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
            random_state=42
        )
        clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)], verbose=50)
        val_pred = clf.predict(X_val)
        val_proba = clf.predict_proba(X_val)[:, 1]

    # =========================================================================
    # 6. Validation results
    # =========================================================================
    macro_f1 = f1_score(y_val, val_pred, average='macro')
    acc = accuracy_score(y_val, val_pred)

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS (Frozen Embed + Classifier)")
    print("=" * 60)
    print(f"Classifier: {args.classifier}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Accuracy:    {acc:.4f}")
    print(classification_report(y_val, val_pred))

    # Save val probabilities
    np.save(os.path.join(args.output_dir, "val_proba.npy"), val_proba)

    # =========================================================================
    # 7. Test inference
    # =========================================================================
    if args.test_data:
        logger.info(f"Running test inference: {args.test_data}")
        test_df = pd.read_parquet(args.test_data)
        test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.RangeIndex(len(test_df))

        if args.load_test_emb:
            test_emb = np.load(args.load_test_emb)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = AutoModel.from_pretrained(args.model_name).to(device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            test_dataset = CodeDataset(
                test_df['code'].tolist(), tokenizer, args.max_length
            )
            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size,
                collate_fn=test_dataset.collate_fn,
                num_workers=2, pin_memory=True
            )
            test_emb = extract_embeddings(model, test_loader, device, "Test")

            if args.save_embeddings:
                np.save(os.path.join(args.output_dir, "test_emb.npy"), test_emb)

        # Add style features for test
        if args.test_style and scaler is not None:
            test_sf = pd.read_parquet(args.test_style)
            avail_cols = [c for c in STYLE_COLS if c in test_sf.columns]
            test_style_arr = scaler.transform(test_sf[avail_cols].fillna(0).values)
            min_len_t = min(len(test_emb), len(test_style_arr))
            X_test = np.hstack([test_emb[:min_len_t], test_style_arr[:min_len_t]])
        else:
            X_test = test_emb

        # Predict
        if args.classifier == "mlp":
            mlp_model.eval()
            with torch.no_grad():
                X_test_t = torch.FloatTensor(X_test).to(mlp_device)
                test_logits = []
                for i in range(0, len(X_test_t), 4096):
                    chunk = X_test_t[i:i+4096]
                    test_logits.append(mlp_model(chunk).cpu())
                test_logits = torch.cat(test_logits, dim=0)
                test_proba = torch.softmax(test_logits, dim=1)[:, 1].numpy()
                test_pred = test_logits.argmax(dim=1).numpy()
        else:
            test_pred = clf.predict(X_test)
            test_proba = clf.predict_proba(X_test)[:, 1]

        # Save probabilities
        np.save(os.path.join(args.output_dir, "test_proba.npy"), test_proba)

        print(f"\nTest prediction distribution (threshold=0.5):")
        print(f"  Human: {(test_pred==0).sum()} ({(test_pred==0).mean()*100:.1f}%)")
        print(f"  AI:    {(test_pred==1).sum()} ({(test_pred==1).mean()*100:.1f}%)")

        print(f"\nDistribution at various thresholds:")
        for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            y_t = (test_proba >= t).astype(int)
            print(f"  t={t:.1f}: Human={(y_t==0).sum()} ({(y_t==0).mean()*100:.1f}%), "
                  f"AI={(y_t==1).sum()} ({(y_t==1).mean()*100:.1f}%)")

        sub = pd.DataFrame({"ID": test_ids, "label": test_pred})
        sub.to_csv(args.submission_out, index=False)
        logger.info(f"Submission saved to {args.submission_out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
