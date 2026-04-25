"""
Hybrid IsolationForest + ComplementNB Pipeline (v5)
====================================================
Architecture:
  style_only features (20-dim)
    → QuantileTransformer (uniform)
    → Dual IsolationForest (one on AI, one on Human)
    → s_ai, s_hum anomaly scores
    → ComplementNB on (s_ai, s_hum)
    → threshold τ → binary label

Usage:
  # Train + evaluate on val
  python train_v5_hybrid.py

  # Train + generate submission
  python train_v5_hybrid.py --test_data test.parquet --submission_out submission.csv

  # Use pre-extracted features
  python train_v5_hybrid.py --train_feat train_style.parquet --val_feat val_style.parquet
"""
import argparse
import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import f1_score, accuracy_score, classification_report

from feature_extractor import extract_style_features

FEAT_COLS = [
    "comment_ratio", "blank_line_ratio", "indentation_std", "line_len_std",
    "style_consistency", "ttr", "comment_completeness", "blank_per_function",
    "comment_per_function", "trailing_ws_ratio", "naming_uniformity",
    "line_len_burstiness", "token_entropy", "inline_comment_ratio", "keyword_density",
    "max_nesting_depth", "avg_block_length", "cyclomatic_proxy",
    "comment_word_count_avg", "function_size_regularity",
]


def extract_style_df(codes, show_progress=True):
    """Extract 20 style features for a Series of code strings."""
    if not isinstance(codes, pd.Series):
        codes = pd.Series(codes)
    results = []
    it = tqdm(codes, desc="Style features", disable=not show_progress)
    for code in it:
        results.append(extract_style_features(code))
    df = pd.DataFrame(results, index=codes.index)
    df = df[FEAT_COLS]  # ensure column order
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def train_pipeline(X_train, y_train, X_val=None, y_val=None):
    """Train the full hybrid pipeline. Returns (pipeline_dict, best_threshold)."""

    # Step 1: QuantileTransformer
    print("  [1/4] Fitting QuantileTransformer...")
    qt = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_train_qt = qt.fit_transform(X_train)

    # Step 2: Dual IsolationForest
    print("  [2/4] Training dual IsolationForest...")
    mask_ai = (y_train == 1).values
    mask_hum = (y_train == 0).values

    iso_ai = IsolationForest(
        contamination=0.05, random_state=42, n_estimators=200, n_jobs=-1
    )
    iso_hum = IsolationForest(
        contamination=0.05, random_state=42, n_estimators=200, n_jobs=-1
    )

    iso_ai.fit(X_train_qt[mask_ai])
    iso_hum.fit(X_train_qt[mask_hum])

    # Anomaly scores for train
    s_ai_train = iso_ai.decision_function(X_train_qt)
    s_hum_train = iso_hum.decision_function(X_train_qt)
    S_train = np.column_stack([s_ai_train, s_hum_train])

    # Step 3: ComplementNB (needs non-negative input)
    print("  [3/4] Training ComplementNB...")
    # Shift scores to be non-negative for ComplementNB
    s_min = S_train.min(axis=0)
    S_train_shifted = S_train - s_min + 1e-6

    cnb = ComplementNB(alpha=1.0)
    cnb.fit(S_train_shifted, y_train)

    pipeline = {
        'qt': qt,
        'iso_ai': iso_ai,
        'iso_hum': iso_hum,
        'cnb': cnb,
        's_min': s_min,
    }

    # Step 4: Optimize threshold on validation
    best_threshold = 0.5
    if X_val is not None and y_val is not None:
        print("  [4/4] Optimizing threshold on validation...")
        X_val_qt = qt.transform(X_val)
        s_ai_val = iso_ai.decision_function(X_val_qt)
        s_hum_val = iso_hum.decision_function(X_val_qt)
        S_val = np.column_stack([s_ai_val, s_hum_val])
        S_val_shifted = S_val - s_min + 1e-6

        proba_val = cnb.predict_proba(S_val_shifted)[:, 1]  # P(AI)

        best_f1 = 0
        for t in np.arange(0.01, 0.99, 0.01):
            y_pred = (proba_val >= t).astype(int)
            f1 = f1_score(y_val, y_pred, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        print(f"  Best threshold: {best_threshold:.4f} (Macro F1 = {best_f1:.4f})")

    return pipeline, best_threshold


def predict_pipeline(pipeline, X, threshold=0.5):
    """Run prediction through the full pipeline."""
    qt = pipeline['qt']
    iso_ai = pipeline['iso_ai']
    iso_hum = pipeline['iso_hum']
    cnb = pipeline['cnb']
    s_min = pipeline['s_min']

    X_qt = qt.transform(X)
    s_ai = iso_ai.decision_function(X_qt)
    s_hum = iso_hum.decision_function(X_qt)
    S = np.column_stack([s_ai, s_hum])
    S_shifted = S - s_min + 1e-6

    proba = cnb.predict_proba(S_shifted)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return y_pred, proba


def main():
    parser = argparse.ArgumentParser(description="Hybrid IsolationForest + ComplementNB")
    parser.add_argument("--train_data", default="task_A/train.parquet")
    parser.add_argument("--val_data", default="task_A/validation.parquet")
    parser.add_argument("--test_data", default=None)
    parser.add_argument("--train_feat", default=None, help="Pre-extracted train style features")
    parser.add_argument("--val_feat", default=None, help="Pre-extracted val style features")
    parser.add_argument("--test_feat", default=None, help="Pre-extracted test style features")
    parser.add_argument("--model_out", default="taskA_hybrid_v5.pkl")
    parser.add_argument("--submission_out", default="submission.csv")
    parser.add_argument("--save_features", action="store_true", help="Save extracted features")
    args = parser.parse_args()

    # =========================================================================
    # 1. Extract / Load style features
    # =========================================================================
    if args.train_feat and os.path.exists(args.train_feat):
        print(f"Loading pre-extracted train features: {args.train_feat}")
        train_df = pd.read_parquet(args.train_feat)
        X_train = train_df[FEAT_COLS]
        y_train = train_df['label']
    else:
        print("Extracting style features for train set...")
        train_raw = pd.read_parquet(args.train_data)
        X_train = extract_style_df(train_raw['code'])
        y_train = train_raw['label']
        if args.save_features:
            out = X_train.copy()
            out['label'] = y_train.values
            out.to_parquet('train_style_features.parquet', index=False)
            print("Saved train_style_features.parquet")

    print(f"Train: {X_train.shape}, labels: {y_train.value_counts().to_dict()}")

    if args.val_feat and os.path.exists(args.val_feat):
        print(f"Loading pre-extracted val features: {args.val_feat}")
        val_df = pd.read_parquet(args.val_feat)
        X_val = val_df[FEAT_COLS]
        y_val = val_df['label']
    else:
        print("Extracting style features for val set...")
        val_raw = pd.read_parquet(args.val_data)
        X_val = extract_style_df(val_raw['code'])
        y_val = val_raw['label']
        if args.save_features:
            out = X_val.copy()
            out['label'] = y_val.values
            out.to_parquet('val_style_features.parquet', index=False)
            print("Saved val_style_features.parquet")

    print(f"Val:   {X_val.shape}, labels: {y_val.value_counts().to_dict()}")

    # =========================================================================
    # 2. Train pipeline
    # =========================================================================
    print("\nTraining hybrid pipeline...")
    pipeline, threshold = train_pipeline(X_train, y_train, X_val, y_val)

    # =========================================================================
    # 3. Evaluate on validation
    # =========================================================================
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS (Hybrid v5)")
    print("=" * 60)
    y_pred_val, proba_val = predict_pipeline(pipeline, X_val, threshold)
    print(f"Threshold: {threshold:.4f}")
    print(f"Accuracy:    {accuracy_score(y_val, y_pred_val):.4f}")
    print(f"F1 Macro:    {f1_score(y_val, y_pred_val, average='macro'):.4f}")
    print(f"F1 Weighted: {f1_score(y_val, y_pred_val, average='weighted'):.4f}")
    print(classification_report(y_val, y_pred_val))
    print(f"Prediction distribution: "
          f"Human={( y_pred_val == 0).sum()}, AI={(y_pred_val == 1).sum()}")

    # =========================================================================
    # 4. Save pipeline
    # =========================================================================
    joblib.dump({'pipeline': pipeline, 'threshold': threshold}, args.model_out)
    print(f"\nPipeline saved to {args.model_out}")

    # =========================================================================
    # 5. Test set inference
    # =========================================================================
    if args.test_data or args.test_feat:
        if args.test_feat and os.path.exists(args.test_feat):
            print(f"\nLoading pre-extracted test features: {args.test_feat}")
            test_feat_df = pd.read_parquet(args.test_feat)
            X_test = test_feat_df[FEAT_COLS]
            test_ids = test_feat_df['ID'] if 'ID' in test_feat_df.columns else test_feat_df.index
        else:
            print(f"\nExtracting style features for test set: {args.test_data}")
            test_raw = pd.read_parquet(args.test_data)
            X_test = extract_style_df(test_raw['code'])
            test_ids = test_raw['ID'] if 'ID' in test_raw.columns else test_raw.index
            if args.save_features:
                out = X_test.copy()
                out['ID'] = test_ids.values
                out.to_parquet('test_style_features.parquet', index=False)
                print("Saved test_style_features.parquet")

        y_pred_test, proba_test = predict_pipeline(pipeline, X_test, threshold)
        print(f"\nTest prediction distribution:")
        print(f"  Human: {(y_pred_test == 0).sum()} ({(y_pred_test == 0).mean()*100:.1f}%)")
        print(f"  AI:    {(y_pred_test == 1).sum()} ({(y_pred_test == 1).mean()*100:.1f}%)")

        sub = pd.DataFrame({"ID": test_ids, "label": y_pred_test})
        sub.to_csv(args.submission_out, index=False)
        print(f"Submission saved to {args.submission_out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
