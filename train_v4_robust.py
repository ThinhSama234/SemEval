"""
XGBoost v4: Language-robust model.
- Detect language from code
- Drop language-dependent features
- Keep only language-agnostic features
- Stronger regularization to avoid overfitting
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import re
from sklearn.metrics import f1_score, accuracy_score, classification_report
from feature_extractor import extract_all_features


# ============================================================
# Language detection (simple heuristic)
# ============================================================
def detect_language(code):
    """Detect programming language from code content."""
    if not code or not code.strip():
        return 'unknown'
    # Check for strong indicators
    if re.search(r'\bdef\s+\w+\s*\(', code) or re.search(r'\bimport\s+\w+', code) or 'print(' in code:
        if '{' not in code or code.count('{') < code.count('def '):
            return 'python'
    if re.search(r'\bpublic\s+(static\s+)?class\b', code) or re.search(r'System\.out\.print', code):
        return 'java'
    if re.search(r'#include\s*<', code) or re.search(r'\bstd::', code) or re.search(r'cout\s*<<', code):
        return 'cpp'
    if re.search(r'\bfunc\s+\w+\(', code) and re.search(r':=', code):
        return 'go'
    if re.search(r'\bfn\s+\w+\(', code) and re.search(r'->', code):
        return 'rust'
    if re.search(r'console\.log\(', code) or re.search(r'\bfunction\s+\w+\(', code):
        return 'javascript'
    if '{' in code and ';' in code:
        return 'c_style'
    return 'python'  # default


# ============================================================
# Features that are language-agnostic (transfer across languages)
# ============================================================
LANG_AGNOSTIC_FEATURES = [
    # Entropy & compression (universal)
    'shannon_entropy', 'compression_ratio',
    # Burstiness & line-level stats (universal)
    'burstiness', 'line_entropy_std', 'line_entropy_mean',
    # Code structure ratios (normalized, universal)
    'vocabulary_richness', 'duplicate_line_ratio',
    'line_length_std', 'indent_consistency',
    # Style consistency (universal concept)
    'style_consistency', 'eq_dirty_ratio', 'op_dirty_ratio', 'spacing_consistency',
    # Identifier patterns (universal)
    'id_char_entropy', 'id_short_ratio', 'id_numeric_ratio',
    # Punctuation (normalized)
    'punct_entropy', 'punct_density',
    # Repetition patterns (universal)
    'bigram_repetition', 'trigram_repetition',
    # Halstead (universal complexity)
    'halstead_volume', 'halstead_difficulty',
    # Whitespace patterns
    'indent_pattern_diversity', 'unique_indent_patterns',
    # Ratios (not absolute counts)
    'empty_line_ratio', 'comment_ratio', 'unique_char_ratio',
    'avg_line_len', 'avg_token_length',
    # Human markers (universal)
    'has_human_marker', 'human_marker_count',
    # Overall perplexity
    'overall_ppl',
    # Normalized counts (as ratios)
    'trailing_ws_lines',  # will normalize below
    'max_blank_streak',
    'max_nesting_depth',
    # 'tab_space_signal',  # REMOVED: too language-dependent (Python spaces vs C tabs)
]

# Features to NORMALIZE by line_count or char_count
NORMALIZE_BY_LINES = [
    'trailing_ws_lines', 'max_blank_streak', 'unique_indent_patterns',
]


def prep_robust_features(df, has_ppl=True):
    """Extract and select only language-agnostic features."""
    if 'char_count' not in df.columns:
        # Need to extract features
        X = extract_all_features(df['code'], show_progress=True)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    else:
        X = df.drop(columns=['label', 'generator', 'language', 'code'], errors='ignore')
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Normalize some features by line_count
    if 'line_count' in X.columns:
        for col in NORMALIZE_BY_LINES:
            if col in X.columns:
                X[col + '_ratio'] = X[col] / X['line_count'].clip(lower=1)

    # NO language dummies — they cause model to learn language-specific patterns

    # Select language-agnostic features + normalized ratios
    keep = []
    for col in X.columns:
        if col in LANG_AGNOSTIC_FEATURES:
            keep.append(col)
        elif col.endswith('_ratio') and col.startswith(tuple(NORMALIZE_BY_LINES)):
            keep.append(col)

    X_selected = X[keep] if keep else X
    return X_selected


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_in", default=None)
    parser.add_argument("--model_out", default="taskA_xgb_v4.pkl")
    parser.add_argument("--test_data", default=None)
    parser.add_argument("--test_ppl", default=None)
    parser.add_argument("--submission_out", default="submission.csv")
    args = parser.parse_args()

    # --- Load train ---
    print("Loading train data...")
    train_eda = pd.read_parquet('task_A/train_features_ml_ready.parquet')
    train_orig = pd.read_parquet('task_A/train.parquet')
    ppl = pd.read_parquet('train_perplexity.parquet')
    train_eda['overall_ppl'] = ppl['overall_ppl'].values
    train_eda['code'] = train_orig['code'].values  # need code for lang detection

    X_train = prep_robust_features(train_eda, has_ppl=True)
    y_train = train_eda['label']
    train_columns = X_train.columns
    print(f"Train: {X_train.shape}")
    print(f"Features ({len(train_columns)}): {sorted(train_columns.tolist())}")

    if args.model_in:
        print(f"\nLoading model from {args.model_in}...")
        model = joblib.load(args.model_in)
    else:
        # --- Load val ---
        print("\nLoading validation data...")
        val_eda = pd.read_parquet('task_A/val_features_ml_ready.parquet')
        val_orig = pd.read_parquet('task_A/validation.parquet')
        val_eda['overall_ppl'] = ppl['overall_ppl'].median()  # no val ppl yet
        val_eda['code'] = val_orig['code'].values

        X_val = prep_robust_features(val_eda, has_ppl=True)
        X_val = X_val.reindex(columns=train_columns, fill_value=0)
        y_val = val_eda['label']
        print(f"Val: {X_val.shape}")

        # --- Train with STRONGER regularization ---
        print("\nTraining XGBoost v4 (robust, regularized)...")
        model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,            # shallower trees (was 7)
            min_child_weight=10,    # prevent overfit (was default 1)
            subsample=0.7,          # more dropout (was 0.8)
            colsample_bytree=0.6,   # more feature dropout (was 0.8)
            reg_alpha=1.0,          # L1 regularization (was 0)
            reg_lambda=5.0,         # L2 regularization (was 1)
            gamma=1.0,              # min split loss (was 0)
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='logloss',
            tree_method='hist',
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

        # --- Evaluate ---
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS (v4 robust)")
        print("=" * 60)
        y_pred = model.predict(X_val)
        print(f"Accuracy:    {accuracy_score(y_val, y_pred):.4f}")
        print(f"F1 Macro:    {f1_score(y_val, y_pred, average='macro'):.4f}")
        print(f"F1 Weighted: {f1_score(y_val, y_pred, average='weighted'):.4f}")
        print(classification_report(y_val, y_pred))

        # Feature importance
        importance = model.get_booster().get_score(importance_type='gain')
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
        print("Top 15 features by gain:")
        for feat, gain in top_features:
            print(f"  {feat}: {gain:.1f}")

        joblib.dump(model, args.model_out)
        print(f"\nModel saved to {args.model_out}")

    # --- Quick test sample check ---
    if args.test_data:
        print(f"\nGenerating submission from {args.test_data}...")
        test_df = pd.read_parquet(args.test_data)

        print("Extracting features for test set...")
        X_test = extract_all_features(test_df['code'], show_progress=True)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

        if args.test_ppl:
            test_ppl = pd.read_parquet(args.test_ppl)
            X_test['overall_ppl'] = test_ppl['overall_ppl'].values
        else:
            X_test['overall_ppl'] = ppl['overall_ppl'].median()

        # Add code for language detection then drop
        X_test['code'] = test_df['code'].values
        test_feat = prep_robust_features(
            pd.DataFrame({'code': test_df['code'].values, **{c: X_test[c] for c in X_test.columns if c != 'code'}}),
        )
        test_feat = test_feat.reindex(columns=train_columns, fill_value=0)

        y_test_pred = model.predict(test_feat)
        print(f"\nPrediction distribution:")
        print(f"  Class 0 (human): {(y_test_pred == 0).sum()} ({(y_test_pred == 0).mean()*100:.1f}%)")
        print(f"  Class 1 (AI):    {(y_test_pred == 1).sum()} ({(y_test_pred == 1).mean()*100:.1f}%)")

        sub = pd.DataFrame({
            "ID": test_df["ID"] if "ID" in test_df.columns else test_df.index,
            "label": y_test_pred,
        })
        sub.to_csv(args.submission_out, index=False)
        print(f"Submission saved to {args.submission_out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
