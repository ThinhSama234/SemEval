"""
v7: Full Feature Ensemble
=========================
Uses ALL 76 handcrafted features + perplexity (where available)
+ XGBoost + LightGBM ensemble with threshold optimization.

Key improvements over v5:
  - 76 features instead of 20
  - Gradient boosting instead of IsolationForest+ComplementNB
  - Proper ensemble with weight optimization
  - Language detection feature for test set

Usage:
  python train_v7_full_ensemble.py
  python train_v7_full_ensemble.py --test_feat test_features_ml_ready.parquet
"""
import argparse
import os
import numpy as np
import pandas as pd
import joblib
import re
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not installed")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("WARNING: catboost not installed")


def detect_language(code):
    """Simple heuristic language detection."""
    if re.search(r'\bdef\s+\w+\s*\(', code) or re.search(r'\bimport\s+\w+', code) or 'print(' in code:
        return 0  # Python
    elif re.search(r'#include\s*<', code) or re.search(r'\bcout\s*<<', code) or re.search(r'\bstd::', code):
        return 1  # C++
    elif re.search(r'\bpublic\s+class\b', code) or re.search(r'System\.out\.print', code):
        return 2  # Java
    elif re.search(r':\s*$', code, re.MULTILINE):
        return 0  # Python (colon at end of line)
    else:
        return 0  # Default Python (most common)


def add_language_feature(df, code_series=None):
    """Add language detection feature."""
    if 'language' in df.columns:
        lang_map = {'Python': 0, 'C++': 1, 'Java': 2}
        df['lang_id'] = df['language'].map(lang_map).fillna(0).astype(int)
    elif code_series is not None:
        df['lang_id'] = code_series.apply(detect_language)
    return df


def add_interaction_features(df):
    """Add interaction features between important features."""
    # Ratios that might help distinguish human vs AI
    if 'compression_ratio' in df.columns and 'shannon_entropy' in df.columns:
        df['entropy_x_compression'] = df['shannon_entropy'] * df['compression_ratio']
    if 'line_count' in df.columns and 'token_count' in df.columns:
        df['tokens_per_line'] = df['token_count'] / df['line_count'].replace(0, 1)
    if 'comment_ratio' in df.columns and 'line_count' in df.columns:
        df['comment_density'] = df['comment_ratio'] * df['line_count']
    if 'avg_line_len' in df.columns and 'line_length_std' in df.columns:
        df['line_len_cv'] = df['line_length_std'] / df['avg_line_len'].replace(0, 1)
    if 'indent_consistency' in df.columns and 'max_indent' in df.columns:
        df['indent_ratio'] = df['indent_consistency'] / df['max_indent'].replace(0, 1)
    return df


def prep_features(df, label_col='label', id_col='ID'):
    """Prepare feature matrix, dropping non-feature columns."""
    drop_cols = [c for c in [label_col, id_col, 'code', 'generator', 'language'] if c in df.columns]
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.replace([float('inf'), float('-inf')], float('nan')).fillna(0)
    return X


def optimize_threshold(proba, y_true):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.20, 0.80, 0.005):
        f1 = f1_score(y_true, (proba >= t).astype(int), average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_feat", default="task_A/train_features_ml_ready.parquet")
    parser.add_argument("--train_ppl", default="train_perplexity.parquet")
    parser.add_argument("--val_feat", default="task_A/val_features_ml_ready.parquet")
    parser.add_argument("--test_feat", default=None)
    parser.add_argument("--test_data", default="test.parquet", help="Raw test data (for language detection)")
    parser.add_argument("--model_out", default="taskA_v7.pkl")
    parser.add_argument("--submission_out", default="submission_v7.csv")
    args = parser.parse_args()

    # =========================================================================
    # 1. Load train features
    # =========================================================================
    print("Loading train features...")
    train = pd.read_parquet(args.train_feat)
    y_train = train['label']

    # Add perplexity
    if os.path.exists(args.train_ppl):
        ppl = pd.read_parquet(args.train_ppl)
        train['overall_ppl'] = ppl['overall_ppl'].values
        ppl_median = train['overall_ppl'].median()
        print(f"  Added perplexity (median={ppl_median:.4f})")
    else:
        ppl_median = None

    # Add language feature from train data
    train_raw = pd.read_parquet("task_A/train.parquet", columns=['language'])
    train['language'] = train_raw['language'].values
    add_language_feature(train)
    add_interaction_features(train)

    X_train = prep_features(train)
    train_columns = X_train.columns.tolist()
    print(f"Train: {X_train.shape} features, {len(y_train)} samples")

    # =========================================================================
    # 2. Load val features
    # =========================================================================
    print("\nLoading val features...")
    val = pd.read_parquet(args.val_feat)
    y_val = val['label']

    # Add perplexity (use median from train)
    if ppl_median is not None:
        val['overall_ppl'] = ppl_median  # No val perplexity available
        print(f"  Filled val perplexity with train median={ppl_median:.4f}")

    # Add language feature
    val_raw = pd.read_parquet("task_A/validation.parquet", columns=['language'])
    val['language'] = val_raw['language'].values
    add_language_feature(val)
    add_interaction_features(val)

    X_val = prep_features(val)
    X_val = X_val.reindex(columns=train_columns, fill_value=0)
    print(f"Val: {X_val.shape} features, {len(y_val)} samples")

    # =========================================================================
    # 3. Train models
    # =========================================================================
    models = {}
    val_probas = {}

    # --- XGBoost ---
    print("\n[1/3] Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=3.0,
        gamma=0.3,
        random_state=42,
        tree_method='hist',
        n_jobs=-1,
        early_stopping_rounds=100,
        eval_metric='logloss',
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    models['xgb'] = xgb_model
    val_probas['xgb'] = xgb_model.predict_proba(X_val)[:, 1]
    t, f1 = optimize_threshold(val_probas['xgb'], y_val)
    print(f"  XGBoost: Macro F1={f1:.4f} (threshold={t:.3f})")

    # --- LightGBM ---
    if HAS_LGB:
        print("\n[2/3] Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=7,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=3.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)],
        )
        models['lgb'] = lgb_model
        val_probas['lgb'] = lgb_model.predict_proba(X_val)[:, 1]
        t, f1 = optimize_threshold(val_probas['lgb'], y_val)
        print(f"  LightGBM: Macro F1={f1:.4f} (threshold={t:.3f})")
    else:
        print("\n[2/3] LightGBM SKIPPED")

    # --- CatBoost ---
    if HAS_CAT:
        print("\n[3/3] Training CatBoost...")
        cat_model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=100,
            early_stopping_rounds=100,
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        models['cat'] = cat_model
        val_probas['cat'] = cat_model.predict_proba(X_val)[:, 1]
        t, f1 = optimize_threshold(val_probas['cat'], y_val)
        print(f"  CatBoost: Macro F1={f1:.4f} (threshold={t:.3f})")
    else:
        print("\n[3/3] CatBoost SKIPPED")

    # =========================================================================
    # 4. Optimize ensemble weights
    # =========================================================================
    print("\n" + "=" * 60)
    print("Optimizing ensemble...")
    print("=" * 60)

    model_names = list(val_probas.keys())
    proba_list = [val_probas[k] for k in model_names]

    # Grid search weights
    best_f1 = 0
    best_weights = [1.0] * len(model_names)
    best_threshold = 0.5

    weight_options = [0.5, 1.0, 1.5, 2.0, 2.5]
    from itertools import product
    for combo in product(weight_options, repeat=len(model_names)):
        total_w = sum(combo)
        avg_p = sum(p * w for p, w in zip(proba_list, combo)) / total_w
        t, f1 = optimize_threshold(avg_p, y_val)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = list(combo)
            best_threshold = t

    print("Best weights:")
    for name, w in zip(model_names, best_weights):
        print(f"  {name}: {w:.1f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Best Macro F1: {best_f1:.4f}")

    # Final val evaluation
    total_w = sum(best_weights)
    avg_proba_val = sum(p * w for p, w in zip(proba_list, best_weights)) / total_w
    y_pred_val = (avg_proba_val >= best_threshold).astype(int)

    print(f"\nAccuracy:    {accuracy_score(y_val, y_pred_val):.4f}")
    print(f"F1 Macro:    {f1_score(y_val, y_pred_val, average='macro'):.4f}")
    print(f"F1 Weighted: {f1_score(y_val, y_pred_val, average='weighted'):.4f}")
    print(classification_report(y_val, y_pred_val))
    print(f"Pred dist: Human={(y_pred_val==0).sum()}, AI={(y_pred_val==1).sum()}")

    # Feature importance (XGBoost)
    importance = xgb_model.get_booster().get_score(importance_type='gain')
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\nTop 20 features (XGBoost gain):")
    for feat, gain in top_features:
        print(f"  {feat}: {gain:.1f}")

    # Save
    joblib.dump({
        'models': models,
        'weights': best_weights,
        'model_names': model_names,
        'threshold': best_threshold,
        'train_columns': train_columns,
        'ppl_median': ppl_median,
    }, args.model_out)
    print(f"\nModel saved to {args.model_out}")

    # =========================================================================
    # 5. Test inference
    # =========================================================================
    if args.test_feat and os.path.exists(args.test_feat):
        print(f"\n{'='*60}")
        print("TEST INFERENCE")
        print(f"{'='*60}")
        print(f"Loading test features: {args.test_feat}")
        test_df = pd.read_parquet(args.test_feat)
        test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.RangeIndex(len(test_df))

        # Add perplexity (median fill)
        if ppl_median is not None:
            test_df['overall_ppl'] = ppl_median

        # Add language detection
        if os.path.exists(args.test_data):
            print("Detecting languages in test set...")
            test_raw = pd.read_parquet(args.test_data, columns=['code'])
            test_df['lang_id'] = test_raw['code'].apply(detect_language)
        else:
            test_df['lang_id'] = 0

        add_interaction_features(test_df)
        X_test = prep_features(test_df, id_col='ID')
        X_test = X_test.reindex(columns=train_columns, fill_value=0)
        print(f"Test features: {X_test.shape}")

        # Predict with each model
        test_probas = []
        for name in model_names:
            model = models[name]
            if name == 'cat':
                p = model.predict_proba(X_test)[:, 1]
            else:
                p = model.predict_proba(X_test)[:, 1]
            test_probas.append(p)

        total_w = sum(best_weights)
        avg_proba_test = sum(p * w for p, w in zip(test_probas, best_weights)) / total_w
        y_pred_test = (avg_proba_test >= best_threshold).astype(int)

        print(f"Test prediction distribution:")
        print(f"  Human: {(y_pred_test==0).sum()} ({(y_pred_test==0).mean()*100:.1f}%)")
        print(f"  AI:    {(y_pred_test==1).sum()} ({(y_pred_test==1).mean()*100:.1f}%)")

        sub = pd.DataFrame({"ID": test_ids, "label": y_pred_test})
        sub.to_csv(args.submission_out, index=False)
        print(f"Submission saved to {args.submission_out} ({len(sub)} rows)")
    else:
        print("\nNo test features available yet. Run with --test_feat once extraction is done.")


if __name__ == "__main__":
    main()
