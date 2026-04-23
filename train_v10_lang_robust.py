"""
v10: Language-Robust Ensemble (fork of v7)
==========================================
Same gradient-boosted soft-voting ensemble (XGBoost + LightGBM + CatBoost)
as v7, but drops features that hurt generalization. Two changes vs v7:

1) Drop 7 features that shift heavily between train (91.5% Python / C++ / Java)
   and test (contains unseen brace-heavy languages like JS/TS/Go/Ruby):
     avg_identifier_len      |d|=1.76   train=3.60,  test=6.48
     id_avg_len              |d|=1.69   train=3.57,  test=7.17
     id_short_ratio          |d|=1.54   train=0.45,  test=0.14
     line_entropy_std        |d|=1.40
     camel_case_ratio        |d|=1.26   train=0.02,  test=0.17
     burstiness              |d|=1.15
     punct_density           |d|=1.05

2) Drop perplexity (overall_ppl). Computing perplexity on test is prohibitively
   expensive (hours on GPU for 500K samples) and we fill test with the train
   median anyway — so the feature provides no real signal at inference time.

Root cause: v7 learned 'long identifier + camelCase => AI'. In unseen
brace-heavy languages, this is just normal human code, so v7 flags everything
as AI (test proba median = 0.9994, 94% AI predicted).

Hyperparameters and training flow are IDENTICAL to v7 to keep the comparison
clean.

Kaggle usage:
    python train_v10_lang_robust.py \
        --train_feat /kaggle/input/.../train_features_ml_ready.parquet \
        --val_feat   /kaggle/input/.../val_features_ml_ready.parquet \
        --test_feat  /kaggle/input/.../test_features_ml_ready.parquet \
        --test_data  /kaggle/input/.../test.parquet \
        --model_out  /kaggle/working/taskA_v10.pkl \
        --submission_out /kaggle/working/submission_v10.csv

The test-side quantile-threshold (50% AI) override is also written as a
secondary submission at submission_v10_q50.csv — upload that if the val-tuned
threshold gives >60% AI on test.
"""
import argparse
import os
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, accuracy_score, classification_report
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


# Features to drop — shift too hard between train and test (|d| > 1.0)
LANG_SHIFTED_FEATURES = {
    'avg_identifier_len',
    'id_avg_len',
    'id_short_ratio',
    'line_entropy_std',
    'camel_case_ratio',
    'burstiness',
    'punct_density',
    # perplexity dropped — cannot be computed on test (prohibitive GPU cost)
    'overall_ppl',
    'line_ppl_mean',
    'line_ppl_std',
    'line_ppl_max',
    'line_ppl_min',
    'ppl_variance',
}


def detect_language(code):
    if re.search(r'\bdef\s+\w+\s*\(', code) or re.search(r'\bimport\s+\w+', code) or 'print(' in code:
        return 0
    elif re.search(r'#include\s*<', code) or re.search(r'\bcout\s*<<', code) or re.search(r'\bstd::', code):
        return 1
    elif re.search(r'\bpublic\s+class\b', code) or re.search(r'System\.out\.print', code):
        return 2
    elif re.search(r':\s*$', code, re.MULTILINE):
        return 0
    return 0


def add_language_feature(df, code_series=None):
    if 'language' in df.columns:
        lang_map = {'Python': 0, 'C++': 1, 'Java': 2}
        df['lang_id'] = df['language'].map(lang_map).fillna(0).astype(int)
    elif code_series is not None:
        df['lang_id'] = code_series.apply(detect_language)
    return df


def add_interaction_features(df):
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
    """Drop metadata + language-shifted features."""
    drop_cols = [c for c in [label_col, id_col, 'code', 'generator', 'language'] if c in df.columns]
    drop_cols += [c for c in LANG_SHIFTED_FEATURES if c in df.columns]
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.replace([float('inf'), float('-inf')], float('nan')).fillna(0)
    return X


def optimize_threshold(proba, y_true, lo=0.20, hi=0.80, step=0.005):
    best_f1, best_t = 0, 0.5
    for t in np.arange(lo, hi, step):
        f1 = f1_score(y_true, (proba >= t).astype(int), average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_feat", default="task_A/train_features_ml_ready.parquet")
    parser.add_argument("--val_feat",   default="task_A/val_features_ml_ready.parquet")
    parser.add_argument("--test_feat",  default=None)
    parser.add_argument("--test_data",  default="test.parquet")
    parser.add_argument("--train_data", default="task_A/train.parquet",
                        help="needed for language metadata")
    parser.add_argument("--val_data",   default="task_A/validation.parquet",
                        help="needed for language metadata")
    parser.add_argument("--model_out",  default="taskA_v10.pkl")
    parser.add_argument("--submission_out", default="submission_v10.csv")
    args = parser.parse_args()

    # =========================================================================
    # 1. Train features
    # =========================================================================
    print("Loading train features...")
    train = pd.read_parquet(args.train_feat)
    y_train = train['label']

    # Perplexity is intentionally NOT used (prep_features drops it even if present).
    # Kept ppl_median=None so downstream code that references it still works.
    ppl_median = None

    train_raw = pd.read_parquet(args.train_data, columns=['language'])
    train['language'] = train_raw['language'].values
    add_language_feature(train)
    add_interaction_features(train)

    X_train = prep_features(train)
    train_columns = X_train.columns.tolist()
    print(f"Train: {X_train.shape} features, {len(y_train)} samples")
    print(f"  DROPPED {len(LANG_SHIFTED_FEATURES)} shifted features: {sorted(LANG_SHIFTED_FEATURES)}")

    # =========================================================================
    # 2. Val features
    # =========================================================================
    print("\nLoading val features...")
    val = pd.read_parquet(args.val_feat)
    y_val = val['label']

    val_raw = pd.read_parquet(args.val_data, columns=['language'])
    val['language'] = val_raw['language'].values
    add_language_feature(val)
    add_interaction_features(val)

    X_val = prep_features(val)
    X_val = X_val.reindex(columns=train_columns, fill_value=0)
    print(f"Val: {X_val.shape} features, {len(y_val)} samples")

    # =========================================================================
    # 3. Train models (identical hyperparams to v7)
    # =========================================================================
    models = {}
    val_probas = {}

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

    # =========================================================================
    # 4. Optimize ensemble weights
    # =========================================================================
    print("\n" + "=" * 60)
    print("Optimizing ensemble...")
    print("=" * 60)

    model_names = list(val_probas.keys())
    proba_list = [val_probas[k] for k in model_names]

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
    print(f"Best Macro F1 on val: {best_f1:.4f}")

    total_w = sum(best_weights)
    avg_proba_val = sum(p * w for p, w in zip(proba_list, best_weights)) / total_w
    y_pred_val = (avg_proba_val >= best_threshold).astype(int)

    print(f"\nVal Accuracy:    {accuracy_score(y_val, y_pred_val):.4f}")
    print(f"Val F1 Macro:    {f1_score(y_val, y_pred_val, average='macro'):.4f}")
    print(f"Val pred dist: Human={(y_pred_val==0).sum()}, AI={(y_pred_val==1).sum()}")

    importance = xgb_model.get_booster().get_score(importance_type='gain')
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    print("\nTop 15 features (XGBoost gain):")
    for feat, gain in top_features:
        print(f"  {feat}: {gain:.1f}")

    joblib.dump({
        'models': models,
        'weights': best_weights,
        'model_names': model_names,
        'threshold': best_threshold,
        'train_columns': train_columns,
        'ppl_median': ppl_median,
        'dropped_features': sorted(LANG_SHIFTED_FEATURES),
    }, args.model_out)
    print(f"\nModel saved to {args.model_out}")

    # =========================================================================
    # 5. Test inference
    # =========================================================================
    if args.test_feat and os.path.exists(args.test_feat):
        print(f"\n{'='*60}\nTEST INFERENCE\n{'='*60}")
        test_df = pd.read_parquet(args.test_feat)
        test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.RangeIndex(len(test_df))

        if os.path.exists(args.test_data):
            print("Detecting languages in test...")
            test_raw = pd.read_parquet(args.test_data, columns=['code'])
            test_df['lang_id'] = test_raw['code'].apply(detect_language)
        else:
            test_df['lang_id'] = 0

        add_interaction_features(test_df)
        X_test = prep_features(test_df, id_col='ID')
        X_test = X_test.reindex(columns=train_columns, fill_value=0)
        print(f"Test features: {X_test.shape}")

        test_probas = [models[name].predict_proba(X_test)[:, 1] for name in model_names]
        total_w = sum(best_weights)
        avg_proba_test = sum(p * w for p, w in zip(test_probas, best_weights)) / total_w

        print(f"\nTest proba stats: mean={avg_proba_test.mean():.4f} median={np.median(avg_proba_test):.4f}")
        for t_check in [0.5, 0.7, 0.85, best_threshold]:
            pct = (avg_proba_test >= t_check).mean() * 100
            print(f"  t={t_check:.3f}: {pct:.2f}% AI")

        # Primary: val-tuned threshold
        y_pred_test = (avg_proba_test >= best_threshold).astype(int)
        sub = pd.DataFrame({"ID": test_ids, "label": y_pred_test})
        sub.to_csv(args.submission_out, index=False)
        print(f"\n[primary] Submission saved to {args.submission_out}  AI%={(y_pred_test==1).mean()*100:.2f}")

        # Secondary: quantile-threshold for 50% AI (safety net if val threshold over-calls AI)
        q50_thresh = np.quantile(avg_proba_test, 0.50)
        y_pred_q50 = (avg_proba_test >= q50_thresh).astype(int)
        sec_name = args.submission_out.replace('.csv', '_q50.csv')
        pd.DataFrame({"ID": test_ids, "label": y_pred_q50}).to_csv(sec_name, index=False)
        print(f"[safety] q50 submission at {sec_name}  t={q50_thresh:.4f}  AI%=50.00")

        # Tertiary: q55 (closer to train AI ratio ~52%)
        q55_thresh = np.quantile(avg_proba_test, 0.45)
        y_pred_q55 = (avg_proba_test >= q55_thresh).astype(int)
        ter_name = args.submission_out.replace('.csv', '_q55.csv')
        pd.DataFrame({"ID": test_ids, "label": y_pred_q55}).to_csv(ter_name, index=False)
        print(f"[safety] q55 submission at {ter_name}  t={q55_thresh:.4f}  AI%=55.00")

        # Save the raw test probas too (for further blending / stacking)
        np.save(args.submission_out.replace('.csv', '_proba.npy'), avg_proba_test)
    else:
        print("\nNo --test_feat supplied. Train-only mode.")


if __name__ == "__main__":
    main()
