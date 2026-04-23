"""
Ensemble Pipeline v6: Multiple diverse models → soft voting
=============================================================
Models:
  1. IsolationForest + ComplementNB (hybrid v5)
  2. LightGBM on style features
  3. CatBoost on style features
  4. XGBoost on style features (shallow, regularized)

All use the same 20 style_only features → diverse algorithms → soft vote.

Usage:
  # Train + eval + submission
  python train_v6_ensemble.py --test_data test.parquet

  # Use pre-extracted features (fast)
  python train_v6_ensemble.py --train_feat train_style_features.parquet \
    --val_feat val_style_features.parquet --test_feat test_style_features.parquet
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
import xgboost as xgb

from feature_extractor import extract_style_features

# Try import optional deps
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not installed. pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("WARNING: catboost not installed. pip install catboost")


FEAT_COLS = [
    "comment_ratio", "blank_line_ratio", "indentation_std", "line_len_std",
    "style_consistency", "ttr", "comment_completeness", "blank_per_function",
    "comment_per_function", "trailing_ws_ratio", "naming_uniformity",
    "line_len_burstiness", "token_entropy", "inline_comment_ratio", "keyword_density",
    "max_nesting_depth", "avg_block_length", "cyclomatic_proxy",
    "comment_word_count_avg", "function_size_regularity",
]


def extract_style_df(codes, show_progress=True):
    if not isinstance(codes, pd.Series):
        codes = pd.Series(codes)
    results = []
    it = tqdm(codes, desc="Style features", disable=not show_progress)
    for code in it:
        results.append(extract_style_features(code))
    df = pd.DataFrame(results, index=codes.index)
    df = df[FEAT_COLS]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# =============================================================================
# Model 1: Hybrid IsolationForest + ComplementNB
# =============================================================================
def train_hybrid(X_train, y_train):
    qt = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_qt = qt.fit_transform(X_train)

    mask_ai = (y_train == 1).values
    mask_hum = (y_train == 0).values

    iso_ai = IsolationForest(contamination=0.05, random_state=42, n_estimators=200, n_jobs=-1)
    iso_hum = IsolationForest(contamination=0.05, random_state=42, n_estimators=200, n_jobs=-1)
    iso_ai.fit(X_qt[mask_ai])
    iso_hum.fit(X_qt[mask_hum])

    s_ai = iso_ai.decision_function(X_qt)
    s_hum = iso_hum.decision_function(X_qt)
    S = np.column_stack([s_ai, s_hum])
    s_min = S.min(axis=0)
    S_shifted = S - s_min + 1e-6

    cnb = ComplementNB(alpha=1.0)
    cnb.fit(S_shifted, y_train)

    return {'qt': qt, 'iso_ai': iso_ai, 'iso_hum': iso_hum, 'cnb': cnb, 's_min': s_min}


def predict_hybrid(model, X):
    X_qt = model['qt'].transform(X)
    s_ai = model['iso_ai'].decision_function(X_qt)
    s_hum = model['iso_hum'].decision_function(X_qt)
    S = np.column_stack([s_ai, s_hum])
    S_shifted = S - model['s_min'] + 1e-6
    return model['cnb'].predict_proba(S_shifted)[:, 1]


# =============================================================================
# Model 2: LightGBM
# =============================================================================
def train_lgb(X_train, y_train, X_val, y_val):
    if not HAS_LGB:
        return None
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=5.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def predict_lgb(model, X):
    if model is None:
        return None
    return model.predict_proba(X)[:, 1]


# =============================================================================
# Model 3: CatBoost
# =============================================================================
def train_catboost(X_train, y_train, X_val, y_val):
    if not HAS_CAT:
        return None
    model = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=4,
        l2_leaf_reg=5.0, random_seed=42,
        verbose=0, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return model


def predict_catboost(model, X):
    if model is None:
        return None
    return model.predict_proba(X)[:, 1]


# =============================================================================
# Model 4: XGBoost (shallow, regularized)
# =============================================================================
def train_xgb(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=4,
        min_child_weight=10, subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0, gamma=1.0,
        random_state=42, tree_method='hist', n_jobs=-1, verbosity=0,
        early_stopping_rounds=50, eval_metric='logloss',
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def predict_xgb(model, X):
    return model.predict_proba(X)[:, 1]


# =============================================================================
# Ensemble
# =============================================================================
def ensemble_predict(probas, weights, threshold=0.5):
    """Weighted average of probabilities → threshold → labels."""
    valid = [(p, w) for p, w in zip(probas, weights) if p is not None]
    if not valid:
        raise ValueError("No valid models!")
    total_w = sum(w for _, w in valid)
    avg_proba = sum(p * w for p, w in valid) / total_w
    return (avg_proba >= threshold).astype(int), avg_proba


def optimize_threshold(proba, y_true):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        f1 = f1_score(y_true, (proba >= t).astype(int), average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def optimize_weights(probas_list, y_true):
    """Grid search over weight combinations."""
    valid_idx = [i for i, p in enumerate(probas_list) if p is not None]
    if len(valid_idx) == 1:
        return [1.0] * len(probas_list)

    best_f1 = 0
    best_weights = [1.0] * len(probas_list)
    # Coarse grid search
    grid = [0.5, 1.0, 1.5, 2.0]
    from itertools import product
    for combo in product(grid, repeat=len(valid_idx)):
        weights = [0.0] * len(probas_list)
        for j, idx in enumerate(valid_idx):
            weights[idx] = combo[j]
        y_pred, avg_p = ensemble_predict(probas_list, weights, 0.5)
        # Also optimize threshold for this weight combo
        t, f1 = optimize_threshold(avg_p, y_true)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights
    return best_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="task_A/train.parquet")
    parser.add_argument("--val_data", default="task_A/validation.parquet")
    parser.add_argument("--test_data", default=None)
    parser.add_argument("--train_feat", default=None)
    parser.add_argument("--val_feat", default=None)
    parser.add_argument("--test_feat", default=None)
    parser.add_argument("--model_out", default="taskA_ensemble_v6.pkl")
    parser.add_argument("--submission_out", default="submission_v6.csv")
    parser.add_argument("--save_features", action="store_true")
    args = parser.parse_args()

    # =========================================================================
    # 1. Load features
    # =========================================================================
    if args.train_feat and os.path.exists(args.train_feat):
        print(f"Loading train features: {args.train_feat}")
        train_df = pd.read_parquet(args.train_feat)
        X_train = train_df[FEAT_COLS].values
        y_train = train_df['label']
    else:
        print("Extracting train style features...")
        raw = pd.read_parquet(args.train_data)
        X_train_df = extract_style_df(raw['code'])
        y_train = raw['label']
        X_train = X_train_df.values
        if args.save_features:
            out = X_train_df.copy(); out['label'] = y_train.values
            out.to_parquet('train_style_features.parquet', index=False)

    if args.val_feat and os.path.exists(args.val_feat):
        print(f"Loading val features: {args.val_feat}")
        val_df = pd.read_parquet(args.val_feat)
        X_val = val_df[FEAT_COLS].values
        y_val = val_df['label']
    else:
        print("Extracting val style features...")
        raw = pd.read_parquet(args.val_data)
        X_val_df = extract_style_df(raw['code'])
        y_val = raw['label']
        X_val = X_val_df.values
        if args.save_features:
            out = X_val_df.copy(); out['label'] = y_val.values
            out.to_parquet('val_style_features.parquet', index=False)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # =========================================================================
    # 2. Train all models
    # =========================================================================
    models = {}

    print("\n[1/4] Training Hybrid (IsolationForest + ComplementNB)...")
    models['hybrid'] = train_hybrid(X_train, y_train)

    print("[2/4] Training LightGBM...")
    models['lgb'] = train_lgb(X_train, y_train, X_val, y_val)
    if models['lgb'] is None:
        print("  SKIPPED (lightgbm not installed)")

    print("[3/4] Training CatBoost...")
    models['catboost'] = train_catboost(X_train, y_train, X_val, y_val)
    if models['catboost'] is None:
        print("  SKIPPED (catboost not installed)")

    print("[4/4] Training XGBoost...")
    models['xgb'] = train_xgb(X_train, y_train, X_val, y_val)

    # =========================================================================
    # 3. Get val probabilities from each model
    # =========================================================================
    print("\nGenerating val predictions...")
    val_probas = [
        predict_hybrid(models['hybrid'], X_val),
        predict_lgb(models['lgb'], X_val),
        predict_catboost(models['catboost'], X_val),
        predict_xgb(models['xgb'], X_val),
    ]
    model_names = ['Hybrid', 'LightGBM', 'CatBoost', 'XGBoost']

    # Individual model performance
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODEL RESULTS (Val)")
    print("=" * 60)
    for name, proba in zip(model_names, val_probas):
        if proba is None:
            print(f"  {name:12s}: SKIPPED")
            continue
        t, f1 = optimize_threshold(proba, y_val)
        y_pred = (proba >= t).astype(int)
        acc = accuracy_score(y_val, y_pred)
        print(f"  {name:12s}: Macro F1={f1:.4f}, Acc={acc:.4f}, threshold={t:.2f}, "
              f"Human={(y_pred==0).sum()}, AI={(y_pred==1).sum()}")

    # =========================================================================
    # 4. Optimize ensemble weights
    # =========================================================================
    print("\nOptimizing ensemble weights...")
    weights = optimize_weights(val_probas, y_val)
    for name, w in zip(model_names, weights):
        print(f"  {name:12s}: weight={w:.1f}")

    # Final ensemble with optimized threshold
    _, avg_proba_val = ensemble_predict(val_probas, weights, 0.5)
    best_threshold, best_f1 = optimize_threshold(avg_proba_val, y_val)

    print(f"\n{'='*60}")
    print(f"ENSEMBLE RESULTS (Val)")
    print(f"{'='*60}")
    y_pred_val = (avg_proba_val >= best_threshold).astype(int)
    print(f"Threshold: {best_threshold:.4f}")
    print(f"Accuracy:    {accuracy_score(y_val, y_pred_val):.4f}")
    print(f"F1 Macro:    {f1_score(y_val, y_pred_val, average='macro'):.4f}")
    print(f"F1 Weighted: {f1_score(y_val, y_pred_val, average='weighted'):.4f}")
    print(classification_report(y_val, y_pred_val))
    print(f"Prediction: Human={(y_pred_val==0).sum()}, AI={(y_pred_val==1).sum()}")

    # Save
    joblib.dump({
        'models': models, 'weights': weights, 'threshold': best_threshold,
        'feat_cols': FEAT_COLS,
    }, args.model_out)
    print(f"\nEnsemble saved to {args.model_out}")

    # =========================================================================
    # 5. Test inference
    # =========================================================================
    if args.test_data or args.test_feat:
        if args.test_feat and os.path.exists(args.test_feat):
            print(f"\nLoading test features: {args.test_feat}")
            test_df = pd.read_parquet(args.test_feat)
            X_test = test_df[FEAT_COLS].values
            test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.RangeIndex(len(test_df))
        else:
            print(f"\nExtracting test features: {args.test_data}")
            test_raw = pd.read_parquet(args.test_data)
            X_test_df = extract_style_df(test_raw['code'])
            X_test = X_test_df.values
            test_ids = test_raw['ID'] if 'ID' in test_raw.columns else test_raw.index
            if args.save_features:
                out = X_test_df.copy(); out['ID'] = test_ids.values
                out.to_parquet('test_style_features.parquet', index=False)

        print("Generating test predictions...")
        test_probas = [
            predict_hybrid(models['hybrid'], X_test),
            predict_lgb(models['lgb'], X_test),
            predict_catboost(models['catboost'], X_test),
            predict_xgb(models['xgb'], X_test),
        ]

        y_pred_test, avg_proba_test = ensemble_predict(test_probas, weights, best_threshold)
        print(f"\nTest prediction distribution:")
        print(f"  Human: {(y_pred_test==0).sum()} ({(y_pred_test==0).mean()*100:.1f}%)")
        print(f"  AI:    {(y_pred_test==1).sum()} ({(y_pred_test==1).mean()*100:.1f}%)")

        sub = pd.DataFrame({"ID": test_ids, "label": y_pred_test})
        sub.to_csv(args.submission_out, index=False)
        print(f"Submission saved to {args.submission_out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
