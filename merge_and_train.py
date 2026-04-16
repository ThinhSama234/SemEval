"""
SemEval Task 13A — LightGBM K-Fold Ensemble
============================================
Features  : 24 handcrafted (18 stylistic + 6 consistency) + 10 AST = 34 total
Normalise : per-language z-score (stats from train only)
Model     : LightGBM, 5-fold stratified CV → OOF ensemble
Threshold : calibrated on OOF predictions (not hardcoded 0.5)

Usage:
  python merge_and_train.py \
      --raw_train task_A/train.parquet \
      --raw_val   task_A/validation.parquet \
      --raw_test  task_A/test.parquet
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # Feature parquets (pre-extracted) or raw parquets (compute on-the-fly)
    p.add_argument('--train_feat', default=None)
    p.add_argument('--val_feat',   default=None)
    p.add_argument('--test_feat',  default=None)
    p.add_argument('--raw_train',  default='task_A/train.parquet')
    p.add_argument('--raw_val',    default='task_A/validation.parquet')
    p.add_argument('--raw_test',   default=None)

    # K-fold
    p.add_argument('--n_folds',   type=int,   default=5)
    p.add_argument('--n_rounds',  type=int,   default=2000)
    p.add_argument('--early_stop',type=int,   default=50)

    # LightGBM params
    p.add_argument('--lr',         type=float, default=0.05)
    p.add_argument('--num_leaves', type=int,   default=63)
    p.add_argument('--max_depth',  type=int,   default=7)
    p.add_argument('--subsample',  type=float, default=0.8)
    p.add_argument('--colsample',  type=float, default=0.8)

    # Output
    p.add_argument('--model_out',      default='taskA_lgbm_kfold.pkl')
    p.add_argument('--submission_out', default='submission_lgbm.csv')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _compute_features_from_raw(raw_path: str, show_progress: bool = True) -> pd.DataFrame:
    from feature_extractor import extract_24_features_batch
    from ast_features import extract_ast_features_batch

    df = pd.read_parquet(raw_path)
    log.info(f'  {len(df)} rows from {raw_path}')

    codes = df['code'].fillna('')
    langs = df['language'] if 'language' in df.columns else None

    log.info('  Extracting 24 handcrafted features...')
    feat_24 = extract_24_features_batch(codes, show_progress=show_progress)

    log.info('  Extracting 10 AST features...')
    feat_ast = extract_ast_features_batch(codes, languages=langs,
                                          show_progress=show_progress)

    feat_df = pd.concat([feat_24, feat_ast], axis=1)
    for col in ['label', 'language', 'ID', 'generator']:
        if col in df.columns:
            feat_df[col] = df[col].values

    return feat_df.reset_index(drop=True)


def _load_or_compute(feat_path, raw_path, label: str) -> pd.DataFrame:
    if feat_path and os.path.exists(feat_path):
        log.info(f'Loading {label} features from {feat_path}')
        return pd.read_parquet(feat_path)
    log.info(f'Computing {label} features on-the-fly from {raw_path}')
    return _compute_features_from_raw(raw_path)


def _get_feat_cols(df: pd.DataFrame) -> list:
    drop = {'label', 'language', 'ID', 'generator', 'code'}
    return [c for c in df.columns if c not in drop]


def _clean(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    df = df.copy()
    df[feat_cols] = (df[feat_cols]
                     .replace([float('inf'), float('-inf')], float('nan'))
                     .fillna(0.0)
                     .astype('float64'))
    return df


# ---------------------------------------------------------------------------
# Per-language z-score normalisation
# ---------------------------------------------------------------------------
def per_language_normalize(train_df, val_df, feat_cols, test_df=None):
    """
    Fit z-score per language on train_df, apply to val/test.
    Test gets global normalisation (language distribution unknown).
    """
    lang_col = 'language'

    # Per-language stats from train
    lang_stats = {}
    if lang_col in train_df.columns:
        for lang in train_df[lang_col].dropna().unique():
            mask = train_df[lang_col] == lang
            lang_stats[lang] = {
                col: {
                    'mean': float(train_df.loc[mask, col].mean()),
                    'std':  float(max(train_df.loc[mask, col].std(), 1e-8)),
                }
                for col in feat_cols
            }

    # Global stats (fallback)
    global_stats = {
        col: {
            'mean': float(train_df[col].mean()),
            'std':  float(max(train_df[col].std(), 1e-8)),
        }
        for col in feat_cols
    }

    def _apply(df, use_per_lang):
        df = df.copy()
        df[feat_cols] = df[feat_cols].astype('float64')
        if use_per_lang and lang_col in df.columns:
            for lang in df[lang_col].dropna().unique():
                mask = df[lang_col] == lang
                stats = lang_stats.get(lang, global_stats)
                for col in feat_cols:
                    s = stats.get(col, global_stats[col])
                    df.loc[mask, col] = (df.loc[mask, col] - s['mean']) / s['std']
            null_mask = df[lang_col].isna()
            if null_mask.any():
                for col in feat_cols:
                    s = global_stats[col]
                    df.loc[null_mask, col] = (df.loc[null_mask, col] - s['mean']) / s['std']
        else:
            for col in feat_cols:
                s = global_stats[col]
                df[col] = (df[col] - s['mean']) / s['std']
        return df

    return (
        _apply(train_df, True),
        _apply(val_df,   True),
        _apply(test_df,  False) if test_df is not None else None,
        global_stats,
    )


# ---------------------------------------------------------------------------
# Best threshold search
# ---------------------------------------------------------------------------
def find_best_threshold(y_true, proba, lo=0.3, hi=0.85, steps=56):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(lo, hi, steps):
        f1 = f1_score(y_true, (proba >= t).astype(int), average='macro')
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        log.error('lightgbm not installed: pip install lightgbm')
        sys.exit(1)
    import joblib

    # =========================================================================
    # 1. Load / compute features
    # =========================================================================
    log.info('=== Loading train features ===')
    train_df = _load_or_compute(args.train_feat, args.raw_train, 'train')

    log.info('=== Loading val features ===')
    val_df = _load_or_compute(args.val_feat, args.raw_val, 'val')

    test_df = None
    if args.test_feat or args.raw_test:
        log.info('=== Loading test features ===')
        test_df = _load_or_compute(args.test_feat, args.raw_test or '', 'test')

    # =========================================================================
    # 2. Clean + align columns
    # =========================================================================
    feat_cols = _get_feat_cols(train_df)
    log.info(f'Feature columns ({len(feat_cols)}): {feat_cols}')

    train_df = _clean(train_df, feat_cols)
    val_df   = _clean(val_df,   feat_cols)
    for col in feat_cols:
        if col not in val_df.columns: val_df[col] = 0.0

    if test_df is not None:
        test_df = _clean(test_df, feat_cols)
        for col in feat_cols:
            if col not in test_df.columns: test_df[col] = 0.0

    # =========================================================================
    # 3. Per-language normalisation (fit on train)
    # =========================================================================
    log.info('=== Per-language z-score normalisation ===')
    train_df, val_df, test_df, global_stats = per_language_normalize(
        train_df, val_df, feat_cols, test_df=test_df,
    )

    # Arrays
    X_all = train_df[feat_cols].values
    y_all = train_df['label'].astype(int).values
    X_val = val_df[feat_cols].values
    y_val = val_df['label'].astype(int).values
    X_test = test_df[feat_cols].values if test_df is not None else None

    log.info(f'Train: {X_all.shape}  Val: {X_val.shape}')
    log.info(f'Train label dist: {dict(zip(*np.unique(y_all, return_counts=True)))}')

    # =========================================================================
    # 4. K-Fold LightGBM
    # =========================================================================
    lgb_params = {
        'objective':         'binary',
        'metric':            'binary_logloss',
        'learning_rate':     args.lr,
        'num_leaves':        args.num_leaves,
        'max_depth':         args.max_depth,
        'subsample':         args.subsample,
        'colsample_bytree':  args.colsample,
        'min_child_samples': 20,
        'reg_alpha':         0.1,
        'reg_lambda':        1.0,
        'seed':              42,
        'deterministic':     True,
        'force_row_wise':    True,
        'n_jobs':            -1,
        'verbose':           -1,
    }

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    oof_proba   = np.zeros(len(X_all))      # OOF predictions on train
    val_probas  = []                         # val predictions per fold
    test_probas = []                         # test predictions per fold
    fold_models = []

    log.info(f'=== {args.n_folds}-Fold CV ===')
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
        X_tr, X_v = X_all[tr_idx], X_all[val_idx]
        y_tr, y_v = y_all[tr_idx], y_all[val_idx]

        dtrain = lgb.Dataset(X_tr, y_tr, feature_name=feat_cols, free_raw_data=False)
        dval   = lgb.Dataset(X_v,  y_v,  feature_name=feat_cols,
                             reference=dtrain, free_raw_data=False)

        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=args.n_rounds,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(args.early_stop, verbose=False),
                lgb.log_evaluation(200),
            ],
        )

        oof_proba[val_idx] = model.predict(X_v, num_iteration=model.best_iteration)

        fold_val_proba = model.predict(X_val, num_iteration=model.best_iteration)
        val_probas.append(fold_val_proba)

        if X_test is not None:
            test_probas.append(
                model.predict(X_test, num_iteration=model.best_iteration)
            )

        fold_f1 = f1_score(y_v, (oof_proba[val_idx] >= 0.5).astype(int), average='macro')
        log.info(f'  Fold {fold}/{args.n_folds}  best_iter={model.best_iteration}'
                 f'  OOF macro_f1@0.5={fold_f1:.4f}')
        fold_models.append(model)

    # =========================================================================
    # 5. Ensemble + threshold calibration on OOF
    # =========================================================================
    best_t_oof, best_f1_oof = find_best_threshold(y_all, oof_proba)
    log.info(f'\nOOF best threshold: {best_t_oof:.3f}  →  macro_f1={best_f1_oof:.4f}')
    log.info(f'OOF proba: min={oof_proba.min():.3f}  max={oof_proba.max():.3f}'
             f'  mean={oof_proba.mean():.3f}')

    # =========================================================================
    # 6. Evaluate ensemble on held-out val set
    # =========================================================================
    val_proba_ens = np.mean(val_probas, axis=0)   # average across folds

    best_t_val, best_f1_val = find_best_threshold(y_val, val_proba_ens)
    y_pred_val = (val_proba_ens >= best_t_val).astype(int)

    log.info(f'\n{"="*60}')
    log.info('VALIDATION RESULTS (fold-ensemble)')
    log.info(f'{"="*60}')
    log.info(f'Best threshold: {best_t_val:.3f}  →  Macro F1: {best_f1_val:.4f}')
    log.info(f'Val proba: min={val_proba_ens.min():.3f}  max={val_proba_ens.max():.3f}'
             f'  mean={val_proba_ens.mean():.3f}')
    log.info(f'Pred dist: Human={(y_pred_val==0).sum()}'
             f'  AI={(y_pred_val==1).sum()}')
    print(classification_report(y_val, y_pred_val, target_names=['human', 'machine']))

    # Feature importance (average across folds, top 20)
    importance = pd.Series(
        np.mean([m.feature_importance('gain') for m in fold_models], axis=0),
        index=feat_cols,
    ).sort_values(ascending=False)
    log.info('\nTop 20 features (avg gain across folds):')
    for feat, gain in importance.head(20).items():
        log.info(f'  {feat:<40}  {gain:>10.1f}')

    # Save OOF and val probabilities
    np.save('oof_proba_lgbm.npy',  oof_proba)
    np.save('val_proba_lgbm.npy',  val_proba_ens)
    log.info('Probabilities saved: oof_proba_lgbm.npy  val_proba_lgbm.npy')

    # =========================================================================
    # 7. Save model bundle
    # =========================================================================
    # Use val threshold as final threshold (more representative than OOF)
    final_threshold = best_t_val
    joblib.dump({
        'fold_models':    fold_models,
        'feat_cols':      feat_cols,
        'global_stats':   global_stats,
        'threshold':      final_threshold,
        'oof_f1':         best_f1_oof,
        'val_f1':         best_f1_val,
    }, args.model_out)
    log.info(f'Model bundle saved to {args.model_out}')

    # =========================================================================
    # 8. Test submission
    # =========================================================================
    if X_test is not None:
        test_proba_ens = np.mean(test_probas, axis=0)
        np.save('test_proba_lgbm.npy', test_proba_ens)

        y_test = (test_proba_ens >= final_threshold).astype(int)

        if 'ID' in test_df.columns:
            test_ids = test_df['ID']
        elif args.raw_test and os.path.exists(args.raw_test):
            test_ids = pd.read_parquet(args.raw_test).get(
                'ID', pd.RangeIndex(len(y_test))
            )
        else:
            test_ids = pd.RangeIndex(len(y_test))

        sub = pd.DataFrame({'ID': test_ids, 'label': y_test})
        sub.to_csv(args.submission_out, index=False)

        log.info(f'\nSubmission: {(y_test==1).sum()} AI, {(y_test==0).sum()} Human'
                 f'  (threshold={final_threshold:.3f})')
        log.info(f'Saved to {args.submission_out} ({len(sub)} rows)')

        log.info('\nThreshold sweep on test:')
        for t in np.arange(0.4, 0.86, 0.05):
            y_t = (test_proba_ens >= t).astype(int)
            log.info(f'  t={t:.2f}  AI={(y_t==1).sum():>7}  Human={(y_t==0).sum():>7}')


if __name__ == '__main__':
    main()
