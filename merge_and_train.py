"""
SemEval Task 13A — LightGBM Pipeline
======================================
Features  : 24 handcrafted (18 stylistic + 6 consistency)
            + 10 AST features
            = 34 features total (no perplexity — not available at test time)
Normalise : per-language z-score on train/val; global z-score on test
Model     : LightGBM, deterministic (seed=42)
OOD stop  : hold C++ out of training → use as early-stopping validation
            cap Python at python_cap × max(other-language counts)

Usage:
  # Pre-extracted feature parquets:
  python merge_and_train.py \
      --train_feat task_A/train_features.parquet \
      --val_feat   task_A/val_features.parquet \
      --test_feat  task_A/test_features.parquet

  # Compute features on-the-fly from raw parquets:
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

    # Pre-extracted feature parquets (each row = one sample, contains feature cols + label + language)
    p.add_argument('--train_feat', default=None,
                   help='Pre-extracted train feature parquet')
    p.add_argument('--val_feat',   default=None,
                   help='Pre-extracted val feature parquet')
    p.add_argument('--test_feat',  default=None,
                   help='Pre-extracted test feature parquet (no label)')

    # Raw parquets (if --train_feat not given, features are computed here)
    p.add_argument('--raw_train', default='task_A/train.parquet')
    p.add_argument('--raw_val',   default='task_A/validation.parquet')
    p.add_argument('--raw_test',  default=None)

    # OOD / training options
    p.add_argument('--ood_language', default='c++',
                   help='Language to hold out as OOD validation (default: c++)')
    p.add_argument('--python_cap', type=float, default=4.0,
                   help='Cap Python samples at python_cap × max-other-language count')
    p.add_argument('--min_ood_samples', type=int, default=50,
                   help='Min OOD samples needed to use OOD stopping; else use random 10%% holdout')

    # LightGBM
    p.add_argument('--n_rounds',    type=int,   default=2000)
    p.add_argument('--lr',          type=float, default=0.05)
    p.add_argument('--num_leaves',  type=int,   default=63)
    p.add_argument('--max_depth',   type=int,   default=7)
    p.add_argument('--subsample',   type=float, default=0.8)
    p.add_argument('--colsample',   type=float, default=0.8)
    p.add_argument('--early_stop',  type=int,   default=50)

    # Output
    p.add_argument('--model_out',      default='taskA_lgbm.pkl')
    p.add_argument('--submission_out', default='submission_lgbm.csv')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------
def _compute_features_from_raw(raw_path: str, show_progress: bool = True) -> pd.DataFrame:
    """Compute 24 handcrafted + 10 AST features from a raw parquet file."""
    from feature_extractor import extract_24_features_batch
    from ast_features import extract_ast_features_batch

    df = pd.read_parquet(raw_path)
    log.info(f'  Loaded {len(df)} rows from {raw_path}')

    codes = df['code'].fillna('')
    langs = df['language'] if 'language' in df.columns else None

    log.info('  Extracting 24 handcrafted features...')
    feat_24 = extract_24_features_batch(codes, show_progress=show_progress)

    log.info('  Extracting 10 AST features...')
    feat_ast = extract_ast_features_batch(codes, languages=langs,
                                          show_progress=show_progress)

    feat_df = pd.concat([feat_24, feat_ast], axis=1)

    # Carry over metadata columns
    for col in ['label', 'language', 'ID', 'generator']:
        if col in df.columns:
            feat_df[col] = df[col].values

    return feat_df.reset_index(drop=True)


def _load_or_compute(feat_path, raw_path, label: str) -> pd.DataFrame:
    if feat_path and os.path.exists(feat_path):
        log.info(f'Loading {label} features from {feat_path}')
        return pd.read_parquet(feat_path)
    log.info(f'Computing {label} features from {raw_path}')
    return _compute_features_from_raw(raw_path)


def _get_feat_cols(df: pd.DataFrame) -> list:
    """Return feature column names (drop metadata)."""
    drop = {'label', 'language', 'ID', 'generator', 'code'}
    cols = [c for c in df.columns if c not in drop]
    return cols


def _clean(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    df = df.copy()
    df[feat_cols] = df[feat_cols].replace(
        [float('inf'), float('-inf')], float('nan')
    ).fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Per-language z-score normalisation
# ---------------------------------------------------------------------------
def per_language_normalize(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feat_cols: list,
    test_df: pd.DataFrame | None = None,
) -> tuple:
    """
    Normalise feat_cols with per-language z-score stats from train.

    Train / Val : per-language stats (fallback to global if language unseen).
    Test        : global stats (distribution of test languages unknown).

    Returns
    -------
    train_norm, val_norm, test_norm (or None), lang_stats, global_stats
    """
    lang_col = 'language'

    # -- Compute per-language stats from training data only --
    lang_stats = {}
    if lang_col in train_df.columns:
        for lang in train_df[lang_col].dropna().unique():
            mask = train_df[lang_col] == lang
            stats = {}
            for col in feat_cols:
                mu  = train_df.loc[mask, col].mean()
                std = train_df.loc[mask, col].std()
                stats[col] = {'mean': float(mu), 'std': float(max(std, 1e-8))}
            lang_stats[lang] = stats
    log.info(f'Per-language stats computed for {len(lang_stats)} languages: '
             f'{list(lang_stats.keys())}')

    # -- Global stats (fallback for test / unseen languages) --
    global_stats = {}
    for col in feat_cols:
        mu  = train_df[col].mean()
        std = train_df[col].std()
        global_stats[col] = {'mean': float(mu), 'std': float(max(std, 1e-8))}

    def _apply_norm(df: pd.DataFrame, use_per_lang: bool) -> pd.DataFrame:
        df = df.copy()
        if use_per_lang and lang_col in df.columns:
            for lang in df[lang_col].dropna().unique():
                mask = df[lang_col] == lang
                stats = lang_stats.get(lang, global_stats)  # fallback if unseen
                for col in feat_cols:
                    s = stats.get(col, global_stats[col])
                    df.loc[mask, col] = (df.loc[mask, col] - s['mean']) / s['std']
            # Rows with null language → global normalise
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

    train_norm = _apply_norm(train_df, use_per_lang=True)
    val_norm   = _apply_norm(val_df,   use_per_lang=True)
    test_norm  = _apply_norm(test_df,  use_per_lang=False) if test_df is not None else None

    return train_norm, val_norm, test_norm, lang_stats, global_stats


# ---------------------------------------------------------------------------
# OOD split builder
# ---------------------------------------------------------------------------
def build_ood_split(
    train_df: pd.DataFrame,
    ood_language: str = 'c++',
    python_cap: float = 4.0,
    min_ood_samples: int = 50,
    random_state: int = 42,
) -> tuple:
    """
    Separate OOD (C++) from training and cap Python samples.

    Returns
    -------
    train_ood_removed : DataFrame without OOD language, Python capped
    ood_df            : DataFrame of OOD samples (used as early-stop valid)
    ood_source        : str describing what was used as OOD
    """
    lang_col = 'language'

    if lang_col not in train_df.columns:
        log.warning('No "language" column — skipping OOD split, using random 10% holdout')
        from sklearn.model_selection import train_test_split
        tr, ood = train_test_split(train_df, test_size=0.10,
                                   stratify=train_df['label'], random_state=random_state)
        return tr, ood, 'random_10pct'

    # Normalise language strings for matching
    ood_lang_norm = ood_language.lower().strip()
    lang_norm = train_df[lang_col].str.lower().str.strip()

    cpp_aliases = {'c++', 'cpp', 'c_plus_plus', 'cplusplus'}
    if ood_lang_norm in cpp_aliases:
        ood_mask = lang_norm.isin(cpp_aliases)
    else:
        ood_mask = lang_norm == ood_lang_norm

    ood_df = train_df[ood_mask].copy()
    train_rest = train_df[~ood_mask].copy()

    if len(ood_df) < min_ood_samples:
        log.warning(
            f'OOD language "{ood_language}" has only {len(ood_df)} samples '
            f'(min={min_ood_samples}). Falling back to random 10% holdout.'
        )
        from sklearn.model_selection import train_test_split
        train_rest2, ood_fallback = train_test_split(
            train_df, test_size=0.10,
            stratify=train_df['label'], random_state=random_state,
        )
        return train_rest2, ood_fallback, 'random_10pct_fallback'

    log.info(f'OOD set: "{ood_language}" → {len(ood_df)} samples held out')

    # -- Cap Python --
    py_mask = train_rest[lang_col].str.lower().str.strip() == 'python'
    n_python = py_mask.sum()

    non_py_counts = (
        train_rest[lang_col].str.lower().str.strip()
        .where(~py_mask)
        .value_counts()
    )
    if len(non_py_counts) > 0:
        cap = int(python_cap * non_py_counts.iloc[0])  # iloc[0] = largest non-Python lang
        if n_python > cap:
            py_idx = train_rest[py_mask].sample(cap, random_state=random_state).index
            non_py_idx = train_rest[~py_mask].index
            train_rest = train_rest.loc[py_idx.union(non_py_idx)]
            log.info(
                f'Python capped: {n_python} → {cap} '
                f'(={python_cap}× {non_py_counts.index[0]}={non_py_counts.iloc[0]})'
            )
        else:
            log.info(f'Python count ({n_python}) within cap ({cap}) — no capping needed')
    else:
        log.info('Only Python in training data — skipping Python cap')

    lang_dist = train_rest[lang_col].value_counts().to_dict()
    log.info(f'Training language distribution after OOD+cap: {lang_dist}')

    return train_rest, ood_df, f'ood_{ood_language}'


# ---------------------------------------------------------------------------
# LightGBM custom metric: macro F1
# ---------------------------------------------------------------------------
def _macro_f1_metric(y_pred, data):
    """LightGBM custom eval metric returning macro F1."""
    y_true = data.get_label().astype(int)
    y_cls  = (y_pred >= 0.5).astype(int)
    score  = f1_score(y_true, y_cls, average='macro', zero_division=0)
    return 'macro_f1', float(score), True   # True = higher is better


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        log.error('lightgbm not installed. Run: pip install lightgbm')
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
    raw_test = getattr(args, 'raw_test', None)
    if args.test_feat or raw_test:
        log.info('=== Loading test features ===')
        test_df = _load_or_compute(args.test_feat, raw_test or '', 'test')

    # =========================================================================
    # 2. Identify feature columns
    # =========================================================================
    feat_cols = _get_feat_cols(train_df)
    log.info(f'Feature columns ({len(feat_cols)}): {feat_cols}')

    # Clean infinities / NaN
    train_df = _clean(train_df, feat_cols)
    val_df   = _clean(val_df,   feat_cols)

    # Align val/test to train columns
    for col in feat_cols:
        if col not in val_df.columns:
            val_df[col] = 0.0
    val_df = val_df.reindex(columns=list(val_df.columns))  # no-op, just tidy

    if test_df is not None:
        test_df = _clean(test_df, feat_cols)
        for col in feat_cols:
            if col not in test_df.columns:
                test_df[col] = 0.0

    # =========================================================================
    # 3. Per-language z-score normalisation
    # =========================================================================
    log.info('=== Per-language normalisation ===')
    train_df, val_df, test_df, lang_stats, global_stats = per_language_normalize(
        train_df, val_df, feat_cols, test_df=test_df,
    )

    # =========================================================================
    # 4. OOD split  (hold C++, cap Python)
    # =========================================================================
    log.info('=== Building OOD split ===')
    train_use, ood_df, ood_source = build_ood_split(
        train_df,
        ood_language=args.ood_language,
        python_cap=args.python_cap,
        min_ood_samples=args.min_ood_samples,
    )

    X_train = train_use[feat_cols].values
    y_train = train_use['label'].astype(int).values
    X_ood   = ood_df[feat_cols].values
    y_ood   = ood_df['label'].astype(int).values

    X_val  = val_df[feat_cols].values
    y_val  = val_df['label'].astype(int).values

    log.info(f'Train: {X_train.shape}  OOD({ood_source}): {X_ood.shape}  Val: {X_val.shape}')

    # =========================================================================
    # 5. Train LightGBM with OOD early stopping
    # =========================================================================
    params = {
        'objective':        'binary',
        'metric':           'None',         # use custom macro_f1
        'learning_rate':    args.lr,
        'num_leaves':       args.num_leaves,
        'max_depth':        args.max_depth,
        'subsample':        args.subsample,
        'colsample_bytree': args.colsample,
        'min_child_samples': 20,
        'reg_alpha':        0.1,
        'reg_lambda':       1.0,
        'seed':             42,
        'deterministic':    True,           # reproducible
        'force_row_wise':   True,           # needed for deterministic
        'n_jobs':           -1,
        'verbose':          -1,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train,
                             feature_name=feat_cols, free_raw_data=False)
    lgb_ood   = lgb.Dataset(X_ood,   label=y_ood,
                             reference=lgb_train, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(stopping_rounds=args.early_stop, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    log.info(f'=== Training LightGBM (max {args.n_rounds} rounds, '
             f'early stop on {ood_source}) ===')
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=args.n_rounds,
        valid_sets=[lgb_ood],
        valid_names=[ood_source],
        feval=_macro_f1_metric,
        callbacks=callbacks,
    )

    log.info(f'Best iteration: {model.best_iteration}  '
             f'Best OOD macro_f1: {model.best_score[ood_source]["macro_f1"]:.4f}')

    # =========================================================================
    # 6. Evaluate on full validation set
    # =========================================================================
    val_proba = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred    = (val_proba >= 0.5).astype(int)

    macro_f1 = f1_score(y_val, y_pred, average='macro')
    log.info(f'\n{"="*60}')
    log.info(f'VALIDATION RESULTS')
    log.info(f'{"="*60}')
    log.info(f'Macro F1 : {macro_f1:.4f}')
    print(classification_report(y_val, y_pred, target_names=['human', 'machine']))

    # Feature importance (top 20 by gain)
    importance = pd.Series(
        model.feature_importance(importance_type='gain'),
        index=feat_cols,
    ).sort_values(ascending=False)
    log.info('\nTop 20 features by gain:')
    for feat, gain in importance.head(20).items():
        log.info(f'  {feat:<40}  {gain:>10.1f}')

    # Save val probabilities
    np.save('val_proba_lgbm.npy', val_proba)
    log.info('Val probabilities saved to val_proba_lgbm.npy')

    # =========================================================================
    # 7. Save model
    # =========================================================================
    joblib.dump({
        'model':        model,
        'feat_cols':    feat_cols,
        'lang_stats':   lang_stats,
        'global_stats': global_stats,
        'ood_source':   ood_source,
        'best_iter':    model.best_iteration,
    }, args.model_out)
    log.info(f'Model bundle saved to {args.model_out}')

    # =========================================================================
    # 8. Test submission
    # =========================================================================
    if test_df is not None:
        test_df_clean = _clean(test_df, feat_cols)
        for col in feat_cols:
            if col not in test_df_clean.columns:
                test_df_clean[col] = 0.0

        X_test     = test_df_clean[feat_cols].values
        test_proba = model.predict(X_test, num_iteration=model.best_iteration)
        y_test     = (test_proba >= 0.5).astype(int)

        np.save('test_proba_lgbm.npy', test_proba)

        raw_test_path = args.raw_test if args.raw_test else None
        if raw_test_path and os.path.exists(raw_test_path):
            test_ids = pd.read_parquet(raw_test_path).get('ID', pd.RangeIndex(len(y_test)))
        elif 'ID' in test_df.columns:
            test_ids = test_df['ID']
        else:
            test_ids = pd.RangeIndex(len(y_test))

        sub = pd.DataFrame({'ID': test_ids, 'label': y_test})
        sub.to_csv(args.submission_out, index=False)
        log.info(f'Submission saved to {args.submission_out} ({len(sub)} rows)')
        log.info(f'  Human: {(y_test==0).sum()}  AI: {(y_test==1).sum()}')

        # Threshold sweep
        log.info('\nThreshold sweep on test:')
        for t in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
            y_t = (test_proba >= t).astype(int)
            log.info(f'  t={t:.2f}  Human={( y_t==0).sum()}  AI={(y_t==1).sum()}')


if __name__ == '__main__':
    main()
