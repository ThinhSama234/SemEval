"""
Extract test probabilities from saved v5 and v7 models.

Outputs:
  - test_proba_v5.npy            (v5 hybrid IF+CNB P(AI))
  - test_proba_v7.npy            (v7 soft-voting ensemble P(AI))
  - test_proba_v7_xgb.npy        (individual xgb)
  - test_proba_v7_lgb.npy        (individual lgb)
  - test_proba_v7_cat.npy        (individual cat)
  - val_proba_v7.npy             (v7 on val, for stacking)
  - val_proba_v5.npy             (v5 on val, for stacking)
"""
import os
import re
import numpy as np
import pandas as pd
import joblib


# ---------- v7 helpers duplicated from train_v7_full_ensemble.py ----------
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


def prep_v7(df, train_cols):
    X = df.drop(columns=[c for c in ['label', 'ID', 'code', 'generator', 'language'] if c in df.columns], errors='ignore')
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X.reindex(columns=train_cols, fill_value=0)


V5_FEAT_COLS = [
    "comment_ratio", "blank_line_ratio", "indentation_std", "line_len_std",
    "style_consistency", "ttr", "comment_completeness", "blank_per_function",
    "comment_per_function", "trailing_ws_ratio", "naming_uniformity",
    "line_len_burstiness", "token_entropy", "inline_comment_ratio", "keyword_density",
    "max_nesting_depth", "avg_block_length", "cyclomatic_proxy",
    "comment_word_count_avg", "function_size_regularity",
]


def v5_predict_proba(pipeline, X):
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
    return cnb.predict_proba(S_shifted)[:, 1]


def summarize(p, name):
    print(f"  {name}: min={p.min():.4f} max={p.max():.4f} mean={p.mean():.4f} median={np.median(p):.4f}")
    for t in [0.3, 0.5, 0.7, 0.85, 0.93, 0.95]:
        pct = (p >= t).mean() * 100
        print(f"    t={t}: {pct:.1f}% AI")


# =============================================================================
# V7 — ensemble of XGB + LGB + CatBoost on 83 features
# =============================================================================
print("=" * 60)
print("V7 — loading model...")
print("=" * 60)
v7 = joblib.load('taskA_v7.pkl')
models = v7['models']
weights = v7['weights']
model_names = v7['model_names']
threshold_v7 = float(v7['threshold'])
train_cols = v7['train_columns']
ppl_median = v7['ppl_median']
print(f"  models={model_names}, weights={weights}, saved threshold={threshold_v7:.4f}")

# ----- test -----
print("\nBuilding test features for v7...")
test_df = pd.read_parquet('test_features_ml_ready.parquet')
test_df['overall_ppl'] = ppl_median
test_raw = pd.read_parquet('test.parquet', columns=['code'])
test_df['lang_id'] = test_raw['code'].apply(detect_language)
add_interaction_features(test_df)
X_test_v7 = prep_v7(test_df, train_cols)

print("Predicting v7 on test...")
probs_by_model = {}
for name in model_names:
    p = models[name].predict_proba(X_test_v7)[:, 1]
    probs_by_model[name] = p
    np.save(f'test_proba_v7_{name}.npy', p)

tot_w = sum(weights)
test_v7 = sum(probs_by_model[n] * w for n, w in zip(model_names, weights)) / tot_w
np.save('test_proba_v7.npy', test_v7)
summarize(test_v7, "v7 ensemble (test)")

# ----- val -----
print("\nBuilding val features for v7...")
val_df = pd.read_parquet('task_A/val_features_ml_ready.parquet')
val_df['overall_ppl'] = ppl_median
val_raw = pd.read_parquet('task_A/validation.parquet', columns=['language'])
# map 'language' string → lang_id
lang_map = {'Python': 0, 'C++': 1, 'Java': 2}
val_df['lang_id'] = val_raw['language'].map(lang_map).fillna(0).astype(int)
add_interaction_features(val_df)
X_val_v7 = prep_v7(val_df, train_cols)

print("Predicting v7 on val...")
val_probs_by_model = {}
for name in model_names:
    val_probs_by_model[name] = models[name].predict_proba(X_val_v7)[:, 1]

val_v7 = sum(val_probs_by_model[n] * w for n, w in zip(model_names, weights)) / tot_w
np.save('val_proba_v7.npy', val_v7)
summarize(val_v7, "v7 ensemble (val)")

# =============================================================================
# V5 — Hybrid IF+CNB on 20 style features
# =============================================================================
print("\n" + "=" * 60)
print("V5 — loading model...")
print("=" * 60)
v5 = joblib.load('taskA_hybrid_v5.pkl')
pipeline_v5 = v5['pipeline']
threshold_v5 = float(v5['threshold'])
print(f"  saved threshold={threshold_v5:.4f}")

# ----- test -----
print("\nLoading test style features...")
test_style = pd.read_parquet('test_style_features.parquet')
X_test_v5 = test_style[V5_FEAT_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"  X_test_v5: {X_test_v5.shape}")
print("Predicting v5 on test...")
test_v5 = v5_predict_proba(pipeline_v5, X_test_v5)
np.save('test_proba_v5.npy', test_v5)
summarize(test_v5, "v5 (test)")

# ----- val -----
print("\nLoading val style features...")
if os.path.exists('val_style_features.parquet'):
    val_style = pd.read_parquet('val_style_features.parquet')
    X_val_v5 = val_style[V5_FEAT_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    print("Predicting v5 on val...")
    val_v5 = v5_predict_proba(pipeline_v5, X_val_v5)
    np.save('val_proba_v5.npy', val_v5)
    summarize(val_v5, "v5 (val)")
else:
    print("  val_style_features.parquet missing — skipping val_v5")


# Save IDs once for convenience
if 'ID' in test_df.columns:
    test_ids = test_df['ID'].values
elif 'ID' in test_style.columns:
    test_ids = test_style['ID'].values
else:
    test_ids = np.arange(len(test_v5))
np.save('test_ids.npy', test_ids)
print(f"\nSaved test_ids.npy ({len(test_ids)} IDs)")
print("\nAll probas extracted.")
