"""
Diagnostic: verify which features actually shift between train and test.
This tells us whether the 'drop whitespace features' retrain is justified.
"""
import numpy as np
import pandas as pd

print("Loading train features (for distribution), test features, val features, v7 model...")
train = pd.read_parquet('task_A/train_features_ml_ready.parquet')
val   = pd.read_parquet('task_A/val_features_ml_ready.parquet')
test  = pd.read_parquet('test_features_ml_ready.parquet')

# Also pull language column where available
train_meta = pd.read_parquet('task_A/train.parquet', columns=['language', 'label', 'generator'])
val_meta   = pd.read_parquet('task_A/validation.parquet', columns=['language', 'label'])

print(f"train shape={train.shape} | val shape={val.shape} | test shape={test.shape}")

# -----------------------------------------------------------------------------
# 1. LANGUAGE DISTRIBUTION — is test really multi-language?
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("1. Language distribution")
print("=" * 70)
print("\nTrain:")
print(train_meta['language'].value_counts(normalize=True).round(3))
print("\nVal:")
print(val_meta['language'].value_counts(normalize=True).round(3))

# For test we don't have the language col in test.parquet (only ID + code).
# We can at least check unique token distribution from features.
print("\nTest has no 'language' column in test.parquet — need to infer from code.")

# -----------------------------------------------------------------------------
# 2. FEATURE-WISE DISTRIBUTION SHIFT (train vs test)
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. Feature distribution shift (train vs test) — ranked by |mean ratio|")
print("=" * 70)

feat_cols = [c for c in train.columns if c not in ['label', 'language', 'generator', 'ID', 'code']]

shifts = []
for c in feat_cols:
    if c not in test.columns:
        continue
    tr = train[c].replace([np.inf, -np.inf], np.nan).dropna()
    te = test[c].replace([np.inf, -np.inf], np.nan).dropna()
    if len(tr) == 0 or len(te) == 0:
        continue
    m_tr, m_te = tr.mean(), te.mean()
    s_tr, s_te = tr.std(), te.std()
    # Standardized mean diff (|d| between train and test)
    pooled = np.sqrt((s_tr**2 + s_te**2) / 2)
    if pooled < 1e-10:
        d = 0
    else:
        d = abs(m_tr - m_te) / pooled
    shifts.append((c, m_tr, m_te, s_tr, s_te, d))

shifts.sort(key=lambda x: x[5], reverse=True)

print(f"\n{'Feature':<30} {'train_mean':>12} {'test_mean':>12} {'train_std':>12} {'test_std':>12} {'|d|':>6}")
print("-" * 90)
for name, m_tr, m_te, s_tr, s_te, d in shifts[:25]:
    print(f"{name:<30} {m_tr:>12.4f} {m_te:>12.4f} {s_tr:>12.4f} {s_te:>12.4f} {d:>6.3f}")

print("\n--- Features with LEAST shift (train ~ test): ---")
for name, m_tr, m_te, s_tr, s_te, d in shifts[-10:]:
    print(f"{name:<30} {m_tr:>12.4f} {m_te:>12.4f}  |d|={d:.3f}")

# -----------------------------------------------------------------------------
# 3. Is the shift language-driven? Compare train Python-only vs train non-Python
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. Within train: Python vs non-Python feature means (for suspected features)")
print("=" * 70)

train['__lang'] = train_meta['language'].values
suspect = ['tab_count', 'leading_tab_lines', 'space_ratio', 'leading_space_lines',
           'space_count', 'comment_ratio', 'empty_line_ratio',
           'indent_consistency', 'max_indent', 'shannon_entropy', 'compression_ratio']
print(f"\n{'Feature':<25} {'Python_mean':>14} {'NonPy_mean':>14} {'TEST_mean':>14}")
print("-" * 70)
for c in suspect:
    if c not in train.columns or c not in test.columns:
        continue
    py_mean = train.loc[train['__lang']=='Python', c].mean()
    npy_mean = train.loc[train['__lang']!='Python', c].mean()
    te_mean = test[c].mean()
    print(f"{c:<25} {py_mean:>14.4f} {npy_mean:>14.4f} {te_mean:>14.4f}")

# -----------------------------------------------------------------------------
# 4. If test's means match non-Python train means → test is mostly non-Python
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. Conclusion flag: is test closer to train-Python or train-NonPython?")
print("=" * 70)
n_closer_nonpy = 0
n_closer_py = 0
for c in suspect:
    if c not in train.columns or c not in test.columns:
        continue
    py_mean  = train.loc[train['__lang']=='Python', c].mean()
    npy_mean = train.loc[train['__lang']!='Python', c].mean()
    te_mean  = test[c].mean()
    if abs(te_mean - py_mean) < abs(te_mean - npy_mean):
        n_closer_py += 1
    else:
        n_closer_nonpy += 1
print(f"  Features where TEST mean is closer to Python-train: {n_closer_py}")
print(f"  Features where TEST mean is closer to NonPython-train: {n_closer_nonpy}")
