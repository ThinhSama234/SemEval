"""Diagnose why model fails on test set."""
import pandas as pd
import numpy as np
from feature_extractor import extract_all_features

# 1. Extract features for a sample of test
print("Loading test data...")
test = pd.read_parquet('test.parquet')
print(f"Test: {len(test)} samples")

# Sample 5000 for speed
sample = test.sample(5000, random_state=42)
print("Extracting EDA features for 5000 test samples...")
X_test_sample = extract_all_features(sample['code'])
X_test_sample = X_test_sample.replace([np.inf, -np.inf], np.nan).fillna(0)

# 2. Load train features
print("\nLoading train features...")
train = pd.read_parquet('task_A/train_features_ml_ready.parquet')
ppl = pd.read_parquet('train_perplexity.parquet')
train['overall_ppl'] = ppl['overall_ppl'].values
X_test_sample['overall_ppl'] = 0  # placeholder

# 3. Compare distributions
feature_cols = [c for c in train.columns if c != 'label']
print("\n" + "=" * 70)
print("FEATURE DISTRIBUTION COMPARISON (train vs test)")
print("=" * 70)
print(f"{'Feature':<30} {'Train Mean':>12} {'Test Mean':>12} {'Train Std':>12} {'Test Std':>12} {'Shift':>8}")
print("-" * 86)

shifts = {}
for col in sorted(feature_cols):
    if col in X_test_sample.columns:
        tr_mean = train[col].mean()
        te_mean = X_test_sample[col].mean()
        tr_std = max(train[col].std(), 1e-8)
        te_std = X_test_sample[col].std()
        shift = abs(tr_mean - te_mean) / tr_std
        shifts[col] = shift
        if shift > 0.5:  # notable shift
            print(f"{col:<30} {tr_mean:>12.4f} {te_mean:>12.4f} {tr_std:>12.4f} {te_std:>12.4f} {shift:>8.2f} ***")

print("\nTop 15 features with biggest distribution shift:")
for col, shift in sorted(shifts.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {col:<30} shift={shift:.2f}")

# 4. Load model and predict on test sample
import joblib
print("\nLoading model and predicting on test sample...")
model = joblib.load('taskA_xgb_v3.pkl')
X_test_aligned = X_test_sample.reindex(columns=[c for c in train.columns if c != 'label'], fill_value=0)
y_pred = model.predict(X_test_aligned)
print(f"\nTest sample prediction distribution:")
print(f"  Class 0 (human): {(y_pred == 0).sum()} ({(y_pred == 0).mean()*100:.1f}%)")
print(f"  Class 1 (AI):    {(y_pred == 1).sum()} ({(y_pred == 1).mean()*100:.1f}%)")

# 5. Compare with train distribution
print(f"\nTrain label distribution:")
print(f"  Class 0 (human): {(train['label'] == 0).sum()} ({(train['label'] == 0).mean()*100:.1f}%)")
print(f"  Class 1 (AI):    {(train['label'] == 1).sum()} ({(train['label'] == 1).mean()*100:.1f}%)")
