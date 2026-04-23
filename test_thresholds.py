"""Test different prediction thresholds on test sample."""
import pandas as pd
import numpy as np
import joblib
from feature_extractor import extract_all_features
from train_v4_robust import prep_robust_features

# Load
test = pd.read_parquet('test.parquet')
sample = test.sample(5000, random_state=42)

print("Extracting features...")
X = extract_all_features(sample['code'])
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X['overall_ppl'] = pd.read_parquet('train_perplexity.parquet')['overall_ppl'].median()
X['code'] = sample['code'].values

feat = prep_robust_features(pd.DataFrame({'code': sample['code'].values, **{c: X[c] for c in X.columns if c != 'code'}}))

model = joblib.load('taskA_xgb_v4.pkl')
feat = feat.reindex(columns=model.get_booster().feature_names, fill_value=0)

# Get probabilities
proba = model.predict_proba(feat)[:, 1]  # P(AI)

print(f"Probability stats: mean={proba.mean():.3f}, median={np.median(proba):.3f}, "
      f"std={proba.std():.3f}")
print(f"P(AI) percentiles: 10%={np.percentile(proba,10):.3f}, "
      f"25%={np.percentile(proba,25):.3f}, "
      f"50%={np.percentile(proba,50):.3f}, "
      f"75%={np.percentile(proba,75):.3f}, "
      f"90%={np.percentile(proba,90):.3f}")
print()

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    y_pred = (proba >= threshold).astype(int)
    pct_ai = y_pred.mean() * 100
    print(f"Threshold={threshold:.1f}: {pct_ai:.1f}% AI, {100-pct_ai:.1f}% Human")
