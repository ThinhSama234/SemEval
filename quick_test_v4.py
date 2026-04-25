"""Quick check v4 model on test sample."""
import pandas as pd
import numpy as np
import joblib
from feature_extractor import extract_all_features
from train_v4_robust import prep_robust_features, detect_language

# Load test sample
test = pd.read_parquet('test.parquet')
sample = test.sample(5000, random_state=42)

# Detect language distribution
langs = sample['code'].apply(detect_language)
print("Test language distribution (sample 5000):")
print(langs.value_counts())
print()

# Compare with train
train = pd.read_parquet('task_A/train.parquet')
train_langs = train['code'].apply(detect_language)
print("Train language distribution:")
print(train_langs.value_counts())
print()

# Extract features
print("Extracting features...")
X = extract_all_features(sample['code'])
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X['overall_ppl'] = pd.read_parquet('train_perplexity.parquet')['overall_ppl'].median()
X['code'] = sample['code'].values

feat = prep_robust_features(pd.DataFrame({'code': sample['code'].values, **{c: X[c] for c in X.columns if c != 'code'}}))

# Load model
model = joblib.load('taskA_xgb_v4.pkl')
feat = feat.reindex(columns=model.get_booster().feature_names, fill_value=0)

y_pred = model.predict(feat)
print(f"\nv4 Prediction distribution on test sample:")
print(f"  Class 0 (human): {(y_pred == 0).sum()} ({(y_pred == 0).mean()*100:.1f}%)")
print(f"  Class 1 (AI):    {(y_pred == 1).sum()} ({(y_pred == 1).mean()*100:.1f}%)")
