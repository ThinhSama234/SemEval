"""Extract full EDA features for test set."""
import pandas as pd
from feature_extractor import extract_all_features

print("Loading test data...")
test = pd.read_parquet("test.parquet")
print(f"Test shape: {test.shape}")

print("Extracting features (this will take a while)...")
X_test = extract_all_features(test['code'], show_progress=True)
X_test['ID'] = test['ID'].values

X_test.to_parquet("test_features_ml_ready.parquet", index=False)
print(f"Saved test_features_ml_ready.parquet: {X_test.shape}")
