"""Extract features for validation.parquet using the same pipeline as train."""
import pandas as pd
from feature_extractor import extract_all_features

INPUT = "./task_A/validation.parquet"
OUTPUT = "./task_A/val_features_ml_ready.parquet"

print(f"Loading {INPUT}...")
df = pd.read_parquet(INPUT)
print(f"Shape: {df.shape}, columns: {df.columns.tolist()}")

print("Extracting features...")
X = extract_all_features(df['code'], show_progress=True)
X = X.replace([float('inf'), float('-inf')], float('nan')).fillna(0)

# Match format of train_features_ml_ready: features + label only
out = X.copy()
out['label'] = df['label'].values

print(f"Output shape: {out.shape}")
out.to_parquet(OUTPUT, index=False)
print(f"Saved to {OUTPUT}")
