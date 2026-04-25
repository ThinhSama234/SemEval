import pandas as pd

df = pd.read_parquet('train_perplexity.parquet')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print()

for label in sorted(df['label'].unique()):
    sub = df[df['label'] == label]
    name = 'Human' if label == 0 else 'AI'
    print(f"{name}: ppl mean={sub['overall_ppl'].mean():.4f}, std={sub['overall_ppl'].std():.4f}, n={len(sub)}")

print()
print("Zero ppl count:", (df['overall_ppl'] == 0).sum())
print("NaN ppl count:", df['overall_ppl'].isna().sum())
