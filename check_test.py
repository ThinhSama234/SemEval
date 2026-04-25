import pandas as pd

df = pd.read_parquet('test.parquet')
print(f"Total: {len(df)}")
print(f"ID min: {df['ID'].min()}, ID max: {df['ID'].max()}")
print(f"ID nunique: {df['ID'].nunique()}")
print(f"Avg code length: {df['code'].str.len().mean():.0f}")
print(f"Columns: {df.columns.tolist()}")
print()

# Compare with sample submission
print("First 10 IDs:", df['ID'].head(10).tolist())
print("Last 10 IDs:", df['ID'].tail(10).tolist())
print()

# Check if IDs 2005, 2384, 3526 exist (from sample submission)
for sid in [2005, 2384, 3526]:
    found = sid in df['ID'].values
    print(f"ID {sid} in test: {found}")
