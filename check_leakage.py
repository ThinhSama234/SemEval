import pandas as pd

train = pd.read_parquet('task_A/train.parquet')
val = pd.read_parquet('task_A/validation.parquet')
print(f'Train: {len(train)}, Val: {len(val)}')

# Check exact duplicates by code
train_codes = set(train['code'])
val_codes = set(val['code'])
overlap = train_codes & val_codes
print(f'Overlapping code snippets: {len(overlap)} ({100*len(overlap)/len(val):.2f}% of val)')

# Check distribution
print('\nTrain label dist:', train['label'].value_counts().to_dict())
print('Val label dist:', val['label'].value_counts().to_dict())
print('\nTrain generators (top5):', train['generator'].value_counts().head(5).to_dict())
print('Val generators (top5):', val['generator'].value_counts().head(5).to_dict())
print('\nTrain languages:', train['language'].value_counts().to_dict())
print('Val languages:', val['language'].value_counts().to_dict())

# Check if all val generators exist in train
train_gens = set(train['generator'].unique())
val_gens = set(val['generator'].unique())
print(f'\nVal generators NOT in train: {val_gens - train_gens}')
print(f'Train generators NOT in val: {train_gens - val_gens}')
