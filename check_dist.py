import pandas as pd
import numpy as np

train = pd.read_parquet('task_A/train_features_ml_ready.parquet')
train_raw = pd.read_parquet('task_A/train.parquet', columns=['label'])
train['label'] = train_raw['label'].values

test = pd.read_parquet('test_features_ml_ready.parquet')

key_feats = ['tab_space_signal', 'leading_tab_lines', 'tab_count', 'leading_space_lines']
print('=== Train Human (label=0) ===')
print(train[train['label']==0][key_feats].describe().loc[['mean','std','50%']])
print()
print('=== Train AI (label=1) ===')
print(train[train['label']==1][key_feats].describe().loc[['mean','std','50%']])
print()
print('=== Test ===')
print(test[key_feats].describe().loc[['mean','std','50%']])
print()

# Check tab_space_signal distribution
print("=== tab_space_signal value distribution ===")
for name, df in [('Train Human', train[train['label']==0]), ('Train AI', train[train['label']==1]), ('Test', test)]:
    vals = df['tab_space_signal']
    print(f"{name}: mean={vals.mean():.4f}, pct_0.5={( vals==0.5).mean()*100:.1f}%, pct_0={( vals<0.1).mean()*100:.1f}%, pct_1={(vals>0.9).mean()*100:.1f}%")

# Check submission distribution
sub = pd.read_csv('submission_v7.csv')
print(f"\nSubmission v7: Human={( sub['label']==0).sum()}, AI={( sub['label']==1).sum()}")
