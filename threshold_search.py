import numpy as np
import pandas as pd

proba = np.load('test_proba.npy')
test_df = pd.read_parquet('test.parquet')
ids = test_df['ID']

# Various thresholds
for t in [0.7, 0.8, 0.85, 0.9, 0.93, 0.95]:
    labels = (proba >= t).astype(int)
    pct = labels.mean() * 100
    fname = f'submission_t{t}.csv'
    pd.DataFrame({'ID': ids, 'label': labels}).to_csv(fname, index=False)
    print(f'{fname}: {pct:.1f}% AI, {100-pct:.1f}% Human')

# Flip predictions
labels_flip = (proba < 0.5).astype(int)
pd.DataFrame({'ID': ids, 'label': labels_flip}).to_csv('submission_flip.csv', index=False)
print(f'submission_flip.csv: {labels_flip.mean()*100:.1f}% AI, {(1-labels_flip.mean())*100:.1f}% Human')
