import numpy as np
import pandas as pd

proba = np.load('test_proba.npy')
test_df = pd.read_parquet('test.parquet')
ids = test_df['ID']

for t in [0.91, 0.92, 0.93, 0.935, 0.94, 0.945, 0.95, 0.96]:
    labels = (proba >= t).astype(int)
    pct = labels.mean() * 100
    fname = f'submission_t{t}.csv'
    pd.DataFrame({'ID': ids, 'label': labels}).to_csv(fname, index=False)
    print(f't={t}: {pct:.1f}% AI, {100-pct:.1f}% Human')
