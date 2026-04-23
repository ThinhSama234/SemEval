"""
Leave-One-Generator-Out CV: Train on all generators except one, test on held-out generator.
Shows how well the model generalizes to unseen AI generators.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score
import time

# Load data
print("Loading data...")
eda = pd.read_parquet('task_A/train_features_ml_ready.parquet')
ppl = pd.read_parquet('train_perplexity.parquet')
orig = pd.read_parquet('task_A/train.parquet')

eda['overall_ppl'] = ppl['overall_ppl'].values
eda['generator'] = orig['generator'].values

drop_cols = ['label', 'generator', 'language', 'code']
feature_cols = [c for c in eda.columns if c not in drop_cols]

generators = sorted(eda[eda['generator'] != 'human']['generator'].unique())
print(f"{len(generators)} AI generators to evaluate\n")

results = []
for i, held_out in enumerate(generators):
    # Train: all generators except held_out (but keep all human)
    # Test: held_out + human (to test if model can distinguish this generator from human)
    train_mask = eda['generator'] != held_out
    test_mask = (eda['generator'] == held_out) | (eda['generator'] == 'human')

    X_train = eda.loc[train_mask, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = eda.loc[train_mask, 'label']
    X_test = eda.loc[test_mask, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = eda.loc[test_mask, 'label']

    model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        tree_method='hist', n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    n_test = test_mask.sum()
    n_ai = (eda.loc[test_mask, 'generator'] == held_out).sum()

    results.append({
        'generator': held_out,
        'macro_f1': macro_f1,
        'accuracy': acc,
        'n_ai_samples': n_ai,
    })
    print(f"[{i+1:2d}/{len(generators)}] {held_out:55s} | F1={macro_f1:.4f} | Acc={acc:.4f} | n={n_ai}")

# Summary
print("\n" + "=" * 80)
df_results = pd.DataFrame(results)
print(f"Mean Macro F1: {df_results['macro_f1'].mean():.4f}")
print(f"Min  Macro F1: {df_results['macro_f1'].min():.4f} ({df_results.loc[df_results['macro_f1'].idxmin(), 'generator']})")
print(f"Max  Macro F1: {df_results['macro_f1'].max():.4f} ({df_results.loc[df_results['macro_f1'].idxmax(), 'generator']})")
print(f"Std  Macro F1: {df_results['macro_f1'].std():.4f}")
print(f"\nGenerators with F1 < 0.7:")
bad = df_results[df_results['macro_f1'] < 0.7].sort_values('macro_f1')
if len(bad):
    for _, row in bad.iterrows():
        print(f"  {row['generator']:55s} | F1={row['macro_f1']:.4f}")
else:
    print("  None!")

df_results.to_csv('logo_cv_results.csv', index=False)
print("\nResults saved to logo_cv_results.csv")
