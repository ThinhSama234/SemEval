"""Quick test: ensemble with only hybrid (high weight) to keep balance."""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score

data = joblib.load('taskA_ensemble_v6.pkl')
models = data['models']
FEAT_COLS = data['feat_cols']

# Load test + val
test_df = pd.read_parquet('test_style_features.parquet')
val_df = pd.read_parquet('val_style_features.parquet')
X_test = test_df[FEAT_COLS].values
X_val = val_df[FEAT_COLS].values
y_val = val_df['label']

from train_v6_ensemble import predict_hybrid, predict_lgb, predict_catboost, predict_xgb, optimize_threshold

# Get all probas
p_hybrid_val = predict_hybrid(models['hybrid'], X_val)
p_lgb_val = predict_lgb(models['lgb'], X_val)
p_cat_val = predict_catboost(models['catboost'], X_val)
p_xgb_val = predict_xgb(models['xgb'], X_val)

p_hybrid_test = predict_hybrid(models['hybrid'], X_test)
p_lgb_test = predict_lgb(models['lgb'], X_test)
p_cat_test = predict_catboost(models['catboost'], X_test)
p_xgb_test = predict_xgb(models['xgb'], X_test)

# Try different weight combos focused on hybrid
combos = [
    ("hybrid only",       [3.0, 0.0, 0.0, 0.0]),
    ("hybrid+lgb",        [2.0, 1.0, 0.0, 0.0]),
    ("hybrid+xgb",        [2.0, 0.0, 0.0, 1.0]),
    ("hybrid+lgb+xgb",    [2.0, 1.0, 0.0, 1.0]),
    ("hybrid heavy",      [3.0, 1.0, 1.0, 1.0]),
    ("equal",             [1.0, 1.0, 1.0, 1.0]),
]

all_val = [p_hybrid_val, p_lgb_val, p_cat_val, p_xgb_val]
all_test = [p_hybrid_test, p_lgb_test, p_cat_test, p_xgb_test]

print(f"{'Config':<22} {'Val F1':>8} {'Thr':>6} {'Test%Human':>11} {'Test%AI':>8}")
print("-" * 60)

best_f1 = 0
best_config = None
for name, weights in combos:
    total_w = sum(w for w, p in zip(weights, all_val) if p is not None and w > 0)
    if total_w == 0:
        continue
    avg_val = sum(p * w for p, w in zip(all_val, weights) if p is not None and w > 0) / total_w
    avg_test = sum(p * w for p, w in zip(all_test, weights) if p is not None and w > 0) / total_w

    t, f1 = optimize_threshold(avg_val, y_val)
    y_test = (avg_test >= t).astype(int)
    pct_human = (y_test == 0).mean() * 100
    pct_ai = (y_test == 1).mean() * 100

    print(f"{name:<22} {f1:>8.4f} {t:>6.2f} {pct_human:>10.1f}% {pct_ai:>7.1f}%")

    if f1 > best_f1:
        best_f1 = f1
        best_config = (name, weights, t)

# Save best balanced submission
name, weights, threshold = best_config
total_w = sum(w for w, p in zip(weights, all_test) if p is not None and w > 0)
avg_test = sum(p * w for p, w in zip(all_test, weights) if p is not None and w > 0) / total_w
y_pred = (avg_test >= threshold).astype(int)

print(f"\nBest: {name} (F1={best_f1:.4f}, threshold={threshold:.2f})")
print(f"Test: Human={(y_pred==0).sum()} ({(y_pred==0).mean()*100:.1f}%), AI={(y_pred==1).sum()} ({(y_pred==1).mean()*100:.1f}%)")

test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.RangeIndex(len(test_df))
sub = pd.DataFrame({"ID": test_ids, "label": y_pred})
sub.to_csv("submission_v6_balanced.csv", index=False)
print(f"Saved submission_v6_balanced.csv")
