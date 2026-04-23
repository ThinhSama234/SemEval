"""
Stacking meta-learner on top of v5 + v7 (+ optional CodeBERT) probabilities.

Training signal: val_labels
Features for meta: val_proba_v5, val_proba_v7 (, val_proba_cb)
Meta-learner: logistic regression (simple, hard to overfit)
Apply to test: predict on test probas → threshold sweep to target balanced AI%.

Usage:
    python stacking_meta.py                      # uses v5+v7 only
    python stacking_meta.py --cb_val val_proba_cb.npy  # adds CodeBERT if available
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cb_val', default=None,
                   help='path to val_proba_cb.npy if you have run CodeBERT on val')
    p.add_argument('--cb_test', default='test_proba.npy',
                   help='path to test_proba.npy (CodeBERT)')
    args = p.parse_args()

    # Load val probas (meta-training)
    val_v5 = np.load('val_proba_v5.npy')
    val_v7 = np.load('val_proba_v7.npy')
    val_df = pd.read_parquet('task_A/val_features_ml_ready.parquet')
    y_val = val_df['label'].values

    # Load test probas
    test_v5 = np.load('test_proba_v5.npy')
    test_v7 = np.load('test_proba_v7.npy')
    test_ids = np.load('test_ids.npy')

    feat_cols = ['v5', 'v7']
    X_val = np.column_stack([val_v5, val_v7])
    X_test = np.column_stack([test_v5, test_v7])

    if args.cb_val is not None:
        val_cb = np.load(args.cb_val)
        test_cb = np.load(args.cb_test)
        X_val = np.column_stack([X_val, val_cb])
        X_test = np.column_stack([X_test, test_cb])
        feat_cols.append('cb')
        print(f"Using 3 base models: {feat_cols}")
    else:
        print(f"Using 2 base models: {feat_cols} (no CodeBERT val probas provided)")
        print("  Tip: run finetune_codebert.py predict-mode on val set, save val_proba_cb.npy, rerun with --cb_val")

    # Train meta-learner
    meta = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    meta.fit(X_val, y_val)
    print(f"\nMeta coefs: {dict(zip(feat_cols, meta.coef_[0]))}")
    print(f"Meta intercept: {meta.intercept_[0]:.4f}")

    # Val-set sanity check
    val_pred_proba = meta.predict_proba(X_val)[:, 1]
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.005):
        f1 = f1_score(y_val, (val_pred_proba >= t).astype(int), average='macro')
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"\nMeta val Macro F1: {best_f1:.4f} @ t={best_t:.3f}")
    pred_ai = (val_pred_proba >= best_t).mean() * 100
    print(f"  Val AI% at tuned t: {pred_ai:.2f}%")

    # Test probas
    test_meta_proba = meta.predict_proba(X_test)[:, 1]
    print(f"\nTest meta probas: min={test_meta_proba.min():.4f} max={test_meta_proba.max():.4f}"
          f" mean={test_meta_proba.mean():.4f} median={np.median(test_meta_proba):.4f}")
    for t in [0.3, 0.5, 0.7, best_t]:
        pct = (test_meta_proba >= t).mean() * 100
        print(f"  t={t:.3f}: {pct:.2f}% AI")

    # Write submissions at several target AI% levels
    print()
    for target in [60, 55, 50, 45]:
        t_q = np.quantile(test_meta_proba, 1 - target/100.0)
        labels = (test_meta_proba >= t_q).astype(int)
        sub = pd.DataFrame({"ID": test_ids, "label": labels})
        name = f'submission_stack_{"".join(feat_cols)}_q{target}.csv'
        sub.to_csv(name, index=False)
        print(f"  wrote {name} (t={t_q:.4f}, AI%={target})")

    # Also write at the val-tuned threshold
    labels_opt = (test_meta_proba >= best_t).astype(int)
    sub = pd.DataFrame({"ID": test_ids, "label": labels_opt})
    name_opt = f'submission_stack_{"".join(feat_cols)}_valt.csv'
    sub.to_csv(name_opt, index=False)
    print(f"  wrote {name_opt} (t={best_t:.4f}, AI%={labels_opt.mean()*100:.2f})")

    np.save('test_proba_stack.npy', test_meta_proba)


if __name__ == '__main__':
    main()
