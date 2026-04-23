"""
v8: Tuned IF+CNB style-only with multiple configurations
=========================================================
Tries different contamination, n_estimators, and classifier combinations
to find the most generalizable configuration.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from itertools import product

FEAT_COLS = [
    "comment_ratio", "blank_line_ratio", "indentation_std", "line_len_std",
    "style_consistency", "ttr", "comment_completeness", "blank_per_function",
    "comment_per_function", "trailing_ws_ratio", "naming_uniformity",
    "line_len_burstiness", "token_entropy", "inline_comment_ratio", "keyword_density",
    "max_nesting_depth", "avg_block_length", "cyclomatic_proxy",
    "comment_word_count_avg", "function_size_regularity",
]


def optimize_threshold(proba, y_true):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.20, 0.80, 0.005):
        f1 = f1_score(y_true, (proba >= t).astype(int), average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def train_if_cnb(X_train, y_train, X_val, y_val,
                 contamination=0.05, n_estimators=200, classifier='cnb'):
    """Train IF+classifier pipeline with given params."""
    qt = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_train_qt = qt.fit_transform(X_train)
    X_val_qt = qt.transform(X_val)

    mask_ai = (y_train == 1).values
    mask_hum = (y_train == 0).values

    iso_ai = IsolationForest(
        contamination=contamination, random_state=42,
        n_estimators=n_estimators, n_jobs=-1, max_features=0.8
    )
    iso_hum = IsolationForest(
        contamination=contamination, random_state=42,
        n_estimators=n_estimators, n_jobs=-1, max_features=0.8
    )
    iso_ai.fit(X_train_qt[mask_ai])
    iso_hum.fit(X_train_qt[mask_hum])

    # Anomaly scores
    s_ai_train = iso_ai.decision_function(X_train_qt)
    s_hum_train = iso_hum.decision_function(X_train_qt)
    S_train = np.column_stack([s_ai_train, s_hum_train])

    s_ai_val = iso_ai.decision_function(X_val_qt)
    s_hum_val = iso_hum.decision_function(X_val_qt)
    S_val = np.column_stack([s_ai_val, s_hum_val])

    # Also add raw score difference as feature
    S_train_ext = np.column_stack([S_train, S_train[:, 0] - S_train[:, 1]])
    S_val_ext = np.column_stack([S_val, S_val[:, 0] - S_val[:, 1]])

    if classifier == 'cnb':
        s_min = S_train_ext.min(axis=0)
        S_train_shifted = S_train_ext - s_min + 1e-6
        S_val_shifted = S_val_ext - s_min + 1e-6
        clf = ComplementNB(alpha=1.0)
        clf.fit(S_train_shifted, y_train)
        proba_val = clf.predict_proba(S_val_shifted)[:, 1]
        extra = {'s_min': s_min}
    elif classifier == 'lr':
        scaler = StandardScaler()
        S_train_sc = scaler.fit_transform(S_train_ext)
        S_val_sc = scaler.transform(S_val_ext)
        clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        clf.fit(S_train_sc, y_train)
        proba_val = clf.predict_proba(S_val_sc)[:, 1]
        extra = {'scaler': scaler}
    elif classifier == 'direct':
        # Just use score difference directly
        diff_val = S_val[:, 0] - S_val[:, 1]
        # Normalize to [0, 1]
        diff_all = np.concatenate([S_train[:, 0] - S_train[:, 1], diff_val])
        d_min, d_max = diff_all.min(), diff_all.max()
        proba_val = (diff_val - d_min) / (d_max - d_min + 1e-10)
        clf = None
        extra = {'d_min': d_min, 'd_max': d_max}

    t, f1 = optimize_threshold(proba_val, y_val)

    pipeline = {
        'qt': qt, 'iso_ai': iso_ai, 'iso_hum': iso_hum,
        'clf': clf, 'classifier_type': classifier,
        'threshold': t, **extra
    }
    return pipeline, t, f1, proba_val


def predict_pipeline(pipeline, X):
    X_qt = pipeline['qt'].transform(X)
    s_ai = pipeline['iso_ai'].decision_function(X_qt)
    s_hum = pipeline['iso_hum'].decision_function(X_qt)
    S = np.column_stack([s_ai, s_hum])
    S_ext = np.column_stack([S, S[:, 0] - S[:, 1]])

    ctype = pipeline['classifier_type']
    if ctype == 'cnb':
        S_shifted = S_ext - pipeline['s_min'] + 1e-6
        proba = pipeline['clf'].predict_proba(S_shifted)[:, 1]
    elif ctype == 'lr':
        S_sc = pipeline['scaler'].transform(S_ext)
        proba = pipeline['clf'].predict_proba(S_sc)[:, 1]
    elif ctype == 'direct':
        diff = S[:, 0] - S[:, 1]
        proba = (diff - pipeline['d_min']) / (pipeline['d_max'] - pipeline['d_min'] + 1e-10)

    return proba


def main():
    print("Loading features...")
    train_df = pd.read_parquet('train_style_features.parquet')
    X_train = train_df[FEAT_COLS]
    y_train = train_df['label']

    val_df = pd.read_parquet('val_style_features.parquet')
    X_val = val_df[FEAT_COLS]
    y_val = val_df['label']

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Grid search over configurations
    configs = []
    for contam in [0.01, 0.05, 0.10, 0.15, 0.20]:
        for n_est in [100, 200, 500]:
            for clf_type in ['cnb', 'lr', 'direct']:
                configs.append((contam, n_est, clf_type))

    print(f"\nTesting {len(configs)} configurations...")
    results = []
    for i, (contam, n_est, clf_type) in enumerate(configs):
        try:
            pipe, t, f1, _ = train_if_cnb(
                X_train, y_train, X_val, y_val,
                contamination=contam, n_estimators=n_est, classifier=clf_type
            )
            results.append((contam, n_est, clf_type, t, f1, pipe))
            print(f"  [{i+1}/{len(configs)}] contam={contam:.2f} n_est={n_est} clf={clf_type:6s} → F1={f1:.4f} (t={t:.3f})")
        except Exception as e:
            print(f"  [{i+1}/{len(configs)}] FAILED: {e}")

    # Sort by F1
    results.sort(key=lambda x: x[4], reverse=True)
    print(f"\n{'='*60}")
    print("TOP 5 CONFIGURATIONS")
    print(f"{'='*60}")
    for contam, n_est, clf_type, t, f1, _ in results[:5]:
        print(f"  contam={contam:.2f} n_est={n_est} clf={clf_type:6s} → F1={f1:.4f} (t={t:.3f})")

    # Use best config
    best = results[0]
    best_pipe = best[5]
    print(f"\nBest: contam={best[0]:.2f} n_est={best[1]} clf={best[2]} F1={best[4]:.4f}")

    # Full val evaluation
    proba_val = predict_pipeline(best_pipe, X_val)
    y_pred_val = (proba_val >= best[3]).astype(int)
    print(f"\nAccuracy: {accuracy_score(y_val, y_pred_val):.4f}")
    print(classification_report(y_val, y_pred_val))

    # Test inference
    print("\nTest inference...")
    test_df = pd.read_parquet('test_style_features.parquet')
    X_test = test_df[FEAT_COLS]
    test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.RangeIndex(len(test_df))

    proba_test = predict_pipeline(best_pipe, X_test)
    y_pred_test = (proba_test >= best[3]).astype(int)

    print(f"Test dist: Human={(y_pred_test==0).sum()} ({(y_pred_test==0).mean()*100:.1f}%), "
          f"AI={(y_pred_test==1).sum()} ({(y_pred_test==1).mean()*100:.1f}%)")

    sub = pd.DataFrame({"ID": test_ids, "label": y_pred_test})
    sub.to_csv("submission_v8.csv", index=False)
    print(f"Saved submission_v8.csv ({len(sub)} rows)")

    # Also save top 3 submissions for comparison
    for rank, (contam, n_est, clf_type, t, f1, pipe) in enumerate(results[:3]):
        proba_t = predict_pipeline(pipe, X_test)
        y_t = (proba_t >= t).astype(int)
        fname = f"submission_v8_rank{rank+1}.csv"
        pd.DataFrame({"ID": test_ids, "label": y_t}).to_csv(fname, index=False)
        print(f"  {fname}: Human={(y_t==0).sum()}, AI={(y_t==1).sum()}")

    joblib.dump({'pipeline': best_pipe, 'threshold': best[3], 'config': best[:5]}, 'taskA_v8.pkl')


if __name__ == "__main__":
    main()
