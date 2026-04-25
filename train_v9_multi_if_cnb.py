"""
v9: Multi-IF + CNB — Improved isolation-based pipeline
======================================================
Key ideas:
  1. Multiple IsolationForest pairs trained on different feature GROUPS
     -> each pair produces (s_ai, s_hum) -> richer anomaly representation
  2. Curated features: remove dataset artifacts (tab/whitespace patterns)
  3. ComplementNB on all anomaly scores -> simple, regularized classifier

Architecture:
  features -> [group_1, group_2, ..., group_K]
           -> IF_ai_k, IF_hum_k for each group
           -> [s_ai_1, s_hum_1, diff_1, ..., s_ai_K, s_hum_K, diff_K]
           -> QuantileTransform (shift non-negative)
           -> ComplementNB -> P(AI)
           -> threshold -> label
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import f1_score, accuracy_score, classification_report

# =============================================================================
# Feature groups — semantically meaningful, NO tab/whitespace artifacts
# =============================================================================
STYLE_FEATURES = [
    "comment_ratio", "blank_line_ratio", "indentation_std", "line_len_std",
    "style_consistency", "ttr", "comment_completeness", "blank_per_function",
    "comment_per_function", "trailing_ws_ratio", "naming_uniformity",
    "line_len_burstiness", "token_entropy", "inline_comment_ratio",
    "keyword_density", "max_nesting_depth", "avg_block_length",
    "cyclomatic_proxy", "comment_word_count_avg", "function_size_regularity",
]

# Feature groups for multi-IF
FEATURE_GROUPS = {
    "entropy_complexity": [
        "token_entropy", "line_len_burstiness", "indentation_std",
        "line_len_std", "ttr", "style_consistency",
    ],
    "comment_doc": [
        "comment_ratio", "comment_completeness", "comment_per_function",
        "inline_comment_ratio", "comment_word_count_avg", "blank_per_function",
    ],
    "structure": [
        "max_nesting_depth", "avg_block_length", "cyclomatic_proxy",
        "function_size_regularity", "keyword_density", "naming_uniformity",
    ],
    "surface": [
        "blank_line_ratio", "trailing_ws_ratio", "line_len_std",
        "indentation_std", "ttr", "token_entropy",
    ],
    "all_style": STYLE_FEATURES,  # full 20 features as one group too
}


def optimize_threshold(proba, y_true, low=0.20, high=0.80, step=0.005):
    best_f1, best_t = 0, 0.5
    for t in np.arange(low, high, step):
        f1 = f1_score(y_true, (proba >= t).astype(int), average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


class MultiIFCNB:
    """Multi-group IsolationForest + ComplementNB pipeline."""

    def __init__(self, feature_groups, contamination=0.05, n_estimators=200,
                 max_features=0.8, cnb_alpha=1.0):
        self.feature_groups = feature_groups
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.cnb_alpha = cnb_alpha

        # Will be set during fit
        self.qt_dict = {}      # QuantileTransformer per group
        self.iso_ai_dict = {}  # IF for AI class per group
        self.iso_hum_dict = {} # IF for Human class per group
        self.cnb = None
        self.score_shift = None

    def _build_anomaly_scores(self, X_df, fit=False, y=None):
        """Build anomaly score matrix from all IF groups."""
        scores = []

        for group_name, feat_cols in self.feature_groups.items():
            X_group = X_df[feat_cols].values

            if fit:
                # Fit QT
                qt = QuantileTransformer(output_distribution='uniform', random_state=42)
                X_qt = qt.fit_transform(X_group)
                self.qt_dict[group_name] = qt

                # Fit dual IF
                mask_ai = (y == 1).values
                mask_hum = (y == 0).values

                iso_ai = IsolationForest(
                    contamination=self.contamination, random_state=42,
                    n_estimators=self.n_estimators, max_features=self.max_features,
                    n_jobs=-1
                )
                iso_hum = IsolationForest(
                    contamination=self.contamination, random_state=42,
                    n_estimators=self.n_estimators, max_features=self.max_features,
                    n_jobs=-1
                )
                iso_ai.fit(X_qt[mask_ai])
                iso_hum.fit(X_qt[mask_hum])
                self.iso_ai_dict[group_name] = iso_ai
                self.iso_hum_dict[group_name] = iso_hum
            else:
                qt = self.qt_dict[group_name]
                X_qt = qt.transform(X_group)
                iso_ai = self.iso_ai_dict[group_name]
                iso_hum = self.iso_hum_dict[group_name]

            s_ai = iso_ai.decision_function(X_qt)
            s_hum = iso_hum.decision_function(X_qt)
            s_diff = s_ai - s_hum

            scores.extend([s_ai, s_hum, s_diff])

        return np.column_stack(scores)

    def fit(self, X_train_df, y_train):
        """Fit the full pipeline."""
        print(f"  Building anomaly scores from {len(self.feature_groups)} groups...")
        S_train = self._build_anomaly_scores(X_train_df, fit=True, y=y_train)
        print(f"  Score matrix shape: {S_train.shape}")

        # Shift to non-negative for CNB
        self.score_shift = S_train.min(axis=0)
        S_shifted = S_train - self.score_shift + 1e-6

        print(f"  Fitting ComplementNB (alpha={self.cnb_alpha})...")
        self.cnb = ComplementNB(alpha=self.cnb_alpha)
        self.cnb.fit(S_shifted, y_train)

        return self

    def predict_proba(self, X_df):
        """Get P(AI) probabilities."""
        S = self._build_anomaly_scores(X_df, fit=False)
        S_shifted = S - self.score_shift + 1e-6
        return self.cnb.predict_proba(S_shifted)[:, 1]

    def predict(self, X_df, threshold=0.5):
        proba = self.predict_proba(X_df)
        return (proba >= threshold).astype(int), proba


def main():
    print("Loading features...")
    train_df = pd.read_parquet('train_style_features.parquet')
    X_train = train_df[STYLE_FEATURES]
    y_train = train_df['label']

    val_df = pd.read_parquet('val_style_features.parquet')
    X_val = val_df[STYLE_FEATURES]
    y_val = val_df['label']

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Train labels: {y_train.value_counts().to_dict()}")

    # =========================================================================
    # Grid search over key hyperparameters
    # =========================================================================
    configs = []
    for contam in [0.02, 0.05, 0.10, 0.15]:
        for n_est in [150, 300]:
            for alpha in [0.5, 1.0, 2.0]:
                configs.append((contam, n_est, alpha))

    print(f"\nTesting {len(configs)} configurations...")
    results = []

    for i, (contam, n_est, alpha) in enumerate(configs):
        model = MultiIFCNB(
            feature_groups=FEATURE_GROUPS,
            contamination=contam,
            n_estimators=n_est,
            cnb_alpha=alpha,
        )
        model.fit(X_train, y_train)
        proba_val = model.predict_proba(X_val)
        t, f1 = optimize_threshold(proba_val, y_val)

        results.append((contam, n_est, alpha, t, f1, model, proba_val))
        print(f"  [{i+1}/{len(configs)}] contam={contam:.2f} n_est={n_est} alpha={alpha:.1f}"
              f" -> F1={f1:.4f} (t={t:.3f})")

    results.sort(key=lambda x: x[4], reverse=True)

    print(f"\n{'='*60}")
    print("TOP 5 CONFIGURATIONS")
    print(f"{'='*60}")
    for contam, n_est, alpha, t, f1, _, _ in results[:5]:
        print(f"  contam={contam:.2f} n_est={n_est} alpha={alpha:.1f} -> F1={f1:.4f} (t={t:.3f})")

    # =========================================================================
    # Best model evaluation
    # =========================================================================
    best = results[0]
    best_model = best[5]
    best_t = best[3]
    best_f1 = best[4]

    print(f"\n{'='*60}")
    print(f"BEST: contam={best[0]:.2f} n_est={best[1]} alpha={best[2]:.1f}")
    print(f"{'='*60}")

    y_pred_val, proba_val = best_model.predict(X_val, threshold=best_t)
    print(f"Threshold: {best_t:.4f}")
    print(f"Accuracy:  {accuracy_score(y_val, y_pred_val):.4f}")
    print(f"F1 Macro:  {f1_score(y_val, y_pred_val, average='macro'):.4f}")
    print(classification_report(y_val, y_pred_val))
    print(f"Pred dist: Human={(y_pred_val==0).sum()}, AI={(y_pred_val==1).sum()}")

    # =========================================================================
    # Test inference
    # =========================================================================
    print("\nTest inference...")
    test_df = pd.read_parquet('test_style_features.parquet')
    X_test = test_df[STYLE_FEATURES]
    test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.RangeIndex(len(test_df))

    y_pred_test, proba_test = best_model.predict(X_test, threshold=best_t)
    print(f"Test dist: Human={(y_pred_test==0).sum()} ({(y_pred_test==0).mean()*100:.1f}%), "
          f"AI={(y_pred_test==1).sum()} ({(y_pred_test==1).mean()*100:.1f}%)")

    sub = pd.DataFrame({"ID": test_ids, "label": y_pred_test})
    sub.to_csv("submission_v9.csv", index=False)
    print(f"Saved submission_v9.csv ({len(sub)} rows)")

    # Also save runner-up submissions
    for rank in range(1, min(4, len(results))):
        r = results[rank]
        y_t, _ = r[5].predict(X_test, threshold=r[3])
        fname = f"submission_v9_rank{rank+1}.csv"
        pd.DataFrame({"ID": test_ids, "label": y_t}).to_csv(fname, index=False)
        h, a = (y_t==0).sum(), (y_t==1).sum()
        print(f"  {fname}: Human={h} ({h/len(y_t)*100:.1f}%), AI={a} ({a/len(y_t)*100:.1f}%)")

    joblib.dump({
        'model': best_model, 'threshold': best_t,
        'config': {'contam': best[0], 'n_est': best[1], 'alpha': best[2]},
    }, 'taskA_v9.pkl')
    print("Model saved to taskA_v9.pkl")


if __name__ == "__main__":
    main()
