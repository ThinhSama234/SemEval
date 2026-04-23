"""
Perplexity-Guided Feature Selection & EDA
==========================================
Uses perplexity as an anchor signal to discover which handcrafted features
provide the most COMPLEMENTARY information, then outputs a curated feature
set ready for ensemble training.

Strategy:
  1. Residual Analysis: Train a PPL-only model → find misclassified samples →
     rank features by discriminative power on those "hard" samples.
  2. Conditional Mutual Information: Features with high MI(feature, label | ppl)
     add information beyond perplexity.
  3. Perplexity-Stratified Importance: Feature importance in low/mid/high PPL bins.
  4. Redundancy Filter: Remove features highly correlated with PPL (redundant).
  5. Final Ranking: Combine all signals into a composite score.

Usage:
  python ppl_guided_selection.py
  python ppl_guided_selection.py --top_k 30
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def load_data(train_feat_path, train_ppl_path, val_feat_path):
    """Load and merge train features with perplexity."""
    train = pd.read_parquet(train_feat_path)
    ppl = pd.read_parquet(train_ppl_path)

    # Merge perplexity columns
    ppl_cols = ['overall_ppl']
    for col in ['line_ppl_mean', 'line_ppl_std', 'line_ppl_max', 'line_ppl_min', 'ppl_variance']:
        if col in ppl.columns:
            ppl_cols.append(col)

    for col in ppl_cols:
        train[col] = ppl[col].values

    val = pd.read_parquet(val_feat_path)
    # Fill val ppl with train median
    for col in ppl_cols:
        val[col] = train[col].median()

    return train, val, ppl_cols


def get_feature_cols(df):
    """Get feature columns (exclude metadata)."""
    exclude = {'label', 'code', 'generator', 'language', 'ID'}
    return [c for c in df.columns if c not in exclude]


def prep(df, cols):
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


# =============================================================================
# 1. RESIDUAL ANALYSIS
# =============================================================================
def residual_analysis(train, ppl_cols):
    """
    Train a model using ONLY perplexity features → find misclassified samples →
    rank other features by how well they separate correct vs incorrect predictions.
    """
    print("\n" + "=" * 70)
    print("1. RESIDUAL ANALYSIS: Which features help where PPL fails?")
    print("=" * 70)

    y = train['label']
    X_ppl = prep(train, ppl_cols)

    # Train PPL-only XGBoost
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        random_state=42, tree_method='hist', n_jobs=-1, verbosity=0,
    )
    model.fit(X_ppl, y)
    ppl_proba = model.predict_proba(X_ppl)[:, 1]

    # Find optimal threshold
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.2, 0.8, 0.01):
        f1 = f1_score(y, (ppl_proba >= t).astype(int), average='macro')
        if f1 > best_f1:
            best_f1, best_t = f1, t

    ppl_pred = (ppl_proba >= best_t).astype(int)
    misclassified = (ppl_pred != y)

    print(f"  PPL-only model: Macro F1 = {best_f1:.4f} (threshold={best_t:.2f})")
    print(f"  Misclassified samples: {misclassified.sum()} / {len(y)} ({misclassified.mean()*100:.1f}%)")

    # For misclassified samples, compute Cohen's d for each feature
    all_feats = get_feature_cols(train)
    non_ppl_feats = [f for f in all_feats if f not in ppl_cols]

    residual_scores = {}
    mis_mask = misclassified.values
    mis_data = train[mis_mask]
    mis_y = y[mis_mask]

    for feat in non_ppl_feats:
        vals = mis_data[feat].replace([np.inf, -np.inf], np.nan).fillna(0)
        h = vals[mis_y == 0]
        a = vals[mis_y == 1]
        if len(h) < 10 or len(a) < 10:
            continue
        pooled_std = np.sqrt((h.std()**2 + a.std()**2) / 2)
        if pooled_std < 1e-10:
            continue
        d = abs(h.mean() - a.mean()) / pooled_std
        residual_scores[feat] = d

    # Sort by discriminative power on hard samples
    ranked = sorted(residual_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  Top 20 features that discriminate PPL-misclassified samples:")
    print(f"  {'Feature':<35} {'Cohen d (on hard samples)':<25}")
    print(f"  {'-'*35} {'-'*25}")
    for feat, d in ranked[:20]:
        print(f"  {feat:<35} {d:.4f}")

    return {feat: score for feat, score in ranked}


# =============================================================================
# 2. CONDITIONAL MUTUAL INFORMATION
# =============================================================================
def conditional_mi(train, ppl_cols):
    """
    Compute MI(feature, label | perplexity) by discretizing PPL into bins,
    computing MI within each bin, and averaging.
    """
    print("\n" + "=" * 70)
    print("2. CONDITIONAL MUTUAL INFORMATION: MI(feature, label | ppl)")
    print("=" * 70)

    y = train['label'].values
    ppl_vals = train['overall_ppl'].replace([np.inf, -np.inf], np.nan).fillna(0).values

    # Discretize PPL into 5 quantile bins
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    ppl_bins = discretizer.fit_transform(ppl_vals.reshape(-1, 1)).ravel().astype(int)

    all_feats = get_feature_cols(train)
    non_ppl_feats = [f for f in all_feats if f not in ppl_cols]

    # Discretize features for MI calculation
    cmi_scores = {}
    for feat in non_ppl_feats:
        feat_vals = train[feat].replace([np.inf, -np.inf], np.nan).fillna(0).values
        # Discretize feature into 10 bins
        try:
            feat_disc = KBinsDiscretizer(
                n_bins=10, encode='ordinal', strategy='quantile'
            ).fit_transform(feat_vals.reshape(-1, 1)).ravel().astype(int)
        except ValueError:
            # Too few unique values
            feat_disc = feat_vals

        # Weighted average MI across PPL bins
        total_mi = 0
        total_n = 0
        for b in range(5):
            mask = ppl_bins == b
            n_b = mask.sum()
            if n_b < 50:
                continue
            mi = mutual_info_score(y[mask], feat_disc[mask])
            total_mi += mi * n_b
            total_n += n_b

        if total_n > 0:
            cmi_scores[feat] = total_mi / total_n

    # Also compute unconditional MI for comparison
    umi_scores = {}
    for feat in non_ppl_feats:
        feat_vals = train[feat].replace([np.inf, -np.inf], np.nan).fillna(0).values
        try:
            feat_disc = KBinsDiscretizer(
                n_bins=10, encode='ordinal', strategy='quantile'
            ).fit_transform(feat_vals.reshape(-1, 1)).ravel().astype(int)
        except ValueError:
            feat_disc = feat_vals
        umi_scores[feat] = mutual_info_score(y, feat_disc)

    # Ratio: CMI / MI → features close to 1.0 retain info even after conditioning on PPL
    ranked = sorted(cmi_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  Top 20 features by Conditional MI (info beyond PPL):")
    print(f"  {'Feature':<35} {'CMI':<12} {'Raw MI':<12} {'Retain%':<10}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")
    for feat, cmi in ranked[:20]:
        umi = umi_scores.get(feat, 0.001)
        retain = (cmi / umi * 100) if umi > 0.001 else 0
        print(f"  {feat:<35} {cmi:.4f}      {umi:.4f}      {retain:.1f}%")

    return cmi_scores


# =============================================================================
# 3. PERPLEXITY-STRATIFIED FEATURE IMPORTANCE
# =============================================================================
def stratified_importance(train, ppl_cols):
    """
    Split data into low/mid/high PPL terciles, train XGBoost in each stratum,
    and see which features are important in each regime.
    """
    print("\n" + "=" * 70)
    print("3. PERPLEXITY-STRATIFIED IMPORTANCE")
    print("=" * 70)

    y = train['label']
    ppl_vals = train['overall_ppl'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Terciles
    q33, q66 = ppl_vals.quantile(0.33), ppl_vals.quantile(0.66)
    strata = {
        'LOW_PPL  (easy for LLM)': ppl_vals <= q33,
        'MID_PPL  (ambiguous)':    (ppl_vals > q33) & (ppl_vals <= q66),
        'HIGH_PPL (hard for LLM)': ppl_vals > q66,
    }

    all_feats = get_feature_cols(train)
    non_ppl_feats = [f for f in all_feats if f not in ppl_cols]

    strat_importance = {}  # feat → list of importances across strata

    for stratum_name, mask in strata.items():
        X_s = prep(train[mask], non_ppl_feats)
        y_s = y[mask]

        if len(y_s.unique()) < 2:
            print(f"  {stratum_name}: Skipped (single class)")
            continue

        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            random_state=42, tree_method='hist', n_jobs=-1, verbosity=0,
        )
        model.fit(X_s, y_s)

        proba = model.predict_proba(X_s)[:, 1]
        best_f1 = 0
        for t in np.arange(0.2, 0.8, 0.01):
            f1 = f1_score(y_s, (proba >= t).astype(int), average='macro')
            if f1 > best_f1:
                best_f1 = f1

        importance = model.get_booster().get_score(importance_type='gain')
        print(f"\n  {stratum_name} (n={mask.sum()}, F1={best_f1:.4f}):")

        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feat, gain in sorted_imp:
            print(f"    {feat:<35} gain={gain:.1f}")

        for feat, gain in importance.items():
            if feat not in strat_importance:
                strat_importance[feat] = []
            strat_importance[feat].append(gain)

    # Average importance across strata
    avg_strat = {f: np.mean(v) for f, v in strat_importance.items()}
    return avg_strat


# =============================================================================
# 4. REDUNDANCY FILTER (correlation with PPL)
# =============================================================================
def redundancy_filter(train, ppl_cols):
    """
    Compute correlation of each feature with overall_ppl.
    Features highly correlated with PPL are redundant — they don't add new info.
    """
    print("\n" + "=" * 70)
    print("4. REDUNDANCY FILTER: Correlation with PPL")
    print("=" * 70)

    all_feats = get_feature_cols(train)
    non_ppl_feats = [f for f in all_feats if f not in ppl_cols]

    ppl_vals = train['overall_ppl'].replace([np.inf, -np.inf], np.nan).fillna(0)

    correlations = {}
    for feat in non_ppl_feats:
        feat_vals = train[feat].replace([np.inf, -np.inf], np.nan).fillna(0)
        corr = abs(feat_vals.corr(ppl_vals))
        correlations[feat] = corr if not np.isnan(corr) else 0

    # Sort by LOWEST correlation (most orthogonal = best complement)
    ranked = sorted(correlations.items(), key=lambda x: x[1])

    print(f"\n  Most ORTHOGONAL features (low corr with PPL = good complement):")
    print(f"  {'Feature':<35} {'|corr with PPL|':<18}")
    print(f"  {'-'*35} {'-'*18}")
    for feat, corr in ranked[:15]:
        print(f"  {feat:<35} {corr:.4f}")

    print(f"\n  Most REDUNDANT features (high corr with PPL):")
    for feat, corr in sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feat:<35} {corr:.4f}")

    return correlations


# =============================================================================
# 5. COMPOSITE RANKING
# =============================================================================
def composite_ranking(residual_scores, cmi_scores, strat_scores, correlations, top_k=30):
    """
    Combine all signals into a final feature ranking.
    Score = normalized(residual) + normalized(CMI) + normalized(strat_imp) + normalized(1 - corr_with_ppl)
    """
    print("\n" + "=" * 70)
    print("5. COMPOSITE RANKING: Final feature selection")
    print("=" * 70)

    all_feats = set(residual_scores) | set(cmi_scores) | set(strat_scores) | set(correlations)

    def normalize(d):
        if not d:
            return {}
        vals = np.array(list(d.values()))
        mn, mx = vals.min(), vals.max()
        if mx - mn < 1e-10:
            return {k: 0.5 for k in d}
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}

    n_res = normalize(residual_scores)
    n_cmi = normalize(cmi_scores)
    n_strat = normalize(strat_scores)
    n_corr = {k: 1 - v for k, v in normalize(correlations).items()}  # invert: low corr = high score

    # Weighted combination
    W_RESIDUAL = 2.0   # Most important: helps where PPL fails
    W_CMI = 1.5        # Info beyond PPL
    W_STRAT = 1.0      # Consistent importance across PPL regimes
    W_ORTHOGONAL = 1.0 # Complementary to PPL

    composite = {}
    for feat in all_feats:
        score = 0
        score += W_RESIDUAL * n_res.get(feat, 0)
        score += W_CMI * n_cmi.get(feat, 0)
        score += W_STRAT * n_strat.get(feat, 0)
        score += W_ORTHOGONAL * n_corr.get(feat, 0)
        composite[feat] = score

    ranked = sorted(composite.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  FINAL FEATURE RANKING (top {top_k}):")
    print(f"  {'Rank':<6} {'Feature':<35} {'Score':<8} {'Residual':<10} {'CMI':<10} {'Strat':<10} {'Orthog':<10}")
    print(f"  {'-'*6} {'-'*35} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    selected_features = []
    for i, (feat, score) in enumerate(ranked[:top_k]):
        r = n_res.get(feat, 0)
        c = n_cmi.get(feat, 0)
        s = n_strat.get(feat, 0)
        o = n_corr.get(feat, 0)
        print(f"  {i+1:<6} {feat:<35} {score:.3f}   {r:.3f}     {c:.3f}     {s:.3f}     {o:.3f}")
        selected_features.append(feat)

    # Show features that SHOULD BE DROPPED
    dropped = [feat for feat, _ in ranked[top_k:]]
    print(f"\n  Features dropped ({len(dropped)}):")
    for feat in dropped[:15]:
        print(f"    - {feat} (score={composite[feat]:.3f})")
    if len(dropped) > 15:
        print(f"    ... and {len(dropped)-15} more")

    return selected_features


# =============================================================================
# 6. VALIDATE: Compare full features vs selected features
# =============================================================================
def validate_selection(train, val, selected_features, ppl_cols):
    """Train models with all features vs selected features + PPL to confirm improvement."""
    print("\n" + "=" * 70)
    print("6. VALIDATION: Selected features + PPL vs All features + PPL")
    print("=" * 70)

    y_train = train['label']
    y_val = val['label']

    all_feats = get_feature_cols(train)
    non_ppl_feats = [f for f in all_feats if f not in ppl_cols]

    configs = {
        'PPL only':           ppl_cols,
        'All features (no PPL)': non_ppl_feats,
        'All features + PPL': non_ppl_feats + ppl_cols,
        'Selected + PPL':     selected_features + ppl_cols,
    }

    for name, feats in configs.items():
        feats = [f for f in feats if f in train.columns]
        X_tr = prep(train, feats)
        X_vl = prep(val, feats).reindex(columns=X_tr.columns, fill_value=0)

        model = xgb.XGBClassifier(
            n_estimators=1000, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            random_state=42, tree_method='hist', n_jobs=-1, verbosity=0,
            early_stopping_rounds=50, eval_metric='logloss',
        )
        model.fit(X_tr, y_train, eval_set=[(X_vl, y_val)], verbose=False)

        proba = model.predict_proba(X_vl)[:, 1]
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.2, 0.8, 0.01):
            f1 = f1_score(y_val, (proba >= t).astype(int), average='macro')
            if f1 > best_f1:
                best_f1, best_t = f1, t

        pred = (proba >= best_t).astype(int)
        print(f"\n  {name} ({len(feats)} features):")
        print(f"    Macro F1 = {best_f1:.4f}  (threshold={best_t:.2f})")
        print(f"    Pred dist: Human={( pred==0).sum()}, AI={(pred==1).sum()}")

    return configs['Selected + PPL']


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_feat", default="task_A/train_features_ml_ready.parquet")
    parser.add_argument("--train_ppl", default="train_perplexity.parquet")
    parser.add_argument("--val_feat", default="task_A/val_features_ml_ready.parquet")
    parser.add_argument("--top_k", type=int, default=30, help="Number of features to select")
    parser.add_argument("--output", default="selected_features.txt", help="Save selected feature names")
    args = parser.parse_args()

    print("PPL-Guided Feature Selection")
    print("=" * 70)

    # Load data
    train, val, ppl_cols = load_data(args.train_feat, args.train_ppl, args.val_feat)
    print(f"Train: {train.shape}, Val: {val.shape}")
    print(f"PPL columns: {ppl_cols}")

    all_feats = get_feature_cols(train)
    non_ppl = [f for f in all_feats if f not in ppl_cols]
    print(f"Total features: {len(all_feats)} ({len(non_ppl)} non-PPL + {len(ppl_cols)} PPL)")

    # Run analyses
    residual_scores = residual_analysis(train, ppl_cols)
    cmi_scores = conditional_mi(train, ppl_cols)
    strat_scores = stratified_importance(train, ppl_cols)
    correlations = redundancy_filter(train, ppl_cols)

    # Composite ranking
    selected = composite_ranking(residual_scores, cmi_scores, strat_scores, correlations, top_k=args.top_k)

    # Validate
    validate_selection(train, val, selected, ppl_cols)

    # Save selected features
    with open(args.output, 'w') as f:
        f.write("# PPL-Guided Selected Features\n")
        f.write("# These features complement perplexity best.\n")
        f.write("# Generated by ppl_guided_selection.py\n\n")
        for feat in selected:
            f.write(feat + '\n')
        f.write("\n# PPL features (always included):\n")
        for col in ppl_cols:
            f.write(col + '\n')

    print(f"\nSelected features saved to {args.output}")
    print(f"\nTo use in ensemble training:")
    print(f"  SELECTED_FEATURES = {selected}")


if __name__ == "__main__":
    main()
