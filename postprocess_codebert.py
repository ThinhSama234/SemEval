"""
Post-processing for CodeBERT predictions to handle distribution shift.

Strategies:
  1. Adaptive threshold — optimize threshold on val F1 (thay vi mac dinh 0.5)
  2. Language-aware threshold — threshold khac nhau cho moi ngon ngu
  3. Temperature scaling — calibrate confidence, giam overconfident
  4. Platt scaling — logistic regression tren logits/proba → calibrated proba
  5. Confidence masking — chi giu predictions co confidence cao, flip nhung cai thap
  6. Prior shift correction — dieu chinh theo ty le label expected trong test
  7. Ensemble with hybrid — ket hop CodeBERT + IsoForest model
  8. Isotonic regression — non-parametric calibration

Usage:
  python postprocess_codebert.py \
    --test_proba codebert_model/test_proba.npy \
    --val_proba codebert_model/val_proba.npy \
    --test_data test.parquet \
    --val_data task_A/validation.parquet
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import re
from collections import Counter


# =========================================================================
# Language detection (simple heuristic)
# =========================================================================
LANG_PATTERNS = {
    'python': [
        r'\bdef\s+\w+\s*\(', r'\bimport\s+\w+', r'\bclass\s+\w+.*:',
        r'\bself\.', r'\bprint\s*\(', r':\s*$',
    ],
    'java': [
        r'\bpublic\s+(static\s+)?void\b', r'\bSystem\.out\.print',
        r'\bimport\s+java\.', r'\bclass\s+\w+\s*\{',
        r'\bnew\s+\w+\s*\(', r'\bString\[\]\s+args\b',
    ],
    'cpp': [
        r'#include\s*<', r'\bstd::', r'\bcout\s*<<', r'\bcin\s*>>',
        r'\bint\s+main\s*\(', r'\bvector\s*<', r'\busing\s+namespace\b',
    ],
    'c': [
        r'#include\s*<stdio\.h>', r'\bprintf\s*\(', r'\bscanf\s*\(',
        r'\bmalloc\s*\(', r'\bfree\s*\(', r'\btypedef\s+struct\b',
    ],
    'javascript': [
        r'\bfunction\s+\w+\s*\(', r'\bconst\s+\w+\s*=', r'\blet\s+\w+\s*=',
        r'\bconsole\.log\b', r'\b=>\s*\{', r'\brequire\s*\(',
    ],
    'csharp': [
        r'\busing\s+System', r'\bnamespace\s+\w+', r'\bConsole\.Write',
        r'\bstatic\s+void\s+Main\b', r'\bvar\s+\w+\s*=',
    ],
}


def detect_language(code):
    if not isinstance(code, str):
        return 'unknown'
    scores = {}
    for lang, patterns in LANG_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, code, re.MULTILINE))
        scores[lang] = score
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'unknown'


# =========================================================================
# 1. Adaptive threshold — optimize threshold on val Macro F1
# =========================================================================
def optimize_threshold(proba, y_true):
    """Tim threshold toi uu Macro F1 tren val set."""
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.01, 0.99, 0.005):
        f1 = f1_score(y_true, (proba >= t).astype(int), average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


# =========================================================================
# 2. Language-aware threshold
# =========================================================================
def language_aware_predict(proba_val, y_val, proba_test, langs_val, langs_test):
    """Threshold khac nhau cho moi ngon ngu, optimize tren val."""
    unique_langs = set(langs_val) | set(langs_test)

    # Global fallback
    global_t, _ = optimize_threshold(proba_val, y_val)

    lang_thresholds = {}
    for lang in unique_langs:
        mask = np.array([l == lang for l in langs_val])
        if mask.sum() < 100:
            lang_thresholds[lang] = global_t
            continue
        t, _ = optimize_threshold(proba_val[mask], np.array(y_val)[mask])
        lang_thresholds[lang] = t

    y_test = np.array([
        int(p >= lang_thresholds.get(lang, global_t))
        for p, lang in zip(proba_test, langs_test)
    ])
    return y_test, lang_thresholds


# =========================================================================
# 3. Temperature Scaling
#    Neural nets thuong overconfident (predict 0.99 thay vi 0.7).
#    Scale logits / T truoc softmax. Tim T minimize NLL tren val.
#    T>1: giam confidence, T<1: tang confidence.
#    Luu y: chi calibrate, KHONG fix distribution shift.
# =========================================================================
def temperature_scale(logits_val, y_val, logits_test):
    """Tim T toi uu NLL tren val, apply cho test."""
    from scipy.optimize import minimize_scalar
    from scipy.special import expit

    def nll(T):
        p = expit(logits_val / T)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    best_T = result.x

    scaled_val = expit(logits_val / best_T)
    scaled_test = expit(logits_test / best_T)

    t, f1 = optimize_threshold(scaled_val, y_val)
    y_test = (scaled_test >= t).astype(int)
    return y_test, best_T, t, f1


# =========================================================================
# 4. Platt Scaling (Logistic Regression on probabilities)
#    Fit logistic regression: P(y=1) = sigmoid(a*proba + b)
#    Linh hoat hon temperature scaling vi co ca bias term.
#    Hoc duoc ca shift (b) va scale (a).
# =========================================================================
def platt_scaling(proba_val, y_val, proba_test):
    """Logistic regression tren proba → calibrated proba."""
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    lr.fit(proba_val.reshape(-1, 1), y_val)

    cal_val = lr.predict_proba(proba_val.reshape(-1, 1))[:, 1]
    cal_test = lr.predict_proba(proba_test.reshape(-1, 1))[:, 1]

    t, f1 = optimize_threshold(cal_val, y_val)
    y_test = (cal_test >= t).astype(int)
    return y_test, cal_test, t, f1, lr


# =========================================================================
# 5. Isotonic Regression (non-parametric calibration)
#    Khong gia dinh dang ham → linh hoat nhat.
#    Fit monotonic step function: proba → calibrated proba.
#    Tot khi model confidence khong linear voi true probability.
# =========================================================================
def isotonic_calibration(proba_val, y_val, proba_test):
    """Non-parametric calibration bang isotonic regression."""
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(proba_val, y_val)

    cal_val = iso.predict(proba_val)
    cal_test = iso.predict(proba_test)

    t, f1 = optimize_threshold(cal_val, y_val)
    y_test = (cal_test >= t).astype(int)
    return y_test, cal_test, t, f1


# =========================================================================
# 6. Prior Shift Correction (adjust for label distribution change)
#    Neu train co 52% AI nhung test co ~50% AI,
#    dieu chinh: P_new(y=1|x) ∝ P_old(y=1|x) * (π_test / π_train)
#    Khong can label cua test, chi can uoc luong ty le.
# =========================================================================
def prior_shift_correction(proba, train_ai_ratio, target_ai_ratio):
    """
    Dieu chinh proba theo ty le label moi.
    Formula: p_new = p * (target/train) / (p*(target/train) + (1-p)*((1-target)/(1-train)))
    """
    r = target_ai_ratio / train_ai_ratio
    r0 = (1 - target_ai_ratio) / (1 - train_ai_ratio)
    adjusted = (proba * r) / (proba * r + (1 - proba) * r0)
    return adjusted


def search_prior_shift(proba_val, y_val, proba_test, train_ai_ratio):
    """Thu nhieu target ratio, chon cai tot nhat theo val F1."""
    best_f1, best_ratio, best_t = 0, 0.5, 0.5
    for target in np.arange(0.30, 0.70, 0.01):
        adj_val = prior_shift_correction(proba_val, train_ai_ratio, target)
        t, f1 = optimize_threshold(adj_val, y_val)
        if f1 > best_f1:
            best_f1 = f1
            best_ratio = target
            best_t = t

    adj_test = prior_shift_correction(proba_test, train_ai_ratio, best_ratio)
    y_test = (adj_test >= best_t).astype(int)
    return y_test, adj_test, best_ratio, best_t, best_f1


# =========================================================================
# 7. Confidence Masking
#    Chi tin predictions co confidence cao (>= high_conf).
#    Nhung cai o giua (low confidence) → flip theo majority
#    hoac theo language-based prior.
# =========================================================================
def confidence_masking(proba_test, high_conf=0.8, low_conf=0.2, default_label=0):
    """
    Confident AI (proba >= high_conf) → 1
    Confident Human (proba <= low_conf) → 0
    Uncertain (giua) → default_label
    """
    y_test = np.full(len(proba_test), default_label, dtype=int)
    y_test[proba_test >= high_conf] = 1
    y_test[proba_test <= low_conf] = 0
    n_uncertain = ((proba_test > low_conf) & (proba_test < high_conf)).sum()
    return y_test, n_uncertain


def search_confidence_masking(proba_val, y_val, proba_test):
    """Grid search high/low confidence thresholds."""
    best_f1 = 0
    best_params = (0.8, 0.2, 0)

    for high in np.arange(0.6, 0.95, 0.05):
        for low in np.arange(0.05, 0.4, 0.05):
            if low >= high:
                continue
            for default in [0, 1]:
                y_pred = np.full(len(proba_val), default, dtype=int)
                y_pred[proba_val >= high] = 1
                y_pred[proba_val <= low] = 0
                f1 = f1_score(y_val, y_pred, average='macro')
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = (high, low, default)

    high, low, default = best_params
    y_test, n_unc = confidence_masking(proba_test, high, low, default)
    return y_test, best_params, best_f1, n_unc


# =========================================================================
# 8. Ensemble with hybrid model
# =========================================================================
def ensemble_with_hybrid(cb_proba_test, cb_proba_val, y_val,
                         hyb_proba_test, hyb_proba_val):
    """Weighted ensemble CodeBERT + hybrid, optimize weight + threshold."""
    best_f1, best_w, best_t = 0, 0.5, 0.5

    for w_cb in np.arange(0.1, 0.95, 0.05):
        avg_val = w_cb * cb_proba_val + (1 - w_cb) * hyb_proba_val
        t, f1 = optimize_threshold(avg_val, y_val)
        if f1 > best_f1:
            best_f1 = f1
            best_w = w_cb
            best_t = t

    avg_test = best_w * cb_proba_test + (1 - best_w) * hyb_proba_test
    y_test = (avg_test >= best_t).astype(int)
    return y_test, best_w, best_t, best_f1


# =========================================================================
# Main — chay tat ca strategies, so sanh, save best
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_proba", required=True, help="CodeBERT test proba (.npy)")
    parser.add_argument("--val_proba", default=None, help="CodeBERT val proba (.npy)")
    parser.add_argument("--test_logits", default=None, help="Raw logits (.npy) for temp scaling")
    parser.add_argument("--val_logits", default=None, help="Raw logits (.npy) for temp scaling")
    parser.add_argument("--test_data", required=True, help="Test parquet (code + ID)")
    parser.add_argument("--val_data", default="task_A/validation.parquet")
    parser.add_argument("--hybrid_val_proba", default=None, help="Hybrid val proba (.npy)")
    parser.add_argument("--hybrid_test_proba", default=None, help="Hybrid test proba (.npy)")
    parser.add_argument("--train_ai_ratio", type=float, default=0.523,
                        help="AI label ratio in train set")
    parser.add_argument("--submission_out", default="submission_codebert_pp.csv")
    args = parser.parse_args()

    # Load data
    test_proba = np.load(args.test_proba)
    test_df = pd.read_parquet(args.test_data)
    test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.RangeIndex(len(test_df))

    print(f"Test samples: {len(test_proba)}")
    raw_ai = (test_proba >= 0.5).mean() * 100
    print(f"Raw distribution (t=0.5): Human={(test_proba<0.5).sum()} ({100-raw_ai:.1f}%), "
          f"AI={(test_proba>=0.5).sum()} ({raw_ai:.1f}%)")

    # Detect languages
    print("\nDetecting test languages...")
    test_langs = [detect_language(c) for c in test_df['code']]
    lang_dist = Counter(test_langs)
    for lang, cnt in lang_dist.most_common():
        print(f"  {lang:12s}: {cnt:6d} ({cnt/len(test_langs)*100:.1f}%)")

    results = {}  # name → (y_test, val_f1, description)

    has_val = args.val_proba is not None
    if has_val:
        val_proba = np.load(args.val_proba)
        val_df = pd.read_parquet(args.val_data)[['code', 'label']].dropna()
        y_val = val_df['label'].astype(int).values

    # ================================================================
    print("\n" + "=" * 70)
    print("1. ADAPTIVE THRESHOLD")
    print("=" * 70)
    if has_val:
        t, f1 = optimize_threshold(val_proba, y_val)
        y_test = (test_proba >= t).astype(int)
        print(f"   Val F1={f1:.4f}, threshold={t:.4f}")
        _show_dist(y_test, "   ")
        results['adaptive'] = (y_test, f1)
    else:
        print("   SKIP (no val_proba)")

    # ================================================================
    print("\n" + "=" * 70)
    print("2. LANGUAGE-AWARE THRESHOLD")
    print("=" * 70)
    if has_val:
        val_langs = [detect_language(c) for c in val_df['code']]
        y_test, lang_t = language_aware_predict(val_proba, y_val, test_proba, val_langs, test_langs)
        for lang, t in sorted(lang_t.items()):
            cnt = lang_dist.get(lang, 0)
            print(f"   {lang:12s}: t={t:.3f} (n={cnt})")
        _show_dist(y_test, "   ")
        # Val F1 for this strategy
        y_val_lang = np.array([
            int(p >= lang_t.get(l, 0.5))
            for p, l in zip(val_proba, val_langs)
        ])
        f1_lang = f1_score(y_val, y_val_lang, average='macro')
        print(f"   Val F1={f1_lang:.4f}")
        results['lang_aware'] = (y_test, f1_lang)
    else:
        print("   SKIP (no val_proba)")

    # ================================================================
    print("\n" + "=" * 70)
    print("3. TEMPERATURE SCALING")
    print("=" * 70)
    if args.val_logits and args.test_logits:
        val_logits = np.load(args.val_logits)
        test_logits = np.load(args.test_logits)
        y_test, T, t, f1 = temperature_scale(val_logits, y_val, test_logits)
        print(f"   T={T:.3f}, threshold={t:.4f}, Val F1={f1:.4f}")
        _show_dist(y_test, "   ")
        results['temp_scale'] = (y_test, f1)
    else:
        print("   SKIP (no logits files)")

    # ================================================================
    print("\n" + "=" * 70)
    print("4. PLATT SCALING (logistic regression on proba)")
    print("=" * 70)
    if has_val:
        y_test, cal_test, t, f1, lr = platt_scaling(val_proba, y_val, test_proba)
        print(f"   a={lr.coef_[0][0]:.3f}, b={lr.intercept_[0]:.3f}")
        print(f"   threshold={t:.4f}, Val F1={f1:.4f}")
        _show_dist(y_test, "   ")
        results['platt'] = (y_test, f1)
    else:
        print("   SKIP (no val_proba)")

    # ================================================================
    print("\n" + "=" * 70)
    print("5. ISOTONIC REGRESSION CALIBRATION")
    print("=" * 70)
    if has_val:
        y_test, cal_test, t, f1 = isotonic_calibration(val_proba, y_val, test_proba)
        print(f"   threshold={t:.4f}, Val F1={f1:.4f}")
        _show_dist(y_test, "   ")
        results['isotonic'] = (y_test, f1)
    else:
        print("   SKIP (no val_proba)")

    # ================================================================
    print("\n" + "=" * 70)
    print("6. PRIOR SHIFT CORRECTION")
    print("=" * 70)
    if has_val:
        y_test, adj_test, ratio, t, f1 = search_prior_shift(
            val_proba, y_val, test_proba, args.train_ai_ratio)
        print(f"   target_ratio={ratio:.2f}, threshold={t:.4f}, Val F1={f1:.4f}")
        _show_dist(y_test, "   ")
        results['prior_shift'] = (y_test, f1)
    else:
        print("   SKIP (no val_proba)")

    # ================================================================
    print("\n" + "=" * 70)
    print("7. CONFIDENCE MASKING")
    print("=" * 70)
    if has_val:
        y_test, params, f1, n_unc = search_confidence_masking(val_proba, y_val, test_proba)
        high, low, default = params
        print(f"   high={high:.2f}, low={low:.2f}, default={default}, uncertain={n_unc}")
        print(f"   Val F1={f1:.4f}")
        _show_dist(y_test, "   ")
        results['conf_mask'] = (y_test, f1)
    else:
        print("   SKIP (no val_proba)")

    # ================================================================
    print("\n" + "=" * 70)
    print("8. ENSEMBLE WITH HYBRID")
    print("=" * 70)
    if has_val and args.hybrid_val_proba and args.hybrid_test_proba:
        hyb_val = np.load(args.hybrid_val_proba)
        hyb_test = np.load(args.hybrid_test_proba)
        y_test, w, t, f1 = ensemble_with_hybrid(
            test_proba, val_proba, y_val, hyb_test, hyb_val)
        print(f"   CodeBERT weight={w:.2f}, threshold={t:.4f}, Val F1={f1:.4f}")
        _show_dist(y_test, "   ")
        results['ensemble'] = (y_test, f1)
    else:
        print("   SKIP (no hybrid proba files)")

    # ================================================================
    # Fixed ratio baselines
    print("\n" + "=" * 70)
    print("FIXED TARGET RATIOS (no val needed)")
    print("=" * 70)
    for target in [0.40, 0.45, 0.50, 0.55, 0.60]:
        sorted_p = np.sort(test_proba)
        idx = int(len(sorted_p) * (1 - target))
        idx = max(0, min(idx, len(sorted_p) - 1))
        t = sorted_p[idx]
        y_pred = (test_proba >= t).astype(int)
        print(f"   Target {target*100:.0f}% AI → t={t:.4f}, "
              f"actual={y_pred.mean()*100:.1f}% AI")

    # ================================================================
    # Summary + save best
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Strategy':<20s} {'Val F1':>8s} {'%Human':>8s} {'%AI':>8s}")
    print(f"  {'-'*46}")
    best_name, best_f1 = None, 0
    for name, (y_test, f1) in sorted(results.items(), key=lambda x: -x[1][1]):
        pct_ai = y_test.mean() * 100
        print(f"  {name:<20s} {f1:>8.4f} {100-pct_ai:>7.1f}% {pct_ai:>7.1f}%")
        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    # Save all variants
    for name, (y_test, f1) in results.items():
        fname = f"submission_codebert_{name}.csv"
        pd.DataFrame({"ID": test_ids, "label": y_test}).to_csv(fname, index=False)
        print(f"\nSaved {fname}")

    # Save best
    if best_name:
        best_y = results[best_name][0]
        sub = pd.DataFrame({"ID": test_ids, "label": best_y})
        sub.to_csv(args.submission_out, index=False)
        print(f"\nBEST: {best_name} (Val F1={best_f1:.4f}) → {args.submission_out}")


def _show_dist(y, prefix=""):
    n = len(y)
    ai = y.sum()
    hum = n - ai
    print(f"{prefix}Test: Human={hum} ({hum/n*100:.1f}%), AI={ai} ({ai/n*100:.1f}%)")


if __name__ == "__main__":
    main()
