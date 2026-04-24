"""
Ensemble Pipeline: voting + majority + stacking on top of v5 + v10 + CodeBERT.

Runs AFTER:
  (a) train_v10_lang_robust.py has produced taskA_v10.pkl and test_proba_v10.npy
  (b) taskA_hybrid_v5.pkl exists (already trained earlier)
  (c) test_proba.npy exists (CodeBERT test predictions, already generated)
  (d) Optional: val_proba_cb.npy exists (CodeBERT val predictions, for 3-way stacking)

Outputs (in /kaggle/working/ or current directory):
  =================== Individual calibrations ===================
    submission_v10_q50.csv      v10 thresholded to 50% AI
    submission_v10_q55.csv      v10 thresholded to 55% AI

  =================== Soft voting (rank-average) ================
    submission_vote_v10cb_q50.csv     (v10 + CodeBERT), 50% AI
    submission_vote_v10cb_q55.csv     (v10 + CodeBERT), 55% AI
    submission_vote_v5v10cb_q50.csv   (v5 + v10 + CodeBERT), 50% AI
    submission_vote_v5v10cb_q55.csv   (v5 + v10 + CodeBERT), 55% AI

  =================== Majority vote (label-level) ===============
    submission_maj_v5v10cb.csv        majority of 3 labels

  =================== Stacking (LR meta-learner) ================
    submission_stack_v10cb_q50.csv    (2-way: v10+cb if cb val missing, else stack)
    submission_stack_v5v10cb_q50.csv  (3-way stacking)

Priority order to submit to Kaggle:
  1. submission_vote_v5v10cb_q55.csv     safest balanced blend
  2. submission_vote_v10cb_q50.csv       drops v5 (weakest model)
  3. submission_stack_v5v10cb_q50.csv    learned meta-weights
  4. submission_maj_v5v10cb.csv          naive label majority
  5. submission_v10_q50.csv              v10 alone (control)
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


V5_FEAT_COLS = [
    "comment_ratio", "blank_line_ratio", "indentation_std", "line_len_std",
    "style_consistency", "ttr", "comment_completeness", "blank_per_function",
    "comment_per_function", "trailing_ws_ratio", "naming_uniformity",
    "line_len_burstiness", "token_entropy", "inline_comment_ratio", "keyword_density",
    "max_nesting_depth", "avg_block_length", "cyclomatic_proxy",
    "comment_word_count_avg", "function_size_regularity",
]


def v5_predict_proba(pipeline, X):
    qt, iso_ai, iso_hum, cnb, s_min = (
        pipeline['qt'], pipeline['iso_ai'], pipeline['iso_hum'],
        pipeline['cnb'], pipeline['s_min'],
    )
    X_qt = qt.transform(X)
    s_ai = iso_ai.decision_function(X_qt)
    s_hum = iso_hum.decision_function(X_qt)
    S = np.column_stack([s_ai, s_hum])
    S_shifted = S - s_min + 1e-6
    return cnb.predict_proba(S_shifted)[:, 1]


def rank_normalize(p):
    order = np.argsort(p)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(p))
    return rank / max(1, len(p) - 1)


def write_sub(ids, labels, path):
    sub = pd.DataFrame({"ID": ids, "label": labels.astype(int)})
    sub.to_csv(path, index=False)
    ai_pct = (labels == 1).mean() * 100
    print(f"  wrote {os.path.basename(path):45s}  AI%={ai_pct:5.2f}")


def quantile_submit(ids, proba, target_pct, out):
    t = np.quantile(proba, 1 - target_pct / 100.0)
    labels = (proba >= t).astype(int)
    write_sub(ids, labels, out)


def main():
    p = argparse.ArgumentParser()
    # Pre-existing CodeBERT probas
    p.add_argument('--cb_test', default='test_proba.npy',
                   help='CodeBERT test probas (already exists)')
    p.add_argument('--cb_val',  default=None,
                   help='CodeBERT val probas (for 3-way stacking). If missing, 2-way stacking only.')
    # v10 probas produced by train_v10_lang_robust.py
    p.add_argument('--v10_test', default='submission_v10_proba.npy',
                   help='v10 test probas saved by train_v10_lang_robust.py')
    p.add_argument('--v10_val',  default=None,
                   help='v10 val probas (for stacking). If missing, tries <v10_test base>_val_proba.npy '
                        'or else reconstructs from --v10_pkl + --val_data')
    p.add_argument('--v10_pkl',  default='taskA_v10.pkl')
    p.add_argument('--val_data', default=None,
                   help='Path to raw validation.parquet (only needed if reconstructing v10 val probas)')
    # v5
    p.add_argument('--v5_pkl',   default='taskA_hybrid_v5.pkl')
    p.add_argument('--test_style', default='test_style_features.parquet')
    p.add_argument('--val_style',  default='val_style_features.parquet')
    p.add_argument('--val_feat',   default='task_A/val_features_ml_ready.parquet',
                   help='For val labels')
    # Output dir
    p.add_argument('--out_dir', default='.', help='Where to write submission CSVs')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    outp = lambda name: os.path.join(args.out_dir, name)

    # =========================================================================
    # 1. Load all probas
    # =========================================================================
    print("Loading probas...")
    p_cb_test = np.load(args.cb_test)

    # v10 test probas
    if os.path.exists(args.v10_test):
        p_v10_test = np.load(args.v10_test)
        print(f"  Loaded v10 test probas from {args.v10_test}")
    else:
        sys.exit(f"ERROR: {args.v10_test} not found. Run train_v10_lang_robust.py first.")

    N = len(p_cb_test)
    assert len(p_v10_test) == N, f"length mismatch: cb={len(p_cb_test)} v10={len(p_v10_test)}"

    # =========================================================================
    # 2. Extract v5 test probas on the fly
    # =========================================================================
    print("\nLoading v5 hybrid pipeline + predicting on test style features...")
    v5 = joblib.load(args.v5_pkl)
    pipeline_v5 = v5['pipeline']
    test_style = pd.read_parquet(args.test_style)
    ids = test_style['ID'].values if 'ID' in test_style.columns else np.arange(N)
    X_test_v5 = test_style[V5_FEAT_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    p_v5_test = v5_predict_proba(pipeline_v5, X_test_v5)
    print(f"  v5 test proba: mean={p_v5_test.mean():.4f} median={np.median(p_v5_test):.4f}")

    # =========================================================================
    # 3. v10 alone (sanity / control submissions)
    # =========================================================================
    print("\n" + "=" * 60)
    print("v10 alone — threshold calibration")
    print("=" * 60)
    for target in [48, 50, 52, 55]:
        quantile_submit(ids, p_v10_test, target, outp(f'submission_v10_q{target}.csv'))

    # =========================================================================
    # 4. Soft voting (rank-average blends)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Soft voting (rank-average)")
    print("=" * 60)
    r_v5   = rank_normalize(p_v5_test)
    r_v10  = rank_normalize(p_v10_test)
    r_cb   = rank_normalize(p_cb_test)

    # Pairs
    blend_v10cb = (r_v10 + r_cb) / 2.0
    blend_all3  = (r_v5 + r_v10 + r_cb) / 3.0
    # Weighted: downweight v5 (weakest signal)
    blend_w = 0.2 * r_v5 + 0.4 * r_v10 + 0.4 * r_cb

    # Sweep extra thresholds around q50 for the best-performing blend (v5v10cb).
    # Others only need q50 & q55 since they already underperform.
    for name, p_arr in [('v10cb', blend_v10cb), ('w244', blend_w)]:
        for target in [50, 55]:
            quantile_submit(ids, p_arr, target, outp(f'submission_vote_{name}_q{target}.csv'))
    for target in [45, 48, 50, 52, 55, 58]:
        quantile_submit(ids, blend_all3, target, outp(f'submission_vote_v5v10cb_q{target}.csv'))

    # =========================================================================
    # 5. Majority vote (label-level)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Majority vote (label-level)")
    print("=" * 60)

    # Use each model's own "balanced" label
    labels_v5  = (p_v5_test  >= np.quantile(p_v5_test,  0.50)).astype(int)
    labels_v10 = (p_v10_test >= np.quantile(p_v10_test, 0.50)).astype(int)
    labels_cb  = (p_cb_test  >= np.quantile(p_cb_test,  0.50)).astype(int)

    maj3 = ((labels_v5 + labels_v10 + labels_cb) >= 2).astype(int)
    write_sub(ids, maj3, outp('submission_maj_v5v10cb.csv'))

    # 2-of-2 agreement v10+cb
    and_v10cb = ((labels_v10 + labels_cb) == 2).astype(int)
    write_sub(ids, and_v10cb, outp('submission_and_v10cb.csv'))

    # =========================================================================
    # 6. Stacking (LR meta-learner on val probas)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Stacking (LR meta-learner)")
    print("=" * 60)

    # We need val labels
    val = pd.read_parquet(args.val_feat, columns=['label'])
    y_val = val['label'].values

    # v5 val probas
    val_style = pd.read_parquet(args.val_style)
    X_val_v5 = val_style[V5_FEAT_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    p_v5_val = v5_predict_proba(pipeline_v5, X_val_v5)

    # v10 val probas — resolution order:
    #   1. explicit --v10_val arg
    #   2. sibling file <v10_test>_val_proba.npy   (saved by train_v10 by default)
    #   3. reconstruct from --v10_pkl + --val_data + --val_feat
    if args.v10_val and os.path.exists(args.v10_val):
        p_v10_val = np.load(args.v10_val)
        print(f"  Loaded v10 val probas from {args.v10_val}")
    else:
        # Check the default sibling location saved by train_v10
        default_sibling = args.v10_test.replace('_proba.npy', '_val_proba.npy')
        if os.path.exists(default_sibling):
            p_v10_val = np.load(default_sibling)
            print(f"  Loaded v10 val probas from {default_sibling} (default sibling)")
        else:
            print(f"  v10 val probas not found. Re-predicting via {args.v10_pkl}...")
            if not args.val_data or not os.path.exists(args.val_data):
                sys.exit(
                    f"ERROR: cannot reconstruct v10 val probas without --val_data "
                    f"(expected raw validation.parquet with 'language' column). "
                    f"Either: (a) pass --v10_val <path>, "
                    f"(b) pass --val_data /path/to/validation.parquet, or "
                    f"(c) rerun train_v10_lang_robust.py so val probas are saved."
                )
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from train_v10_lang_robust import (
                add_language_feature, add_interaction_features, prep_features,
            )
            v10 = joblib.load(args.v10_pkl)
            models_v10 = v10['models']
            weights_v10 = v10['weights']
            model_names_v10 = v10['model_names']
            train_cols = v10['train_columns']

            val_full = pd.read_parquet(args.val_feat)
            val_meta = pd.read_parquet(args.val_data, columns=['language'])
            val_full['language'] = val_meta['language'].values
            add_language_feature(val_full)
            add_interaction_features(val_full)
            X_val_v10 = prep_features(val_full)
            X_val_v10 = X_val_v10.reindex(columns=train_cols, fill_value=0)

            val_probs = [models_v10[name].predict_proba(X_val_v10)[:, 1] for name in model_names_v10]
            tot_w = sum(weights_v10)
            p_v10_val = sum(p * w for p, w in zip(val_probs, weights_v10)) / tot_w
            # cache for next time
            np.save(default_sibling, p_v10_val)
            print(f"  Saved reconstructed v10 val probas to {default_sibling}")

    # 2-way stacking (v5 + v10)
    X_val_stk = np.column_stack([p_v5_val, p_v10_val])
    X_test_stk = np.column_stack([p_v5_test, p_v10_test])
    feat_names = ['v5', 'v10']

    if args.cb_val and os.path.exists(args.cb_val):
        p_cb_val = np.load(args.cb_val)
        X_val_stk  = np.column_stack([X_val_stk, p_cb_val])
        X_test_stk = np.column_stack([X_test_stk, p_cb_test])
        feat_names.append('cb')
        print(f"  3-way stacking: {feat_names}")
    else:
        print(f"  2-way stacking only (v5 + v10). Supply --cb_val for 3-way.")

    meta = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    meta.fit(X_val_stk, y_val)
    print(f"  meta coefs: {dict(zip(feat_names, meta.coef_[0]))}")
    print(f"  meta intercept: {meta.intercept_[0]:.4f}")

    val_meta_proba = meta.predict_proba(X_val_stk)[:, 1]
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.005):
        f1 = f1_score(y_val, (val_meta_proba >= t).astype(int), average='macro')
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"  meta val F1: {best_f1:.4f} @ t={best_t:.3f}")

    test_meta_proba = meta.predict_proba(X_test_stk)[:, 1]
    stack_tag = ''.join(feat_names)
    for target in [50, 55]:
        quantile_submit(ids, test_meta_proba, target,
                        outp(f'submission_stack_{stack_tag}_q{target}.csv'))

    np.save(outp(f'test_proba_stack_{stack_tag}.npy'), test_meta_proba)

    # =========================================================================
    # 7. Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("DONE. Upload order to Kaggle (most likely to beat 0.535):")
    print("=" * 60)
    print("""
      1. submission_vote_v5v10cb_q55.csv     3-way rank-avg, 55% AI  [safest]
      2. submission_vote_v10cb_q50.csv       v10+CodeBERT (drop v5)
      3. submission_stack_""" + stack_tag + """_q50.csv    meta-learner stacking
      4. submission_maj_v5v10cb.csv          naive majority vote
      5. submission_v10_q50.csv              v10 alone (control)
    """)


if __name__ == '__main__':
    main()
