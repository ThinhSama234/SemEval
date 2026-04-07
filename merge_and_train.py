"""
Merge EDA features + Perplexity → Train XGBoost → Save model & submission.

Usage:
  python merge_and_train.py
  python merge_and_train.py --test_data path/to/test.parquet --test_ppl path/to/test_ppl.parquet
"""
import argparse
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_and_merge(eda_path, ppl_path):
    """Merge EDA features with perplexity feature."""
    eda = pd.read_parquet(eda_path)
    ppl = pd.read_parquet(ppl_path)

    # Only take perplexity columns (not code/label/etc which are duplicates)
    ppl_cols = [c for c in ppl.columns if c not in eda.columns and c not in ['code', 'generator', 'language']]
    if 'overall_ppl' not in eda.columns:
        eda['overall_ppl'] = ppl['overall_ppl'].values

    for col in ppl_cols:
        eda[col] = ppl[col].values

    return eda


def prep_X(df):
    X = df.drop(columns=['label', 'generator', 'language', 'code'], errors='ignore')
    X = X.replace([float('inf'), float('-inf')], float('nan')).fillna(0)
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_eda", default="task_A/train_features_ml_ready.parquet")
    parser.add_argument("--train_ppl", default="train_perplexity.parquet")
    parser.add_argument("--val_eda", default="task_A/val_features_ml_ready.parquet")
    parser.add_argument("--val_ppl", default=None, help="Val perplexity parquet (optional)")
    parser.add_argument("--test_data", default=None, help="Test parquet for submission")
    parser.add_argument("--test_ppl", default=None, help="Test perplexity parquet")
    parser.add_argument("--model_out", default="taskA_xgb_v3.pkl")
    parser.add_argument("--submission_out", default="submission.csv")
    args = parser.parse_args()

    # --- Load & merge train ---
    print("Loading train EDA + perplexity...")
    train = load_and_merge(args.train_eda, args.train_ppl)
    X_train = prep_X(train)
    y_train = train['label']
    print(f"Train: {X_train.shape} features, {len(y_train)} samples")
    print(f"Features: {X_train.columns.tolist()}")

    # --- Load & merge val ---
    print("\nLoading validation EDA...")
    val = pd.read_parquet(args.val_eda)
    if args.val_ppl:
        val_ppl = pd.read_parquet(args.val_ppl)
        val['overall_ppl'] = val_ppl['overall_ppl'].values
    else:
        # No val perplexity yet — fill with median from train
        print("WARNING: No val perplexity file. Filling overall_ppl with train median.")
        val['overall_ppl'] = train['overall_ppl'].median()

    X_val = prep_X(val)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    y_val = val['label']
    print(f"Val: {X_val.shape} features, {len(y_val)} samples")

    # --- Train XGBoost ---
    print("\nTraining XGBoost (EDA + Perplexity)...")
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='logloss',
        tree_method='hist',
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    # --- Evaluate ---
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS (EDA + Perplexity)")
    print("=" * 60)
    y_pred = model.predict(X_val)
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_val, y_pred, average='weighted'):.4f}")
    print(classification_report(y_val, y_pred))

    # --- Feature importance top 10 ---
    importance = model.get_booster().get_score(importance_type='gain')
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    print("Top 15 features by gain:")
    for feat, gain in top_features:
        print(f"  {feat}: {gain:.1f}")

    # --- Save model ---
    joblib.dump(model, args.model_out)
    print(f"\nModel saved to {args.model_out}")

    # --- Generate submission if test data provided ---
    if args.test_data:
        print(f"\nGenerating submission from {args.test_data}...")
        from feature_extractor import extract_all_features
        test_df = pd.read_parquet(args.test_data)

        # Extract EDA features for test
        X_test = extract_all_features(test_df['code'])
        X_test = X_test.replace([float('inf'), float('-inf')], float('nan')).fillna(0)

        # Add perplexity if available
        if args.test_ppl:
            test_ppl = pd.read_parquet(args.test_ppl)
            X_test['overall_ppl'] = test_ppl['overall_ppl'].values
        else:
            X_test['overall_ppl'] = train['overall_ppl'].median()

        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        y_test_pred = model.predict(X_test)

        sub = pd.DataFrame({
            "ID": test_df["ID"] if "ID" in test_df.columns else test_df.index,
            "label": y_test_pred
        })
        sub.to_csv(args.submission_out, index=False)
        print(f"Submission saved to {args.submission_out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
