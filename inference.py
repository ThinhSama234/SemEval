import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

from feature_extractor import extract_all_features

def _prep_features(df, features_ready):
    if features_ready:
        X = df.drop(columns=['label', 'generator', 'language', 'code'], errors='ignore')
    else:
        X = extract_all_features(df['code'])
    X = X.replace([float('inf'), float('-inf')], float('nan')).fillna(0)
    return X


def train_and_save_model(data_path, output_model_path="xgb_model.pkl", features_ready=False, val_path=None):
    """
    Train an XGBoost model.
    If features_ready is True, it expects a parquet file where features are already extracted.
    Otherwise, it extracts features on the fly.
    If val_path is provided, uses that file as validation; otherwise random splits train data.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    X = _prep_features(df, features_ready)
    y = df['label']

    print(f"Features shape: {X.shape}")
    print(f"Labels: {y.value_counts().to_dict()}")

    if val_path:
        print(f"Loading validation data from {val_path}...")
        val_df = pd.read_parquet(val_path)
        X_val = _prep_features(val_df, features_ready)
        # Align columns with training features
        X_val = X_val.reindex(columns=X.columns, fill_value=0)
        y_val = val_df['label']
        X_train, y_train = X, y
        print(f"Using external validation set: {X_val.shape}")
    else:
        print("Splitting data into train/val (random 90/10)...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    print("Initializing XGBoost model...")
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
    
    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    print("\nEvaluating on validation set...")
    y_pred = model.predict(X_val)
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_val, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Save the model
    joblib.dump(model, output_model_path)
    print(f"\nModel saved to {output_model_path}")
    
    return model

def batch_inference(model_path, data_path, output_csv="submission_xgb.csv"):
    """
    Run inference on a file containing source code. Extracts features using the pipeline, runs predictions, and saves to CSV.
    """
    print(f"\nLoading model from {model_path}...")
    model = joblib.load(model_path)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    print("Extracting features...")
    X = extract_all_features(df['code'])
    X = X.replace([float('inf'), float('-inf')], float('nan')).fillna(0)
    
    print("Running predictions...")
    y_pred = model.predict(X)
    
    if "ID" in df.columns:
        output_df = pd.DataFrame({"ID": df["ID"], "prediction": y_pred})
    else:
        # Fallback if no ID column
        output_df = pd.DataFrame({"ID": df.index, "prediction": y_pred})
        
    output_df.to_csv(output_csv, index=False)
    print(f"Inference complete. Predictions saved to {output_csv}")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and run inference for Code Features Model using XGBoost")
    parser.add_argument("--mode", choices=["train", "infer"], required=True, help="Mode to run: train or infer")
    parser.add_argument("--data", required=True, help="Path to input parquet file")
    parser.add_argument("--model", default="taskA_xgb_model.pkl", help="Path to save/load the model")
    parser.add_argument("--output", default="submission.csv", help="Path to save predictions (inference only)")
    parser.add_argument("--features_ready", action="store_true", help="If training data already has features extracted")
    parser.add_argument("--val", default=None, help="Optional path to validation parquet (train mode)")

    args = parser.parse_args()

    if args.mode == "train":
        train_and_save_model(args.data, args.model, args.features_ready, val_path=args.val)
    elif args.mode == "infer":
        batch_inference(args.model, args.data, args.output)
