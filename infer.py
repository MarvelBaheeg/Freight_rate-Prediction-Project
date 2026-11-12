from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew

from src.preprocess import clean_and_preprocess
from src.feature_engineering import (
    add_some_features,
    encode_categorical_features,
)


def bounds_calc(series):
    Q1 = np.percentile(series, q=25, method='midpoint')
    Q3 = np.percentile(series, q=75, method='midpoint')
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR


def clamp_outliers(series, lower_bound, upper_bound):
    return series.clip(lower_bound, upper_bound)


def skewness_calc(series):
    return skew(series, bias=False)


def run_inference():
    print("=== Loading Pre-trained Model & Artifacts ===")
    
    # 1. Load the pre-trained model from disk (NO RETRAINING)
    model_path = "config/xgboost_best_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run main.py first!")
    
    model = joblib.load(model_path)
    print(f"Successfully loaded model from {model_path}")

    # 2. Load training data ONLY to extract consistent preprocessing stats (bounds, encodings)
    train_df = pd.read_csv("data/train-test.csv")
    train_df["date"] = pd.to_datetime(train_df["date"])
    train_df = train_df.sort_values("date").reset_index(drop=True)

    cleaned_train = clean_and_preprocess(train_df)
    featured_train = add_some_features(cleaned_train)

    drop_cols = [
        "posted_rate",
        "date",
        "load_id",
        "delivery_lon",
        "delivery_lat",
        "pickup_lon",
        "pickup_lat",
        "market_index",
    ]

    X_train = featured_train.drop(columns=[c for c in drop_cols if c in featured_train.columns], errors="ignore").copy()
    y_train = featured_train["posted_rate"].copy()

    # Calculate training statistics for clamping & transformations
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    train_bounds = {}

    for col in num_cols:
        lower_limit, upper_limit = bounds_calc(X_train[col])
        train_bounds[col] = (lower_limit, upper_limit)
        X_train[col] = clamp_outliers(X_train[col], lower_limit, upper_limit)

        if abs(skewness_calc(X_train[col])) >= 1 and X_train[col].min() >= 0:
            X_train[col] = np.log1p(X_train[col])

    lower_y, upper_y = bounds_calc(y_train)
    y_train = clamp_outliers(y_train, lower_y, upper_y)
    
    target_log_transformed = abs(skewness_calc(y_train)) >= 1 and y_train.min() >= 0
    if target_log_transformed:
        y_train = np.log1p(y_train)

    # Fit categorical encoders on training structure
    X_train_encoded, _ = encode_categorical_features(X_train, X_train, y_train)
    train_columns = X_train_encoded.columns

    # ==========================================
    # 3. PROCESS & PREDICT VALIDATION SET
    # ==========================================
    print("=== Processing & Predicting Validation Set ===")
    val_template = pd.read_csv("data/validation-predictions-template.csv")
    val_features = pd.read_csv("data/validation.csv")
    val_features["date"] = pd.to_datetime(val_features["date"], errors="coerce")
    val_df = val_template[["load_id"]].merge(val_features, on="load_id", how="left")

    cleaned_val = clean_and_preprocess(val_df)
    featured_val = add_some_features(cleaned_val)
    X_val = featured_val.drop(columns=[c for c in drop_cols if c in featured_val.columns], errors="ignore").copy()

    for col in num_cols:
        if col in X_val.columns:
            lower_limit, upper_limit = train_bounds[col]
            X_val[col] = clamp_outliers(X_val[col], lower_limit, upper_limit)
            if abs(skewness_calc(X_train[col])) >= 1 and X_val[col].min() >= 0:
                X_val[col] = np.log1p(X_val[col])

    _, X_val_encoded = encode_categorical_features(X_train, X_val, y_train)
    X_val_encoded = X_val_encoded.reindex(columns=train_columns, fill_value=0)

    val_preds_raw = model.predict(X_val_encoded)
    val_preds = np.expm1(val_preds_raw) if target_log_transformed else val_preds_raw

    val_output = val_template.copy()
    val_output["predicted_rate"] = np.round(val_preds, 2)
    val_output.to_csv("validation_predictions.csv", index=False)
    print("Successfully saved validation_predictions.csv")

    # ==========================================
    # 4. PROCESS & PREDICT DECEMBER SET
    # ==========================================
    print("=== Processing & Predicting December Chart Inputs ===")
    december_path = "data/december-chart-inputs.csv"
    december_df = pd.read_csv(december_path)
    december_df["date"] = pd.to_datetime(december_df["date"], errors="coerce")

    cleaned_dec = clean_and_preprocess(december_df)
    featured_dec = add_some_features(cleaned_dec)
    X_dec = featured_dec.drop(columns=[c for c in drop_cols if c in featured_dec.columns], errors="ignore").copy()

    for col in num_cols:
        if col in X_dec.columns:
            lower_limit, upper_limit = train_bounds[col]
            X_dec[col] = clamp_outliers(X_dec[col], lower_limit, upper_limit)
            if abs(skewness_calc(X_train[col])) >= 1 and X_dec[col].min() >= 0:
                X_dec[col] = np.log1p(X_dec[col])

    _, X_dec_encoded = encode_categorical_features(X_train, X_dec, y_train)
    X_dec_encoded = X_dec_encoded.reindex(columns=train_columns, fill_value=0)

    dec_preds_raw = model.predict(X_dec_encoded)
    dec_preds = np.expm1(dec_preds_raw) if target_log_transformed else dec_preds_raw

    december_df["predicted_rate"] = np.round(dec_preds, 2)
    december_df["date"] = december_df["date"].dt.strftime("%Y-%m-%d")
    december_df.to_csv(december_path, index=False)
    print(f"Successfully updated {december_path}")
    print("Inference pipeline complete!")


if __name__ == "__main__":
    run_inference()