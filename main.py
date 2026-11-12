from __future__ import annotations

import os
import gc
import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

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


def main_pipeline():

    df = pd.read_csv("data/train_test.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    cleaned_df = clean_and_preprocess(df)
    featured_df = add_some_features(cleaned_df)

    split_date = pd.Timestamp("2025-08-31")
    train_data = featured_df[featured_df["date"] <= split_date].copy()
    test_data = featured_df[featured_df["date"] > split_date].copy()

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

    X_train = train_data.drop(columns=[c for c in drop_cols if c in train_data.columns], errors="ignore").copy()
    y_train = train_data["posted_rate"].copy()

    X_test = test_data.drop(columns=[c for c in drop_cols if c in test_data.columns], errors="ignore").copy()
    y_test = test_data["posted_rate"].copy()

    num_cols = X_train.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        lower_limit, upper_limit = bounds_calc(X_train[col])
        X_train[col] = clamp_outliers(X_train[col], lower_limit, upper_limit)
        X_test[col] = clamp_outliers(X_test[col], lower_limit, upper_limit)

        sk = skewness_calc(X_train[col])
        if abs(sk) >= 1 and X_train[col].min() >= 0:
            X_train[col] = np.log1p(X_train[col])
            X_test[col] = np.log1p(X_test[col])

    lower_y, upper_y = bounds_calc(y_train)
    y_train = clamp_outliers(y_train, lower_y, upper_y)
    y_test = clamp_outliers(y_test, lower_y, upper_y)

    sk_y = skewness_calc(y_train)
    target_log_transformed = False
    if abs(sk_y) >= 1 and y_train.min() >= 0:
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)
        target_log_transformed = True

    X_train_encoded, X_test_encoded = encode_categorical_features(X_train, X_test, y_train)

    del train_data, test_data, cleaned_df, featured_df
    gc.collect()



    best_params_path = "config/best_params.pkl"
    if os.path.exists(best_params_path):
        with open(best_params_path, "rb") as f:
            best_params_dict = joblib.load(f)
        xgb_params = best_params_dict.get("XGBoost", {})
        
    else:
        xgb_params = {"n_estimators":700,         
            "learning_rate":0.01,        
            "max_depth":5,              
            "subsample":0.8,             
            "colsample_bytree":0.8,      
            "reg_alpha":0.5,             
            "reg_lambda":0.8,            
            "tree_method":'hist',        
            "objective":'reg:squarederror', "random_state": 42}

    model = xgb.XGBRegressor(**xgb_params, n_jobs=-1)
    model.fit(X_train_encoded, y_train)

    preds_raw = model.predict(X_test_encoded)
    
    # Reverse log-transform on predictions if target was log-transformed
    preds = np.expm1(preds_raw) if target_log_transformed else preds_raw
    y_test_actual = np.expm1(y_test) if target_log_transformed else y_test

    rmse = root_mean_squared_error(y_test_actual, preds)
    mae = mean_absolute_error(y_test_actual, preds)
    r2 = r2_score(y_test_actual, preds)

    return rmse, mae, r2

if __name__ == "__main__":
    score_rmse, score_mae, score_r2 = main_pipeline()
    print(f"Pipeline executed successfully! Validation RMSE: {score_rmse:.4f} | R2: {score_r2:.4f}")