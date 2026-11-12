import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from src.feature_engineering import (
    add_some_features,
    drop_unused_columns,
    encode_categorical_features,
)
from src.preprocess import clean_and_preprocess


def run_randomizedsearch():
  df = pd.read_csv('data/train_test.csv')

  df['date'] = pd.to_datetime(df['date'])
  df = df.sort_values('date').reset_index(drop=True)

  cleaned_df = clean_and_preprocess(df)
  featured_df = add_some_features(cleaned_df)

  # Separate features and target *before* encoding to safely extract y_train
  X_raw = featured_df.drop(columns=['posted_rate'])
  y_train = featured_df['posted_rate']

  # Encode categorical features (passing X_raw for both train and self-eval here)
  X_encoded, _ = encode_categorical_features(X_raw, X_raw, y_train)

  # Drop unused cols (note: 'date' is already dropped here or handled)
  X_final, _ = drop_unused_columns(X_encoded, X_encoded)

  # Final feature matrix (dropping 'date' if it still exists)
  X = X_final.drop(columns=['date'], errors='ignore')
  y = y_train

  configs = {
      'XGBoost': {
          'model': xgb.XGBRegressor(
              objective='reg:squarederror', random_state=42
          ),
          'params': {
              'subsample': [0.6, 0.8, 1.0],
              'colsample_bytree': [0.6, 0.8, 1.0],
              'max_depth': [5, 7, 10],
              'learning_rate': [0.001, 0.01, 0.05, 0.1],
              'n_estimators': [500, 700, 1000, 2000],
              'reg_alpha': [0.5, 0.8, 1.0],
              'reg_lambda': [0.5, 0.8, 1.0],
          },
      },
      'RandomForest': {
          'model': RandomForestRegressor(random_state=42),
          'params': {
              'n_estimators': [200, 300, 500, 1000],
              'max_depth': [10, 15, 20, None],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4, 8],
              'max_features': ['sqrt', 0.5, 0.7],
              'max_samples': [0.6, 0.8, 1.0],
          },
      },
  }

  best_params_dict = {}

  for model_name, config in configs.items():
    print(f'Running RandomizedSearchCV for {model_name}...')
    grid = RandomizedSearchCV(
        estimator=config['model'],
        param_distributions=config['params'],
        n_iter=15,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42,
    )
    grid.fit(X, y)

    best_params_dict[model_name] = grid.best_params_

  # FIX: Use binary write mode 'wb' and remove invalid 'indent' argument
  with open('config/best_params.pkl', 'wb') as f:
    joblib.dump(best_params_dict, f)

  print('Best parameters saved successfully to config/best_params.pkl')


if __name__ == '__main__':
  run_randomizedsearch()