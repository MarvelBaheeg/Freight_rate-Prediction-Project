import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, TargetEncoder


def add_some_features(df: pd.DataFrame) -> pd.DataFrame:
  fe_df = df.copy()

  fe_df['month'] = fe_df['date'].dt.month
  fe_df['day'] = fe_df['date'].dt.day

  fe_df['weight_distance_ratio'] = fe_df['weight'] / fe_df['distance']
  fe_df['route'] = (
      fe_df['pickup'].astype(str) + '_' + fe_df['delivery'].astype(str)
  )

  return fe_df


def encode_categorical_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
):
  X_train = X_train.copy()
  X_test = X_test.copy()
  y_train = y_train.copy()

  freq_cols = ['pickup', 'delivery']
  for col in freq_cols:
    freq_map = X_train[col].value_counts(normalize=True)
    X_train[f'{col}_encoded'] = X_train[col].map(freq_map)
    X_test[f'{col}_encoded'] = X_test[col].map(freq_map).fillna(0)

  X_train = X_train.drop(columns=freq_cols, errors='ignore')
  X_test = X_test.drop(columns=freq_cols, errors='ignore')

  target_enc = TargetEncoder(cv=4, random_state=42)
  X_train['route_encoded'] = target_enc.fit_transform(
      X_train[['route']], y_train
  )
  X_test['route_encoded'] = target_enc.transform(X_test[['route']])
  X_train = X_train.drop(columns=['route'])
  X_test = X_test.drop(columns=['route'])

  encoder = OneHotEncoder(
      sparse_output=False, handle_unknown='ignore', dtype=np.float64
  )
  train_encoded = encoder.fit_transform(X_train[['equipment']])
  test_encoded = encoder.transform(X_test[['equipment']])

  train_encoded_df = pd.DataFrame(
      train_encoded,
      columns=encoder.get_feature_names_out(['equipment']),
      index=X_train.index,
  )
  test_encoded_df = pd.DataFrame(
      test_encoded,
      columns=encoder.get_feature_names_out(['equipment']),
      index=X_test.index,
  )

  X_train = pd.concat(
      [X_train.drop(columns=['equipment']), train_encoded_df], axis=1
  )
  X_test = pd.concat(
      [X_test.drop(columns=['equipment']), test_encoded_df], axis=1
  )

  return X_train, X_test
