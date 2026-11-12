import joblib
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):

    xg_model = xgb.XGBRegressor(n_estimators=700,         
    learning_rate=0.01,        
    max_depth=5,              
    subsample=0.8,             
    colsample_bytree=0.8,      
    reg_alpha=0.5,             
    reg_lambda=0.8,            
    random_state=42,
    n_jobs=-1,                 
    tree_method='hist',        
    objective='reg:squarederror')

    xg_model.fit(X_train, y_train)

    return xg_model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series):
    
    rf_model = RandomForestRegressor(
    n_estimators=300, 
    min_samples_split=10, 
    min_samples_leaf=4,
    max_samples=0.8,
    max_features=0.5,
    max_depth=15,
    n_jobs=-1, 
    random_state=42)

    rf_model.fit(X_train, y_train)

    return rf_model