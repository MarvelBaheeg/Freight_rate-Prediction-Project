import numpy as np
import pandas as pd
from scipy.stats import skew

def clean_and_preprocess(df: pd.DataFrame, train_weight_median: float = None) -> pd.DataFrame:
    df_clean = df.copy()
   
    if 'weight' in df_clean.columns:
        df_clean.loc[df_clean['weight'] < 0, 'weight'] = df_clean.loc[
            df_clean['weight'] < 0, 'weight'
        ].abs()
        
    if train_weight_median is not None:
       df_clean['weight'] = df_clean['weight'].fillna(train_weight_median)
    else:
        df_clean['weight'] = df_clean['weight'].fillna(df_clean['weight'].median())
      
    return df_clean
