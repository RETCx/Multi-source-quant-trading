import numpy as np
import pandas as pd

def get_label_percentile(df, h=5, window=252):
    """
    Rolling Percentile - 2-Class (Binary Classification)
     h   (Median)  1   1 (),  0
    """
    #  pct_change(h)  h   (shift -h)
    future_return = df['adjClose'].pct_change(periods=h).shift(-h)
    
    #  Median  ( shift h   Look-ahead Bias)
    #  min_periods=1  window ( 1  ~ 251 )
    rolling_median = future_return.shift(h).rolling(window=window, min_periods=1).median()
    
    label = pd.Series(np.nan, index=df.index)
    valid = future_return.notna() & rolling_median.notna()
    
    # Convert to 1.0 (Beat Market) or 0.0 (Underperform Market)
    label[valid] = (future_return[valid] > rolling_median[valid]).astype(float)
    
    return label

def create_multi_horizon_targets(df, horizons=[1, 3, 5, 7, 9], window=252):
    """
     Target  5  
    """
    for h in horizons:
        col_name = f'Target_{h}D'
        df[col_name] = get_label_percentile(df, h=h, window=window)
        
    return df