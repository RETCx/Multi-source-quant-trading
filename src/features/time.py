import numpy as np
import pandas as pd
def calc_time_features(df):    
# ====================================================================
# TIME-BASED FEATURES with Cyclic Encoding
# ====================================================================
    df["day_of_week_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 5)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["quarter_sin"] = np.sin(2 * np.pi * df.index.quarter / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df.index.quarter / 4)
    return df