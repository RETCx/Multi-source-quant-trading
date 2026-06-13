import numpy as np
import pandas as pd

def calc_kaufman_er(df, er_window=10, column='adjClose'):
    change = (df[column] - df[column].shift(er_window)).abs()
    volatility = (df[column] - df[column].shift(1)).abs().rolling(window=er_window).sum()
    df['feature_er'] = change / volatility

def calc_garman_klass_vol(df, gk_window=20):
    log_hl = (np.log(df['adjHigh'] / df['adjLow']))**2
    log_co = (np.log(df['adjClose'] / df['adjOpen']))**2
    const = 2 * np.log(2) - 1
    gk_var = (0.5 * log_hl - const * log_co).rolling(window=gk_window).mean()
    df['feature_gk_vol'] = np.sqrt(gk_var)

def calc_amihud_illiquidity(df, amihud_window=5):
    returns = df['adjClose'].pct_change().abs()
    dollar_volume = df['adjClose'] * df['adjVolume']
    amihud_raw = returns / dollar_volume.replace(0, np.nan)
    df['feature_amihud'] = amihud_raw.rolling(window=amihud_window).mean()

def calc_rolling_skewness(df, skew_window=60):
    daily_ret = df['adjClose'].pct_change()
    df['feature_skewness'] = daily_ret.rolling(window=skew_window).skew()

def calc_historical_volatility(df, window=20):
    df['feature_volatility'] = df['adjClose'].pct_change().rolling(window=window).std()

def calc_atr(df, period=14):
    high_low = df['adjHigh'] - df['adjLow']
    high_close = np.abs(df['adjHigh'] - df['adjClose'].shift())
    low_close = np.abs(df['adjLow'] - df['adjClose'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f'ATR{period}'] = tr.rolling(window=period).mean()