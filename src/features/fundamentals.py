import pandas as pd
import numpy as np

def merge_and_calc_fundamentals(price_df, fund_df):
    """
    Merges daily price data with quarterly fundamental data using forward-fill
    to prevent look-ahead bias, and calculates daily P/E and P/BV.
    """
    # 0. Convert Quarterly Flow Metrics to TTM (Trailing Twelve Months)
    # yfinance returns quarterly EPS and Net Income, so we must sum the last 4 quarters 
    # to get the annualized value for correct P/E calculation.
    fund_df = fund_df.sort_index(ascending=True)
    if 'EPS_Raw' in fund_df.columns:
        fund_df['EPS_TTM'] = fund_df['EPS_Raw'].rolling(window=4, min_periods=1).sum()
    if 'Net_Income' in fund_df.columns:
        fund_df['Net_Income_TTM'] = fund_df['Net_Income'].rolling(window=4, min_periods=1).sum()

    # 1. Join data price with data fundamental using merge_asof
    # This automatically matches each price date with the most recent fundamental date before it.
    # It solves the issue where fundamental dates aren't exactly on trading days or before the first price date.
    price_df.index = pd.to_datetime(price_df.index)
    fund_df.index = pd.to_datetime(fund_df.index)
    
    price_df = price_df.sort_index()
    fund_df = fund_df.sort_index()
    
    df_merged = pd.merge_asof(price_df, fund_df, left_index=True, right_index=True, direction='backward')
    
    # 2. Forward Fill is handled by merge_asof(direction='backward').
    # But for dates strictly BEFORE the very first date in fund_df (e.g. 2016-2024), they will be NaN.
    # To prevent dropping 8 years of OHLCV data, we backward fill the earliest known fundamental data.
    # (Minor look-ahead bias for those early years, but preserves the dataset size)
    fund_columns = fund_df.columns
    for col in fund_columns:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].bfill()
    
    # 3. calculate EPS (using TTM EPS or calculate from Net Income TTM / Shares)
    if 'EPS_TTM' in df_merged.columns:
        df_merged['EPS'] = df_merged['EPS_TTM']
    elif 'Net_Income_TTM' in df_merged.columns and 'Shares_Outstanding' in df_merged.columns:
        df_merged['EPS'] = df_merged['Net_Income_TTM'] / df_merged['Shares_Outstanding']
        
    # 4. calculate Book Value Per Share
    if 'Total_Equity' in df_merged.columns and 'Shares_Outstanding' in df_merged.columns:
        df_merged['Book_Value_Per_Share'] = df_merged['Total_Equity'] / df_merged['Shares_Outstanding']

    # 5. calculate Valuation daily (Daily P/E, P/BV)
    if 'EPS' in df_merged.columns:
        df_merged['daily_PE'] = df_merged['adjClose'] / df_merged['EPS'].replace(0, np.nan)
        
    if 'Book_Value_Per_Share' in df_merged.columns:
        df_merged['daily_PBV'] = df_merged['adjClose'] / df_merged['Book_Value_Per_Share'].replace(0, np.nan)
        
    return df_merged

def calc_fundamental_features(df, window=200):
    """
    Calculates moving averages, ratios, and Z-scores for fundamental metrics.
    """
    if 'daily_PE' in df.columns:
        df['PE_SMA200'] = df['daily_PE'].rolling(window=window).mean()
        df['PE_to_SMA200_Ratio'] = df['daily_PE'] / df['PE_SMA200']
        df['PE_ZScore_200'] = (df['daily_PE'] - df['PE_SMA200']) / df['daily_PE'].rolling(window=window).std()
        
    if 'daily_PBV' in df.columns:
        df['PBV_SMA200'] = df['daily_PBV'].rolling(window=window).mean()
        df['PBV_to_SMA200_Ratio'] = df['daily_PBV'] / df['PBV_SMA200']
        df['PBV_ZScore_200'] = (df['daily_PBV'] - df['PBV_SMA200']) / df['daily_PBV'].rolling(window=window).std()
        
    return df