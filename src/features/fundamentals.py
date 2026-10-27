import pandas as pd
import numpy as np

def merge_and_calc_fundamentals(price_df, fund_df, reporting_lag_days=45):
    """
    Merges daily price data with quarterly fundamental data using Point-in-Time 
    effective availability dates (or reporting lag) to eliminate look-ahead bias,
    and calculates daily P/E and P/BV.
    
    Args:
        price_df (pd.DataFrame): Daily stock price OHLCV data with DatetimeIndex
        fund_df (pd.DataFrame): Quarterly fundamentals with DatetimeIndex
        reporting_lag_days (int): Calendar days lag added to fiscal period end 
                                  to simulate publication delay (default: 45 days).
    """
    fund_df = fund_df.copy()
    fund_df.index = pd.to_datetime(fund_df.index)
    fund_df = fund_df.sort_index(ascending=True)

    # 0. Convert Quarterly Flow Metrics to TTM (Trailing Twelve Months)
    # TTM must be computed on consecutive fiscal quarters BEFORE shifting dates!
    if 'EPS_Raw' in fund_df.columns:
        fund_df['EPS_TTM'] = fund_df['EPS_Raw'].rolling(window=4, min_periods=1).sum()
    if 'Net_Income' in fund_df.columns:
        fund_df['Net_Income_TTM'] = fund_df['Net_Income'].rolling(window=4, min_periods=1).sum()

    # 1. Determine Effective Publication / Announcement Date (Point-in-Time)
    # If the file has a specific 'reportedDate' or 'filingDate', use that.
    # Otherwise, shift fiscalDateEnding by reporting_lag_days to prevent look-ahead bias.
    if 'reportedDate' in fund_df.columns and fund_df['reportedDate'].notna().any():
        effective_dates = pd.to_datetime(fund_df['reportedDate']).fillna(
            fund_df.index + pd.Timedelta(days=reporting_lag_days)
        )
    elif 'filingDate' in fund_df.columns and fund_df['filingDate'].notna().any():
        effective_dates = pd.to_datetime(fund_df['filingDate']).fillna(
            fund_df.index + pd.Timedelta(days=reporting_lag_days)
        )
    else:
        effective_dates = fund_df.index + pd.Timedelta(days=reporting_lag_days)

    fund_df.index = effective_dates
    fund_df = fund_df.sort_index(ascending=True)
    # Drop duplicate effective dates if any, keeping latest
    fund_df = fund_df[~fund_df.index.duplicated(keep='last')]

    # 2. Join price data with fundamental data using merge_asof backward
    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()

    df_merged = pd.merge_asof(
        price_df, fund_df,
        left_index=True, right_index=True,
        direction='backward'
    )

    # 3. Calculate EPS (using TTM EPS or calculate from Net Income TTM / Shares)
    if 'EPS_TTM' in df_merged.columns:
        df_merged['EPS'] = df_merged['EPS_TTM']
    elif 'Net_Income_TTM' in df_merged.columns and 'Shares_Outstanding' in df_merged.columns:
        df_merged['EPS'] = df_merged['Net_Income_TTM'] / df_merged['Shares_Outstanding']

    # 4. Calculate Book Value Per Share
    if 'Total_Equity' in df_merged.columns and 'Shares_Outstanding' in df_merged.columns:
        df_merged['Book_Value_Per_Share'] = df_merged['Total_Equity'] / df_merged['Shares_Outstanding']

    # 5. Calculate Valuation daily (Daily P/E, P/BV)
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
