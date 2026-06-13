import numpy as np
import pandas as pd

def calculate_rsi(series, period=14):
    """Custom RSI calculation specific to the VIX logic."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def build_market_features(spy_df, vix_df, window=20):
    """
    Takes raw SPY and VIX dataframes, merges them, and calculates 
    advanced market context features (VIX Z-Score, IV-HV Spread, etc.).
    """
    # 1. Standardize column names
    spy = spy_df[['adjClose']].rename(columns={'adjClose': 'SPY_Close'})
    vix = vix_df[['open', 'close', 'high', 'low']].rename(
        columns={'open': 'VIX_Open', 'close': 'VIX_Close', 'high': 'VIX_High', 'low': 'VIX_Low'}
    )

    # 2. Merge SPY & VIX on index (date)
    market_df = spy.join(vix, how='inner')

    # 3. Base calculations for correlation and spread
    market_df['SPY_Log_Ret'] = np.log(market_df['SPY_Close'] / market_df['SPY_Close'].shift(1))
    market_df['VIX_Change'] = market_df['VIX_Close'].diff()

    # Feature 1: VIX Z-Score (The Level)
    vix_mean = market_df['VIX_Close'].rolling(window=window).mean()
    vix_std = market_df['VIX_Close'].rolling(window=window).std()
    market_df['feat_VIX_ZScore'] = (market_df['VIX_Close'] - vix_mean) / (vix_std + 1e-6)

    # Feature 2: VIX Momentum (The Panic)
    market_df['feat_VIX_ROC5'] = market_df['VIX_Close'].pct_change(periods=5)

    # Feature 3: Market Structure (The Regime) - Correlation SPY vs VIX
    market_df['feat_Corr_SPY_VIX'] = market_df['SPY_Log_Ret'].rolling(window=window).corr(market_df['VIX_Change'])

    # Feature 4: IV-HV Spread (Fear Premium)
    spy_hist_vol = market_df['SPY_Log_Ret'].rolling(window=window).std() * np.sqrt(252) * 100
    market_df['feat_IV_HV_Spread'] = market_df['VIX_Close'] - spy_hist_vol

    # Feature 5: VIX Bollinger Band Width (Regime Squeeze)
    upper_band = vix_mean + (2 * vix_std)
    lower_band = vix_mean - (2 * vix_std)
    market_df['feat_VIX_BB_Width'] = (upper_band - lower_band) / vix_mean

    # Feature 6: Overnight Gap (Shock)
    if 'VIX_Open' in market_df.columns:
        market_df['feat_VIX_Gap'] = market_df['VIX_Open'] - market_df['VIX_Close'].shift(1)

    # Feature 7: VIX RSI
    market_df['feat_VIX_RSI'] = calculate_rsi(market_df['VIX_Close'])

    # Feature 8 & 9: VIX Intraday Body & Range (Smart Money Sentiment)
    if 'VIX_Open' in market_df.columns:
        market_df['feat_VIX_Intraday'] = market_df['VIX_Close'] - market_df['VIX_Open']
        market_df['feat_VIX_Range'] = market_df['VIX_High'] - market_df['VIX_Low']

    # Clean up temporary calculation columns
    market_df.drop(columns=['SPY_Log_Ret', 'VIX_Change'], inplace=True, errors='ignore')

    return market_df