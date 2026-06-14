import yaml
import os
import pandas as pd

# Import our custom engineering modules
import src.features.technical as tech
import src.features.advanced as adv
import src.features.market as mkt
import src.features.fundamentals as fund
import src.features.sentiment as sent
import src.features.time as time
import src.features.target as tgt

# ==========================================
# 0. Setup & Load Config
# ==========================================
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

tickers = config['data_settings']['selected_tickers']
paths = config['data_path']

DIR_OHLCV = paths['raw']['OHLCV']
DIR_FUND = paths['raw']['FUNDAMENTALS']
DIR_NEWS = paths['raw']['NEWS']
DIR_FEAT = paths['features']

os.makedirs(DIR_FEAT, exist_ok=True)

print("--- Starting Feature Engineering Pipeline ---")

# ==========================================
# 1. Build Market Features (SPY & VIX)
# ==========================================
print("Building Market Context Features...")
spy_path = f"{DIR_OHLCV}/SPY_raw.csv"
vix_path = f"{DIR_OHLCV}/VIX_raw.csv"
market_df = None

if os.path.exists(spy_path) and os.path.exists(vix_path):
    spy_df = pd.read_csv(spy_path, index_col='date', parse_dates=True)
    vix_df = pd.read_csv(vix_path, index_col='date', parse_dates=True)
    market_df = mkt.build_market_features(spy_df, vix_df)
else:
    print("Warning: SPY or VIX data not found. Market features will be skipped.")

# Initialize Sentiment Analyzer
sentiment_analyzer = sent.StockSentimentAnalyzer()

# ==========================================
# 2. Pipeline Loop for Each Ticker
# ==========================================
for ticker in tickers:
    filepath_raw = f"{DIR_OHLCV}/{ticker}_raw.csv"
    filepath_out = f"{DIR_FEAT}/{ticker}_with_indicators.csv"
    
    if not os.path.exists(filepath_raw):
        print(f"Raw data file not found for {ticker}, skipping.")
        continue
        
    print(f"Processing indicators for: {ticker}")
    
    try:
        # Load Raw Data
        df = pd.read_csv(filepath_raw, index_col='date', parse_dates=True)
        
        # --- Apply Technical Indicators ---
        tech.calc_sma(df, period=5)
        tech.calc_sma(df, period=10)
        tech.calc_sma(df, period=20)
        tech.calc_sma(df, period=50)
        tech.calc_sma(df, period=200)

        tech.calc_ema(df, period=5)
        tech.calc_ema(df, period=10)
        tech.calc_ema(df, period=20)
        tech.calc_ema(df, period=50)
        tech.calc_ema(df, period=200)

        tech.calc_wma(df, period=5)
        tech.calc_wma(df, period=10)
        tech.calc_wma(df, period=20)

        tech.calc_hma(df, period=9)
        tech.calc_hma(df, period=16)

        tech.calc_tema(df, period=10)
        tech.calc_tema(df, period=20)

        tech.calc_dmi(df, period=14)
        tech.calc_rsi(df, period=14)
        tech.calc_cmo(df, period=14)
        tech.calc_cci(df, period=20)
        tech.calc_williams_r(df, period=14)
        
        tech.calc_roc(df, period=10)
        tech.calc_roc(df, period=12)
        
        tech.calc_macd(df, fast_period=12, slow_period=26, signal_period=9)
        tech.calc_cmfi(df, period=21)
        tech.calc_psi(df, period=12)

        # --- Apply Advanced Features ---
        adv.calc_kaufman_er(df, er_window=10)
        adv.calc_garman_klass_vol(df, gk_window=20)
        adv.calc_amihud_illiquidity(df, amihud_window=5)
        adv.calc_rolling_skewness(df, skew_window=60)
        adv.calc_historical_volatility(df, window=20)
        adv.calc_atr(df, period=14)

        # --- Merge Market Features ---
        if market_df is not None:
            df = df.join(market_df, how='left')

        # --- Apply Fundamental Features ---
        fund_path = f"{DIR_FUND}/{ticker}_fundamentals.csv"
        if os.path.exists(fund_path):
            fund_df = pd.read_csv(fund_path, index_col='date', parse_dates=True)
            df = fund.merge_and_calc_fundamentals(df, fund_df)
            df = fund.calc_fundamental_features(df)

        # --- Apply Sentiment Features ---
        news_path = f"{DIR_NEWS}/{ticker}_news_raw.csv"
        if os.path.exists(news_path):
            news_df = pd.read_csv(news_path)

            # To strictly prevent look-ahead bias, we must define when the 'train' period ends.
            # In a walk-forward setting, this should ideally be done per-fold dynamically.
            # But since features are built once, we approximate by using data up to the last year
            # for source quality calculation, leaving the last year for OOS.
            train_end_date = pd.to_datetime(config['backtest'].get('start_date', '2024-01-01'))

            # This function uses merge on 'date', which resets the index.
            df = sentiment_analyzer.process_sentiment_features(ticker, df, news_df, train_end_date=train_end_date)
            if 'date' in df.columns:
                # Convert 'date' back to datetime index 
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
        
        # --- Apply Time Features ---
        time.calc_time_features(df)

        # --- 6. Create Multi-Horizon Targets (from config) ---
        target_horizons = config.get('target_horizons', [1, 3, 5, 10, 20])
        target_window = config.get('target_rolling_window', 252)
        tgt.create_multi_horizon_targets(df, horizons=target_horizons, window=target_window)
        # --- Filter Date Range & Drop Initial Window NaNs ---
        # Moving average of 200 days needs 200 rows of seed data, so we filter after calculation
        # We also want to drop rows that have NaNs due to rolling windows.
        # But we might have NaNs from Market or Fundamentals if they don't cover the full range.
        feature_cols = [c for c in df.columns if not c.startswith('Target_')]
        df.dropna(subset=feature_cols, inplace=True)
        
        # Save output to features folder
        df.to_csv(filepath_out)
        print(f"   Success: Saved {ticker}_with_indicators.csv (Total rows: {len(df)})")
        
    except Exception as e:
        print(f"   Error processing {ticker}: {e}")

print("--- Feature Engineering Pipeline Completed ---")