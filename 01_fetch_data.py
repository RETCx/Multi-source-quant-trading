import yaml
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import functions
from src.data_fetch.tiingo import fetch_spy, fetch_stocks
from src.data_fetch.yfinance import fetch_vix, fetch_fundamentals
from src.data_fetch.bigquery import fetch_news  

# ==========================================
# 0. Setup & Load Config
# ==========================================
load_dotenv()
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

api_key = os.getenv("tiingo_api_key")
tickers = config['data_settings']['selected_tickers']
lookback = config['data_settings']['lookback_years']
sources = config['sources']
paths = config['data_path']

# Create directories
DIR_OHLCV = paths['raw']['OHLCV']
DIR_FUND = paths['raw']['FUNDAMENTALS']
DIR_NEWS = paths['raw']['NEWS']
news_keywords = config['data_settings']['news_keywords']
require_confirmation = sources.get('require_bq_confirmation', True)

# Create directories
os.makedirs(DIR_OHLCV, exist_ok=True)
os.makedirs(DIR_FUND, exist_ok=True)
os.makedirs(DIR_NEWS, exist_ok=True)
os.makedirs(paths['features'], exist_ok=True)
os.makedirs(paths['models'], exist_ok=True)

end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365 * lookback)).strftime('%Y-%m-%d')

print(f"--- Starting data fetch process ({start_date} to {end_date}) ---")

# ==========================================
# 1. Execute Pipeline based on Config
# ==========================================

# Fetch market data (SPY, VIX) to OHLCV folder
if sources.get('use_tiingo_spy'):
    fetch_spy(api_key, start_date, end_date, DIR_OHLCV)

if sources.get('use_yfinance_vix'):
    fetch_vix(start_date, end_date, DIR_OHLCV)

# Fetch stock data to OHLCV and FUNDAMENTALS folders
if sources.get('use_tiingo_stocks'):
    fetch_stocks(tickers, api_key, start_date, end_date, DIR_OHLCV)
    # Add fetching fundamentals here
    fetch_fundamentals(tickers, DIR_FUND)

# Fetch news from BigQuery (GDELT)
if sources.get('use_bigquery_news'):
    fetch_news(news_keywords, start_date, end_date, DIR_NEWS, require_confirmation)

print("--- Data fetching process completed ---")