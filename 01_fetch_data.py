import yaml
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import functions from our src modules
from src.data_fetch.tiingo import fetch_spy, fetch_stocks
from src.data_fetch.yfinance import fetch_vix

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
data_path = config['data_path']

RAW_DIR = data_path['raw']
if not os.path.exists(RAW_DIR):
    os.makedirs(RAW_DIR)

end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365 * lookback)).strftime('%Y-%m-%d')

print(f"--- Starting data fetch process ({start_date} to {end_date}) ---")

# ==========================================
# 1. Execute Pipeline based on Config
# ==========================================

if sources.get('use_tiingo_spy'):
    fetch_spy(api_key, start_date, end_date, RAW_DIR)

if sources.get('use_yfinance_vix'):
    fetch_vix(start_date, end_date, RAW_DIR)

if sources.get('use_tiingo_stocks'):
    fetch_stocks(tickers, api_key, start_date, end_date, RAW_DIR)

if sources.get('use_bigquery_news'):
    print("News fetching is currently disabled in config or not yet implemented.")

print("--- Data fetching process completed ---")