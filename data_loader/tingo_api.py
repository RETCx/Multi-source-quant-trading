import yaml
import os
from dotenv import load_dotenv
import requests
import pandas as pd
from datetime import datetime, timedelta

# 1. load Config and API Key
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
load_dotenv()

TIINGO_API_KEY = os.getenv("tiingo_api_key")
tickers = config['data_settings']['selected_tickers']
lookback_years = config['data_settings']['lookback_years']

# 2. Calculate the date
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365 * lookback_years)).strftime('%Y-%m-%d')

print(f"Getting data from {start_date} to {end_date}...")

# 3. Create a folder to store data
if not os.path.exists("data"):
    os.makedirs("data")
else :
    print("Folder 'data' already exists")


# 4. Loop to get data one by one
for ticker in tickers:
    filepath = f"data/{ticker}.csv"
    should_download = True
    
    # Check if file exists and when it was last modified
    if os.path.exists(filepath):
        mtime = os.path.getmtime(filepath)
        last_mod_date = datetime.fromtimestamp(mtime).date()
        today_date = datetime.today().date()
        
        # If the file was modified today, we skip the download
        if last_mod_date == today_date:
            print(f"Data for {ticker} is up to date (modified today). Skipping API call.")
            should_download = False
        else:
            print(f"Data for {ticker} is outdated (modified {last_mod_date}). Fetching new data...")
            
    if should_download:
        print(f"Getting data for: {ticker}")
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token={TIINGO_API_KEY}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Convert to Pandas DataFrame for easier processing in the future
                df = pd.DataFrame(response.json())
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                df.set_index('date', inplace=True)
                
                # Save as CSV file in the data folder
                df.to_csv(filepath)
                print(f"Save {ticker}.csv successfully")
            else:
                print(f"Failed to retrieve {ticker} data (Code: {response.status_code})")
        except Exception as e:
            print(f"Error: {e}")

print("Get data successfully!")