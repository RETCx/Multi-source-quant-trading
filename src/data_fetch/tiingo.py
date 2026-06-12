import requests
import pandas as pd
from src.utils import needs_download

def fetch_spy(api_key, start_date, end_date, raw_dir):
    filepath = f"{raw_dir}/SPY_raw.csv"
    
    if not needs_download(filepath):
        print("Data for SPY is up to date. Skipping API call.")
        return

    print("Fetching SPY from Tiingo...")
    url = "https://api.tiingo.com/tiingo/daily/spy/prices"
    params = {'startDate': start_date, 'endDate': end_date, 'token': api_key}
    
    try:
        res = requests.get(url, params=params)
        if res.status_code == 200:
            df = pd.DataFrame(res.json())
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df.set_index('date', inplace=True)
            df.to_csv(filepath)
            print("Saved: SPY_raw.csv")
        else:
            print(f"Failed to retrieve SPY data (Code: {res.status_code})")
    except Exception as e:
        print(f"Error fetching SPY: {e}")

def fetch_stocks(tickers, api_key, start_date, end_date, raw_dir):
    print("Fetching Individual Stocks from Tiingo...")
    for ticker in tickers:
        filepath = f"{raw_dir}/{ticker}_raw.csv"
        
        if not needs_download(filepath):
            print(f"Data for {ticker} is up to date. Skipping API call.")
            continue
            
        print(f"Getting data for: {ticker}")
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        params = {'startDate': start_date, 'endDate': end_date, 'token': api_key}
        
        try:
            res = requests.get(url, params=params)
            if res.status_code == 200:
                df = pd.DataFrame(res.json())
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                df.set_index('date', inplace=True)
                df.to_csv(filepath)
                print(f"Saved: {ticker}_raw.csv successfully")
            else:
                print(f"Failed to retrieve {ticker} data (Code: {res.status_code})")
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")