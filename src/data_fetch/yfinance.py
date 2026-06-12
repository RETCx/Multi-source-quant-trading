import yfinance as yf
import pandas as pd
from src.utils import needs_download

def fetch_vix(start_date, end_date, raw_dir):
    filepath = f"{raw_dir}/VIX_raw.csv"
    
    if not needs_download(filepath):
        print("Data for VIX is up to date. Skipping API call.")
        return

    print("Fetching VIX from Yahoo Finance...")
    try:
        df = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None)
        df.to_csv(filepath)
        print("Saved: VIX_raw.csv")
    except Exception as e:
        print(f"Error fetching VIX: {e}")