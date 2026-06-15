import os
import requests
import pandas as pd
import time
from src.utils import needs_download

def fetch_fundamentals(tickers, api_key, raw_fund_dir):
    print("Fetching Historical Fundamental Data from Alpha Vantage...")
    
    if not api_key or api_key == 'your_key_here':
        print("Error: ALPHAVANTAGE_API_KEY is not set correctly in .env")
        return

    for ticker in tickers:
        filepath = f"{raw_fund_dir}/{ticker}_fundamentals.csv"
        
        if not needs_download(filepath):
            print(f"Fundamentals for {ticker} is up to date. Skipping.")
            continue
            
        try:
            print(f"  [{ticker}] Fetching Income Statement...")
            url_inc = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={api_key}"
            r_inc = requests.get(url_inc)
            data_inc = r_inc.json()
            
            # Alpha Vantage limits free tier to 25 requests per day
            if "Information" in data_inc and "rate limit" in data_inc["Information"].lower():
                print(f"  [ERROR] Alpha Vantage Rate Limit Exceeded! {data_inc['Information']}")
                continue
                
            print(f"  [{ticker}] Fetching Balance Sheet...")
            url_bs = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={api_key}"
            r_bs = requests.get(url_bs)
            data_bs = r_bs.json()

            # Check if data exists
            if "quarterlyReports" not in data_inc or "quarterlyReports" not in data_bs:
                print(f"  [WARN] No quarterly reports found for {ticker}")
                continue

            # Process Income Statement
            df_inc = pd.DataFrame(data_inc["quarterlyReports"])
            df_inc = df_inc[['fiscalDateEnding', 'netIncome']].copy()
            
            # Process Balance Sheet
            df_bs = pd.DataFrame(data_bs["quarterlyReports"])
            
            # Sometimes 'commonStockSharesOutstanding' is missing in some quarters or named differently.
            cols_bs = ['fiscalDateEnding', 'totalShareholderEquity']
            if 'commonStockSharesOutstanding' in df_bs.columns:
                cols_bs.append('commonStockSharesOutstanding')
            df_bs = df_bs[cols_bs].copy()

            # Merge
            df_fund = pd.merge(df_inc, df_bs, on='fiscalDateEnding', how='outer')
            
            # Rename columns to match pipeline expectations
            rename_map = {
                'fiscalDateEnding': 'date',
                'netIncome': 'Net_Income',
                'totalShareholderEquity': 'Total_Equity',
                'commonStockSharesOutstanding': 'Shares_Outstanding'
            }
            df_fund.rename(columns=rename_map, inplace=True)
            
            # Convert to numeric
            for col in ['Net_Income', 'Total_Equity', 'Shares_Outstanding']:
                if col in df_fund.columns:
                    df_fund[col] = pd.to_numeric(df_fund[col], errors='coerce')

            # Clean and sort
            df_fund['date'] = pd.to_datetime(df_fund['date'])
            df_fund.set_index('date', inplace=True)
            df_fund = df_fund.dropna(how='all')
            df_fund.sort_index(ascending=True, inplace=True)
            
            df_fund.to_csv(filepath)
            print(f"  Saved: {ticker}_fundamentals.csv ({len(df_fund)} quarters)")
            
        except Exception as e:
            print(f"Error fetching historical fundamentals for {ticker}: {e}")
            
        # Sleep to avoid hitting 5 requests/minute free tier limit
        time.sleep(12)