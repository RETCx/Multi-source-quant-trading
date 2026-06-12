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

def fetch_fundamentals(tickers, raw_fund_dir):
    print("Fetching Historical Fundamental Data from Yahoo Finance...")
    for ticker in tickers:
        filepath = f"{raw_fund_dir}/{ticker}_fundamentals.csv"
        
        if not needs_download(filepath):
            print(f"Fundamentals for {ticker} is up to date. Skipping.")
            continue
            
        try:
            yticker = yf.Ticker(ticker)
            
            # fetch quarterly income statement and balance sheet
            inc_stmt = yticker.quarterly_income_stmt
            bs = yticker.quarterly_balance_sheet
            
            if inc_stmt.empty or bs.empty:
                print(f"Warning: No quarterly data found for {ticker}")
                continue
            
            # Transpose  Index
            inc_stmt = inc_stmt.T
            bs = bs.T
            
            # collect raw data
            df_fund = pd.DataFrame(index=inc_stmt.index)
            
            # 1. fetch EPS
            if 'Basic EPS' in inc_stmt.columns:
                df_fund['EPS_Raw'] = inc_stmt['Basic EPS']
            elif 'Diluted EPS' in inc_stmt.columns:
                df_fund['EPS_Raw'] = inc_stmt['Diluted EPS']

            # 2. fetch Net Income
            if 'Net Income' in inc_stmt.columns:
                df_fund['Net_Income'] = inc_stmt['Net Income']

            # 3. fetch Total Equity
            equity_cols = ['Stockholders Equity', 'Total Stockholder Equity', 'Total Equity Gross Minority Interest', 'Common Stock Equity', 'Net Tangible Assets']
            for col in equity_cols:
                if col in bs.columns:
                    df_fund['Total_Equity'] = bs[col]
                    break

            # 4. fetch Shares Outstanding
            shares_cols = ['Basic Average Shares', 'Diluted Average Shares', 'Ordinary Shares Number', 'Share Issued']
            for col in shares_cols:
                if col in inc_stmt.columns:
                    df_fund['Shares_Outstanding'] = inc_stmt[col]
                    break
            if 'Shares_Outstanding' not in df_fund.columns:
                 for col in shares_cols:
                    if col in bs.columns:
                        df_fund['Shares_Outstanding'] = bs[col]
                        break

            df_fund.index.name = 'date'
            if df_fund.index.tz is not None:
                df_fund.index = df_fund.index.tz_localize(None)
                
            df_fund = df_fund.dropna(how='all')
            df_fund.sort_index(ascending=True, inplace=True)
            
            df_fund.to_csv(filepath)
            print(f"Saved: {ticker}_fundamentals.csv")
            
        except Exception as e:
            print(f"Error fetching historical fundamentals for {ticker}: {e}")