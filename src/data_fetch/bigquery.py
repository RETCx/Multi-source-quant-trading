import os
import pandas as pd
from google.cloud import bigquery
from src.utils import needs_download

def check_if_needs_update(keywords_dict, raw_dir):
    """Check if all ticker files are already updated today."""
    for ticker in keywords_dict.keys():
        filepath = f"{raw_dir}/{ticker}_news_raw.csv"
        if needs_download(filepath):
            return True # At least one needs update
    return False

def get_incremental_start_date(keywords_dict, default_start_date, raw_dir):
    """Find the earliest 'last updated date' among all tickers to resume downloading."""
    latest_dates = []
    
    for ticker in keywords_dict.keys():
        filepath = f"{raw_dir}/{ticker}_news_raw.csv"
        if os.path.exists(filepath):
            try:
                df_existing = pd.read_csv(filepath)
                if not df_existing.empty:
                    last_date = df_existing['seendate'].max()
                    last_date_str = pd.to_datetime(last_date).strftime('%Y-%m-%d')
                    latest_dates.append(last_date_str)
            except Exception:
                pass
                
    if latest_dates:
        actual_start_date = min(latest_dates)
        print(f"Incremental Update Triggered: Adjusting start_date from {default_start_date} to {actual_start_date}")
        return actual_start_date
        
    return default_start_date

def build_gdelt_query(keywords_dict, start_date, end_date):
    """Build the BigQuery SQL string for GDELT."""
    combined_keywords = "|".join([v for v in keywords_dict.values()])
    return f"""
    SELECT
      PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS seendate,
      DocumentIdentifier AS url,
      V2Organizations AS organizations,
      V2Tone AS tone_data 
    FROM
      `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE
      _PARTITIONTIME BETWEEN TIMESTAMP('{start_date} 00:00:00') AND TIMESTAMP('{end_date} 23:59:59')
      AND REGEXP_CONTAINS(LOWER(V2Organizations), r'({combined_keywords})')
    """

def perform_dry_run(client, query, require_confirmation):
    """Perform a dry run to estimate cost and ask for user confirmation."""
    print("Step 2: Performing Dry Run to estimate data volume and cost...")
    try:
        dry_run_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        dry_run_job = client.query(query, job_config=dry_run_config)

        bytes_processed = dry_run_job.total_bytes_processed
        gb_processed = bytes_processed / (1024 ** 3)
        tb_processed = bytes_processed / (1024 ** 4)

        estimated_cost_usd = tb_processed * 5
        estimated_cost_thb = estimated_cost_usd * 35

        print(f"Estimated data scanned: {gb_processed:.2f} GB ({tb_processed:.4f} TB)")
        print(f"Estimated cost: ~{estimated_cost_thb:.2f} THB")

        if require_confirmation:
            confirm = input("\nDo you want to proceed with this query? (Type 'yes' to confirm): ")
            if confirm.lower() != 'yes':
                print("Operation cancelled by user. No charges incurred.")
                return False
        else:
            print("Confirmation disabled via config. Proceeding automatically...")
        
        return True
    except Exception as e:
        print(f"Dry run failed: {e}")
        return False

def process_query_in_chunks(client, query, keywords_dict):
    """
    Execute query and process results in chunks (Low-RAM Mode).
    Returns a dictionary of filtered DataFrames per ticker.
    """
    print("\nStep 3: Executing query and processing in chunks (Low-RAM Mode)...")
    query_job = client.query(query)
    
    # Use to_dataframe_iterable to download in chunks (avoids OOM)
    chunk_iterator = query_job.result().to_dataframe_iterable()
    
    # Store filtered results per ticker in memory
    filtered_dfs = {ticker: [] for ticker in keywords_dict.keys()}
    total_rows_processed = 0
    chunk_count = 0
    
    for df_chunk in chunk_iterator:
        chunk_count += 1
        if df_chunk.empty:
            continue
            
        total_rows_processed += len(df_chunk)
        print(f"  -> Processing chunk {chunk_count} (Rows: {len(df_chunk)})...")
        
        # Parse sentiment scores for the chunk
        df_chunk['sentiment_score'] = df_chunk['tone_data'].astype(str).apply(
            lambda x: float(x.split(',')[0]) if pd.notnull(x) and ',' in x else None
        )
        df_chunk = df_chunk.drop_duplicates(subset=['url'], keep='first')
        
        # Filter and collect to individual ticker lists
        for ticker, keyword in keywords_dict.items():
            sub_df = df_chunk[df_chunk['organizations'].str.lower().str.contains(keyword, na=False, regex=True)].copy()
            if not sub_df.empty:
                filtered_dfs[ticker].append(sub_df[['seendate', 'url', 'sentiment_score']])
                
    print(f"Query successful. Processed {chunk_count} chunks ({total_rows_processed} total rows).")
    return filtered_dfs

def save_filtered_data(filtered_dfs, keywords_dict, raw_dir):
    """Combine filtered chunks and append to respective CSV files."""
    print("Step 4: Saving filtered data to individual ticker files...")
    for ticker in keywords_dict.keys():
        if len(filtered_dfs[ticker]) > 0:
            final_df = pd.concat(filtered_dfs[ticker])
            final_df = final_df.sort_values('seendate', ascending=True)
            final_df = final_df.drop_duplicates(subset=['url'], keep='last')
            
            filename = f"{raw_dir}/{ticker}_news_raw.csv"
            
            if os.path.exists(filename):
                df_old = pd.read_csv(filename)
                df_combined = pd.concat([df_old, final_df])
                df_combined = df_combined.drop_duplicates(subset=['url'], keep='last')
                df_combined = df_combined.sort_values('seendate', ascending=True)
                df_combined.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"  [+] Appended: {filename} (Total records now: {len(df_combined)})")
            else:
                final_df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"  [+] Created: {filename} ({len(final_df)} records)")
        else:
            print(f"  [-] No new data found for {ticker} in this batch.")

def fetch_news(keywords_dict, start_date, end_date, raw_dir, require_confirmation=True):
    """
    Main orchestration function for fetching news sentiment data from BigQuery.
    """
    if not check_if_needs_update(keywords_dict, raw_dir):
        print("News data for all tickers is up to date (modified today). Skipping BigQuery.")
        return

    try:
        client = bigquery.Client()
    except Exception as e:
        print(f"Failed to initialize BigQuery client. Error: {e}")
        return

    # Prepare Query
    actual_start_date = get_incremental_start_date(keywords_dict, start_date, raw_dir)
    print(f"Step 1: Generating BigQuery SQL string (Scanning from {actual_start_date} to {end_date})...")
    query = build_gdelt_query(keywords_dict, actual_start_date, end_date)

    # Safety Check
    if not perform_dry_run(client, query, require_confirmation):
        return

    # Execution (Low-RAM chunk processing)
    try:
        filtered_dfs = process_query_in_chunks(client, query, keywords_dict)
        save_filtered_data(filtered_dfs, keywords_dict, raw_dir)
    except Exception as e:
        print(f"Error executing BigQuery: {e}")

    print("BigQuery data fetching complete.")