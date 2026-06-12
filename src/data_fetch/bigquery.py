import os
import pandas as pd
from google.cloud import bigquery
from src.utils import needs_download

def fetch_news(keywords_dict, start_date, end_date, raw_dir, require_confirmation=True):
    """
    Fetches news sentiment data from Google BigQuery in a single optimized query.
    Implements Incremental Load to only fetch new data and avoid scanning years of data daily.
    """
    
    # ---------------------------------------------------------
    # 1. Gatekeeper Check: Do we even need to run today?
    # ---------------------------------------------------------
    all_up_to_date = True
    for ticker in keywords_dict.keys():
        filepath = f"{raw_dir}/{ticker}_news_raw.csv"
        if needs_download(filepath):
            all_up_to_date = False
            break # If even one is outdated, we need to run BQ
            
    if all_up_to_date:
        print("News data for all tickers is up to date (modified today). Skipping BigQuery.")
        return

    try:
        client = bigquery.Client()
    except Exception as e:
        print(f"Failed to initialize BigQuery client. Error: {e}")
        return

    # ---------------------------------------------------------
    # 2. Incremental Start Date Logic
    # ---------------------------------------------------------
    actual_start_date = start_date
    latest_dates = []
    
    for ticker in keywords_dict.keys():
        filepath = f"{raw_dir}/{ticker}_news_raw.csv"
        if os.path.exists(filepath):
            try:
                # Read existing data to find the last updated date
                df_existing = pd.read_csv(filepath)
                if not df_existing.empty:
                    last_date = df_existing['seendate'].max()
                    # Convert to YYYY-MM-DD for BQ string
                    last_date_str = pd.to_datetime(last_date).strftime('%Y-%m-%d')
                    latest_dates.append(last_date_str)
            except Exception:
                pass
                
    if latest_dates:
        # We take the oldest "latest date" among all tickers to ensure we don't miss any gaps
        actual_start_date = min(latest_dates)
        print(f"Incremental Update Triggered: Adjusting start_date from {start_date} to {actual_start_date}")

    # Combine all keywords into a single regex string
    combined_keywords = "|".join([v for v in keywords_dict.values()])

    print(f"Step 1: Generating BigQuery SQL string (Scanning from {actual_start_date} to {end_date})...")

    query = f"""
    SELECT
      PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS seendate,
      DocumentIdentifier AS url,
      V2Organizations AS organizations,
      V2Tone AS tone_data 
    FROM
      `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE
      _PARTITIONTIME BETWEEN TIMESTAMP('{actual_start_date} 00:00:00') AND TIMESTAMP('{end_date} 23:59:59')
      AND REGEXP_CONTAINS(LOWER(V2Organizations), r'({combined_keywords})')
    """

    # --- SAFETY CHECK (Dry Run) ---
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
                return
        else:
            print("Confirmation disabled via config. Proceeding automatically...")

    except Exception as e:
        print(f"Dry run failed: {e}")
        return

    # --- EXECUTION ---
    print("\nStep 3: Executing query...")
    try:
        query_job = client.query(query)
        df_new = query_job.to_dataframe(progress_bar_type='tqdm')
        print(f"Query successful. Total new rows fetched: {len(df_new)}")
        
        if not df_new.empty:
            print("Step 4: Processing sentiment scores...")
            df_new['sentiment_score'] = df_new['tone_data'].astype(str).apply(
                lambda x: float(x.split(',')[0]) if pd.notnull(x) and ',' in x else None
            )
            df_new = df_new.drop_duplicates(subset=['url'], keep='first')
            
            print("Step 5: Filtering and appending to individual ticker files...")
            for ticker, keyword in keywords_dict.items():
                sub_df = df_new[df_new['organizations'].str.lower().str.contains(keyword, na=False, regex=True)].copy()
                
                if not sub_df.empty:
                    sub_df = sub_df.sort_values('seendate', ascending=True)
                    final_df = sub_df[['seendate', 'url', 'sentiment_score']]
                    
                    filename = f"{raw_dir}/{ticker}_news_raw.csv"
                    
                    # ---------------------------------------------------------
                    # 3. Append Logic (Combine old + new, drop overlaps)
                    # ---------------------------------------------------------
                    if os.path.exists(filename):
                        df_old = pd.read_csv(filename)
                        # Concat old and new
                        df_combined = pd.concat([df_old, final_df])
                        # Drop duplicates based on URL just in case of date overlap
                        df_combined = df_combined.drop_duplicates(subset=['url'], keep='last')
                        df_combined = df_combined.sort_values('seendate', ascending=True)
                        df_combined.to_csv(filename, index=False, encoding='utf-8-sig')
                        print(f"Appended: {filename} (Total records now: {len(df_combined)})")
                    else:
                        final_df.to_csv(filename, index=False, encoding='utf-8-sig')
                        print(f"Created: {filename} ({len(final_df)} records)")
                else:
                    print(f"No new data found for {ticker} in this batch.")
                    
    except Exception as e:
        print(f"Error executing BigQuery: {e}")

    print("BigQuery data fetching complete.")