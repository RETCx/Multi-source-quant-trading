import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch
import pickle
import json
import concurrent.futures
from datetime import datetime
from src.evaluation.metrics import evaluate_and_report, create_model_comparison_df
from src.evaluation.tracker import ExperimentManager
from src.models.pipeline import run_walk_forward_pipeline
from src.data_pipeline.walk_forward import WalkForwardCV
from src.utils import set_seed

def train_stock(ticker, config):
    try:
        SEED = config['random_seed']
        set_seed(SEED)
        
        DATA_PATH = f"{config['data_path']['features']}/{ticker}_with_indicators.csv"
        
        TARGET_COLS = config['target_columns']
        EXCLUDE_COLS = config.get('exclude_columns', [])
        PARAMS = config['model_params']
        
        CV_CONFIG = {
            'min_train_size': config['walk_forward']['min_train_size'],
            'test_size': config['walk_forward']['test_size'],
            'gap': config['walk_forward']['gap'],
            'step_size': config['walk_forward']['step_size'],
        }
        max_train = config['walk_forward'].get('max_train_size')
        if max_train is not None:
            CV_CONFIG['max_train_size'] = max_train
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n[{ticker}] Loading data and preparing features from {DATA_PATH}...")
        if not os.path.exists(DATA_PATH):
            return ticker, False, f"Data file not found: {DATA_PATH}"
            
        df = pd.read_csv(DATA_PATH, index_col='date', parse_dates=True)
        df = df.dropna(subset=TARGET_COLS)
        
        fallback_features = [col for col in df.columns if col not in TARGET_COLS + EXCLUDE_COLS]
        
        best_features_path = f"data/models/best_features_{ticker}.json"
        if os.path.exists(best_features_path):
            print(f"[{ticker}] Loading optimal features from {best_features_path}...")
            with open(best_features_path, 'r') as f:
                best_data = json.load(f)
                features = best_data.get('features', fallback_features)
        else:
            features = fallback_features
        
        X_raw = df[features].values
        y_raw = df[TARGET_COLS].values 
        
        print(f"[{ticker}] Data Shape: {X_raw.shape[0]} rows, {X_raw.shape[1]} features")
        
        cv_splitter = WalkForwardCV(**CV_CONFIG)
        folds = cv_splitter.summary(len(X_raw))
        
        if not folds:
            return ticker, False, "Not enough data to create Walk-Forward Folds"
        
        print(f"\n[{ticker}] Starting Walk-Forward Training...")
        pipeline_results = run_walk_forward_pipeline(
            X_raw=X_raw,
            y_raw=y_raw,
            features=features,
            target_cols=TARGET_COLS,
            params=PARAMS,
            cv_config=CV_CONFIG,
            device=device,
            seed=SEED,
            verbose=False  # Reduce verbosity when running parallel
        )
        
        if not pipeline_results:
            return ticker, False, "Pipeline returned no results."
        
        ensembled_oos = pipeline_results['ensembled_oos']
        raw_results = pipeline_results['raw_results']
        train_accs_mean = pipeline_results['train_accs_mean']
        
        results_summary = {}
        for head_idx, target_name in enumerate(TARGET_COLS):
            f_trues = ensembled_oos[target_name]['trues']
            f_preds = ensembled_oos[target_name]['preds']
            f_probas = ensembled_oos[target_name]['probas']
            
            acc = evaluate_and_report(f_trues, f_preds, target_name, verbose=False)
            
            results_summary[target_name] = {
                'train_acc': train_accs_mean[target_name],
                'test_acc': acc,
                'avg_confidence': np.mean(f_probas)
            }
        
        df_summary = create_model_comparison_df(results_summary)
        
        exp = ExperimentManager(base_dir="data/models", prefix="train", ticker=ticker)
        exp.save_dataframe(df_summary, "model_summary.csv")
        exp.save_config('config.yaml')
        
        oos_data = {
            'raw': {
                'indices': raw_results['indices'],
                'probas': raw_results['probas'],
                'trues': raw_results['trues'],
                'fold_ids': raw_results['fold_ids'],
            },
            'ensembled': ensembled_oos,
            'target_cols': TARGET_COLS,
        }
        exp.save_pickle(oos_data, "oos_results.pkl")
        
        all_oos_indices = set()
        for t_name in TARGET_COLS:
            if t_name in ensembled_oos:
                all_oos_indices.update(ensembled_oos[t_name]['indices'])
        all_oos_indices = sorted(all_oos_indices)
        
        oos_dates = df.index[all_oos_indices]
        df_oos_csv = pd.DataFrame(index=oos_dates)
        df_oos_csv.index.name = 'Date'
        
        for t_name in TARGET_COLS:
            if t_name in ensembled_oos:
                oos = ensembled_oos[t_name]
                t_indices = oos['indices']
                t_dates = df.index[t_indices]
                
                pred_series = pd.Series(oos['preds'], index=t_dates)
                prob_series = pd.Series(oos['probas'], index=t_dates)
                true_series = pd.Series(oos['trues'], index=t_dates)
                
                df_oos_csv[f"{t_name}_Pred"] = pred_series
                df_oos_csv[f"{t_name}_Prob"] = prob_series.round(4)
                df_oos_csv[f"{t_name}_True"] = true_series
        
        df_oos_csv = df_oos_csv.sort_index()
        exp.save_dataframe(df_oos_csv, "oos_predictions.csv")
        
        return ticker, True, f"Successfully trained and saved artifacts. Run ID: {exp.run_id}"
        
    except Exception as e:
        return ticker, False, f"Exception occurred: {str(e)}"

if __name__ == '__main__':
    # Fix for multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    tickers = config['data_settings']['selected_tickers']
    print(f"===========================================================")
    print(f"MULTI-STOCK TRAINING PIPELINE | Total Stocks: {len(tickers)}")
    print(f"===========================================================")
    
    # Run sequentially if only 1 stock, or parallel if > 1
    max_workers = min(len(tickers), 4)  # Limit to 4 parallel jobs to save RAM/VRAM
    
    start_time = datetime.now()
    results = []
    
    if len(tickers) == 1:
        res = train_stock(tickers[0], config)
        results.append(res)
    else:
        print(f"Starting parallel training with {max_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(train_stock, ticker, config): ticker for ticker in tickers}
            for future in concurrent.futures.as_completed(futures):
                ticker = futures[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    results.append((ticker, False, f"Process generated an exception: {exc}"))
                    
    print("\n===========================================================")
    print("TRAINING SUMMARY")
    print("===========================================================")
    for ticker, success, msg in results:
        status = "OK" if success else "FAILED"
        print(f"[{status}] {ticker}: {msg}")
        
    elapsed = datetime.now() - start_time
    print(f"\nTotal execution time: {elapsed}")