import os
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.trading.backtest import run_dynamic_backtest
from src.evaluation.validation import calculate_trading_metrics
import json

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_rf_proxy_wfa(X, y, df_prices, features, target_cols, cv_config, bt_config, df_full_index):
    """
    Runs Walk-Forward Analysis using a fast Random Forest proxy instead of LSTM,
    then evaluates the OOS predictions via the dynamic backtester.
    """
    n_samples = X.shape[0]
    min_train = cv_config['min_train_size']
    test_size = cv_config['test_size']
    step_size = cv_config['step_size']
    gap = cv_config['gap']
    
    ensembled_oos = {target: {'indices': [], 'probas': [], 'preds': [], 'trues': []} for target in target_cols}
    feature_importances = np.zeros(len(features))
    fold_count = 0
    
    total_folds = len(range(0, n_samples - min_train - test_size - gap + 1, step_size))
    
    for start_idx in range(0, n_samples - min_train - test_size - gap + 1, step_size):
        train_end = start_idx + min_train
        test_start = train_end + gap
        test_end = test_start + test_size
        
        if test_end > n_samples:
            test_end = n_samples
            
        X_train, y_train = X[start_idx:train_end], y[start_idx:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]
        
        for i, target in enumerate(target_cols):
            y_train_tgt = y_train[:, i]
            y_test_tgt = y_test[:, i]
            
            # Fast Proxy Model
            clf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train_tgt)
            
            # Predict OOS
            probas = clf.predict_proba(X_test)[:, 1] if len(clf.classes_) > 1 else np.zeros(len(X_test))
            preds = (probas > 0.5).astype(int)
            
            ensembled_oos[target]['indices'].extend(range(test_start, test_end))
            ensembled_oos[target]['probas'].extend(probas)
            ensembled_oos[target]['preds'].extend(preds)
            ensembled_oos[target]['trues'].extend(y_test_tgt)
            
            # Aggregate importance
            feature_importances += clf.feature_importances_
            
        fold_count += 1
        if fold_count % 5 == 0 or fold_count == total_folds:
            logging.info(f"   ... Processing WFA Fold {fold_count}/{total_folds}")
            
        if test_end == n_samples:
            break
            
    # Average feature importances
    feature_importances /= (fold_count * len(target_cols))
    
    # Run Backtest
    bt_result = run_dynamic_backtest(
        df_prices          = df_prices,
        ensembled_oos      = ensembled_oos,
        target_horizons    = [int(t.split('_')[1].replace('D','')) for t in target_cols],
        initial_capital    = bt_config.get('initial_capital', 100_000),
        position_size      = bt_config.get('position_size', 1.0),
        transaction_cost   = bt_config.get('transaction_cost', 0.001),
        prob_threshold_pct = bt_config.get('prob_threshold', 85),
        sl_multiplier      = bt_config.get('sl_multiplier', 2.0),
        atr_col            = bt_config.get('atr_column', 'ATR14'),
        full_index_for_mapping = df_full_index
    )
    
    if bt_result is None:
        return 0, feature_importances
        
    metrics = calculate_trading_metrics(bt_result['equity_curve'], bt_result['trade_log'])
    sharpe = metrics.get('Sharpe Ratio', 0)
    
    return sharpe, feature_importances

import logging

def main():
    config = load_config()
    stock = config['data_settings']['selected_tickers'][0]
    data_path = f"{config['data_path']['features']}/{stock}_with_indicators.csv"
    
    # Setup logging
    log_file = f"data/models/feature_selection_{stock}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"[{stock}] Starting Automated Feature Selection (RFECV Proxy)...")
    
    df = pd.read_csv(data_path, index_col='date', parse_dates=True)
    target_cols = config.get('target_columns', ['Target_1D', 'Target_3D', 'Target_5D', 'Target_7D', 'Target_10D'])
    
    # Drop rows with NaN targets or NaN features (like the first 200 days of SMA200)
    df = df.dropna()
        
    base_exclude = config.get('exclude_columns', [])
    current_features = [c for c in df.columns if c not in target_cols and c not in base_exclude]
    
    history = []
    
    while len(current_features) >= 5:
        X = df[current_features].values
        y = df[target_cols].values
        
        logging.info(f"\nEvaluating {len(current_features)} features...")
        sharpe, importances = run_rf_proxy_wfa(
            X, y, df, current_features, target_cols, 
            config['walk_forward'], config['backtest'], df.index
        )
        
        logging.info(f"-> Sharpe Ratio: {sharpe:.3f}")
        history.append({
            'num_features': len(current_features),
            'sharpe': sharpe,
            'features': current_features.copy()
        })
        
        # Drop the least important feature(s)
        if len(current_features) > 40:
            drop_count = 5
        else:
            drop_count = 1
            
        least_important_indices = np.argsort(importances)[:drop_count]
        # Sort indices in reverse to pop safely without affecting other indices
        dropped_features = []
        for idx in sorted(least_important_indices, reverse=True):
            dropped_features.append(current_features.pop(idx))
            
        logging.info(f"-> Dropped {drop_count} worst feature(s): {dropped_features}")
        
    # Find Best
    best_run = max(history, key=lambda x: x['sharpe'])
    logging.info("\n" + "="*50)
    logging.info("OPTIMAL FEATURE SET FOUND!")
    logging.info(f"Optimal Size: {best_run['num_features']} features")
    logging.info(f"Max Sharpe Ratio: {best_run['sharpe']:.3f}")
    logging.info("="*50)
    
    # Save to JSON
    out_file = f"data/models/best_features_{stock}.json"
    with open(out_file, 'w') as f:
        json.dump(best_run, f, indent=4)
    logging.info(f"Saved optimal features to: {out_file}")

if __name__ == "__main__":
    main()
