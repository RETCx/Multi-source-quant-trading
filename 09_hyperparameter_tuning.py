import os
import sys
import yaml
import json
import optuna
import logging
import pandas as pd
import numpy as np
import torch
from src.models.pipeline import run_walk_forward_pipeline
from src.trading.backtest import run_dynamic_backtest
from src.evaluation.validation import calculate_trading_metrics
from src.utils import set_seed

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def objective(trial):
    config = load_config()
    SEED = config['random_seed']
    set_seed(SEED)
    
    STOCK_SYMBOL = config['data_settings']['selected_tickers'][0]
    DATA_PATH = f"{config['data_path']['features']}/{STOCK_SYMBOL}_with_indicators.csv"
    
    TARGET_COLS = config['target_columns']
    EXCLUDE_COLS = config.get('exclude_columns', [])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Suggest hyperparameters
    params = {
        'sequence_length': trial.suggest_int('sequence_length', 5, 20),
        'hidden_size': trial.suggest_categorical('hidden_size', [16, 32, 64]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
        'epochs': 100,
        'patience': 15,
        'batch_size': 64
    }
    
    df = pd.read_csv(DATA_PATH, index_col='date', parse_dates=True)
    df = df.dropna(subset=TARGET_COLS)
    
    best_features_path = config.get('best_features_path', f"data/models/best_features_{STOCK_SYMBOL}.json")
    if os.path.exists(best_features_path):
        with open(best_features_path, 'r') as f:
            best_data = json.load(f)
            features = best_data.get('features', [])
    else:
        fallback_features = [col for col in df.columns if col not in TARGET_COLS + EXCLUDE_COLS]
        features = fallback_features
        
    df = df.dropna()
    X_raw = df[features].values
    y_raw = df[TARGET_COLS].values
    
    cv_config = {
        'min_train_size': config['walk_forward']['min_train_size'],
        'test_size': config['walk_forward']['test_size'],
        'gap': config['walk_forward']['gap'],
        'step_size': config['walk_forward']['step_size'],
    }
    if config['walk_forward'].get('max_train_size') is not None:
        cv_config['max_train_size'] = config['walk_forward']['max_train_size']
        
    # Disable printing inside the pipeline
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        pipeline_results = run_walk_forward_pipeline(
            X_raw=X_raw,
            y_raw=y_raw,
            features=features,
            target_cols=TARGET_COLS,
            params=params,
            cv_config=cv_config,
            device=device,
            seed=SEED,
            verbose=False
        )
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
        
    if not pipeline_results:
        return -1.0
        
    ensembled_oos = pipeline_results['ensembled_oos']
    
    # Evaluate with dynamic backtest
    bt_config = config['backtest']
    
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        bt_result = run_dynamic_backtest(
            df_prices=df,
            ensembled_oos=ensembled_oos,
            target_horizons=[int(t.split('_')[1].replace('D','')) for t in TARGET_COLS],
            initial_capital=bt_config.get('initial_capital', 100_000),
            position_size=bt_config.get('position_size', 1.0),
            transaction_cost=bt_config.get('transaction_cost', 0.001),
            prob_threshold_pct=bt_config.get('prob_threshold', 85),
            sl_multiplier=bt_config.get('sl_multiplier', 2.0),
            atr_col=bt_config.get('atr_column', 'ATR14'),
            full_index_for_mapping=df.index
        )
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
        
    if not bt_result:
        return -1.0
        
    metrics = calculate_trading_metrics(bt_result['equity_curve'], bt_result['trade_log'])
    sharpe = metrics.get('Sharpe Ratio', -1.0)
    
    return sharpe

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    study_name = "lstm-hyperparameter-tuning"
    storage_name = "sqlite:///data/models/optuna_study.db"
    
    # Needs optuna to be installed
    import subprocess
    try:
        import optuna
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
        import optuna
        
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        direction='maximize', 
        load_if_exists=True
    )
    
    logging.info("Starting Hyperparameter Tuning...")
    # Run 10 trials to keep it relatively fast
    study.optimize(objective, n_trials=10)
    
    logging.info("="*50)
    logging.info("TUNING COMPLETED")
    logging.info(f"Best Trial ID: {study.best_trial.number}")
    logging.info(f"Best Sharpe Ratio: {study.best_trial.value}")
    logging.info("Best Params:")
    for key, value in study.best_trial.params.items():
        logging.info(f"    {key}: {value}")
        
    # Update config.yaml with best params
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    config['model_params'].update(study.best_trial.params)
    
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, sort_keys=False)
        
    logging.info("Updated config.yaml with best params!")
