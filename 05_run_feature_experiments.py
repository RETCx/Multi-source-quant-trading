"""
05_run_feature_experiments.py
==============================
Automated Feature Ablation Lab

This script runs WFA Training + Backtest automatically 
Tests various Feature Exclusions and compares them to find the best Sharpe ratio

Usage:
    python 05_run_feature_experiments.py

Output:
    - data/models/feature_experiment_scoreboard.csv
"""
import os
import sys
import glob
import pickle
import copy
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

from src.data_pipeline.walk_forward import WalkForwardCV
from src.data_pipeline.sequence import create_multi_horizon_sequences, get_dataloaders
from src.models.architecture import MultiHorizonLSTM
from src.models.trainer import MultiHeadTrainer
from src.evaluation.ensemble import ensemble_overlapping_predictions
from src.evaluation.validation import calculate_trading_metrics
from src.trading.backtest import run_dynamic_backtest
from src.utils import set_seed
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
import torch

# ==========================================
# 0. LOAD BASE CONFIG
# ==========================================
with open('config.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

SEED         = base_config['random_seed']
STOCK_SYMBOL = base_config['data_settings']['selected_tickers'][0]
DATA_PATH    = f"{base_config['data_path']['features']}/{STOCK_SYMBOL}_with_indicators.csv"
MODEL_DIR    = base_config['data_path']['models']
TARGET_COLS  = base_config['target_columns']
HORIZONS     = base_config.get('target_horizons', [1, 3, 5, 10])
PARAMS       = base_config['model_params']
BT_CFG       = base_config.get('backtest', {})

# ==========================================
# 1. DEFINE FEATURE EXPERIMENTS
# ==========================================
# Each experiment is a dict containing:
#   'name'     : Name for Scoreboard
#   'exclude'  : List of columns added beyond base exclusion

BASE_EXCLUDE = base_config.get('exclude_columns', [])

EXPERIMENTS = [
    {
        'name': 'Baseline (config.yaml)',
        'exclude': BASE_EXCLUDE,
    },
    {
        'name': 'Drop Sentiment Features',
        'exclude': BASE_EXCLUDE + [
            'Sent_Decay_3D', 'Sent_Momentum', 'Sent_ZScore',
            'Sent_Gap_Interaction', 'feat_VIX_Gap', 'feat_VIX_Intraday',
        ],
    },
    {
        'name': 'Drop Market Context (SPY/VIX)',
        'exclude': BASE_EXCLUDE + [
            'SPY_Return', 'SPY_Log_Ret', 'SPY_Close',
            'VIX_Close', 'VIX_High', 'VIX_Low', 'VIX_Open', 'VIX_Change',
        ],
    },
    {
        'name': 'Keep Only Core Technicals',
        'exclude': BASE_EXCLUDE + [
            'Sent_Decay_3D', 'Sent_Momentum', 'Sent_ZScore',
            'Sent_Gap_Interaction', 'feat_VIX_Gap', 'feat_VIX_Intraday',
            'SPY_Return', 'SPY_Log_Ret', 'SPY_Close',
            'VIX_Close', 'VIX_High', 'VIX_Low', 'VIX_Open', 'VIX_Change',
            'daily_PE', 'daily_PBV', 'PE_SMA200', 'PBV_SMA200',
        ],
    },
]

# ==========================================
# 2. LOAD PRICE DATA (ONCE)
# ==========================================
print(f"Loading feature data for {STOCK_SYMBOL}...")
df_full = pd.read_csv(DATA_PATH, index_col='date', parse_dates=True)
df_full = df_full.dropna(subset=TARGET_COLS)
df_prices_bt = df_full.copy()  # Use this file for backtest as well (contains adjOpen, adjClose, etc.)

# Filter date range (if specified in config) for Backtest
bt_start_date = BT_CFG.get('start_date', None)
bt_end_date = BT_CFG.get('end_date', None)

if bt_start_date or bt_end_date:
    date_mask = pd.Series(True, index=df_prices_bt.index)
    if bt_start_date:
        date_mask &= df_prices_bt.index >= pd.Timestamp(bt_start_date)
    if bt_end_date:
        date_mask &= df_prices_bt.index <= pd.Timestamp(bt_end_date)
    df_prices_bt = df_prices_bt[date_mask]
    print(f"[FILTER] Scoreboard Evaluation Range: {bt_start_date or 'start'} -> {bt_end_date or 'end'} ({len(df_prices_bt)} rows)")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | Seed: {SEED}\n")

# ==========================================
# 3. CV SPLITTER (SHARED)
# ==========================================
CV_CONFIG = {
    'min_train_size': base_config['walk_forward']['min_train_size'],
    'test_size':      base_config['walk_forward']['test_size'],
    'gap':            base_config['walk_forward']['gap'],
    'step_size':      base_config['walk_forward']['step_size'],
}
max_train = base_config['walk_forward'].get('max_train_size')
if max_train:
    CV_CONFIG['max_train_size'] = max_train

scoreboard = []

# ==========================================
# 4. EXPERIMENT LOOP
# ==========================================
for exp_i, experiment in enumerate(EXPERIMENTS):
    exp_name    = experiment['name']
    exclude_set = set(experiment['exclude']).union(set(BASE_EXCLUDE))
    
    print(f"\n{'#'*60}")
    print(f"EXPERIMENT {exp_i+1}/{len(EXPERIMENTS)}: {exp_name}")
    print(f"{'#'*60}")

    # Filter features for this experiment
    features = [c for c in df_full.columns if c not in TARGET_COLS and c not in exclude_set]
    X_raw = df_full[features].values
    y_raw = df_full[TARGET_COLS].values
    n_heads = len(TARGET_COLS)
    
    print(f"Features: {len(features)} columns")
    
    from src.models.pipeline import run_walk_forward_pipeline
    
    # ---------- WFA Loop ----------
    pipeline_results = run_walk_forward_pipeline(
        X_raw=X_raw,
        y_raw=y_raw,
        features=features,
        target_cols=TARGET_COLS,
        params=PARAMS,
        cv_config=CV_CONFIG,
        device=device,
        seed=SEED,
        verbose=True
    )
    
    if not pipeline_results:
        print("[WARN] Pipeline returned no results. Skipping.")
        continue
        
    ensembled_oos = pipeline_results['ensembled_oos']

    # ---------- Backtest ----------
    bt_result = run_dynamic_backtest(
        df_prices          = df_prices_bt,
        ensembled_oos      = ensembled_oos,
        target_horizons    = HORIZONS,
        initial_capital    = BT_CFG.get('initial_capital', 100_000),
        transaction_cost   = BT_CFG.get('transaction_cost', 0.001),
        prob_threshold_pct = BT_CFG.get('prob_threshold', 85),
        sl_multiplier      = BT_CFG.get('sl_multiplier', 2.0),
        atr_col            = BT_CFG.get('atr_column', 'ATR14'),
        full_index_for_mapping = df_full.index.copy(),
    )
    
    if bt_result is None:
        print("[WARN] Backtest returned no results. Skipping metrics.")
        metrics = {}
    else:
        metrics = calculate_trading_metrics(bt_result['equity_curve'], bt_result['trade_log'])
        print(f"  -> Sharpe: {metrics.get('Sharpe Ratio', 0):.3f} | "
              f"Return: {metrics.get('Total Return', 0):.2%} | "
              f"MaxDD: {metrics.get('Max Drawdown', 0):.2%} | "
              f"Trades: {metrics.get('Total Trades', 0)}")
              
        # --- Save Artifacts per Experiment ---
        from src.evaluation.validation import save_backtest_artifacts
        
        safe_exp_name = exp_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        exp_prefix = f"backtest_exp_{safe_exp_name}"
        
        save_backtest_artifacts(
            bt_result=bt_result,
            metrics=metrics,
            model_dir=MODEL_DIR,
            prefix=exp_prefix,
            ticker=STOCK_SYMBOL,
            title=exp_name
        )

    # ---------- Log to Scoreboard ----------
    scoreboard.append({
        'Experiment':     exp_name,
        'Features':       len(features),
        'Sharpe Ratio':   round(metrics.get('Sharpe Ratio', np.nan), 4),
        'Total Return':   round(metrics.get('Total Return', np.nan), 4),
        'CAGR':           round(metrics.get('CAGR', np.nan), 4),
        'Max Drawdown':   round(metrics.get('Max Drawdown', np.nan), 4),
        'Win Rate':       round(metrics.get('Win Rate', np.nan), 4),
        'Profit Factor':  round(metrics.get('Profit Factor', np.nan), 4),
        'Total Trades':   int(metrics.get('Total Trades', 0)),
        'Avg Hold Days':  round(metrics.get('Avg Hold Days', np.nan), 1),
        'Final Value':    round(bt_result['final_value'] if bt_result else np.nan, 2),
    })

# ==========================================
# 5. EXPORT SCOREBOARD
# ==========================================
df_score = pd.DataFrame(scoreboard).sort_values('Sharpe Ratio', ascending=False)

scoreboard_path = os.path.join(MODEL_DIR, "feature_experiment_scoreboard.csv")
df_score.to_csv(scoreboard_path, index=False)

print(f"\n{'='*70}")
print(f"FEATURE EXPERIMENT SCOREBOARD — {STOCK_SYMBOL}")
print(f"{'='*70}")
df_print = df_score.copy()
df_print['Total Return'] = df_print['Total Return'].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
df_print['Max Drawdown'] = df_print['Max Drawdown'].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
df_print['Win Rate']     = df_print['Win Rate'].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
print(df_print[['Experiment', 'Features', 'Sharpe Ratio', 'Total Return',
                'Max Drawdown', 'Win Rate', 'Total Trades', 'Avg Hold Days']].to_string(index=False))
print(f"\n[DONE] Scoreboard saved to: {scoreboard_path}")
