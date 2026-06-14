"""
04_run_backtest.py
==================
Script for running Dynamic Horizon Backtest Engine

Usage:
    python 04_run_backtest.py

Input:
    - data/features/{TICKER}_with_indicators.csv  ( + ATR)
    - data/models/train_{TICKER}_*/oos_results.pkl  ( OOS)
    - config.yaml

Output:
    - Saved to original or new ExperimentManager run directory
    - backtest_summary.csv
    - trade_log.csv
    - equity_curve.png
"""
import os
import sys
import glob
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # No Display needed, save to file instead
import matplotlib.pyplot as plt

from src.trading.backtest import run_dynamic_backtest
from src.evaluation.validation import calculate_trading_metrics, print_trading_metrics
from src.evaluation.tracker import ExperimentManager

# ==========================================
# 0. CONFIG
# ==========================================
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

STOCK_SYMBOL    = config['data_settings']['selected_tickers'][0]
TARGET_HORIZONS = config.get('target_horizons', [1, 3, 5, 10])
TARGET_COLS     = config.get('target_columns', [f"Target_{h}D" for h in TARGET_HORIZONS])
DATA_PATH       = f"{config['data_path']['features']}/{STOCK_SYMBOL}_with_indicators.csv"
MODEL_DIR       = config['data_path']['models']

BT_CONFIG = config.get('backtest', {})
INITIAL_CAPITAL    = BT_CONFIG.get('initial_capital', 100_000)
POSITION_SIZE      = BT_CONFIG.get('position_size', 1.0)
TRANSACTION_COST   = BT_CONFIG.get('transaction_cost', 0.001)
PROB_THRESHOLD_PCT = BT_CONFIG.get('prob_threshold', 85)
SL_MULTIPLIER      = BT_CONFIG.get('sl_multiplier', 2.0)
ATR_COLUMN         = BT_CONFIG.get('atr_column', 'ATR14')
BT_START_DATE      = BT_CONFIG.get('start_date', None)   # None = 
BT_END_DATE        = BT_CONFIG.get('end_date', None)

# ==========================================
# 1. LOAD LATEST OOS PICKLE
# ==========================================
# Find latest run in data/models/
pattern = os.path.join(MODEL_DIR, f"train_{STOCK_SYMBOL}_*", "oos_results.pkl")
pkl_files = sorted(glob.glob(pattern))

if not pkl_files:
    print(f"[ERROR] Could not find oos_results.pkl for {STOCK_SYMBOL}")
    print(f"         Please run python 03_run_training.py first.")
    sys.exit(1)

latest_pkl = pkl_files[-1]
latest_run_dir = os.path.dirname(latest_pkl)
print(f"[OK] Loading OOS Predictions from: {latest_pkl}")

with open(latest_pkl, 'rb') as f:
    oos_data = pickle.load(f)

ensembled_oos = oos_data.get('ensembled', {})
if not ensembled_oos:
    print("[ERROR] Ensembled data not found in pickle")
    print("         Please run 03_run_training.py with latest code version.")
    sys.exit(1)

print(f"[OK] Available Horizons: {list(ensembled_oos.keys())}")

# ==========================================
# 2. LOAD PRICE DATA (for Backtest)
# ==========================================
print(f"\n[INFO] Load price data from: {DATA_PATH}")
df_prices = pd.read_csv(DATA_PATH, index_col='date', parse_dates=True)
print(f"       Total rows: {len(df_prices)} | Columns: {len(df_prices.columns)}")

required_cols = ['adjOpen', 'adjHigh', 'adjLow', 'adjClose']
missing = [c for c in required_cols if c not in df_prices.columns]
if missing:
    print(f"[ERROR] Missing columns: {missing}")
    sys.exit(1)

# Must dropna(subset=TARGET_COLS) to match 03_run_training for index alignment
df_prices = df_prices.dropna(subset=TARGET_COLS)
full_index_for_mapping = df_prices.index.copy()  # Keep full index to map with oos_indices

# Filter date range (if specified in config) for Backtest
if BT_START_DATE or BT_END_DATE:
    date_mask = pd.Series(True, index=df_prices.index)
    if BT_START_DATE:
        date_mask &= df_prices.index >= pd.Timestamp(BT_START_DATE)
    if BT_END_DATE:
        date_mask &= df_prices.index <= pd.Timestamp(BT_END_DATE)
    df_prices = df_prices[date_mask]
    print(f"[FILTER] Date range: {BT_START_DATE or 'start'} -> {BT_END_DATE or 'end'} ({len(df_prices)} rows)")

# ==========================================
# 3. RUN DYNAMIC HORIZON BACKTEST
# ==========================================
print(f"\n{'='*55}")
print(f"DYNAMIC HORIZON BACKTEST | {STOCK_SYMBOL}")
print(f"{'='*55}")
print(f"  Capital     : ${INITIAL_CAPITAL:,.0f}")
print(f"  Tx Cost     : {TRANSACTION_COST*100:.2f}% per leg")
print(f"  ATR SL Mult : {SL_MULTIPLIER}x")
print(f"  Prob Thresh : {PROB_THRESHOLD_PCT}th percentile")
print(f"  Horizons    : {TARGET_HORIZONS} days")
if BT_START_DATE or BT_END_DATE:
    print(f"  Date Range  : {BT_START_DATE or 'start'} -> {BT_END_DATE or 'end'}")
print(f"{'='*55}\n")

result = run_dynamic_backtest(
    df_prices           = df_prices,
    ensembled_oos       = ensembled_oos,
    target_horizons     = TARGET_HORIZONS,
    initial_capital     = INITIAL_CAPITAL,
    position_size       = POSITION_SIZE,
    transaction_cost    = TRANSACTION_COST,
    prob_threshold_pct  = PROB_THRESHOLD_PCT,
    sl_multiplier       = SL_MULTIPLIER,
    atr_col             = ATR_COLUMN,
    full_index_for_mapping = full_index_for_mapping,
)

if result is None:
    print("[ERROR] Backtest  ")
    sys.exit(1)

# ==========================================
# 4. VALIDATION METRICS
# ==========================================
metrics = calculate_trading_metrics(
    equity_curve = result['equity_curve'],
    trade_log    = result['trade_log'],
)
print_trading_metrics(metrics, title=f"{STOCK_SYMBOL} — Dynamic Horizon Strategy")

# Trade Distribution
if len(result['trade_log']) > 0:
    tl = result['trade_log']
    print("[TRADE DISTRIBUTION by Horizon]")
    if 'horizon_h' in tl.columns:
        h_dist = tl.groupby('horizon_h').agg(
            trades     = ('return_pct', 'count'),
            win_rate   = ('return_pct', lambda x: (x > 0).mean()),
            avg_return = ('return_pct', 'mean')
        ).reset_index()
        h_dist.columns = ['Hold Days', 'Trades', 'Win Rate', 'Avg Return']
        h_dist['Win Rate']   = h_dist['Win Rate'].map(lambda x: f"{x:.1%}")
        h_dist['Avg Return'] = h_dist['Avg Return'].map(lambda x: f"{x:+.2%}")
        print(h_dist.to_string(index=False))
    print()

# ==========================================
# 5. SAVE RESULTS
# ==========================================
from src.evaluation.validation import save_backtest_artifacts

run_dir = save_backtest_artifacts(
    bt_result=result,
    metrics=metrics,
    model_dir=MODEL_DIR,
    prefix="backtest",
    ticker=STOCK_SYMBOL
)

print(f"\n[DONE] Backtest results saved to: {run_dir}")
print(f"       Trades: {len(result['trade_log'])} | Final: ${result['final_value']:,.2f} | Return: {result['total_return']:.2%}")
