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
from src.evaluation.validation import calculate_trading_metrics, print_trading_metrics, print_backtest_config
from src.evaluation.tracker import ExperimentManager

def run_backtest_for_stock(stock_symbol, config):
    TARGET_HORIZONS = config.get('target_horizons', [1, 3, 5, 10])
    TARGET_COLS     = config.get('target_columns', [f"Target_{h}D" for h in TARGET_HORIZONS])
    DATA_PATH       = f"{config['data_path']['features']}/{stock_symbol}_with_indicators.csv"
    MODEL_DIR       = config['data_path']['models']
    
    BT_CONFIG = config.get('backtest', {})
    INITIAL_CAPITAL    = BT_CONFIG.get('initial_capital', 100_000)
    POSITION_SIZE      = BT_CONFIG.get('position_size', 1.0)
    TRANSACTION_COST   = BT_CONFIG.get('transaction_cost', 0.001)
    SLIPPAGE           = BT_CONFIG.get('slippage', 0.0005)
    PROB_THRESHOLD_PCT = BT_CONFIG.get('prob_threshold', 85)
    SL_MULTIPLIER      = BT_CONFIG.get('sl_multiplier', 2.0)
    ATR_COLUMN         = BT_CONFIG.get('atr_column', 'ATR14')
    STRENGTH_MODE      = BT_CONFIG.get('strength_mode', 'z_score')
    ROLLING_WINDOW     = config.get('target_rolling_window', 252)
    BT_START_DATE      = BT_CONFIG.get('start_date', None)
    BT_END_DATE        = BT_CONFIG.get('end_date', None)
    
    print(f"\n=======================================================")
    print(f"RUNNING BACKTEST FOR: {stock_symbol}")
    print(f"=======================================================")
    
    pattern = os.path.join(MODEL_DIR, f"train_{stock_symbol}_*", "oos_results.pkl")
    pkl_files = sorted(glob.glob(pattern))
    
    if not pkl_files:
        print(f"[ERROR] Could not find oos_results.pkl for {stock_symbol}")
        return
        
    latest_pkl = pkl_files[-1]
    print(f"[OK] Loading OOS Predictions from: {latest_pkl}")
    
    with open(latest_pkl, 'rb') as f:
        oos_data = pickle.load(f)
        
    ensembled_oos = oos_data.get('ensembled', {})
    if not ensembled_oos:
        print("[ERROR] Ensembled data not found in pickle")
        return
        
    print(f"[OK] Available Horizons: {list(ensembled_oos.keys())}")
    print(f"\n[INFO] Load price data from: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        return
        
    df_prices = pd.read_csv(DATA_PATH, index_col='date', parse_dates=True)
    print(f"       Total rows: {len(df_prices)} | Columns: {len(df_prices.columns)}")
    
    required_cols = ['adjOpen', 'adjHigh', 'adjLow', 'adjClose']
    missing = [c for c in required_cols if c not in df_prices.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return
        
    df_prices = df_prices.dropna(subset=TARGET_COLS)
    full_index_for_mapping = df_prices.index.copy()
    
    if BT_START_DATE or BT_END_DATE:
        date_mask = pd.Series(True, index=df_prices.index)
        if BT_START_DATE:
            date_mask &= df_prices.index >= pd.Timestamp(BT_START_DATE)
        if BT_END_DATE:
            date_mask &= df_prices.index <= pd.Timestamp(BT_END_DATE)
        df_prices = df_prices[date_mask]
        print(f"[FILTER] Date range: {BT_START_DATE or 'start'} -> {BT_END_DATE or 'end'} ({len(df_prices)} rows)")
        
    config_dict = {
        'initial_capital': INITIAL_CAPITAL,
        'transaction_cost': TRANSACTION_COST,
        'slippage': SLIPPAGE,
        'sl_multiplier': SL_MULTIPLIER,
        'prob_threshold_pct': PROB_THRESHOLD_PCT,
        'target_horizons': TARGET_HORIZONS,
        'start_date': BT_START_DATE,
        'end_date': BT_END_DATE,
        'strength_mode': STRENGTH_MODE,
        'rolling_window': ROLLING_WINDOW
    }
    print_backtest_config(stock_symbol, config_dict)
    
    result = run_dynamic_backtest(
        df_prices           = df_prices,
        ensembled_oos       = ensembled_oos,
        target_horizons     = TARGET_HORIZONS,
        initial_capital     = INITIAL_CAPITAL,
        position_size       = POSITION_SIZE,
        transaction_cost    = TRANSACTION_COST,
        slippage            = SLIPPAGE,
        prob_threshold_pct  = PROB_THRESHOLD_PCT,
        sl_multiplier       = SL_MULTIPLIER,
        atr_col             = ATR_COLUMN,
        full_index_for_mapping = full_index_for_mapping,
        rolling_window      = ROLLING_WINDOW,
        strength_mode       = STRENGTH_MODE,
    )
    
    if result is None:
        print(f"[ERROR] Backtest failed for {stock_symbol}")
        return
        
    metrics = calculate_trading_metrics(
        equity_curve = result['equity_curve'],
        trade_log    = result['trade_log'],
    )
    print_trading_metrics(metrics, title=f"{stock_symbol} — Dynamic Horizon Strategy")
    
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
        
    from src.evaluation.validation import save_backtest_artifacts
    run_dir = save_backtest_artifacts(
        bt_result=result,
        metrics=metrics,
        model_dir=MODEL_DIR,
        prefix="backtest",
        ticker=stock_symbol
    )
    
    print(f"\n[DONE] {stock_symbol} Backtest results saved to: {run_dir}")
    print(f"       Trades: {len(result['trade_log'])} | Final: ${result['final_value']:,.2f} | Return: {result['total_return']:.2%}")


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    tickers = config['data_settings']['selected_tickers']
    print(f"===========================================================")
    print(f"MULTI-STOCK BACKTEST PIPELINE | Total Stocks: {len(tickers)}")
    print(f"===========================================================")
    
    for ticker in tickers:
        run_backtest_for_stock(ticker, config)
