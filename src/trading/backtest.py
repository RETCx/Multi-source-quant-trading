"""
Dynamic Horizon Backtest Engine
================================
 Backtest  Dynamic 

Logic:
  - Day T: Check confidence (probability) of all Horizons (1D, 3D, 5D, 10D)
  - Find Horizon with highest confidence
  - If value > Threshold -> Open Trade and set Time Exit by Horizon
  - ATR Stop Loss providing continuous protection
"""
import numpy as np
import pandas as pd


def run_dynamic_backtest(
    df_prices: pd.DataFrame,
    ensembled_oos: dict,
    target_horizons: list,
    initial_capital: float = 100_000,
    position_size: float = 1.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    prob_threshold_pct: int = 85,
    sl_multiplier: float = 2.0,
    atr_col: str = 'ATR14',
    full_index_for_mapping = None,
    rolling_window: int = 252,
    strength_mode: str = 'z_score',
) -> dict:
    """
    Dynamic Horizon Backtest Engine
    
     probability  horizon  horizon 
     Time Exit  dynamic
    
    Parameters
    ----------
    df_prices : DataFrame
        DataFrame Contains adjOpen, adjHigh, adjLow, adjClose, and ATR columns
        Index  DatetimeIndex
    ensembled_oos : dict
        dict  oos_results.pkl['ensembled']  target_name
        : {'indices': [...], 'preds': [...], 'probas': [...], 'trues': [...]}
    target_horizons : list
          [1, 3, 5, 10]  Target  ensembled_oos
    initial_capital : float
        
    transaction_cost : float
         commission  leg (entry + exit = x2)
    prob_threshold_pct : int
        Percentile  Adaptive Threshold ( 85 =  85% )
    sl_multiplier : float
        ATR Multiplier  Stop Loss
    atr_col : str
        ATR column name in df_prices
    rolling_window : int
        Rolling window size for calculating historical probability statistics
    strength_mode : str
        'z_score' or 'diff' (absolute difference from average)
        
    Returns
    -------
    dict :
        - 'equity_curve': Series Daily
        - 'trade_log': DataFrame  Trade 
        - 'final_value': Final portfolio value
        - 'total_return': Total return (decimal)
    """
    # ====================================================
    # 1. Build aligned signal arrays per day
    # ====================================================
    target_names = [f"Target_{h}D" for h in target_horizons]

    #  mapping: date -> row index  df_prices ( filter )
    valid_dates = set(df_prices.index)
    date_to_filtered_row = {date: i for i, date in enumerate(df_prices.index)}

    n_rows = len(df_prices)
    n_horizons = len(target_horizons)
    prob_matrix = np.zeros((n_rows, n_horizons))
    dir_matrix = np.zeros((n_rows, n_horizons), dtype=int)

    #  date  OOS index —  df_prices_full  full_index_for_mapping
    #  fallback:  indices   df_prices  filter (len )
    full_index = full_index_for_mapping if full_index_for_mapping is not None else ensembled_oos.get('_full_index', None)

    for h_idx, (t_name, h) in enumerate(zip(target_names, target_horizons)):
        if t_name not in ensembled_oos:
            continue
        oos = ensembled_oos[t_name]
        oos_indices = oos['indices']
        oos_preds   = np.array(oos['preds'])
        oos_probas  = np.array(oos['probas'])

        for i, data_idx in enumerate(oos_indices):
            prob = float(oos_probas[i])
            pred = int(oos_preds[i])
            direction = 1 if pred == 1 else -1

            #  full_index ->  oos_row_number -> date -> filtered_row
            if full_index is not None:
                if data_idx >= len(full_index):
                    continue
                oos_date = pd.Timestamp(full_index[data_idx])
                if oos_date not in valid_dates:
                    continue
                row = date_to_filtered_row[oos_date]
            else:
                #  full_index:  data_idx  ( filter)
                if data_idx >= n_rows:
                    continue
                row = data_idx

            prob_matrix[row, h_idx] = prob
            dir_matrix[row, h_idx]  = direction

    # Compute rolling stats for each horizon to get baseline
    df_probs = pd.DataFrame(prob_matrix, columns=target_names)
    df_probs_valid = df_probs.replace(0.0, np.nan)
    
    rolling_mean = df_probs_valid.rolling(window=rolling_window, min_periods=50).mean()
    rolling_std  = df_probs_valid.rolling(window=rolling_window, min_periods=50).std()
    
    expanding_mean = df_probs_valid.expanding(min_periods=5).mean()
    expanding_std  = df_probs_valid.expanding(min_periods=5).std()
    
    final_mean = rolling_mean.fillna(expanding_mean).fillna(df_probs_valid.mean()).fillna(0.5)
    final_std  = rolling_std.fillna(expanding_std).fillna(df_probs_valid.std()).fillna(0.05)

    # Calculate Strength Score matrix (Absolute Deviation to capture both Long and Short)
    strength_matrix = np.zeros_like(prob_matrix)
    for h_idx in range(n_horizons):
        h_mean = final_mean.iloc[:, h_idx].values
        h_std  = final_std.iloc[:, h_idx].values
        h_prob = prob_matrix[:, h_idx]
        
        if strength_mode == 'z_score':
            safe_std = np.where(h_std > 1e-6, h_std, 1e-6)
            strength_matrix[:, h_idx] = np.where(h_prob > 0, np.abs((h_prob - h_mean) / safe_std), -999.0)
        else: # 'diff'
            strength_matrix[:, h_idx] = np.where(h_prob > 0, np.abs(h_prob - h_mean), -999.0)

    # Now, for each day, pick the horizon with the highest Absolute Strength Score
    signal_prob  = np.zeros(n_rows)
    signal_dir   = np.zeros(n_rows, dtype=int)
    signal_h     = np.zeros(n_rows, dtype=int)
    signal_strength = np.zeros(n_rows)

    for row in range(n_rows):
        valid_h_idxs = [h_idx for h_idx in range(n_horizons) if prob_matrix[row, h_idx] > 0]
        if not valid_h_idxs:
            continue
        
        best_h_idx = max(valid_h_idxs, key=lambda idx: strength_matrix[row, idx])
        
        signal_prob[row]     = prob_matrix[row, best_h_idx]
        signal_dir[row]      = dir_matrix[row, best_h_idx]
        signal_h[row]        = target_horizons[best_h_idx]
        signal_strength[row] = strength_matrix[row, best_h_idx]

    # ====================================================
    # 2. Compute ATR (fallback if not in df_prices)
    # ====================================================
    if atr_col in df_prices.columns:
        atr_series = df_prices[atr_col].values
    else:
        #  ATR14 
        print(f"[INFO] Column '{atr_col}' not found. Computing ATR14 on the fly.")
        high = df_prices['adjHigh']
        low  = df_prices['adjLow']
        close = df_prices['adjClose']
        hl  = high - low
        hc  = (high - close.shift()).abs()
        lc  = (low  - close.shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr_series = tr.rolling(14).mean().values

    # ====================================================
    # 3. Event-Driven Simulation
    # ====================================================
    balance       = initial_capital
    in_position   = False
    entry_idx     = 0
    entry_price   = 0.0
    sl_price      = 0.0
    trade_side    = ""
    hold_days_target = 0
    invested_amount = 0.0
    
    daily_equity  = []
    trade_details = []
    daily_signals = []
    
    #  probability  Adaptive Threshold
    past_probs = []
    
    opens  = df_prices['adjOpen'].values
    highs  = df_prices['adjHigh'].values
    lows   = df_prices['adjLow'].values
    closes = df_prices['adjClose'].values
    
    for i in range(n_rows):
        today_open  = opens[i]
        today_high  = highs[i]
        today_low   = lows[i]
        today_close = closes[i]
        
        yesterday_prob = 0.0
        yesterday_dir = 0
        yesterday_h = 0
        yesterday_confidence = 0.0
        thresh = 0.0
        action = "Skipped (Day 0)"
        
        # ---- Entry Logic () ----
        if i > 0:
            yesterday_prob  = signal_prob[i - 1]
            yesterday_dir   = signal_dir[i - 1]
            yesterday_h     = signal_h[i - 1]
            
            if yesterday_prob > 0:
                yesterday_confidence = yesterday_prob if yesterday_dir == 1 else 1.0 - yesterday_prob
            
            #  Adaptive Threshold (Rolling Window of last 120 signals)
            valid_past = [p for p in past_probs if p > 1e-6]
            if len(valid_past) >= 50:
                # Use only the last 120 valid signals so threshold can drop when market gets noisy
                recent_past = valid_past[-120:]
                thresh = np.percentile(recent_past, prob_threshold_pct)
            else:
                thresh = 0.55  # default if insufficient data
                
            if in_position:
                action = "Skipped (Already in position)"
            else:
                if yesterday_confidence >= thresh and yesterday_dir != 0 and yesterday_h > 0:
                    action = f"Traded ({'Long' if yesterday_dir == 1 else 'Short'})"
                    in_position      = True
                    entry_idx        = i
                    entry_price      = today_open
                    trade_side       = "Long" if yesterday_dir == 1 else "Short"
                    hold_days_target = yesterday_h
                    entry_prob       = yesterday_confidence
                    entry_thresh     = thresh
                    invested_amount  = balance * position_size
                    
                    # ATR Stop Loss  ATR 
                    atr_val = atr_series[i - 1]
                    if not np.isnan(atr_val):
                        if trade_side == "Long":
                            sl_price = entry_price - sl_multiplier * atr_val
                        else:
                            sl_price = entry_price + sl_multiplier * atr_val
                    else:
                        sl_price = entry_price * (0.95 if trade_side == "Long" else 1.05)
                else:
                    if yesterday_confidence == 0:
                        action = "Skipped (No signal)"
                    elif yesterday_confidence < thresh:
                        action = f"Skipped (Confidence {yesterday_confidence:.4f} < Threshold {thresh:.4f})"
                    else:
                        action = "Skipped (Invalid signal)"
        
        daily_signals.append({
            'Date': df_prices.index[i].strftime('%Y-%m-%d'),
            'Max_Confidence': yesterday_confidence,
            'Threshold': thresh,
            'Target_Horizon': yesterday_h,
            'Direction': yesterday_dir,
            'Action': action
        })
        
        #  confidence  threshold 
        if signal_prob[i] > 0:
            today_conf = signal_prob[i] if signal_dir[i] == 1 else 1.0 - signal_prob[i]
            past_probs.append(today_conf)
        
        # ---- Exit Logic ----
        if in_position:
            days_held  = i - entry_idx
            force_exit = False
            exit_price = 0.0
            exit_reason = ""
            
            # A. ATR Stop Loss
            if trade_side == "Long" and today_low <= sl_price:
                exit_price  = min(today_open, sl_price)
                force_exit  = True
                exit_reason = f"Stop Loss @ {sl_price:.2f}"
            elif trade_side == "Short" and today_high >= sl_price:
                exit_price  = max(today_open, sl_price)
                force_exit  = True
                exit_reason = f"Stop Loss @ {sl_price:.2f}"
            
            # B. Time Exit ( h )
            if not force_exit and (days_held >= hold_days_target - 1 or i == n_rows - 1):
                exit_price  = today_close
                force_exit  = True
                exit_reason = f"Time Exit (h={hold_days_target}d)"
            
            if force_exit:
                # Calculate return with slippage applied to entry and exit
                if trade_side == "Long":
                    actual_entry = entry_price * (1 + slippage)
                    actual_exit  = exit_price * (1 - slippage)
                    ret_pct = (actual_exit / actual_entry) - 1
                else:
                    actual_entry = entry_price * (1 - slippage)
                    actual_exit  = exit_price * (1 + slippage)
                    ret_pct = (actual_entry / actual_exit) - 1
                
                net_ret  = ret_pct - (transaction_cost * 2)
                pnl      = invested_amount * net_ret
                balance += pnl
                
                trade_details.append({
                    'entry_date':  df_prices.index[entry_idx],
                    'exit_date':   df_prices.index[i],
                    'side':        trade_side,
                    'confidence':  entry_prob,
                    'threshold':   entry_thresh,
                    'entry_price': entry_price,
                    'exit_price':  exit_price,
                    'sl_price':    sl_price,
                    'hold_days':   days_held + 1,
                    'horizon_h':   hold_days_target,
                    'exit_reason': exit_reason,
                    'return_pct':  net_ret,
                    'pnl':         pnl,
                    'balance':     balance,
                })
                in_position = False
        
        # Mark-to-Market Daily
        if in_position:
            if trade_side == "Long":
                unrealized = (today_close / entry_price) - 1
            else:
                unrealized = (entry_price / today_close) - 1
            daily_equity.append(balance + (invested_amount * unrealized))
        else:
            daily_equity.append(balance)
    
    # ====================================================
    # 4. Package Results
    # ====================================================
    equity_curve = pd.Series(daily_equity, index=df_prices.index)
    trade_log    = pd.DataFrame(trade_details)
    daily_signals_df = pd.DataFrame(daily_signals)
    total_return = (balance / initial_capital) - 1
    
    # Benchmark: Buy & Hold adjClose
    bh_start = df_prices['adjClose'].iloc[0]
    benchmark = initial_capital * (df_prices['adjClose'] / bh_start)
    
    return {
        'equity_curve':   equity_curve,
        'benchmark':      benchmark,
        'trade_log':      trade_log,
        'daily_signals':  daily_signals_df,
        'final_value':    balance,
        'total_return':   total_return,
        'initial_capital': initial_capital,
    }
