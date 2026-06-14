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
    transaction_cost: float = 0.001,
    prob_threshold_pct: int = 85,
    sl_multiplier: float = 2.0,
    atr_col: str = 'ATR14',
    full_index_for_mapping = None,
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
    signal_prob = np.zeros(n_rows)
    signal_dir  = np.zeros(n_rows, dtype=int)
    signal_h    = np.zeros(n_rows, dtype=int)

    #  date  OOS index —  df_prices_full  full_index_for_mapping
    #  fallback:  indices   df_prices  filter (len )
    full_index = full_index_for_mapping if full_index_for_mapping is not None else ensembled_oos.get('_full_index', None)

    for t_name, h in zip(target_names, target_horizons):
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

            if prob > signal_prob[row]:
                signal_prob[row] = prob
                signal_dir[row]  = direction
                signal_h[row]    = h

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
    
    daily_equity  = []
    trade_details = []
    
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
        
        # ---- Entry Logic () ----
        if not in_position and i > 0:
            yesterday_prob  = signal_prob[i - 1]
            yesterday_dir   = signal_dir[i - 1]
            yesterday_h     = signal_h[i - 1]
            
            #  Adaptive Threshold 
            valid_past = [p for p in past_probs if p > 1e-6]
            if len(valid_past) >= 50:
                thresh = np.percentile(valid_past, prob_threshold_pct)
            else:
                thresh = 0.55  # default if insufficient data
            
            if yesterday_prob >= thresh and yesterday_dir != 0 and yesterday_h > 0:
                in_position      = True
                entry_idx        = i
                entry_price      = today_open
                trade_side       = "Long" if yesterday_dir == 1 else "Short"
                hold_days_target = yesterday_h
                
                # ATR Stop Loss  ATR 
                atr_val = atr_series[i - 1]
                if not np.isnan(atr_val):
                    if trade_side == "Long":
                        sl_price = entry_price - sl_multiplier * atr_val
                    else:
                        sl_price = entry_price + sl_multiplier * atr_val
                else:
                    sl_price = entry_price * (0.95 if trade_side == "Long" else 1.05)
        
        #  probability  threshold 
        if signal_prob[i] > 0:
            past_probs.append(signal_prob[i])
        
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
                if trade_side == "Long":
                    ret_pct = (exit_price / entry_price) - 1
                else:
                    ret_pct = (entry_price / exit_price) - 1
                
                net_ret  = ret_pct - (transaction_cost * 2)
                pnl      = balance * net_ret
                balance += pnl
                
                trade_details.append({
                    'entry_date':  df_prices.index[entry_idx],
                    'exit_date':   df_prices.index[i],
                    'side':        trade_side,
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
            daily_equity.append(balance * (1 + unrealized))
        else:
            daily_equity.append(balance)
    
    # ====================================================
    # 4. Package Results
    # ====================================================
    equity_curve = pd.Series(daily_equity, index=df_prices.index)
    trade_log    = pd.DataFrame(trade_details)
    total_return = (balance / initial_capital) - 1
    
    # Benchmark: Buy & Hold adjClose
    bh_start = df_prices['adjClose'].iloc[0]
    benchmark = initial_capital * (df_prices['adjClose'] / bh_start)
    
    return {
        'equity_curve':   equity_curve,
        'benchmark':      benchmark,
        'trade_log':      trade_log,
        'final_value':    balance,
        'total_return':   total_return,
        'initial_capital': initial_capital,
    }
