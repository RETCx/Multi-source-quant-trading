import numpy as np
import pandas as pd

def calc_sma(data, period=14, column='adjClose'):
    data[f'SMA{period}'] = data[column].rolling(window=period).mean()

def calc_ema(data, period=14, column='adjClose'):
    data[f'EMA{period}'] = data[column].ewm(span=period, adjust=False).mean()

def calc_wma(data, period=14, column='adjClose'):
    weights = np.arange(1, period + 1)
    def weighted_ma(x):
        return np.dot(x, weights) / weights.sum()
    data[f'WMA{period}'] = data[column].rolling(window=period).apply(weighted_ma, raw=True)

def calc_hma(data, period=14, column='adjClose'):
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    
    def calc_wma_series(series, p):
        weights = np.arange(1, p + 1)
        return series.rolling(window=p).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    wma_half = calc_wma_series(data[column], half_period)
    wma_full = calc_wma_series(data[column], period)
    raw_hma = 2 * wma_half - wma_full
    data[f'HMA{period}'] = calc_wma_series(raw_hma, sqrt_period)

def calc_tema(data, period=14, column='adjClose'):
    ema1 = data[column].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    data[f'TEMA{period}'] = 3 * (ema1 - ema2) + ema3

def calc_dmi(data, period=14):
    delta_high = data['adjHigh'].diff()
    delta_low = data['adjLow'].diff().abs()
    
    plus_dm = np.where((delta_high > delta_low) & (delta_high > 0), delta_high, 0)
    minus_dm = np.where((delta_low > delta_high) & (delta_low > 0), delta_low, 0)
    
    tr = pd.concat([
        data['adjHigh'] - data['adjLow'],
        (data['adjHigh'] - data['adjClose'].shift(1)).abs(),
        (data['adjLow'] - data['adjClose'].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    tr_s = tr.rolling(window=period).sum()
    pdm_s = pd.Series(plus_dm, index=data.index).rolling(window=period).sum()
    mdm_s = pd.Series(minus_dm, index=data.index).rolling(window=period).sum()
    
    data[f'PlusDI{period}'] = 100 * (pdm_s / tr_s)
    data[f'MinusDI{period}'] = 100 * (mdm_s / tr_s)
    
    dx = 100 * (data[f'PlusDI{period}'] - data[f'MinusDI{period}']).abs() / (data[f'PlusDI{period}'] + data[f'MinusDI{period}'])
    data[f'ADX{period}'] = dx.rolling(window=period).mean()

def calc_rsi(data, period=14, column='adjClose'):
    delta = data[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Wilder's Smoothing
    for i in range(period, len(data)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    data[f'RSI{period}'] = 100 - (100 / (1 + rs))

def calc_cmo(data, period=14, column='adjClose'):
    delta = data[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    sum_up = gain.rolling(window=period).sum()
    sum_down = loss.rolling(window=period).sum()
    data[f'CMO{period}'] = (sum_up - sum_down) / (sum_up + sum_down) * 100

def calc_cci(data, period=14):
    tp = (data['adjHigh'] + data['adjLow'] + data['adjClose']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mean_dev = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    data[f'CCI{period}'] = (tp - sma_tp) / (0.015 * mean_dev)

def calc_williams_r(data, period=14):
    highest_high = data['adjHigh'].rolling(window=period).max()
    lowest_low = data['adjLow'].rolling(window=period).min()
    data[f'WilliamsR{period}'] = -100 * (highest_high - data['adjClose']) / (highest_high - lowest_low)

def calc_roc(data, period=14, column='adjClose'):
    data[f'ROC{period}'] = ((data[column] - data[column].shift(period)) / data[column].shift(period)) * 100

def calc_macd(data, fast_period=12, slow_period=26, signal_period=9, column='adjClose'):
    ema_fast = data[column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data[column].ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    data[f'MACD_Line{fast_period}_{slow_period}'] = macd_line
    data[f'MACD_Signal{fast_period}_{slow_period}_{signal_period}'] = signal_line
    data[f'MACD_Histogram{fast_period}_{slow_period}_{signal_period}'] = macd_line - signal_line

def calc_cmfi(data, period=21):
    denominator = data['adjHigh'] - data['adjLow']
    mfu = ((data['adjClose'] - data['adjLow']) - (data['adjHigh'] - data['adjClose'])) / denominator.replace(0, np.nan)
    mfu = mfu.fillna(0)
    mfv_volume = mfu * data['adjVolume']
    data[f'CMFI{period}'] = mfv_volume.rolling(window=period).sum() / data['adjVolume'].rolling(window=period).sum() * 100

def calc_psi(data, period=12, column='adjClose'):
    price_change = data[column].diff()
    is_up_day = (price_change > 0).astype(int)
    up_days_count = is_up_day.rolling(window=period).sum()
    data[f'PSI{period}'] = (up_days_count / period) * 100