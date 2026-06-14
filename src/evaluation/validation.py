"""
Trading Performance Validation Metrics
=======================================
ฟังก์ชันสำหรับประเมินผล Strategy อย่างครอบคลุม
(Migrated from VALIDATION_TOOLS_FOR_RESEARCH.ipynb)
"""
import numpy as np
import pandas as pd


def calculate_trading_metrics(
    equity_curve: pd.Series,
    trade_log: pd.DataFrame = None,
    risk_free_rate: float = 0.02
) -> dict:
    """
    คำนวณ Metrics ครบถ้วนสำหรับ Trading Strategy
    
    Parameters
    ----------
    equity_curve : Series  รายวัน
    trade_log    : DataFrame  ที่มีคอลัมน์ return_pct (optional)
    risk_free_rate : float  อัตราดอกเบี้ยปราศจากความเสี่ยงต่อปี (default 2%)
    
    Returns
    -------
    dict ของ Metrics ทั้งหมด
    """
    returns = equity_curve.pct_change().fillna(0)
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_years = len(returns) / 252

    # ---- Return Metrics ----
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # ---- Risk Metrics ----
    daily_rf = risk_free_rate / 252
    excess   = returns - daily_rf
    sharpe   = float(np.sqrt(252) * excess.mean() / (returns.std() + 1e-9))

    # Sortino (Downside only)
    downside = returns[returns < 0]
    d_std    = downside.std() if len(downside) > 0 else 1e-9
    sortino  = float(np.sqrt(252) * excess.mean() / (d_std + 1e-9))

    # Max Drawdown
    cummax = equity_curve.cummax()
    dd     = (equity_curve - cummax) / cummax
    max_dd = float(dd.min())

    # Max DD Duration
    is_dd = dd < 0
    dd_dur, cur_dur = 0, 0
    for in_dd in is_dd:
        if in_dd:
            cur_dur += 1
            dd_dur = max(dd_dur, cur_dur)
        else:
            cur_dur = 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    vol    = float(returns.std() * np.sqrt(252))
    var95  = float(np.percentile(returns, 5))
    es95   = float(returns[returns <= var95].mean()) if len(returns[returns <= var95]) > 0 else 0.0

    # ---- Trade Metrics ----
    if trade_log is not None and len(trade_log) > 0 and 'return_pct' in trade_log.columns:
        t_ret   = trade_log['return_pct']
        wins    = t_ret[t_ret > 0]
        losses  = t_ret[t_ret < 0]
        n_trades = len(t_ret)
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        avg_win  = float(wins.mean())  if len(wins) > 0  else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        gp = float(wins.sum())
        gl = abs(float(losses.sum()))
        profit_factor  = gp / gl if gl > 0 else np.inf
        expectancy     = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        wl_ratio       = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        avg_hold       = float(trade_log['hold_days'].mean()) if 'hold_days' in trade_log.columns else np.nan
    else:
        win_rate = avg_win = avg_loss = profit_factor = expectancy = wl_ratio = avg_hold = np.nan
        n_trades = 0

    return {
        # Return
        'Total Return':         total_return,
        'CAGR':                 cagr,
        'Avg Daily Return':     float(returns.mean()),
        # Risk
        'Sharpe Ratio':         sharpe,
        'Sortino Ratio':        sortino,
        'Calmar Ratio':         calmar,
        'Max Drawdown':         max_dd,
        'Max DD Duration (days)': dd_dur,
        'Volatility (Annual)':  vol,
        'VaR (95%)':            var95,
        'Expected Shortfall':   es95,
        # Trade
        'Total Trades':         n_trades,
        'Win Rate':             win_rate,
        'Avg Win':              avg_win,
        'Avg Loss':             avg_loss,
        'Profit Factor':        profit_factor,
        'Expectancy':           expectancy,
        'Win/Loss Ratio':       wl_ratio,
        'Avg Hold Days':        avg_hold,
    }


def print_trading_metrics(metrics: dict, title: str = "Strategy Performance"):
    """แสดงผล Metrics แบบสวยงามใน Terminal"""
    print(f"\n{'='*60}")
    print(f"[REPORT] {title}")
    print(f"{'='*60}")

    def _fmt_pct(v):  return f"{v:>10.2%}" if not np.isnan(v) else "       N/A"
    def _fmt_f2(v):   return f"{v:>10.3f}" if not np.isnan(v) else "       N/A"
    def _fmt_int(v):  return f"{int(v):>10d}" if not np.isnan(v) else "       N/A"

    print("\n[RETURN METRICS]")
    print(f"   Total Return         : {_fmt_pct(metrics['Total Return'])}")
    print(f"   CAGR                 : {_fmt_pct(metrics['CAGR'])}")
    print(f"   Avg Daily Return     : {metrics['Avg Daily Return']:>10.4%}")

    print("\n[RISK METRICS]")
    print(f"   Sharpe Ratio         : {_fmt_f2(metrics['Sharpe Ratio'])}")
    print(f"   Sortino Ratio        : {_fmt_f2(metrics['Sortino Ratio'])}")
    print(f"   Calmar Ratio         : {_fmt_f2(metrics['Calmar Ratio'])}")
    print(f"   Max Drawdown         : {_fmt_pct(metrics['Max Drawdown'])}")
    print(f"   Max DD Duration      : {_fmt_int(metrics['Max DD Duration (days)'])} days")
    print(f"   Volatility (Annual)  : {_fmt_pct(metrics['Volatility (Annual)'])}")
    print(f"   VaR (95%)            : {metrics['VaR (95%)']:>10.4%}")
    print(f"   Expected Shortfall   : {metrics['Expected Shortfall']:>10.4%}")

    if not np.isnan(metrics['Win Rate']):
        print("\n[TRADE METRICS]")
        print(f"   Total Trades         : {_fmt_int(metrics['Total Trades'])}")
        print(f"   Win Rate             : {_fmt_pct(metrics['Win Rate'])}")
        print(f"   Avg Win              : {_fmt_pct(metrics['Avg Win'])}")
        print(f"   Avg Loss             : {_fmt_pct(metrics['Avg Loss'])}")
        print(f"   Profit Factor        : {_fmt_f2(metrics['Profit Factor'])}")
        print(f"   Expectancy           : {_fmt_pct(metrics['Expectancy'])}")
        print(f"   Win/Loss Ratio       : {_fmt_f2(metrics['Win/Loss Ratio'])}")
        if not np.isnan(metrics['Avg Hold Days']):
            print(f"   Avg Hold Days        : {metrics['Avg Hold Days']:>10.1f} days")

    print(f"{'='*60}\n")


def save_backtest_artifacts(bt_result: dict, metrics: dict, model_dir: str, prefix: str, ticker: str, title: str = None) -> str:
    """
    เซฟผลลัพธ์จาก Backtest (CSV, Metrics, Plot) ลงโฟลเดอร์เดียว
    และคืนค่า Path ของโฟลเดอร์ที่เซฟ
    """
    import matplotlib.pyplot as plt
    from src.evaluation.tracker import ExperimentManager
    
    exp_mgr = ExperimentManager(base_dir=model_dir, prefix=prefix, ticker=ticker)
    
    # Save logs
    if len(bt_result['trade_log']) > 0:
        exp_mgr.save_dataframe(bt_result['trade_log'], "trade_log.csv")
    metrics_df = pd.DataFrame([metrics])
    exp_mgr.save_dataframe(metrics_df, "backtest_metrics.csv")
    
    # Save Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    axes[0].plot(bt_result['equity_curve'].index, bt_result['equity_curve'].values, color='#00C8FF', linewidth=1.8, label=f'{ticker} Strategy')
    axes[0].plot(bt_result['benchmark'].index,    bt_result['benchmark'].values,    color='#888888', linewidth=1.2, linestyle='--', alpha=0.7, label='Buy & Hold')
    
    peak = bt_result['equity_curve'].cummax()
    axes[0].fill_between(bt_result['equity_curve'].index, peak.values, bt_result['equity_curve'].values, alpha=0.25, color='red', label='Drawdown')
    
    plot_title = title if title else f'{ticker} — Dynamic Horizon Strategy'
    axes[0].set_title(f'{plot_title}\nSharpe: {metrics["Sharpe Ratio"]:.2f} | Return: {metrics["Total Return"]:.1%}', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.2)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    drawdown = (bt_result['equity_curve'] - bt_result['equity_curve'].cummax()) / bt_result['equity_curve'].cummax()
    axes[1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.5)
    axes[1].set_title('Drawdown', fontsize=10)
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    axes[1].grid(True, alpha=0.2)
    
    plt.tight_layout()
    exp_mgr.save_figure(fig, "equity_curve.png")
    plt.close(fig)
    
    return exp_mgr.run_dir
