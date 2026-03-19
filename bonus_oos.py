"""
Bonus Challenge 1: Out-of-Sample Testing
-----------------------------------------
Train: 2015-2020  |  Test: 2021-2024  (no re-optimisation on test data)

Grid search runs on the training period ONLY. Best parameters are then
frozen and applied to the unseen test period — a clean simulation of
live deployment without data leakage.

Clearly separated from the main solution in main.py.
"""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from strategy   import MeanReversionStrategy
from backtester import SimpleBacktester

os.makedirs('results', exist_ok=True)

ENTRY_Z_GRID = [1.0, 1.5, 2.0]
WINDOW_GRID  = [50, 80, 120]


def _run_oos_grid_search(train, train_end):
    """Grid search on training data only. Returns (best_entry_z, best_window)."""
    print(f"\n  [OOS Grid Search — TRAIN ONLY  (<{train_end})]")
    records = []
    for ez in ENTRY_Z_GRID:
        for w in WINDOW_GRID:
            df_s = MeanReversionStrategy(train, window=w, entry_z=ez).generate_signals()
            bt   = SimpleBacktester(df_s)
            m    = bt.run()
            records.append({
                'window' : w,
                'entry_z': ez,
                'Sharpe' : round(m['Sharpe Ratio'],    3),
                'Return' : round(m['Total Return'],    4),
                'MDD'    : round(m['Maximum Drawdown'],4),
                'Trades' : int(m['Total Trades']),
            })
    results = (pd.DataFrame(records)
                 .sort_values('Sharpe', ascending=False)
                 .reset_index(drop=True))
    print(results.to_string(index=False))
    best = results.iloc[0]
    print(f"\n  Selected: window={int(best['window'])}  entry_z={best['entry_z']}  "
          f"Sharpe={best['Sharpe']}  (parameters now FROZEN)")
    return float(best['entry_z']), int(best['window'])


def _evaluate_period(prices_slice, window, entry_z, label):
    """Run strategy + backtest on a prices slice. Returns (metrics, bt.df)."""
    df_s = MeanReversionStrategy(prices_slice, window=window, entry_z=entry_z).generate_signals()
    bt   = SimpleBacktester(df_s)
    m    = bt.run()
    n_active = int((df_s['signal'] != 0).sum())
    print(f"\n  [{label}]  bars={len(prices_slice):,}  active={n_active:,}  "
          f"Sharpe={m['Sharpe Ratio']:.4f}  Return={m['Total Return']:.2%}  "
          f"MDD={m['Maximum Drawdown']:.2%}")
    return m, bt.df


def _print_comparison_table(train_m, test_m):
    pct_keys = {'Total Return', 'Annualized Return', 'Maximum Drawdown', 'Win Rate'}
    keys = [
        'Total Return', 'Annualized Return', 'Sharpe Ratio', 'Maximum Drawdown',
        'Win Rate', 'Average Trade Duration (min)', 'Total Trades',
        'Sortino Ratio', 'Calmar Ratio', 'Profit Factor',
    ]
    print("\n" + "="*70)
    print("  OOS COMPARISON TABLE")
    print(f"  {'Metric':<38} {'Train 2015-20':>15} {'Test OOS 2021-24':>16}")
    print("="*70)
    for k in keys:
        if k not in train_m:
            continue
        tv, ov = train_m[k], test_m[k]
        if isinstance(tv, float):
            fmt = lambda v: f"{v:.2%}" if k in pct_keys else f"{v:.4f}"
        else:
            fmt = lambda v: str(v)
        print(f"  {k:<38} {fmt(tv):>15} {fmt(ov):>16}")
    print("="*70)

    # Flag performance degradation
    sr_diff = test_m['Sharpe Ratio'] - train_m['Sharpe Ratio']
    note = "degraded" if sr_diff < -0.5 else ("improved" if sr_diff > 0.5 else "stable")
    print(f"\n  OOS Sharpe change: {sr_diff:+.4f}  ({note})")
    print("  (Some degradation is expected — IS parameters are optimised to IS data.)")


def _plot_oos(bt_train_df, bt_test_df, best_window, best_entry_z,
              save_path='results/bonus_oos.png'):
    # Scale test equity curve so it continues visually from where train ended
    train_final    = bt_train_df['equity_curve'].iloc[-1]
    test_ec_scaled = bt_test_df['equity_curve'] * train_final
    split_ts       = bt_test_df.index[0]

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45, height_ratios=[3, 2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(bt_train_df.index, bt_train_df['equity_curve'],
             color='steelblue', linewidth=0.6, label='Train IS (2015-2020)')
    ax1.plot(bt_test_df.index, test_ec_scaled,
             color='darkorange', linewidth=0.6, label='Test OOS (2021-2024)')
    ax1.axvline(split_ts, color='black', linestyle='--', linewidth=1.0,
                alpha=0.7, label='Train / Test Split')
    ax1.axhline(1.0, color='grey', linestyle=':', alpha=0.4, linewidth=0.7)
    ax1.set_title(f'Equity Curve — IS & OOS  (window={best_window}, entry_z={best_entry_z})',
                  fontsize=11)
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(loc='upper right', fontsize=8)

    ax2.fill_between(bt_train_df.index, bt_train_df['drawdown'], 0,
                     color='steelblue', alpha=0.45, label='Train Drawdown')
    ax2.fill_between(bt_test_df.index, bt_test_df['drawdown'], 0,
                     color='darkorange', alpha=0.45, label='OOS Drawdown')
    ax2.axvline(split_ts, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    ax2.set_title('Drawdown by Period')
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')
    ax2.legend(loc='lower left', fontsize=8)

    fig.suptitle('BONUS 1 — Out-of-Sample Test  (no look-ahead across the split)',
                 fontsize=13, fontweight='bold')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def run_bonus_oos(prices, train_end='2021-01-01'):
    """
    Top-level entry point. Called from main.py.
    Splits data at train_end, grid-searches on train only,
    evaluates both periods with frozen params, prints table, saves plot.
    """
    print("\n" + "=" * 60)
    print("BONUS 1 — OUT-OF-SAMPLE TESTING")
    print("=" * 60)

    train = prices[prices.index < train_end]
    test  = prices[prices.index >= train_end]
    print(f"  Train: {train.index[0].date()} → {train.index[-1].date()}  "
          f"({len(train):,} bars)")
    print(f"  Test : {test.index[0].date()}  → {test.index[-1].date()}   "
          f"({len(test):,} bars)")

    best_entry_z, best_window = _run_oos_grid_search(train, train_end)

    print("\n  Evaluating frozen params on each period:")
    train_m, bt_train_df = _evaluate_period(train, best_window, best_entry_z,
                                            label='Train IS  2015-2020')
    test_m,  bt_test_df  = _evaluate_period(test,  best_window, best_entry_z,
                                            label='Test  OOS 2021-2024')

    _print_comparison_table(train_m, test_m)
    _plot_oos(bt_train_df, bt_test_df, best_window, best_entry_z)
    print("\n  Saved → results/bonus_oos.png")
