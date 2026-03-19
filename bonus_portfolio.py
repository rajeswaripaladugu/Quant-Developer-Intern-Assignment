"""
Bonus Challenge 2: Multiple Simultaneous Strategies + Risk Allocation
----------------------------------------------------------------------
Runs 3 mean reversion variants in parallel on the same BankNifty series:

  A  window=50,  entry_z=1.0  — aggressive: reacts to short spikes
  B  window=80,  entry_z=1.5  — balanced:  base strategy
  C  window=120, entry_z=2.0  — conservative: only large, sustained deviations

The three variants are partially decorrelated — A fires far more often
(low threshold, short window) while C only activates on rare extremes.

Risk allocation: rolling inverse-volatility weighting.
  Every REBAL_FREQ bars, each strategy's weight = (1/vol_i) / Σ(1/vol_j)
  where vol_i is the rolling std of that strategy's net returns over the
  last VOL_LOOKBACK bars. Higher volatility → smaller allocation.

This prevents a single unstable strategy from dominating the portfolio
during periods of regime change.

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

MPY          = 252 * 375   # annualisation constant, matches SimpleBacktester
VOL_LOOKBACK = 3000        # bars for vol estimation at each rebalance event
REBAL_FREQ   = 1000        # bars between rebalance events

PORTFOLIO_PARAMS = [
    {'window': 50,  'entry_z': 1.0, 'label': 'A: w50/ez1.0  (aggressive)'},
    {'window': 80,  'entry_z': 1.5, 'label': 'B: w80/ez1.5  (balanced)'},
    {'window': 120, 'entry_z': 2.0, 'label': 'C: w120/ez2.0 (conservative)'},
]
COLOURS = ['#d96060', '#5b8fd9', '#4caa5c']


def _run_strategies(prices, param_list):
    """
    Run each strategy variant on prices.
    Returns:
        rets_df : DataFrame of net_return, one column per strategy
        ec_df   : DataFrame of equity_curve, one column per strategy
        ind_metrics : list of metric dicts, one per strategy
    """
    rets, ecs, metrics = {}, {}, []
    for p in param_list:
        lbl = p['label']
        print(f"  Running {lbl}...")
        df_s = MeanReversionStrategy(
            prices, window=p['window'], entry_z=p['entry_z']
        ).generate_signals()
        bt = SimpleBacktester(df_s)
        m  = bt.run()
        rets[lbl]    = bt.df['net_return']
        ecs[lbl]     = bt.df['equity_curve']
        metrics.append((lbl, m))
        print(f"    Sharpe={m['Sharpe Ratio']:.4f}  "
              f"Return={m['Total Return']:.2%}  "
              f"MDD={m['Maximum Drawdown']:.2%}")
    return pd.DataFrame(rets), pd.DataFrame(ecs), metrics


def _compute_inv_vol_weights(returns_df):
    """
    Compute inverse-volatility weights rebalanced every REBAL_FREQ bars.

    Key design:
      1. Compute rolling vol at every bar (for smooth snapshot values).
      2. SAMPLE weights only at rebalance bar indices — hold constant in between.
      3. Reindex to full bar index then forward-fill between rebal events.
      4. Rows 0..(VOL_LOOKBACK-1) are NaN — insufficient history.

    Pitfall avoided: (rets * weights).sum(axis=1) with NaN weight rows uses
    skipna=True by default, summing only the non-NaN columns — those partial
    weights don't add to 1.0, giving wrong portfolio returns. The caller must
    zero out warm-up bars explicitly (see _compute_portfolio_returns).
    """
    vols = returns_df.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std()

    # Sample vol at each rebalance timestamp
    rebal_pos = np.arange(0, len(vols), REBAL_FREQ)
    w_snap    = vols.iloc[rebal_pos].copy()

    # Inverse vol — protect against zero vol with replace
    inv_vol  = 1.0 / w_snap.replace(0.0, np.nan)
    row_sums = inv_vol.sum(axis=1)
    # axis=0 required: broadcasts Series(row_sums) over rows, not columns
    w_norm   = inv_vol.div(row_sums, axis=0)

    # Forward-fill between rebalance events; NaN before first valid snapshot
    w_rebal  = w_norm.reindex(returns_df.index).ffill()
    return w_rebal


def _compute_portfolio_returns(returns_df, weights_df):
    """
    Weighted sum of strategy returns.
    Zero during warm-up (NaN weight rows) — see pitfall note in
    _compute_inv_vol_weights docstring.
    """
    weighted     = returns_df * weights_df        # NaN * value = NaN during warm-up
    port_ret_raw = weighted.sum(axis=1)           # skipna=True gives wrong warm-up values
    valid_mask   = weights_df.notna().all(axis=1) # True only when ALL weights are valid
    port_ret     = port_ret_raw.where(valid_mask, other=0.0)
    return port_ret


def _portfolio_metrics(port_ret):
    """Compute full metric set consistent with SimpleBacktester._calculate_metrics."""
    ec  = (1 + port_ret.fillna(0)).cumprod()
    dd  = ec / ec.cummax() - 1

    # Exclude flat warm-up bars (zero returns) from distributional stats
    r   = port_ret.replace(0.0, np.nan).dropna()
    n   = len(port_ret)
    mean_r, std_r = r.mean(), r.std()
    total_ret     = ec.iloc[-1] - 1
    ann_ret       = (1 + total_ret) ** (MPY / n) - 1
    down_r        = r[r < 0]
    gp            = r[r > 0].sum()
    gl            = r[r < 0].abs().sum()
    mdd           = dd.min()

    m = {
        'Total Return'     : total_ret,
        'Annualized Return': ann_ret,
        'Sharpe Ratio'     : (mean_r / std_r) * np.sqrt(MPY) if std_r > 0 else 0,
        'Maximum Drawdown' : mdd,
        'Sortino Ratio'    : (mean_r / down_r.std()) * np.sqrt(MPY) if len(down_r) > 0 else 0,
        'Calmar Ratio'     : ann_ret / abs(mdd) if mdd != 0 else 0,
        'Profit Factor'    : gp / gl if gl > 0 else 0,
    }
    return m, ec, dd


def _print_portfolio_metrics(ind_metrics, port_m):
    pct_keys = {'Total Return', 'Annualized Return', 'Maximum Drawdown'}
    print(f"\n  {'Metric':<35} ", end='')
    labels = [p['label'][:12] for p in PORTFOLIO_PARAMS]
    for lbl in labels:
        print(f"{lbl:>16}", end='')
    print(f"{'Portfolio':>16}")
    print("  " + "-" * (35 + 16 * (len(labels) + 1)))

    common_keys = ['Total Return', 'Annualized Return', 'Sharpe Ratio',
                   'Maximum Drawdown', 'Sortino Ratio', 'Profit Factor']
    for k in common_keys:
        fmt = lambda v: (f"{v:.2%}" if k in pct_keys else f"{v:.4f}") if isinstance(v, float) else str(v)
        print(f"  {k:<35} ", end='')
        for _, m in ind_metrics:
            print(f"{fmt(m.get(k, 0)):>16}", end='')
        print(f"{fmt(port_m[k]):>16}")


def _plot_portfolio(port_ret, weights_df, ec_df, port_m, port_ec, port_dd,
                    save_path='results/bonus_portfolio.png'):
    n = len(port_ret)

    # Downsample weights for weight chart (every 375 bars ≈ 1 session)
    STEP  = 375
    w_ds  = weights_df.iloc[::STEP].dropna()
    x_ds  = w_ds.index

    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45,
                            height_ratios=[3, 2, 2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])

    # Panel 1: equity curves
    for (col, ec_series), col_hex in zip(ec_df.items(), COLOURS):
        ax1.plot(ec_series.index, ec_series.values,
                 color=col_hex, linewidth=0.4, alpha=0.45, label=col[:20])
    ax1.plot(port_ec.index, port_ec.values,
             color='navy', linewidth=1.1, label=f'Portfolio  Sharpe={port_m["Sharpe Ratio"]:.3f}')
    ax1.axhline(1.0, color='grey', linestyle=':', alpha=0.4, linewidth=0.7)
    ax1.set_title(f'Portfolio vs Individual Strategy Equity Curves  '
                  f'(MDD={port_m["Maximum Drawdown"]:.2%})', fontsize=11)
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(loc='upper right', fontsize=7)

    # Panel 2: portfolio drawdown
    ax2.fill_between(port_dd.index, port_dd.values, 0,
                     color='crimson', alpha=0.45)
    ax2.set_title('Portfolio Drawdown')
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')

    # Panel 3: stacked area weight chart
    # stackplot expects x as DatetimeIndex and each array as a row
    w_arrays = [w_ds[col].values for col in weights_df.columns]
    ax3.stackplot(x_ds, *w_arrays,
                  labels=[p['label'][:20] for p in PORTFOLIO_PARAMS],
                  alpha=0.75, colors=COLOURS)
    ax3.set_ylim(0, 1)
    ax3.set_title('Strategy Weights — Inverse-Vol Rebalanced '
                  f'(every {REBAL_FREQ} bars, {VOL_LOOKBACK}-bar lookback)',
                  fontsize=10)
    ax3.set_ylabel('Weight')
    ax3.set_xlabel('Date')
    ax3.legend(loc='upper right', fontsize=7)

    fig.suptitle('BONUS 2 — Multi-Strategy Inverse-Volatility Portfolio',
                 fontsize=13, fontweight='bold')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def run_bonus_portfolio(prices):
    """
    Top-level entry point. Called from main.py.
    Runs 3 strategy variants → computes inv-vol weights → builds portfolio
    → prints metrics → saves plot.
    """
    print("\n" + "=" * 60)
    print("BONUS 2 — MULTI-STRATEGY INVERSE-VOL PORTFOLIO")
    print(f"  Strategies : {len(PORTFOLIO_PARAMS)}  "
          f"(windows={[p['window'] for p in PORTFOLIO_PARAMS]}, "
          f"entry_z={[p['entry_z'] for p in PORTFOLIO_PARAMS]})")
    print(f"  Risk alloc : inverse-vol, rebal every {REBAL_FREQ} bars, "
          f"{VOL_LOOKBACK}-bar lookback")
    print("=" * 60)

    rets_df, ec_df, ind_metrics = _run_strategies(prices, PORTFOLIO_PARAMS)

    print("\n  Computing inverse-volatility weights...")
    weights = _compute_inv_vol_weights(rets_df)
    valid_w = weights.dropna()
    print(f"  Warm-up period : {len(weights) - len(valid_w):,} bars  "
          f"(portfolio flat until {VOL_LOOKBACK}-bar vol history is available)")
    print(f"  Rebalance events: {len(np.arange(0, len(weights), REBAL_FREQ)):,}")

    print("\n  Weight stats at rebalance events:")
    print(valid_w.iloc[::REBAL_FREQ].describe().round(4).to_string())

    port_ret          = _compute_portfolio_returns(rets_df, weights)
    port_m, port_ec, port_dd = _portfolio_metrics(port_ret)

    print("\n  PERFORMANCE — INDIVIDUAL STRATEGIES vs PORTFOLIO")
    _print_portfolio_metrics(ind_metrics, port_m)

    _plot_portfolio(port_ret, weights, ec_df, port_m, port_ec, port_dd)
    print("\n  Saved → results/bonus_portfolio.png")
