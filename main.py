import time, os, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_loader import load_and_clean
from analysis    import run_eda
from strategy    import MeanReversionStrategy
from backtester  import SimpleBacktester

t_start = time.time()
os.makedirs('results', exist_ok=True)


def grid_search(df_prices, entry_z_vals=[1.0, 1.5], window_vals=[50, 80]):
    """Grid search over entry_z and window. Selects by Sharpe (not return)."""
    print("\n" + "="*60)
    print("STEP 2 - PARAMETER OPTIMISATION (Grid Search)")
    print("="*60)
    records = []
    for ez in entry_z_vals:
        for w in window_vals:
            strat  = MeanReversionStrategy(df_prices, window=w, entry_z=ez)
            df_sig = strat.generate_signals()
            bt     = SimpleBacktester(df_sig)
            m      = bt.run()
            records.append({
                'entry_z': ez, 'window': w,
                'Sharpe' : round(m['Sharpe Ratio'],    3),
                'Return' : round(m['Total Return'],    4),
                'MDD'    : round(m['Maximum Drawdown'],4),
                'Trades' : int(m['Total Trades']),
                'AvgDur' : round(m['Average Trade Duration (min)'], 1),
                'WinRate': round(m['Win Rate'], 3),
            })
    results = (pd.DataFrame(records)
                 .sort_values('Sharpe', ascending=False)
                 .reset_index(drop=True))
    print(results.to_string(index=False))
    best = results.iloc[0]
    print(f"\n  Selected: entry_z={best['entry_z']}  window={best['window']}  Sharpe={best['Sharpe']}")
    return best['entry_z'], int(best['window'])


def plot_performance(bdf, asset_name, entry_z=1.5, exit_z=0.5, stop_z=3.5):
    price_col = bdf.columns[0]
    fig = plt.figure(figsize=(14, 18))
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    fig.suptitle(f'Z-Score Mean Reversion - {asset_name}', fontsize=14)

    ax1.plot(bdf.index, bdf[price_col], color='black', linewidth=0.3, alpha=0.6, label='Close')
    ax1.plot(bdf.index, bdf['rolling_mean'], color='blue', linewidth=0.6, alpha=0.7, label='Rolling Mean')
    buys  = bdf[(bdf['signal'] ==  1) & (bdf['signal'].shift(1) !=  1)]
    sells = bdf[(bdf['signal'] == -1) & (bdf['signal'].shift(1) != -1)]
    ax1.scatter(buys.index,  buys[price_col],  marker='^', color='green', s=20, zorder=5, label='Long')
    ax1.scatter(sells.index, sells[price_col], marker='v', color='red',   s=20, zorder=5, label='Short')
    ax1.set_title('Price Series and Trading Signals')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left', fontsize=8)

    ax2.plot(bdf.index, bdf['z_score'], color='purple', linewidth=0.4, alpha=0.7)
    for lvl, col, ls in [(entry_z,'red','--'),(-entry_z,'red','--'),
                         (exit_z,'gray',':'),(- exit_z,'gray',':'),
                         (stop_z,'black',':'),(-stop_z,'black',':')]:
        ax2.axhline(lvl, color=col, linestyle=ls, alpha=0.5, linewidth=0.7)
    ax2.set_title(f'Z-Score  (entry +/-{entry_z} | exit +/-{exit_z} | stop +/-{stop_z})')
    ax2.set_ylabel('Z-Score')

    ax3.plot(bdf.index, bdf['equity_curve'], color='blue', linewidth=0.8)
    ax3.axhline(1.0, color='black', linestyle='--', alpha=0.4)
    ax3.fill_between(bdf.index, bdf['equity_curve'], 1.0,
                     where=bdf['equity_curve'] >= 1.0, alpha=0.15, color='green')
    ax3.fill_between(bdf.index, bdf['equity_curve'], 1.0,
                     where=bdf['equity_curve'] <  1.0, alpha=0.15, color='red')
    ax3.set_title('Equity Curve  (TC=0.01% + slippage=0.005%)')
    ax3.set_ylabel('Cumulative Return')

    ax4.fill_between(bdf.index, bdf['drawdown'], 0, color='red', alpha=0.4)
    ax4.set_title('Drawdown Curve')
    ax4.set_ylabel('Drawdown')
    ax4.set_xlabel('Bar Index')

    plt.savefig('results/strategy_performance.png', dpi=100, bbox_inches='tight')
    plt.close()

    MPY = 252 * 375
    r   = bdf['net_return'].fillna(0)
    rs  = (r.rolling(3000).mean() / r.rolling(3000).std()) * np.sqrt(MPY)
    fig2, ax = plt.subplots(figsize=(14, 3))
    ax.plot(bdf.index, rs, color='purple', linewidth=0.6)
    ax.axhline(0,  color='black', linestyle='--', alpha=0.4)
    ax.axhline( 1, color='green', linestyle=':', alpha=0.6, label='Sharpe=1')
    ax.axhline(-1, color='red',   linestyle=':', alpha=0.6, label='Sharpe=-1')
    ax.set_title('Rolling Sharpe (3000-bar) - Strategy Regime Analysis')
    ax.set_ylabel('Sharpe')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('results/rolling_sharpe.png', dpi=100, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print("  QUANT DEVELOPER INTERN - SCREENING ASSIGNMENT")
print("=" * 60)

print("\n[1/5] Loading and cleaning data...")
prices     = load_and_clean('banknifty_candlestick_data.csv')
asset_name = prices.columns[0]

print("\n[2/5] Running EDA...")
run_eda(prices[asset_name])

best_entry_z, best_window = grid_search(prices)

print("\n" + "="*60)
print("STEP 3 - FINAL STRATEGY RUN")
print("="*60)
strategy   = MeanReversionStrategy(prices, window=best_window, entry_z=best_entry_z)
df_signals = strategy.generate_signals()
n_active   = int((df_signals['signal'] != 0).sum())
print(f"  Parameters  : entry_z={best_entry_z}  window={best_window}  exit_z=0.5  stop_z=3.5")
print(f"  Active bars : {n_active:,} / {len(df_signals):,} ({n_active/len(df_signals)*100:.1f}%)")

print("\n[4/5] Running backtest...")
backtester = SimpleBacktester(df_signals)
metrics    = backtester.run()
SimpleBacktester.print_metrics(metrics)

print("\n[5/5] Generating charts...")
plot_performance(backtester.df, asset_name, entry_z=best_entry_z)
print("  Saved -> results/strategy_performance.png")
print("  Saved -> results/rolling_sharpe.png")

total = time.time() - t_start
print(f"\n{'='*52}")
print(f"  TOTAL RUNTIME : {total:.1f} seconds")
print(f"  {'PASS - under 30s' if total < 30 else 'OVER 30s (Colab slower than local)'}")
print(f"{'='*52}")

# =============================================================================
# BONUS CHALLENGES  (clearly separated from main solution above)
# -----------------------------------------------------------------------------
# Bonus 1: Out-of-sample testing     (train 2015-2020 / test 2021-2024)
# Bonus 2: Multi-strategy portfolio  (3 variants, inverse-vol risk allocation)
#
# Remove this block to run the main solution only.
# =============================================================================

print("\n" + "=" * 60)
print("  BONUS CHALLENGES")
print("=" * 60)

from bonus_oos       import run_bonus_oos
from bonus_portfolio import run_bonus_portfolio

print("\n[BONUS 1/2] Out-of-sample testing...")
run_bonus_oos(prices)

print("\n[BONUS 2/2] Multi-strategy inverse-vol portfolio...")
run_bonus_portfolio(prices)

bonus_total = time.time() - t_start
print(f"\n{'='*52}")
print(f"  TOTAL RUNTIME (main + bonus) : {bonus_total:.1f} seconds")
print(f"{'='*52}")
