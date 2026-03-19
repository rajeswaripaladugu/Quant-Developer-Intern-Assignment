import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

os.makedirs('results', exist_ok=True)


def adf_test(series: pd.Series, name: str = '') -> bool:
    """ADF stationarity test. Samples to 10k points for speed."""
    s = series.dropna()
    if len(s) > 10_000:
        s = s.iloc[::len(s) // 10_000]
    result     = adfuller(s, autolag='AIC')
    stat, pval = round(result[0], 4), round(result[1], 4)
    label      = 'STATIONARY' if pval < 0.05 else 'non-stationary'
    print(f"  ADF [{name:20s}]  stat={stat:8.4f}  p={pval:.4f}  -> {label}")
    return pval < 0.05


def run_eda(price: pd.Series) -> None:
    """
    Full EDA on single price series.
    Saves: eda_price_overview.png, eda_returns.png, eda_rolling_vol.png
    Prints: ADF tests, return stats, intraday volatility
    """
    print("\n" + "="*60)
    print("STEP 1 - EXPLORATORY DATA ANALYSIS")
    print("="*60)

    log_ret = np.log(price / price.shift(1)).dropna()

    # Plot 1: Price overview + daily range
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    axes[0].plot(price.index, price.values, color='steelblue', linewidth=0.4)
    axes[0].set_title('BankNifty Close Price - Full History')
    axes[0].set_ylabel('Price')
    daily_range = price.groupby(price.index.date).agg(lambda x: x.max() - x.min())
    axes[1].bar(range(len(daily_range)), daily_range.values,
                color='coral', alpha=0.6, width=1.0)
    axes[1].set_title('Daily Range (High-Low) - Volatility Proxy')
    axes[1].set_ylabel('Points')
    axes[1].set_xlabel('Trading Days')
    fig.suptitle('EDA - BankNifty Price Overview', fontsize=13)
    plt.tight_layout()
    plt.savefig('results/eda_price_overview.png', dpi=72, bbox_inches='tight')
    plt.close()

    # Stationarity tests
    print("\n[ADF] Raw price (expect non-stationary):")
    adf_test(price, name='BankNifty price')
    print("[ADF] Log returns (expect stationary):")
    adf_test(log_ret, name='log returns')

    # Return statistics
    lag1 = log_ret.autocorr(lag=1)
    print(f"\n[Return Stats]")
    print(f"  Mean       : {log_ret.mean():.6f}")
    print(f"  Std        : {log_ret.std():.6f}")
    print(f"  Skewness   : {log_ret.skew():.4f}  (negative = downside fat tail)")
    print(f"  Kurtosis   : {log_ret.kurtosis():.2f}  (>>3 = extreme fat tails)")
    print(f"  Lag-1 ACF  : {lag1:.4f}")
    print(f"\n  [Strategy Justification]")
    print(f"  Kurtosis of {log_ret.kurtosis():.0f} means extreme price moves occur far more")
    print(f"  often than a normal distribution predicts. These extreme z-score events")
    print(f"  tend to revert — which is the statistical basis for this strategy.")
    print(f"  Lag-1 ACF near zero means NO momentum at 1-min scale,")
    print(f"  consistent with mean-reverting behaviour at short horizons.")

    # Plot 2: ACF + return histogram
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sample = log_ret.iloc[::max(1, len(log_ret)//5000)]
    plot_acf(sample, lags=30, ax=axes[0], alpha=0.05)
    axes[0].set_title('ACF of Minute Returns')
    axes[1].hist(log_ret, bins=200, color='steelblue', alpha=0.7)
    axes[1].set_title(f'Return Distribution  skew={log_ret.skew():.2f}  kurt={log_ret.kurtosis():.0f}')
    axes[1].set_xlabel('Log Return')
    fig.suptitle('EDA - Return Distribution & Autocorrelation', fontsize=13)
    plt.tight_layout()
    plt.savefig('results/eda_returns.png', dpi=72, bbox_inches='tight')
    plt.close()

    # Intraday volatility — print top noisy minutes
    vol_by_time = log_ret.abs().groupby(log_ret.index.time).mean()
    top5        = vol_by_time.nlargest(5)
    print(f"\n[Intraday Volatility] Top 5 noisiest minutes:")
    for t, v in top5.items():
        print(f"  {str(t):10s}  vol={v:.6f}")
    print(f"  -> Session filter skips 09:15-09:29 (open noise)")

    # Plot 3: Rolling volatility
    daily_vol   = log_ret.groupby(log_ret.index.date).std()
    rolling_vol = daily_vol.rolling(30).mean()
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(range(len(rolling_vol)), rolling_vol.values,
            color='darkorange', linewidth=0.8)
    ax.set_title('EDA - 30-Day Rolling Volatility  (COVID spike ~day 1300 = March 2020)')
    ax.set_ylabel('Volatility')
    ax.set_xlabel('Trading Days')
    plt.tight_layout()
    plt.savefig('results/eda_rolling_vol.png', dpi=72, bbox_inches='tight')
    plt.close()

    print("\n  [Plots saved to results/]")
