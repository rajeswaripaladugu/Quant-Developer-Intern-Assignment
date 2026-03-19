# BankNifty Z-Score Mean Reversion Strategy

## Strategy Overview

A z-score mean reversion strategy on BankNifty minute-level data (2015–2024).

Buys when price is unusually low relative to its recent rolling mean, and sells short when unusually high, expecting reversion to the mean.

---

## File Structure

```
project/
├── data_loader.py       # load CSV, clean outliers, enforce market hours
├── analysis.py          # EDA: stationarity, ACF, return stats, volatility plots
├── strategy.py          # z-score signal generation, session filter, stop-loss
├── backtester.py        # vectorized P&L engine, all required metrics
├── main.py              # full pipeline + bonus challenges
├── bonus_oos.py         # BONUS 1: out-of-sample testing
├── bonus_portfolio.py   # BONUS 2: portfolio + risk allocation
├── README.md
└── results/
    ├── strategy_performance.png
    ├── rolling_sharpe.png
    ├── eda_price_overview.png
    ├── eda_returns.png
    ├── eda_rolling_vol.png
    ├── bonus_oos.png
    └── bonus_portfolio.png
```

---

## Approach

### 1. Data Processing

* Loads minute-level OHLC CSV (DD-MM-YYYY)
* Removes outliers using z-score (|z| > 5)
* Enforces market hours: 09:15–15:30 IST
* Drops incomplete sessions (< 200 bars/day)
* Forward-fills gaps (≤ 5 bars)

---

### 2. EDA and Statistical Justification

| Statistic             | Value  | Implication       |
| --------------------- | ------ | ----------------- |
| ADF p-value (price)   | 0.91   | Non-stationary    |
| ADF p-value (returns) | 0.00   | Stationary        |
| Kurtosis              | 1441   | Extreme fat tails |
| Skewness              | −6.05  | Heavy downside    |
| Lag-1 ACF             | −0.002 | No momentum       |

These extreme deviations justify mean reversion.

---

### 3. Strategy Logic

```
rolling_mean = Close.rolling(window).mean()
rolling_std  = Close.rolling(window).std()
z_score      = (Close - rolling_mean) / rolling_std

Entry long  : z < -entry_z
Entry short : z > +entry_z
Exit        : |z| < 0.5
Stop-loss   : |z| > 3.5
```

**Session filter:** removes first & last 15 minutes (high noise)

**Important:** Exit/stop checked before entry to ensure stop-loss works.

---

### 4. Risk & Position Sizing

* Vol targeting:

```
position = 0.01 / (rolling_std / close)
```

* Cap at 1.0
* Stop-loss enforced

---

### 5. Backtesting

| Parameter        | Value           |
| ---------------- | --------------- |
| Transaction cost | 0.01%           |
| Slippage         | 0.005%          |
| Execution lag    | signal.shift(1) |
| Annualisation    | 94,500 min/year |

---

### 6. Parameter Optimisation

Grid search:

* entry_z ∈ {1.0, 1.5}
* window ∈ {50, 80}

Selected via **Sharpe Ratio**

---

## Bonus

### Out-of-Sample

* Train: 2015–2020
* Test: 2021–2024

---

### Portfolio Strategy

Three strategies:

| Strategy | Window | Entry Z |
| -------- | ------ | ------- |
| A        | 50     | 1.0     |
| B        | 80     | 1.5     |
| C        | 120    | 2.0     |

Uses inverse volatility weighting.

---

## Limitations

* Weak edge at 1-min scale
* Single asset (no pairs trading)
* Static parameters
* Sensitive to market regimes

---

## Improvements

* Momentum strategy
* Higher timeframe
* Adaptive thresholds
* Regime detection
* Walk-forward optimisation

---

## AI Assistance

Tools used: Claude 
Used for: syntax, debugging signal state machine logic, identifying and fixing bugs in the backtesting engine, README formatting.
All design decisions made independently: strategy choice, signal logic, risk controls, backtesting architecture, bonus challenge design, limitations assessment.
