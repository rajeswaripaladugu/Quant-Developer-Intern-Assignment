import numpy as np
import pandas as pd


class SimpleBacktester:
    """
    Vectorized backtester. No loops over rows.

    transaction_cost = 0.01% per trade  (assignment specification)
    slippage         = 0.005% per trade (half-spread conservative estimate)

    Uses price log-returns — always well-defined and bounded.
    signal.shift(1) ensures no look-ahead bias (act on previous bar only).
    trade_id uses signal column (not effective_position) so duration
    is measured correctly even when position_size varies bar-to-bar.
    """

    TRANSACTION_COST = 0.0001   # 0.01%
    SLIPPAGE         = 0.00005  # 0.005%

    def __init__(self, df, transaction_cost=None, slippage=None):
        self.df   = df.copy()
        self.tc   = transaction_cost if transaction_cost is not None else self.TRANSACTION_COST
        self.slip = slippage if slippage is not None else self.SLIPPAGE

    def run(self) -> dict:
        df        = self.df
        price_col = df.columns[0]

        df['price_return'] = np.log(df[price_col] / df[price_col].shift(1))
        df['strat_return'] = df['effective_position'].shift(1) * df['price_return']
        # Use signal (not effective_position) for trade detection.
        # effective_position = signal * position_size, and position_size
        # changes every bar (rolling std evolves), so effective_position.diff()
        # is non-zero on every in-position bar — costs would be charged
        # bar-by-bar instead of only at entry/exit.
        df['trade_flag']   = df['signal'].diff().fillna(0).abs()
        df['cost']         = df['trade_flag'] * (self.tc + self.slip)
        df['net_return']   = df['strat_return'] - df['cost']
        df['equity_curve'] = (1 + df['net_return'].fillna(0)).cumprod()
        df['drawdown']     = df['equity_curve'] / df['equity_curve'].cummax() - 1

        self.df = df
        return self._calculate_metrics()

    def _calculate_metrics(self) -> dict:
        df  = self.df
        MPY = 252 * 375
        n   = len(df)
        r   = df['net_return'].fillna(0)
        m   = {}

        # 6 required metrics
        m['Total Return']      = df['equity_curve'].iloc[-1] - 1
        m['Annualized Return'] = (1 + m['Total Return']) ** (MPY / n) - 1
        mean_r, std_r          = r.mean(), r.std()
        m['Sharpe Ratio']      = (mean_r / std_r) * np.sqrt(MPY) if std_r > 0 else 0
        m['Maximum Drawdown']  = df['drawdown'].min()

        # trade_id on signal — position_size varies every bar so
        # effective_position would create a new trade_id each bar
        df['trade_id'] = (df['signal'] != df['signal'].shift(1)).cumsum()
        active         = df[df['effective_position'] != 0]

        if not active.empty:
            trade_ret = active.groupby('trade_id')['net_return'].sum()
            trade_dur = active.groupby('trade_id').size()
            m['Win Rate']                     = (trade_ret > 0).mean()
            m['Average Trade Duration (min)'] = trade_dur.mean()
            m['Total Trades']                 = int(df['trade_flag'].sum())
        else:
            m['Win Rate'] = m['Average Trade Duration (min)'] = m['Total Trades'] = 0

        # Additional risk metrics
        down_r             = r[r < 0]
        m['Sortino Ratio'] = (mean_r / down_r.std()) * np.sqrt(MPY) if len(down_r) > 0 else 0
        m['Calmar Ratio']  = (m['Annualized Return'] / abs(m['Maximum Drawdown'])
                              if m['Maximum Drawdown'] != 0 else 0)
        gp = r[r > 0].sum()
        gl = abs(r[r < 0].sum())
        m['Profit Factor'] = gp / gl if gl > 0 else 0

        return m

    @staticmethod
    def print_metrics(metrics: dict) -> None:
        pct_keys = {'Total Return', 'Annualized Return', 'Maximum Drawdown', 'Win Rate'}
        print("\n" + "="*52)
        print("  STRATEGY PERFORMANCE METRICS")
        print("="*52)
        for k, v in metrics.items():
            if isinstance(v, float):
                val = f"{v:.2%}" if k in pct_keys else f"{v:.4f}"
            else:
                val = str(v)
            print(f"  {k:<38} {val}")
        print("="*52)
