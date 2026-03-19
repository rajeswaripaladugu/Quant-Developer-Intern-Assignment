import numpy as np
import pandas as pd
import datetime


class MeanReversionStrategy:
    """
    Z-score mean reversion on a single asset.

    Entry long  : z < -entry_z  AND session open (09:30-15:15)
    Entry short : z > +entry_z  AND session open
    Exit        : |z| < exit_z  (price reverted to mean)
    Stop-loss   : |z| > stop_z  (price moved further against position)

    Signal uses a state machine (numpy loop) so exit only fires
    when a position is already held — prevents same-bar exits.

    min_periods=window on rolling stats prevents look-ahead bias
    during the warm-up period.
    """

    def __init__(self, prices, window=80, entry_z=1.5, exit_z=0.5, stop_z=3.5):
        self.df       = prices.copy()
        self.window   = window
        self.entry_z  = entry_z
        self.exit_z   = exit_z
        self.stop_z   = stop_z

    def generate_signals(self) -> pd.DataFrame:
        df    = self.df.copy()
        close = df.iloc[:, 0]

        # min_periods=window — no signal until full warm-up window is available
        roll_mean          = close.rolling(self.window, min_periods=self.window).mean()
        roll_std           = close.rolling(self.window, min_periods=self.window).std()
        df['rolling_mean'] = roll_mean
        df['z_score']      = (close - roll_mean) / roll_std
        df['spread']       = close

        # Session filter: skip noisy open (09:15-09:29) and close (15:16-15:30)
        t          = df.index.time
        session_ok = (t >= datetime.time(9, 30)) & (t <= datetime.time(15, 15))

        # State machine signal generation
        # Correctly carries position state forward — exit only fires when in a trade
        z    = df['z_score'].values
        sess = np.array(session_ok)

        entry  = np.where((z < -self.entry_z) & sess,  1,
                 np.where((z >  self.entry_z) & sess, -1, 0))
        exit_c = np.abs(z) < self.exit_z
        stop_c = np.abs(z) > self.stop_z

        signal = np.zeros(len(z), dtype=float)
        pos    = 0.0
        for i in range(len(z)):
            # Exit/stop checked FIRST while in a position.
            # Without this, stop_z > entry_z means a stop-loss bar always
            # satisfies the entry condition too, and entry would win — the
            # stop-loss would never fire.
            if pos != 0 and (exit_c[i] or stop_c[i]):
                pos = 0.0
            elif entry[i] != 0:
                pos = float(entry[i])
            signal[i] = pos

        df['signal'] = signal

        # Position sizing: 1% volatility targeting using return-based std.
        # roll_std is in price-point units (e.g. 100 BankNifty pts), so
        # dividing 0.01 by it gives a near-zero position (0.0001x leverage).
        # Return std (roll_std / close) is dimensionless, giving a meaningful
        # fraction-of-capital position that scales with actual return risk.
        roll_ret_std             = (roll_std / close).replace(0, np.nan)
        df['position_size']      = (0.01 / roll_ret_std.clip(lower=1e-8)).clip(upper=1.0).fillna(0)
        df['effective_position'] = df['signal'] * df['position_size']

        return df
