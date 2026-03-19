import pandas as pd
import numpy as np
import datetime


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load CSV, clean outliers, enforce strict market hours, align timestamps.
    All operations vectorized — no loops over rows.
    """
    df = pd.read_csv(filepath)

    # Parse datetime — Indian DD-MM-YYYY format
    df['datetime'] = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['Time'].astype(str), dayfirst=True
    )
    df = df.sort_values('datetime').drop_duplicates(subset=['datetime', 'Instrument'])

    # Outlier removal: vectorized z-score per asset, drop |z| > 5
    grp = df.groupby('Instrument')['Close']
    z   = (df['Close'] - grp.transform('mean')) / grp.transform('std')
    df  = df[z.abs() < 5].copy()

    # Pivot to wide format: index=datetime, columns=assets
    wide = df.pivot_table(index='datetime', columns='Instrument',
                          values='Close', aggfunc='last')

    # Strict market hours: 09:15-15:30 IST only
    # BankNifty cash closes at 15:30. After-hours/futures data is noise.
    t    = wide.index.time
    wide = wide[(t >= datetime.time(9, 15)) & (t <= datetime.time(15, 30))]

    # Drop incomplete sessions (< 200 bars = partial/corrupt day)
    bars_per_day = wide.groupby(wide.index.date).size()
    valid_days   = bars_per_day[bars_per_day >= 200].index
    wide         = wide[pd.Series(wide.index.date, index=wide.index).isin(valid_days)]

    # Forward-fill short gaps (<=5 bars), drop remaining NaN rows
    wide = wide.ffill(limit=5).dropna()

    print(f"[DataLoader] {len(wide):,} rows | Assets: {list(wide.columns)}")
    print(f"[DataLoader] Range: {wide.index[0]}  to  {wide.index[-1]}")
    return wide
