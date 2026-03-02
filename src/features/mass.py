# src/features/mass.py
import pandas as pd
import numpy as np

def apply_volume_stack(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 1] Institutional Mass Stack.
    Converts raw Tick Volume into 18 stationary energy sensors.
    Includes 'Gap Guard' to ensure volume flow is continuous (>60s check).
    """
    # 0. Preparation: The Gap Guard
    time_diffs = df.index.to_series().diff().dt.total_seconds().fillna(60)
    is_break = time_diffs > 60

    # --- 1. RELATIVE VOLUME (Pulse Sensors) -> 2 Columns ---
    for n in [5, 20]:
        valid = (is_break.rolling(window=n).sum() == 0)
        vol_ma = df['<TICKVOL>'].rolling(n).mean()
        df[f'rel_vol_{n}'] = (df['<TICKVOL>'] / (vol_ma + 1e-9)).where(valid)

    # --- 2. VOLUME VELOCITY (Interest Slopes) -> 2 Columns ---
    for n in [5, 10, 20]:
        valid = (is_break.rolling(window=n).sum() == 0)
        vol_ema = df['<TICKVOL>'].ewm(span=n, adjust=False).mean()
        df[f'vol_slope_{n}'] = ((vol_ema - vol_ema.shift(n)) / df['ATR_14']).where(valid)

    # --- 3. PRICE-VOLUME EFFICIENCY (The Effort Metric) -> 2 Columns ---
    df['vol_efficiency_1'] = abs(df['<CLOSE>'] - df['<OPEN>']) / (df['<TICKVOL>'] + 1e-9)
    valid_5 = (is_break.rolling(window=5).sum() == 0)
    price_move_5 = abs(df['<CLOSE>'] - df['<CLOSE>'].shift(5))
    df['vol_efficiency_5'] = (price_move_5 / (df['<TICKVOL>'].rolling(5).sum() + 1e-9)).where(valid_5)

    # --- 4. VOLUME RARITY (Z-Scores) -> 2 Columns ---
    for n in [20, 200]:
        valid = (is_break.rolling(window=n).sum() == 0)
        v_mean = df['<TICKVOL>'].rolling(n).mean()
        v_std = df['<TICKVOL>'].rolling(n).std()
        df[f'vol_z_{n}'] = ((df['<TICKVOL>'] - v_mean) / (v_std + 1e-9)).where(valid)

    # --- 5. BUY/SELL PRESSURE (Polarity Sensors) -> 4 Columns ---
    for n in [5, 15]:
        valid = (is_break.rolling(window=n).sum() == 0)
        buy_vol = df['<TICKVOL>'].where(df['<CLOSE>'] > df['<OPEN>'], 0)
        sell_vol = df['<TICKVOL>'].where(df['<CLOSE>'] < df['<OPEN>'], 0)
        df[f'buy_pressure_{n}'] = (buy_vol.rolling(n).sum() / (df['<TICKVOL>'].rolling(n).sum() + 1e-9)).where(valid)
        df[f'sell_pressure_{n}'] = (sell_vol.rolling(n).sum() / (df['<TICKVOL>'].rolling(n).sum() + 1e-9)).where(valid)

    # --- 6. MARKET FRICTION (Wall Detectors) -> 2 Columns ---
    for n in [5, 20]:
        valid = (is_break.rolling(window=n).sum() == 0)
        total_range_n = df['<HIGH>'].rolling(n).max() - df['<LOW>'].rolling(n).min()
        df[f'vol_friction_{n}'] = (total_range_n / (df['<TICKVOL>'].rolling(n).sum() + 1e-9)).where(valid)

    # --- 7. VOLUME ACCELERATION (Force Multiplier) -> 2 Columns ---
    for n in [5, 10, 20]:
        df[f'vol_acceleration_{n}'] = df[f'vol_slope_{n}'].diff().where(~is_break)

    # --- 8. SESSION NORMALIZATION (FIXED: Causal Logic) ---
    # Logic: Compares current volume to the historical average for this specific time-slot.
    # We use a Causal Expanding Mean to prevent Future Leakage.
    
    df['temp_hour'] = df.index.hour
    df['temp_day'] = df.index.dayofweek

    # Create a historical baseline that only looks at the past.
    # We shift(1) so the current candle's volume doesn't 'contaminate' its own baseline.
    for group_col, baseline_name in [('temp_hour', 'hist_hour_avg'), ('temp_day', 'hist_day_avg')]:
        grouped = df.groupby(group_col)['<TICKVOL>']
        
        # Calculate Expanding Mean: Sum of past / Count of past
        cum_sum = grouped.cumsum().shift(1)
        cum_count = grouped.cumcount().shift(1)
        
        # If it's the very first time we see this hour/day, baseline defaults to current volume
        df[baseline_name] = (cum_sum / (cum_count + 1e-9)).fillna(df['<TICKVOL>'])

    # The Final Stationary Ratios (1.0 = Normal, 2.0 = Double the usual volume for this time)
    df['session_vol_ratio'] = df['<TICKVOL>'] / (df['hist_hour_avg'] + 1e-9)
    df['weekday_vol_ratio'] = df['<TICKVOL>'] / (df['hist_day_avg'] + 1e-9)
    
    # Cleaning scaffolding
    df.drop(columns=['temp_hour', 'temp_day', 'hist_hour_avg', 'hist_day_avg'], inplace=True)

    return df