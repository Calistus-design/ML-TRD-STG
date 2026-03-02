# src/features/temporal.py
import pandas as pd
import numpy as np

def apply_time_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Step 1] Upgrades Time Resolution from Hourly to Minute-by-Minute.
    Uses Circular Vectors to ensure continuity across day/week boundaries.
    """
    # 1. Extract raw components
    # dayofweek: Monday=0, Friday=4
    day = df.index.dayofweek
    hour = df.index.hour
    minute = df.index.minute
    
    # 2. Minute of the Day (1,440 unique states)
    # This allows the model to see session maturity (e.g., 10 mins into London)
    min_of_day = (hour * 60) + minute
    df['minute_sin'] = np.sin(2 * np.pi * min_of_day / 1440)
    df['minute_cos'] = np.cos(2 * np.pi * min_of_day / 1440)
    
    # 3. Minute of the Week (7,200 unique states)
    # Captures weekly cycles (Monday gaps vs Friday closes)
    min_of_week = (day * 1440) + min_of_day
    df['week_min_sin'] = np.sin(2 * np.pi * min_of_week / 7200)
    df['week_min_cos'] = np.cos(2 * np.pi * min_of_week / 7200)
    
    # 4. Day of the Week (5 unique states)
    # Keeps the 'Broad Day' context
    df['day_sin'] = np.sin(2 * np.pi * day / 5)
    df['day_cos'] = np.cos(2 * np.pi * day / 5)
    
    return df



def apply_rolling_lenses(df: pd.DataFrame) -> pd.DataFrame:
    """
    The 'Jackpot Forge'. Implements the full 50-column Rolling Window stack.
    Includes Anatomy, Context, H1 Regime, Timing, and Accelerator Transitions.
    """
    # 0. Preparation: Pre-calculate the Gap Guard
    # We use this to ensure no feature looks across weekends or news blackouts.
    time_diffs = (df.index.to_series().diff().dt.total_seconds().fillna(60))
    is_break = time_diffs > 60

    # 1. Anatomy Lenses (M2, 3, 5, 10, 15) -> 20 Columns
    # Captures the 'Shape' of the move between 1-minute bars.
    for n in [2, 3, 5, 10, 15, 30, 60]:
        r_high = df['<HIGH>'].rolling(n).max()
        r_low = df['<LOW>'].rolling(n).min()
        r_open = df['<OPEN>'].shift(n-1)
        r_close = df['<CLOSE>']
        
        # Valid if no breaks occurred in the last N rows
        valid_window = (is_break.rolling(window=n).sum() == 0)
        
        total_range = (r_high - r_low).replace(0, 0.001)
        body_size = abs(r_open - r_close)
        
        df[f'M{n}_Body_Ratio'] = (body_size / total_range).where(valid_window)
        df[f'M{n}_Top_Wick_Ratio'] = ((r_high - np.maximum(r_open, r_close)) / total_range).where(valid_window)
        df[f'M{n}_Bottom_Wick_Ratio'] = ((np.minimum(r_open, r_close) - r_low) / total_range).where(valid_window)
        df[f'M{n}_Relative_Size'] = (total_range / df['ATR_14']).where(valid_window)

    # 2. M15 and M30 Context Lenses -> 8 Columns
    for n in [15, 30]:
        valid_window = (is_break.rolling(window=n).sum() == 0)
        ema = df['<CLOSE>'].ewm(span=n, adjust=False).mean()
        
        df[f'EMA_{n}_slope_norm'] = ((ema - ema.shift(n)) / df['ATR_14']).where(valid_window)
        df[f'dist_to_EMA_{n}_norm'] = ((df['<CLOSE>'] - ema) / df['ATR_14']).where(valid_window)
        
        # RSI Proxy for the scale
        delta = df['<CLOSE>'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=n*14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=n*14).mean()
        rs = gain / loss.replace(0, 0.001)
        df[f'RSI_{n}_proxy'] = (100 - (100 / (1 + rs))).where(valid_window)
        
        # Energy Ratio (Efficiency of the move at this scale)
        net_move = abs(df['<CLOSE>'] - df['<CLOSE>'].shift(n))
        path = abs(df['<CLOSE>'] - df['<CLOSE>'].shift(1)).rolling(n).sum()
        df[f'M{n}_energy_ratio'] = (net_move / path.replace(0, 0.001)).where(valid_window)

    # 3. H1 God-View (Regime) -> 6 Columns
    valid_h1 = (is_break.rolling(window = 60).sum() == 0)
    df['H1_Volatility_Rank'] = df['ATR_14'].rolling(1440).rank(pct=True) # Rank vs 24h
    df['H1_Body_Avg'] = (df['body_size'].rolling(60).mean() / df['ATR_14']).where(valid_h1)
    
    net_h1 = abs(df['<CLOSE>'] - df['<CLOSE>'].shift(60))
    path_h1 = abs(df['<CLOSE>'] - df['<CLOSE>'].shift(1)).rolling(60).sum()
    df['H1_Efficiency_Ratio'] = (net_h1 / path_h1.replace(0, 0.001)).where(valid_h1)
    
    df['H1_Trend_Direction'] = np.sign(df['<CLOSE>'] - df['<CLOSE>'].shift(60)).where(valid_h1)
    df['H1_Range_Cleanliness'] = (df['<HIGH>'].rolling(60).max() - df['<LOW>'].rolling(60).min()) / df['ATR_14']
    
    # H1 Gap to VWAP (Requires VWAP calculation - implemented in Locus step)
    df['H1_Gap_to_VWAP'] = df['dist_to_vwap_norm'].rolling(60).mean().where(valid_h1)

    # 4. Timing Multipliers -> 4 Columns
    df['M1_Velocity_vs_M5'] = abs(df['<CLOSE>'] - df['<OPEN>']) / (df['body_size'].rolling(5).mean() + 1e-9)
    df['M1_Velocity_vs_M15'] = abs(df['<CLOSE>'] - df['<OPEN>']) / (df['body_size'].rolling(15).mean() + 1e-9)
    df['M1_Volatility_Acceleration'] = df['ATR_14'].diff() / df['ATR_14'].shift(1)
    df['M1_Spread_Tax_Ratio'] = df['<SPREAD>'] / (df['ATR_14'] * 10) # Pip-relative tax

    # 5. Cross-Scale Energy Divergence -> 6 Columns
    # Logic: Comparing the short-term pulse to the structural tide
    df['RSI_M1_vs_M5_delta'] = df['RSI_14'] - df['RSI_14'].rolling(5).mean()
    df['RSI_M1_vs_M15_delta'] = df['RSI_14'] - df['RSI_14'].rolling(15).mean()
    df['Slope_M1_vs_M5_delta'] = df['EMA_10_slope_norm'] - df['EMA_10_slope_norm'].rolling(5).mean()
    df['Slope_M1_vs_M15_delta'] = df['EMA_10_slope_norm'] - df['EMA_10_slope_norm'].rolling(15).mean()
    df['Vol_M1_vs_M15_ratio'] = df['<TICKVOL>'] / (df['<TICKVOL>'].rolling(15).mean() + 1)
    df['Vol_M1_vs_M30_ratio'] = df['<TICKVOL>'] / (df['<TICKVOL>'].rolling(30).mean() + 1)

    # 6. Accelerator Transitions -> 6 Columns
    # Logic: Detecting the hand-off between time-cycles
    df['Accel_M2_to_M5'] = df['EMA_10_slope_norm'].rolling(2).mean() - df['EMA_10_slope_norm'].rolling(5).mean()
    df['Accel_M5_to_M15'] = df['EMA_10_slope_norm'].rolling(5).mean() - df['EMA_10_slope_norm'].rolling(15).mean()
    df['Accel_M15_to_M60'] = df['EMA_10_slope_norm'].rolling(15).mean() - df['EMA_10_slope_norm'].rolling(60).mean()
    
    # Squeeze Flow (Multi-scale volatility comparison)
    df['Squeeze_Flow_M1_to_M10'] = df['BB_width'] / (df['BB_width'].rolling(10).mean() + 1e-9)
    df['Squeeze_Flow_M10_to_M30'] = df['BB_width'].rolling(10).mean() / (df['BB_width'].rolling(30).mean() + 1e-9)
    df['Squeeze_Flow_M30_to_M200'] = df['BB_width'].rolling(30).mean() / (df['BB_width'].rolling(200).mean() + 1e-9)

    return df