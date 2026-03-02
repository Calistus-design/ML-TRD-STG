# src/features/anatomy.py
import pandas as pd
import numpy as np

def apply_m1_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 1] M1 Atomic Anatomy.
    Digitizes the shape and force of the current 1-minute candle.
    Total Columns in this block: 7.
    """
    # 1. RAW MEASUREMENTS (Scaffolding)
    # We use these to build the ratios, then purge them in the Oracle block
    df['body_size'] = abs(df['<OPEN>'] - df['<CLOSE>'])
    df['total_range'] = (df['<HIGH>'] - df['<LOW>']).replace(0, 1e-9)

    # 2. STATIONARY RATIOS (The 'Fingerprint') -> 4 Columns
    df['body_ratio'] = df['body_size'] / df['total_range']
    df['top_wick_ratio'] = (df['<HIGH>'] - df[['<OPEN>', '<CLOSE>']].max(axis=1)) / df['total_range']
    df['bottom_wick_ratio'] = (df[['<OPEN>', '<CLOSE>']].min(axis=1) - df['<LOW>']) / df['total_range']
    
    # Normalized Magnitude (Is this candle 2x the normal size?)
    df['relative_size'] = df['total_range'] / (df['ATR_14'] + 1e-9)

    # 3. DIRECTIONAL BIAS -> 1 Column
    # We use -1, 0, 1 instead of Boolean to keep the math fluid
    df['candle_direction'] = np.sign(df['<CLOSE>'] - df['<OPEN>']).astype('int8')

    return df

def apply_m1_memory_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 2] The HD Video Feed (Lags 1-10).
    Stores the exact physical shape of the last 10 minutes.
    Includes 'Gap Guard' to ensure history is continuous.
    Total Columns in this block: 40.
    """
    # Preparation: Identify gaps in the time-chain
    time_diffs = df.index.to_series().diff().dt.total_seconds().fillna(60)
    is_break = time_diffs > 60

    # We capture 4 dimensions for every lag
    for lag in range(1, 11):
        # The window is valid only if NO breaks happened between 'Lag' and 'Now'
        valid_lag = (is_break.rolling(window=lag+1).sum() == 0)
        
        df[f'body_ratio_lag_{lag}'] = df['body_ratio'].shift(lag).where(valid_lag)
        df[f'wick_low_lag_{lag}'] = df['bottom_wick_ratio'].shift(lag).where(valid_lag)
        df[f'wick_high_lag_{lag}'] = df['top_wick_ratio'].shift(lag).where(valid_lag)
        df[f'size_atr_lag_{lag}'] = df['relative_size'].shift(lag).where(valid_lag)

    return df

def apply_momentum_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 3] Structural Conviction Metrics.
    Measures the 'Health' of the trend and the 'Force' of the pullback.
    Total Columns in this block: 3.
    """
    time_diffs = df.index.to_series().diff().dt.total_seconds().fillna(60)
    is_break = time_diffs > 60

    # 1. Pullback Intensity (Body Size of Pullback vs Trend)
    # Valid only if the last 6 candles are continuous
    valid_6 = (is_break.rolling(window=6).sum() == 0)
    avg_body_3 = df['body_size'].rolling(3).mean()
    avg_body_prev_3 = df['body_size'].shift(3).rolling(3).mean()
    df['pullback_intensity'] = (avg_body_3 / (avg_body_prev_3 + 1e-9)).where(valid_6)
    
    # 2. Wick Pressure Ratio (Choppiness Filter)
    valid_3 = (is_break.rolling(window=3).sum() == 0)
    sum_wicks_3 = (df['top_wick_ratio'] + df['bottom_wick_ratio']).rolling(3).sum()
    sum_bodies_3 = df['body_ratio'].rolling(3).sum()
    df['wick_pressure_ratio'] = (sum_wicks_3 / (sum_bodies_3 + 1e-9)).where(valid_3)
    
    # 3. Momentum Cleanliness (Trend Efficiency)
    valid_5 = (is_break.rolling(window=5).sum() == 0)
    net_move_5 = abs(df['<CLOSE>'] - df['<CLOSE>'].shift(5))
    total_path_5 = df['total_range'].rolling(5).sum()
    df['momentum_cleanliness'] = (net_move_5 / (total_path_5 + 1e-9)).where(valid_5)

    return df