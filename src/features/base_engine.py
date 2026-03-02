# src/features/base_engine.py
import pandas as pd
import numpy as np

def apply_base_physics(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 1] The Raw Indicator Engine.
    Calculates stationary sensors on continuous raw data to ensure 
    mathematical integrity (No gap contamination).
    """
    
    # --- 1. THE UNIVERSAL RULER (ATR) ---
    # We calculate this first because it is the denominator for almost everything.
    high_low = df['<HIGH>'] - df['<LOW>']
    high_close = abs(df['<HIGH>'] - df['<CLOSE>'].shift(1))
    low_close = abs(df['<LOW>'] - df['<CLOSE>'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean().clip(lower=1e-7).astype('float32')

    # --- 2. THE MONEY LOCUS (Daily Reset VWAP) ---
    # Logic: Resets at the first candle of every new day.
    # We use a vectorized approach for speed.
    df['price_vol'] = df['<CLOSE>'] * df['<TICKVOL>']
    df['date'] = df.index.date
    
    # Calculate cumulative sums that reset when the date changes
    cum_pv = df.groupby('date')['price_vol'].cumsum()
    cum_vol = df.groupby('date')['<TICKVOL>'].cumsum()
    
    df['vwap'] = (cum_pv / (cum_vol + 1e-9)).astype('float32')
    df['dist_to_vwap_norm'] = ((df['<CLOSE>'] - df['vwap']) / df['ATR_14']).astype('float32')
    
    # Cleanup scaffolding
    df.drop(columns=['price_vol', 'date', 'vwap'], inplace=True)

    # --- 3. THE GRAVITY STACK (EMAs & Slopes) ---
    for p in [10, 20, 50, 100, 200]:
        ema_col = f'EMA_{p}_raw' # Scaffolding name
        df[ema_col] = df['<CLOSE>'].ewm(span=p, adjust=False).mean()
        
        # A. Velocity (Slope): Change over 3 mins normalized by ATR
        df[f'EMA_{p}_slope_norm'] = ((df[ema_col] - df[ema_col].shift(3)) / df['ATR_14']).astype('float32')
        
        # B. Distance (Gap): Distance from Price to EMA normalized by ATR
        df[f'dist_to_EMA_{p}_norm'] = ((df['<CLOSE>'] - df[ema_col]) / df['ATR_14']).astype('float32')
        
        # --- RECURSIVE PURGE ---
        # We kill the raw EMA price now that we have the Slope and Distance
        df.drop(columns=[ema_col], inplace=True)

    # --- 4. MOMENTUM SENSORS (RSI) ---
    for p in [7, 14, 21]:
        delta = df['<CLOSE>'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
        rs = gain / (loss + 1e-9)
        df[f'RSI_{p}'] = (100 - (100 / (1 + rs))).astype('float32')

    # --- 5. CYCLE PHASE SENSORS (Stochastic %K) ---
    for p in [7, 14, 21]:
        low_p = df['<LOW>'].rolling(window=p).min()
        high_p = df['<HIGH>'].rolling(window=p).max()
        # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low)
        df[f'stoch_{p}_cycle_phase'] = ((df['<CLOSE>'] - low_p) / (high_p - low_p + 1e-9)).astype('float32')

    # --- 6. BOUNDARY SENSORS (Bollinger Bands) ---
    bb_sma = df['<CLOSE>'].rolling(window=20).mean()
    bb_std = df['<CLOSE>'].rolling(window=20).std()
    upper_bb = bb_sma + (2 * bb_std)
    lower_bb = bb_sma - (2 * bb_std)
    
    # BB Width & Squeeze Rank (Percentile of last 24 hours)
    df['BB_width'] = ((upper_bb - lower_bb) / (bb_sma + 1e-9)).astype('float32')
    df['BB_squeeze_rank'] = df['BB_width'].rolling(window=1440).rank(pct=True).astype('float32')
    
    # Fluid Gaps to the Band Walls
    df['dist_to_bb_upper_norm'] = ((upper_bb - df['<CLOSE>']) / df['ATR_14']).astype('float32')
    df['dist_to_bb_lower_norm'] = ((df['<CLOSE>'] - lower_bb) / df['ATR_14']).astype('float32')

    # --- 7. STRATEGY SCANNER (The Gatekeeper) ---
    # Standardize the name to SMA_10 so all departments can find it
    df['SMA_10'] = df['<CLOSE>'].rolling(window=10).mean().astype('float32')
    
    # Standardize the name to dist_to_sma10 to match your V3 list
    df['dist_to_sma10'] = ((df['<CLOSE>'] - df['SMA_10']) / df['ATR_14']).astype('float32')
    # is_near_sma is used by the Refinery to pick rows, but we keep the dist_to column
    
    return df 