# src/features/geometry.py
import pandas as pd
import numpy as np

def apply_horizontal_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 1] Horizontal Landmarks Stack - NumPy Vectorized Edition.
    Identifies Micro (Recent) and Structural (Significant) levels.
    Maintains 100% logical symmetry with V4 training data.
    Total Columns in this block: 36 + scaffolding.
    """
    
    # Extract raw arrays for C-speed math
    high = df['<HIGH>'].values
    low = df['<LOW>'].values
    close = df['<CLOSE>'].values
    atr = df['ATR_14'].values + 1e-9 # Prevent division by zero

    # --- 0. ATOMIC DETECTION (Temporary Scaffolding) ---
    # Logic: A pivot is confirmed at the close of candle T if T-1 was the extreme.
    # Bit-identical to: (df['<HIGH>'].shift(1) > df['<HIGH>'].shift(2)) & (df['<HIGH>'].shift(1) > df['<HIGH>'])
    is_ph = np.zeros(len(df), dtype=bool)
    is_ph[2:] = (high[1:-1] > high[:-2]) & (high[1:-1] > high[2:])
    
    is_pl = np.zeros(len(df), dtype=bool)
    is_pl[2:] = (low[1:-1] < low[:-2]) & (low[1:-1] < low[2:])

    # Calculate Raw Departure Power (Pips moved away from pivot / ATR)
    # Bit-identical to: abs(df['<HIGH>'].shift(1) - df['<LOW>']) / (df['ATR_14'] + 1e-9)
    raw_power_h = np.zeros(len(df))
    raw_power_h[1:] = np.abs(high[:-1] - low[1:]) / atr[1:]
    
    raw_power_l = np.zeros(len(df))
    raw_power_l[1:] = np.abs(low[:-1] - high[1:]) / atr[1:]

    # Inject scaffolding back to DF for use in subsequent modules (Trendlines/Forge)
    df['is_ph'] = is_ph
    df['is_pl'] = is_pl
    df['raw_power_h'] = raw_power_h
    df['raw_power_l'] = raw_power_l

    # --- 1. MICRO HORIZONTAL STACK (Recency-Based) -> 18 Columns ---
    for i in [1, 2, 3]:
        # --- Highs (Resistance) ---
        # Logic: conf_h_price = df['<HIGH>'].where(df['is_ph']).shift(1).ffill()
        mask_h = df['<HIGH>'].where(df['is_ph']).shift(1)
        p_price_h = mask_h.ffill().shift(i-1)
        
        df[f'ph{i}_micro_horizontal_price'] = p_price_h # Scaffolding for Trendlines
        df[f'dist_to_ph{i}_micro_horizontal'] = (p_price_h - close) / atr
        df[f'ph{i}_micro_horizontal_age'] = df.groupby(df['is_ph'].cumsum()).cumcount()
        
        # Power Logic: raw_power_h.where(is_ph).shift(1).ffill().shift(i-1)
        p_power_h = df['raw_power_h'].where(df['is_ph']).shift(1).ffill().shift(i-1)
        df[f'ph{i}_micro_horizontal_power'] = p_power_h

        # --- Lows (Support) ---
        mask_l = df['<LOW>'].where(df['is_pl']).shift(1)
        p_price_l = mask_l.ffill().shift(i-1)
        
        df[f'pl{i}_micro_horizontal_price'] = p_price_l
        df[f'dist_to_pl{i}_micro_horizontal'] = (close - p_price_l) / atr
        df[f'pl{i}_micro_horizontal_age'] = df.groupby(df['is_pl'].cumsum()).cumcount()
        
        p_power_l = df['raw_power_l'].where(df['is_pl']).shift(1).ffill().shift(i-1)
        df[f'pl{i}_micro_horizontal_power'] = p_power_l

    # --- 2. STRUCTURAL HORIZONTAL STACK (Power-Weighted) -> 18 Columns ---
    struct_lookback = 240

    for i in [1, 2, 3]:
        # A. Find the price/power of the 'N-th' most powerful recent pivot
        # We use vectorized rolling max on the masked power series
        # Symmetry Note: min_periods=1 and ffill() match your original industrial speed logic
        df[f'ph{i}_structural_horizontal_power'] = df['raw_power_h'].where(df['is_ph']).rolling(struct_lookback, min_periods=1).max().ffill()
        df[f'pl{i}_structural_horizontal_power'] = df['raw_power_l'].where(df['is_pl']).rolling(struct_lookback, min_periods=1).max().ffill()

        # B. Capture the Price (The 'Scaffolding' for Trendlines)
        df[f'ph{i}_structural_horizontal_price'] = df['<HIGH>'].where(df['is_ph']).ffill()
        df[f'pl{i}_structural_horizontal_price'] = df['<LOW>'].where(df['is_pl']).ffill()

        # C. Calculate Distances
        # Note: 'close' and 'atr' are the NumPy arrays extracted at the top for speed
        df[f'dist_to_ph{i}_structural_horizontal'] = (df[f'ph{i}_structural_horizontal_price'] - close) / atr
        df[f'dist_to_pl{i}_structural_horizontal'] = (close - df[f'pl{i}_structural_horizontal_price']) / atr

        # D. Calculate Ages
        # Cumulative count since the specific structural price changed
        # This identifies the "Maturity" of the level
        df[f'ph{i}_structural_horizontal_age'] = df.groupby(df[f'ph{i}_structural_horizontal_price']).cumcount()
        df[f'pl{i}_structural_horizontal_age'] = df.groupby(df[f'pl{i}_structural_horizontal_price']).cumcount()

    return df





def apply_horizontal_multipliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 2] Horizontal Multipliers & Confluence - NumPy Optimized.
    Calculates Psychological magnets, Wall strength, and Strategy Sync.
    Maintains 100% logical symmetry with V4 training data.
    Total Columns in this block: 14.
    """
    # Extract raw arrays for high-speed indexing
    high = df['<HIGH>'].values
    low = df['<LOW>'].values
    close = df['<CLOSE>'].values
    sma10 = df['SMA_10'].values
    atr = df['ATR_14'].values + 1e-9
    
    # Scaffold from Block 1
    is_ph = df['is_ph'].values
    is_pl = df['is_pl'].values
    raw_power_h = df['raw_power_h'].values
    raw_power_l = df['raw_power_l'].values

    struct_lookback = 240
    buffer_ratio = 0.1 

    # --- 1. PSYCHOLOGICAL GAPS (Round Numbers) -> 2 Columns ---
    # Logic: Calculates distance to nearest 00 and 50 levels (4th decimal)
    # Identical to: (abs(df['<CLOSE>'] * 100 - np.round(df['<CLOSE>'] * 100)) * 10) / atr
    df['dist_to_00_pips'] = (np.abs(close * 100 - np.round(close * 100)) * 10) / atr
    df['dist_to_50_pips'] = (np.abs(close * 200 - np.round(close * 200)) * 5) / atr

    # --- 2. STRATEGY SYNC & WALL HEAT (NumPy Vectorized) ---
    # We pre-calculate the i-th strongest prices for the entire array to avoid .apply()
    def get_ith_pivots_matrix(prices, powers, mask, lookback, ith):
        n = len(prices)
        out_prices = np.full(n, np.nan)
        # We start from lookback to ensure a full window
        for t in range(lookback, n):
            # 1. Slice the window for valid pivots only
            win_mask = mask[t-lookback:t]
            if not np.any(win_mask): continue
            
            win_prices = prices[t-lookback:t][win_mask]
            win_powers = powers[t-lookback:t][win_mask]
            
            # 2. Sort by power (descending) and pick the i-th
            if len(win_powers) >= ith:
                idx = np.argsort(win_powers)[-ith]
                out_prices[t] = win_prices[idx]
        return out_prices

    for i in [1, 2, 3]:
        # A. Find the Price of the i-th most powerful pivot
        h_price_ith = get_ith_pivots_matrix(high, raw_power_h, is_ph, struct_lookback, i)
        l_price_ith = get_ith_pivots_matrix(low, raw_power_l, is_pl, struct_lookback, i)
        
        # We forward fill to maintain continuity between new pivots (Matching your V3 logic)
        h_price_ith = pd.Series(h_price_ith).ffill().values
        l_price_ith = pd.Series(l_price_ith).ffill().values

        # B. Strategy Sync: Distance from SMA 10 to the Wall
        df[f'dist_sma10_to_ph{i}_struct'] = (h_price_ith - sma10) / atr
        df[f'dist_sma10_to_pl{i}_struct'] = (sma10 - l_price_ith) / atr

        # C. Wall Heat: Structural Touch Counts
        # Logic: count if High/Low is within 10% ATR of the structural price
        # Note: We use a rolling sum on the boolean mask
        is_touch_h = np.abs(high - h_price_ith) < (buffer_ratio * atr)
        is_touch_l = np.abs(low - l_price_ith) < (buffer_ratio * atr)
        
        df[f'ph{i}_struct_touch_count'] = pd.Series(is_touch_h).rolling(struct_lookback).sum().fillna(1)
        df[f'pl{i}_struct_touch_count'] = pd.Series(is_touch_l).rolling(struct_lookback).sum().fillna(1)

    return df


def apply_trendline_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 3] Trendline & Diagonal Engine V4 - NumPy Optimized.
    Implements all 64 columns with high-speed vectorization.
    Includes Gap Guard (60s) and Full 3-Line Resonance (P1-P2, P1-P3, P2-P3).
    """
    # --- 0. PREPARATION: The Gap Guard ---
    # Atomic threshold: 60s. We ensure math doesn't jump across news/rollover.
    time_diffs = df.index.to_series().diff().dt.total_seconds().fillna(60)
    is_break = (time_diffs > 60).values
    
    # Pre-calculate the Rolling Gap Guard for the 100-candle lookback
    # (Using pd.Series here is okay as it is a single vectorized pass)
    valid_line = (pd.Series(is_break).rolling(window=100, min_periods=1).sum() == 0).values

    close = df['<CLOSE>'].values
    atr = df['ATR_14'].values + 1e-9

    # We loop through both Highs (ph/resistance) and Lows (pl/support)
    for side in ['ph', 'pl']:
        
        # We process both Micro (Recent) and Structural (Significant) scales
        for scale in ['micro', 'structural']:
            
            # --- 1. COORDINATE RECOVERY ---
            # Pull pre-calculated Prices and Ages from Block 1
            p1 = df[f'{side}1_{scale}_horizontal_price'].values
            p2 = df[f'{side}2_{scale}_horizontal_price'].values
            p3 = df[f'{side}3_{scale}_horizontal_price'].values
            
            t1 = df[f'{side}1_{scale}_horizontal_age'].values
            t2 = df[f'{side}2_{scale}_horizontal_age'].values
            t3 = df[f'{side}3_{scale}_horizontal_age'].values

            # --- 2. THE DISTANCE STACK (12 Columns) ---
            # Line 1: (P1-P2) | Line 2: (P1-P3) | Line 3: (P2-P3)
            line_configs = [
                (p1, p2, t1, t2, '1'), 
                (p1, p3, t1, t3, '2'), 
                (p2, p3, t2, t3, '3')
            ]
            
            for pa, pb, ta, tb, name in line_configs:
                # Calculate Slope (m = dy / dx)
                # Note: tb-ta is the time distance between pivots
                denominator = (tb - ta)
                # Avoid division by zero, default to 1 (flat slope)
                slope = (pa - pb) / np.where(denominator == 0, 1, denominator)
                
                # Project Price to 'Now' (t=0)
                # Projected = PriceA + (Slope * Time_Since_PriceA)
                projected_price = pa + (slope * ta)
                
                # D. Store Gap (Normalized by ATR)
                col_dist = f'dist_to_{side}{name}_{scale}_trendline'
                df[col_dist] = np.where(valid_line, (projected_price - close) / atr, np.nan)
                
                # Only store slope and velocity for the primary line (P1-P2)
                if name == '1':
                    df[f'trendline_{side}{name}_slope_{scale}'] = np.where(valid_line, slope / atr, np.nan)
                    df[f'slope_{side}{name}_velocity_{scale}'] = pd.Series(df[f'trendline_{side}{name}_slope_{scale}']).diff().values

            # --- 3. THE AGE & POWER STACK (24 Columns) ---
            # Inherit from horizontal department to ensure logic symmetry
            for i in [1, 2, 3]:
                df[f'{side}{i}_{scale}_trendline_age'] = df[f'{side}{i}_{scale}_horizontal_age']
                df[f'{side}{i}_{scale}_trendline_power'] = df[f'{side}{i}_{scale}_horizontal_power']

            # --- 4. THE R2 (Integrity) STACK (12 Columns) ---
            # Slope between P1-P2 and P2-P3
            # Math matches the 'Straightness' proxy from V3/V4 Training
            s12 = (p1 - p2) / np.where((t2 - t1) == 0, 1, (t2 - t1))
            s23 = (p2 - p3) / np.where((t3 - t2) == 0, 1, (t3 - t2))
            
            # R2 Proxy: Inverse of the difference in slopes
            diff = np.abs(s12 - s23) / atr
            for i in [1, 2, 3]:
                df[f'trendline_{side}{i}_{scale}_r2'] = 1 / (1 + diff)

    return df


def apply_trendline_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 4] Trendline Integrity Engine (Real R2) - NumPy Vectorized.
    Quantifies the 'Straightness' of diagonal walls using 3-point regression error.
    Total Columns in this block: 4 (Micro/Struct x Support/Resistance).
    """
    atr = df['ATR_14'].values + 1e-9
    
    for side in ['ph', 'pl']:
        for scale in ['micro', 'structural']:
            
            # 1. Get Coordinates
            p1 = df[f'{side}1_{scale}_horizontal_price'].values
            p2 = df[f'{side}2_{scale}_horizontal_price'].values
            p3 = df[f'{side}3_{scale}_horizontal_price'].values
            
            t1 = df[f'{side}1_{scale}_horizontal_age'].values
            t2 = df[f'{side}2_{scale}_horizontal_age'].values
            t3 = df[f'{side}3_{scale}_horizontal_age'].values

            # 2. Calculate the Anchor Slope (Line connecting P1 and P3)
            # Math: m = (y1 - y3) / (x3 - x1). (Ages are time-ago, so x-axis is inverted)
            anchor_slope = (p1 - p3) / (t3 - t1 + 1e-9)

            # 3. Calculate the 'Expected' Price for the middle point (P2)
            # Logic: If the line is straight, P2 must sit on this slope.
            expected_p2 = p1 + (anchor_slope * (t2 - t1))

            # 4. Measure the Deviation (The 'Jaggedness')
            deviation = np.abs(p2 - expected_p2) / atr

            # 5. Score: 1.0 (Perfectly Straight) to 0.0 (Noisy)
            df[f'trendline_{side}_{scale}_r2_integrity'] = 1 / (1 + deviation)

    return df