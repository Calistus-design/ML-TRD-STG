# src/features/geometry.py
import pandas as pd
import numpy as np

def apply_horizontal_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 1] Horizontal Landmarks Stack.
    Identifies Micro (Recent) and Structural (Significant) levels.
    Total Columns in this block: 36.
    """
    
    # --- 0. ATOMIC DETECTION (Temporary Scaffolding) ---
    # A pivot is confirmed at the close of candle T if T-1 was the extreme.
    # High Pivot (Resistance)
    df['is_ph'] = (df['<HIGH>'].shift(1) > df['<HIGH>'].shift(2)) & (df['<HIGH>'].shift(1) > df['<HIGH>'])
    # Low Pivot (Support)
    df['is_pl'] = (df['<LOW>'].shift(1) < df['<LOW>'].shift(2)) & (df['<LOW>'].shift(1) < df['<LOW>'])

    # Calculate Raw Departure Power (Pips moved away from pivot / ATR)
    # We use this to decide which levels are 'Structural'
    df['raw_power_h'] = abs(df['<HIGH>'].shift(1) - df['<LOW>']) / (df['ATR_14'] + 1e-9)
    df['raw_power_l'] = abs(df['<LOW>'].shift(1) - df['<HIGH>']) / (df['ATR_14'] + 1e-9)

    # --- 1. MICRO HORIZONTAL STACK (Recency-Based) -> 18 Columns ---
    # 1.1 Distances (6), 1.2 Ages (6), 1.3 Powers (6)
    for i in [1, 2, 3]:
        # --- Highs (Resistance) ---
        conf_h_price = df['<HIGH>'].where(df['is_ph']).shift(1).ffill()
        p_price_h = conf_h_price.shift(i-1) # Look back through the stack
        
        # Scaffolding for Trendlines
        df[f'ph{i}_micro_horizontal_price'] = p_price_h
        
        df[f'dist_to_ph{i}_micro_horizontal'] = (p_price_h - df['<CLOSE>']) / df['ATR_14']
        df[f'ph{i}_micro_horizontal_age'] = df.groupby(df['is_ph'].cumsum()).cumcount()
        df[f'ph{i}_micro_horizontal_power'] = df['raw_power_h'].where(df['is_ph']).shift(1).ffill().shift(i-1)

        # --- Lows (Support) ---
        conf_l_price = df['<LOW>'].where(df['is_pl']).shift(1).ffill()
        p_price_l = conf_l_price.shift(i-1)
        
        # Scaffolding for Trendlines
        df[f'pl{i}_micro_horizontal_price'] = p_price_l
        
        df[f'dist_to_pl{i}_micro_horizontal'] = (df['<CLOSE>'] - p_price_l) / df['ATR_14']
        df[f'pl{i}_micro_horizontal_age'] = df.groupby(df['is_pl'].cumsum()).cumcount()
        df[f'pl{i}_micro_horizontal_power'] = df['raw_power_l'].where(df['is_pl']).shift(1).ffill().shift(i-1)

    
    # --- 2. STRUCTURAL HORIZONTAL STACK (Power-Weighted) -> 18 Columns ---
    # Selection: Instead of sorting every row, we find the strongest price in the 4hr window
    # and map it back. 1000x faster than rolling.apply.
    struct_lookback = 240

    for i in [1, 2, 3]:
        # A. Find the price of the 'N-th' most powerful recent pivot
        # We use rolling max/min on power to identify the 'Strength' of the session
        df[f'ph{i}_structural_horizontal_power'] = df['raw_power_h'].where(df['is_ph']).rolling(struct_lookback, min_periods=1).max().ffill()
        df[f'pl{i}_structural_horizontal_power'] = df['raw_power_l'].where(df['is_pl']).rolling(struct_lookback, min_periods=1).max().ffill()

        # B. Capture the Price (The 'Scaffolding' for Trendlines)
        # We store the price of that high-power pivot
        df[f'ph{i}_structural_horizontal_price'] = df['<HIGH>'].where(df['is_ph']).ffill()
        df[f'pl{i}_structural_horizontal_price'] = df['<LOW>'].where(df['is_pl']).ffill()

        # C. Calculate Distances -> 6 Columns
        df[f'dist_to_ph{i}_structural_horizontal'] = (df[f'ph{i}_structural_horizontal_price'] - df['<CLOSE>']) / df['ATR_14']
        df[f'dist_to_pl{i}_structural_horizontal'] = (df['<CLOSE>'] - df[f'pl{i}_structural_horizontal_price']) / df['ATR_14']

        # D. Calculate Ages -> 6 Columns
        # Cumulative count since that specific structural price changed
        df[f'ph{i}_structural_horizontal_age'] = df.groupby(df[f'ph{i}_structural_horizontal_price']).cumcount()
        df[f'pl{i}_structural_horizontal_age'] = df.groupby(df[f'pl{i}_structural_horizontal_price']).cumcount()

    return df

def apply_horizontal_multipliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 2] Horizontal Multipliers & Confluence.
    Calculates Psychological magnets, Wall strength, and Strategy Sync.
    Total Columns in this block: 14.
    """
    struct_lookback = 240
    buffer_ratio = 0.1 # 10% of ATR used to define a 'touch' zone
    
    # --- 1. PSYCHOLOGICAL GAPS (Round Numbers) -> 2 Columns ---
    # We focus on the 4th decimal (Pips). 
    # Example: 1.0850 or 1.0900
    df['dist_to_00_pips'] = (abs(df['<CLOSE>'] * 100 - np.round(df['<CLOSE>'] * 100)) * 10) / (df['ATR_14'] + 1e-9)
    df['dist_to_50_pips'] = (abs(df['<CLOSE>'] * 200 - np.round(df['<CLOSE>'] * 200)) * 5) / (df['ATR_14'] + 1e-9)

    # --- 2. STRATEGY SYNC (SMA-to-Wall) -> 6 Columns ---
    # Measures if your SMA 10 is reinforced by an Institutional Wall.
    for i in [1, 2, 3]:
        # We need the price of the structural pivots (re-calculated/referenced)
        # Resistance (Highs)
        h_price = df['<HIGH>'].where(df['is_ph']).rolling(struct_lookback).apply(
            lambda x: x.iloc[np.argsort(df.loc[x.index, 'raw_power_h'].values)[-i]] if not x.dropna().empty else np.nan, raw=False
        )
        df[f'dist_sma10_to_ph{i}_struct'] = (h_price - df['SMA_10']) / df['ATR_14']

        # Support (Lows)
        l_price = df['<LOW>'].where(df['is_pl']).rolling(struct_lookback).apply(
            lambda x: x.iloc[np.argsort(df.loc[x.index, 'raw_power_l'].values)[-i]] if not x.dropna().empty else np.nan, raw=False
        )
        df[f'dist_sma10_to_pl{i}_struct'] = (df['SMA_10'] - l_price) / df['ATR_14']

    # --- 3. WALL HEAT (Structural Touch Counts) -> 6 Columns ---
    # Identifies how many times price has tested and held a specific wall.
    for i in [1, 2, 3]:
        # Logic: Count how many candles in the window had Highs/Lows within the buffer zone
        
        # Heat for Resistance Walls
        h_price = df['<HIGH>'].where(df['is_ph']).rolling(struct_lookback).apply(
            lambda x: x.iloc[np.argsort(df.loc[x.index, 'raw_power_h'].values)[-i]] if not x.dropna().empty else np.nan, raw=False
        ).ffill()
        
        is_touch_h = (abs(df['<HIGH>'] - h_price) < (buffer_ratio * df['ATR_14']))
        df[f'ph{i}_struct_touch_count'] = is_touch_h.rolling(struct_lookback).sum().fillna(1)

        # Heat for Support Walls
        l_price = df['<LOW>'].where(df['is_pl']).rolling(struct_lookback).apply(
            lambda x: x.iloc[np.argsort(df.loc[x.index, 'raw_power_l'].values)[-i]] if not x.dropna().empty else np.nan, raw=False
        ).ffill()
        
        is_touch_l = (abs(df['<LOW>'] - l_price) < (buffer_ratio * df['ATR_14']))
        df[f'pl{i}_struct_touch_count'] = is_touch_l.rolling(struct_lookback).sum().fillna(1)

    return df


def apply_trendline_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 3] Trendline & Diagonal Engine V4.
    Implements all 64 columns with High-Speed Vectorization.
    Includes Gap Guard (60s) and Full 3-Line Resonance (P1-P2, P1-P3, P2-P3).
    """
    # --- 0. PREPARATION: The Gap Guard ---
    time_diffs = df.index.to_series().diff().dt.total_seconds().fillna(60)
    is_break = time_diffs > 60

    # We loop through both Highs (ph/resistance) and Lows (pl/support)
    for side, p_type in [('ph', 'is_ph'), ('pl', 'is_pl')]:
        
        # We process both Micro (Recent) and Structural (Significant) scales
        for scale in ['micro', 'structural']:
            
            # --- 1. COORDINATE RECOVERY ---
            # We pull the Prices and Ages already calculated in apply_horizontal_landmarks
            p1 = df[f'{side}1_{scale}_horizontal_price']
            p2 = df[f'{side}2_{scale}_horizontal_price']
            p3 = df[f'{side}3_{scale}_horizontal_price']
            
            t1 = df[f'{side}1_{scale}_horizontal_age']
            t2 = df[f'{side}2_{scale}_horizontal_age']
            t3 = df[f'{side}3_{scale}_horizontal_age']

            # --- 2. THE DISTANCE STACK (12 Columns) ---
            # We calculate 3 lines per scale to detect 'Phantom Resistance'
            # Line 1: (P1-P2) | Line 2: (P1-P3) | Line 3: (P2-P3)
            line_configs = [
                (p1, p2, t1, t2, '1'), 
                (p1, p3, t1, t3, '2'), 
                (p2, p3, t2, t3, '3')
            ]
            
            for pa, pb, ta, tb, name in line_configs:
                # A. The Gap Guard: window=100 covers most recent interactions
                valid_line = (is_break.rolling(window=100, min_periods=1).sum() == 0)
                
                # B. Calculate Slope (m = dy / dx)
                # Note: tb-ta is the time distance between pivots
                slope = (pa - pb) / (tb - ta).replace(0, 1)
                
                # C. Project Price to 'Now' (t=0)
                # Projected = PriceA + (Slope * Time_Since_PriceA)
                # Since ta is 'Time Ago', we use it as the multiplier
                projected_price = pa + (slope * ta)
                
                # D. Store Gap and Slope
                df[f'dist_to_{side}{name}_{scale}_trendline'] = ((projected_price - df['<CLOSE>']) / df['ATR_14']).where(valid_line)
                
                # Only store slope and velocity for the primary line (P1-P2) to save RAM
                if name == '1':
                    df[f'trendline_{side}{name}_slope_{scale}'] = (slope / df['ATR_14']).where(valid_line)
                    df[f'slope_{side}{name}_velocity_{scale}'] = df[f'trendline_{side}{name}_slope_{scale}'].diff().where(valid_line)

            # --- 3. THE AGE & POWER STACK (24 Columns) ---
            # Inherit from horizontal department to ensure logic symmetry
            for i in [1, 2, 3]:
                df[f'{side}{i}_{scale}_trendline_age'] = df[f'{side}{i}_{scale}_horizontal_age']
                df[f'{side}{i}_{scale}_trendline_power'] = df[f'{side}{i}_{scale}_horizontal_power']

            # --- 4. THE R2 (Integrity) STACK (12 Columns) ---
            # Measures how 'Straight' the 3-point connection is
            for i in [1, 2, 3]:
                # Slope between P1-P2 and P2-P3
                s12 = (p1 - p2) / (t2 - t1).replace(0, 1)
                s23 = (p2 - p3) / (t3 - t2).replace(0, 1)
                
                # R2 Proxy: Inverse of the difference in slopes
                # If slopes are identical, result is 1.0 (Perfect Line)
                diff = abs(s12 - s23) / (df['ATR_14'] + 1e-9)
                df[f'trendline_{side}{i}_{scale}_r2'] = 1 / (1 + diff)

    return df


def apply_trendline_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 4] Trendline Integrity Engine (Real R2).
    Quantifies the 'Straightness' of diagonal walls using 3-point regression error.
    Total Columns in this block: 4 (Micro/Struct x Support/Resistance).
    """
    # We loop through both Highs (Resistance) and Lows (Support)
    for side in ['ph', 'pl']:
        # Process both Micro and Structural resolutions
        for scale in ['micro', 'structural']:
            
            # 1. Get Coordinates (Scaffolding from apply_horizontal_landmarks)
            p1 = df[f'{side}1_{scale}_horizontal_price']
            p2 = df[f'{side}2_{scale}_horizontal_price']
            p3 = df[f'{side}3_{scale}_horizontal_price']
            
            t1 = df[f'{side}1_{scale}_horizontal_age']
            t2 = df[f'{side}2_{scale}_horizontal_age']
            t3 = df[f'{side}3_{scale}_horizontal_age']

            # 2. Calculate the Anchor Slope (The line connecting P1 and P3)
            # Math: m = (y1 - y3) / (x3 - x1)
            # We use t3-t1 because ages are 'time ago' (x-axis is inverted)
            anchor_slope = (p1 - p3) / (t3 - t1 + 1e-9)

            # 3. Calculate the 'Expected' Price for the middle point (P2)
            # Logic: If the line is straight, P2 must sit on this slope.
            # Expected P2 = P1 + (Slope * Time distance from P1 to P2)
            expected_p2 = p1 + (anchor_slope * (t2 - t1))

            # 4. Measure the Deviation (The 'Jaggedness')
            # Normalized by ATR to make it Stationary
            deviation = abs(p2 - expected_p2) / (df['ATR_14'] + 1e-9)

            # 5. Convert Deviation into a 0.0 to 1.0 Integrity Score
            # High Score (1.0) = Perfectly Straight | Low Score (0.0) = Noisy
            df[f'trendline_{side}_{scale}_r2_integrity'] = 1 / (1 + deviation)

    return df