# src/features/oracle.py
import pandas as pd
import numpy as np

def apply_oracle_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 4] The Multi-Label Oracle.
    Generates 32 targets (4 expiries x 2 directions x 4 tiers).
    Implements Cumulative Labeling logic: High-Intensity wins trigger Low-Intensity wins.
    """
    
    # 1. Configuration
    expiries = [3, 4, 5, 10]
    tiers = [0.3, 0.75, 1.5, 3.0]
    
    # We use a loop to build the matrix efficiently
    for m in expiries:
        # Calculate the future price change
        # Logic: Price at T+m minus Price at T
        future_delta = df['<CLOSE>'].shift(-m) - df['<CLOSE>']
        
        for k in tiers:
            # The 'Comfort' buffer for this specific tier
            buffer = k * df['ATR_14']
            
            # --- CALL Targets ---
            # Result is 1 if price moved UP more than the buffer
            col_call = f'target_{m}m_call_{str(k).replace(".", "")}'
            df[col_call] = (future_delta > buffer).astype('int8')
            
            # --- PUT Targets ---
            # Result is 1 if price moved DOWN more than the buffer
            col_put = f'target_{m}m_put_{str(k).replace(".", "")}'
            df[col_put] = (future_delta < -buffer).astype('int8')

    return df

def perform_scaffolding_purge(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Hygiene] The Final RAM Recovery.
    Surgically removes every raw coordinate and temporary math column.
    Ensures the model only sees Stationary Ratios and Gaps.
    """
    # 1. Define the 'Toxic' Scaffolding
    # These were used for math but cause overfitting if shown to the brain
    raw_coordinates = [
        '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', '<SPREAD>', '<VOL>',
        'is_ph', 'is_pl', 'raw_power_h', 'raw_power_l', 'SMA_10', 'temp_time',
        'last_news_time', 'time_x', 'time_y', 
        # DEAD SENSORS: Verified 100% NaN in V4 Tournament
        'dist_sma10_to_ph1_struct', 'dist_sma10_to_pl1_struct', 
        'dist_sma10_to_ph2_struct', 'dist_sma10_to_pl2_struct', 
        'dist_sma10_to_ph3_struct', 'dist_sma10_to_pl3_struct',

        'trendline_ph1_structural_r2', 'trendline_ph2_structural_r2', 
        'trendline_ph3_structural_r2', 'trendline_pl1_slope_structural',
        'trendline_ph1_slope_structural', 'trendline_pl_structural_r2_integrity',
        'trendline_ph_structural_r2_integrity', 'trendline_pl3_structural_r2',
        'trendline_pl2_structural_r2', 'trendline_pl1_structural_r2',
        'slope_ph1_velocity_structural', 'slope_pl1_velocity_structural'
        
    ]
    
    # 2. Identify Price-Based pivots (ending in _price)
    # These are non-stationary addresses
    price_columns = [c for c in df.columns if '_price' in c]
    
    # 3. Identify redundant indicators (if any remain)
    # We only want Gaps and Slopes
    scaffolding = raw_coordinates + price_columns
    
    # 4. The Purge
    # We use errors='ignore' in case a column was already deleted by a department
    df.drop(columns=scaffolding, errors='ignore', inplace=True)
    
    
    # 5. The NaN Cleanup [V4.3 SURGICAL UPDATE]
    # We ONLY drop rows if the target (the answer) or primary anchor is missing.
    # This protects your trade volume while letting CatBoost handle feature-specific NaNs.
    initial_rows = len(df)
    
    # Identify all your target columns and the primary SMA anchor
    target_cols = [c for c in df.columns if 'target_' in c]
    essential_cols = ['dist_to_sma10'] + target_cols
    
    # Drop only if essential columns are missing
    df.dropna(subset=essential_cols, inplace=True)
    
    print(f"    -> [Hygiene] Purged scaffolding and removed {initial_rows - len(df)} rows containing NaNs.")
    
    return df