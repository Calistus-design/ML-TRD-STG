# src/features/dynamics.py
import pandas as pd
import numpy as np

def apply_market_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """
    The 'Kinetic Sensor' Department.
    Calculates 15 high-order features to measure Force, Friction, and Exhaustion.
    Includes the 'Gap Guard' to ensure derivatives are only calculated on continuous data.
    """
    
    # 0. THE GAP GUARD: 60 seconds is the atomic threshold.
    # We identify any breaks (>60s) to prevent the math from jumping across news/rollover.
    time_diffs = df.index.to_series().diff().dt.total_seconds().fillna(60)
    is_break = time_diffs > 60

    # --- 1. EMA ACCELERATION STACK (Speed vs. Force) -> 10 Columns ---
    # Logic: 1st Order (Slope) = Speed | 2nd Order (Acceleration) = Force.
    # Detects if the trend engine is 'Redlining' or 'Igniting.'
    # We use the normalized slopes already calculated in the Base Engine.
    for p in [10, 20, 50, 100, 200]:
        # A. Reference the Velocity (1st Order Derivative)
        # We use the standard name from base_engine.py
        slope_name = f'EMA_{p}_slope_norm'
        
        # B. Acceleration (2nd Order Derivative): Change in Slope over the last 3 bars
        # Tells us if the trend is gaining mass or losing steam.
        valid_accel = (is_break.rolling(window=6).sum() == 0) # Needs a longer continuous window
        df[f'EMA_{p}_acceleration'] = (df[slope_name] - df[slope_name].shift(3)).where(valid_accel)


    # --- 2. KAUFMAN’S EFFICIENCY RATIO (Market Friction) -> 3 Columns ---
    # Logic: Direct Distance / Total Path Traveled.
    # 1.0 = Straight line (High Alpha). 0.1 = Chaotic noise (Sideways/Chop).
    for n in [5, 10, 30]:
        valid_er = (is_break.rolling(window=n).sum() == 0)
        
        # Net change from start of window to end
        net_change = abs(df['<CLOSE>'] - df['<CLOSE>'].shift(n))
        # Sum of every individual 1-minute move (volatility path)
        path = abs(df['<CLOSE>'] - df['<CLOSE>'].shift(1)).rolling(window=n).sum()
        
        # Column: ER_5, ER_10, ER_30
        df[f'ER_{n}'] = (net_change / (path + 1e-9)).where(valid_er)

    # --- 3. RELATIVE VELOCITY DELTA (Exhaustion Sensor) -> 2 Columns ---
    # Logic: Current Intensity / Recent Average Intensity.
    # Compares the 'Horsepower' of the Pullback to the preceding Trend.
    
    # We use the EMA 10 slope as the proxy for micro-velocity
    # Current absolute speed
    current_velocity = abs(df['EMA_10_slope_norm'])
    # Average speed of the last 10 minutes (The 'Trend' baseline)
    avg_trend_velocity = current_velocity.shift(1).rolling(window=10).mean()
    
    valid_delta = (is_break.rolling(window=10).sum() == 0)
  
    # Delta for Bulls (Confirmation of Rejection)
    # If < 1.0: Pullback is slower than trend (Healthy). 
    # If > 2.0: Pullback is crashing (Reversal/Dangerous).
    df['relative_velocity_delta'] = (current_velocity / (avg_trend_velocity + 1e-9)).where(valid_delta)
    
    # Alpha Multiplier: Momentum Efficiency (Divergence Proxy)
    # Compares Price velocity to RSI velocity
    df['momentum_efficiency_ratio'] = (df['EMA_10_slope_norm'] / (df['RSI_14'].diff() + 1e-9)).where(valid_delta)


    # --- 4. RSI MOMENTUM EFFICIENCY (Divergence Sensors) -> 3 Columns ---
    # Logic: Compares Price Displacement to RSI Displacement.
    # Why: If price makes a new high but RSI moves less than history suggests,
    # it indicates a 'Momentum Leak' (Divergence). High values = High Efficiency.
    # Low values = Exhaustion (The 'Dead-End' setup).
    for n in [7, 14, 21]:
        # Window is valid only if the market flow was continuous over n bars
        valid_div = (is_break.rolling(window=n).sum() == 0)
            
        
        # Price change normalized by ATR
        price_dist = (df['<CLOSE>'] - df['<CLOSE>'].shift(n)) / (df['ATR_14'] + 1e-9)
        # RSI change over the same window
        rsi_dist = df[f'RSI_{n}'] - df[f'RSI_{n}'].shift(n)
        
        # The Ratio: Distance traveled per unit of RSI momentum
        df[f'RSI_{n}_momentum_efficiency'] = (price_dist / (rsi_dist + 1e-9)).where(valid_div)


    # --- 5. MOMENTUM MASS (Institutional Weight) -> 2 Columns ---
    # Logic: Sum of Body Sizes / Sum of ATRs over the last 3 candles.
    # Why: Replaces rigid 1/0 'Healthy' flags. This tells the machine the 
    # actual 'Weight' of the move. 1.0 is average; 3.0 is institutional force.
    # It distinguishes between a 'Retail Ripple' and an 'Institutional Wave.'
    valid_mass = (is_break.rolling(window=3).sum() == 0 )
                 
    body_sum_3 = df['body_size'].rolling(3).sum()
    atr_sum_3 = df['ATR_14'].rolling(3).sum()
    
    # We separate the Mass by direction so the model sees Bullish vs Bearish conviction
    df['bull_momentum_mass'] = ((body_sum_3 / (atr_sum_3 + 1e-9)).where(df['<CLOSE>'] > df['<OPEN>'])).fillna(0).where(valid_mass)
    df['bear_momentum_mass'] = ((body_sum_3 / (atr_sum_3 + 1e-9)).where(df['<CLOSE>'] < df['<OPEN>'])).fillna(0).where(valid_mass)


    # --- 6. VOLATILITY CORRIDOR (The Reachability Metric) -> 16 Columns ---
    # Logic: Ratio = (ATR_Target_Distance) / (ATR_Volatility * Expiry_Time).
    # Why: In Binary Options, you are racing the clock. If R > 1.0, the price 
    # is physically unlikely to reach the target before expiry.
    # This filter identifies 'Impossible Wins' and 'Easy Scraps.'
    expiries = [3, 4, 5, 10]
    tiers = [0.3, 0.75, 1.5, 3.0] # Your 4 V4 distance tiers
    
    for m in expiries:
        for k in tiers:
            # We calculate the 'Effort' required to reach this specific tier
            # Logic: If market moves at 1 ATR/min, it can reach 'm' distance.
            # If our target is 'k' distance, the reachability ratio is k/m.
            # We use a moving ATR average to adjust for the 'Session Speed.'
            session_speed = df['ATR_14'].rolling(m).mean() + 1e-9
            target_dist = k * df['ATR_14']
            
            df[f'reach_m{m}_tier{str(k).replace(".","")}'] = (target_dist / (session_speed * m))

    # --- 7. MARKET SMOOTHNESS (Recycled V3 Alpha) -> 2 Columns ---
    # Logic: Inverse Relative Volatility (1 / (StdDev / ATR)).
    # Why: This was the 'Fake R2' from V3 that ranked Rank 5 in importance.
    # It identifies 'Orderly' sessions where price action is smooth and 
    # lacks spiky noise, which is the perfect environment for SMA pullbacks.
    
    # Resistance Smoothness (Highs)
    highs_std = df['<HIGH>'].rolling(10).std()
    # Support Smoothness (Lows)
    lows_std = df['<LOW>'].rolling(10).std()
    
    # Gap Guard: Window is valid only if last 10 candles were continuous
    valid_smooth = (is_break.rolling(window=10).sum() == 0)
    
    # Normalized Smoothness Score
    # Updated Lines in dynamics.py
    # High Score (1.0) = Perfectly Smooth | Low Score (0.0) = Chaotic Noise
    df['smoothness_highs_10'] = (1 / (1 + (highs_std / (df['ATR_14'] + 1e-9)))).where(valid_smooth)
    df['smoothness_lows_10'] = (1 / (1 + (lows_std / (df['ATR_14'] + 1e-9)))).where(valid_smooth)
    return df