# src/feature_factory.py

import pandas as pd
import numpy as np
import os
from datetime import datetime

class FeatureFactory:
    """
    The Alpha Forge.
    Transforms cleaned M1 data into a 500+ feature training set.
    Implements multi-scale sensors, geometric pivots, and ATR-normalization.
    """
    def __init__(self, cleaned_dir: str = 'data/cleaned', output_dir: str = 'data/features'):
        self.cleaned_dir = cleaned_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define our universal asset list from your 27 pairs
        self.asset_list = [f for f in os.listdir(cleaned_dir) if f.endswith('.csv')]
        print(f"Feature Factory initialized. Found {len(self.asset_list)} cleaned assets.")

    def load_cleaned_data(self, filename: str) -> pd.DataFrame:
        """Loads a single cleaned CSV and ensures correct UTC index."""
        path = os.path.join(self.cleaned_dir, filename)
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df
    
    def add_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block 1] Global Environment Stack.
        Calculates mathematical sensors to describe trend, momentum, and energy.
        """
        # --- 1. VOLATILITY (The Universal Yardstick) ---
        # We use 14-period ATR to normalize almost all other features.
        high_low = df['<HIGH>'] - df['<LOW>']
        high_close = abs(df['<HIGH>'] - df['<CLOSE>'].shift(1))
        low_close = abs(df['<LOW>'] - df['<CLOSE>'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(window=14).mean().ffill()

        # --- 2. STRATEGY ANCHOR (SMA 10) ---
        # This is your core line. We measure the distance from price to this line.
        df['SMA_10'] = df['<CLOSE>'].rolling(window=10).mean()
        df['dist_to_sma10'] = (df['<CLOSE>'] - df['SMA_10']) / df['ATR_14']

        # --- 3. TREND MOMENTUM SENSORS (EMAs) ---
        # We use EMAs here because their weighted nature produces cleaner 'Slope' data.
        for p in [10, 20, 50, 100, 200]:
            df[f'EMA_{p}'] = df['<CLOSE>'].ewm(span=p, adjust=False).mean()
            
            # Calculate Slope: Change over 3 candles, normalized by ATR.
            # Tells the model: "How many ATR-units is the trend moving per 3 minutes?"
            df[f'EMA_{p}_slope'] = (df[f'EMA_{p}'] - df[f'EMA_{p}'].shift(3)) / df['ATR_14']

        # --- 4. MOMENTUM & EXHAUSTION (RSI) ---
        for p in [7, 14, 21]:
            delta = df['<CLOSE>'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=p).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
            rs = gain / loss.replace(0, 0.001) 
            df[f'RSI_{p}'] = 100 - (100 / (1 + rs))

        # --- 5. ENERGY SQUEEZE (Bollinger Bands) ---
        bb_sma = df['<CLOSE>'].rolling(window=20).mean()
        bb_std = df['<CLOSE>'].rolling(window=20).std()
        df['BB_width'] = ((bb_sma + (2 * bb_std)) - (bb_sma - (2 * bb_std))) / bb_sma
        
        # Rank the current width against the last 24 hours (1440 mins).
        # 0.0 = Tightest (Squeeze), 1.0 = Widest (Expansion).
        df['BB_squeeze_rank'] = df['BB_width'].rolling(window=1440).rank(pct=True)


        # --- NEW: VOLATILITY ACCELERATION ---
        atr_5 = tr.rolling(window=5).mean()
        atr_20 = tr.rolling(window=20).mean()
        df['volatility_change'] = atr_5 / atr_20.replace(0, 0.001)

        return df
    

    def add_geometry_engine(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block 2] Causal Geometric Alpha Engine.
        Zero-Peeking Logic: Only uses past data to identify structural landmarks.
        """
        # --- 1. CAUSAL PIVOT DETECTION ---
        # We define a peak as: Candle[T-2] < Candle[T-1] > Candle[T]
        # This information is only available at the CLOSE of Candle[T].
        # We use a 2-candle confirmation (Standard for high-frequency).
        
        # Identify Swing Highs (Resistance Anchors)
        df['is_swing_h'] = (df['<HIGH>'].shift(1) > df['<HIGH>'].shift(2)) & \
                           (df['<HIGH>'].shift(1) > df['<HIGH>'])
        
        # Identify Swing Lows (Support Anchors)
        df['is_swing_l'] = (df['<LOW>'].shift(1) < df['<LOW>'].shift(2)) & \
                           (df['<LOW>'].shift(1) < df['<LOW>'])

        # --- 2. THE PIVOT STACK (Memory without Peeking) ---
        # We extract the price and time of the last 3 confirmed pivots.
        # We use 'ffill' to carry the last known pivot forward through time.
        # This now includes Departure Power for both Highs and Lows.        
    
        for i in [1, 2, 3]:
            # --- RESISTANCE STACK (Highs) ---
            # 1. Identify the i-th most recent confirmed Swing High
            # We use .shift(1) on the mask to ensure we aren't using the CURRENT candle
            # if it happens to be a pivot (which we wouldn't know yet anyway)
            confirmed_highs = df['<HIGH>'].where(df['is_swing_h']).shift(1)
            df[f'pivot_h{i}_price'] = confirmed_highs.ffill().shift(i-1)

            # 2. Distance to Pivot (Normalized by ATR)
            df[f'dist_to_ph{i}'] = (df['<CLOSE>'] - df[f'pivot_h{i}_price']) / df['ATR_14']

            # 3. Pivot Age (How many candles ago was it confirmed?)
            # We use a simple counter that resets when a new pivot is found.
            df[f'ph{i}_age'] = df.groupby(df['is_swing_h'].cumsum()).cumcount()

            # 4. Departure Power (The "Spring" Strength)
            # Measures the price movement after the pivot formed
            pivot_candle_range_h = (df['<HIGH>'] - df['<LOW>']).where(df['is_swing_h']).shift(1).ffill().shift(i-1)
            df[f'pivot_h{i}_power'] = pivot_candle_range_h / df['ATR_14']  # Normalized by CURRENT volatility

            # --- SUPPORT STACK (Lows) ---
            # 1. Identify the i-th most recent confirmed Swing Low
            confirmed_lows = df['<LOW>'].where(df['is_swing_l']).shift(1)
            df[f'pivot_l{i}_price'] = confirmed_lows.ffill().shift(i-1)

            # 2. Distance to Pivot (Normalized by ATR)
            df[f'dist_to_pl{i}'] = (df['<CLOSE>'] - df[f'pivot_l{i}_price']) / df['ATR_14']

            # 3. Pivot Age (optional for symmetry, can mirror High age if needed)
            df[f'pl{i}_age'] = df.groupby(df['is_swing_l'].cumsum()).cumcount()

            # 4. Departure Power (The "Spring" Strength for Lows)
            pivot_candle_range_l = (df['<HIGH>'] - df['<LOW>']).where(df['is_swing_l']).shift(1).ffill().shift(i-1)
            df[f'pivot_l{i}_power'] = pivot_candle_range_l / df['ATR_14']  # Normalized by CURRENT volatility            
           


        # --- 3. THE DIAGONAL ENGINE (Causal Trendlines) ---
        # Resistance Trendline (Highs)
        price_diff_h = df['pivot_h1_price'] - df['pivot_h2_price']
        time_diff_h = df['ph1_age'] - df['ph2_age']
        df['trendline_slope_resistance'] = (price_diff_h / time_diff_h.replace(0, 1)) / df['ATR_14']

        # Support Trendline (Lows) - NEW
        price_diff_l = df['pivot_l1_price'] - df['pivot_l2_price']
        time_diff_l = df['pl1_age'] - df['pl2_age']
        df['trendline_slope_support'] = (price_diff_l / time_diff_l.replace(0, 1)) / df['ATR_14']

        # Feature: Is the slope currently 'steep' or 'flat'? (rolling mean)
        df['slope_velocity_resistance'] = df['trendline_slope_resistance'].rolling(window=5).mean()
        df['slope_velocity_support'] = df['trendline_slope_support'].rolling(window=5).mean()

        # R-Squared Proxy for Resistance (Highs)
        highs_std = df['<HIGH>'].rolling(10).std()
        df['trendline_r2_resistance'] = 1 / (highs_std / df['ATR_14']).replace(0, 0.001)

        # R-Squared Proxy for Support (Lows)
        lows_std = df['<LOW>'].rolling(10).std()
        df['trendline_r2_support'] = 1 / (lows_std / df['ATR_14']).replace(0, 0.001)

        return df
    


    def add_candle_anatomy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block 3] Deep Anatomy Memory (HD Video of Pullback & Momentum)
        Includes Raw Lags (1-10) and Multi-Scale Rolling Context.
        """
        # 1. RAW MEASUREMENTS
        df['body_size'] = abs(df['<OPEN>'] - df['<CLOSE>'])
        df['total_range'] = (df['<HIGH>'] - df['<LOW>']).replace(0, 0.0001)

        # 2. RATIO FEATURES (Stationarity)
        df['body_ratio'] = df['body_size'] / df['total_range']
        df['top_wick_ratio'] = (df['<HIGH>'] - df[['<OPEN>', '<CLOSE>']].max(axis=1)) / df['total_range']
        df['bottom_wick_ratio'] = (df[['<OPEN>', '<CLOSE>']].min(axis=1) - df['<LOW>']) / df['total_range']
        
        # Relative size (Normalized by ATR)
        df['relative_size'] = df['total_range'] / df['ATR_14']

        # 3. MOMENTUM SCORE (Your Logic)
        df['is_healthy_bull'] = (df['<CLOSE>'] > df['<OPEN>']) & (df['body_ratio'] > 0.5) & (df['relative_size'] > 0.5)
        df['is_healthy_bear'] = (df['<CLOSE>'] < df['<OPEN>']) & (df['body_ratio'] > 0.5) & (df['relative_size'] > 0.5)
        
        # Shift(1) ensures we look at the 'Setup' before the trigger
        df['bull_momentum_count'] = df['is_healthy_bull'].shift(1).rolling(window=3).sum()
        df['bear_momentum_count'] = df['is_healthy_bear'].shift(1).rolling(window=3).sum()

        # 4. THE HD VIDEO (Raw Lags 1-10)
        # Captures the exact shape of the last 10 candles (Deep Pullback Story)
        for lag in range(1, 11):
            df[f'body_ratio_lag_{lag}'] = df['body_ratio'].shift(lag)
            df[f'wick_low_lag_{lag}'] = df['bottom_wick_ratio'].shift(lag)
            df[f'wick_high_lag_{lag}'] = df['top_wick_ratio'].shift(lag)
            df[f'size_atr_lag_{lag}'] = df['relative_size'].shift(lag)

        # 5. THE STRATEGIC SUMMARY (Rolling Multi-Scale Anatomy)
        # Captures the Vibe of the session [5, 10, 20, 30, 60]
        # Added 200 to capture the 'Institutional Vibe' of the last few hours
        for w in [5, 10, 20, 30, 60, 200]:
            df[f'avg_body_ratio_{w}'] = df['body_ratio'].shift(1).rolling(w).mean()
            df[f'avg_volatility_{w}'] = df['relative_size'].shift(1).rolling(w).mean()

        # 6. SMA 10 INTERACTION (The Location)
        df['SMA_10'] = df['<CLOSE>'].rolling(10).mean()
        
        df['close_sma_dist'] = (df['<CLOSE>'] - df['SMA_10']) / df['ATR_14']

        # --- NEW: MOMENTUM QUALITY METRICS ---
        
        # 1. Pullback Intensity (Body Size of Pullback vs Trend)
        # Avg body size of last 3 candles (Pullback) / Avg body size of candles 4-6 (Trend)
        avg_body_3 = df['body_size'].rolling(3).mean()
        avg_body_prev_3 = df['body_size'].shift(3).rolling(3).mean()
        df['pullback_intensity'] = avg_body_3 / avg_body_prev_3.replace(0, 0.001)
        
        # 2. Wick Pressure Ratio
        # Do the last 3 candles have more wicks than bodies? (Choppy/Rejection)
        sum_wicks_3 = (df['top_wick_ratio'] + df['bottom_wick_ratio']).rolling(3).sum()
        sum_bodies_3 = df['body_ratio'].rolling(3).sum()
        df['wick_pressure_ratio'] = sum_wicks_3 / sum_bodies_3.replace(0, 0.001)
        
        # 3. Momentum Cleanliness (Trend Efficiency)
        # Total Pips Moved (High-Low) / Total Distance Traveled (Sum of Ranges) for last 5 candles
        # If 1.0, it moved in a straight line. If 0.1, it went back and forth.
        net_move = abs(df['<CLOSE>'] - df['<CLOSE>'].shift(5))
        total_path = df['total_range'].rolling(5).sum()
        df['momentum_cleanliness'] = net_move / total_path.replace(0, 0.001)

        return df
    

    # --- ADD THESE TO THE END OF YOUR CLASS ---

    def add_time_vectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block 4] Cyclical Time Encoding.
        Converts linear time (0-23) into circular Sin/Cos components.
        This prevents the model from being confused by the midnight 'clock flip'.
        """
        # Encode Hour of the Day (24-hour cycle)
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Encode Day of the Week (5-day trading cycle)
        # Monday=0, Friday=4
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 5)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 5)
        
        # Drop raw hour to avoid linear bias
        df.drop(columns=['hour'], inplace=True)
        return df

    def add_multi_horizon_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Block 5] The Multi-Horizon Oracle (Answer Key).
        Calculates the trade outcome for 1, 2, 3, 4, and 5 minute expiries.
        
        Logic: 
        A 'Win' (1) requires the price to move at least 0.5 * ATR in favor.
        Otherwise, it is a 'Loss' (0).
        """
        # The 'Comfortable Win' buffer (Volatility Adjusted)
        buffer = 0.5 * df['ATR_14']
        
        # We loop through each potential binary expiry
        for m in [1, 2, 3, 4, 5]:
            # Look ahead 'm' candles into the future
            future_close = df['<CLOSE>'].shift(-m)
            
            # Label for CALL: Price moved UP > buffer
            df[f'target_{m}m_call'] = ((future_close - df['<CLOSE>']) > buffer).astype(int)
            
            # Label for PUT: Price moved DOWN > buffer
            df[f'target_{m}m_put'] = ((df['<CLOSE>'] - future_close) > buffer).astype(int)
            
        return df

    def generate_asset_textbook(self, filename: str) -> pd.DataFrame:
        """
        The Master Assembly Line.
        Ingests a cleaned file and runs it through all 5 Logic Blocks.
        """
        # 1. Load data
        df = self.load_cleaned_data(filename)
        symbol = filename.split('_')[0]
        
        # 2. Run the Sensor Blocks
        df = self.add_base_indicators(df) # Ensure you fixed 'return df' here!
        df = self.add_geometry_engine(df)
        df = self.add_candle_anatomy(df)
        df = self.add_time_vectors(df)
        df = self.add_multi_horizon_labels(df)
        
        # THE INDUSTRIAL FUNNEL: Range-Aware Proximity
        # Captures if ANY part of the candle (High to Low) is within 0.5 ATR of the SMA 10
        zone = 0.5 * df['ATR_14']
        is_near_sma = (df['SMA_10'] >= (df['<LOW>'] - zone)) & (df['SMA_10'] <= (df['<HIGH>'] + zone))

        # Apply the new filter
        candidates = df[is_near_sma].copy()
        
        
        # 4. Add Asset Identity (Categorical feature for the Universal Model)
        # We extract the name and map it to an ID
        # (Make sure self.asset_map is defined in your __init__)
        candidates['asset_id'] = symbol
        
        # 5. Drop NaNs AND Reset Index to make 'time' a column
        return candidates.dropna().reset_index()