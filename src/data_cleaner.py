import pandas as pd
import pytz
import os
import numpy as np

class DataCleaner:
    """
    The Alpha Factory Refinery Module V4.
    
    This class implements the 'Data Preparation' phase of the CRISP-DM framework.
    It transforms enriched M1 market data into 'Gold Standard' datasets by 
    surgically removing toxic noise, liquidity gaps, and high-impact news, 
    while identifying the 0.5 * ATR candidate pool.
    """

    def __init__(self, raw_dir: str = 'data/raw', clean_dir: str = 'data/cleaned'):
        """
        Initializes the Refinery with source and destination directories.
        """
        self.raw_dir = raw_dir
        self.clean_dir = clean_dir
        os.makedirs(self.clean_dir, exist_ok=True)

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Loads enriched M1 data and enforces a strict UTC timezone standard.
        This is the temporal foundation of the factory.
        """
        df = pd.read_csv(file_path)
        
        # 1. Convert the 'time' string column to proper datetime objects
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # 2. Force UTC Awareness. 
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
            
        return df

    def remove_rollover_and_weekends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Layer 1] Time Scrubber: Removes low-liquidity and chaotic open periods.

        Logic:
        1. Rollover: 17:00-18:00 NY Time daily (The Witching Hour).
        2. Monday Open: UTC 00:00-03:00 (Weekend gap shock).
        """
        initial_count = len(df)
        
        # Convert to NY time to find 5:00 PM rollover regardless of DST
        ny_tz = pytz.timezone('America/New_York')
        # We use a temporary series to avoid changing the main UTC index
        ny_hours = df.index.tz_convert(ny_tz).hour
        
        # Define masks
        is_rollover = (ny_hours == 17)
        is_monday_morning = (df.index.dayofweek == 0) & (df.index.hour < 3)
        
        # Apply filters
        clean_df = df[~is_rollover & ~is_monday_morning].copy()
        
        removed = initial_count - len(clean_df)
        print(f"    -> [Layer 1] Time Scrubber: Removed {removed} candles (Rollover/Monday Open).")
        
        return clean_df

    def remove_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Layer 2] Noise Filter: Removes dead markets and impossible price spikes.

        Logic:
        1. Flat Periods: TICKVOL <= 1 (Halted markets).
        2. Bad Ticks: 20x ATR spike threshold (Data glitches).
        """
        initial_count = len(df)
        
        # 1. Kill Flat Candles
        df = df[df['<TICKVOL>'] > 2].copy()
        flat_removed = initial_count - len(df)
        
        # 2. Kill Bad Ticks
        pre_tick_count = len(df)
        candle_range = df['<HIGH>'] - df['<LOW>']
        
        # We use the ATR_14 already calculated by the Raw Engine
        is_bad_tick = (candle_range > (df['ATR_14'] * 20.0))
        
        clean_df = df[~is_bad_tick].copy()
        ticks_removed = pre_tick_count - len(clean_df)
        
        print(f"    -> [Layer 2] Noise Filter: Removed {flat_removed} flat and {ticks_removed} spike candles.")
        
        return clean_df
    

    def remove_news_events(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        [Layer 3] News Filter & Ringing Sensor.
        Updated: Explicitly preserves DatetimeIndex to prevent 'dayofweek' errors.
        """
        calendar_path = 'data/news_calendar.csv'
        
        # Default value
        df['minutes_since_news'] = 1440.0 

        if not os.path.exists(calendar_path):
            return df

        # 1. Load Calendar
        news_df = pd.read_csv(calendar_path)
        news_df['time'] = pd.to_datetime(news_df['time']).dt.tz_localize('UTC')

        # 2. Filter news
        currencies = [symbol[:3], symbol[3:]]
        relevant_news = news_df[news_df['currency'].isin(currencies)].copy()
        
        if relevant_news.empty:
            return df

        # --- 3. CALCULATE THE RINGING SENSOR (Index-Safe) ---
        # Move index to a column so we can merge safely
        df_reset = df.reset_index()
        
        news_times = relevant_news[['time']].sort_values('time')
        news_times['last_news_time'] = news_times['time']
        
        # Perform the merge on the columns
        df_merged = pd.merge_asof(
            df_reset.sort_values('time'), 
            news_times, 
            on='time', 
            direction='backward'
        )
        
        # Calculate minutes delta
        df_merged['minutes_since_news'] = (df_merged['time'] - df_merged['last_news_time']).dt.total_seconds() / 60.0
        df_merged['minutes_since_news'] = df_merged['minutes_since_news'].fillna(1440.0).clip(upper=1440.0)
        
        # --- 4. THE BLACKOUT PURGE ---
        indices_to_drop = []
        buffer = pd.Timedelta(minutes=30)

        for event_time in relevant_news['time']:
            start, end = event_time - buffer, event_time + buffer
            # Find rows within the blackout window
            mask = (df_merged['time'] >= start) & (df_merged['time'] <= end)
            indices_to_drop.extend(df_merged.index[mask].tolist())

        # Drop and clean scaffolding
        df_clean = df_merged.drop(index=indices_to_drop).copy()
        df_clean.drop(columns=['last_news_time'], errors='ignore', inplace=True)

        # --- 5. THE CRITICAL RESTORATION ---
        # We MUST set the index back to 'time' and ensure it's a DatetimeIndex
        df_clean.set_index('time', inplace=True)
        # Re-sort to ensure continuity for the Lenses (MTF Windows)
        df_clean.sort_index(inplace=True)

        print(f"    -> [Layer 3] News Filter: 'minutes_since_news' active. Index Restored.")
        
        return df_clean
    
    def identify_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Layer 4] The High-Volume Funnel.
        Surgically selects only the rows where price is in the SMA 10 'Aura'.
        Uses 0.5 * ATR Range-Aware Proximity.
        """
        # The 0.5 * ATR Proximity Rule (Range-Aware)
        # Captures if ANY part of the candle (High to Low) is within the aura.
        zone = 0.5 * df['ATR_14']
        
        # Logic: (SMA is below High + zone) AND (SMA is above Low - zone)
        is_near_sma = (df['SMA_10'] <= (df['<HIGH>'] + zone)) & \
                      (df['SMA_10'] >= (df['<LOW>'] - zone))

        # Filter the textbook pool
        candidates = df[is_near_sma].copy()
        
        return candidates



    def harden_physics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Refinery V3: Final Industrial Standard.
        Automated logic to stabilize all 320+ features for 27 assets.
        """
        # 1. TIME: Unix Integer conversion
        if df.index.name == 'time':
            df = df.reset_index()
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time']).view('int64') // 10**9
        
        # 2. THE INFINITY KILL SWITCH
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 3. SURGICAL PHYSICS SHIELD (Distances)
        # Delete rows where price is > 25 ATRs away (Broken Physics)
        dist_cols = [c for c in df.columns if 'dist_to_' in c and 'norm' in c]
        if dist_cols:
            physics_mask = (df[dist_cols].abs() <= 25.0).all(axis=1)
            df = df[physics_mask].copy()

        # 4. UNIVERSAL NOISE STABILIZER (All other features)
        # Identify feature columns automatically
        system_cols = ['time', 'asset_id', 'minutes_since_news']
        target_cols = [c for c in df.columns if 'target' in c]
        
        # Clip everything that isn't a System Col, a Target, or a Distance
        clip_candidates = [c for c in df.columns if c not in system_cols 
                        and c not in target_cols 
                        and c not in dist_cols]
        
        if clip_candidates:
            df[clip_candidates] = df[clip_candidates].clip(lower=-100.0, upper=100.0)

        # 5. FINAL HYGIENE
        # We do NOT drop columns here to maintain Parquet schema consistency.
        # We only drop rows missing the primary anchor.
        df.dropna(subset=['dist_to_sma10'], inplace=True)
        
        return df
    
    def calculate_news_sensor(self, df: pd.DataFrame, symbol: str, news_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculates 'minutes_since_news' sensor. 
        Supports both Batch (Disk-based) and Live (RAM-based) modes.
        """
        # 1. Mode Detection: Use provided RAM data or load from Disk
        if news_df is None:
            calendar_path = 'data/news_calendar.csv'
            if not os.path.exists(calendar_path):
                df['minutes_since_news'] = 1440.0
                return df
            news_df = pd.read_csv(calendar_path)
            # Ensure time is localized to UTC to match candle data
            news_df['time'] = pd.to_datetime(news_df['time']).dt.tz_localize('UTC')

        # 2. Filter for relevant currencies
        currencies = [symbol[:3], symbol[3:]]
        relevant_news = news_df[news_df['currency'].isin(currencies)].copy()
        
        if relevant_news.empty:
            df['minutes_since_news'] = 1440.0
            return df

        # 3. Execution: Temporal Mapping
        df_reset = df.reset_index()
        news_times = relevant_news[['time']].sort_values('time')
        news_times['last_news_time'] = news_times['time']
        
        # Backward-looking merge (Safe: No Peeking)
        df_merged = pd.merge_asof(
            df_reset.sort_values('time'), 
            news_times, 
            on='time', 
            direction='backward'
        )
        
        # 4. Sensor Calculation
        df_merged['minutes_since_news'] = (df_merged['time'] - df_merged['last_news_time']).dt.total_seconds() / 60.0
        df_merged['minutes_since_news'] = df_merged['minutes_since_news'].fillna(1440.0).clip(upper=1440.0)
        
        # --- TECHNICAL HARDENING ---
        # Remove the datetime scaffolding so it doesn't crash the Silencer/Clipper
        df_merged.drop(columns=['last_news_time'], errors='ignore', inplace=True)
        
        # Restore index and return
        df_merged.set_index('time', inplace=True)
        return df_merged
