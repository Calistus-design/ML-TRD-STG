# src/data_cleaner.py

import pandas as pd
import pytz
import os

class DataCleaner:
    """
    The Alpha Factory Refinery Module.
    
    This class implements the 'Data Preparation' phase of the CRISP-DM framework.
    It is responsible for transforming raw M1 market data into 'Gold Standard' 
    datasets by surgically removing toxic noise, liquidity gaps, and 
    high-impact news volatility.
    """

    def __init__(self, raw_dir: str = 'data/raw', clean_dir: str = 'data/cleaned'):
        """
        Initializes the Refinery with source and destination directories.

        Args:
            raw_dir (str): Path to the folder containing raw MT5 CSV exports.
            clean_dir (str): Path where the sanitized CSV files will be saved.
        """
        self.raw_dir = raw_dir
        self.clean_dir = clean_dir
        os.makedirs(self.clean_dir, exist_ok=True)

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Loads raw M1 data and enforces a strict UTC timezone standard.

        This ensures that regardless of the broker's server time, the entire 
        Alpha Factory engine operates on a single, global temporal anchor (UTC).

        Args:
            file_path (str): Full path to the CSV file.

        Returns:
            pd.DataFrame: A DataFrame with a UTC-localized DatetimeIndex.
        """
        df = pd.read_csv(file_path)
        
        # 1. Convert the 'time' string column to proper datetime objects
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # 2. Force UTC Awareness. 
        # If data is 'naive' (no TZ), localize it. If it has TZ, convert it.
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
            
        return df

    def remove_rollover_and_weekends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Layer 1] Time Scrubber: Removes low-liquidity and chaotic open periods.

        Logic:
        1. Rollover: Removes 17:00-18:00 New York time daily. This is the 
           'Witching Hour' where spreads explode and price action is unreliable.
        2. Monday Open: Removes the first 3 hours of the trading week (UTC 00:00-03:00) 
            to eliminate weekend gaps and 'open-shock' volatility.

        Args:
            df (pd.DataFrame): The input market data.

        Returns:
            pd.DataFrame: Data filtered for high-liquidity sessions only.
        """
        initial_count = len(df)
        
        # We must convert to NY time to find the 5:00 PM rollover regardless of DST
        ny_tz = pytz.timezone('America/New_York')
        df['time_ny'] = df.index.tz_convert(ny_tz)
        
        # Define the Rollover mask (5:00 PM to 5:59 PM)
        is_rollover = (df['time_ny'].dt.hour == 17)
        
        # Define the Monday Open mask (Monday before 3:00 AM UTC)
        is_monday_morning = (df.index.dayofweek == 0) & (df.index.hour < 3)
        
        # Apply filters
        clean_df = df[~is_rollover & ~is_monday_morning].copy()
        
        # Cleanup temporary timezone column
        clean_df.drop(columns=['time_ny'], inplace=True)
        
        removed = initial_count - len(clean_df)
        print(f"    -> [Layer 1] Time Scrubber: Removed {removed} candles (Rollover/Monday Open).")
        
        return clean_df

    def remove_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Layer 2] Noise Filter: Removes dead markets and impossible price spikes.

        Logic:
        1. Flat Periods: Removes candles with Tick Volume <= 1, which indicate 
           halted markets or data gaps that 'flatten' technical indicators.
        2. Bad Ticks: Uses a 20x ATR (Average True Range) multiple to identify 
           and delete price spikes that are physically impossible in a liquid market, 
           preventing the ML model from learning from data glitches.

        Args:
            df (pd.DataFrame): The input market data.

        Returns:
            pd.DataFrame: Data sanitized of microstructure anomalies.
        """
        initial_count = len(df)
        
        # --- 1. Remove Flat/Zero Volume Candles ---
        df = df[df['<TICKVOL>'] > 1].copy()
        flat_removed = initial_count - len(df)
        
        # --- 2. Remove Bad Ticks (Outlier Spikes) ---
        pre_tick_count = len(df)
        
        # Calculate candle range (High - Low)
        candle_range = df['<HIGH>'] - df['<LOW>']
        
        # Use a 14-period rolling average of range as a volatility baseline
        avg_range = candle_range.rolling(window=14).mean().ffill()
        
        # Detect candles exceeding the 20x multiplier safety valve
        is_bad_tick = (candle_range > (avg_range * 20.0)) & (avg_range > 0)
        
        clean_df = df[~is_bad_tick].copy()
        ticks_removed = pre_tick_count - len(clean_df)
        
        print(f"    -> [Layer 2] Noise Filter: Removed {flat_removed} flat and {ticks_removed} spike candles.")
        
        return clean_df

    def remove_news_events(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        [Layer 3] News Filter: Implements a surgical 60-minute blackout around news.

        This method connects the market data to the 'news_calendar.csv' parsed 
        from ForexFactory. It ensures the model is not trained on chaos-driven 
        fundamental volatility.

        Args:
            df (pd.DataFrame): The input market data.
            symbol (str): The currency pair being processed (to filter relevant news).

        Returns:
            pd.DataFrame: Data with news-impacted windows removed.
        """
        calendar_path = 'data/news_calendar.csv'
        
        if not os.path.exists(calendar_path):
            print("    -> [Layer 3] News Filter: ! SKIPPED (news_calendar.csv not found)")
            return df

        # Load the pre-parsed UTC news calendar
        news_df = pd.read_csv(calendar_path)
        news_df['time'] = pd.to_datetime(news_df['time']).dt.tz_localize('UTC')

        # Identify which currencies from the pair are in the news file (e.g., EUR or USD)
        currencies = [symbol[:3], symbol[3:]]
        relevant_news = news_df[news_df['currency'].isin(currencies)]
        
        if relevant_news.empty:
            print(f"    -> [Layer 3] News Filter: No high-impact events for {symbol}.")
            return df

        initial_count = len(df)
        indices_to_drop = set()
        
        # Define the 'Danger Zone' (30 minutes before and after the event)
        buffer = pd.Timedelta(minutes=30)

        # Build the set of timestamps that fall within ANY news blackout window
        for event_time in relevant_news['time']:
            start, end = event_time - buffer, event_time + buffer
            
            # Find the mask for candles in this specific event's window
            window_mask = (df.index >= start) & (df.index <= end)
            
            # Update our master list of indices to be removed
            indices_to_drop.update(df.index[window_mask])

        # Remove the marked candles in one operation
        clean_df = df.drop(index=list(indices_to_drop))
        
        removed = initial_count - len(clean_df)
        print(f"    -> [Layer 3] News Filter: Removed {removed} candles (High-Impact news).")
        
        return clean_df