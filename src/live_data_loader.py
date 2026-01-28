# src/live_data_loader.py

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os
import pytz # Still needed to correctly create UTC-aware timestamps

class MT5DataLoader:
    """
    Connects to MT5, downloads M1 data for a specific UTC date range, and saves it.
    This version is robust and timezone-safe, forcing all requests to use UTC.
    """
    def __init__(self):
        """Initializes the loader and verifies the connection to MT5."""
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            raise ConnectionError("Could not connect to MetaTrader 5 Terminal.")
        
        print("MetaTrader 5 connection successful.")
        print(f"Connected to: {mt5.terminal_info().name}")

    def disconnect(self):
        """Shuts down the connection to the MT5 terminal."""
        mt5.shutdown()
        print("MetaTrader 5 connection closed.")

    def fetch_data_for_asset(self, symbol: str, date_from: datetime, date_to: datetime):
        """
        Fetches M1 data for a single asset within a specified UTC date range.
        
        Args:
            symbol (str): The symbol to fetch (e.g., 'EURUSD').
            date_from (datetime): The start date of the period (will be treated as UTC).
            date_to (datetime): The end date of the period (will be treated as UTC).

        Returns:
            pd.DataFrame: A DataFrame with the M1 data, or None if it fails.
        """
        print(f"  Attempting to fetch M1 data for {symbol} from {date_from} (UTC) to {date_to} (UTC)...")
        
        # --- THE FIX #1: Programmatically enable the symbol in Market Watch ---
        # This is the automated way to ensure the symbol is available for data requests.
        if not mt5.symbol_select(symbol, True):
            print(f"    -> FAILED: Could not select/enable symbol {symbol}. It may not be offered by your broker.")
            # We add a brief pause to allow the terminal to catch up if needed
            mt5.symbol_select(symbol, False) # Deselect to clean up
            return None

        # --- THE FIX #2: Explicitly make the datetime objects UTC-aware ---
        # Instead of guessing the broker's timezone, we tell the API exactly what we mean.
        utc_from = date_from.replace(tzinfo=pytz.UTC)
        utc_to = date_to.replace(tzinfo=pytz.UTC)
        
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, utc_from, utc_to)

        # Deselect the symbol to keep the Market Watch clean
        mt5.symbol_select(symbol, False)

        if rates is None or len(rates) == 0:
            print(f"    -> FAILED: No data returned for {symbol}.")
            return None

        df = pd.DataFrame(rates)
        # The 'time' column from MT5 is already in UTC seconds (Unix time).
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        
        df.rename(columns={
            'open': '<OPEN>', 'high': '<HIGH>', 'low': '<LOW>', 'close': '<CLOSE>',
            'tick_volume': '<TICKVOL>', 'real_volume': '<VOL>', 'spread': '<SPREAD>'
        }, inplace=True)
        
        print(f"    -> SUCCESS: Fetched {len(df)} candles for {symbol}.")
        return df

    def run_download_process(self, date_from: datetime, date_to: datetime, output_dir: str = 'data/raw'):
        """
        Orchestrates the download process for a predefined list of assets for a specific date range.
        """
        asset_list = [
            'AUDJPY', 'AUDUSD', 'AUDCHF', 'AUDNZD', 'AUDCAD', 'CADCHF', 'EURJPY',
            'EURUSD', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD', 'EURGBP', 'GBPUSD',
            'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD', 'GBPJPY', 'NZDUSD', 'NZDCAD',
            'NZDJPY', 'USDJPY', 'USDCAD', 'CHFJPY', 'CADJPY', 'USDCHF'
        ]
        
        print(f"Starting download process for {len(asset_list)} assets...")
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol in asset_list:
            df = self.fetch_data_for_asset(symbol, date_from, date_to)
            
            if df is not None:
                file_path = os.path.join(output_dir, f"{symbol}_M1.csv")
                df.to_csv(file_path)
                print(f"    -> SAVED: {symbol} data to {file_path}\n")