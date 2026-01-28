# verify_cleaning.py

import pandas as pd
import pytz

def verify_rollover_removal(file_path):
    print(f"Inspecting: {file_path}")
    
    # 1. Load the CLEANED file
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Ensure UTC awareness
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    # 2. Convert to NY Time
    ny_tz = pytz.timezone('America/New_York')
    df['time_ny'] = df.index.tz_convert(ny_tz)
    
    # 3. Search for the "Forbidden Hour" (17:00 - 17:59)
    forbidden_candles = df[df['time_ny'].dt.hour == 17]
    
    count = len(forbidden_candles)
    
    if count == 0:
        print("✅ SUCCESS: No candles found in the 5 PM NY Rollover hour.")
    else:
        print(f"❌ FAILURE: Found {count} candles during Rollover (17:00 NY)!")
        print(forbidden_candles.head())

if __name__ == "__main__":
    # Check the cleaned file we just made
    verify_rollover_removal("data/cleaned/EURUSD_M1.csv")