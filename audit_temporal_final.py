# audit_temporal_final.py
import pandas as pd
import numpy as np

def run_final_temporal_audit():
    #file_path = "data/EURUSD_M1_v4.csv"
    file_path = "data/EURCAD_M1_v4.csv"
    print(f"🕵️  STARTING FINAL TEMPORAL AUDIT: {file_path}")
    
    # 1. Load a slice of the data
    # We use the index to get human dates. 
    # NOTE: Your Parquet saves time as Unix integers (seconds).
    df = pd.read_parquet(file_path)
    df['human_time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    print("-" * 60)
    
    # --- TEST 1: MONDAY OPEN (00:00 - 03:00 UTC) ---
    monday_data = df[df['human_time'].dt.dayofweek == 0]
    monday_early_hours = monday_data[monday_data['human_time'].dt.hour < 3]
    
    if monday_early_hours.empty:
        print("✅ PASS: Monday Morning Chaos (00:00-03:00) successfully removed.")
    else:
        print(f"❌ FAIL: Detected {len(monday_early_hours)} rows on Monday before 03:00 UTC.")

    # --- TEST 2: NY ROLLOVER (17:00 NY TIME) ---
    # Convert UTC to NY to check the 17:00 blackout
    df['ny_time'] = df['human_time'].dt.tz_convert('America/New_York')
    rollover_rows = df[df['ny_time'].dt.hour == 17]
    
    if rollover_rows.empty:
        print("✅ PASS: Daily Rollover (17:00 NY Time) successfully removed.")
    else:
        print(f"❌ FAIL: Detected {len(rollover_rows)} rows during the NY Rollover hour.")

    # --- TEST 3: SPECIFIC NEWS EVENT (Dec 5, 2025 - CAD Employment) ---
    # From your image: Dec 5, 2025, at 13:30 UTC
    news_event = pd.Timestamp('2025-12-05 13:30:00', tz='UTC')
    buffer = pd.Timedelta(minutes=30)
    
    news_window = df[(df['human_time'] >= news_event - buffer) & 
                     (df['human_time'] <= news_event + buffer)]
    
    if news_window.empty:
        print(f"✅ PASS: News Blackout for {news_event} is clean. No rows exist in the window.")
    else:
        print(f"❌ FAIL: Detected {len(news_window)} rows inside the Dec 5 news window.")
        print(f"      Check if your 'minutes_since_news' logic is using the same timezone.")

    print("-" * 60)
    print("📊 DATASET BOUNDARIES:")
    print(f"Start: {df['human_time'].min()}")
    print(f"End  : {df['human_time'].max()}")
    print("-" * 60)

if __name__ == "__main__":
    run_final_temporal_audit()