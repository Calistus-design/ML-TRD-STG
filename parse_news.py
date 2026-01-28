# parse_news.py

import pandas as pd
import re
import os

def parse_forex_factory_to_csv(file_path, year):
    if not os.path.exists(file_path):
        return pd.DataFrame()

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    events = []
    curr_date = None
    curr_time = None
    
    # List of currencies in your 27 pairs
    target_currencies = {'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD'}

    for i in range(len(lines)):
        line = lines[i].strip()
        if not line: continue

        # 1. Capture Date (e.g., "Jan 28")
        date_match = re.match(r"([A-Z][a-z]{2})\s+(\d+)", line)
        if date_match:
            curr_date = f"{date_match.group(1)} {date_match.group(2)}, {year}"
            continue

        # 2. Capture Time (e.g., "0:30" or "19:00")
        time_match = re.match(r"(\d{1,2}:\d{2})", line)
        if time_match:
            curr_time = time_match.group(1)
            continue

        # 3. Capture Event: If line is a Currency, the NEXT line is the Event Name
        if line in target_currencies:
            event_name = "Unknown Event"
            if i + 1 < len(lines):
                event_name = lines[i+1].strip()
            
            if curr_date and curr_time:
                try:
                    full_ts_str = f"{curr_date} {curr_time}"
                    dt = pd.to_datetime(full_ts_str, format='%b %d, %Y %H:%M')
                    
                    events.append({
                        'time': dt,
                        'currency': line,
                        'event': event_name
                    })
                except:
                    continue

    return pd.DataFrame(events)

if __name__ == "__main__":
    print("🚀 Alpha Factory: Refining News (2024-2026) into Table Structure...")
    
    # 1. Parse each year's raw text dump separately
    df_2024 = parse_forex_factory_to_csv("data/raw_news_2024.txt", 2024)
    df_2025 = parse_forex_factory_to_csv("data/raw_news_2025.txt", 2025)
    df_2026 = parse_forex_factory_to_csv("data/raw_news_2026.txt", 2026) # Added 2026
    
    # 2. Collect all non-empty DataFrames
    dfs_to_combine = []
    for df in [df_2024, df_2025, df_2026]:
        if not df.empty:
            dfs_to_combine.append(df)
    
    # 3. Combine and Finalize
    if not dfs_to_combine:
        print("❌ ERROR: No data found in any of the raw text files.")
    else:
        full_df = pd.concat(dfs_to_combine)
        
        # Chronological sort is critical for the cleaning logic later
        full_df.sort_values('time', inplace=True)
        
        # Drop duplicates where multiple events affect the same currency at the same time
        full_df.drop_duplicates(subset=['time', 'currency'], inplace=True)
        
        # SAVE AS THE MASTER CALENDAR
        output_path = "data/news_calendar.csv"
        full_df.to_csv(output_path, index=False)
        
        print(f"✅ SUCCESS: Refined {len(full_df)} events for the 2024-2026 period.")
        print(f"File saved to: {output_path}")

        print("\nStructure preview (Matches FF Table):")
        print(full_df.head(10))