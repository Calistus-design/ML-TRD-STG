# audit_parquet.py
import pandas as pd
import numpy as np

def generate_integrity_report(file_path):
    print(f"🕵️  STARTING FORENSIC AUDIT: {file_path}")
    df = pd.read_parquet(file_path)
    
    # 1. Generate Summary Stats
    # Percentiles .01 and .99 help us see if the data is "clumping" at the edges
    report = df.describe(percentiles=[.01, .99]).T
    report['nan_count'] = df.isna().sum()
    
    # 2. Automated Violation Check
    # We define a "Price Leak" as any mean value > 10 (excluding system cols)
    check_cols = [c for c in df.columns if c not in ['time', 'asset_id']]
    means = df[check_cols].mean().abs()
    leaks = means[means > 10.0].index.tolist()
    # Ignore columns where large values are normal (like time-based features)
    leaks = [c for c in leaks if 'minutes_since' not in c and 'sin' not in c and 'cos' not in c]

    # 3. Output Results
    print("-" * 60)
    print(f"📊 DATASET STATS: {len(df):,} rows | {len(df.columns)} columns")
    
    if leaks:
        print(f"❌ CRITICAL ERROR: {len(leaks)} Potential Price Leaks Detected!")
        print(f"   Check columns: {leaks[:5]}...")
    else:
        print("✅ STATIONARITY CHECK: Passed. No raw price leakage found.")

    extreme_outliers = (df[check_cols].abs() > 50.0).any().sum()
    if extreme_outliers > 0:
        print(f"⚠️  WARNING: {extreme_outliers} columns still contain values > 50.0.")
    else:
        print("✅ OUTLIER CHECK: Passed. All features within physical reality.")

    # 4. Save the full spreadsheet for manual inspection
    report.to_csv("data/factory_integrity_report.csv")
    print("-" * 60)
    print("💾 Full Forensic Report saved: data/factory_integrity_report.csv")

if __name__ == "__main__":
    import os
    target_file = "data/training_data_v4.parquet"
    if os.path.exists(target_file):
        generate_integrity_report(target_file)
    else:
        print(f"❌ Error: {target_file} not found.")