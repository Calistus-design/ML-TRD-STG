# audit_invariance.py (V4 Hardened)
import pandas as pd
import numpy as np
import os
import warnings
from src.feature_factory import FeatureFactory

# Suppress fragmentation warnings during audit
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def run_invariance_audit(symbol='EURUSD'):
    print(f"🕵️  STARTING INVARIANCE AUDIT: {symbol}")
    
    raw_path = f'data/raw/{symbol}_M1.csv'
    temp_short_path = f'data/raw/TEMP_SHORT_M1.csv'
    temp_long_path = f'data/raw/TEMP_LONG_M1.csv'
    
    if not os.path.exists(raw_path):
        print(f"❌ ERROR: {raw_path} not found.")
        return

    # 1. Prepare Test Data
    raw_df = pd.read_csv(raw_path)
    # We take 10k vs 20k to see if the extra 10k "Future" rows change the past
    raw_df.iloc[:50000].to_csv(temp_short_path, index=False)
    raw_df.iloc[:100000].to_csv(temp_long_path, index=False)
    
    factory = FeatureFactory(raw_dir='data/raw')
    
    try:
        # 2. Process through the Factory
        print("   -> Processing Short Segment (10k raw rows)...")
        df_short = factory.generate_asset_textbook('TEMP_SHORT_M1.csv')
        
        print("   -> Processing Long Segment (20k raw rows)...")
        df_long = factory.generate_asset_textbook('TEMP_LONG_M1.csv')
        
        # 3. DIRECT ALIGNMENT
        # Since 'time' was purged by the Oracle, we rely on row-order integrity.
        # We find how many rows from the short run exist and take that many from the long run.
        min_rows = len(df_short)
        if min_rows == 0:
            print("❌ FAIL: Short segment produced 0 candidates. Increase raw slice size.")
            return

        print(f"   -> Comparing first {min_rows:,} processed rows...")
        
        # 4. MATHEMATICAL COMPARISON
        # Select only feature columns (numeric)
        features = [c for c in df_short.columns if 'target' not in c and c != 'asset_id']
        
        short_matrix = df_short[features].values
        long_matrix = df_long[features].iloc[:min_rows].values
        
        # Calculate Variance
        diff_matrix = np.abs(short_matrix - long_matrix)
        # We replace NaNs with 0 for the comparison because NaN != NaN in numpy
        diff_matrix = np.nan_to_num(diff_matrix, nan=0.0)
        
        max_diff = np.max(diff_matrix)
        
        # 5. REPORT RESULTS
        print("-" * 60)
        print(f"📊 INVARIANCE RESULTS:")
        print(f"   Max Variance Detected: {max_diff:.10f}")
        
        # Threshold 1e-5 accounts for microscopic float32 precision shifts
        if max_diff < 1e-5:
            print("✅ PASS: Feature Invariance Confirmed. No Future Leakage.")
        else:
            print("❌ FAIL: Future Leakage Detected!")
            # Identify the column
            max_idx = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
            failed_feature = features[max_idx[1]]
            print(f"   🚨 Primary Offender: {failed_feature}")
            print(f"   Value in Short run: {short_matrix[max_idx]}")
            print(f"   Value in Long run : {long_matrix[max_idx]}")

    finally:
        # Cleanup
        if os.path.exists(temp_short_path): os.remove(temp_short_path)
        if os.path.exists(temp_long_path): os.remove(temp_long_path)
        print("-" * 60)

if __name__ == "__main__":
    run_invariance_audit()