# src/data_validator.py
import numpy as np
import pandas as pd

def run_post_refinery_audit(df: pd.DataFrame, asset_name: str):
    """
    Refinery V3 Sentry: Minimal Noise, Maximum Integrity.
    Only alerts on true mathematical anomalies.
    """
    if df.empty:
        print(f"❌ CRITICAL ERROR: {asset_name} is EMPTY after refinery!")
        return 1

    issues_found = 0
    check_cols = [c for c in df.columns if c not in ['time', 'asset_id']]
    
    # 1. THE "PRICE LEAK" EXORCIST (Non-Stationarity)
    # We whitelist features that naturally have high values
    whitelist = ['rsi', 'age', 'since_news', 'smoothness', 'stoch', 'tax', 
                 'power', 'mass', 'pips', 'slope', 'rank', 'efficiency', 
                 'intensity', 'ratio', 'acceleration', 'velocity']
    
    potential_leaks = []
    for col in check_cols:
        if not any(w in col.lower() for w in whitelist):
            # Price Leaks have very specific means (e.g., 1.08 or 145.0)
            # We only alert if it's not whitelisted AND has a huge mean
            if abs(df[col].mean()) > 150.0:
                potential_leaks.append(col)

    if potential_leaks:
        print(f"   🚨 LEAK ALERT: {len(potential_leaks)} features have non-stationary means: {potential_leaks[:5]}...")
        issues_found += 1

    # 2. THE "BROKEN PHYSICS" AUDIT (Outliers)
    # Since we CLIP at 100, anything up to 100 is "Legal".
    # We only alert if distance features escape the +/- 25 zone.
    dist_cols = [c for c in df.columns if 'dist_to_' in c and 'norm' in c]
    if dist_cols:
        outlier_mask = (df[dist_cols].abs() > 26.0).any() # Allow 1.0 slack for math
        bad_physics = outlier_mask[outlier_mask == True].index.tolist()
        if bad_physics:
            print(f"   ⚠️  PHYSICS BREACH: {len(bad_physics)} distance sensors > 25 ATR: {bad_physics[:5]}...")
            issues_found += 1

    # 3. THE "INFINITE ENERGY" AUDIT (Math Glitches)
    # We check if anything actually hit Infinity (which should be impossible after V3)
    if np.isinf(df[check_cols].values).any():
        print(f"   🚨 MATH ALERT: Infinity (Inf) detected in the matrix!")
        issues_found += 1

    # 4. DATA INTEGRITY (NaNs)
    # A healthy high-frequency row should have < 5% NaNs
    total_cells = df[check_cols].size
    total_nans = df[check_cols].isna().sum().sum()
    nan_pct = (total_nans / total_cells) * 100
    
    if nan_pct > 5.0:
        print(f"   ⚠️  INTEGRITY ALERT: High NaN Density ({nan_pct:.2f}%).")
        issues_found += 1

    if issues_found == 0:
        # Final confirmation if everything is perfect
        rows_str = f"{len(df):,}"
        print(f"   ✅ {asset_name:<7} | Rows: {rows_str:>9} | Physics: Verified | Status: GOLDEN")
    
    return issues_found