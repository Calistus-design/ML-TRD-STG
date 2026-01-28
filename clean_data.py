# clean_data.py

"""
Alpha Factory Refinery Runner.
Orchestrates the multi-asset cleaning process, transforming raw exports 
into verified, 'Gold Standard' datasets for Machine Learning.
"""

from src.data_cleaner import DataCleaner
import os

if __name__ == "__main__":
    # Initialize the Refinery Engine
    cleaner = DataCleaner(raw_dir='data/raw', clean_dir='data/cleaned')
    
    # Automatically discover all raw CSV files exported from MT5
    raw_files = [f for f in os.listdir(cleaner.raw_dir) if f.endswith('.csv')]
    
    if not raw_files:
        print("❌ ERROR: No raw CSV files found. Run download_data.py first.")
    else:
        print("="*70)
        print(f"🚀 ALPHA FACTORY: COMMENCING REFINERY FOR {len(raw_files)} ASSETS")
        print("="*70)

        for filename in raw_files:
            # Parse symbol from filename (expects format SYMBOL_M1.csv)
            symbol = filename.split('_')[0]
            raw_path = os.path.join(cleaner.raw_dir, filename)
            
            print(f"REFINING ASSET: {symbol}")
            
            try:
                # --- STEP 1: LOAD ---
                print(f"    -> Ingesting raw data...")
                df = cleaner.load_csv(raw_path)
                orig_len = len(df)
                
                # --- STEP 2: TIME SCRUBBING ---
                df = cleaner.remove_rollover_and_weekends(df)
                
                # --- STEP 3: NOISE FILTERING ---
                df = cleaner.remove_noise(df)
                
                # --- STEP 4: NEWS BLACKOUT ---
                df = cleaner.remove_news_events(df, symbol)
                
                # --- STEP 5: FINALIZATION ---
                print(f"    -> Exporting Gold-Standard CSV...")
                clean_path = os.path.join(cleaner.clean_dir, filename)
                df.to_csv(clean_path)
                
                final_len = len(df)
                removed_total = orig_len - final_len
                retention_rate = (final_len / orig_len) * 100
                
                print(f"✅ REFINERY SUCCESS: {symbol}")
                print(f"       Rows Removed: {removed_total}")
                print(f"       Data Quality Score (Retention): {retention_rate:.2f}%")
                print("-" * 50)
                
            except Exception as e:
                print(f"    -> ❌ CRITICAL FAILURE on {symbol}: {e}\n")

        print("="*70)
        print("✅ DATA REFINERY COMPLETE: ALL ASSETS READY FOR ALPHA PRODUCTION.")
        print("="*70)