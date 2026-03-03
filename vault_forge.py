# vault_forge.py
from src.production_factory import ProductionFactory
from src.features import base_engine, oracle 
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import time
import gc
import joblib

def run_vault_forge():
    # 1. INITIALIZATION
    print("🚀 INITIALIZING VAULT REFINERY: FEB 2026 AUDIT...")
    
    roster_path = 'models/v4_master_blueprint.joblib'
    master_dna = joblib.load(roster_path)
    factory = ProductionFactory(raw_dir='data/raw1', roster_path=roster_path)
    output_path = "data/vault_feb_2026.parquet"
    
    # Load news for the sensor
    news_df = pd.read_csv('data/news_calendar.csv')
    news_df['time'] = pd.to_datetime(news_df['time']).dt.tz_localize('UTC')

    total_rows = 0
    start_time = time.time()
    writer = None
    raw_files = [f for f in os.listdir('data/raw1') if f.endswith('.csv')]

    # 2. THE ASSET LOOP
    for i, filename in enumerate(raw_files, start=1):
        symbol = filename.split('_')[0]
        print(f"\n[{i}/{len(raw_files)}] 🛠️  Processing {symbol}...")

        try:
            # --- STAGE 1: INGESTION (Continuous Raw) ---
            raw_df = factory.cleaner.load_csv(os.path.join('data/raw1', filename))
            print(f"      -> Ingested: {len(raw_df):,} continuous rows.")

            # --- STAGE 2: FULL FORGE (Physics Integrity) ---
            # IMPORTANT: We calculate features on the CONTINUOUS data.
            # This ensures EMAs and Rolling Lenses have perfect inertia.
            # is_live=False ensures we process all rows, not just the tail.
            forged_df = factory._execute_forge_pipeline(raw_df, symbol, news_df, is_live=False)

            # --- STAGE 3: ATTACHING THE TRUTH (Targets) ---
            # Answers are calculated on raw history, then joined
            labeled_raw = oracle.apply_oracle_labels(raw_df)
            target_col = 'target_10m_call_03'
            
            # This is our 'Master Data Store' for this asset
            asset_matrix = forged_df.join(labeled_raw[[target_col]], how='inner')

            # --- STAGE 4: THE VAULT REFINERY (Surgical Slicing) ---
            # Now that the math is finished and stable, we decide which rows to keep.
            # We use the raw_df (which has <TICKVOL> and SMA_10) to find valid indices.
            
            # A. Prepare raw data for index discovery (calc ATR/SMA10)
            refinery_df = base_engine.apply_base_physics(raw_df.copy())
            
            # B. Apply Cleaners to find valid time-stamps
            refinery_df = factory.cleaner.remove_rollover_and_weekends(refinery_df)
            refinery_df = factory.cleaner.remove_noise(refinery_df)
            refinery_df = factory.cleaner.remove_news_events(refinery_df, symbol)
            
            # C. Apply Funnel to find pullback moments
            refinery_df = factory.cleaner.identify_candidates(refinery_df)
            
            # D. THE INTERSECTION: Keep only forged rows that passed the refinery
            final_df = asset_matrix.loc[asset_matrix.index.intersection(refinery_df.index)]

            if final_df.empty:
                print(f"      ⚠️ No tradeable pullbacks found for {symbol}.")
                continue

            # --- STAGE 5: SYMMETRY & BOUNDARY AUDIT ---
            if list(final_df.columns[:471]) == master_dna:
                print(f"      ✅ SYMMETRY: Verified.")
            else:
                print(f"      🚨 FATAL: Positional Drift detected!")
                continue

            f_max, f_min = final_df.iloc[:, :471].max().max(), final_df.iloc[:, :471].min().min()
            print(f"      ✅ BOUNDARY AUDIT: Max {f_max:.1f} | Min {f_min:.1f}")

            # --- STAGE 6: STREAM TO DISK ---
            table = pa.Table.from_pandas(final_df, preserve_index=True)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
            writer.write_table(table)

            total_rows += len(final_df)
            print(f"      ✅ ASSET COMPLETE: {len(final_df):,} valid samples saved.")

            del final_df, asset_matrix, forged_df, raw_df, refinery_df, labeled_raw
            gc.collect()

        except Exception as e:
            print(f"      ❌ FAILED {symbol}: {e}")

    # 3. FINALIZE
    if writer: writer.close()
    
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"✅ VAULT FORGE COMPLETE")
    print(f"📊 Final Test Matrix: {output_path}")
    print(f"📊 Total Samples    : {total_rows:,}")
    print(f"⏱️  Forge Time       : {elapsed:.2f}s")
    print("="*60)

if __name__ == "__main__":
    run_vault_forge()