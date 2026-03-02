# generate_training_data.py (Final V4 Industrial Edition)

from src.feature_factory import FeatureFactory
from src.data_validator import run_post_refinery_audit
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import time
import gc
import warnings

# Suppress performance warnings from pandas during high-dimensional assembly
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def run_industrial_forge():
    # 1. Initialize the Factory
    factory = FeatureFactory(raw_dir='data/raw')
    output_path = "data/training_data_v4.parquet"
    
    total_rows = 0
    start_time = time.time()
    writer = None  # The Parquet streaming engine

    print("=" * 60)
    print(f"🚀 ALPHA FORGE V4: GENERATING PARQUET FROM {len(factory.asset_list)} ASSETS")
    print(f"   Storage Standard: float32 (Features) | int8 (Targets)")
    print("=" * 60)

    # 2. The Assembly Loop
    for i, filename in enumerate(factory.asset_list, start=1):
        symbol = filename.split('_')[0]
        
        # We start the asset-specific timer
        asset_start = time.time()
        print(f"\n[{i}/{len(factory.asset_list)}] Processing {symbol}...")

        # --- THE WORK: Execute the Assembly Line ---
        # This calls the modular departments we built (temporal, geometry, etc.)
        asset_df = factory.generate_asset_textbook(filename)

        # ONE LINE calls the hardened logic for both scripts
        asset_df = factory.cleaner.harden_physics(asset_df) 

        # Validating the data is clean
        run_post_refinery_audit(asset_df, symbol)
        
        if asset_df.empty:
            print(f"    ⚠️  Skipping {symbol}: No candidates found in the 0.5 ATR zone.")
            continue

        # --- INSURANCE: Final Memory Downcasting ---
        # Even though FeatureFactory does this, we run a final check here 
        # to ensure the Parquet writer stays within the 16GB laptop limit.
        float_cols = asset_df.select_dtypes(include=['float64']).columns
        if not float_cols.empty:
            asset_df[float_cols] = asset_df[float_cols].astype('float32')

        target_cols = [c for c in asset_df.columns if 'target_' in c]
        if len(target_cols) > 0:
            asset_df[target_cols] = asset_df[target_cols].astype('int8')

        # --- THE STREAM: Direct Write to Disk ---
        # Convert the shrunken DataFrame to a PyArrow Table
        table = pa.Table.from_pandas(asset_df, preserve_index=False)

        # Initialize the writer on the first successful asset
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')

        writer.write_table(table)

        # --- LOGGING ---
        rows = len(asset_df)
        total_rows += rows
        asset_time = time.time() - asset_start
        
        # We use your exact preferred print statements
        print(f"    ✔ Rows added to Parquet: {rows:,}")
        print(f"    ✔ Asset process time  : {asset_time:.2f}s")
        print(f"    ✔ TOTAL ROWS CONSOLIDATED: {total_rows:,}")

        # --- HYGIENE: The Kill Shot ---
        # We physically erase the local asset data before moving to the next pair.
        # This is the ONLY way to process 15M rows on a 16GB laptop.
        del asset_df
        gc.collect()

    # --- 3. FINALIZE ---
    if writer:
        writer.close()

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"✅ MISSION COMPLETE: {output_path}")
    print(f"📊 FINAL TEXTBOOK SIZE: {total_rows:,} high-purity rows")
    print(f"⏱️  TOTAL FORGE TIME: {total_time/60:.2f} minutes")
    print("=" * 60)

if __name__ == "__main__":
    run_industrial_forge()