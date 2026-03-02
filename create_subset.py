# create_subset.py
from src.feature_factory import FeatureFactory
import pandas as pd
import os
import time
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

if __name__ == "__main__":
    # Define your "Dev Squad"
    subset_assets = ['EURUSD', 'AUDJPY', 'GBPUSD', 'EURGBP']
    factory = FeatureFactory()
    output_file = "data/training_data_4.csv"

    first_write = True
    total_rows = 0
    start_time = time.time()

    print("=" * 60)
    print(f"🚀 CREATING MINI-TEXTBOOK FOR {len(subset_assets)} ASSETS")
    print("=" * 60)

    total_assets = len(subset_assets)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for i, symbol in enumerate(subset_assets, start=1):
        filename = f"{symbol}_M1.csv"
        path = os.path.join("data/cleaned", filename)

        if os.path.exists(path):
            asset_start = time.time()
            df = factory.generate_asset_textbook(filename)
            rows = len(df)

            # Stream-write to CSV in chunks
            df.to_csv(
                output_file,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
                chunksize=100_000
            )
            first_write = False
            total_rows += rows
            asset_time = time.time() - asset_start

            print(f"\n[{i}/{total_assets}] Processing {symbol}...")
            print(f"    ✔ Rows written: {rows:,}")
            print(f"    ✔ Asset time : {asset_time:.2f}s")
            print(f"    ✔ TOTAL ROWS SO FAR: {total_rows:,}")

            # Free memory explicitly
            del df

        else:
            print(f"\n[{i}/{total_assets}] ❌ Missing file: {filename}")

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"✅ MINI-TEXTBOOK COMPLETE: {output_file}")
    print(f"📊 TOTAL ROWS: {total_rows:,}")
    print(f"⏱️ TOTAL RUN TIME: {total_time/60:.2f} minutes")
    print("=" * 60)
