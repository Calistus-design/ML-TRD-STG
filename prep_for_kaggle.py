# prep_for_kaggle.py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

def compress_textbook(
    input_path='data/training_data.csv',
    output_path='data/training_data.parquet',
    chunksize=500_000
):
    print("⏳ Converting CSV → Parquet (chunked, memory-safe)...")

    writer = None
    chunk_count = 0

    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        chunk_count += 1
        print(f"⚙️  Processing chunk {chunk_count}")

        # Downcast float64 → float32 (optional but recommended)
        float_cols = chunk.select_dtypes(include=['float64']).columns
        chunk[float_cols] = chunk[float_cols].astype('float32')

        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                table.schema,
                compression='snappy'
            )

        writer.write_table(table)

    if writer:
        writer.close()

    print(f"🚀 SUCCESS: {chunk_count} chunks written to ONE Parquet file → {output_path}")

if __name__ == "__main__":
    compress_textbook()
