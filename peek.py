import pandas as pd
df = pd.read_parquet("data/EURUSD_M1_v4.parquet")
print(df.head(20)) # Look at the actual numbers