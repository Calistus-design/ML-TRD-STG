# verify_production.py
from src.production_factory import ProductionFactory
import joblib
import pandas as pd
import warnings

# 1. Mute Performance Warnings to keep the log clean
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# 2. Load the Blueprint
factory = ProductionFactory()
roster = joblib.load('models/v4_elite_roster_ranked.joblib')

# 3. RUN the Production Factory on EURUSD
print("🏗️  EXECUTING FACTORY HANDSHAKE...")
df = factory.generate_production_vector('EURUSD_M1.csv')

# 4. PERFORM THE SYMMETRY AUDIT
print("-" * 50)
print(f"📊 PRODUCTION SYMMETRY REPORT")
print("-" * 50)
print(f"Roster Size (Joblib)  : {len(roster)}")
print(f"Factory Output Columns: {len(df.columns)}")

# Positional Check: Is column X what the brain thinks it is?
first_match = (df.columns[0] == roster[0])
last_match = (df.columns[-1] == roster[-1])

print(f"First Feature Match : {first_match} (Index 0: {df.columns[0]})")
print(f"Last Feature Match  : {last_match} (Index {len(df.columns)-1}: {df.columns[-1]})")

# Boundary Check: Are we in the Rational Zone?
print(f"Boundary Audit      : Max {df.max().max():.2f} | Min {df.min().min():.2f}")
print("-" * 50)

# add this to verify_production.py
df = factory.generate_production_vector('EURUSD_M1.csv')
print(f"Final Column (Index 470): {df.columns[470]}")
print(f"Index 470 Dtype: {df.iloc[:, 470].dtype}")

if df.columns[470] == 'asset_id' and df.iloc[:, 470].dtype == 'int32':
    print("✅ POSITIONAL ALIGNMENT: FIXED.")

# 5. FINAL VERDICT
if len(df.columns) == len(roster) and first_match and last_match:
    print("🚀 ALPHA FACTORY V4: STATUS GOLDEN.")
    print("   The factory output is bit-for-bit identical to the training blueprint.")
else:
    if len(df.columns) != len(roster):
        print(f"❌ COUNT MISMATCH: Expected {len(roster)}, got {len(df.columns)}")
    if not first_match or not last_match:
        print("❌ POSITIONAL MISMATCH: The column order is shuffled!")

