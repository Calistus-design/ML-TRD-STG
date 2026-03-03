# vault_audit.py
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib

# 1. Load Model & Data
MODEL_PATH = 'models/10m_call_v4.cbm'
DATA_PATH = 'data/vault_feb_2026.parquet'

print(f"🚀 LOADING VAULT AUDITOR...")
model = CatBoostClassifier()
model.load_model(MODEL_PATH)
df = pd.read_parquet(DATA_PATH)

# 2. Get Features & Target
# The model DNA ensures we use the correct 471-feature order
features = model.feature_names_
X = df[features]
y_true = df['target_10m_call_03'].values 

# 3. Predict
print(f"🔬 Analyzing {len(df):,} February Pullbacks...")
# We use Column 0 for the 0.3 ATR prediction
all_probs = model.predict_proba(X)
probs = all_probs[:, 0] # 0.3 ATR Win Probabilities

# 4. Generate Dashboard
results = []
# We sweep from 50% to 90% to find your 'Predator Entry Point'
for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    mask = probs >= t
    trade_count = np.sum(mask)
    
    if trade_count > 0:
        win_rate = np.mean(y_true[mask])
        # Calculate Trades Per Day (Assuming 20 trading days in Feb)
        yield_per_day = trade_count / 20.0
        
        results.append({
            'Threshold': f">{int(t*100)}% Sure", 
            'WinRate': f"{win_rate:.2%}", 
            'Total_Trades': f"{trade_count:,}",
            'Yield/Day': f"{yield_per_day:.2f}"
        })

print("\n" + "="*60)
print("🏆 FEBRUARY 2026 VAULT TRUTH DASHBOARD")
print("="*60)
print(pd.DataFrame(results).to_string(index=False))
print("="*60)

# 5. The $17M Feasibility Check
elite_trades = np.sum(probs >= 0.70)
if elite_trades / 20.0 >= 50:
    print("🚀 TARGET REACHED: High-Volume Predator identified.")
else:
    print("🔍 DIAGNOSIS: Sniper logic confirmed. Volume expansion required.")