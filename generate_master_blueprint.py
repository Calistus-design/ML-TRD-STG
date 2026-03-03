# generate_master_blueprint.py
import joblib
import os
from catboost import CatBoostClassifier

# --- CONFIGURATION ---
MODEL_PATH = 'models/10m_call_v4.cbm' # Path to your reference model
OUTPUT_PATH = 'models/v4_master_blueprint.joblib' # The NEW Source of Truth

def generate_blueprint():
    print("🔬 EXTRACTING MODEL DNA...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Reference model not found at {MODEL_PATH}")
        return

    # 1. Load the trained brain
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    
    # 2. Extract the exact feature names and order
    model_dna = model.feature_names_

    # 3. Save the DNA as the new Master Blueprint
    joblib.dump(model_dna, OUTPUT_PATH)

    print("-" * 60)
    print(f"✅ SUCCESS: {len(model_dna)}-feature Master Blueprint saved.")
    print(f"   -> File: {OUTPUT_PATH}")
    
    # Audit to confirm asset_id position
    if 'asset_id' in model_dna:
        idx = model_dna.index('asset_id')
        print(f"   -> 'asset_id' confirmed at Index: {idx}")
    print("-" * 60)

if __name__ == "__main__":
    generate_blueprint()