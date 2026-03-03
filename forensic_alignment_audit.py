# forensic_alignment_audit.py 
# checks if the .joblib features match the models features in order
import joblib
import os
from catboost import CatBoostClassifier

# --- 1. CONFIGURATION ---
MODEL_PATH = 'models/10m_call_v4.cbm' # Path to your dummy/production model
JOBLIB_PATH = 'models/v4_elite_roster_ranked.joblib' # Path to your roster

def run_alignment_audit():
    print("🔬 STARTING FORENSIC ALIGNMENT AUDIT...")
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        return
    if not os.path.exists(JOBLIB_PATH):
        print(f"❌ ERROR: Joblib not found at {JOBLIB_PATH}")
        return

    # 2. LOAD BOTH SOURCES OF TRUTH
    # Load the roster from the joblib
    roster_list = joblib.load(JOBLIB_PATH)
    
    # Load the model and extract its internal feature DNA
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    model_dna = model.feature_names_

    # 3. STATISTICAL COMPARISON
    roster_len = len(roster_list)
    dna_len = len(model_dna)

    print("-" * 60)
    print(f"📊 DIMENSIONAL AUDIT")
    print(f"   Roster Feature Count: {roster_len}")
    print(f"   Model DNA Count    : {dna_len}")
    print("-" * 60)

    # 4. POSITIONAL INTEGRITY CHECK
    mismatches = []
    
    # Find the maximum range to check
    check_range = min(roster_len, dna_len)
    
    for i in range(check_range):
        roster_feat = roster_list[i]
        model_feat = model_dna[i]
        
        if roster_feat != model_feat:
            mismatches.append({
                'index': i,
                'roster': roster_feat,
                'model': model_feat
            })

    # 5. FINAL VERDICT
    if roster_len != dna_len:
        print("🚨 CRITICAL FAILURE: FEATURE COUNT MISMATCH!")
        print(f"   Difference: {abs(roster_len - dna_len)} features.")

    if not mismatches and roster_len == dna_len:
        print("✅ STATUS: PERFECT SYMMETRY.")
        print("   The local factory matches the model's brain bit-for-bit.")
    else:
        mismatch_count = len(mismatches)
        mismatch_pct = (mismatch_count / roster_len) * 100
        
        print(f"🚨 STATUS: ALIGNMENT CORRUPTED.")
        print(f"   Mismatched Indices: {mismatch_count} / {roster_len} ({mismatch_pct:.2f}%)")
        print("-" * 60)
        
        if mismatches:
            print("🛑 TOP 10 DIVERGENCE POINTS:")
            print(f"{'Index':<6} | {'Joblib (Factory)':<30} | {'Model DNA (Brain)':<30}")
            for m in mismatches[:10]:
                print(f"{m['index']:<6} | {m['roster']:<30} | {m['model']:<30}")
            
            print("\n💡 ARCHITECT'S NOTE:")
            if any('asset_id' in str(m.values()) for m in mismatches):
                print("   Found 'asset_id' shift. Check Stage 4 of production_factory.py.")
            else:
                print("   General shift detected. Ensure interaction forge order matches training.")
    print("-" * 60)

if __name__ == "__main__":
    run_alignment_audit()