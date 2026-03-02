import pandas as pd
import numpy as np
import os
import gc
import joblib

# Import the Specialized Departments
from src.features import base_engine, temporal, geometry, mass, anatomy, dynamics, oracle
from src.data_cleaner import DataCleaner

class ProductionFactory:
    """
    Alpha Factory V4 - Production Orchestrator.
    Converts live M1 data into the exact 471-feature vector expected by the Specialists.
    Enforces Roster Alignment and Physical Silencing.
    """

    def __init__(self, raw_dir='data/raw', roster_path='models/v4_elite_roster_ranked.joblib'):
        self.raw_dir = raw_dir
        self.cleaner = DataCleaner()
        
        # 1. Load the "Source of Truth" Blueprint
        if not os.path.exists(roster_path):
            raise FileNotFoundError(f"❌ Critical Error: Roster not found at {roster_path}")
        
        self.elite_roster = joblib.load(roster_path)
        print(f"✅ Production Factory Initialized. Roster Locked: {len(self.elite_roster)} features.")

        # Fixed Registry for asset_id consistency
        self.assets = [
            'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADCHF', 'CADJPY', 
            'CHFJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 
            'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD', 'GBPUSD', 
            'NZDCAD', 'NZDJPY', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'
        ]
        self.asset_map = {symbol: i for i, symbol in enumerate(self.assets)}

    def generate_production_vector_from_df(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Industrial High-Speed Forge. 
        Optimized for 471 features and <2s portfolio latency.
        """
        
        # --- STAGE 1: ATOMIC STABILITY (300 Rows) ---
        # We run the full history through the base modules so indicators 
        # like EMA_200, VWAP, and H1_Body_Avg are mathematically correct.
        df = base_engine.apply_base_physics(df)
        df = self.cleaner.calculate_news_sensor(df, symbol)
        
        # Identity & Anatomical Parents
        df['asset_id'] = np.int8(self.asset_map.get(symbol, -1))
        df = anatomy.apply_m1_anatomy(df) 
        df = temporal.apply_time_vectors(df)
        df = temporal.apply_rolling_lenses(df)
        df = anatomy.apply_m1_memory_lags(df)
        df = anatomy.apply_momentum_quality(df)
        
        # Structural Parents (Pivots & Walls)
        df = geometry.apply_horizontal_landmarks(df)
        df = geometry.apply_horizontal_multipliers(df)
        df = geometry.apply_trendline_engine(df)
        df = geometry.apply_trendline_integrity(df)
        
        # Force & Mass Parents
        df = mass.apply_volume_stack(df)
        df = dynamics.apply_market_dynamics(df)

        # --- STAGE 2: THE SURGICAL SPEED SLICE ---
        # We drop the history and keep only the last 2 rows.
        # This is the 'Speed Valve' that kills the 22-second lag.
        # We need 2 rows to calculate Deltas (Velocity) and D_ (Acceleration).
        df = df.tail(2).copy()

        # --- STAGE 3: THE MOLECULE FORGE (2 Rows Only) ---
        # Interactions are calculated only for the current candle.
        df = self._forge_elite_molecules(df, symbol)

        # --- STAGE 4: SILENCER & ALIGNMENT ---
        # 1. Reset memory to fix fragmentation warnings
        df = df.copy() 
        
        # 2. Identify and clip numeric features (Rational Boundary)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        protected = [c for c in numeric_cols if c == 'asset_id']
        clip_targets = [c for c in numeric_cols if c not in protected]
        df[clip_targets] = df[clip_targets].clip(lower=-100.0, upper=100.0)

        # 3. POSITIONAL HANDSHAKE: The Training Mirror
        # Ensure 'asset_id' is moved to Index 470 (The Very Last Column)
        # to match the Specialist's internal address book.
        features_no_id = [f for f in self.elite_roster if f != 'asset_id']
        final_production_cols = features_no_id + ['asset_id']
        
        # Filter and Re-order
        production_df = df[final_production_cols].copy()
        
        # 4. TYPE HARDENING
        # Cast features to float32 and asset_id to integer for CatBoost
        production_df[features_no_id] = production_df[features_no_id].astype('float32')
        production_df['asset_id'] = production_df['asset_id'].astype(int)
        
        # Return only the current candle for prediction
        return production_df.tail(1)
    
    
    def _forge_elite_molecules(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Recreates interactions using a dictionary buffer to prevent fragmentation."""
        forged_cols = {} # Dictionary to hold new features

        for f_name in self.elite_roster:
            if f_name in df.columns: continue
            
            try:
                if f_name.startswith('D_'):
                    parts = f_name[2:].split('_X_D_')
                    p_a, p_b = parts[0], parts[1]
                    forged_cols[f_name] = ((df[p_a] - df[p_a].shift(1)) * (df[p_b] - df[p_b].shift(1))).fillna(0)

                elif f_name.startswith('delta_'):
                    p_a = f_name[6:]
                    forged_cols[f_name] = (df[p_a] - df[p_a].shift(1)).fillna(0)

                elif 'RGM_' in f_name or '_div_' in f_name:
                    parts = f_name[4:].split('_div_') if f_name.startswith('RGM_') else f_name.split('_div_')
                    forged_cols[f_name] = df[parts[0]] / (df[parts[1]] + 1e-9)

                elif '_X_' in f_name:
                    parts = f_name.split('_X_')
                    forged_cols[f_name] = df[parts[0]] * df[parts[1]]
            
            except Exception as e:
                forged_cols[f_name] = 0.0

        # Join all new columns at once to prevent fragmentation
        new_features_df = pd.DataFrame(forged_cols, index=df.index)
        df = pd.concat([df, new_features_df], axis=1)
        return df