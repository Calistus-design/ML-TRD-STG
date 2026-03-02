import pandas as pd
import numpy as np
import os
import gc
import joblib

# Import the Specialized Departments
from src.features import base_engine, temporal, geometry, mass, anatomy, dynamics, oracle
from src.data_cleaner import DataCleaner

class FeatureFactory:
    """
    The Alpha Factory V4 Orchestrator.
    Coordinates the multi-stage assembly line to convert 20M candles into 
    a 15M-row high-purity, stationary, multi-target feature matrix.
    """

    def __init__(self, raw_dir='data/raw', cleaned_dir='data/cleaned', output_dir='data/features'):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.cleaner = DataCleaner() # Initialize the Refinery
        
        # Fixed Registry: EURUSD is ALWAYS 0, GBPUSD is 1... (Stationary Identity)
        self.assets = [
            'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADCHF', 'CADJPY', 
            'CHFJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 
            'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD', 'GBPUSD', 
            'NZDCAD', 'NZDJPY', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'
        ]
        self.asset_map = {symbol: i for i, symbol in enumerate(self.assets)}
        self.asset_list = [f"{s}_M1.csv" for s in self.assets]
        
        # Save Registry for the Live HUD
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.asset_map, 'models/asset_mapping.joblib')

        # src/feature_factory.py [SYNCED VERSION]
    def generate_asset_textbook(self, filename: str) -> pd.DataFrame:
        symbol = filename.split('_')[0]
        raw_path = os.path.join(self.raw_dir, filename)
        
        # --- STAGE 1: INGESTION ---
        # Load RAW data (Continuous)
        df = self.cleaner.load_csv(raw_path)
        print(f"    -> [Stage 1] Loaded {len(df):,} raw candles for {symbol}.")

        # --- STAGE 2: CONTINUOUS PHYSICS ---
        # Calculate Base Indicators (ATR, RSI, VWAP) on unbroken data
        df = base_engine.apply_base_physics(df)

        # --- STAGE 5: DEPARTMENTAL FORGE [MASTER SYNCED SEQUENCE] ---
        # We run this BEFORE cleaning so features see the full history.
        
        # A. Identity & Basic Anatomy
        df['asset_id'] = np.int8(self.asset_map.get(symbol, -1))
        df = anatomy.apply_m1_anatomy(df) 
        
        # B. Temporal Department (Clock & Lenses)
        df = temporal.apply_time_vectors(df)
        df = temporal.apply_rolling_lenses(df)
        
        # C. Anatomy Part 2 (Lags)
        df = anatomy.apply_m1_memory_lags(df)
        df = anatomy.apply_momentum_quality(df)
        
        # D. Geometry Department (Space)
        df = geometry.apply_horizontal_landmarks(df)
        df = geometry.apply_horizontal_multipliers(df)
        df = geometry.apply_trendline_engine(df)
        df = geometry.apply_trendline_integrity(df)
        
        # E. Mass Department (Volume/Energy) <--- RE-INSERTED THIS
        df = mass.apply_volume_stack(df)

        # F. Dynamics Department (Force/Motion)
        df = dynamics.apply_market_dynamics(df)
        
        print(f"    -> [Stage 5] Features Engineered. Matrix Size: {df.shape[1]} cols.")

        # --- STAGE 6: THE ORACLE ---
        # Calculate Future Targets on continuous data
        df = oracle.apply_oracle_labels(df)

        # --- STAGE 3: THE REFINERY (Late-Stage Cleaning) ---
        # Now we remove the toxic rows (News/Rollover/Bad Ticks)
        df = self.cleaner.remove_rollover_and_weekends(df)
        df = self.cleaner.remove_noise(df)
        df = self.cleaner.remove_news_events(df, symbol)

        # --- STAGE 4: THE FUNNEL ---
        # Extract only the 0.5 ATR Candidates
        df = self.cleaner.identify_candidates(df)
        
        if df.empty:
            print(f"    ⚠️ Skipping {symbol}: No candidates found.")
            return df

        # --- STAGE 7: HYGIENE & SHRINKING ---
        # 1. Purge Scaffolding
        df = oracle.perform_scaffolding_purge(df)
        
        # 2. Final Type Safety (RAM Protection)
        feat_cols = [c for c in df.columns if 'target_' not in c and c != 'asset_id']
        df[feat_cols] = df[feat_cols].astype('float32')
        
        targ_cols = [c for c in df.columns if 'target_' in c]
        df[targ_cols] = df[targ_cols].astype('int8')

        gc.collect() 
        return df