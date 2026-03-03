import pandas as pd
import numpy as np
import os
import gc
import joblib
import warnings

# Mute Performance Warnings for production speed
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from src.features import base_engine, temporal, geometry, mass, anatomy, dynamics
from src.data_cleaner import DataCleaner

class ProductionFactory:
    """
    Alpha Factory V4 - Industrial Production Engine.
    Hardened for 471-feature positional symmetry and <1s latency.
    Supports both Live Pulse and Batch Vault Audit modes.
    """

    def __init__(self, raw_dir='data/raw', roster_path='models/v4_master_blueprint.joblib'):
        self.raw_dir = raw_dir
        self.cleaner = DataCleaner()
        
        if not os.path.exists(roster_path):
            raise FileNotFoundError(f"❌ Blueprint not found at {roster_path}")
        
        self.final_roster = joblib.load(roster_path)

        # Asset Mapping for stationary identity
        self.assets = [
            'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADCHF', 'CADJPY', 
            'CHFJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 
            'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD', 'GBPUSD', 
            'NZDCAD', 'NZDJPY', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'
        ]
        self.asset_map = {symbol: i for i, symbol in enumerate(self.assets)}

        # Compile instructions once at startup
        self._compile_production_instructions()

    def _compile_production_instructions(self):
        """Creates a numerical plan based on the model's internal DNA."""
        raw_cols = ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', '<VOL>', '<SPREAD>']
        dummy_df = pd.DataFrame(np.random.randn(70, len(raw_cols)), columns=raw_cols)
        dummy_df.index = pd.date_range("2024-01-01", periods=70, freq='T')
        dummy_df['<SPREAD>'] = 1.0 
        
        dummy_df = base_engine.apply_base_physics(dummy_df)
        dummy_df['minutes_since_news'] = 1440.0
        dummy_df['asset_id'] = 0
        dummy_df = anatomy.apply_m1_anatomy(dummy_df)
        dummy_df = temporal.apply_time_vectors(dummy_df)
        dummy_df = temporal.apply_rolling_lenses(dummy_df)
        dummy_df = anatomy.apply_m1_memory_lags(dummy_df)
        dummy_df = anatomy.apply_momentum_quality(dummy_df)
        dummy_df = geometry.apply_horizontal_landmarks(dummy_df)
        dummy_df = geometry.apply_horizontal_multipliers(dummy_df)
        dummy_df = geometry.apply_trendline_engine(dummy_df)
        dummy_df = geometry.apply_trendline_integrity(dummy_df)
        dummy_df = mass.apply_volume_stack(dummy_df)
        dummy_df = dynamics.apply_market_dynamics(dummy_df)

        atom_names = dummy_df.columns.tolist()
        name_to_idx = {name: i for i, name in enumerate(atom_names)}
        
        self.atomic_copy_plan = [] 
        self.molecule_plan = []    

        for roster_idx, f_name in enumerate(self.final_roster):
            if f_name in name_to_idx:
                self.atomic_copy_plan.append((name_to_idx[f_name], roster_idx))
            else:
                try:
                    if f_name.startswith('D_'):
                        parts = f_name[2:].split('_X_D_')
                        self.molecule_plan.append((3, name_to_idx[parts[0]], name_to_idx[parts[1]], roster_idx))
                    elif f_name.startswith('delta_'):
                        parent = f_name[6:]
                        self.molecule_plan.append((2, name_to_idx[parent], -1, roster_idx))
                    elif 'RGM_' in f_name or '_div_' in f_name:
                        parts = f_name[4:].split('_div_') if f_name.startswith('RGM_') else f_name.split('_div_')
                        self.molecule_plan.append((1, name_to_idx[parts[0]], name_to_idx[parts[1]], roster_idx))
                    elif '_X_' in f_name:
                        parts = f_name.split('_X_')
                        self.molecule_plan.append((0, name_to_idx[parts[0]], name_to_idx[parts[1]], roster_idx))
                except KeyError: continue

        self.atomic_copy_plan = np.array(self.atomic_copy_plan, dtype=np.int32)
        self.molecule_plan = np.array(self.molecule_plan, dtype=np.int32)

    def generate_production_vector(self, filename: str, news_df: pd.DataFrame = None, is_live: bool = False) -> pd.DataFrame:
        """Universal entry point for Batch processing (Vault Forge)."""
        symbol = filename.split('_')[0]
        df = self.cleaner.load_csv(os.path.join(self.raw_dir, filename))
        return self._execute_forge_pipeline(df, symbol, news_df, is_live)

    def generate_production_vector_from_df(self, df: pd.DataFrame, symbol: str, news_df: pd.DataFrame = None, is_live: bool = True) -> pd.DataFrame:
        """Entry point for live HUD signals."""
        return self._execute_forge_pipeline(df, symbol, news_df, is_live)

    def _execute_forge_pipeline(self, df: pd.DataFrame, symbol: str, news_df: pd.DataFrame, is_live: bool) -> pd.DataFrame:
        """The Central Engine Room. Bit-for-bit identical for both Live and Batch."""
        
        # --- TIER 1: HIGH-HISTORY PHYSICS (300 Rows) ---
        df = base_engine.apply_base_physics(df)
        df = self.cleaner.calculate_news_sensor(df, symbol, news_df)
        df['asset_id'] = np.int8(self.asset_map.get(symbol, -1))
        df = anatomy.apply_m1_anatomy(df) 
        df = anatomy.apply_m1_memory_lags(df)
        df = anatomy.apply_momentum_quality(df)
        df = geometry.apply_horizontal_landmarks(df)
        df = geometry.apply_horizontal_multipliers(df)
        df = geometry.apply_trendline_engine(df)
        df = geometry.apply_trendline_integrity(df)

        # --- TIER 2: MID-HISTORY SLICE (Only if Live) ---
        if is_live:
            df = df.tail(61).copy()

        df = temporal.apply_time_vectors(df)
        df = temporal.apply_rolling_lenses(df)
        df = mass.apply_volume_stack(df)
        df = dynamics.apply_market_dynamics(df)

        # --- TIER 3: SURGICAL SLICE (Only if Live) ---
        if is_live:
            df = df.tail(2).copy()

        # --- TIER 4: NUMPY FORGE ---
        num_rows, num_features = df.shape[0], len(self.final_roster)
        master_array = np.zeros((num_rows, num_features), dtype=np.float32)
        atom_data = df.values
        for atom_idx, target_idx in self.atomic_copy_plan:
            master_array[:, target_idx] = atom_data[:, atom_idx]

        for op, a_idx, b_idx, target_idx in self.molecule_plan:
            if op == 0: master_array[:, target_idx] = master_array[:, a_idx] * master_array[:, b_idx]
            elif op == 1: master_array[:, target_idx] = master_array[:, a_idx] / (master_array[:, b_idx] + 1e-9)
            elif op == 2: 
                master_array[:, target_idx] = master_array[:, a_idx] - np.roll(master_array[:, a_idx], 1)
                master_array[0, target_idx] = 0
            elif op == 3: 
                d_a = master_array[:, a_idx] - np.roll(master_array[:, a_idx], 1)
                d_b = master_array[:, b_idx] - np.roll(master_array[:, b_idx], 1)
                master_array[:, target_idx] = d_a * d_b
                master_array[0, target_idx] = 0

        # --- STAGE 5: SILENCER & TYPE SAFETY ---
        master_array = np.clip(master_array, -100.0, 100.0)
        production_df = pd.DataFrame(master_array, columns=self.final_roster, index=df.index)
        production_df['asset_id'] = production_df['asset_id'].astype(int)

        # Final Return Selection
        return production_df.tail(1) if is_live else production_df