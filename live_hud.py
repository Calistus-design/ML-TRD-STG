# live_hud.py - Alpha Factory V4 High-Performance Multiprocessing HUD
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import warnings
import time
import joblib
from concurrent.futures import ProcessPoolExecutor
from catboost import CatBoostClassifier
from src.production_factory import ProductionFactory

# Mute Performance Warnings to keep the dashboard clean
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# --- CONFIGURATION ---
ASSETS = [
    'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADCHF', 'CADJPY', 
    'CHFJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 
    'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD', 'GBPUSD', 
    'NZDCAD', 'NZDJPY', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'
]

SPECIALISTS = ['10m_call'] 
MODELS_DIR = 'models/'
THRESHOLD = 0.01  # Test Threshold (Set to 0.85 for production)

# --- GLOBAL WORKER STATE ---
# This variable lives in the memory of the child process
worker_factory = None

def forge_worker(symbol, news_df):
    """
    Surgical worker function. 
    Reuses the factory across pulses to achieve sub-second speeds.
    """
    global worker_factory
    
    # 1. LAZY INITIALIZATION: Runs only once per CPU core session
    if worker_factory is None:
        if not mt5.initialize():
            return None
        worker_factory = ProductionFactory()

    # 2. DATA INGESTION
    # Fetch 300 candles for indicator stability (EMA 200)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 300)
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={
        'open':'<OPEN>', 'high':'<HIGH>', 'low':'<LOW>', 
        'close':'<CLOSE>', 'tick_volume':'<TICKVOL>', 'spread':'<SPREAD>'
    }, inplace=True)
    
    # 3. PRODUCTION FORGE
    # is_live=True (default) ensures we only calculate the 1-row result
    try:
        vector = worker_factory.generate_production_vector_from_df(df, symbol, news_df=news_df)
        return (symbol, vector)
    except:
        return None

class AlphaHUD:
    def __init__(self):
        print("🚀 BOOTING ALPHA FACTORY V4 PREDATOR HUD...")
        
        if not mt5.initialize():
            print("❌ MT5 Init Failed")
            quit()
        
        # 1. Cache news in RAM to avoid redundant Disk I/O
        print("  -> Loading Global News Calendar...")
        self.news_df = pd.read_csv('data/news_calendar.csv')
        self.news_df['time'] = pd.to_datetime(self.news_df['time']).dt.tz_localize('UTC')

        # 2. Load the Master Blueprint for the Pre-Flight Audit
        self.master_blueprint = joblib.load('models/v4_master_blueprint.joblib')
        self.factory_auditor = ProductionFactory()
        
        # 3. Load specialists into the Main Process for high-speed inference
        self.brains = {}
        for s in SPECIALISTS:
            path = os.path.join(MODELS_DIR, f"{s}_v4.cbm")
            if os.path.exists(path):
                model = CatBoostClassifier()
                model.load_model(path)
                self.brains[s] = model
                print(f"   ✅ Specialist Loaded: {s.upper()}")
        
        # 4. Run Symmetry Audit before opening the gate
        self._run_preflight_audit()

        # 5. INITIALIZE PERSISTENT EXECUTOR
        # We keep these 8 processes alive to avoid the 'NoneType' and 'Lag' issues
        self.executor = ProcessPoolExecutor(max_workers=8)
        print("🟢 HEARTBEAT ACTIVE. Monitoring 58s pulse...")

    def _run_preflight_audit(self):
        """Mandatory check: Does the factory match the Brain's Address Book?"""
        print("\n🔬 CONDUCTING PRE-FLIGHT SYMMETRY AUDIT...")
        test_symbol = 'EURUSD'
        # Request 300 rows for the test
        rates = mt5.copy_rates_from_pos(test_symbol, mt5.TIMEFRAME_M1, 0, 300)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df.rename(columns={'open':'<OPEN>', 'high':'<HIGH>', 'low':'<LOW>', 'close':'<CLOSE>', 'tick_volume':'<TICKVOL>', 'spread':'<SPREAD>'}, inplace=True)
        
        production_df = self.factory_auditor.generate_production_vector_from_df(df, test_symbol, self.news_df)
        
        if list(production_df.columns) == self.master_blueprint:
            print("  ✅ STATUS: PERFECT SYMMETRY. Deployment Authorized.")
        else:
            print("  🚨 CRITICAL FAILURE: POSITIONAL MISMATCH.")
            quit()

    def pulse(self):
        """The 60-second Alpha Harvest."""
        start_time = time.time()
        print(f"\n🔔 PULSE [{time.strftime('%H:%M:%S')}]")

        # PARALLEL FORGE: Reuses persistent processes
        futures = [self.executor.submit(forge_worker, s, self.news_df) for s in ASSETS]
        results = [f.result() for f in futures if f.result() is not None]

        if not results: return

        # BATCH ASSEMBLY
        symbols = [r[0] for r in results]
        portfolio_matrix = pd.concat([r[1] for r in results], axis=0)

        # BATCH INFERENCE: One call for all assets
        for name, model in self.brains.items():
            all_probs = model.predict_proba(portfolio_matrix)
            for i, symbol in enumerate(symbols):
                # win_prob for 0.3 ATR target
                win_prob = all_probs[i][0]
                if win_prob >= THRESHOLD:
                    self.alert(symbol, name, win_prob)

        print(f"✅ PULSE COMPLETE | Scanned: {len(results)} | Speed: {time.time() - start_time:.2f}s")

    def alert(self, symbol, s_type, prob):
        color = "\033[92m" if "call" in s_type else "\033[91m"
        print(f"  🎯 {color}{symbol:<7} | {s_type.upper():<8} | CONF: {prob:.2%}\033[0m")

    def run(self):
        while True:
            # Pulse logic triggers at second 58
            if time.localtime().tm_sec == 58:
                self.pulse()
                time.sleep(2) # Prevent double trigger
            time.sleep(0.1)

if __name__ == "__main__":
    # Windows requirement for multiprocessing
    hud = AlphaHUD()
    hud.run()