# live_hud.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
from catboost import CatBoostClassifier
from src.production_factory import ProductionFactory

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# 1. SETTINGS
ASSETS = [
    'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADCHF', 'CADJPY', 
    'CHFJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 
    'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD', 'GBPUSD', 
    'NZDCAD', 'NZDJPY', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'
]

# The HUD will try to load all 8, but will only use what you have in /models
SPECIALISTS = ['3m_call', '3m_put', '4m_call', '4m_put', '5m_call', '5m_put', '10m_call', '10m_put']
MODELS_DIR = 'models/'
THRESHOLD = 0.1 # Alert when probability > 85%

class AlphaHUD:
    def __init__(self):
        print("🚀 BOOTING ALPHA FACTORY V4 HUD...")
        if not mt5.initialize():
            print("❌ MT5 Init Failed")
            quit()
        
        self.factory = ProductionFactory(roster_path='models/v4_elite_roster_ranked.joblib')
        
        # Load available models
        self.brains = {}
        for s in SPECIALISTS:
            path = os.path.join(MODELS_DIR, f"{s}_v4.cbm")
            if os.path.exists(path):
                model = CatBoostClassifier()
                model.load_model(path)
                self.brains[s] = model
                print(f"   ✅ Specialist Loaded: {s.upper()}")
            else:
                print(f"   ❌ Specialist Missing: {s.upper()} (Skipping)")

    def fetch_m1_data(self, symbol):
        """Fetches last 300 candles to satisfy the EMA 200 and forge."""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 300)
        if rates is None or len(rates) == 0: return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df.rename(columns={'open':'<OPEN>', 'high':'<HIGH>', 'low':'<LOW>', 'close':'<CLOSE>', 'tick_volume':'<TICKVOL>', 'spread':'<SPREAD>'}, inplace=True)
        return df

    def pulse(self):
        """Scans all assets and prints high-confidence signals."""
        print(f"\n🔔 HEARTBEAT [{time.strftime('%H:%M:%S')}] | Scanning {len(ASSETS)} assets...")
        start_time = time.time()
        
        for symbol in ASSETS:
            raw_df = self.fetch_m1_data(symbol)
            if raw_df is None: continue

            # Create 471 Features for the CURRENT candle (last row)
            try:
                # 1. Forge the vector (471 features, last one is int)
                production_matrix = self.factory.generate_production_vector_from_df(raw_df, symbol)
                
                # 2. Slice the last row (The current candle)
                # We do NOT use .astype or .fillna here to avoid changing the types
                current_candle = production_matrix.tail(1)
                
            except Exception as e:
                print(f"   ⚠️ Forge Fail [{symbol}]: {e}")
                continue

            # Check all loaded specialists
            for name, model in self.brains.items():
                # 1. Get raw probabilities for all 4 tiers
                # MultiLogloss returns a list: [array(tier_0.3), array(tier_0.75), ...]
                all_probs = model.predict_proba(current_candle)
                
                # 2. Extract the WIN probability for the 0.3 ATR Tier (Index 0)
                # all_probs[0] = 0.3 ATR target
                # all_probs[0][0][1] = The probability of '1' (Win) for the current candle
                win_prob = all_probs[0][0]
                
                if win_prob >= THRESHOLD:
                    self.alert(symbol, name, win_prob)


        duration = time.time() - start_time
        print(f"✅ SCAN COMPLETE | Time: {duration:.2f}s")

    def alert(self, symbol, s_type, prob):
        color = "\033[92m" if "call" in s_type else "\033[91m"
        print(f"  🎯 {color}{symbol:<7} | {s_type.upper():<8} | CONF: {prob:.2%} | ACTION: NOW\033[0m")

    def run(self):
        print("🟢 HUD ACTIVE. Monitoring 58s pulse...")
        while True:
            # Trigger at the 58th second of the minute
            if time.localtime().tm_sec == 58:
                self.pulse()
                time.sleep(2) # Prevent double trigger
            time.sleep(0.5)

if __name__ == "__main__":
    hud = AlphaHUD()
    hud.run()