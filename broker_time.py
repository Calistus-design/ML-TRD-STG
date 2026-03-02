import MetaTrader5 as mt5
from datetime import datetime
import pytz

if mt5.initialize():
    terminal_info = mt5.terminal_info()
    # Get the last candle of EURUSD
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M1, 0, 1)
    broker_time = datetime.fromtimestamp(rates[0]['time'])
    utc_time = datetime.now(pytz.UTC)
    
    print(f"Broker Terminal Time: {broker_time}")
    print(f"Actual UTC Time    : {utc_time.strftime('%Y-%m-%d %H:%M:%S')}")
    mt5.shutdown()