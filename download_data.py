# download_data.py

from src.live_data_loader import MT5DataLoader
from datetime import datetime

if __name__ == "__main__":
    """
    This is the main execution script for the data factory.
    Run this file from your terminal to download all historical data for a specific date range.
    """
    
    # --- DEFINE YOUR DESIRED DATE RANGE HERE ---
    # Start date of your historical data
    # start_date = datetime(2024, 1, 1)
    start_date = datetime(2026, 2, 1)
    
    # End date will be the current moment the script is run
    end_date = datetime.now()
    
    # --- EXECUTION ---
    loader = None
    try:
        loader = MT5DataLoader()
        
        # Pass the specific dates to the download process
        loader.run_download_process(date_from=start_date, date_to=end_date)
        
    except ConnectionError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please ensure your MT5 terminal is running and you are logged in.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if loader:
            loader.disconnect()