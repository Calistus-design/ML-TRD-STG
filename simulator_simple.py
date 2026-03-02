import pandas as pd
import random

def run_interactive_simulation():
    print("\n" + "="*60)
    print("🚀 ALPHA FACTORY V1: FULL HISTORY SIMULATOR")
    print("="*60)

    try:
        # --- USER INPUTS ---
        initial_capital = float(input("Enter Starting Capital ($): "))
        win_rate_input = float(input("Enter Win Rate (%): ")) / 100
        num_trades = int(input("Enter Number of Trades: "))
        risk_pct = float(input("Enter Anti-Ruin/Risk per Trade (%): ")) / 100
        profit_pct = float(input("Enter Profit/Payout per Win (%): ")) / 100
        
        print("\n" + "-"*60)
        print(f"STRESS TEST: {num_trades} trades at {win_rate_input*100}% WR and {risk_pct*100}% Risk")
        print("-"*60)

        # 1. Generate outcomes based on exact win rate
        num_wins = int(num_trades * win_rate_input)
        num_losses = num_trades - num_wins
        outcomes = [1] * num_wins + [0] * num_losses
        
        # 2. Randomize the sequence (The "Luck" Factor)
        random.shuffle(outcomes)

        balance = initial_capital
        ath = initial_capital
        history = []

        # 3. Execution Loop (Dynamic Compounding)
        for i, win in enumerate(outcomes):
            start_bal = balance
            
            # Recalculate 2% based on CURRENT equity
            risk_amount = balance * risk_pct
            
            if win == 1:
                profit = risk_amount * profit_pct
                balance += profit
                result_str = "WIN " # Added space for alignment
            else:
                loss = risk_amount
                balance -= loss
                result_str = "LOSS"
            
            # Track peak and drawdown
            ath = max(ath, balance)
            dd = (ath - balance) / ath * 100
            
            history.append({
                'Trade': i + 1,
                'Result': result_str,
                'Risk_Amt': risk_amount,
                'Balance': balance,
                'Drawdown': dd
            })

        # --- DATA PRESENTATION (FULL HISTORY) ---
        df = pd.DataFrame(history)
        
        print("\nFULL TRADE-BY-TRADE AUDIT:")
        print("-" * 60)
        # Using to_string() prints the entire DataFrame without truncating
        print(df[['Trade', 'Result', 'Risk_Amt', 'Balance', 'Drawdown']].to_string(
            index=False, 
            formatters={
                'Risk_Amt': '${:,.2f}'.format, 
                'Balance': '${:,.2f}'.format,
                'Drawdown': '{:.2f}%'.format
            }
        ))

        # --- FINAL REPORT ---
        total_growth = ((balance - initial_capital) / initial_capital) * 100
        max_dd = df['Drawdown'].max()
        
        print("\n" + "="*60)
        print("📊 FINAL PERFORMANCE AUDIT")
        print("="*60)
        print(f"Starting Capital:   ${initial_capital:,.2f}")
        print(f"Final Balance:      ${balance:,.2f}")
        print(f"Total Net Profit:   ${(balance - initial_capital):,.2f}")
        print(f"Total ROI:          {total_growth:,.2f}%")
        print(f"Max Peak-to-Valley Drawdown: {max_dd:.2f}%")
        print(f"Final Trade Size:   ${df['Risk_Amt'].iloc[-1]:,.2f}")
        print("="*60 + "\n")

    except ValueError:
        print("❌ Error: Please enter numerical values.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    run_interactive_simulation()