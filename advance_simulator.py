import numpy as np
import matplotlib.pyplot as plt

def run_alpha_factory(
    initial_capital=1000,
    base_wr=0.68,
    num_trades=100,
    risk_pct=0.02,
    profit_pct=0.84,
    num_universes=1000,
    wr_std=0.03,
    wr_haircut=0.03,
    decay_on=False,
    decay_factor=0.9,
    decay_start_trade=None,
    ruin_threshold_pct=0.75
):
    """
    Full Monte Carlo simulation implementing the 13 steps.
    - Bernoulli trades
    - Win-rate sampling + haircut
    - Multiple universes
    - Percentiles / cone
    - Risk-of-ruin, drawdowns
    - Edge decay switch (off by default)
    """

    final_balances = []
    max_drawdowns = []
    equity_matrix = []

    for u in range(num_universes):
        # --- 3 & 4: sample WR with haircut + variance ---
        wr = np.random.normal(base_wr - wr_haircut, wr_std)
        wr = np.clip(wr, 0.05, 0.95)

        balance = initial_capital
        ath = initial_capital
        equity_curve = []

        for t in range(num_trades):
            # --- 1: Bernoulli trade ---
            win = np.random.rand() < wr

            # --- 10: edge decay ---
            eff_profit_pct = profit_pct
            if decay_on and decay_start_trade is not None and t >= decay_start_trade:
                eff_profit_pct *= decay_factor

            risk_amount = balance * risk_pct
            balance += risk_amount * eff_profit_pct if win else -risk_amount

            ath = max(ath, balance)
            dd = (ath - balance) / ath * 100

            equity_curve.append(balance)

        equity_matrix.append(equity_curve)
        final_balances.append(balance)
        max_drawdowns.append(np.max((np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve) * 100))

    equity_matrix = np.array(equity_matrix)
    final_balances = np.array(final_balances)
    max_drawdowns = np.array(max_drawdowns)

    # --- 11: percentiles ---
    p5 = np.percentile(equity_matrix, 5, axis=0)
    p50 = np.percentile(equity_matrix, 50, axis=0)
    p95 = np.percentile(equity_matrix, 95, axis=0)

    # --- 12: cone plot ---
    plt.figure(figsize=(10,6))
    plt.fill_between(range(1, num_trades+1), p5, p95, color='skyblue', alpha=0.4, label='5-95th percentile')
    plt.plot(range(1, num_trades+1), p50, color='blue', label='Median')
    plt.title("Cone of Possibility (Equity Bands)")
    plt.xlabel("Trade Number")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.show()

    # --- 7 & 8: probability of losing money / hitting drawdown ---
    prob_loss = np.mean(final_balances < initial_capital) * 100
    prob_dd25 = np.mean(max_drawdowns >= 25) * 100
    prob_dd50 = np.mean(max_drawdowns >= 50) * 100

    # --- 13: capital survival / risk-of-ruin ---
    ruin_threshold = initial_capital * ruin_threshold_pct
    prob_ruin = np.mean(final_balances < ruin_threshold) * 100
    capital_survival = 100 - prob_ruin

    # --- 5 & 6: final balance distribution / max drawdown summary ---
    summary = {
        "Final Balance Mean": np.mean(final_balances),
        "Final Balance Median": np.median(final_balances),
        "Final Balance 5th %": np.percentile(final_balances, 5),
        "Final Balance 95th %": np.percentile(final_balances, 95),
        "Max Drawdown Mean (%)": np.mean(max_drawdowns),
        "Max Drawdown Median (%)": np.median(max_drawdowns),
        "Max Drawdown 5th %": np.percentile(max_drawdowns, 5),
        "Max Drawdown 95th %": np.percentile(max_drawdowns, 95),
        "Probability of Losing Money (%)": prob_loss,
        "Probability of -25% DD (%)": prob_dd25,
        "Probability of -50% DD (%)": prob_dd50,
        "Risk-of-Ruin (%)": prob_ruin,
        "Capital Survival (%)": capital_survival
    }

    # --- 9: WR floor sweep ---

    print("\n--- WR Floor Sweep Analysis ---")
    for wr_floor in [0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = np.random.normal(base_wr - wr_haircut, wr_std, num_universes)
        mask = np.clip(mask, wr_floor, 0.95)
        survival = np.mean(mask > 0.5) * 100
        print(f"WR floor {wr_floor:.2f}: Survival chance ~{survival:.2f}% (approx)")

    # --- Print summary of key 13 metrics ---
    print("\n--- Monte Carlo Simulation Summary ---")
    for k,v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:,.2f}")
        else:
            print(f"{k}: {v}")
            print()

# ---------------- RUN SIMULATION ----------------
if __name__ == "__main__":
    run_alpha_factory(
        initial_capital=1000,
        base_wr=0.60,
        num_trades=100,
        risk_pct=0.01,
        profit_pct=1.00,
        num_universes=1000,
        wr_std=0.04,
        wr_haircut=0.04,
        decay_on=False
    )
