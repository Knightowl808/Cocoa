"""
09_var_backtest.py
==================
VaR backtesting for the upper quantile (tau=0.95) forecasts.

Implements (from Methodology Section 3.3.4):
  - Kupiec unconditional coverage test (Eq. 3.13)
  - Christoffersen conditional coverage test (Eq. 3.14)
  - Violation series plots
  - Crisis freeze test: parameters from Dec 2021, forecast 2022-2024

Output: test results, violation plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "02_Output", "figures")
TBL_DIR = os.path.join(BASE_DIR, "02_Output", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

HORIZONS = {1: "1d", 5: "1w", 22: "1m", 66: "3m", 132: "6m", 264: "12m"}


# ===================================================================
# 1. Kupiec Unconditional Coverage Test
# ===================================================================
def kupiec_test(violations, n, alpha=0.05):
    """
    Kupiec (1995) unconditional coverage test (Eq. 3.13).

    H0: The violation rate equals alpha (= 1 - tau = 0.05 for tau=0.95).

    Parameters
    ----------
    violations : array-like of 0/1 (1 = violation)
    n : total number of observations
    alpha : expected violation rate (default 0.05)

    Returns
    -------
    dict with LR statistic, p-value, actual violation rate
    """
    n1 = np.sum(violations)  # number of violations
    n0 = n - n1

    if n1 == 0 or n1 == n:
        return {"LR_uc": np.nan, "p_value": np.nan, "viol_rate": n1 / n}

    pi_hat = n1 / n  # empirical violation rate

    # Likelihood ratio
    LR_uc = 2 * (
        n1 * np.log(pi_hat / alpha)
        + n0 * np.log((1 - pi_hat) / (1 - alpha))
    )

    p_value = 1 - stats.chi2.cdf(LR_uc, df=1)

    return {
        "LR_uc": LR_uc,
        "p_value_uc": p_value,
        "violations": int(n1),
        "total": n,
        "viol_rate": pi_hat,
        "expected_rate": alpha,
    }


# ===================================================================
# 2. Christoffersen Conditional Coverage Test
# ===================================================================
def christoffersen_test(violations, alpha=0.05):
    """
    Christoffersen (1998) conditional coverage test (Eq. 3.14).

    Tests both:
      - Correct violation frequency (unconditional coverage)
      - No clustering of violations (independence)

    LR_cc = LR_uc + LR_ind ~ chi2(2)
    """
    n = len(violations)
    v = np.array(violations, dtype=int)

    # Transition counts
    n00 = np.sum((v[:-1] == 0) & (v[1:] == 0))
    n01 = np.sum((v[:-1] == 0) & (v[1:] == 1))
    n10 = np.sum((v[:-1] == 1) & (v[1:] == 0))
    n11 = np.sum((v[:-1] == 1) & (v[1:] == 1))

    # Unconditional part
    kupiec = kupiec_test(v, n, alpha)

    # Independence test
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return {**kupiec, "LR_ind": np.nan, "LR_cc": np.nan, "p_value_cc": np.nan}

    pi01 = n01 / (n00 + n01)  # P(violation | no violation yesterday)
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0  # P(violation | violation yesterday)
    pi_hat = (n01 + n11) / (n - 1)  # unconditional from transitions

    # Avoid log(0)
    eps = 1e-10

    if pi01 < eps or pi11 < eps or (1 - pi01) < eps or (1 - pi11) < eps:
        LR_ind = 0
    else:
        log_L_ind = (
            n00 * np.log(1 - pi01 + eps) + n01 * np.log(pi01 + eps)
            + n10 * np.log(1 - pi11 + eps) + n11 * np.log(pi11 + eps)
        )
        log_L_0 = (
            (n00 + n10) * np.log(1 - pi_hat + eps)
            + (n01 + n11) * np.log(pi_hat + eps)
        )
        LR_ind = 2 * (log_L_ind - log_L_0)

    LR_cc = kupiec["LR_uc"] + LR_ind if not np.isnan(kupiec["LR_uc"]) else np.nan
    p_cc = 1 - stats.chi2.cdf(LR_cc, df=2) if not np.isnan(LR_cc) else np.nan

    return {
        **kupiec,
        "LR_ind": LR_ind,
        "p_value_ind": 1 - stats.chi2.cdf(LR_ind, df=1) if not np.isnan(LR_ind) else np.nan,
        "LR_cc": LR_cc,
        "p_value_cc": p_cc,
    }


# ===================================================================
# 3. Violation Plot
# ===================================================================
def plot_violations(results, horizon_label):
    """Plot actual vs 95th percentile forecast, highlighting violations."""
    violations = results["actual"] > results["q95"]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(results["Date"], results["actual"],
            linewidth=0.5, color="black", alpha=0.6, label="Actual")
    ax.plot(results["Date"], results["q95"],
            linewidth=0.8, color="red", label="95th percentile forecast")

    # Mark violations
    viol_dates = results.loc[violations, "Date"]
    viol_vals = results.loc[violations, "actual"]
    ax.scatter(viol_dates, viol_vals, color="red", s=10, alpha=0.5,
               zorder=5, label=f"Violations ({violations.sum()}/{len(results)})")

    rate = violations.mean() * 100
    ax.set_title(f"VaR Backtest — Horizon {horizon_label} "
                 f"(Violation rate: {rate:.1f}%, expected: 5.0%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Daily Volatility")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"23_var_backtest_{horizon_label}.png"))
    plt.close(fig)
    print(f"  Saved: 23_var_backtest_{horizon_label}.png")


# ===================================================================
# 4. Summary Table
# ===================================================================
def backtest_all_horizons():
    """Run VaR backtests for all available horizons."""
    all_rows = []

    for h, label in HORIZONS.items():
        path = os.path.join(PROC_DIR, f"har_qr_forecasts_{label}.csv")
        if not os.path.exists(path):
            print(f"  Skipping {label}: no forecast file found")
            continue

        results = pd.read_csv(path)
        results["Date"] = pd.to_datetime(results["Date"])

        violations = (results["actual"] > results["q95"]).astype(int).values

        # Christoffersen test (includes Kupiec)
        test = christoffersen_test(violations)
        test["Horizon"] = label
        all_rows.append(test)

        print(f"\n  Horizon {label}:")
        print(f"    Violations: {test['violations']}/{test['total']} "
              f"({test['viol_rate']*100:.1f}%)")
        print(f"    Kupiec LR: {test['LR_uc']:.3f} (p={test['p_value_uc']:.4f})")
        print(f"    Christoffersen LR: {test['LR_cc']:.3f} "
              f"(p={test['p_value_cc']:.4f})")

        # Plot violations
        plot_violations(results, label)

    if all_rows:
        table = pd.DataFrame(all_rows)
        table = table[["Horizon", "violations", "total", "viol_rate",
                        "LR_uc", "p_value_uc", "LR_ind", "p_value_ind",
                        "LR_cc", "p_value_cc"]]
        table.to_csv(os.path.join(TBL_DIR, "var_backtest_results.csv"), index=False)
        print("\n=== VaR Backtest Summary ===")
        print(table.round(4).to_string(index=False))

    return table if all_rows else None


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("Running VaR backtests...")
    table = backtest_all_horizons()
    print("\nDone!")
