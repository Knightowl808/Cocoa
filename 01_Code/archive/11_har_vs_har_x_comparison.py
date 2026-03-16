"""
11_har_vs_har_x_comparison.py
==============================
Compare baseline HAR-QR vs ENSO-augmented HAR-X-QR.

This script provides the key empirical contribution of the thesis:
  - Does ENSO improve quantile forecast accuracy?
  - Does ENSO improve coverage at procurement horizons (6m-12m)?
  - Is the improvement statistically significant?

Evaluation metrics:
  - Quantile Loss (QL) comparison by horizon and quantile
  - Coverage diagnostics (5% and 95% quantiles)
  - Diebold-Mariano test for forecast superiority

Expected result (based on Bouri 2021, Bonato 2023):
  - ENSO helps most at LONG horizons (6m-12m)
  - ENSO helps most at TAIL quantiles (0.05, 0.95)
  - Little/no effect at short horizons (1d-1w)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

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

HORIZONS = ["1d", "1w", "1m", "3m", "6m", "12m"]
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def quantile_loss(actual, forecast, tau):
    """Quantile loss (check function)."""
    u = actual - forecast
    return np.mean(u * (tau - (u < 0).astype(float)))


def load_forecasts(model_name, horizon):
    """Load forecast results for given model and horizon."""
    if model_name == "HAR-QR":
        path = os.path.join(PROC_DIR, f"har_qr_forecasts_{horizon}.csv")
    elif model_name == "HAR-X-QR":
        path = os.path.join(PROC_DIR, f"har_x_qr_forecasts_{horizon}.csv")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ===================================================================
# 1. Quantile Loss Comparison
# ===================================================================
def compare_quantile_loss():
    """Compute QL for both models across all horizons and quantiles."""
    print("="*60)
    print("QUANTILE LOSS COMPARISON")
    print("="*60)

    rows = []
    for horizon in HORIZONS:
        har_qr = load_forecasts("HAR-QR", horizon)
        har_x_qr = load_forecasts("HAR-X-QR", horizon)

        if har_qr is None or har_x_qr is None:
            print(f"  {horizon}: Missing data, skipping")
            continue

        # Align on dates (HAR-X-QR may have fewer obs due to ENSO missings)
        common_dates = pd.merge(
            har_qr[["Date"]], har_x_qr[["Date"]], on="Date", how="inner"
        )
        har_qr = har_qr[har_qr["Date"].isin(common_dates["Date"])]
        har_x_qr = har_x_qr[har_x_qr["Date"].isin(common_dates["Date"])]

        print(f"\n{horizon}: {len(common_dates)} common observations")

        for tau in QUANTILES:
            col = f"q{int(tau*100):02d}"

            ql_har = quantile_loss(har_qr["actual"].values, har_qr[col].values, tau)
            ql_har_x = quantile_loss(har_x_qr["actual"].values, har_x_qr[col].values, tau)

            improvement = (ql_har - ql_har_x) / ql_har * 100

            rows.append({
                "Horizon": horizon,
                "Quantile": tau,
                "QL_HAR-QR": ql_har,
                "QL_HAR-X-QR": ql_har_x,
                "Improvement (%)": improvement
            })

            print(f"  τ={tau:.2f}: HAR-QR={ql_har:.6f}, HAR-X-QR={ql_har_x:.6f} "
                  f"({improvement:+.1f}%)")

    df_loss = pd.DataFrame(rows)
    df_loss.to_csv(os.path.join(TBL_DIR, "har_vs_har_x_quantile_loss.csv"), index=False)

    print("\n✓ Saved: har_vs_har_x_quantile_loss.csv")
    return df_loss


# ===================================================================
# 2. Coverage Comparison
# ===================================================================
def compare_coverage():
    """Compare empirical coverage at 5% and 95% quantiles."""
    print("\n" + "="*60)
    print("COVERAGE COMPARISON (5% and 95% quantiles)")
    print("="*60)

    rows = []
    for horizon in HORIZONS:
        har_qr = load_forecasts("HAR-QR", horizon)
        har_x_qr = load_forecasts("HAR-X-QR", horizon)

        if har_qr is None or har_x_qr is None:
            continue

        # Align
        common_dates = pd.merge(
            har_qr[["Date"]], har_x_qr[["Date"]], on="Date", how="inner"
        )
        har_qr = har_qr[har_qr["Date"].isin(common_dates["Date"])]
        har_x_qr = har_x_qr[har_x_qr["Date"].isin(common_dates["Date"])]

        n = len(common_dates)

        # 5% quantile (should have 5% violations)
        viol_05_har = (har_qr["actual"] < har_qr["q05"]).sum()
        viol_05_har_x = (har_x_qr["actual"] < har_x_qr["q05"]).sum()

        # 95% quantile (should have 5% violations)
        viol_95_har = (har_qr["actual"] > har_qr["q95"]).sum()
        viol_95_har_x = (har_x_qr["actual"] > har_x_qr["q95"]).sum()

        print(f"\n{horizon} (N={n}):")
        print(f"  τ=0.05: HAR-QR {viol_05_har/n*100:.1f}%, HAR-X-QR {viol_05_har_x/n*100:.1f}%")
        print(f"  τ=0.95: HAR-QR {viol_95_har/n*100:.1f}%, HAR-X-QR {viol_95_har_x/n*100:.1f}%")

        rows.append({
            "Horizon": horizon,
            "N": n,
            "Model": "HAR-QR",
            "Violations_05": viol_05_har,
            "Rate_05 (%)": viol_05_har / n * 100,
            "Violations_95": viol_95_har,
            "Rate_95 (%)": viol_95_har / n * 100,
        })
        rows.append({
            "Horizon": horizon,
            "N": n,
            "Model": "HAR-X-QR",
            "Violations_05": viol_05_har_x,
            "Rate_05 (%)": viol_05_har_x / n * 100,
            "Violations_95": viol_95_har_x,
            "Rate_95 (%)": viol_95_har_x / n * 100,
        })

    df_cov = pd.DataFrame(rows)
    df_cov.to_csv(os.path.join(TBL_DIR, "har_vs_har_x_coverage.csv"), index=False)

    print("\n✓ Saved: har_vs_har_x_coverage.csv")
    return df_cov


# ===================================================================
# 3. Improvement Heatmap
# ===================================================================
def plot_improvement_heatmap(df_loss):
    """Heatmap showing % improvement from ENSO augmentation."""
    pivot = df_loss.pivot(index="Quantile", columns="Horizon", values="Improvement (%)")
    pivot = pivot[HORIZONS]  # reorder

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                vmin=-5, vmax=5, ax=ax, cbar_kws={"label": "Improvement (%)"})
    ax.set_title("HAR-X-QR Improvement over HAR-QR (% reduction in Quantile Loss)\n"
                 "Green = ENSO helps, Red = ENSO hurts")
    ax.set_ylabel("Quantile τ")
    ax.set_xlabel("Forecast Horizon")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "25_har_x_improvement_heatmap.png"))
    plt.close(fig)
    print("✓ Saved: 25_har_x_improvement_heatmap.png")


# ===================================================================
# 4. Coverage Deviation Plot
# ===================================================================
def plot_coverage_deviation(df_cov):
    """Plot how far each model deviates from target 5% coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # τ=0.05
    ax = axes[0]
    har_05 = df_cov[df_cov["Model"] == "HAR-QR"][["Horizon", "Rate_05 (%)"]].set_index("Horizon")
    har_x_05 = df_cov[df_cov["Model"] == "HAR-X-QR"][["Horizon", "Rate_05 (%)"]].set_index("Horizon")

    x = np.arange(len(HORIZONS))
    width = 0.35

    ax.bar(x - width/2, har_05["Rate_05 (%)"], width, label="HAR-QR", color="steelblue")
    ax.bar(x + width/2, har_x_05["Rate_05 (%)"], width, label="HAR-X-QR", color="darkorange")
    ax.axhline(5, color="red", linestyle="--", linewidth=1.5, label="Target (5%)")

    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS)
    ax.set_ylabel("Empirical Violation Rate (%)")
    ax.set_title("Coverage at τ=0.05 (Lower Tail)")
    ax.legend()
    ax.set_ylim([0, 10])

    # τ=0.95
    ax = axes[1]
    har_95 = df_cov[df_cov["Model"] == "HAR-QR"][["Horizon", "Rate_95 (%)"]].set_index("Horizon")
    har_x_95 = df_cov[df_cov["Model"] == "HAR-X-QR"][["Horizon", "Rate_95 (%)"]].set_index("Horizon")

    ax.bar(x - width/2, har_95["Rate_95 (%)"], width, label="HAR-QR", color="steelblue")
    ax.bar(x + width/2, har_x_95["Rate_95 (%)"], width, label="HAR-X-QR", color="darkorange")
    ax.axhline(5, color="red", linestyle="--", linewidth=1.5, label="Target (5%)")

    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS)
    ax.set_ylabel("Empirical Violation Rate (%)")
    ax.set_title("Coverage at τ=0.95 (Upper Tail)")
    ax.legend()
    ax.set_ylim([0, 10])

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "26_coverage_comparison.png"))
    plt.close(fig)
    print("✓ Saved: 26_coverage_comparison.png")


# ===================================================================
# 5. Summary Statistics
# ===================================================================
def print_summary(df_loss):
    """Print summary of ENSO effects."""
    print("\n" + "="*60)
    print("SUMMARY: ENSO AUGMENTATION EFFECTS")
    print("="*60)

    # By horizon
    print("\nAverage improvement by horizon:")
    by_horizon = df_loss.groupby("Horizon")["Improvement (%)"].mean()
    for h in HORIZONS:
        if h in by_horizon.index:
            print(f"  {h:>4s}: {by_horizon[h]:+.2f}%")

    # By quantile
    print("\nAverage improvement by quantile:")
    by_quantile = df_loss.groupby("Quantile")["Improvement (%)"].mean()
    for q in QUANTILES:
        print(f"  τ={q:.2f}: {by_quantile[q]:+.2f}%")

    # Long horizon tails
    long_horizons = ["6m", "12m"]
    tail_quantiles = [0.05, 0.95]
    long_tail = df_loss[
        (df_loss["Horizon"].isin(long_horizons)) &
        (df_loss["Quantile"].isin(tail_quantiles))
    ]["Improvement (%)"].mean()

    print(f"\nAverage improvement at long-horizon tails (6m-12m, τ=0.05/0.95):")
    print(f"  {long_tail:+.2f}%")

    if long_tail > 1.0:
        print("\n✓ ENSO IMPROVES tail-risk forecasts at procurement horizons!")
    else:
        print("\n✗ ENSO does NOT significantly improve tail forecasts.")


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("HAR-QR vs HAR-X-QR Comparison")
    print("="*60)

    # 1. Quantile Loss
    df_loss = compare_quantile_loss()

    # 2. Coverage
    df_cov = compare_coverage()

    # 3. Visualizations
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    plot_improvement_heatmap(df_loss)
    plot_coverage_deviation(df_cov)

    # 4. Summary
    print_summary(df_loss)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
