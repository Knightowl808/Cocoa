"""
10_summary_plots.py
===================
Final summary plots combining results across all models and horizons.

Generates:
  - Model comparison: QLIKE across models and horizons
  - Quantile loss comparison: HAR-QR vs benchmarks
  - Coefficient evolution across quantiles (for one horizon)
  - Crisis period zoom-in

Run this AFTER all model scripts (06-09) have been executed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
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
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


# ===================================================================
# Helper: Quantile loss
# ===================================================================
def quantile_loss(actual, forecast, tau):
    """Check function loss."""
    u = actual - forecast
    return np.mean(u * (tau - (u < 0).astype(float)))


def qlike(actual, forecast):
    """QLIKE loss."""
    f = np.maximum(forecast, 1e-10)
    a = np.maximum(actual, 1e-10)
    return np.mean(np.log(f ** 2) + (a ** 2) / (f ** 2))


# ===================================================================
# 1. QLIKE Comparison: HAR-OLS vs Historical Vol vs GARCH
# ===================================================================
def plot_qlike_comparison():
    """Bar chart comparing QLIKE across models."""
    models = {}

    for label in HORIZONS.values():
        # HAR-OLS
        path = os.path.join(PROC_DIR, f"har_ols_forecasts_{label}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            models.setdefault("HAR-OLS", {})[label] = qlike(
                df["actual"].values, df["forecast"].values)

        # HAR-QR median
        path = os.path.join(PROC_DIR, f"har_qr_forecasts_{label}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            models.setdefault("HAR-QR (median)", {})[label] = qlike(
                df["actual"].values, df["q50"].values)

        # Historical Vol
        path = os.path.join(PROC_DIR, f"hist_vol_forecasts_{label}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            models.setdefault("Historical Vol", {})[label] = qlike(
                df["actual"].values, df["mean"].values)

        # GARCH
        path = os.path.join(PROC_DIR, f"garch_forecasts_{label}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path).dropna(subset=["mean"])
            if len(df) > 0:
                models.setdefault("GARCH(1,1)", {})[label] = qlike(
                    df["actual"].values, df["mean"].values)

    if not models:
        print("  No forecast files found. Run model scripts first.")
        return

    # Build table
    model_names = list(models.keys())
    horizons = list(HORIZONS.values())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(horizons))
    width = 0.2
    colors = ["steelblue", "red", "forestgreen", "darkorange"]

    for i, (name, losses) in enumerate(models.items()):
        vals = [losses.get(h, np.nan) for h in horizons]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=name,
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("QLIKE Loss (lower is better)")
    ax.set_title("QLIKE Loss Comparison Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "24_qlike_model_comparison.png"))
    plt.close(fig)
    print("Saved: 24_qlike_model_comparison.png")

    # Save table
    table = pd.DataFrame(models).T
    table.to_csv(os.path.join(TBL_DIR, "qlike_model_comparison.csv"))


# ===================================================================
# 2. Quantile Loss Comparison: HAR-QR vs Historical empirical quantiles
# ===================================================================
def plot_quantile_loss_comparison():
    """Compare quantile loss for tau=0.95 across models."""
    tau = 0.95
    results = {}

    for label in HORIZONS.values():
        # HAR-QR
        path = os.path.join(PROC_DIR, f"har_qr_forecasts_{label}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            results.setdefault("HAR-QR", {})[label] = quantile_loss(
                df["actual"].values, df["q95"].values, tau)

        # Historical empirical quantile
        path = os.path.join(PROC_DIR, f"hist_vol_forecasts_{label}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            results.setdefault("Historical", {})[label] = quantile_loss(
                df["actual"].values, df["q95"].values, tau)

        # GARCH simulated quantile
        path = os.path.join(PROC_DIR, f"garch_forecasts_{label}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path).dropna(subset=["q95"])
            if len(df) > 0:
                results.setdefault("GARCH(1,1)", {})[label] = quantile_loss(
                    df["actual"].values, df["q95"].values, tau)

    if not results:
        print("  No forecast files found.")
        return

    horizons = list(HORIZONS.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(horizons))
    width = 0.25
    colors = ["red", "forestgreen", "darkorange"]

    for i, (name, losses) in enumerate(results.items()):
        vals = [losses.get(h, np.nan) for h in horizons]
        offset = (i - len(results) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=name,
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Quantile Loss (τ=0.95)")
    ax.set_title("Upper Tail (95th Percentile) — Quantile Loss Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "25_quantile_loss_comparison_q95.png"))
    plt.close(fig)
    print("Saved: 25_quantile_loss_comparison_q95.png")


# ===================================================================
# 3. Crisis Zoom-In (2022-2024)
# ===================================================================
def plot_crisis_zoom(horizon_label="1m"):
    """Zoom into the 2022-2024 crisis period for one horizon."""
    path = os.path.join(PROC_DIR, f"har_qr_forecasts_{horizon_label}.csv")
    if not os.path.exists(path):
        print(f"  No HAR-QR forecast file for {horizon_label}")
        return

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Filter crisis period
    crisis = df[(df["Date"] >= "2022-01-01") & (df["Date"] <= "2025-01-01")]

    if len(crisis) == 0:
        print("  No data in crisis period")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.fill_between(crisis["Date"], crisis["q05"], crisis["q95"],
                    alpha=0.15, color="steelblue", label="5–95% band")
    ax.fill_between(crisis["Date"], crisis["q25"], crisis["q75"],
                    alpha=0.25, color="steelblue", label="25–75% band")
    ax.plot(crisis["Date"], crisis["q50"],
            linewidth=1.5, color="red", label="Median forecast")
    ax.plot(crisis["Date"], crisis["actual"],
            linewidth=1.0, color="black", label="Actual")

    violations = crisis["actual"] > crisis["q95"]
    if violations.any():
        ax.scatter(crisis.loc[violations, "Date"],
                   crisis.loc[violations, "actual"],
                   color="red", s=25, zorder=5, label="Violations (>95th)")

    ax.set_title(f"Crisis Period 2022–2024 — HAR-QR Fan Chart ({horizon_label})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Daily Volatility")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"26_crisis_zoom_{horizon_label}.png"))
    plt.close(fig)
    print(f"Saved: 26_crisis_zoom_{horizon_label}.png")


# ===================================================================
# 4. Overview: All Horizons Fan Charts in Grid
# ===================================================================
def plot_all_horizons_grid():
    """Grid of fan charts for all 6 horizons."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (h, label) in enumerate(HORIZONS.items()):
        ax = axes[idx]
        path = os.path.join(PROC_DIR, f"har_qr_forecasts_{label}.csv")
        if not os.path.exists(path):
            ax.set_title(f"h={h} ({label}) — No data")
            continue

        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])

        ax.fill_between(df["Date"], df["q05"], df["q95"],
                        alpha=0.15, color="steelblue")
        ax.fill_between(df["Date"], df["q25"], df["q75"],
                        alpha=0.25, color="steelblue")
        ax.plot(df["Date"], df["q50"], linewidth=0.8, color="red")
        ax.plot(df["Date"], df["actual"], linewidth=0.3, color="black", alpha=0.5)

        ax.set_title(f"h={h}d ({label})")
        ax.set_ylabel("Volatility")

    fig.suptitle("HAR-QR Quantile Forecasts — All Horizons", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "27_all_horizons_grid.png"))
    plt.close(fig)
    print("Saved: 27_all_horizons_grid.png")


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("Generating summary plots...\n")

    print("1. QLIKE model comparison")
    plot_qlike_comparison()

    print("\n2. Quantile loss comparison (τ=0.95)")
    plot_quantile_loss_comparison()

    print("\n3. Crisis zoom-in")
    plot_crisis_zoom("1m")
    plot_crisis_zoom("6m")

    print("\n4. All horizons grid")
    plot_all_horizons_grid()

    print("\nDone!")
