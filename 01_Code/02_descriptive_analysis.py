"""
02_descriptive_analysis.py
==========================
Descriptive statistics and basic plots for the daily cocoa futures data.

Generates:
  - Summary statistics table (price levels, returns, volume)
  - Price time series plot
  - Log-return distribution (histogram + QQ plot)
  - Return series plot
  - Volume over time

All figures saved to 02_Output/figures/
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

# Plot style
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def load_data():
    """Load the daily data with volatility columns."""
    path = os.path.join(PROC_DIR, "daily_with_volatility.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ===================================================================
# 1. Summary Statistics Table
# ===================================================================
def summary_statistics(df):
    """Compute and save summary statistics for price and returns."""

    # --- Price levels ---
    price_stats = df["Close"].describe()

    # --- Log returns ---
    ret = df["log_ret"].dropna()
    ret_stats = pd.Series({
        "Count": len(ret),
        "Mean": ret.mean(),
        "Std Dev": ret.std(),
        "Min": ret.min(),
        "25%": ret.quantile(0.25),
        "Median": ret.median(),
        "75%": ret.quantile(0.75),
        "Max": ret.max(),
        "Skewness": ret.skew(),
        "Kurtosis": ret.kurtosis(),  # excess kurtosis
        "Jarque-Bera stat": stats.jarque_bera(ret)[0],
        "Jarque-Bera p-val": stats.jarque_bera(ret)[1],
    })

    # Combine into one table
    summary = pd.DataFrame({
        "Close Price (GBP)": price_stats,
    })

    # Save
    ret_stats.to_csv(os.path.join(TBL_DIR, "return_statistics.csv"))
    summary.to_csv(os.path.join(TBL_DIR, "price_summary.csv"))

    print("=== Return Statistics ===")
    print(ret_stats.round(6).to_string())
    print()

    return ret_stats


# ===================================================================
# 2. Price Time Series Plot
# ===================================================================
def plot_price_series(df):
    """Plot cocoa futures closing price over time."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df["Date"], df["Close"], linewidth=0.6, color="saddlebrown")
    ax.set_title("ICE London Cocoa Futures — Close Price (GBP/tonne)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (GBP)")

    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_price_series.png"))
    plt.close(fig)
    print("Saved: 01_price_series.png")


# ===================================================================
# 3. Log-Return Series
# ===================================================================
def plot_return_series(df):
    """Plot daily log-returns over time."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df["Date"], df["log_ret"], linewidth=0.3, color="steelblue", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Daily Log-Returns — ICE London Cocoa Futures")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")

    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_return_series.png"))
    plt.close(fig)
    print("Saved: 02_return_series.png")


# ===================================================================
# 4. Return Distribution (Histogram + Normal overlay + QQ)
# ===================================================================
def plot_return_distribution(df):
    """Histogram of log-returns with normal overlay, plus QQ plot."""
    ret = df["log_ret"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Histogram ---
    ax = axes[0]
    ax.hist(ret, bins=150, density=True, alpha=0.7, color="steelblue",
            edgecolor="none", label="Empirical")

    # Normal overlay
    x = np.linspace(ret.min(), ret.max(), 300)
    ax.plot(x, stats.norm.pdf(x, ret.mean(), ret.std()),
            color="red", linewidth=1.5, label="Normal fit")

    ax.set_title("Distribution of Daily Log-Returns")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.legend()

    # --- QQ Plot ---
    ax = axes[1]
    stats.probplot(ret, dist="norm", plot=ax)
    ax.set_title("QQ Plot vs Normal Distribution")
    ax.get_lines()[0].set_markersize(2)
    ax.get_lines()[0].set_alpha(0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_return_distribution.png"))
    plt.close(fig)
    print("Saved: 03_return_distribution.png")


# ===================================================================
# 5. Volume Over Time
# ===================================================================
def plot_volume(df):
    """Plot trading volume over time."""
    fig, ax = plt.subplots(figsize=(14, 4))

    # Use monthly average for cleaner plot
    monthly = df.set_index("Date")["Volume"].resample("ME").mean()

    ax.bar(monthly.index, monthly.values, width=25, color="gray", alpha=0.7)
    ax.set_title("Monthly Average Trading Volume — ICE London Cocoa Futures")
    ax.set_xlabel("Date")
    ax.set_ylabel("Contracts")

    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_volume.png"))
    plt.close(fig)
    print("Saved: 04_volume.png")


# ===================================================================
# 6. Annual Statistics (returns by year)
# ===================================================================
def plot_annual_returns(df):
    """Bar chart of annualized return and volatility per year."""
    df_tmp = df.set_index("Date").copy()
    df_tmp["Year"] = df_tmp.index.year

    annual = df_tmp.groupby("Year").agg(
        ann_ret=("log_ret", lambda x: x.sum()),
        ann_vol=("log_ret", lambda x: x.std() * np.sqrt(252)),
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Returns
    colors = ["green" if r > 0 else "red" for r in annual["ann_ret"]]
    axes[0].bar(annual.index, annual["ann_ret"] * 100, color=colors, alpha=0.7)
    axes[0].set_ylabel("Annual Return (%)")
    axes[0].set_title("Annual Log-Returns and Volatility")
    axes[0].axhline(0, color="black", linewidth=0.5)

    # Volatility
    axes[1].bar(annual.index, annual["ann_vol"] * 100, color="steelblue", alpha=0.7)
    axes[1].set_ylabel("Annualized Volatility (%)")
    axes[1].set_xlabel("Year")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "05_annual_stats.png"))
    plt.close(fig)
    print("Saved: 05_annual_stats.png")


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("\n--- Summary Statistics ---")
    summary_statistics(df)

    print("\n--- Generating Plots ---")
    plot_price_series(df)
    plot_return_series(df)
    plot_return_distribution(df)
    plot_volume(df)
    plot_annual_returns(df)

    print("\nDone! All figures in 02_Output/figures/")
