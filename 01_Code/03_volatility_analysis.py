"""
03_volatility_analysis.py
=========================
Volatility-specific analysis and stylized facts.

Generates:
  - Yang-Zhang volatility time series (annualized)
  - Comparison of 4 volatility estimators
  - Volatility clustering: ACF of volatility and squared returns
  - Distribution of daily volatility (log-scale)
  - Rolling correlation between estimators
  - Sub-period comparison table

All figures saved to 02_Output/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf
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


def load_data():
    """Load daily data with volatility columns."""
    path = os.path.join(PROC_DIR, "daily_with_volatility.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ===================================================================
# 1. Yang-Zhang Volatility Time Series
# ===================================================================
def plot_yz_volatility(df):
    """Plot the Yang-Zhang annualized volatility over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Price
    axes[0].plot(df["Date"], df["Close"], linewidth=0.6, color="saddlebrown")
    axes[0].set_ylabel("Close Price (GBP)")
    axes[0].set_title("ICE London Cocoa: Price and Yang-Zhang Volatility")

    # Volatility
    axes[1].plot(df["Date"], df["vol_yz_annual"] * 100,
                 linewidth=0.6, color="darkred")
    axes[1].set_ylabel("Annualized Volatility (%)")
    axes[1].set_xlabel("Date")

    # Highlight crisis
    for ax in axes:
        ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31"),
                    alpha=0.1, color="red")
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "06_yz_volatility_timeseries.png"))
    plt.close(fig)
    print("Saved: 06_yz_volatility_timeseries.png")


# ===================================================================
# 2. Comparison of Volatility Estimators
# ===================================================================
def plot_estimator_comparison(df):
    """Compare all four volatility estimators."""
    fig, ax = plt.subplots(figsize=(14, 6))

    estimators = {
        "Yang-Zhang": "vol_yz_annual",
        "Close-to-Close": "vol_cc_annual",
        "Parkinson": "vol_park_annual",
        "Garman-Klass": "vol_gk_annual",
    }
    colors = ["darkred", "steelblue", "forestgreen", "darkorange"]

    for (name, col), color in zip(estimators.items(), colors):
        ax.plot(df["Date"], df[col] * 100, linewidth=0.5, alpha=0.8,
                label=name, color=color)

    ax.set_title("Comparison of Volatility Estimators (22-day rolling, annualized)")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_xlabel("Date")
    ax.legend()

    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "07_estimator_comparison.png"))
    plt.close(fig)
    print("Saved: 07_estimator_comparison.png")


# ===================================================================
# 3. Autocorrelation of Volatility (Stylized Fact: Clustering)
# ===================================================================
def plot_volatility_acf(df):
    """ACF of volatility and squared returns — shows long memory / clustering."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    sigma = df["sigma_yz"].dropna()
    sq_ret = (df["log_ret"] ** 2).dropna()
    abs_ret = df["log_ret"].abs().dropna()

    # ACF of daily YZ proxy
    plot_acf(sigma, lags=100, ax=axes[0, 0], title="ACF: Daily Yang-Zhang σ",
             alpha=0.05, zero=False)

    # ACF of squared returns
    plot_acf(sq_ret, lags=100, ax=axes[0, 1], title="ACF: Squared Returns",
             alpha=0.05, zero=False)

    # ACF of absolute returns
    plot_acf(abs_ret, lags=100, ax=axes[1, 0], title="ACF: Absolute Returns",
             alpha=0.05, zero=False)

    # ACF of log returns (should die off quickly)
    log_ret = df["log_ret"].dropna()
    plot_acf(log_ret, lags=100, ax=axes[1, 1], title="ACF: Log Returns",
             alpha=0.05, zero=False)

    fig.suptitle("Autocorrelation Analysis — Volatility Clustering", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "08_volatility_acf.png"))
    plt.close(fig)
    print("Saved: 08_volatility_acf.png")


# ===================================================================
# 4. Distribution of Daily Volatility
# ===================================================================
def plot_volatility_distribution(df):
    """Histogram of daily YZ volatility (original and log-transformed)."""
    sigma = df["sigma_yz"].dropna()
    log_sigma = np.log(sigma[sigma > 0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Original scale
    axes[0].hist(sigma, bins=100, density=True, alpha=0.7, color="darkred",
                 edgecolor="none")
    axes[0].set_title("Distribution of Daily Yang-Zhang σ")
    axes[0].set_xlabel("Daily Volatility")
    axes[0].set_ylabel("Density")

    # Log scale (should be closer to normal)
    axes[1].hist(log_sigma, bins=100, density=True, alpha=0.7, color="steelblue",
                 edgecolor="none")
    # Normal overlay
    x = np.linspace(log_sigma.min(), log_sigma.max(), 200)
    axes[1].plot(x, stats.norm.pdf(x, log_sigma.mean(), log_sigma.std()),
                 color="red", linewidth=1.5, label="Normal fit")
    axes[1].set_title("Distribution of log(σ) — closer to Normal")
    axes[1].set_xlabel("log(Daily Volatility)")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "09_volatility_distribution.png"))
    plt.close(fig)
    print("Saved: 09_volatility_distribution.png")


# ===================================================================
# 5. Sub-Period Volatility Comparison
# ===================================================================
def subperiod_table(df):
    """Compare volatility statistics across sub-periods."""
    df_tmp = df.set_index("Date").copy()

    periods = {
        "Full Sample":   (df_tmp.index.min(), df_tmp.index.max()),
        "1974–1990":     ("1974-01-01", "1990-12-31"),
        "1991–2005":     ("1991-01-01", "2005-12-31"),
        "2006–2021":     ("2006-01-01", "2021-12-31"),
        "2022–2025 (Crisis)": ("2022-01-01", "2025-12-31"),
    }

    rows = []
    for name, (start, end) in periods.items():
        sub = df_tmp.loc[start:end]
        ret = sub["log_ret"].dropna()
        vol = sub["vol_yz_annual"].dropna()
        rows.append({
            "Period": name,
            "Obs": len(ret),
            "Mean Return (%)": ret.mean() * 100,
            "Std Dev (%)": ret.std() * 100,
            "Skewness": ret.skew(),
            "Kurtosis": ret.kurtosis(),
            "Mean Ann. Vol (%)": vol.mean() * 100,
            "Max Ann. Vol (%)": vol.max() * 100,
        })

    table = pd.DataFrame(rows).set_index("Period")
    table.to_csv(os.path.join(TBL_DIR, "subperiod_statistics.csv"))

    print("=== Sub-Period Statistics ===")
    print(table.round(3).to_string())
    return table


# ===================================================================
# 6. Volatility Regime Plot (rolling 1y average)
# ===================================================================
def plot_volatility_regimes(df):
    """Show long-run volatility regimes using 1-year rolling average."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # 252-day rolling average of annualized YZ volatility
    vol_1y = df.set_index("Date")["vol_yz_annual"].rolling(252).mean() * 100

    ax.plot(vol_1y.index, vol_1y.values, linewidth=1.2, color="darkred")
    ax.axhline(vol_1y.median(), color="gray", linestyle="--", linewidth=0.8,
               label=f"Median = {vol_1y.median():.1f}%")

    ax.set_title("Volatility Regimes — 1-Year Rolling Average (Annualized)")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_xlabel("Date")
    ax.legend()

    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "10_volatility_regimes.png"))
    plt.close(fig)
    print("Saved: 10_volatility_regimes.png")


# ===================================================================
# 7. Correlation Matrix of Estimators
# ===================================================================
def plot_estimator_correlations(df):
    """Correlation heatmap between the four volatility estimators."""
    cols = {
        "Yang-Zhang": "vol_yz_daily",
        "Close-Close": "vol_cc_daily",
        "Parkinson": "vol_park_daily",
        "Garman-Klass": "vol_gk_daily",
    }
    corr_df = df[list(cols.values())].dropna()
    corr_df.columns = list(cols.keys())
    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=0.8, vmax=1.0)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    # Annotate
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.3f}",
                    ha="center", va="center", fontsize=11)

    ax.set_title("Correlation Between Volatility Estimators")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "11_estimator_correlation.png"))
    plt.close(fig)
    print("Saved: 11_estimator_correlation.png")

    # Save table too
    corr.to_csv(os.path.join(TBL_DIR, "estimator_correlations.csv"))


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("\n--- Volatility Plots ---")
    plot_yz_volatility(df)
    plot_estimator_comparison(df)
    plot_volatility_acf(df)
    plot_volatility_distribution(df)
    plot_volatility_regimes(df)
    plot_estimator_correlations(df)

    print("\n--- Sub-Period Table ---")
    subperiod_table(df)

    print("\nDone! All figures in 02_Output/figures/")
