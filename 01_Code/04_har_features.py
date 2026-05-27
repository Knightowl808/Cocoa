"""
04_har_features.py
==================
Build HAR model components and multi-horizon target variables.

HAR (Heterogeneous Autoregressive) components:
  - sigma_d:  daily volatility (sigma_t)
  - sigma_w:  weekly average (past 5 days)
  - sigma_m:  monthly average (past 22 days)

Target variables (direct projection, Eq. 3.7 in Methodology):
  - sigma_bar_{t:t+h} = (1/h) * sum_{i=1}^{h} sigma_{t+i}
  for h in {1, 5, 22, 66, 132, 264}

Also generates:
  - Scatter plots of HAR components vs target
  - Correlation heatmap of features
  - Horizon-decay plot

Output: har_dataset.csv in 00_Data/processed/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Horizons from the methodology (Section 3.3.2)
HORIZONS = {
    1: "1d",
    5: "1w",
    22: "1m",
    66: "3m",
    132: "6m",
    264: "12m",
}


def load_data():
    """Load daily data with volatility."""
    path = os.path.join(PROC_DIR, "daily_with_volatility.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ===================================================================
# 1. Build HAR Features
# ===================================================================
def build_har_features(df):
    """
    Construct HAR predictor variables from daily Yang-Zhang volatility.

    Features (all computed at time t, using info up to t):
      - sigma_d: sigma_t            (daily component)
      - sigma_w: mean(sigma_{t-4:t}) (weekly = 5-day average)
      - sigma_m: mean(sigma_{t-21:t}) (monthly = 22-day average)
    """
    sigma = df["sigma_yz"].copy()

    # Daily component: just sigma_t
    df["sigma_d"] = sigma

    # Weekly component: average of past 5 days (including today)
    df["sigma_w"] = sigma.rolling(window=5, min_periods=5).mean()

    # Monthly component: average of past 22 days (including today)
    df["sigma_m"] = sigma.rolling(window=22, min_periods=22).mean()

    return df


# ===================================================================
# 2. Build Target Variables (Forward-Looking Averages)
# ===================================================================
def build_targets(df):
    """
    Construct target variables: average volatility over future h days.

    sigma_bar_{t:t+h} = (1/h) * sum_{i=1}^{h} sigma_{t+i}

    This is the dependent variable in the HAR-QR model (Eq. 3.7).
    We shift the rolling mean backwards so that at time t, the target
    covers days t+1 through t+h.
    """
    sigma = df["sigma_yz"].copy()

    for h, label in HORIZONS.items():
        # Forward-looking rolling mean: average of next h days
        # rolling(h).mean() computes backward average, so shift(-h)
        # gives us the forward average from t+1 to t+h
        forward_avg = sigma.rolling(window=h, min_periods=h).mean().shift(-h)
        df[f"target_{label}"] = forward_avg

    return df


# ===================================================================
# 3. Scatter Plots: Features vs Targets
# ===================================================================
def plot_feature_target_scatter(df):
    """Scatter plots of HAR components vs. the 1-month target."""
    features = ["sigma_d", "sigma_w", "sigma_m"]
    labels = ["Daily σ", "Weekly σ (5d avg)", "Monthly σ (22d avg)"]
    target = "target_1m"

    # Sample for readability (too many points otherwise)
    plot_df = df[features + [target]].dropna().sample(
        n=min(3000, len(df)), random_state=42
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, (feat, lab) in enumerate(zip(features, labels)):
        ax = axes[i]
        ax.scatter(plot_df[feat], plot_df[target],
                   alpha=0.15, s=8, color="steelblue")

        # Add regression line
        mask = plot_df[[feat, target]].dropna().index
        x = plot_df.loc[mask, feat]
        y = plot_df.loc[mask, target]
        if len(x) > 10:
            z = np.polyfit(x, y, 1)
            ax.plot(np.sort(x), np.polyval(z, np.sort(x)),
                    color="red", linewidth=1.5)
            corr = x.corr(y)
            ax.set_title(f"{lab} vs Target (r={corr:.3f})")
        else:
            ax.set_title(f"{lab} vs Target")

        ax.set_xlabel(lab)
        ax.set_ylabel("Target: avg σ (22d ahead)")

    fig.suptitle("HAR Features vs 1-Month Target Volatility", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "12_har_feature_scatter.png"))
    plt.close(fig)
    print("Saved: 12_har_feature_scatter.png")


# ===================================================================
# 4. Feature Correlation Heatmap
# ===================================================================
def plot_feature_correlations(df):
    """Correlation between HAR features and all target horizons."""
    feat_cols = ["sigma_d", "sigma_w", "sigma_m"]
    target_cols = [f"target_{label}" for label in HORIZONS.values()]
    all_cols = feat_cols + target_cols

    corr = df[all_cols].dropna().corr()

    # Rename for readability
    rename = {
        "sigma_d": "σ_daily",
        "sigma_w": "σ_weekly",
        "sigma_m": "σ_monthly",
    }
    for h, label in HORIZONS.items():
        rename[f"target_{label}"] = f"Target {label}"
    corr = corr.rename(index=rename, columns=rename)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r",
                vmin=0, vmax=1, ax=ax, square=True)
    ax.set_title("Correlation: HAR Features and Target Horizons")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "13_feature_correlation_heatmap.png"))
    plt.close(fig)
    print("Saved: 13_feature_correlation_heatmap.png")

    # Save table
    corr.to_csv(os.path.join(TBL_DIR, "har_correlations.csv"))


# ===================================================================
# 5. Horizon Decay Plot
# ===================================================================
def plot_horizon_decay(df):
    """
    Show how predictive power (correlation) decays with horizon.
    This motivates the thesis: at long horizons, autoregressive signal fades.
    """
    feat_cols = ["sigma_d", "sigma_w", "sigma_m"]
    feat_labels = ["Daily σ", "Weekly σ", "Monthly σ"]
    colors = ["steelblue", "forestgreen", "darkorange"]

    horizon_days = list(HORIZONS.keys())
    horizon_labels = list(HORIZONS.values())

    fig, ax = plt.subplots(figsize=(10, 6))

    for feat, lab, col in zip(feat_cols, feat_labels, colors):
        corrs = []
        for h, h_label in HORIZONS.items():
            target = f"target_{h_label}"
            r = df[[feat, target]].dropna().corr().iloc[0, 1]
            corrs.append(r)
        ax.plot(horizon_days, corrs, marker="o", label=lab, color=col, linewidth=2)

    ax.set_xlabel("Forecast Horizon (trading days)")
    ax.set_ylabel("Correlation with Target")
    ax.set_title("Predictive Power Decay by Horizon")
    ax.set_xticks(horizon_days)
    ax.set_xticklabels([f"{h}d\n({l})" for h, l in zip(horizon_days, horizon_labels)])
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "14_horizon_decay.png"))
    plt.close(fig)
    print("Saved: 14_horizon_decay.png")


# ===================================================================
# 6. HAR Components Time Series
# ===================================================================
def plot_har_components(df):
    """Plot the three HAR components side by side."""
    # Use a 5-year recent window for clarity
    recent = df[df["Date"] >= "2015-01-01"].copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(recent["Date"], recent["sigma_d"],
            linewidth=0.4, alpha=0.5, color="steelblue", label="Daily σ")
    ax.plot(recent["Date"], recent["sigma_w"],
            linewidth=1.0, color="forestgreen", label="Weekly σ (5d)")
    ax.plot(recent["Date"], recent["sigma_m"],
            linewidth=1.5, color="darkorange", label="Monthly σ (22d)")

    ax.set_title("HAR Components — Daily, Weekly, Monthly (2015–2025)")
    ax.set_ylabel("Daily Volatility (Yang-Zhang)")
    ax.set_xlabel("Date")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "15_har_components.png"))
    plt.close(fig)
    print("Saved: 15_har_components.png")


# ===================================================================
# 7. Merge Nino 3.4 onto Daily Data
# ===================================================================
def merge_nino34(df):
    """
    Merge monthly Nino 3.4 anomaly onto daily data.

    Each trading day gets the Nino 3.4 value of its month.
    This is the ENSO_t variable in the HAR-X-QR model (Eq. 3.6).
    """
    nino_path = os.path.join(PROC_DIR, "nino34_clean.csv")
    nino = pd.read_csv(nino_path)
    nino["Date"] = pd.to_datetime(nino["Date"])

    # Create year-month key for merging
    df["ym"] = df["Date"].dt.to_period("M")
    nino["ym"] = nino["Date"].dt.to_period("M")

    # Merge on year-month
    nino_monthly = nino[["ym", "nino34"]].rename(columns={"nino34": "enso"})
    df = df.merge(nino_monthly, on="ym", how="left")
    df = df.drop(columns=["ym"])

    n_missing = df["enso"].isna().sum()
    print(f"  ENSO merged: {df['enso'].notna().sum()} days matched, {n_missing} missing")

    return df


# ===================================================================
# 8. Build ENSO Volatility Columns (rolling stdev of monthly Nino 3.4)
# ===================================================================
def build_enso_vol(df):
    """
    Add rolling stdev of Nino 3.4 as enso_vol_12m and enso_vol_36m.

    Computed on the MONTHLY series before broadcasting to daily.
    Computing on the daily-broadcast column would inflate N artificially
    (22 identical daily values per month), so we work at source.

    enso_vol_12m: 12-month rolling stdev — captures annual cycle variation
    enso_vol_36m: 36-month rolling stdev — captures cross-cycle amplitude
    """
    nino_path = os.path.join(PROC_DIR, "nino34_clean.csv")
    nino = pd.read_csv(nino_path)
    nino["Date"] = pd.to_datetime(nino["Date"])
    nino["ym"] = nino["Date"].dt.to_period("M")
    nino = nino.sort_values("ym").reset_index(drop=True)

    nino["enso_vol_12m"] = nino["nino34"].rolling(window=12, min_periods=12).std()
    nino["enso_vol_36m"] = nino["nino34"].rolling(window=36, min_periods=36).std()

    df = df.copy()
    if "ym" in df.columns:
        df = df.drop(columns=["ym"])
    df["ym"] = df["Date"].dt.to_period("M")
    df = df.merge(nino[["ym", "enso_vol_12m", "enso_vol_36m"]], on="ym", how="left")
    df = df.drop(columns=["ym"])

    n12 = df["enso_vol_12m"].notna().sum()
    n36 = df["enso_vol_36m"].notna().sum()
    print(f"  enso_vol_12m: {n12} daily obs matched ({df['enso_vol_12m'].isna().sum()} missing)")
    print(f"  enso_vol_36m: {n36} daily obs matched ({df['enso_vol_36m'].isna().sum()} missing)")

    return df


def plot_enso_vs_volatility(df):
    """Dual-axis plot: Nino 3.4 index and cocoa volatility over time."""
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Volatility on left axis
    ax1.plot(df["Date"], df["vol_yz_annual"] * 100 if "vol_yz_annual" in df.columns
             else df["sigma_m"] * np.sqrt(252) * 100,
             linewidth=0.5, color="saddlebrown", alpha=0.7, label="Cocoa Volatility")
    ax1.set_ylabel("Annualized Volatility (%)", color="saddlebrown")
    ax1.tick_params(axis="y", labelcolor="saddlebrown")

    # ENSO on right axis
    ax2 = ax1.twinx()
    # Monthly ENSO for cleaner plot
    monthly_enso = df.set_index("Date")["enso"].resample("ME").last()
    ax2.fill_between(monthly_enso.index, 0, monthly_enso.values,
                     where=monthly_enso.values >= 0, alpha=0.3, color="red",
                     label="El Nino (warm)")
    ax2.fill_between(monthly_enso.index, 0, monthly_enso.values,
                     where=monthly_enso.values < 0, alpha=0.3, color="blue",
                     label="La Nina (cool)")
    ax2.set_ylabel("Nino 3.4 Anomaly (°C)", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2.axhline(0, color="gray", linewidth=0.5)

    ax1.set_title("Cocoa Volatility and ENSO (Nino 3.4)")
    ax1.set_xlabel("Date")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "15b_enso_vs_volatility.png"))
    plt.close(fig)
    print("Saved: 15b_enso_vs_volatility.png")


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("Building HAR features...")
    df = build_har_features(df)

    print("Building horizon targets...")
    df = build_targets(df)

    print("Merging Nino 3.4 ENSO data...")
    df = merge_nino34(df)

    print("Building ENSO volatility columns...")
    df = build_enso_vol(df)

    # Save the full dataset for modeling
    out_path = os.path.join(PROC_DIR, "har_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved HAR dataset: {out_path}")

    # Count valid observations per horizon
    print("\nValid observations per horizon:")
    for h, label in HORIZONS.items():
        n = df[f"target_{label}"].notna().sum()
        print(f"  h={h:>3d} ({label:>3s}): {n:,} observations")

    print("\n--- Generating Plots ---")
    plot_feature_target_scatter(df)
    plot_feature_correlations(df)
    plot_horizon_decay(df)
    plot_har_components(df)
    plot_enso_vs_volatility(df)

    print("\nDone!")
