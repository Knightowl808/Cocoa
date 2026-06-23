"""
07b_rolling_har_x_qr.py
=======================
Rolling-window HAR-X-QR estimation with ENSO augmentation.

Implements Eq. 3.6 from Methodology (HAR-X-QR with ENSO):
  Q_tau(sigma_{t+h} | F_t) = beta_0(tau) + beta_d(tau)*sigma_d
                              + beta_w(tau)*sigma_w + beta_m(tau)*sigma_m
                              + beta_enso(tau)*ENSO_t

ENSO inclusion based on:
  - Bouri et al. (2021): ENSO helps HAR-RV at 2-4 year horizons (oil)
  - Bonato et al. (2023): El Niño/La Niña improve agricultural vol forecasts
  - Ubilava (2018): ENSO affects cocoa supply through rainfall patterns

Quantiles: tau in {0.05, 0.25, 0.50, 0.75, 0.95}
Horizons:  h in {1, 5, 22, 66, 132, 264}

Rolling window: 2,520 days (~10 years), daily roll-forward.

Output: quantile forecasts with ENSO, quantile loss, coverage diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Maximum number of iterations")

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
WINDOW = 2520

# How often to re-estimate (every N days) — speeds up computation
# Set to 1 for full daily rolling, or e.g. 5 for weekly re-estimation
REESTIMATE_EVERY = 5


def load_data():
    """Load the HAR dataset."""
    path = os.path.join(PROC_DIR, "har_dataset.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ===================================================================
# 1. Quantile Loss (Check Function)
# ===================================================================
def quantile_loss(actual, forecast, tau):
    """
    Quantile loss / check function (Eq. 3.6 in Methodology).
    rho_tau(u) = u * (tau - I(u < 0))
    """
    u = actual - forecast
    return np.mean(u * (tau - (u < 0).astype(float)))


# ===================================================================
# 2. Rolling HAR-X-QR for One Horizon (WITH ENSO)
# ===================================================================
def rolling_har_x_qr(df, horizon_key, horizon_label):
    """
    Rolling-window HAR-X-QR estimation for a single horizon, all quantiles.

    KEY CHANGE: Includes ENSO as 4th predictor (Nino 3.4 anomaly).

    Returns DataFrame with columns: Date, actual, q05, q25, q50, q75, q95
    """
    target_col = f"target_{horizon_label}"
    # *** ADD ENSO HERE ***
    feature_cols = ["sigma_d", "sigma_w", "sigma_m", "enso"]

    # Prepare valid data
    valid = df[["Date"] + feature_cols + [target_col]].dropna().reset_index(drop=True)
    n = len(valid)

    if n < WINDOW + horizon_key + 100:
        print(f"  WARNING: Only {n} valid obs for h={horizon_key}")
        return None

    # Storage
    dates = []
    actuals = []
    q_forecasts = {tau: [] for tau in QUANTILES}

    # Cache for model coefficients (re-use between re-estimation days)
    cached_params = {tau: None for tau in QUANTILES}
    last_estimated = -REESTIMATE_EVERY  # force estimation on first iteration

    # Estimation window ends at t - h: rows s > t - h have forward targets
    # not fully realized at the forecast origin t (avoids look-ahead bias).
    h = horizon_key
    for t in tqdm(range(WINDOW + h, n), desc=f"  HAR-X-QR h={horizon_label}", leave=True):
        y_actual = valid.iloc[t][target_col]
        if np.isnan(y_actual):
            continue

        X_test = valid.iloc[t][feature_cols].values
        X_test_const = np.concatenate([[1.0], X_test])

        # Re-estimate every REESTIMATE_EVERY days
        if t - last_estimated >= REESTIMATE_EVERY:
            train = valid.iloc[t - h - WINDOW : t - h]
            X_train = sm.add_constant(train[feature_cols].values)
            y_train = train[target_col].values

            for tau in QUANTILES:
                try:
                    model = QuantReg(y_train, X_train)
                    result = model.fit(q=tau, max_iter=1000)
                    cached_params[tau] = result.params
                except Exception:
                    # If QR fails, keep previous params
                    pass

            last_estimated = t

        # Generate forecasts using cached parameters
        forecast_ok = True
        for tau in QUANTILES:
            if cached_params[tau] is None:
                forecast_ok = False
                break

        if not forecast_ok:
            continue

        dates.append(valid.iloc[t]["Date"])
        actuals.append(y_actual)
        for tau in QUANTILES:
            y_hat = X_test_const @ cached_params[tau]
            y_hat = max(y_hat, 1e-10)  # floor at zero
            q_forecasts[tau].append(y_hat)

    # Build results DataFrame
    results = pd.DataFrame({"Date": dates, "actual": actuals})
    for tau in QUANTILES:
        col_name = f"q{int(tau*100):02d}"
        results[col_name] = q_forecasts[tau]

    return results


# ===================================================================
# 3. Fan Chart: Quantile Forecasts Over Time
# ===================================================================
def plot_fan_chart(results, horizon_label):
    """Fan chart showing quantile bands and actual volatility."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Quantile bands (shaded)
    ax.fill_between(results["Date"], results["q05"], results["q95"],
                    alpha=0.15, color="steelblue", label="5–95% band")
    ax.fill_between(results["Date"], results["q25"], results["q75"],
                    alpha=0.25, color="steelblue", label="25–75% band")

    # Median and actual
    ax.plot(results["Date"], results["q50"],
            linewidth=1.0, color="red", label="Median forecast (τ=0.50)")
    ax.plot(results["Date"], results["actual"],
            linewidth=0.5, alpha=0.6, color="black", label="Actual")

    ax.set_title(f"HAR-X-QR Fan Chart (with ENSO) — Horizon {horizon_label}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Daily Volatility")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"23_har_x_qr_fanchart_{horizon_label}.png"))
    plt.close(fig)
    print(f"  Saved: 23_har_x_qr_fanchart_{horizon_label}.png")


# ===================================================================
# 4. Quantile Loss by Horizon and Quantile
# ===================================================================
def compute_all_losses(all_results):
    """Compute quantile loss for every (horizon, quantile) pair."""
    rows = []
    for label, results in all_results.items():
        for tau in QUANTILES:
            col = f"q{int(tau*100):02d}"
            loss = quantile_loss(results["actual"].values,
                                 results[col].values, tau)
            rows.append({"Horizon": label, "Quantile": tau, "QL": loss})

    loss_df = pd.DataFrame(rows)
    return loss_df


def plot_quantile_loss_heatmap(loss_df):
    """Heatmap of quantile loss across horizons and quantiles."""
    pivot = loss_df.pivot(index="Quantile", columns="Horizon", values="QL")
    # Reorder columns
    col_order = [v for v in HORIZONS.values() if v in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(10, 5))
    import seaborn as sns
    sns.heatmap(pivot, annot=True, fmt=".6f", cmap="YlOrRd", ax=ax)
    ax.set_title("Quantile Loss by Horizon and Quantile (HAR-X-QR with ENSO)")
    ax.set_ylabel("Quantile τ")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "24_har_x_qr_loss_heatmap.png"))
    plt.close(fig)
    print("Saved: 24_har_x_qr_loss_heatmap.png")


# ===================================================================
# 5. Coverage Analysis (for tau=0.95)
# ===================================================================
def coverage_analysis(all_results):
    """Check empirical coverage of the 95th percentile forecast."""
    print("\n=== Coverage Analysis (τ=0.95) ===")
    print(f"  Expected violation rate: 5.0%\n")

    rows = []
    for label, results in all_results.items():
        violations = (results["actual"] > results["q95"]).sum()
        n = len(results)
        rate = violations / n * 100
        rows.append({
            "Horizon": label,
            "N": n,
            "Violations": violations,
            "Rate (%)": rate,
        })
        print(f"  {label}: {violations}/{n} violations ({rate:.1f}%)")

    return pd.DataFrame(rows)


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("="*60)
    print("HAR-X-QR with ENSO Augmentation")
    print("="*60)
    print("\nModel specification:")
    print("  Q_τ(σ²_{t+h}) = β₀(τ) + β_d(τ)·σ²_d + β_w(τ)·σ²_w")
    print("                  + β_m(τ)·σ²_m + β_enso(τ)·ENSO_t")
    print("\n" + "="*60 + "\n")

    print("Loading HAR dataset...")
    df = load_data()

    # Check ENSO availability
    enso_available = df["enso"].notna().sum()
    print(f"ENSO data available: {enso_available}/{len(df)} days")

    all_results = {}

    for h, label in HORIZONS.items():
        print(f"\n{'='*50}")
        print(f"Horizon: h={h} ({label})")
        print(f"{'='*50}")

        results = rolling_har_x_qr(df, h, label)

        if results is not None and len(results) > 100:
            all_results[label] = results
            print(f"  Out-of-sample forecasts: {len(results)}")

            # Save
            results.to_csv(
                os.path.join(PROC_DIR, f"har_x_qr_forecasts_{label}.csv"),
                index=False
            )

            # Fan chart for select horizons
            if label in ["1d", "1m", "6m", "12m"]:
                plot_fan_chart(results, label)
        else:
            print(f"  Skipped: insufficient data")

    # Evaluation
    if all_results:
        print(f"\n{'='*50}")
        print("Evaluation")
        print(f"{'='*50}")

        loss_df = compute_all_losses(all_results)
        loss_df.to_csv(os.path.join(TBL_DIR, "har_x_qr_quantile_loss.csv"), index=False)

        print("\n=== Quantile Loss Summary ===")
        print(loss_df.pivot(index="Quantile", columns="Horizon", values="QL")
              .round(6).to_string())

        plot_quantile_loss_heatmap(loss_df)

        coverage = coverage_analysis(all_results)
        coverage.to_csv(os.path.join(TBL_DIR, "har_x_qr_coverage.csv"), index=False)

    print("\nDone!")
