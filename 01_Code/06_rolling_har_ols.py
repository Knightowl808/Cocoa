"""
06_rolling_har_ols.py
=====================
Rolling-window HAR-OLS estimation (benchmark model).

This implements the out-of-sample framework from Methodology Section 3.5:
  - Rolling window: 2,520 trading days (~10 years)
  - Estimate HAR-OLS at each forecast origin
  - Generate mean forecasts for all 6 horizons
  - Compute QLIKE loss

This script demonstrates the rolling-window loop that will also be
used for HAR-QR. HAR-OLS serves as the mean-forecast benchmark.

Output: rolling forecasts + evaluation metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy.linalg import lstsq
from tqdm import tqdm
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
WINDOW = 2520  # ~10 years of trading days


def load_data():
    """Load the HAR dataset."""
    path = os.path.join(PROC_DIR, "har_dataset.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ===================================================================
# 1. QLIKE Loss Function
# ===================================================================
def qlike(actual, forecast):
    """
    QLIKE loss (Eq. 3.8 in Methodology).
    QLIKE = mean( log(forecast^2) + actual^2 / forecast^2 )

    Lower is better. Robust to noisy volatility proxies (Patton 2011).
    """
    # Avoid division by zero or log of zero
    f = np.maximum(forecast, 1e-10)
    a = np.maximum(actual, 1e-10)
    return np.mean(np.log(f ** 2) + (a ** 2) / (f ** 2))


# ===================================================================
# 2. Rolling Window HAR-OLS
# ===================================================================
def rolling_har_ols(df, horizon_key, horizon_label):
    """
    Rolling-window HAR-OLS estimation for a single horizon.

    Parameters
    ----------
    df : DataFrame with sigma_d, sigma_w, sigma_m, and target columns
    horizon_key : int (e.g. 22)
    horizon_label : str (e.g. '1m')

    Returns
    -------
    results : DataFrame with Date, actual, forecast columns
    """
    target_col = f"target_{horizon_label}"
    feature_cols = ["sigma_d", "sigma_w", "sigma_m"]

    # Drop rows where features or target are NaN. 'enso' is NOT a regressor
    # here; it is included in the dropna only to align the evaluation sample
    # with the ENSO-augmented models (07b/07d/07h), so that loss tables are
    # computed over identical forecast dates across all models.
    valid = (df[["Date", "enso"] + feature_cols + [target_col]]
             .dropna().reset_index(drop=True))
    n = len(valid)

    if n < WINDOW + horizon_key + 100:
        print(f"  WARNING: Only {n} valid obs for h={horizon_key}, need {WINDOW + horizon_key}+")
        return None

    # Storage for out-of-sample forecasts
    dates = []
    actuals = []
    forecasts = []
    q_forecasts = {tau: [] for tau in QUANTILES}

    # Rolling loop. The estimation window ends at t - h, not t: the target of
    # row s covers sigma over s+1 ... s+h, so rows s > t - h have targets that
    # are not fully realized at the forecast origin t. Ending the window at
    # t - h ensures only information available at t enters the estimation.
    h = horizon_key
    for t in tqdm(range(WINDOW + h, n), desc=f"  HAR-OLS h={horizon_label}", leave=True):
        # Estimation window: [t - h - WINDOW, t - h)
        train = valid.iloc[t - h - WINDOW : t - h]
        X_train = train[feature_cols].values
        y_train = train[target_col].values

        # Forecast origin: t
        X_test = valid.iloc[t : t + 1][feature_cols].values
        y_actual = valid.iloc[t][target_col]

        # Skip if actual is NaN (can happen near end of sample)
        if np.isnan(y_actual):
            continue

        # Fit OLS via numpy (X with intercept column)
        ones_train = np.column_stack([np.ones(len(X_train)), X_train])
        ones_test = np.column_stack([np.ones(len(X_test)), X_test])
        beta, _, _, _ = lstsq(ones_train, y_train, rcond=None)
        y_hat = (ones_test @ beta)[0]

        # Floor forecast at zero (volatility can't be negative)
        y_hat = max(y_hat, 1e-10)

        # Pseudo-quantile forecasts from in-sample residuals (Johannes, 17.03.2026):
        #   q̂_τ = ŷ + ε̂_τ  where ε̂_τ is the τ-th empirical quantile of training residuals
        # This gives a distributional forecast consistent with the OLS point forecast.
        in_sample_fits = (ones_train @ beta)
        residuals = y_train - in_sample_fits
        for tau in QUANTILES:
            eps_tau = np.quantile(residuals, tau)
            q_forecasts[tau].append(max(y_hat + eps_tau, 1e-10))

        dates.append(valid.iloc[t]["Date"])
        actuals.append(y_actual)
        forecasts.append(y_hat)

    results = pd.DataFrame({
        "Date": dates,
        "actual": actuals,
        "forecast": forecasts,
    })
    for tau in QUANTILES:
        results[f"q{int(tau*100):02d}"] = q_forecasts[tau]

    return results


# ===================================================================
# 3. Forecast vs Actual Plot
# ===================================================================
def plot_forecast_vs_actual(results, horizon_label):
    """Time series of forecast vs actual volatility."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(results["Date"], results["actual"],
            linewidth=0.5, alpha=0.7, color="steelblue", label="Actual")
    ax.plot(results["Date"], results["forecast"],
            linewidth=0.5, alpha=0.7, color="red", label="HAR-OLS Forecast")

    loss = qlike(results["actual"].values, results["forecast"].values)
    ax.set_title(f"HAR-OLS Forecast vs Actual — Horizon {horizon_label} (QLIKE={loss:.4f})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Daily Volatility")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"18_har_ols_forecast_{horizon_label}.png"))
    plt.close(fig)
    print(f"  Saved: 18_har_ols_forecast_{horizon_label}.png")


# ===================================================================
# 4. Loss Summary Across Horizons
# ===================================================================
def plot_loss_by_horizon(loss_dict):
    """Bar chart of QLIKE loss by horizon."""
    fig, ax = plt.subplots(figsize=(8, 5))

    horizons = list(loss_dict.keys())
    losses = list(loss_dict.values())

    ax.bar(horizons, losses, color="steelblue", alpha=0.8)
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("QLIKE Loss")
    ax.set_title("HAR-OLS Out-of-Sample QLIKE Loss by Horizon")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "19_har_ols_loss_by_horizon.png"))
    plt.close(fig)
    print("Saved: 19_har_ols_loss_by_horizon.png")


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("Loading HAR dataset...")
    df = load_data()

    loss_dict = {}
    all_results = {}

    for h, label in HORIZONS.items():
        print(f"\n--- Horizon: h={h} ({label}) ---")
        results = rolling_har_ols(df, h, label)

        if results is not None and len(results) > 100:
            loss = qlike(results["actual"].values, results["forecast"].values)
            loss_dict[label] = loss
            all_results[label] = results

            print(f"  Out-of-sample forecasts: {len(results)}")
            print(f"  QLIKE loss: {loss:.6f}")

            # Save forecasts
            results.to_csv(
                os.path.join(PROC_DIR, f"har_ols_forecasts_{label}.csv"),
                index=False
            )

            # Plot for select horizons (1d, 1m, 12m)
            if label in ["1d", "1m", "12m"]:
                plot_forecast_vs_actual(results, label)
        else:
            print(f"  Skipped: insufficient data")

    # Summary
    if loss_dict:
        print("\n=== QLIKE Loss Summary ===")
        for label, loss in loss_dict.items():
            print(f"  {label:>4s}: {loss:.6f}")

        # Save loss table
        pd.Series(loss_dict, name="QLIKE").to_csv(
            os.path.join(TBL_DIR, "har_ols_qlike_loss.csv")
        )

        plot_loss_by_horizon(loss_dict)

    print("\nDone!")
