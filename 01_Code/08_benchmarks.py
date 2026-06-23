"""
08_benchmarks.py
================
Benchmark models for comparison with HAR-QR.

Implements (from Methodology Section 3.4):
  1. Historical Volatility: rolling 252-day window + empirical quantiles
  2. GARCH(1,1): conditional variance model + simulated quantiles

Output:
  - Raw forecast CSVs in 00_Data/processed/
  - Summary metric tables in 02_Output/tables/:
      benchmark_mean_qlike.csv   -- QLIKE for HistVol and GARCH mean forecasts
      benchmark_quantile_loss.csv -- Quantile loss for HistVol and GARCH at all tau
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.stats import norm
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "02_Output", "figures")
TBL_DIR = os.path.join(BASE_DIR, "02_Output", "tables")
os.makedirs(PROC_DIR, exist_ok=True)
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
WINDOW = 2520        # rolling estimation window for HAR / comparison
HIST_WINDOW = 252    # historical volatility look-back (1 year)


def load_data():
    """Load the HAR dataset."""
    path = os.path.join(PROC_DIR, "har_dataset.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ===================================================================
# 1. Historical Volatility Benchmark
# ===================================================================
def rolling_historical_vol(df, horizon_key, horizon_label):
    """
    Historical volatility benchmark (Methodology Eq. 3.10).

    The forecast target is the h-day AVERAGE of daily volatility, so the
    benchmark must forecast that object — not daily volatility itself.
    We therefore build the backward-looking h-day rolling mean of sigma_yz
    (fully observable at the origin) and use:

      Mean forecast:     mean of the past HIST_WINDOW h-day averages
      Quantile forecast: empirical quantiles of the past HIST_WINDOW
                         h-day averages

    Using daily-sigma quantiles instead would mechanically overstate the
    dispersion of an h-day average and miscalibrate the benchmark at h > 1.

    The loop starts at WINDOW + h so that forecast origins coincide exactly
    with the HAR-family models and GARCH (identical evaluation samples).
    'enso' enters the dropna only for that same sample alignment.
    """
    target_col = f"target_{horizon_label}"

    valid = (df[["Date", "enso", "sigma_yz", target_col]]
             .dropna().reset_index(drop=True))
    n = len(valid)
    h = horizon_key

    if n < WINDOW + h + 100:
        return None

    # Backward h-day rolling average of daily sigma — same object as the
    # target, but built from past observations only (known at the origin).
    avg_h = valid["sigma_yz"].rolling(window=h, min_periods=h).mean()

    dates = []
    actuals = []
    q_forecasts = {tau: [] for tau in QUANTILES}
    mean_forecasts = []

    for t in tqdm(range(WINDOW + h, n), desc=f"  HistVol h={horizon_label}", leave=True):
        y_actual = valid.iloc[t][target_col]
        if np.isnan(y_actual):
            continue

        # Past HIST_WINDOW values of the h-day average (ends at t-1)
        past_avg = avg_h.iloc[t - HIST_WINDOW : t].values

        # Mean forecast = mean of past h-day averages
        mean_forecasts.append(past_avg.mean())

        # Quantile forecasts = empirical quantiles of past h-day averages
        for tau in QUANTILES:
            q_forecasts[tau].append(np.quantile(past_avg, tau))

        dates.append(valid.iloc[t]["Date"])
        actuals.append(y_actual)

    results = pd.DataFrame({"Date": dates, "actual": actuals, "mean": mean_forecasts})
    for tau in QUANTILES:
        results[f"q{int(tau*100):02d}"] = q_forecasts[tau]

    return results


# ===================================================================
# 2. GARCH(1,1) Benchmark
# ===================================================================
def rolling_garch(df, horizon_key, horizon_label, reestimate_every=22):
    """
    GARCH(1,1) benchmark (Methodology Eq. 3.11-3.12).

    Mean forecast: closed-form multi-step variance recursion, averaged over
    the horizon, starting from the proper one-step-ahead variance
        sigma^2_{t+1|t} = omega + alpha * eps_t^2 + beta * sigma_t^2.
    Between re-estimations the conditional-variance state is updated daily
    with the realized return, so the forecast conditions on all information
    up to the origin t even though parameters are refit only every
    `reestimate_every` days.

    Quantile forecasts: Gaussian quantiles centered on the mean forecast.
    The uncertainty scale is the in-window std of the h-day AVERAGE of
    sigma_yz (the target object), not of daily sigma — daily dispersion
    would mechanically overstate the spread of an h-day average.

    The loop starts at WINDOW + h so that forecast origins coincide exactly
    with the HAR-family models. Parameters are fit on returns up to t only
    (no target involved), so there is no look-ahead in the estimation.
    'enso' enters the dropna only for sample alignment.
    """
    target_col = f"target_{horizon_label}"

    # GARCH needs returns, not volatility
    valid = (df[["Date", "enso", "log_ret", "sigma_yz", target_col]]
             .dropna().reset_index(drop=True))
    n = len(valid)
    h = horizon_key

    if n < WINDOW + h + 100:
        return None

    # Backward h-day average of daily sigma — used as the uncertainty scale
    # for the Gaussian quantiles (same object as the forecast target).
    avg_h = valid["sigma_yz"].rolling(window=h, min_periods=h).mean()

    dates = []
    actuals = []
    mean_forecasts = []
    q_forecasts = {tau: [] for tau in QUANTILES}

    # Cached parameters (pct units) and the conditional-variance state.
    # sigma2_state is sigma^2_{t+1|t} in pct^2 for the CURRENT loop index t,
    # i.e. the one-step-ahead variance given information up to t.
    cached = None          # dict: mu, omega_p, alpha, beta
    sigma2_state = None    # pct^2
    last_estimated = None

    for t in tqdm(range(WINDOW + h, n), desc=f"  GARCH h={horizon_label}", leave=True):
        y_actual = valid.iloc[t][target_col]
        if np.isnan(y_actual):
            continue

        if last_estimated is None or t - last_estimated >= reestimate_every:
            # Refit on the WINDOW returns ending at t-1 (info up to t)
            returns_train = valid.iloc[t - WINDOW : t]["log_ret"].values * 100
            try:
                garch = arch_model(returns_train, vol="Garch", p=1, q=1,
                                   mean="Constant", dist="normal")
                res = garch.fit(disp="off", show_warning=False)
                mu      = float(res.params.get("mu", 0.0))
                omega_p = float(res.params["omega"])
                alpha   = float(res.params.get("alpha[1]", res.params.get("alpha", 0.0)))
                beta    = float(res.params.get("beta[1]",  res.params.get("beta",  0.0)))
                cached  = {"mu": mu, "omega_p": omega_p,
                           "alpha": alpha, "beta": beta}
                # One-step-ahead variance for day t (the first forecast step):
                # omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
                eps_last = returns_train[-1] - mu
                sig_last = float(res.conditional_volatility[-1])
                sigma2_state = omega_p + alpha * eps_last ** 2 + beta * sig_last ** 2
                last_estimated = t
            except Exception:
                pass  # keep previous parameters and state
        elif cached is not None and sigma2_state is not None:
            # No refit: advance the variance recursion by one day using the
            # realized return of day t-1 (known at the origin t).
            ret_prev = valid.iloc[t - 1]["log_ret"] * 100
            eps_prev = ret_prev - cached["mu"]
            sigma2_state = (cached["omega_p"]
                            + cached["alpha"] * eps_prev ** 2
                            + cached["beta"] * sigma2_state)

        if cached is None or sigma2_state is None:
            continue

        try:
            # Rescale from pct^2 to fraction^2
            omega = cached["omega_p"] / 10000.0
            persistence = cached["alpha"] + cached["beta"]
            sigma_lr2 = omega / max(1.0 - persistence, 1e-8)
            sigma2_t1 = sigma2_state / 10000.0

            # k-step variance forecasts (steps 1 ... h), then average
            step_vars = np.array([
                sigma_lr2 + persistence ** (k - 1) * (sigma2_t1 - sigma_lr2)
                for k in range(1, h + 1)
            ])
            step_vars = np.maximum(step_vars, 1e-10)

            mean_vol = np.sqrt(step_vars.mean())
            mean_forecasts.append(mean_vol)

            # Gaussian quantiles around the mean forecast; scale = std of the
            # h-day average of sigma over the estimation window.
            sigma_uncertainty = avg_h.iloc[t - WINDOW : t].std()
            for tau in QUANTILES:
                q_forecasts[tau].append(
                    max(norm.ppf(tau, loc=mean_vol, scale=sigma_uncertainty), 1e-10)
                )
        except Exception:
            mean_forecasts.append(np.nan)
            for tau in QUANTILES:
                q_forecasts[tau].append(np.nan)

        dates.append(valid.iloc[t]["Date"])
        actuals.append(y_actual)

    results = pd.DataFrame({"Date": dates, "actual": actuals, "mean": mean_forecasts})
    for tau in QUANTILES:
        results[f"q{int(tau*100):02d}"] = q_forecasts[tau]

    return results


# ===================================================================
# 3. Evaluation Metrics
# ===================================================================
def qlike(actual, forecast):
    """QLIKE loss (Patton 2011). Lower is better."""
    f = np.maximum(forecast, 1e-10)
    a = np.maximum(actual, 1e-10)
    return np.mean(np.log(f ** 2) + (a ** 2) / (f ** 2))


def quantile_loss(actual, forecast, tau):
    """Check function / pinball loss. Lower is better."""
    u = actual - forecast
    return np.mean(u * (tau - (u < 0).astype(float)))


def compute_metrics(results_dict, model_name):
    """
    Compute QLIKE (mean forecast) and quantile loss for all horizons.

    results_dict: {label: DataFrame with columns actual, mean, q05, q25, q50, q75, q95}
    Returns two DataFrames: qlike_rows, ql_rows.
    """
    qlike_rows = []
    ql_rows = []

    for label, df in results_dict.items():
        df = df.dropna(subset=["mean"])

        # QLIKE on mean forecast
        ql_val = qlike(df["actual"].values, df["mean"].values)
        qlike_rows.append({"Model": model_name, "Horizon": label, "QLIKE": ql_val})

        # Quantile loss for each tau
        for tau in QUANTILES:
            col = f"q{int(tau * 100):02d}"
            if col in df.columns:
                df_q = df.dropna(subset=[col])
                loss = quantile_loss(df_q["actual"].values, df_q[col].values, tau)
                ql_rows.append({
                    "Model": model_name, "Horizon": label,
                    "Quantile": tau, "QL": loss
                })

    return pd.DataFrame(qlike_rows), pd.DataFrame(ql_rows)


# ===================================================================
# 4. Comparison Plots
# ===================================================================
def plot_benchmark_comparison(har_qr_results, hist_results, garch_results,
                              horizon_label):
    """Compare HAR-QR median vs benchmarks (mean forecasts)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Actual
    ax.plot(har_qr_results["Date"], har_qr_results["actual"],
            linewidth=0.4, alpha=0.5, color="black", label="Actual")

    # HAR-QR median
    ax.plot(har_qr_results["Date"], har_qr_results["q50"],
            linewidth=0.8, color="red", label="HAR-QR (median)")

    # Historical
    if hist_results is not None:
        ax.plot(hist_results["Date"], hist_results["mean"],
                linewidth=0.8, color="forestgreen", alpha=0.7,
                label="Historical Vol")

    # GARCH
    if garch_results is not None:
        garch_clean = garch_results.dropna(subset=["mean"])
        ax.plot(garch_clean["Date"], garch_clean["mean"],
                linewidth=0.8, color="darkorange", alpha=0.7,
                label="GARCH(1,1)")

    ax.set_title(f"Model Comparison — Horizon {horizon_label}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Daily Volatility")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"22_benchmark_comparison_{horizon_label}.png"))
    plt.close(fig)
    print(f"  Saved: 22_benchmark_comparison_{horizon_label}.png")


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("Loading HAR dataset...")
    df = load_data()

    # All six horizons (matches Results section tables)
    all_horizons = {1: "1d", 5: "1w", 22: "1m", 66: "3m", 132: "6m", 264: "12m"}

    hist_results = {}
    garch_results = {}

    for h, label in all_horizons.items():
        print(f"\n{'='*50}")
        print(f"Horizon: h={h} ({label})")
        print(f"{'='*50}")

        # Historical Vol
        print("  Running Historical Volatility benchmark...")
        hist = rolling_historical_vol(df, h, label)
        if hist is not None:
            print(f"    {len(hist)} forecasts")
            hist.to_csv(
                os.path.join(PROC_DIR, f"hist_vol_forecasts_{label}.csv"),
                index=False
            )
            hist_results[label] = hist

        # GARCH
        print("  Running GARCH(1,1) benchmark...")
        garch = rolling_garch(df, h, label, reestimate_every=22)
        if garch is not None:
            print(f"    {len(garch)} forecasts")
            garch.to_csv(
                os.path.join(PROC_DIR, f"garch_forecasts_{label}.csv"),
                index=False
            )
            garch_results[label] = garch

        # Comparison plot (optional — needs HAR-QR raw forecasts)
        har_qr_path = os.path.join(PROC_DIR, f"har_qr_forecasts_{label}.csv")
        if os.path.exists(har_qr_path):
            har_qr = pd.read_csv(har_qr_path)
            har_qr["Date"] = pd.to_datetime(har_qr["Date"])
            plot_benchmark_comparison(har_qr, hist, garch, label)
        else:
            print(f"  (Skipping comparison plot — HAR-QR raw forecasts not found)")

    # ------------------------------------------------------------------
    # Compute and save evaluation metrics
    # ------------------------------------------------------------------
    print("\n" + "="*50)
    print("Computing evaluation metrics")
    print("="*50)

    all_qlike = []
    all_ql = []

    if hist_results:
        q, ql = compute_metrics(hist_results, "Historical Vol")
        all_qlike.append(q)
        all_ql.append(ql)

    if garch_results:
        q, ql = compute_metrics(garch_results, "GARCH(1,1)")
        all_qlike.append(q)
        all_ql.append(ql)

    if all_qlike:
        qlike_df = pd.concat(all_qlike, ignore_index=True)
        qlike_df.to_csv(
            os.path.join(TBL_DIR, "benchmark_mean_qlike.csv"), index=False
        )
        print("\nMean QLIKE summary:")
        print(qlike_df.pivot(index="Model", columns="Horizon", values="QLIKE")
              .round(6).to_string())

    if all_ql:
        ql_df = pd.concat(all_ql, ignore_index=True)
        ql_df.to_csv(
            os.path.join(TBL_DIR, "benchmark_quantile_loss.csv"), index=False
        )
        print("\nQuantile loss at tau=0.95:")
        subset = ql_df[ql_df["Quantile"] == 0.95]
        print(subset.pivot(index="Model", columns="Horizon", values="QL")
              .round(6).to_string())

    print("\nDone!")
