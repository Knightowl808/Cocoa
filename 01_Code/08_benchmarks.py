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

    For mean forecast: rolling 252-day standard deviation of Yang-Zhang σ
    For quantile forecasts: empirical quantiles of the past 252 days of σ

    This is the simplest possible benchmark — no model, just look backward.
    """
    target_col = f"target_{horizon_label}"
    sigma = df["sigma_yz"].copy()

    valid = df[["Date", "sigma_yz", target_col]].dropna().reset_index(drop=True)
    n = len(valid)

    if n < HIST_WINDOW + 100:
        return None

    dates = []
    actuals = []
    q_forecasts = {tau: [] for tau in QUANTILES}
    mean_forecasts = []

    for t in tqdm(range(HIST_WINDOW, n), desc=f"  HistVol h={horizon_label}", leave=True):
        y_actual = valid.iloc[t][target_col]
        if np.isnan(y_actual):
            continue

        # Past 252 days of daily volatility
        past_vol = valid.iloc[t - HIST_WINDOW : t]["sigma_yz"].values

        # Mean forecast = mean of past volatilities
        mean_forecasts.append(past_vol.mean())

        # Quantile forecasts = empirical quantiles
        for tau in QUANTILES:
            q_forecasts[tau].append(np.quantile(past_vol, tau))

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

    For mean forecast: GARCH multi-step forecast (analytical)
    For quantile forecasts: simulate 5,000 paths, take quantiles

    Re-estimates every `reestimate_every` days to speed up.
    """
    target_col = f"target_{horizon_label}"

    # GARCH needs returns, not volatility
    valid = df[["Date", "log_ret", "sigma_yz", target_col]].dropna().reset_index(drop=True)
    n = len(valid)

    if n < WINDOW + 100:
        return None

    dates = []
    actuals = []
    mean_forecasts = []
    q_forecasts = {tau: [] for tau in QUANTILES}

    # Cache GARCH parameters
    cached_model_result = None
    last_estimated = -reestimate_every

    n_sims = 5000  # simulation paths for quantiles

    for t in tqdm(range(WINDOW, n), desc=f"  GARCH h={horizon_label}", leave=True):
        y_actual = valid.iloc[t][target_col]
        if np.isnan(y_actual):
            continue

        # Re-estimate GARCH
        if t - last_estimated >= reestimate_every:
            returns_train = valid.iloc[t - WINDOW : t]["log_ret"].values * 100  # scale for arch
            try:
                garch = arch_model(returns_train, vol="Garch", p=1, q=1,
                                   mean="Constant", dist="normal")
                cached_model_result = garch.fit(disp="off", show_warning=False)
                last_estimated = t
            except Exception:
                pass  # keep previous model

        if cached_model_result is None:
            continue

        # Multi-step mean forecast (analytical)
        try:
            forecasts = cached_model_result.forecast(
                horizon=horizon_key, reindex=False
            )
            # Average variance over horizon, convert back from pct scale
            mean_var = forecasts.variance.values[-1, :].mean() / 10000
            mean_vol = np.sqrt(max(mean_var, 1e-10))
            mean_forecasts.append(mean_vol)
        except Exception:
            mean_forecasts.append(np.nan)

        # Simulated quantile forecasts
        try:
            sim = cached_model_result.forecast(
                horizon=horizon_key, method="simulation",
                simulations=n_sims, reindex=False
            )
            # sim.simulations.variances: shape (1, n_sims, horizon)
            sim_vars = sim.simulations.variances[-1, :, :] / 10000  # rescale
            sim_vols = np.sqrt(np.maximum(sim_vars, 1e-10))
            avg_sim_vol = sim_vols.mean(axis=1)  # average over horizon for each sim

            for tau in QUANTILES:
                q_forecasts[tau].append(np.quantile(avg_sim_vol, tau))
        except Exception:
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
