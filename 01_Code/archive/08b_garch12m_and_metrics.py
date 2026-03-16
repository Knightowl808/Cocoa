"""
08b_garch12m_and_metrics.py
===========================
Targeted script: runs only the missing GARCH 12m forecast,
then computes benchmark metrics from all existing forecast CSVs.

Run once in your terminal:
    py 08b_garch12m_and_metrics.py
"""

import pandas as pd
import numpy as np
from arch import arch_model
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
TBL_DIR  = os.path.join(BASE_DIR, "02_Output", "tables")

HORIZONS  = ["1d", "1w", "1m", "3m", "6m", "12m"]
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
WINDOW    = 2520


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------
def qlike(actual, forecast):
    f = np.maximum(forecast, 1e-10)
    a = np.maximum(actual,   1e-10)
    return np.mean(np.log(f**2) + (a**2) / (f**2))


def quantile_loss(actual, forecast, tau):
    u = actual - forecast
    return np.mean(u * (tau - (u < 0).astype(float)))


# ---------------------------------------------------------------------------
# Step 1: Run GARCH 12m if missing
# ---------------------------------------------------------------------------
garch_12m_path = os.path.join(PROC_DIR, "garch_forecasts_12m.csv")

if os.path.exists(garch_12m_path):
    print("garch_forecasts_12m.csv already exists — skipping computation.")
else:
    print("=" * 60)
    print("Running GARCH(1,1) for h=264 (12m) — this takes ~30-60 min")
    print("=" * 60)

    har_path = os.path.join(PROC_DIR, "har_dataset.csv")
    df = pd.read_csv(har_path)
    df["Date"] = pd.to_datetime(df["Date"])

    horizon_key   = 264
    horizon_label = "12m"
    target_col    = "target_12m"

    valid = df[["Date", "log_ret", "sigma_yz", target_col]].dropna().reset_index(drop=True)
    n     = len(valid)

    reestimate_every    = 22
    n_sims              = 5000
    cached_model_result = None
    last_estimated      = -reestimate_every

    dates          = []
    actuals        = []
    mean_forecasts = []
    q_forecasts    = {tau: [] for tau in QUANTILES}

    for t in tqdm(range(WINDOW, n), desc="GARCH h=12m"):
        y_actual = valid.iloc[t][target_col]
        if np.isnan(y_actual):
            continue

        if t - last_estimated >= reestimate_every:
            returns_train = valid.iloc[t - WINDOW : t]["log_ret"].values * 100
            try:
                garch = arch_model(returns_train, vol="Garch", p=1, q=1,
                                   mean="Constant", dist="normal")
                cached_model_result = garch.fit(disp="off", show_warning=False)
                last_estimated = t
            except Exception:
                pass

        if cached_model_result is None:
            continue

        try:
            fc = cached_model_result.forecast(horizon=horizon_key, reindex=False)
            mean_var = fc.variance.values[-1, :].mean() / 10000
            mean_vol = np.sqrt(max(mean_var, 1e-10))
            mean_forecasts.append(mean_vol)
        except Exception:
            mean_forecasts.append(np.nan)

        try:
            sim = cached_model_result.forecast(
                horizon=horizon_key, method="simulation",
                simulations=n_sims, reindex=False
            )
            sim_vars    = sim.simulations.variances[-1, :, :] / 10000
            sim_vols    = np.sqrt(np.maximum(sim_vars, 1e-10))
            avg_sim_vol = sim_vols.mean(axis=1)
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

    results.to_csv(garch_12m_path, index=False)
    print(f"  Saved: garch_forecasts_12m.csv  ({len(results)} forecasts)")


# ---------------------------------------------------------------------------
# Step 2: Compute metrics from ALL forecast CSVs
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Computing benchmark metrics")
print("=" * 60)

models = {
    "Historical Vol": "hist_vol_forecasts",
    "GARCH(1,1)":     "garch_forecasts",
}

qlike_rows = []
ql_rows    = []

for model_name, prefix in models.items():
    for label in HORIZONS:
        path = os.path.join(PROC_DIR, f"{prefix}_{label}.csv")
        if not os.path.exists(path):
            print(f"  MISSING: {prefix}_{label}.csv — skipping")
            continue

        df = pd.read_csv(path)

        # QLIKE on mean forecast
        sub = df.dropna(subset=["mean"])
        if len(sub):
            ql_val = qlike(sub["actual"].values, sub["mean"].values)
            qlike_rows.append({"Model": model_name, "Horizon": label, "QLIKE": ql_val})

        # Quantile loss
        for tau in QUANTILES:
            col = f"q{int(tau*100):02d}"
            if col in df.columns:
                sub_q = df.dropna(subset=[col])
                if len(sub_q):
                    loss = quantile_loss(sub_q["actual"].values, sub_q[col].values, tau)
                    ql_rows.append({
                        "Model": model_name, "Horizon": label,
                        "Quantile": tau, "QL": loss
                    })

# Save
qlike_df = pd.DataFrame(qlike_rows)
ql_df    = pd.DataFrame(ql_rows)

qlike_df.to_csv(os.path.join(TBL_DIR, "benchmark_mean_qlike.csv"), index=False)
ql_df.to_csv(   os.path.join(TBL_DIR, "benchmark_quantile_loss.csv"), index=False)
print("  Saved: benchmark_mean_qlike.csv")
print("  Saved: benchmark_quantile_loss.csv")

# Print summary tables
print("\n--- QLIKE (mean forecast, lower = better) ---")
pivot = qlike_df.pivot(index="Model", columns="Horizon", values="QLIKE")
pivot = pivot.reindex(columns=HORIZONS)
print(pivot.round(6).to_string())

print("\n--- Quantile Loss at tau=0.95 ---")
sub95 = ql_df[ql_df["Quantile"] == 0.95]
pivot95 = sub95.pivot(index="Model", columns="Horizon", values="QL")
pivot95 = pivot95.reindex(columns=HORIZONS)
print(pivot95.round(6).to_string())

print("\nDone.")
