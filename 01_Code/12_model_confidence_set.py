"""
12_model_confidence_set.py
===========================
Model Confidence Set (MCS) of Hansen et al. (2011).

For each horizon and evaluation criterion, this script determines the set
of models M* that contains the best model at the 90% confidence level.

Two MCS exercises:
  (A) QLIKE on mean/median forecasts  (HistVol, GARCH, HAR-OLS, HAR-QR median)
  (B) Quantile Loss at each tau        (HistVol, GARCH, HAR-QR, HAR-X-QR)

Requires: forecast CSV files from scripts 06-08 + 07b.
Uses: arch.bootstrap.MCS (stationary bootstrap, 10 000 replications).

Output:
  - tables/mcs_qlike.csv          (panel A)
  - tables/mcs_quantile_loss.csv  (panel B)
"""

import os
import warnings
import numpy as np
import pandas as pd
from arch.bootstrap import MCS

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
TBL_DIR = os.path.join(BASE_DIR, "02_Output", "tables")
os.makedirs(TBL_DIR, exist_ok=True)

HORIZONS = ["1d", "1w", "1m", "3m", "6m", "12m"]
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
MCS_SIZE = 0.10          # 90% confidence level
MCS_REPS = 10_000
MCS_METHOD = "max"       # max-t statistic (Hansen et al. 2011 recommend)
MCS_BOOTSTRAP = "stationary"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_forecast(model_tag, horizon):
    """Load a forecast CSV; return None if missing."""
    path = os.path.join(PROC_DIR, f"{model_tag}_forecasts_{horizon}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def qlike_losses(actual, forecast):
    """Element-wise QLIKE loss (not averaged). Returns array of length T."""
    f = np.maximum(forecast, 1e-10)
    a = np.maximum(actual, 1e-10)
    return np.log(f ** 2) + (a ** 2) / (f ** 2)


def quantile_losses(actual, forecast, tau):
    """Element-wise quantile (check-function) loss. Returns array of length T."""
    u = actual - forecast
    return u * (tau - (u < 0).astype(float))


def run_mcs(loss_df):
    """Run MCS on a DataFrame of losses (columns = models, rows = dates).
    Returns dict with 'pvalues' (Series) and 'included' (list)."""
    loss_df = loss_df.dropna(axis=0, how="any")
    if loss_df.shape[1] < 2:
        # Need at least 2 models to compare
        return None
    if loss_df.shape[0] < 100:
        print(f"    WARNING: only {loss_df.shape[0]} obs, skipping MCS")
        return None
    mcs = MCS(loss_df, size=MCS_SIZE, reps=MCS_REPS,
              method=MCS_METHOD, bootstrap=MCS_BOOTSTRAP)
    mcs.compute()
    return {"pvalues": mcs.pvalues, "included": list(mcs.included)}


# ---------------------------------------------------------------------------
# Panel A: QLIKE on mean / median forecasts
# ---------------------------------------------------------------------------
def mcs_qlike():
    """MCS using QLIKE loss across mean-forecast models."""
    print("=" * 60)
    print("PANEL A: MCS on QLIKE (mean / median forecasts)")
    print("=" * 60)

    # Model tag -> (file prefix, column for point forecast)
    models = {
        "Historical Vol": ("hist_vol", "mean"),
        "GARCH(1,1)":     ("garch", "mean"),
        "HAR-OLS":        ("har_ols", "forecast"),
        "HAR-QR (med.)":  ("har_qr", "q50"),
        "HAR-X-QR (med.)": ("har_x_qr", "q50"),
    }

    rows = []
    for horizon in HORIZONS:
        print(f"\n  Horizon: {horizon}")
        loss_dict = {}
        dates_list = []

        for model_name, (tag, col) in models.items():
            df = load_forecast(tag, horizon)
            if df is None:
                print(f"    {model_name}: file missing, skipping")
                continue
            if col not in df.columns:
                print(f"    {model_name}: column '{col}' missing, skipping")
                continue
            sub = df[["Date", "actual", col]].dropna()
            loss = qlike_losses(sub["actual"].values, sub[col].values)
            loss_dict[model_name] = pd.Series(loss, index=sub["Date"].values)
            dates_list.append(set(sub["Date"].values))

        if len(loss_dict) < 2:
            print(f"    Not enough models for MCS at {horizon}")
            continue

        # Align on common dates
        common = sorted(set.intersection(*dates_list))
        loss_df = pd.DataFrame({k: v.reindex(common) for k, v in loss_dict.items()})
        loss_df = loss_df.dropna()

        print(f"    Models: {list(loss_df.columns)} | Obs: {len(loss_df)}")
        result = run_mcs(loss_df)
        if result is None:
            continue

        pvals = result["pvalues"]
        included = result["included"]
        print(f"    M* (90%): {included}")

        for model_name in loss_df.columns:
            pv = pvals.loc[model_name, "Pvalue"] if model_name in pvals.index else np.nan
            rows.append({
                "Horizon": horizon,
                "Model": model_name,
                "Mean QLIKE": loss_df[model_name].mean(),
                "MCS p-value": pv,
                "In M*": model_name in included,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(TBL_DIR, "mcs_qlike.csv"), index=False)
    print(f"\nSaved: mcs_qlike.csv")
    return df_out


# ---------------------------------------------------------------------------
# Panel B: Quantile Loss across quantile-forecast models
# ---------------------------------------------------------------------------
def mcs_quantile_loss():
    """MCS using quantile loss, separately for each (horizon, tau)."""
    print("\n" + "=" * 60)
    print("PANEL B: MCS on Quantile Loss")
    print("=" * 60)

    # Models that produce quantile forecasts
    models = {
        "Historical Vol": "hist_vol",
        "GARCH(1,1)":     "garch",
        "HAR-QR":         "har_qr",
        "HAR-X-QR":       "har_x_qr",
    }

    rows = []
    for horizon in HORIZONS:
        for tau in QUANTILES:
            col = f"q{int(tau * 100):02d}"
            loss_dict = {}
            dates_list = []

            for model_name, tag in models.items():
                df = load_forecast(tag, horizon)
                if df is None:
                    continue
                if col not in df.columns:
                    continue
                sub = df[["Date", "actual", col]].dropna()
                loss = quantile_losses(sub["actual"].values, sub[col].values, tau)
                loss_dict[model_name] = pd.Series(loss, index=sub["Date"].values)
                dates_list.append(set(sub["Date"].values))

            if len(loss_dict) < 2:
                continue

            common = sorted(set.intersection(*dates_list))
            loss_df = pd.DataFrame({k: v.reindex(common) for k, v in loss_dict.items()})
            loss_df = loss_df.dropna()

            result = run_mcs(loss_df)
            if result is None:
                continue

            pvals = result["pvalues"]
            included = result["included"]

            print(f"  {horizon} | tau={tau:.2f} | M*: {included}")

            for model_name in loss_df.columns:
                pv = pvals.loc[model_name, "Pvalue"] if model_name in pvals.index else np.nan
                rows.append({
                    "Horizon": horizon,
                    "Quantile": tau,
                    "Model": model_name,
                    "Mean QL": loss_df[model_name].mean(),
                    "MCS p-value": pv,
                    "In M*": model_name in included,
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(TBL_DIR, "mcs_quantile_loss.csv"), index=False)
    print(f"\nSaved: mcs_quantile_loss.csv")
    return df_out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(df_qlike, df_ql):
    """Print a concise summary of MCS results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if df_qlike is not None and len(df_qlike) > 0:
        print("\nPanel A — QLIKE (mean forecasts):")
        for h in HORIZONS:
            sub = df_qlike[(df_qlike["Horizon"] == h) & (df_qlike["In M*"])]
            if len(sub) > 0:
                print(f"  {h:>4s}: M* = {list(sub['Model'])}")

    if df_ql is not None and len(df_ql) > 0:
        print("\nPanel B — Quantile Loss (selected quantiles):")
        for tau in [0.05, 0.50, 0.95]:
            print(f"\n  tau = {tau:.2f}:")
            for h in HORIZONS:
                sub = df_ql[(df_ql["Horizon"] == h) &
                            (df_ql["Quantile"] == tau) &
                            (df_ql["In M*"])]
                if len(sub) > 0:
                    print(f"    {h:>4s}: M* = {list(sub['Model'])}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Model Confidence Set Analysis")
    print("Hansen et al. (2011)")
    print(f"Confidence level: {(1 - MCS_SIZE)*100:.0f}% | "
          f"Bootstrap reps: {MCS_REPS:,} | Method: {MCS_METHOD}")
    print("=" * 60)

    df_qlike = mcs_qlike()
    df_ql = mcs_quantile_loss()
    print_summary(df_qlike, df_ql)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
