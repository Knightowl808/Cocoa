"""
07f_rolling_har_x_lagged_qr.py
================================
Lag-decay diagnostic: HAR-X-QR with lagged ENSO predictors.

Tests whether ENSO predictive power decays as the lag increases.
This directly addresses Dr. Bleher's question (28.04.26): "Welche Lags
hast du getestet?"

Theory check:
  The agronomic transmission chain is ~12 months (ENSO → rainfall →
  harvest → price). If this is the true mechanism, power should
  strengthen at intermediate lags (3–6 months), not decay.

  If instead power decays monotonically with lag, this contradicts the
  6–12 month transmission story and suggests the contemporaneous ENSO
  level is acting as a general regime indicator, not a causal predictor.

Lag construction: .shift(L) on the MONTHLY nino34 series before
  broadcasting to daily. At forecast date t in month M, the predictor
  used is the ENSO value from month M-L (L months ago). Fully causal.

Reduced scope (runtime management):
  Horizons:  1m, 6m, 12m  (the three most relevant for thesis)
  Quantile:  tau=0.95 only  (the upper tail Dr. Bleher cares about)
  Lags:      0, 1, 3, 6 months

Output:
  02_Output/tables/lagged_enso_summary.csv
    — 12 rows: Lag x Horizon, columns: Lag, Horizon, Quantile, QL, N
  Pivot (printed to console): 3x4 table for direct thesis inclusion
"""

import pandas as pd
import numpy as np
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
TBL_DIR  = os.path.join(BASE_DIR, "02_Output", "tables")
os.makedirs(TBL_DIR, exist_ok=True)

HORIZONS_SUBSET  = {22: "1m", 132: "6m", 264: "12m"}
QUANTILES_SUBSET = [0.95]
LAGS             = [0, 1, 3, 6, 12]
WINDOW           = 2520
REESTIMATE_EVERY = 5

HORIZON_ORDER = pd.CategoricalDtype(
    categories=["1m", "6m", "12m"], ordered=True
)


# ===================================================================
# 1. Data loading
# ===================================================================
def load_data():
    path = os.path.join(PROC_DIR, "har_dataset.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ===================================================================
# 2. Build lagged ENSO column
# ===================================================================
def merge_lagged_enso(df, lag_months):
    """
    Merge L-month lagged ENSO into daily DataFrame.

    .shift(L) on sorted monthly series: row i gets value from row i-L,
    so month M receives the value from month M-L. Fully causal.
    lag_months=0 gives the contemporaneous value (same as 'enso' column).
    """
    nino_path = os.path.join(PROC_DIR, "nino34_clean.csv")
    nino = pd.read_csv(nino_path)
    nino["Date"] = pd.to_datetime(nino["Date"])
    nino["ym"]   = nino["Date"].dt.to_period("M")
    nino = nino.sort_values("ym").reset_index(drop=True)

    col_name  = f"enso_lag{lag_months}"
    nino[col_name] = nino["nino34"].shift(lag_months)

    df = df.copy()
    if "ym" in df.columns:
        df = df.drop(columns=["ym"])
    df["ym"] = df["Date"].dt.to_period("M")
    df = df.merge(nino[["ym", col_name]], on="ym", how="left")
    df = df.drop(columns=["ym"])

    n_ok = df[col_name].notna().sum()
    print(f"    {col_name}: {n_ok} daily obs matched")
    return df, col_name


# ===================================================================
# 3. Quantile loss
# ===================================================================
def quantile_loss(actual, forecast, tau):
    u = actual - forecast
    return np.mean(u * (tau - (u < 0).astype(float)))


# ===================================================================
# 4. Rolling HAR-X-QR with arbitrary ENSO column, tau=0.95 only
# ===================================================================
def rolling_har_x_lagged_qr(df, horizon_key, horizon_label, enso_col):
    """
    Identical structure to 07b but:
    - enso_col is a parameter
    - Only estimates QUANTILES_SUBSET = [0.95] (faster)
    Returns DataFrame with columns: Date, actual, q95
    """
    target_col   = f"target_{horizon_label}"
    feature_cols = ["sigma_d", "sigma_w", "sigma_m", enso_col]

    valid = df[["Date"] + feature_cols + [target_col]].dropna().reset_index(drop=True)
    n = len(valid)

    if n < WINDOW + 100:
        print(f"  WARNING: Only {n} valid obs for h={horizon_key}")
        return None

    dates         = []
    actuals       = []
    q_forecasts   = {tau: [] for tau in QUANTILES_SUBSET}
    cached_params = {tau: None for tau in QUANTILES_SUBSET}
    last_estimated = -REESTIMATE_EVERY

    for t in tqdm(range(WINDOW, n),
                  desc=f"  lag={enso_col} h={horizon_label}",
                  leave=True):
        y_actual = valid.iloc[t][target_col]
        if np.isnan(y_actual):
            continue

        X_test       = valid.iloc[t][feature_cols].values
        X_test_const = np.concatenate([[1.0], X_test])

        if t - last_estimated >= REESTIMATE_EVERY:
            train   = valid.iloc[t - WINDOW : t]
            X_train = sm.add_constant(train[feature_cols].values)
            y_train = train[target_col].values

            for tau in QUANTILES_SUBSET:
                try:
                    result = QuantReg(y_train, X_train).fit(q=tau, max_iter=1000)
                    cached_params[tau] = result.params
                except Exception:
                    pass

            last_estimated = t

        if any(cached_params[tau] is None for tau in QUANTILES_SUBSET):
            continue

        dates.append(valid.iloc[t]["Date"])
        actuals.append(y_actual)
        for tau in QUANTILES_SUBSET:
            y_hat = X_test_const @ cached_params[tau]
            q_forecasts[tau].append(max(y_hat, 1e-10))

    results = pd.DataFrame({"Date": dates, "actual": actuals})
    for tau in QUANTILES_SUBSET:
        results[f"q{int(tau*100):02d}"] = q_forecasts[tau]

    return results


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Lag-Decay Diagnostic: HAR-X-QR with Lagged ENSO")
    print(f"Lags tested:    {LAGS} months")
    print(f"Horizons:       {list(HORIZONS_SUBSET.values())}")
    print(f"Quantile:       tau=0.95")
    print("=" * 60)

    df_base = load_data()
    rows = []

    for lag in LAGS:
        print(f"\n--- Lag {lag} month(s) ---")
        df_lag, enso_col = merge_lagged_enso(df_base, lag)

        for h, label in HORIZONS_SUBSET.items():
            results = rolling_har_x_lagged_qr(df_lag, h, label, enso_col)

            if results is not None and len(results) > 100:
                tau   = 0.95
                q_col = "q95"
                ql    = quantile_loss(results["actual"].values,
                                      results[q_col].values, tau)
                n_obs = len(results)
                rows.append({
                    "Lag":      lag,
                    "Horizon":  label,
                    "Quantile": tau,
                    "QL":       ql,
                    "N":        n_obs,
                })
                print(f"  h={label:<4}  QL(0.95) = {ql:.6f}  (N={n_obs})")
            else:
                print(f"  h={label}: insufficient data — skipped")

    if not rows:
        print("No results produced.")
        raise SystemExit(1)

    summary = pd.DataFrame(rows)
    summary["Horizon"] = summary["Horizon"].astype(HORIZON_ORDER)
    summary = summary.sort_values(["Horizon", "Lag"]).reset_index(drop=True)

    out = os.path.join(TBL_DIR, "lagged_enso_summary.csv")
    summary.to_csv(out, index=False)
    print(f"\nSaved: lagged_enso_summary.csv ({len(summary)} rows)")

    # Print pivot table — direct thesis inclusion format
    print("\n=== Lag-Decay Table (QL at tau=0.95) ===")
    print("  Lower QL = better forecast. Does power decay as lag increases?")
    pivot = summary.pivot(index="Horizon", columns="Lag", values="QL")
    pivot.columns = [f"lag_{c}m" for c in pivot.columns]
    print(pivot.round(6).to_string())

    # Interpretation hint
    print("\n=== Interpretation ===")
    for h_label in ["1m", "6m", "12m"]:
        if h_label not in pivot.index:
            continue
        row = pivot.loc[h_label]
        ql_vals = row.values
        if all(np.diff(ql_vals) > 0):
            verdict = "DECAY confirmed (QL increases with lag)"
        elif all(np.diff(ql_vals) < 0):
            verdict = "IMPROVES with lag (unusual — check data)"
        else:
            verdict = "mixed pattern"
        print(f"  {h_label}: {verdict}")

    print("\nDone!")
