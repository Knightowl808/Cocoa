"""
07h_rolling_har_x_vol_qr.py
============================
Rolling-window HAR-X-QR with ENSO VOLATILITY as predictor.

Motivation (Dr. Bleher, 28.04.26): the predictive content may lie in
the *switching frequency and amplitude* of ENSO (adjustment costs when
regimes change), not in the level. A rolling stdev of Nino 3.4 captures
this and is:
  (a) constructed by a reproducible rule
  (b) observable in real time (no look-ahead)
  (c) not a calendar dummy

Two specifications tested:
  vol12: 12-month rolling stdev of monthly Nino 3.4
  vol36: 36-month rolling stdev of monthly Nino 3.4

Model:
  Q_tau(sigma_{t+h} | F_t) = beta_0 + beta_d*sigma_d + beta_w*sigma_w
                              + beta_m*sigma_m + beta_vol*ENSO_vol_t

Prereq: run 04_har_features.py first to add enso_vol_12m / enso_vol_36m
        columns to har_dataset.csv.

Output per spec:
  00_Data/processed/har_x_vol_qr_{spec}_forecasts_{h}.csv
  02_Output/tables/har_x_vol_qr_{spec}_quantile_loss.csv
  02_Output/tables/har_x_vol_qr_{spec}_coverage.csv

Final comparison:
  02_Output/tables/har_x_vol_qr_comparison.csv
    — HAR-QR vs HAR-X-QR (level) vs HAR-X-vol-12m vs HAR-X-vol-36m
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
FIG_DIR  = os.path.join(BASE_DIR, "02_Output", "figures")
TBL_DIR  = os.path.join(BASE_DIR, "02_Output", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

HORIZONS         = {1: "1d", 5: "1w", 22: "1m", 66: "3m", 132: "6m", 264: "12m"}
QUANTILES        = [0.05, 0.25, 0.50, 0.75, 0.95]
WINDOW           = 2520
REESTIMATE_EVERY = 5

SPECS = {
    "vol12": "enso_vol_12m",
    "vol36": "enso_vol_36m",
}

HORIZON_ORDER = pd.CategoricalDtype(
    categories=["1d", "1w", "1m", "3m", "6m", "12m"], ordered=True
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
# 2. Quantile Loss
# ===================================================================
def quantile_loss(actual, forecast, tau):
    u = actual - forecast
    return np.mean(u * (tau - (u < 0).astype(float)))


# ===================================================================
# 3. Rolling HAR-X-vol-QR for one horizon
# ===================================================================
def rolling_har_x_vol_qr(df, horizon_key, horizon_label, enso_vol_col):
    """
    Identical structure to 07b's rolling_har_x_qr.
    Only difference: uses enso_vol_col instead of 'enso'.
    """
    target_col   = f"target_{horizon_label}"
    feature_cols = ["sigma_d", "sigma_w", "sigma_m", enso_vol_col]

    # Also require 'enso' to be non-missing so the evaluation sample is
    # identical to HAR-QR / HAR-X-QR (enso_vol columns have the same monthly
    # source but dropna alignment keeps this explicit and robust).
    valid = (df[["Date", "enso"] + feature_cols + [target_col]]
             .dropna().reset_index(drop=True))
    n = len(valid)

    if n < WINDOW + horizon_key + 100:
        print(f"  WARNING: Only {n} valid obs for h={horizon_key} with {enso_vol_col}")
        return None

    dates        = []
    actuals      = []
    q_forecasts  = {tau: [] for tau in QUANTILES}
    cached_params = {tau: None for tau in QUANTILES}
    last_estimated = -REESTIMATE_EVERY

    # Estimation window ends at t - h: rows s > t - h have forward targets
    # not fully realized at the forecast origin t (avoids look-ahead bias).
    h = horizon_key
    for t in tqdm(range(WINDOW + h, n),
                  desc=f"  HAR-X-vol-QR [{enso_vol_col}] h={horizon_label}",
                  leave=True):
        y_actual = valid.iloc[t][target_col]
        if np.isnan(y_actual):
            continue

        X_test       = valid.iloc[t][feature_cols].values
        X_test_const = np.concatenate([[1.0], X_test])

        if t - last_estimated >= REESTIMATE_EVERY:
            train   = valid.iloc[t - h - WINDOW : t - h]
            X_train = sm.add_constant(train[feature_cols].values)
            y_train = train[target_col].values

            for tau in QUANTILES:
                try:
                    result = QuantReg(y_train, X_train).fit(q=tau, max_iter=1000)
                    cached_params[tau] = result.params
                except Exception:
                    pass

            last_estimated = t

        if any(cached_params[tau] is None for tau in QUANTILES):
            continue

        dates.append(valid.iloc[t]["Date"])
        actuals.append(y_actual)
        for tau in QUANTILES:
            y_hat = X_test_const @ cached_params[tau]
            q_forecasts[tau].append(max(y_hat, 1e-10))

    results = pd.DataFrame({"Date": dates, "actual": actuals})
    for tau in QUANTILES:
        results[f"q{int(tau*100):02d}"] = q_forecasts[tau]

    return results


# ===================================================================
# 4. Aggregated loss and coverage (same as 07b)
# ===================================================================
def compute_all_losses(all_results):
    rows = []
    for label, results in all_results.items():
        for tau in QUANTILES:
            col  = f"q{int(tau*100):02d}"
            loss = quantile_loss(results["actual"].values,
                                 results[col].values, tau)
            rows.append({"Horizon": label, "Quantile": tau, "QL": loss})
    return pd.DataFrame(rows)


def coverage_analysis(all_results):
    rows = []
    for label, results in all_results.items():
        violations = (results["actual"] > results["q95"]).sum()
        n    = len(results)
        rate = violations / n * 100
        rows.append({"Horizon": label, "N": n,
                     "Violations": violations, "Rate (%)": rate})
    return pd.DataFrame(rows)


# ===================================================================
# 5. Three-way comparison table
# ===================================================================
def build_comparison_table(all_specs_losses):
    """
    Merge HAR-QR (baseline), HAR-X-QR (ENSO level), and both vol specs
    into one wide table with Delta columns.
    """
    base_path  = os.path.join(TBL_DIR, "har_qr_quantile_loss.csv")
    level_path = os.path.join(TBL_DIR, "har_x_qr_quantile_loss.csv")

    if not os.path.exists(base_path) or not os.path.exists(level_path):
        print("  WARNING: baseline loss CSVs not found — skipping comparison table.")
        print(f"    Expected: {base_path}")
        print(f"    Expected: {level_path}")
        return

    base  = pd.read_csv(base_path).rename(columns={"QL": "QL_HAR-QR"})
    level = pd.read_csv(level_path).rename(columns={"QL": "QL_HAR-X-QR"})

    merged = base.merge(level, on=["Horizon", "Quantile"])

    for spec_key, loss_df in all_specs_losses.items():
        window_label = spec_key.replace("vol", "")
        col = f"QL_HAR-X-vol-{window_label}m"
        merged = merged.merge(
            loss_df.rename(columns={"QL": col}),
            on=["Horizon", "Quantile"]
        )

    merged["Delta_ENSO-level"]  = merged["QL_HAR-X-QR"]      - merged["QL_HAR-QR"]
    if "QL_HAR-X-vol-12m" in merged.columns:
        merged["Delta_vol-12m"] = merged["QL_HAR-X-vol-12m"] - merged["QL_HAR-QR"]
    if "QL_HAR-X-vol-36m" in merged.columns:
        merged["Delta_vol-36m"] = merged["QL_HAR-X-vol-36m"] - merged["QL_HAR-QR"]

    merged["Horizon"] = merged["Horizon"].astype(HORIZON_ORDER)
    merged = merged.sort_values(["Horizon", "Quantile"]).reset_index(drop=True)

    out = os.path.join(TBL_DIR, "har_x_vol_qr_comparison.csv")
    merged.to_csv(out, index=False)
    print(f"Saved: har_x_vol_qr_comparison.csv ({len(merged)} rows)")

    # Print summary for tau=0.95
    print("\n=== Comparison at tau=0.95 ===")
    print(f"  Delta = QL(model) - QL(HAR-QR)  [negative = improvement]")
    tail = merged[merged["Quantile"] == 0.95].copy()
    delta_cols = [c for c in merged.columns if c.startswith("Delta")]
    print(tail[["Horizon"] + delta_cols].to_string(index=False))

    return merged


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("HAR-X-QR with ENSO Volatility")
    print("=" * 60)

    df = load_data()

    # Verify ENSO vol columns exist
    for col in ["enso_vol_12m", "enso_vol_36m"]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not in har_dataset.csv.\n"
                "Run 04_har_features.py first to add ENSO volatility columns."
            )
        n_ok = df[col].notna().sum()
        print(f"  {col}: {n_ok}/{len(df)} obs available")

    all_specs_losses = {}

    for spec_key, enso_col in SPECS.items():
        print(f"\n{'='*60}")
        print(f"Spec: {spec_key}  (predictor: {enso_col})")
        print(f"{'='*60}")

        all_results = {}

        for h, label in HORIZONS.items():
            print(f"\n  Horizon h={h} ({label})")
            results = rolling_har_x_vol_qr(df, h, label, enso_col)

            if results is not None and len(results) > 100:
                all_results[label] = results
                print(f"  Out-of-sample forecasts: {len(results)}")
                results.to_csv(
                    os.path.join(PROC_DIR,
                                 f"har_x_vol_qr_{spec_key}_forecasts_{label}.csv"),
                    index=False
                )
            else:
                print(f"  Skipped: insufficient data")

        if all_results:
            loss_df = compute_all_losses(all_results)
            loss_df.to_csv(
                os.path.join(TBL_DIR,
                             f"har_x_vol_qr_{spec_key}_quantile_loss.csv"),
                index=False
            )
            all_specs_losses[spec_key] = loss_df

            coverage = coverage_analysis(all_results)
            coverage.to_csv(
                os.path.join(TBL_DIR,
                             f"har_x_vol_qr_{spec_key}_coverage.csv"),
                index=False
            )

            print(f"\n  Quantile Loss (tau=0.95):")
            tail = loss_df[loss_df["Quantile"] == 0.95]
            tail["Horizon"] = tail["Horizon"].astype(HORIZON_ORDER)
            for _, row in tail.sort_values("Horizon").iterrows():
                print(f"    {row['Horizon']}: {row['QL']:.6f}")

    # Build combined comparison table
    print(f"\n{'='*60}")
    print("Building comparison table...")
    build_comparison_table(all_specs_losses)

    print("\nDone!")
