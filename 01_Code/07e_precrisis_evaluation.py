"""
07e_precrisis_evaluation.py
===========================
Pre-crisis subsample evaluation of HAR-QR vs HAR-X-QR.

NO new model estimation — loads existing forecast CSVs and recomputes
quantile losses on the pre-crisis period only (Date < 2022-01-01).

Key diagnostic: does the ENSO gain at 12m, tau=0.95 survive when the
2022-2024 cocoa price crisis is excluded from evaluation?

  Delta_full < 0 and Delta_pre ≈ 0  → gain was crisis-driven
  Delta_full ≈ Delta_pre < 0        → ENSO gain is genuine, pre-dates crisis

Filter: forecast Date < 2024-02-08
  (Bai-Perron sup-Wald estimated break date; Andrews 1993.
   The date the rolling model was estimated and forecast was issued;
   a fully real-time criterion — no crisis information was visible at t)

Output:
  precrisis_ql_comparison.csv  — long-form (model x horizon x quantile x period)
  precrisis_delta_table.csv    — wide diagnostic table, key result for thesis
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
TBL_DIR  = os.path.join(BASE_DIR, "02_Output", "tables")
os.makedirs(TBL_DIR, exist_ok=True)

HORIZONS  = ["1d", "1w", "1m", "3m", "6m", "12m"]
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
CRISIS_DATE = pd.Timestamp("2024-02-08")  # Bai-Perron sup-Wald break (Andrews 1993)

MODELS = {
    "HAR-QR":   "har_qr_forecasts_{h}.csv",
    "HAR-X-QR": "har_x_qr_forecasts_{h}.csv",
}

HORIZON_ORDER = pd.CategoricalDtype(
    categories=["1d", "1w", "1m", "3m", "6m", "12m"], ordered=True
)


# ===================================================================
# 1. Quantile Loss
# ===================================================================
def quantile_loss(actual, forecast, tau):
    u = actual - forecast
    return np.mean(u * (tau - (u < 0).astype(float)))


# ===================================================================
# 2. Load forecast CSV
# ===================================================================
def load_forecasts(model_key, horizon):
    fname = MODELS[model_key].format(h=horizon)
    path  = os.path.join(PROC_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Forecast file not found: {path}\n"
            f"Run 07_rolling_har_qr.py and 07b_rolling_har_x_qr.py first."
        )
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ===================================================================
# 3. Compute losses for full and pre-crisis subsets
# ===================================================================
def compute_precrisis_losses():
    rows = []
    for model_key in MODELS:
        for h in HORIZONS:
            try:
                df_full = load_forecasts(model_key, h)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                continue

            df_pre = df_full[df_full["Date"] < CRISIS_DATE].copy()

            n_full = len(df_full)
            n_pre  = len(df_pre)

            if n_pre < 100:
                print(f"  WARNING: {model_key} h={h} — only {n_pre} pre-crisis obs")

            for tau in QUANTILES:
                q_col = f"q{int(tau*100):02d}"
                if q_col not in df_full.columns:
                    continue

                ql_full = quantile_loss(
                    df_full["actual"].values, df_full[q_col].values, tau)
                ql_pre = quantile_loss(
                    df_pre["actual"].values, df_pre[q_col].values, tau
                ) if n_pre >= 10 else np.nan

                rows.append({
                    "Horizon":     h,
                    "Quantile":    tau,
                    "Model":       model_key,
                    "QL_full":     ql_full,
                    "QL_precrisis": ql_pre,
                    "N_full":      n_full,
                    "N_precrisis": n_pre,
                })

    return pd.DataFrame(rows)


# ===================================================================
# 4. Build wide delta table (key diagnostic output)
# ===================================================================
def build_delta_table(loss_df):
    """
    Wide table: one row per (Horizon, Quantile).
    Delta = QL(HAR-X-QR) - QL(HAR-QR): negative means ENSO helps.
    Delta_pre tells us if the gain survives outside the 2022-2024 crisis.
    """
    har = (loss_df[loss_df["Model"] == "HAR-QR"]
           .rename(columns={"QL_full": "QL_base_full",
                             "QL_precrisis": "QL_base_pre"})
           [["Horizon", "Quantile", "QL_base_full", "QL_base_pre",
             "N_full", "N_precrisis"]])

    harx = (loss_df[loss_df["Model"] == "HAR-X-QR"]
            .rename(columns={"QL_full": "QL_enso_full",
                              "QL_precrisis": "QL_enso_pre"})
            [["Horizon", "Quantile", "QL_enso_full", "QL_enso_pre"]])

    merged = har.merge(harx, on=["Horizon", "Quantile"])
    merged["Delta_full"] = merged["QL_enso_full"] - merged["QL_base_full"]
    merged["Delta_pre"]  = merged["QL_enso_pre"]  - merged["QL_base_pre"]

    merged["Horizon"] = merged["Horizon"].astype(HORIZON_ORDER)
    merged = merged.sort_values(["Horizon", "Quantile"]).reset_index(drop=True)

    return merged


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Pre-Crisis Subsample Evaluation")
    print(f"Crisis period excluded: Date >= {CRISIS_DATE.date()}")
    print("=" * 60)

    print("\nComputing quantile losses (full sample and pre-crisis)...")
    loss_df = compute_precrisis_losses()

    if loss_df.empty:
        print("No results — check that 07 and 07b forecast CSVs exist.")
        raise SystemExit(1)

    # Save long-form table
    loss_df["Horizon"] = loss_df["Horizon"].astype(HORIZON_ORDER)
    loss_df = loss_df.sort_values(["Horizon", "Quantile", "Model"]).reset_index(drop=True)
    out_long = os.path.join(TBL_DIR, "precrisis_ql_comparison.csv")
    loss_df.to_csv(out_long, index=False)
    print(f"\nSaved: precrisis_ql_comparison.csv ({len(loss_df)} rows)")

    # Build and save delta table
    delta = build_delta_table(loss_df)
    out_delta = os.path.join(TBL_DIR, "precrisis_delta_table.csv")
    delta.to_csv(out_delta, index=False)
    print(f"Saved: precrisis_delta_table.csv ({len(delta)} rows)")

    # Print key diagnostic rows (12m and 6m, tau=0.95)
    print("\n=== Key diagnostic: ENSO gain at tau=0.95 ===")
    print(f"  Delta = QL(HAR-X-QR) - QL(HAR-QR)  [negative = ENSO helps]")
    print(f"  {'Horizon':<8} {'N_pre':>7}  {'Delta_full':>12}  {'Delta_pre':>12}  {'Survives?'}")
    print(f"  {'-'*60}")
    for _, row in delta[delta["Quantile"] == 0.95].iterrows():
        survives = "YES" if (
            pd.notna(row["Delta_pre"]) and
            row["Delta_pre"] < -1e-8 and
            abs(row["Delta_pre"]) > 0.1 * abs(row["Delta_full"])
        ) else "no"
        print(f"  {str(row['Horizon']):<8} {int(row['N_precrisis']):>7}  "
              f"{row['Delta_full']:>12.6f}  "
              f"{row['Delta_pre'] if pd.notna(row['Delta_pre']) else float('nan'):>12.6f}  "
              f"{survives}")

    print("\nDone!")
