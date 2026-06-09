"""
07e2_precrisis_vol_evaluation.py
=================================
Pre-crisis subsample evaluation extended to ENSO volatility specifications.

Extends 07e_precrisis_evaluation.py to also cover:
  - HAR-X-vol-12m  (12-month rolling stdev of Nino 3.4)
  - HAR-X-vol-36m  (36-month rolling stdev of Nino 3.4)

Same logic as 07e: NO new estimation. Loads existing forecast CSVs,
filters to Date < 2022-01-01, recomputes quantile losses.

Key diagnostic: does the gain at 12m, tau=0.95 survive pre-crisis for
the vol specs, the same way it vanishes for the ENSO level?

Output:
  precrisis_vol_delta_table.csv  — wide table (all 4 models x horizon x quantile)
"""

import pandas as pd
import numpy as np
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(BASE_DIR, "00_Data", "processed")
TBL_DIR   = os.path.join(BASE_DIR, "02_Output", "tables")
os.makedirs(TBL_DIR, exist_ok=True)

HORIZONS    = ["1d", "1w", "1m", "3m", "6m", "12m"]
QUANTILES   = [0.05, 0.25, 0.50, 0.75, 0.95]
CRISIS_DATE = pd.Timestamp("2024-02-08")  # onset of extreme-volatility phase, supply crisis

HORIZON_ORDER = pd.CategoricalDtype(
    categories=["1d", "1w", "1m", "3m", "6m", "12m"], ordered=True
)

MODELS = {
    "HAR-QR":        "har_qr_forecasts_{h}.csv",
    "HAR-X-QR":      "har_x_qr_forecasts_{h}.csv",
    "HAR-X-vol-12m": "har_x_vol_qr_vol12_forecasts_{h}.csv",
    "HAR-X-vol-36m": "har_x_vol_qr_vol36_forecasts_{h}.csv",
}


def quantile_loss(actual, forecast, tau):
    u = actual - forecast
    return float(np.mean(u * (tau - (u < 0).astype(float))))


def compute_losses():
    rows = []
    for model_key, fname_tpl in MODELS.items():
        for h in HORIZONS:
            path = os.path.join(PROC_DIR, fname_tpl.format(h=h))
            if not os.path.exists(path):
                print(f"  SKIP {model_key} h={h}: {path} not found")
                continue

            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"])
            df_pre = df[df["Date"] < CRISIS_DATE].copy()

            n_full = len(df)
            n_pre  = len(df_pre)

            for tau in QUANTILES:
                col = f"q{int(tau*100):02d}"
                if col not in df.columns:
                    continue
                ql_full = quantile_loss(df["actual"].values, df[col].values, tau)
                ql_pre  = quantile_loss(
                    df_pre["actual"].values, df_pre[col].values, tau
                ) if n_pre >= 10 else np.nan

                rows.append({
                    "Model":       model_key,
                    "Horizon":     h,
                    "Quantile":    tau,
                    "QL_full":     ql_full,
                    "QL_precrisis": ql_pre,
                    "N_full":      n_full,
                    "N_precrisis": n_pre,
                })
    return pd.DataFrame(rows)


def build_delta_table(loss_df):
    """
    Wide table: one row per (Horizon, Quantile).
    Columns: baseline QL, then Delta_full and Delta_pre for each ENSO spec.
    """
    base = (loss_df[loss_df["Model"] == "HAR-QR"]
            .rename(columns={"QL_full": "QL_base_full", "QL_precrisis": "QL_base_pre"})
            [["Horizon", "Quantile", "QL_base_full", "QL_base_pre", "N_full", "N_precrisis"]])

    enso_models = ["HAR-X-QR", "HAR-X-vol-12m", "HAR-X-vol-36m"]
    merged = base.copy()

    for m in enso_models:
        sub = (loss_df[loss_df["Model"] == m]
               .rename(columns={"QL_full": f"QL_{m}_full", "QL_precrisis": f"QL_{m}_pre"})
               [["Horizon", "Quantile", f"QL_{m}_full", f"QL_{m}_pre"]])
        merged = merged.merge(sub, on=["Horizon", "Quantile"], how="left")
        merged[f"Delta_full_{m}"] = merged[f"QL_{m}_full"] - merged["QL_base_full"]
        merged[f"Delta_pre_{m}"]  = merged[f"QL_{m}_pre"]  - merged["QL_base_pre"]

    merged["Horizon"] = merged["Horizon"].astype(HORIZON_ORDER)
    return merged.sort_values(["Horizon", "Quantile"]).reset_index(drop=True)


if __name__ == "__main__":
    print("=" * 65)
    print("Pre-Crisis Evaluation — ENSO Vol Specs (07e2)")
    print(f"Crisis period excluded: Date >= {CRISIS_DATE.date()}")
    print("=" * 65)

    loss_df = compute_losses()
    if loss_df.empty:
        raise SystemExit("No results — check forecast CSVs exist.")

    loss_df["Horizon"] = loss_df["Horizon"].astype(HORIZON_ORDER)
    loss_df = loss_df.sort_values(["Horizon", "Quantile", "Model"]).reset_index(drop=True)

    delta = build_delta_table(loss_df)
    out   = os.path.join(TBL_DIR, "precrisis_vol_delta_table.csv")
    delta.to_csv(out, index=False)
    print(f"\nSaved: precrisis_vol_delta_table.csv ({len(delta)} rows)")

    # ===================================================================
    # Key diagnostic: tau=0.95, all horizons, all three ENSO specs
    # ===================================================================
    enso_cols = ["HAR-X-QR", "HAR-X-vol-12m", "HAR-X-vol-36m"]
    key = delta[delta["Quantile"] == 0.95].copy()

    print("\n" + "=" * 65)
    print("KEY DIAGNOSTIC: Delta at tau=0.95")
    print("Delta = QL(model) - QL(HAR-QR)  [negative = model helps]")
    print("=" * 65)

    n_pre_ex = key["N_precrisis"].iloc[0] if not key.empty else "?"
    n_full_ex = key["N_full"].iloc[0] if not key.empty else "?"
    print(f"\n  N_full ~= {int(n_full_ex):,}   N_precrisis ~= {int(n_pre_ex):,}\n")

    col_w = 14
    header = f"  {'Horizon':<8}" + "".join(
        f"  {m.replace('HAR-X-',''):>{col_w}}" for m in enso_cols
    )
    sep = "  " + "-" * (len(header) - 2)

    period_label = f"PRE-CRISIS (< {CRISIS_DATE.strftime('%b %Y')})"
    for period, suffix in [("FULL SAMPLE", "_full"), (period_label, "_pre")]:
        print(f"  {period}")
        print(header)
        print(sep)
        for _, row in key.iterrows():
            vals = [row.get(f"Delta{suffix}_{m}", np.nan) for m in enso_cols]
            line = f"  {str(row['Horizon']):<8}" + "".join(
                f"  {v:>+{col_w}.6f}" if pd.notna(v) else f"  {'n/a':>{col_w}}" for v in vals
            )
            print(line)
        print()

    # ===================================================================
    # Survival check: does pre-crisis gain >= 10% of full-sample gain?
    # ===================================================================
    print("  SURVIVAL CHECK  (|Delta_pre| > 10% of |Delta_full| AND Delta_pre < 0)")
    print(sep)
    print(f"  {'Horizon':<8}" + "".join(f"  {m.replace('HAR-X-',''):>{col_w}}" for m in enso_cols))
    print(sep)
    for _, row in key.iterrows():
        results = []
        for m in enso_cols:
            df_full = row.get(f"Delta_full_{m}", np.nan)
            df_pre  = row.get(f"Delta_pre_{m}",  np.nan)
            if pd.isna(df_full) or pd.isna(df_pre):
                results.append("n/a")
            elif df_pre < 0 and abs(df_pre) > 0.1 * abs(df_full):
                results.append("SURVIVES")
            else:
                results.append("vanishes")
        line = f"  {str(row['Horizon']):<8}" + "".join(f"  {r:>{col_w}}" for r in results)
        print(line)

    print("\nDone.")
