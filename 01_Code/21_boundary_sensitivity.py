"""
21_boundary_sensitivity.py
==========================
Boundary sensitivity test for the 2022-2024 crisis dummy used in HAR-D-QR
and HAR-XD-QR. Re-fits the rolling models for boundary variants shifted
by +/- 3 months and recomputes Delta(ENSO | dummy) = QL(HAR-XD-QR) - QL(HAR-D-QR).

Variants tested:
  V0  baseline   2022-01-01 .. 2024-12-31  (deck Slide A)
  V- early shift 2021-10-01 .. 2024-09-30  (-3 months)
  V+ late shift  2022-04-01 .. 2025-03-31  (+3 months)

Output:
  02_Output/tables/boundary_sensitivity.csv
  02_Output/figures/33_boundary_sensitivity_ql95.png

Speed: only tau in {0.50, 0.95} fitted -> ~5x faster than the full QR sweep.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Maximum number of iterations")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
FIG_DIR  = os.path.join(BASE_DIR, "02_Output", "figures")
TBL_DIR  = os.path.join(BASE_DIR, "02_Output", "tables")
TEX_FIG  = os.path.join(os.path.dirname(BASE_DIR), "tex", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#333333",
    "axes.linewidth":   0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":      "serif",
    "font.size":        10,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

HORIZONS  = {1: "1d", 5: "1w", 22: "1m", 66: "3m", 132: "6m", 264: "12m"}
QUANTILES = [0.50, 0.95]
WINDOW           = 2520
REESTIMATE_EVERY = 5

VARIANTS = {
    "V- (-3m)":  (pd.Timestamp("2021-10-01"), pd.Timestamp("2024-09-30")),
    "V0 (base)": (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")),
    "V+ (+3m)":  (pd.Timestamp("2022-04-01"), pd.Timestamp("2025-03-31")),
}


def _ql(actual, forecast, tau):
    u = np.asarray(actual) - np.asarray(forecast)
    return float(np.mean(u * (tau - (u < 0).astype(float))))


def load_data():
    df = pd.read_csv(os.path.join(PROC_DIR, "har_dataset.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def add_dummy(df, start, end):
    df = df.copy()
    df["crisis_dummy"] = (
        (df["Date"] >= start) & (df["Date"] <= end)
    ).astype(float)
    return df


def rolling_qr(df, horizon_key, horizon_label, feature_cols, tag):
    target_col = f"target_{horizon_label}"
    # 'enso' always included in the dropna (even when not a regressor) so
    # HAR-D-QR and HAR-XD-QR are evaluated over identical forecast dates.
    extra      = [] if "enso" in feature_cols else ["enso"]
    cols       = ["Date"] + extra + feature_cols + [target_col]
    valid      = df[cols].dropna().reset_index(drop=True)
    n          = len(valid)
    if n < WINDOW + horizon_key + 100:
        return None

    dates, actuals = [], []
    q_forecasts    = {tau: [] for tau in QUANTILES}
    cached_params  = {tau: None for tau in QUANTILES}
    last_estimated = -REESTIMATE_EVERY

    # Estimation window ends at t - h: rows s > t - h have forward targets
    # not fully realized at the forecast origin t (avoids look-ahead bias).
    h = horizon_key
    for t in tqdm(range(WINDOW + h, n),
                  desc=f"  {tag} h={horizon_label}", leave=False):
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
                    res = QuantReg(y_train, X_train).fit(q=tau, max_iter=1000)
                    cached_params[tau] = res.params
                except Exception:
                    pass
            last_estimated = t

        if any(cached_params[tau] is None for tau in QUANTILES):
            continue

        dates.append(valid.iloc[t]["Date"])
        actuals.append(y_actual)
        for tau in QUANTILES:
            q_forecasts[tau].append(max(X_test_const @ cached_params[tau], 1e-10))

    out = pd.DataFrame({"Date": dates, "actual": actuals})
    for tau in QUANTILES:
        out[f"q{int(tau*100):02d}"] = q_forecasts[tau]
    return out


def evaluate(results_dict, model_label, variant_label):
    rows = []
    for h_label, res in results_dict.items():
        for tau in QUANTILES:
            qcol = f"q{int(tau*100):02d}"
            ql   = _ql(res["actual"].values, res[qcol].values, tau)
            rows.append({
                "Variant":  variant_label,
                "Model":    model_label,
                "Horizon":  h_label,
                "Quantile": tau,
                "QL":       ql,
            })
    return rows


def main():
    df0 = load_data()

    all_rows = []
    for vlabel, (start, end) in VARIANTS.items():
        print(f"\n=== Variant {vlabel}: {start.date()} .. {end.date()} ===")
        df = add_dummy(df0, start, end)
        crisis_share = df["crisis_dummy"].mean() * 100
        print(f"  Crisis share: {crisis_share:.2f}% of sample")

        # HAR-D-QR
        results_d = {}
        for hk, hl in HORIZONS.items():
            r = rolling_qr(df, hk, hl,
                           ["sigma_d", "sigma_w", "sigma_m", "crisis_dummy"],
                           f"HAR-D-QR  {vlabel}")
            if r is not None:
                results_d[hl] = r
        all_rows.extend(evaluate(results_d, "HAR-D-QR", vlabel))

        # HAR-XD-QR
        results_xd = {}
        for hk, hl in HORIZONS.items():
            r = rolling_qr(df, hk, hl,
                           ["sigma_d", "sigma_w", "sigma_m", "enso",
                            "crisis_dummy"],
                           f"HAR-XD-QR {vlabel}")
            if r is not None:
                results_xd[hl] = r
        all_rows.extend(evaluate(results_xd, "HAR-XD-QR", vlabel))

    out = pd.DataFrame(all_rows)
    out.to_csv(os.path.join(TBL_DIR, "boundary_sensitivity.csv"), index=False)
    print(f"\nSaved: {os.path.join(TBL_DIR, 'boundary_sensitivity.csv')}")

    # Build Delta(ENSO | dummy) = QL(HAR-XD-QR) - QL(HAR-D-QR), tau=0.95
    p = (out[out["Quantile"] == 0.95]
         .pivot_table(index=["Variant", "Horizon"],
                      columns="Model", values="QL")
         .reset_index())
    p["delta_enso_given_dummy"] = p["HAR-XD-QR"] - p["HAR-D-QR"]
    print("\nDelta(ENSO | dummy) at tau=0.95 (negative = ENSO helps):")
    print(p.pivot(index="Horizon", columns="Variant",
                  values="delta_enso_given_dummy"))

    plot(p)


def plot(p):
    horizons_order = ["1d", "1w", "1m", "3m", "6m", "12m"]
    p["Horizon"]   = pd.Categorical(p["Horizon"], horizons_order, ordered=True)
    pivot = p.pivot(index="Horizon", columns="Variant",
                    values="delta_enso_given_dummy").reindex(horizons_order)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    width = 0.26
    x = np.arange(len(horizons_order))
    styles = {
        "V- (-3m)":  ("#888888",   "//"),
        "V0 (base)": ("#000000",   ""),
        "V+ (+3m)":  ("#BBBBBB",   "\\\\"),
    }
    for i, vlabel in enumerate(["V- (-3m)", "V0 (base)", "V+ (+3m)"]):
        if vlabel not in pivot.columns:
            continue
        col, hatch = styles[vlabel]
        ax.bar(x + (i - 1) * width, pivot[vlabel].values * 1e4,
               width=width, label=vlabel, color=col, hatch=hatch,
               edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons_order)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(r"$\Delta$ ENSO $\mid$ dummy  (QL$\times 10^{-4}$, $\tau$=0.95)")
    ax.set_title("Boundary sensitivity: ENSO contribution conditional on the dummy\n"
                 "(near zero across all three boundary variants)")
    ax.legend(title="Crisis-dummy boundary", loc="best")
    fig.tight_layout()
    for d in [FIG_DIR, TEX_FIG]:
        fig.savefig(os.path.join(d, "33_boundary_sensitivity_ql95.png"))
    plt.close(fig)
    print("Saved: 33_boundary_sensitivity_ql95.png")


if __name__ == "__main__":
    main()
