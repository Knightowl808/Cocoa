"""
07d_rolling_har_xd_qr.py
========================
Rolling-window HAR-XD-QR: HAR-QR augmented with ENSO *and* crisis dummy.

Model (HAR-XD-QR):
  Q_tau(sigma_{t+h} | F_t) = beta_0(tau) + beta_d(tau)*sigma_d
                              + beta_w(tau)*sigma_w + beta_m(tau)*sigma_m
                              + beta_enso(tau)*ENSO_t
                              + beta_D(tau)*D_t

Purpose — ENSO regime test (from 17.03.2026 meeting with Dr. Bleher):
  This is the "full" specification that includes both the ENSO anomaly and
  the 2022–2024 crisis regime dummy simultaneously. Comparing its quantile
  loss against HAR-D-QR (dummy only, script 07c) isolates the *incremental*
  contribution of ENSO over and above the crisis regime effect.

  Interpretation guide:
    QL(HAR-XD-QR) << QL(HAR-D-QR)  → ENSO adds beyond the regime dummy
                                       → ENSO informativeness is genuine
    QL(HAR-XD-QR) ≈  QL(HAR-D-QR)  → dummy absorbs ENSO's signal
                                       → ENSO benefit may be spurious

Crisis dummy:
  D_t = 1  for 2022-01-01 <= Date <= 2024-12-31 (cocoa supply crisis)
  D_t = 0  otherwise

ENSO variable: monthly Nino 3.4 anomaly (climatological baseline 1991–2020),
  sourced from NOAA and pre-processed in 00_data_loading.py.

Quantiles: tau in {0.05, 0.25, 0.50, 0.75, 0.95}
Horizons:  h in {1, 5, 22, 66, 132, 264}
Rolling window: 2,520 days (~10 years), re-estimate every 5 days.

Output:
  00_Data/processed/har_xd_qr_forecasts_{h}.csv  — per-horizon forecasts
  02_Output/tables/har_xd_qr_quantile_loss.csv   — QL across all (h, tau)
  02_Output/tables/har_xd_qr_coverage.csv        — tau=0.95 coverage
  02_Output/figures/28_har_xd_qr_fanchart_{h}.png
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
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

# ---------------------------------------------------------------------------
# Thesis plot style (consistent with 09_comparison_plots.py)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "figure.facecolor":    "white",
    "figure.dpi":          150,
    "axes.facecolor":      "white",
    "axes.edgecolor":      "#333333",
    "axes.linewidth":      0.8,
    "axes.grid":           False,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "font.family":         "serif",
    "font.size":           9,
    "axes.titlesize":      9,
    "axes.labelsize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     8,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.size":    3,
    "ytick.major.size":    3,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "lines.linewidth":     1.0,
    "legend.frameon":      False,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.facecolor":   "white",
    "pdf.fonttype":        42,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZONS = {1: "1d", 5: "1w", 22: "1m", 66: "3m", 132: "6m", 264: "12m"}
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
WINDOW = 2520
REESTIMATE_EVERY = 5

# Crisis period: 2022–2024 cocoa supply crisis
CRISIS_START = pd.Timestamp("2022-01-01")
CRISIS_END   = pd.Timestamp("2024-12-31")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load the HAR dataset (includes 'enso' column) and add crisis dummy."""
    path = os.path.join(PROC_DIR, "har_dataset.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = add_crisis_dummy(df)
    return df


def add_crisis_dummy(df):
    """
    Add binary crisis dummy D_t.

    D_t = 1  for 2022-01-01 <= Date <= 2024-12-31  (cocoa supply crisis)
    D_t = 0  otherwise

    Predetermined dates — no look-ahead bias. The crisis_dummy column has
    no NaNs; sample size is bound only by ENSO availability (same as 07b).
    """
    df = df.copy()
    df["crisis_dummy"] = (
        (df["Date"] >= CRISIS_START) & (df["Date"] <= CRISIS_END)
    ).astype(float)
    return df


# ---------------------------------------------------------------------------
# Quantile loss
# ---------------------------------------------------------------------------
def quantile_loss(actual, forecast, tau):
    """Check function / pinball loss (Koenker & Bassett 1978)."""
    u = actual - forecast
    return np.mean(u * (tau - (u < 0).astype(float)))


# ---------------------------------------------------------------------------
# Rolling HAR-XD-QR
# ---------------------------------------------------------------------------
def rolling_har_xd_qr(df, horizon_key, horizon_label):
    """
    Rolling-window HAR-XD-QR for a single horizon, all quantiles.

    Features: [sigma_d, sigma_w, sigma_m, enso, crisis_dummy]

    ENSO NaNs reduce the effective sample (same as in 07b). The crisis_dummy
    is fully observed and does not reduce the sample further.

    Returns DataFrame with columns: Date, actual, q05, q25, q50, q75, q95
    """
    target_col   = f"target_{horizon_label}"
    feature_cols = ["sigma_d", "sigma_w", "sigma_m", "enso", "crisis_dummy"]

    valid = df[["Date"] + feature_cols + [target_col]].dropna().reset_index(drop=True)
    n = len(valid)

    if n < WINDOW + horizon_key + 100:
        print(f"  WARNING: Only {n} valid obs for h={horizon_key}")
        return None

    dates       = []
    actuals     = []
    q_forecasts = {tau: [] for tau in QUANTILES}

    cached_params  = {tau: None for tau in QUANTILES}
    last_estimated = -REESTIMATE_EVERY

    # Estimation window ends at t - h: rows s > t - h have forward targets
    # not fully realized at the forecast origin t (avoids look-ahead bias).
    h = horizon_key
    for t in tqdm(range(WINDOW + h, n), desc=f"  HAR-XD-QR h={horizon_label}", leave=True):
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
                    model  = QuantReg(y_train, X_train)
                    result = model.fit(q=tau, max_iter=1000)
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


# ---------------------------------------------------------------------------
# Fan chart
# ---------------------------------------------------------------------------
def plot_fan_chart(results, horizon_label):
    """Fan chart: quantile bands + actual volatility (thesis style)."""
    fig, ax = plt.subplots(figsize=(6.5, 2.8))

    ax.fill_between(results["Date"], results["q05"], results["q95"],
                    alpha=0.12, color="#333333", label="5–95% band")
    ax.fill_between(results["Date"], results["q25"], results["q75"],
                    alpha=0.22, color="#333333", label="25–75% band")
    ax.plot(results["Date"], results["q50"],
            linewidth=1.0, color="#000000", linestyle="--",
            label="Median forecast (τ=0.50)")
    ax.plot(results["Date"], results["actual"],
            linewidth=0.5, alpha=0.55, color="#777777", label="Actual")

    ax.set_title(f"HAR-XD-QR Fan Chart (ENSO + crisis dummy) — Horizon {horizon_label}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg. daily volatility")
    ax.legend(loc="upper left")

    fname = os.path.join(FIG_DIR, f"28_har_xd_qr_fanchart_{horizon_label}.png")
    fig.savefig(fname)
    plt.close(fig)
    print(f"  Saved: 28_har_xd_qr_fanchart_{horizon_label}.png")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def compute_all_losses(all_results):
    """Quantile loss for every (horizon, quantile) pair."""
    rows = []
    for label, res in all_results.items():
        for tau in QUANTILES:
            col  = f"q{int(tau*100):02d}"
            loss = quantile_loss(res["actual"].values, res[col].values, tau)
            rows.append({"Horizon": label, "Quantile": tau, "QL": loss})
    return pd.DataFrame(rows)


def coverage_analysis(all_results):
    """Empirical violation rate at tau=0.95 (expected: 5%)."""
    print("\n=== Coverage Analysis (tau=0.95) ===")
    rows = []
    for label, res in all_results.items():
        violations = (res["actual"] > res["q95"]).sum()
        n    = len(res)
        rate = violations / n * 100
        rows.append({"Horizon": label, "N": n,
                     "Violations": violations, "Rate (%)": rate})
        print(f"  {label}: {violations}/{n} violations ({rate:.1f}%)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("HAR-XD-QR — HAR-QR with ENSO and Crisis Regime Dummy")
    print("=" * 60)
    print("\nModel specification:")
    print("  Q_tau(sigma_{t+h}) = b0(tau) + b_d(tau)*sigma_d + b_w(tau)*sigma_w")
    print("                       + b_m(tau)*sigma_m + b_enso(tau)*ENSO_t + b_D(tau)*D_t")
    print(f"\nCrisis period: {CRISIS_START.date()} to {CRISIS_END.date()}")
    print("Purpose: full specification for ENSO regime test (ENSO + dummy)")
    print("\n" + "=" * 60 + "\n")

    df = load_data()

    enso_available = df["enso"].notna().sum()
    n_crisis       = df["crisis_dummy"].sum()
    print(f"ENSO data available: {enso_available}/{len(df)} days")
    print(f"Crisis observations: {int(n_crisis)} / {len(df)} days "
          f"({100*n_crisis/len(df):.1f}%)\n")

    all_results = {}

    for h, label in HORIZONS.items():
        print(f"\n{'='*50}")
        print(f"Horizon: h={h} ({label})")
        print(f"{'='*50}")

        res = rolling_har_xd_qr(df, h, label)

        if res is not None and len(res) > 100:
            all_results[label] = res
            print(f"  Out-of-sample forecasts: {len(res)}")

            res.to_csv(
                os.path.join(PROC_DIR, f"har_xd_qr_forecasts_{label}.csv"),
                index=False
            )

            if label in ["1d", "1m", "6m", "12m"]:
                plot_fan_chart(res, label)
        else:
            print(f"  Skipped: insufficient data")

    if all_results:
        print(f"\n{'='*50}")
        print("Evaluation")
        print(f"{'='*50}")

        loss_df = compute_all_losses(all_results)
        loss_df.to_csv(
            os.path.join(TBL_DIR, "har_xd_qr_quantile_loss.csv"), index=False
        )

        print("\n=== Quantile Loss (tau=0.95) ===")
        ql95 = loss_df[loss_df["Quantile"] == 0.95].set_index("Horizon")["QL"]
        for h in ["1d", "1w", "1m", "3m", "6m", "12m"]:
            if h in ql95.index:
                print(f"  {h}: {ql95.get(h, float('nan')):.6f}")

        coverage = coverage_analysis(all_results)
        coverage.to_csv(
            os.path.join(TBL_DIR, "har_xd_qr_coverage.csv"), index=False
        )

    print("\nDone!")
