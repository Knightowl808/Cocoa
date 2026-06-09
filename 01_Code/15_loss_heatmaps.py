"""
15_loss_heatmaps.py
===================
Heatmaps for Quantile Loss and QLIKE across models and horizons.

Panel A: Quantile Loss heatmap (models x horizons), one sub-panel per tau.
         Green = lower QL = better.  Annotates absolute values.
Panel B: QLIKE heatmap (models x horizons), mean-forecast models only.
         Green = more negative QLIKE = better.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap

# ── Thesis plot style ────────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor":     "white",
    "figure.dpi":           150,
    "axes.facecolor":       "white",
    "axes.edgecolor":       "#333333",
    "axes.linewidth":       0.8,
    "axes.grid":            False,
    "axes.spines.top":      True,
    "axes.spines.right":    True,
    "font.family":          "serif",
    "font.size":            9,
    "axes.titlesize":       10,
    "axes.labelsize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "white",
    "pdf.fonttype":         42,
})

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TBL_DIR  = os.path.join(BASE_DIR, "02_Output", "tables")
FIG_DIR  = os.path.join(BASE_DIR, "02_Output", "figures")
TEX_FIG  = os.path.join(os.path.dirname(BASE_DIR), "tex", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

HORIZONS  = ["1d", "1w", "1m", "3m", "6m", "12m"]
QUANTILES = [0.05, 0.50, 0.95]   # lower tail, median, upper tail

# Models that produce quantile forecasts (full set incl. RQ2 regime-test specs)
QL_MODELS = [
    "Historical Vol",
    "GARCH(1,1)",
    "HAR-OLS",
    "HAR-QR",
    "HAR-X-QR",
    "HAR-D-QR",
    "HAR-XD-QR",
]
# Map model display name to forecast-file prefix in 00_Data/processed/
FORECAST_PREFIX = {
    "Historical Vol": "hist_vol",
    "GARCH(1,1)":     "garch",
    "HAR-OLS":        "har_ols",
    "HAR-QR":         "har_qr",
    "HAR-X-QR":       "har_x_qr",
    "HAR-D-QR":       "har_d_qr",
    "HAR-XD-QR":      "har_xd_qr",
}
# Models that produce mean/median forecasts (QLIKE-comparable)
QLIKE_MODELS = ["Historical Vol", "GARCH(1,1)", "HAR-OLS", "HAR-QR (med.)", "HAR-X-QR (med.)"]


# ==========================================================================
# Load Quantile Loss data
# ==========================================================================
def _ql(actual, forecast, tau):
    """Pinball / check-loss."""
    u = np.asarray(actual) - np.asarray(forecast)
    return float(np.mean(u * (tau - (u < 0).astype(float))))


def _ql_from_forecasts(prefix, tau):
    """Recompute QL at a given tau from the per-horizon forecast CSVs."""
    PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
    out = {}
    qcol = f"q{int(round(tau*100)):02d}"
    for h in HORIZONS:
        path = os.path.join(PROC_DIR, f"{prefix}_forecasts_{h}.csv")
        if not os.path.exists(path):
            out[h] = np.nan
            continue
        df = pd.read_csv(path)
        if qcol not in df.columns or "actual" not in df.columns:
            out[h] = np.nan
            continue
        out[h] = _ql(df["actual"].values, df[qcol].values, tau)
    return out


def load_quantile_losses():
    """Recompute QL at QUANTILES for every model from forecast CSVs.

    Source of truth = per-horizon forecast files in 00_Data/processed/.
    This guarantees consistency across models even when CSV summary tables
    contain different quantile sets.
    """
    rows = []
    for model_name, prefix in FORECAST_PREFIX.items():
        for tau in QUANTILES:
            losses = _ql_from_forecasts(prefix, tau)
            for h, v in losses.items():
                rows.append({
                    "Model": model_name,
                    "Horizon": h,
                    "Quantile": tau,
                    "QL": v,
                })
    return pd.DataFrame(rows)


# ==========================================================================
# Load QLIKE data
# ==========================================================================
def load_qlike():
    """Load QLIKE for all mean-forecast models."""
    # qlike_model_comparison.csv has HAR-OLS, HAR-QR (median), Historical Vol, GARCH
    # in wide format: rows=models, columns=horizons
    comp = pd.read_csv(os.path.join(TBL_DIR, "qlike_model_comparison.csv"), index_col=0)

    # Also load MCS QLIKE which has all 5 models with proper naming
    mcs = pd.read_csv(os.path.join(TBL_DIR, "mcs_qlike.csv"))

    rows = []
    for _, r in mcs.iterrows():
        rows.append({
            "Model": r["Model"],
            "Horizon": r["Horizon"],
            "QLIKE": r["Mean QLIKE"],
        })
    return pd.DataFrame(rows)


# ==========================================================================
# FIGURE 1: Quantile Loss Heatmap (one panel per quantile)
# ==========================================================================
def plot_ql_heatmap():
    df = load_quantile_losses()

    # Green = low QL (good), Red = high QL (bad)
    cmap_ql = LinearSegmentedColormap.from_list(
        "ql_good",
        [
            (0.0, "#1B5E20"),   # deep green (best)
            (0.3, "#66BB6A"),   # medium green
            (0.5, "#C8E6C9"),   # pale green
            (0.7, "#FFFFFF"),   # white (mid)
            (0.85, "#FFCDD2"),  # pale red
            (1.0, "#C62828"),   # deep red (worst)
        ],
    )

    # --- Build all matrices first to compute global min/max ---
    all_mats = {}
    for tau in QUANTILES:
        sub = df[df["Quantile"] == tau]
        mat = np.full((len(QL_MODELS), len(HORIZONS)), np.nan)
        for _, row in sub.iterrows():
            model_name = row["Model"]
            if model_name not in QL_MODELS:
                continue
            mi = QL_MODELS.index(model_name)
            hi = HORIZONS.index(row["Horizon"])
            mat[mi, hi] = row["QL"]
        all_mats[tau] = mat

    global_min = min(np.nanmin(m) for m in all_mats.values())
    global_max = max(np.nanmax(m) for m in all_mats.values())
    global_range = global_max - global_min

    fig, axes = plt.subplots(1, len(QUANTILES),
                             figsize=(3.4 * len(QUANTILES) + 1.6, 4.4),
                             constrained_layout=True)
    if len(QUANTILES) == 1:
        axes = [axes]

    for qi, tau in enumerate(QUANTILES):
        ax = axes[qi]
        mat = all_mats[tau]

        # Global normalisation: same scale across all panels and horizons
        mat_norm = (mat - global_min) / global_range

        # Best (lowest QL) per column for bold annotation
        col_min = np.nanmin(mat, axis=0)

        im = ax.imshow(mat_norm, cmap=cmap_ql, vmin=0, vmax=1, aspect="auto")

        # Annotate with actual QL values
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                nval = mat_norm[i, j]
                if np.isnan(val):
                    continue
                is_best = (val == col_min[j])
                txt_col = "white" if (nval < 0.15 or nval > 0.85) else "#333333"
                weight = "bold" if is_best else "normal"
                ax.text(j, i, f"{val:.5f}", ha="center", va="center",
                        fontsize=6.5, color=txt_col, fontweight=weight)

        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels(HORIZONS)
        ax.set_yticks(range(len(QL_MODELS)))
        if qi == 0:
            ax.set_yticklabels(QL_MODELS, fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Horizon")
        ax.set_title(rf"$\tau = {tau:.2f}$", fontweight="bold")

    fig.suptitle("Quantile Loss by Model, Horizon, and Quantile Level",
                 fontsize=11, fontweight="bold")

    # Legend note
    fig.text(0.5, -0.02,
             "Green = lower quantile loss (better).  Bold = best model per horizon.  "
             "Colours normalised globally across all panels.",
             ha="center", fontsize=7.5, style="italic", color="#555555")

    for d in [FIG_DIR, TEX_FIG]:
        fig.savefig(os.path.join(d, "ql_heatmap_all.png"))
    plt.close(fig)
    print("Saved: ql_heatmap_all.png")


# ==========================================================================
# FIGURE 2: QLIKE Heatmap (models x horizons)
# ==========================================================================
def plot_qlike_heatmap():
    df = load_qlike()

    # Green = more negative QLIKE (better), Red = less negative (worse)
    cmap_qlike = LinearSegmentedColormap.from_list(
        "qlike_good",
        [
            (0.0, "#C62828"),   # deep red (worst / least negative)
            (0.15, "#FFCDD2"),  # pale red
            (0.3, "#FFFFFF"),   # white
            (0.5, "#C8E6C9"),   # pale green
            (0.7, "#66BB6A"),   # medium green
            (1.0, "#1B5E20"),   # deep green (best / most negative)
        ],
    )

    models = QLIKE_MODELS
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)

    mat = np.full((len(models), len(HORIZONS)), np.nan)
    for _, row in df.iterrows():
        model_name = row["Model"]
        if model_name not in models:
            continue
        mi = models.index(model_name)
        hi = HORIZONS.index(row["Horizon"])
        mat[mi, hi] = row["QLIKE"]

    # Normalise per column: most negative = 1 (green), least negative = 0 (red)
    col_min = np.nanmin(mat, axis=0)  # most negative = best
    col_max = np.nanmax(mat, axis=0)  # least negative = worst
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    # Invert: best (most negative) should map to 1.0
    mat_norm = (mat - col_max) / (-col_range)  # best=1, worst=0

    im = ax.imshow(mat_norm, cmap=cmap_qlike, vmin=0, vmax=1, aspect="auto")

    # Annotate
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            nval = mat_norm[i, j]
            if np.isnan(val):
                continue
            is_best = (val == col_min[j])
            txt_col = "white" if (nval < 0.15 or nval > 0.85) else "#333333"
            weight = "bold" if is_best else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=txt_col, fontweight=weight)

    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels(HORIZONS)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel("Forecast horizon")
    ax.set_title("QLIKE Loss by Model and Horizon", fontweight="bold", fontsize=11)

    fig.text(0.5, -0.04,
             "Green = lower QLIKE (better).  Bold = best model per horizon.  "
             "QLIKE is robust to volatility proxy noise (Patton, 2011).",
             ha="center", fontsize=7.5, style="italic", color="#555555")

    for d in [FIG_DIR, TEX_FIG]:
        fig.savefig(os.path.join(d, "qlike_heatmap.png"))
    plt.close(fig)
    print("Saved: qlike_heatmap.png")


# ==========================================================================
if __name__ == "__main__":
    plot_ql_heatmap()
    plot_qlike_heatmap()
    print("Done.")
