"""
16_mse_mae_comparison.py
========================
MSE and MAE comparison across point-forecast models and horizons.

Models evaluated on point forecast (mean/median):
  - Historical Vol   (column: mean)
  - GARCH(1,1)       (column: mean)
  - HAR-OLS          (column: forecast)
  - HAR-QR (med.)    (column: q50)
  - HAR-X-QR (med.)  (column: q50)

For each (model, horizon) computes:
  - MSE  = mean squared error
  - MAE  = mean absolute error
  - RMSE = sqrt(MSE)

Output:
  - tables/mse_mae_comparison.csv
  - figures/mse_mae_heatmap.png  (also copied to tex/figures/)
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Thesis plot style ─────────────────────────────────────────────────────────
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

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
TBL_DIR  = os.path.join(BASE_DIR, "02_Output", "tables")
FIG_DIR  = os.path.join(BASE_DIR, "02_Output", "figures")
TEX_FIG  = os.path.join(os.path.dirname(BASE_DIR), "tex", "figures")
os.makedirs(TBL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

HORIZONS = ["1d", "1w", "1m", "3m", "6m", "12m"]

# Model label -> (file tag, column name)
MODELS = {
    "Historical Vol":   ("hist_vol", "mean"),
    "GARCH(1,1)":       ("garch",    "mean"),
    "HAR-OLS":          ("har_ols",  "forecast"),
    "HAR-QR (med.)":    ("har_qr",   "q50"),
    "HAR-X-QR (med.)":  ("har_x_qr", "q50"),
}
MODEL_ORDER = list(MODELS.keys())


# ── Load forecast CSV ─────────────────────────────────────────────────────────
def load_forecast(tag, horizon):
    path = os.path.join(PROC_DIR, f"{tag}_forecasts_{horizon}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ── Compute MSE / MAE / RMSE ──────────────────────────────────────────────────
def compute_metrics():
    rows = []
    for horizon in HORIZONS:
        for model_name, (tag, col) in MODELS.items():
            df = load_forecast(tag, horizon)
            if df is None:
                print(f"  MISSING: {tag}_forecasts_{horizon}.csv — skipping {model_name}")
                continue
            if col not in df.columns:
                print(f"  MISSING column '{col}' in {tag}_forecasts_{horizon}.csv — skipping")
                continue

            sub = df[["Date", "actual", col]].dropna()
            actual   = sub["actual"].values
            forecast = sub[col].values

            mse  = np.mean((actual - forecast) ** 2)
            mae  = np.mean(np.abs(actual - forecast))
            rmse = np.sqrt(mse)

            rows.append({
                "Model":   model_name,
                "Horizon": horizon,
                "MSE":     mse,
                "RMSE":    rmse,
                "MAE":     mae,
                "N":       len(sub),
            })
            print(f"  {model_name:22s} | {horizon:>4s} | MSE={mse:.6f}  MAE={mae:.6f}  N={len(sub)}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(TBL_DIR, "mse_mae_comparison.csv"), index=False)
    print(f"\nSaved: mse_mae_comparison.csv")
    return df_out


# ── Heatmap helper ────────────────────────────────────────────────────────────
def _build_matrix(df, metric):
    """Build (n_models x n_horizons) matrix for a given metric."""
    mat = np.full((len(MODEL_ORDER), len(HORIZONS)), np.nan)
    for _, row in df.iterrows():
        if row["Model"] not in MODEL_ORDER:
            continue
        mi = MODEL_ORDER.index(row["Model"])
        hi = HORIZONS.index(row["Horizon"])
        mat[mi, hi] = row[metric]
    return mat


def _annotate(ax, mat, mat_norm, fmt):
    """Annotate heatmap cells; bold = best (lowest) per column."""
    col_min = np.nanmin(mat, axis=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            nval = mat_norm[i, j]
            if np.isnan(val):
                continue
            is_best = (val == col_min[j])
            txt_col = "white" if (nval < 0.15 or nval > 0.85) else "#333333"
            weight  = "bold" if is_best else "normal"
            ax.text(j, i, fmt(val), ha="center", va="center",
                    fontsize=7, color=txt_col, fontweight=weight)


# ── Main figure: two-panel heatmap (MSE and MAE) ─────────────────────────────
def plot_heatmap(df):
    # Green = low error (good), Red = high error (bad)
    cmap = LinearSegmentedColormap.from_list(
        "err_good",
        [
            (0.0,  "#1B5E20"),   # deep green (best)
            (0.30, "#66BB6A"),
            (0.50, "#C8E6C9"),
            (0.70, "#FFFFFF"),
            (0.85, "#FFCDD2"),
            (1.0,  "#C62828"),   # deep red (worst)
        ],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)

    for ax, metric, title, fmt in [
        (axes[0], "MSE", "Mean Squared Error (MSE)",  lambda v: f"{v:.2e}"),
        (axes[1], "MAE", "Mean Absolute Error (MAE)",  lambda v: f"{v:.5f}"),
    ]:
        mat = _build_matrix(df, metric)

        # Normalise column-wise: lowest (best) = 0 (green), highest = 1 (red)
        col_min = np.nanmin(mat, axis=0)
        col_max = np.nanmax(mat, axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1
        mat_norm = (mat - col_min) / col_range

        im = ax.imshow(mat_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        _annotate(ax, mat, mat_norm, fmt)

        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels(HORIZONS)
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_yticklabels(MODEL_ORDER, fontsize=8)
        ax.set_xlabel("Forecast horizon")
        ax.set_title(title, fontweight="bold")

    fig.suptitle("Point Forecast Accuracy: MSE and MAE by Model and Horizon",
                 fontsize=11, fontweight="bold")

    fig.text(0.5, -0.03,
             "Green = lower error (better).  Bold = best model per horizon.  "
             "Colours normalised column-wise (within each horizon).",
             ha="center", fontsize=7.5, style="italic", color="#555555")

    for d in [FIG_DIR, TEX_FIG]:
        fig.savefig(os.path.join(d, "mse_mae_heatmap.png"))
    plt.close(fig)
    print("Saved: mse_mae_heatmap.png")


# ── Summary printout ──────────────────────────────────────────────────────────
def print_summary(df):
    print("\n" + "=" * 60)
    print("SUMMARY — Best model per horizon")
    print("=" * 60)
    for metric in ["MSE", "MAE"]:
        print(f"\n  {metric}:")
        for h in HORIZONS:
            sub = df[df["Horizon"] == h].dropna(subset=[metric])
            if sub.empty:
                continue
            best = sub.loc[sub[metric].idxmin(), "Model"]
            best_val = sub[metric].min()
            print(f"    {h:>4s}: {best:22s}  ({metric}={best_val:.6f})")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("MSE / MAE Comparison across Models and Horizons")
    print("=" * 60)

    df = compute_metrics()
    plot_heatmap(df)
    print_summary(df)

    print("\nDone.")
