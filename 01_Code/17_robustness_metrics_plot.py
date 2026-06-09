"""
17_robustness_metrics_plot.py
==============================
Three-panel line plot: QLIKE, MSE, MAE across models and horizons.

Shows that the HAR-OLS / HAR-QR dominance is consistent regardless
of the loss function — the key robustness argument for Johannes.

Models on mean/median forecasts only (apples-to-apples comparison):
  Historical Vol, GARCH(1,1), HAR-OLS, HAR-QR (med.), HAR-X-QR (med.)
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Thesis plot style ─────────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor":  "white",
    "figure.dpi":        150,
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.linewidth":    0.8,
    "axes.grid":         False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "pdf.fonttype":      42,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
FIG_DIR  = os.path.join(BASE_DIR, "02_Output", "figures")
TEX_FIG  = os.path.join(os.path.dirname(BASE_DIR), "tex", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

HORIZONS     = ["1d", "1w", "1m", "3m", "6m", "12m"]
HORIZ_LABELS = ["1d", "1w", "1m", "3m", "6m", "12m"]
X            = np.arange(len(HORIZONS))

# Model -> (file tag, column)
MODELS = {
    "Historical Vol":  ("hist_vol", "mean"),
    "GARCH(1,1)":      ("garch",    "mean"),
    "HAR-OLS":         ("har_ols",  "forecast"),
    "HAR-QR (med.)":   ("har_qr",   "q50"),
    "HAR-X-QR (med.)": ("har_x_qr", "q50"),
}

# Monochrome-friendly styles
STYLES = {
    "Historical Vol":  dict(color="#aaaaaa", ls="--",  lw=1.4, marker="s", ms=4, zorder=2),
    "GARCH(1,1)":      dict(color="#888888", ls="-.",  lw=1.4, marker="^", ms=4, zorder=2),
    "HAR-OLS":         dict(color="#111111", ls="-",   lw=2.0, marker="o", ms=5, zorder=4),
    "HAR-QR (med.)":   dict(color="#444444", ls="-",   lw=1.6, marker="D", ms=4, zorder=3),
    "HAR-X-QR (med.)": dict(color="#111111", ls=":",   lw=1.8, marker="P", ms=5, zorder=3),
}


def load_forecast(tag, horizon):
    path = os.path.join(PROC_DIR, f"{tag}_forecasts_{horizon}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def qlike_loss(actual, forecast):
    f = np.maximum(forecast, 1e-10)
    a = np.maximum(actual,   1e-10)
    return np.mean(np.log(f**2) + (a**2) / (f**2))


def compute_all():
    results = {m: {"QLIKE": [], "MSE": [], "MAE": []} for m in MODELS}
    for horizon in HORIZONS:
        for model_name, (tag, col) in MODELS.items():
            df = load_forecast(tag, horizon)
            if df is None or col not in df.columns:
                for k in results[model_name]:
                    results[model_name][k].append(np.nan)
                continue
            sub = df[["actual", col]].dropna()
            a = sub["actual"].values
            f = sub[col].values
            results[model_name]["QLIKE"].append(qlike_loss(a, f))
            results[model_name]["MSE"].append(np.mean((a - f)**2))
            results[model_name]["MAE"].append(np.mean(np.abs(a - f)))
    return results


def plot(results):
    # (metric_key, panel_label, y_label, invert_axis, scale_factor)
    metrics = [
        ("QLIKE", "(a) QLIKE",                "QLIKE loss",           True,  1),
        ("MSE",   "(b) Mean Squared Error",   "MSE  ($\\times 10^5$)", False, 1e5),
        ("MAE",   "(c) Mean Absolute Error",  "MAE",                  False, 1),
    ]

    # Vertically stacked: each panel spans the full text width so it stays
    # legible when scaled to \textwidth in LaTeX.
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.2), sharex=True)

    for ax, (metric, panel, ylabel, invert, scale) in zip(axes, metrics):
        for model_name, vals in results.items():
            y = np.array(vals[metric]) * scale
            sty = STYLES[model_name]
            ax.plot(X, y, label=model_name, **sty)

        ax.set_ylabel(ylabel)

        # Pad y-axis so lines don't cluster at edges
        yvals = np.concatenate([np.array(v[metric]) * scale for v in results.values()])
        yvals = yvals[np.isfinite(yvals)]
        ymin, ymax = yvals.min(), yvals.max()
        pad = (ymax - ymin) * 0.12
        direction = "lower is better" if not invert else "less negative is worse"
        if invert:
            ax.set_ylim(ymax + pad, ymin - pad)
        else:
            ax.set_ylim(ymin - pad, ymax + pad)
        better = "↑ better" if invert else "↓ better"
        ax.set_title(f"{panel}   ({better})", loc="left", fontweight="bold")

    axes[-1].set_xticks(X)
    axes[-1].set_xticklabels(HORIZ_LABELS)
    axes[-1].set_xlabel("Forecast horizon")

    # Shared legend below the panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=8)

    fig.tight_layout(rect=(0, 0.03, 1, 1))

    for d in [FIG_DIR, TEX_FIG]:
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(d, f"robustness_metrics_lineplot.{ext}"),
                        bbox_inches="tight")
    plt.close(fig)
    print("Saved: robustness_metrics_lineplot.{png,pdf}")


if __name__ == "__main__":
    print("Computing metrics...")
    results = compute_all()
    print("Plotting...")
    plot(results)
    print("Done.")
