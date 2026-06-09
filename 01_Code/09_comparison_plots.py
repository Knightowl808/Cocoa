"""
09_comparison_plots.py
======================
Model comparison line plots for Results section.
  (a) QLIKE across horizons — 3 mean-forecast models
  (b) Quantile loss (tau=0.95) across horizons — 4 quantile-forecast models
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Thesis plot style (from .claude/skills/plot-style) ──────────────────────
mpl.rcParams.update({
    "figure.facecolor":     "white",
    "figure.dpi":           150,
    "axes.facecolor":       "white",
    "axes.edgecolor":       "#333333",
    "axes.linewidth":       0.8,
    "axes.grid":            False,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "font.family":          "serif",
    "font.size":            9,
    "axes.titlesize":       9,
    "axes.labelsize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.minor.visible":  False,
    "ytick.minor.visible":  False,
    "lines.linewidth":      1.0,
    "legend.frameon":       False,
    "legend.borderpad":     0.3,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "white",
    "pdf.fonttype":         42,
})

C = {
    "black":       "#000000",
    "dark_gray":   "#333333",
    "mid_gray":    "#777777",
    "light_gray":  "#AAAAAA",
    "accent":      "#2B6CB0",
}

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR  = os.path.join(BASE_DIR, "02_Output", "figures")
TBL_DIR  = os.path.join(BASE_DIR, "02_Output", "tables")
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
TEX_FIG  = os.path.join(os.path.dirname(BASE_DIR), "tex", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

horizons = ["1d", "1w", "1m", "3m", "6m", "12m"]
x        = np.arange(len(horizons))


def _ql(actual, forecast, tau):
    u = np.asarray(actual) - np.asarray(forecast)
    return float(np.mean(u * (tau - (u < 0).astype(float))))


def _ql95_from_forecasts(prefix):
    out = []
    for h in horizons:
        path = os.path.join(PROC_DIR, f"{prefix}_forecasts_{h}.csv")
        df = pd.read_csv(path)
        out.append(_ql(df["actual"].values, df["q95"].values, 0.95))
    return out


# QLIKE (mean forecasts, more negative = better) - load from canonical CSVs
def _load_qlike():
    bench = pd.read_csv(os.path.join(TBL_DIR, "benchmark_mean_qlike.csv"))
    bench = bench.set_index(["Model", "Horizon"])["QLIKE"]
    har_ols = pd.read_csv(
        os.path.join(TBL_DIR, "har_ols_qlike_loss.csv"), index_col=0
    )["QLIKE"]
    return {
        "Historical Vol": [bench.loc[("Historical Vol", h)] for h in horizons],
        "GARCH(1,1)":     [bench.loc[("GARCH(1,1)",     h)] for h in horizons],
        "HAR-OLS":        [har_ols.loc[h]                    for h in horizons],
    }


qlike = _load_qlike()

# Quantile loss at tau=0.95 (lower = better) - compute from per-model forecast CSVs
ql95 = {
    "Historical Vol": _ql95_from_forecasts("hist_vol"),
    "GARCH(1,1)":     _ql95_from_forecasts("garch"),
    "HAR-OLS":        _ql95_from_forecasts("har_ols"),
    "HAR-QR":         _ql95_from_forecasts("har_qr"),
    "HAR-X-QR":       _ql95_from_forecasts("har_x_qr"),
}

# ── Figure 1: QLIKE comparison ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 2.8))

styles = [
    ("Historical Vol", C["light_gray"], "-.",  "s",  5.0),
    ("GARCH(1,1)",     C["mid_gray"],   ":",   "D",  4.5),
    ("HAR-OLS",        C["black"],      "-",   "o",  5.0),
]

for name, color, ls, marker, ms in styles:
    ax.plot(x, qlike[name], color=color, ls=ls, marker=marker, ms=ms,
            markerfacecolor=color, markeredgecolor=color, label=name, lw=1.2)

ax.set_xticks(x)
ax.set_xticklabels(horizons)
ax.set_xlabel("Forecast horizon")
ax.set_ylabel("QLIKE loss")
ax.legend(loc="upper right", handlelength=2.0, handletextpad=0.5,
          labelspacing=0.4)

# Save to both output and tex directories
for d in [FIG_DIR, TEX_FIG]:
    fig.savefig(os.path.join(d, "qlike_comparison.png"))
plt.close(fig)
print("Saved: qlike_comparison.png")

# ── Figure 2: QL tau=0.95 comparison ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 2.8))

styles = [
    ("Historical Vol", C["light_gray"], "-.",  "s",  5.0),
    ("GARCH(1,1)",     C["mid_gray"],   ":",   "D",  4.5),
    ("HAR-OLS",        C["mid_gray"],   "-",   "v",  4.5),
    ("HAR-QR",         C["dark_gray"],  "--",  "^",  5.0),
    ("HAR-X-QR",       C["black"],      "-",   "o",  5.0),
]

for name, color, ls, marker, ms in styles:
    ax.plot(x, ql95[name], color=color, ls=ls, marker=marker, ms=ms,
            markerfacecolor=color, markeredgecolor=color, label=name, lw=1.2)

ax.set_xticks(x)
ax.set_xticklabels(horizons)
ax.set_xlabel("Forecast horizon")
ax.set_ylabel(r"Quantile loss ($\tau = 0.95$)")
ax.legend(loc="upper right", handlelength=2.0, handletextpad=0.5,
          labelspacing=0.4)

# Scale y-axis to start near zero for honest comparison
ax.set_ylim(bottom=0)

for d in [FIG_DIR, TEX_FIG]:
    fig.savefig(os.path.join(d, "ql95_comparison.png"))
plt.close(fig)
print("Saved: ql95_comparison.png")

print("Done.")
