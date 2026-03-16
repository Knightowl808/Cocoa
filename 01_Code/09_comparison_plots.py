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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR  = os.path.join(BASE_DIR, "02_Output", "figures")
TEX_FIG  = os.path.join(os.path.dirname(BASE_DIR), "tex", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Data ────────────────────────────────────────────────────────────────────
horizons      = ["1d", "1w", "1m", "3m", "6m", "12m"]
x             = np.arange(len(horizons))

# QLIKE (mean forecast, more negative = better)
qlike = {
    "Historical Vol": [-7.002, -7.256, -7.327, -7.334, -7.324, -7.310],
    "GARCH(1,1)":     [-7.242, -7.388, -7.406, -7.381, -7.356, -7.345],
    "HAR-OLS":        [-7.302, -7.482, -7.521, -7.515, -7.492, -7.485],
}

# Quantile loss at tau=0.95 (lower = better)
ql95 = {
    "Historical Vol": [0.001502, 0.001023, 0.000871, 0.000889, 0.000914, 0.000930],
    "GARCH(1,1)":     [0.001986, 0.000975, 0.000590, 0.000570, 0.000607, 0.000663],
    "HAR-QR":         [0.001193, 0.000692, 0.000458, 0.000433, 0.000454, 0.000446],
    "HAR-X-QR":       [0.001195, 0.000695, 0.000464, 0.000418, 0.000387, 0.000333],
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
