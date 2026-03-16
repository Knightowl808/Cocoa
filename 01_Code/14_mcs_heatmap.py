"""
14_mcs_heatmap.py
=================
MCS heatmap for ALL quantile losses (all taus x all horizons x all models).

Produces a 4-panel heatmap (one per model). Each panel is a 5x6 grid
(rows = quantiles, columns = horizons). Cell colour encodes MCS p-value:
  green  = high p-value (model is in or near the best-model set)
  red    = low p-value  (model is excluded)

Annotates each cell with the p-value and marks M* inclusion with a bold border.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

HORIZONS = ["1d", "1w", "1m", "3m", "6m", "12m"]
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
MODELS = ["Historical Vol", "GARCH(1,1)", "HAR-QR", "HAR-X-QR"]

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(TBL_DIR, "mcs_quantile_loss.csv"))

# ── Custom green-white-red colormap ──────────────────────────────────────────
# High p-value (green) = good, Low p-value (red) = bad
cmap = LinearSegmentedColormap.from_list(
    "mcs_pval",
    [
        (0.0,  "#C62828"),   # deep red   (p ≈ 0, excluded)
        (0.05, "#E57373"),   # light red
        (0.10, "#FFCDD2"),   # pale red   (just at threshold)
        (0.15, "#FFFFFF"),   # white      (borderline)
        (0.30, "#C8E6C9"),   # pale green
        (0.60, "#66BB6A"),   # medium green
        (1.0,  "#1B5E20"),   # deep green (p = 1, best model)
    ],
)

# ── Build figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5),
                          constrained_layout=True)
axes = axes.ravel()

for idx, model in enumerate(MODELS):
    ax = axes[idx]
    sub = df[df["Model"] == model]

    # Build matrix: rows = quantiles (top=0.95, bottom=0.05), cols = horizons
    mat = np.full((len(QUANTILES), len(HORIZONS)), np.nan)
    inc = np.full((len(QUANTILES), len(HORIZONS)), False)

    for _, row in sub.iterrows():
        qi = QUANTILES.index(row["Quantile"])
        hi = HORIZONS.index(row["Horizon"])
        mat[qi, hi] = row["MCS p-value"]
        inc[qi, hi] = row["In M*"]

    # Flip so tau=0.95 is at top
    mat = mat[::-1]
    inc = inc[::-1]
    q_labels = [f"{q:.2f}" for q in reversed(QUANTILES)]

    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                continue
            # Choose text colour for contrast
            txt_col = "white" if (val > 0.7 or val < 0.08) else "#333333"
            weight = "bold" if inc[i, j] else "normal"
            label = f"{val:.2f}"
            # Mark M* inclusion with a star
            if inc[i, j]:
                label += " *"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=7.5, color=txt_col, fontweight=weight)

    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels(HORIZONS)
    ax.set_yticks(range(len(q_labels)))
    ax.set_yticklabels(q_labels)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(r"Quantile $\tau$")
    ax.set_title(model, fontweight="bold")

# Shared colorbar
cbar = fig.colorbar(im, ax=axes, orientation="horizontal",
                     fraction=0.04, pad=0.08, shrink=0.6)
cbar.set_label("MCS p-value  (* = included in M* at 90% confidence)")

fig.suptitle("Model Confidence Set: Quantile Loss across All Quantiles and Horizons",
             fontsize=11, fontweight="bold", y=1.02)

for d in [FIG_DIR, TEX_FIG]:
    fig.savefig(os.path.join(d, "mcs_heatmap_all_quantiles.png"))
plt.close(fig)
print("Saved: mcs_heatmap_all_quantiles.png")
print("Done.")
