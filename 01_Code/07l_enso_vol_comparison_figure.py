"""
07l_enso_vol_comparison_figure.py
Plot ΔQL (τ=0.95) vs. forecast horizon for three ENSO specifications:
  - ENSO level (Niño 3.4)
  - ENSO vol-12m (rolling 12m stdev)
  - ENSO vol-36m (rolling 36m stdev)
Negative = beats HAR-QR baseline.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT    = os.path.join(os.path.dirname(__file__), "..", "02_Output")
TAB_DIR = os.path.join(ROOT, "tables")
FIG_DIR = os.path.join(ROOT, "figures")

# ── style ──────────────────────────────────────────────────────────────────────
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
    "legend.borderpad":    0.3,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.transparent": True,
    "pdf.fonttype":        42,
})

C = {
    "black":      "#000000",
    "dark_gray":  "#333333",
    "light_gray": "#AAAAAA",
}

# ── data ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(TAB_DIR, "har_x_vol_qr_comparison.csv"))
sub = df[df["Quantile"] == 0.95].copy()

# horizon order
horizon_order = ["1d", "1w", "1m", "3m", "6m", "12m"]
horizon_labels = ["1d", "1w", "1m", "3m", "6m", "12m"]
sub["Horizon"] = pd.Categorical(sub["Horizon"], categories=horizon_order, ordered=True)
sub = sub.sort_values("Horizon")

x = range(len(horizon_order))

# scale to ×10^4
level  = sub["Delta_ENSO-level"].values * 1e4
vol12  = sub["Delta_vol-12m"].values    * 1e4
vol36  = sub["Delta_vol-36m"].values    * 1e4

# ── plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 2.8))

ax.plot(x, level, color=C["black"],      ls="-",   lw=1.1, marker="D", ms=4,
        label="ENSO level (Niño 3.4)")
ax.plot(x, vol12, color=C["dark_gray"],  ls="--",  lw=1.0, marker="s", ms=4,
        label="ENSO vol-12m")
ax.plot(x, vol36, color=C["light_gray"], ls="-.",  lw=0.9, marker="o", ms=4,
        label="ENSO vol-36m")

ax.axhline(0, color="#AAAAAA", lw=0.8, ls=":")

ax.set_xticks(list(x))
ax.set_xticklabels(horizon_labels)
ax.set_xlabel("Forecast horizon")
ax.set_ylabel(r"$\Delta$QL $\times 10^{4}$ (negative = better)")
ax.legend(loc="lower left", handlelength=2)

fig.tight_layout()

stem = os.path.join(FIG_DIR, "39_enso_vol_comparison")
fig.savefig(stem + ".png")
fig.savefig(stem + ".pdf")
plt.close(fig)
print("Saved:", stem)
