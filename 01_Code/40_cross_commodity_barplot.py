"""
40_cross_commodity_barplot.py
Grouped bar chart: ΔQL (τ=0.95) for Cocoa, Coffee, Sugar at 6m and 12m horizons.
Data from Appendix Table A.3 (cross-commodity ENSO gain).
Negative = HAR-X-QR beats HAR-QR.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT    = os.path.join(os.path.dirname(__file__), "..", "02_Output")
FIG_DIR = os.path.join(ROOT, "figures")

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
    "legend.frameon":      False,
    "legend.borderpad":    0.3,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.transparent": True,
    "pdf.fonttype":        42,
})

# ── data (from Appendix Table A.3) ────────────────────────────────────────────
commodities = ["Cocoa", "Coffee", "Sugar"]
delta_6m  = np.array([+0.000007, +0.000015, -0.000010]) * 1e4
delta_12m = np.array([-0.000034, +0.000079, -0.000007]) * 1e4

# ── plot ──────────────────────────────────────────────────────────────────────
x = np.arange(len(commodities))
width = 0.32

fig, ax = plt.subplots(figsize=(5.5, 3.0))

bars_6m  = ax.bar(x - width/2, delta_6m,  width, label="6-month horizon",
                  color="#777777", edgecolor="white", linewidth=0.5)
bars_12m = ax.bar(x + width/2, delta_12m, width, label="12-month horizon",
                  color="#111111", edgecolor="white", linewidth=0.5)

ax.axhline(0, color="#AAAAAA", lw=0.8, ls=":")

# value labels on bars
for bar in list(bars_6m) + list(bars_12m):
    h = bar.get_height()
    sign = "+" if h >= 0 else ""
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + (0.003 if h >= 0 else -0.006),
        f"{sign}{h:.3f}",
        ha="center", va="bottom" if h >= 0 else "top",
        fontsize=7, color="#333333",
    )

ax.set_xticks(x)
ax.set_xticklabels(commodities)
ax.set_ylabel(r"$\Delta$QL $\times 10^{4}$ (negative = better)")
ax.legend(loc="upper right")

# note below plot
fig.text(
    0.5, -0.04,
    r"$\Delta\mathrm{QL} = \mathrm{QL}_{\mathrm{HAR\text{-}X\text{-}QR}} - \mathrm{QL}_{\mathrm{HAR\text{-}QR}}$"
    r"$\;\;$($\tau = 0.95$)",
    ha="center", fontsize=7.5, color="#555555",
)

fig.tight_layout()

stem = os.path.join(FIG_DIR, "40_cross_commodity_enso_gain")
fig.savefig(stem + ".png")
fig.savefig(stem + ".pdf")
plt.close(fig)
print("Saved:", stem)
