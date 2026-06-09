"""
07k_lag_decay_figure.py
Plot QL improvement (ΔQL = QL_lagged − QL_baseline) vs. ENSO lag at τ=0.95,
for horizons 1M, 6M, 12M. Negative = model beats baseline.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.join(os.path.dirname(__file__), "..", "02_Output")
TAB_DIR = os.path.join(ROOT, "tables")
FIG_DIR = os.path.join(ROOT, "figures")

# ── style ─────────────────────────────────────────────────────────────────────
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
    "black":     "#000000",
    "dark_gray": "#333333",
    "mid_gray":  "#777777",
}

# ── data ──────────────────────────────────────────────────────────────────────
lag_df = pd.read_csv(os.path.join(TAB_DIR, "lagged_enso_summary.csv"))
base_df = pd.read_csv(os.path.join(TAB_DIR, "har_qr_quantile_loss.csv"))

# baseline QL at τ=0.95 for 1m, 6m, 12m
baselines = (
    base_df[base_df["Quantile"] == 0.95]
    .set_index("Horizon")["QL"]
    .to_dict()
)

# filter lag table to τ=0.95, horizons of interest
horizons = ["1m", "6m", "12m"]
sub = lag_df[(lag_df["Quantile"] == 0.95) & (lag_df["Horizon"].isin(horizons))].copy()
sub["delta"] = sub.apply(lambda r: r["QL"] - baselines[r["Horizon"]], axis=1)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 2.8))

styles = {
    "1m":  dict(color=C["mid_gray"],  ls="-.",  lw=0.9, marker="o", ms=4),
    "6m":  dict(color=C["dark_gray"], ls="--",  lw=1.0, marker="s", ms=4),
    "12m": dict(color=C["black"],     ls="-",   lw=1.1, marker="D", ms=4),
}
labels = {"1m": "1-Month horizon", "6m": "6-Month horizon", "12m": "12-Month horizon"}

for h in horizons:
    d = sub[sub["Horizon"] == h].sort_values("Lag")
    ax.plot(d["Lag"], d["delta"] * 1e4, label=labels[h], **styles[h])

# baseline reference
ax.axhline(0, color="#AAAAAA", lw=0.8, ls=":")

# physical-channel window annotation
ax.axvspan(6, 12, color="#EEEEEE", zorder=0)
ax.text(8.8, ax.get_ylim()[1] if ax.get_ylim()[1] < 0 else 0.05,
        "agronomic\nwindow", fontsize=7, color="#999999",
        ha="center", va="bottom")

ax.set_xlabel("ENSO lag (months)")
ax.set_ylabel(r"$\Delta$QL $\times 10^{4}$ (negative = better)")
ax.set_xticks([0, 1, 3, 6, 12])
ax.legend(loc="lower right", handlelength=2)

fig.tight_layout()

stem = os.path.join(FIG_DIR, "38_lag_decay_enso_gain")
fig.savefig(stem + ".png")
fig.savefig(stem + ".pdf")
plt.close(fig)
print("Saved:", stem)
