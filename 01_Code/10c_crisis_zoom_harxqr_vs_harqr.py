"""
10c_crisis_zoom_harxqr_vs_harqr.py
------------------------------------
2×3 matrix of crisis-period zoom panels comparing HAR-X-QR vs HAR-QR
at the τ=0.95 quantile for all six forecasting horizons.

Layout:
    Row 1: 1d | 1w | 1m
    Row 2: 3m | 6m | 12m

Each panel shows:
  - Realized volatility (black solid)
  - HAR-X-QR τ=0.95 (dark gray dashed)  <- with ENSO
  - HAR-QR    τ=0.95 (mid gray dotted)   <- without ENSO

Time window: 2021-06-01 – 2025-06-01 (covers pre-crisis, crisis, and partial recovery)

Output:
    python/02_Output/figures/10c_crisis_zoom_harxqr_vs_harqr.png
    tex/figures/crisis_zoom_harxqr_vs_harqr.png
"""

import pathlib
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = pathlib.Path(__file__).resolve().parents[2]
DATA    = ROOT / "python" / "00_Data" / "processed"
OUT_PY  = ROOT / "python" / "02_Output" / "figures"
OUT_TEX = ROOT / "tex" / "figures"

OUT_PY.mkdir(parents=True, exist_ok=True)
OUT_TEX.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Thesis monochrome style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "figure.facecolor": "white", "figure.dpi": 150,
    "axes.facecolor": "white", "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "lines.linewidth": 1.0, "legend.frameon": False, "legend.borderpad": 0.3,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

C = {
    "black":     "#000000",
    "dark_gray": "#333333",
    "mid_gray":  "#777777",
}

ZOOM_START = "2021-06-01"
ZOOM_END   = "2025-06-01"

HORIZONS = [
    ("1d",  "1-day"),
    ("1w",  "1-week"),
    ("1m",  "1-month"),
    ("3m",  "3-month"),
    ("6m",  "6-month"),
    ("12m", "12-month"),
]

PANEL_LABELS = ["a", "b", "c", "d", "e", "f"]


def load(tag: str, h: str) -> pd.DataFrame:
    path = DATA / f"{tag}_forecasts_{h}.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    return df[(df["Date"] >= ZOOM_START) & (df["Date"] <= ZOOM_END)]


def plot_panel(ax, h: str, h_label: str, panel_label: str, show_legend: bool) -> None:
    harxqr = load("har_x_qr", h)
    harqr  = load("har_qr",   h)

    # Realized volatility (from HAR-X-QR file, same actual series)
    ax.plot(harxqr["Date"], harxqr["actual"] * 100,
            color=C["black"], ls="-", lw=0.9, label="YZ volatility", zorder=3)

    # HAR-X-QR τ=0.95
    ax.plot(harxqr["Date"], harxqr["q95"] * 100,
            color=C["dark_gray"], ls="--", lw=1.0, label=r"HAR-X-QR ($\tau$=0.95)",
            zorder=2)

    # HAR-QR τ=0.95
    ax.plot(harqr["Date"], harqr["q95"] * 100,
            color=C["mid_gray"], ls=":", lw=1.0, label=r"HAR-QR ($\tau$=0.95)",
            zorder=2)

    ax.set_ylabel("Volatility (ann., %)", labelpad=3)
    ax.set_title(f"({panel_label}) {h_label}", loc="left", fontsize=8.5)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    if show_legend:
        ax.legend(loc="upper left", ncol=1, handlelength=1.8, fontsize=7.5)


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.0), sharex=True)
    axes_flat = axes.flatten()

    for idx, ((h, h_label), panel_label) in enumerate(zip(HORIZONS, PANEL_LABELS)):
        show_legend = (idx == 0)   # legend only in first panel
        plot_panel(axes_flat[idx], h, h_label, panel_label, show_legend)

    # Remove redundant y-labels on middle/right columns
    for idx in [1, 2, 4, 5]:
        axes_flat[idx].set_ylabel("")

    fig.autofmt_xdate(rotation=0, ha="center")
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    fig.tight_layout()

    fig.savefig(OUT_PY  / "10c_crisis_zoom_harxqr_vs_harqr.png")
    fig.savefig(OUT_TEX / "crisis_zoom_harxqr_vs_harqr.png")
    plt.close(fig)
    print("Saved: 10c_crisis_zoom_harxqr_vs_harqr.png")
