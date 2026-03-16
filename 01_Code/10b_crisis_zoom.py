"""
10b_crisis_zoom.py
------------------
Two-panel figure zoomed to the 2022-2024 cocoa supply crisis.
Top: 1-month horizon.  Bottom: 12-month horizon.

Shows tau=0.95 boundaries from all four models vs realized volatility.
At this scale, model separation is clearly visible.

Output:
    python/02_Output/figures/10b_crisis_zoom.png
    tex/figures/crisis_zoom.png
"""

import pathlib
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "python" / "00_Data" / "processed"
OUT_PY = ROOT / "python" / "02_Output" / "figures"
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
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "lines.linewidth": 1.0, "legend.frameon": False, "legend.borderpad": 0.3,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

C = {
    "black": "#000000",
    "dark_gray": "#333333",
    "mid_gray": "#777777",
    "light_gray": "#AAAAAA",
    "band_inner": "#DDDDDD",
}

ZOOM_START = "2021-06-01"
ZOOM_END = "2025-06-01"


def load(tag, h):
    path = DATA / f"{tag}_forecasts_{h}.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    df = df[(df["Date"] >= ZOOM_START) & (df["Date"] <= ZOOM_END)]
    return df


def plot_panel(ax, h, panel_label):
    """Plot one horizon panel."""
    models = [
        ("har_x_qr", r"HAR-X-QR", C["dark_gray"], "--", 1.0),
        ("har_qr",   r"HAR-QR",   C["mid_gray"],  ":",  1.0),
        ("garch",    r"GARCH",    C["light_gray"], "-.", 1.0),
        ("hist_vol", r"Hist. Vol", C["band_inner"], "-.", 0.7),
    ]

    # Realized
    ref = load("har_x_qr", h)
    ax.plot(ref["Date"], ref["actual"] * 100,
            color=C["black"], ls="-", lw=1.0, label="Realized")

    # Model boundaries
    for tag, name, color, ls, lw in models:
        df = load(tag, h)
        ax.plot(df["Date"], df["q95"] * 100,
                color=color, ls=ls, lw=lw, label=name)

    ax.set_ylabel("Volatility (ann., %)")
    ax.set_title(f"({panel_label}) {h} horizon", loc="left", fontsize=9)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True)

    plot_panel(ax1, "1m", "a")
    plot_panel(ax2, "12m", "b")

    # Legend only on top panel
    ax1.legend(loc="upper left", ncol=3, handlelength=1.8)

    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()

    fig.savefig(OUT_PY / "10b_crisis_zoom.png")
    fig.savefig(OUT_TEX / "crisis_zoom.png")
    plt.close(fig)
    print("Saved crisis zoom plot.")
