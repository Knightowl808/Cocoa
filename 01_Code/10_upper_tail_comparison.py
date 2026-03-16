"""
10_upper_tail_comparison.py
---------------------------
Produces two figures comparing the tau=0.95 upper-quantile forecast boundary
from all four models against realized volatility, zoomed to 2015-2025.

Figure 1: 1-month horizon
Figure 2: 12-month horizon

Output locations:
    python/02_Output/figures/  ->  10_upper_tail_1m.png, 10_upper_tail_12m.png
    tex/figures/               ->  upper_tail_1m.png, upper_tail_12m.png
"""

import pathlib
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]          # Master/
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
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
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
    "band_outer": "#EEEEEE",
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
ZOOM_START = "2015-01-01"
ZOOM_END = "2025-12-31"


def _load(path: pathlib.Path) -> pd.DataFrame:
    """Load a forecast CSV, parse dates, sort, and zoom to window."""
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    df = df[(df["Date"] >= ZOOM_START) & (df["Date"] <= ZOOM_END)].copy()
    return df


def load_horizon(h: str) -> dict:
    """Return dict of DataFrames keyed by model name for a given horizon tag."""
    return {
        "har_x_qr": _load(DATA / f"har_x_qr_forecasts_{h}.csv"),
        "har_qr":   _load(DATA / f"har_qr_forecasts_{h}.csv"),
        "garch":    _load(DATA / f"garch_forecasts_{h}.csv"),
        "hist_vol": _load(DATA / f"hist_vol_forecasts_{h}.csv"),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_figure(h: str) -> plt.Figure:
    data = load_horizon(h)

    fig, ax = plt.subplots(figsize=(6.5, 2.8))

    ref = data["har_x_qr"]

    # Realized volatility -- solid black
    ax.plot(
        ref["Date"], ref["actual"] * 100,
        color=C["black"], linestyle="-", linewidth=1.0,
        label="Realized",
    )

    # HAR-X-QR tau=0.95 -- dark gray dashed
    ax.plot(
        data["har_x_qr"]["Date"], data["har_x_qr"]["q95"] * 100,
        color=C["dark_gray"], linestyle="--", linewidth=1.0,
        label=r"HAR-X-QR $\tau\!=\!0.95$",
    )

    # HAR-QR tau=0.95 -- mid gray dotted
    ax.plot(
        data["har_qr"]["Date"], data["har_qr"]["q95"] * 100,
        color=C["mid_gray"], linestyle=":", linewidth=1.0,
        label=r"HAR-QR $\tau\!=\!0.95$",
    )

    # GARCH tau=0.95 -- light gray dashdot
    ax.plot(
        data["garch"]["Date"], data["garch"]["q95"] * 100,
        color=C["light_gray"], linestyle="-.", linewidth=1.0,
        label=r"GARCH $\tau\!=\!0.95$",
    )

    # Historical Vol tau=0.95 -- very light gray dashdot, thinner
    ax.plot(
        data["hist_vol"]["Date"], data["hist_vol"]["q95"] * 100,
        color=C["band_inner"], linestyle="-.", linewidth=0.7,
        label=r"Hist. Vol $\tau\!=\!0.95$",
    )

    ax.set_ylabel("Volatility (ann., %)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")

    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig1 = make_figure("1m")
    fig1.savefig(OUT_PY / "10_upper_tail_1m.png")
    fig1.savefig(OUT_TEX / "upper_tail_1m.png")
    plt.close(fig1)
    print("Saved 1-month upper-tail comparison.")

    fig2 = make_figure("12m")
    fig2.savefig(OUT_PY / "10_upper_tail_12m.png")
    fig2.savefig(OUT_TEX / "upper_tail_12m.png")
    plt.close(fig2)
    print("Saved 12-month upper-tail comparison.")

    print("Done.")
