"""
11_exceedance_plot.py
---------------------
Cumulative violation (exceedance) plot for the tau=0.95 upper quantile.

For each model, counts how many times realized volatility exceeds the
predicted 95th percentile boundary over time. A perfectly calibrated
model's cumulative violations grow linearly at 5%.

Output: tex/figures/exceedance_12m.png (preview only)
"""

import pathlib
import numpy as np
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
}


def load(tag, h):
    path = DATA / f"{tag}_forecasts_{h}.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    return df


def cumulative_violations(df):
    """Return dates and cumulative violation count."""
    exceeded = (df["actual"] > df["q95"]).astype(int)
    return df["Date"].values, np.cumsum(exceeded.values)


def make_exceedance_plot(h, horizon_label):
    models = {
        "HAR-X-QR": ("har_x_qr", C["dark_gray"], "--"),
        "HAR-QR":   ("har_qr",   C["mid_gray"],  ":"),
        "GARCH":    ("garch",    C["light_gray"], "-."),
        "Hist. Vol": ("hist_vol", C["band_inner"], "-."),
    }

    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    # Find common date range for the 5% reference line
    ref_df = load("har_x_qr", h)
    n = len(ref_df)
    dates_ref = ref_df["Date"].values
    nominal = np.arange(1, n + 1) * 0.05

    # Nominal 5% line
    ax.plot(dates_ref, nominal, color=C["black"], linestyle="-",
            linewidth=0.8, label="Nominal 5%")

    for name, (tag, color, ls) in models.items():
        df = load(tag, h)
        dates, cumviol = cumulative_violations(df)
        ax.plot(dates, cumviol, color=color, linestyle=ls,
                linewidth=1.0, label=name)

    ax.set_ylabel("Cumulative violations")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")

    ax.legend(loc="upper left", handlelength=2.0)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    for h, label in [("1m", "1-month"), ("12m", "12-month")]:
        fig = make_exceedance_plot(h, label)
        fig.savefig(OUT_PY / f"11_exceedance_{h}.png")
        fig.savefig(OUT_TEX / f"exceedance_{h}.png")
        plt.close(fig)
        print(f"Saved {label} exceedance plot.")

    print("Done.")
