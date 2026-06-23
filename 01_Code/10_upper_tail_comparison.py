"""
10_upper_tail_comparison.py
---------------------------
Fan charts for HAR-X-QR: realized vs. tau=0.50/0.75/0.95 forecast bands.

2x2 layout with a period split at 2022-01-01:
  Rows:    one-month (top) / twelve-month (bottom) horizon
  Columns: pre-crisis sample (left) / 2022-2025 supply shock (right)
The right column uses its own (larger) y-axis so the crisis-period
dynamics remain readable without distorting the long-sample view.

Output:
    python/02_Output/figures/  ->  10_upper_tail.png/.pdf
    tex/figures/               ->  upper_tail.png/.pdf
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
# Plot style
# ---------------------------------------------------------------------------
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
    "legend.fontsize":      7.5,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "lines.linewidth":      1.0,
    "legend.frameon":       False,
    "legend.borderpad":     0.3,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.transparent":  True,
    "pdf.fonttype":         42,
})

C = {
    "black":      "#000000",
    "dark_gray":  "#333333",
    "mid_gray":   "#777777",
    "band_inner": "#BBBBBB",
    "band_outer": "#DDDDDD",
}

SPLIT = pd.Timestamp("2023-01-01")

# y-axis limits per (horizon, period). Pre-crisis tightens the view;
# crisis range derived from data with a small headroom.
YLIM = {
    ("1m",  "pre"):    3.0,
    ("1m",  "crisis"): 11.0,
    ("6m",  "pre"):    3.0,
    ("6m",  "crisis"): 5.5,
    ("12m", "pre"):    3.0,
    ("12m", "crisis"): 5.5,
}


def plot_panel(ax, df_sub, ylim, ymin=0, is_top_left=False, show_ylabel=True):
    dates  = df_sub["Date"]
    actual = df_sub["actual"] * 100
    q50    = df_sub["q50"]    * 100
    q75    = df_sub["q75"]    * 100
    q95    = df_sub["q95"]    * 100

    ax.fill_between(dates, q75, q95,
                    color=C["band_outer"], linewidth=0,
                    label="75–95% band")
    ax.fill_between(dates, q50, q75,
                    color=C["band_inner"], linewidth=0,
                    label="50–75% band")
    ax.plot(dates, q50,
            color=C["dark_gray"], lw=0.9, ls="--",
            label=r"Median ($\tau = 0.50$)")
    ax.plot(dates, actual,
            color=C["black"], lw=0.4, alpha=0.85,
            label="YZ volatility")

    ax.set_ylim(ymin, ylim)
    if show_ylabel:
        ax.set_ylabel("Volatility (ann., %)")
    if is_top_left:
        ax.legend(loc="upper left", handlelength=1.5,
                  handletextpad=0.4, labelspacing=0.3)


def build_figure(split: pd.Timestamp, left_label: str, right_label: str):
    fig, axes = plt.subplots(
        3, 2,
        figsize=(7.2, 7.2),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.18, "hspace": 0.38},
    )

    horizons = [("1m",  "One-month horizon"),
                ("6m",  "Six-month horizon"),
                ("12m", "Twelve-month horizon")]
    row_prefixes = ["a", "b", "c"]

    for row, ((horizon, hlabel), prefix) in enumerate(zip(horizons, row_prefixes)):
        df = pd.read_csv(DATA / f"har_x_qr_forecasts_{horizon}.csv",
                         parse_dates=["Date"]).sort_values("Date")

        df_pre    = df[df["Date"] <  split]
        df_crisis = df[df["Date"] >= split]

        ax_left  = axes[row, 0]
        ax_right = axes[row, 1]

        plot_panel(ax_left,  df_pre,
                   ylim=YLIM[(horizon, "pre")],
                   ymin=0.5,
                   is_top_left=(row == 0),
                   show_ylabel=True)
        plot_panel(ax_right, df_crisis,
                   ylim=YLIM[(horizon, "crisis")],
                   is_top_left=False,
                   show_ylabel=True)

        ax_left.set_title(f"({prefix}1) {hlabel}: {left_label}",
                          loc="left", fontsize=9)
        ax_right.set_title(f"({prefix}2) {right_label}",
                           loc="left", fontsize=9)

        ax_left.xaxis.set_major_locator(mdates.YearLocator(5))
        ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_right.xaxis.set_major_locator(mdates.YearLocator(1))
        ax_right.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    variants = [
        (pd.Timestamp("2023-01-01"), "1995–2022", "2023–2025",
         "10_upper_tail",      "upper_tail"),
        (pd.Timestamp("2024-01-01"), "1995–2023", "2024–2025",
         "10_upper_tail_2024", "upper_tail_2024"),
    ]

    for split, left_lbl, right_lbl, py_stem, tex_stem in variants:
        fig = build_figure(split, left_lbl, right_lbl)
        for stem in [OUT_PY / py_stem, OUT_TEX / tex_stem]:
            fig.savefig(str(stem) + ".png")
            fig.savefig(str(stem) + ".pdf")
        plt.close(fig)
        print(f"Saved: {tex_stem} (split={split.date()})")
