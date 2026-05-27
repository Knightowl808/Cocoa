"""
07l_expanding_window_enso_matrix.py
=====================================
Matrix version of the expanding-window ENSO gain diagnostic.

Produces a 4-row x 3-column panel grid:
  Rows    = forecast horizons  : 1m, 3m, 6m, 12m
  Columns = quantile levels    : tau = 0.50, 0.75, 0.95

Each panel shows the cumulative Delta_QL(T) for that (horizon, tau) pair,
computed identically to 07k but over the full set of combinations.

Output:
  python/02_Output/figures/37_expanding_window_enso_matrix.pdf
  python/02_Output/figures/37_expanding_window_enso_matrix.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
FIG_DIR  = os.path.join(BASE_DIR, "02_Output", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
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
    "font.size":           8,
    "axes.titlesize":      8,
    "axes.labelsize":      8,
    "xtick.labelsize":     7,
    "ytick.labelsize":     7,
    "legend.fontsize":     8,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.size":    3,
    "ytick.major.size":    3,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "lines.linewidth":     0.9,
    "legend.frameon":      False,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.facecolor":   "white",
    "pdf.fonttype":        42,
})

C = {
    "black":      "#000000",
    "dark_gray":  "#333333",
    "mid_gray":   "#777777",
    "light_gray": "#AAAAAA",
    "band_outer": "#EEEEEE",
}

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
HORIZONS  = ["1m", "3m", "6m", "12m"]
TAUS      = [0.50, 0.75, 0.95]
MIN_OBS   = 500
STEP_DAYS = 21   # ~1 month

HORIZON_LABELS = {"1m": "1-month", "3m": "3-month",
                  "6m": "6-month", "12m": "12-month"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quantile_loss(actual: np.ndarray, forecast: np.ndarray, tau: float) -> float:
    u = actual - forecast
    return float(np.mean(u * (tau - (u < 0).astype(float))))


def load_forecasts(horizon: str) -> dict[str, pd.DataFrame]:
    out = {}
    for model, template in [
        ("HAR-QR",   "har_qr_forecasts_{h}.csv"),
        ("HAR-X-QR", "har_x_qr_forecasts_{h}.csv"),
    ]:
        path = os.path.join(PROC_DIR, template.format(h=horizon))
        df   = pd.read_csv(path, parse_dates=["Date"])
        df   = df.sort_values("Date").reset_index(drop=True)
        out[model] = df
    return out


def expanding_delta(har: pd.DataFrame, harx: pd.DataFrame,
                    tau: float, min_obs: int, step: int) -> pd.DataFrame:
    q_col     = f"q{int(tau * 100):02d}"
    dates_all = har["Date"].values
    idx_range = range(min_obs - 1, len(dates_all), step)

    rows = []
    for i in idx_range:
        end_date = dates_all[i]
        mask     = har["Date"] <= end_date
        n        = mask.sum()
        if n < min_obs:
            continue
        if q_col not in har.columns or q_col not in harx.columns:
            continue

        sub_har  = har[mask]
        sub_harx = harx[harx["Date"] <= end_date]

        ql_har  = quantile_loss(sub_har["actual"].values,  sub_har[q_col].values,  tau)
        ql_harx = quantile_loss(sub_harx["actual"].values, sub_harx[q_col].values, tau)

        rows.append({
            "end_date":    pd.Timestamp(end_date),
            "delta_ql_e4": (ql_harx - ql_har) * 1e4,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compute all (horizon, tau) combinations
# ---------------------------------------------------------------------------
print("Computing expanding-window Delta_QL for all horizon x tau combinations...")
data = {}
for h in HORIZONS:
    forecasts = load_forecasts(h)
    for tau in TAUS:
        key = (h, tau)
        df  = expanding_delta(forecasts["HAR-QR"], forecasts["HAR-X-QR"],
                              tau=tau, min_obs=MIN_OBS, step=STEP_DAYS)
        data[key] = df
        final = df["delta_ql_e4"].iloc[-1] if len(df) else float("nan")
        print(f"  {h:>3s}  tau={tau:.2f}  n={len(df):4d}  "
              f"final Delta_QL = {final:+.4f} x10^-4")

# ---------------------------------------------------------------------------
# Plot: 4 rows (horizons) x 3 columns (taus)
# ---------------------------------------------------------------------------
nrows, ncols = len(HORIZONS), len(TAUS)
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(6.5, 7.8),
                         sharex=True)

crisis_start = pd.Timestamp("2022-01-01")
x_max = max(df["end_date"].max()
            for df in data.values() if len(df) > 0)

# Column headers (quantiles) — drawn as text above the top row
tau_labels = {0.50: r"$\tau = 0.50$  (median)",
              0.75: r"$\tau = 0.75$",
              0.95: r"$\tau = 0.95$  (upper tail)"}

for col_idx, tau in enumerate(TAUS):
    axes[0, col_idx].set_title(tau_labels[tau], fontsize=8, pad=4)

for row_idx, h in enumerate(HORIZONS):
    for col_idx, tau in enumerate(TAUS):
        ax  = axes[row_idx, col_idx]
        df  = data[(h, tau)]

        # Supply-crisis shading
        ax.axvspan(crisis_start, x_max,
                   color=C["band_outer"], alpha=1.0, zorder=0)

        # Zero reference
        ax.axhline(0, color=C["mid_gray"], lw=0.7, ls="--", zorder=1)

        # Delta QL curve
        if len(df) > 0:
            ax.plot(df["end_date"], df["delta_ql_e4"],
                    color=C["black"], lw=0.9, zorder=2)

        # Row label on the left-most column
        if col_idx == 0:
            ax.set_ylabel(HORIZON_LABELS[h], fontsize=8)

        # x-axis formatting (only on bottom row — sharex handles the rest)
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.YearLocator(5))

        # Tighten tick density
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="both"))

# Shared x-label on the bottom row only
for col_idx in range(ncols):
    axes[-1, col_idx].set_xlabel("")

fig.tight_layout(h_pad=0.6, w_pad=0.5)

png_path = os.path.join(FIG_DIR, "37_expanding_window_enso_matrix.png")
pdf_path = os.path.join(FIG_DIR, "37_expanding_window_enso_matrix.pdf")
fig.savefig(png_path)
fig.savefig(pdf_path)
plt.close(fig)
print(f"\nSaved: {png_path}")
print(f"Saved: {pdf_path}")
print("Done.")
