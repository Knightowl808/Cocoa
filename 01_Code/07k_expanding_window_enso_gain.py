"""
07k_expanding_window_enso_gain.py
==================================
Expanding-window ENSO gain diagnostic.

For each monthly evaluation end-date from the start of the OOS period
to the end of the sample, computes:

    Delta_QL(T_end) = QL(HAR-X-QR) - QL(HAR-QR)

at tau = 0.95 using only forecast-realization pairs with Date <= T_end.

The resulting curve shows WHEN the ENSO gain accumulated over time ---
whether it was always present (structural signal) or appeared suddenly
during the 2022-2024 supply crisis (sample-specific). No break date
is assumed or required anywhere in this script.

Horizons plotted: 12m (primary), 6m (secondary).
Quantile: tau = 0.95 (upper tail, thesis primary criterion).

Output:
  python/02_Output/figures/36_expanding_window_enso_gain.pdf
  python/02_Output/figures/36_expanding_window_enso_gain.png
  python/02_Output/figures/36b_expanding_window_enso_gain_price.pdf
  python/02_Output/figures/36b_expanding_window_enso_gain_price.png
  python/02_Output/tables/expanding_window_enso_gain.csv
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
TBL_DIR  = os.path.join(BASE_DIR, "02_Output", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

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
    "savefig.facecolor":   "white",
    "pdf.fonttype":        42,
})

C = {
    "black":      "#000000",
    "dark_gray":  "#333333",
    "mid_gray":   "#777777",
    "light_gray": "#AAAAAA",
    "band_inner": "#DDDDDD",
    "band_outer": "#EEEEEE",
}

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
HORIZONS   = ["6m", "12m"]
TAU        = 0.95
Q_COL      = f"q{int(TAU * 100):02d}"   # "q95"
MIN_OBS    = 500    # minimum observations before we plot (statistical stability)
STEP_DAYS  = 21     # advance end-date by ~1 month at a time (21 trading days)

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
    """
    For each monthly end-date, compute Delta_QL = QL(harx) - QL(har)
    over all observations with Date <= end_date.
    """
    dates_all = har["Date"].values

    # Build candidate end-dates: every `step` trading days from the series start
    idx_range = range(min_obs - 1, len(dates_all), step)

    rows = []
    for i in idx_range:
        end_date = dates_all[i]
        mask = har["Date"] <= end_date

        sub_har  = har[mask]
        sub_harx = harx[harx["Date"] <= end_date]

        n = mask.sum()
        if n < min_obs:
            continue

        q_col = f"q{int(tau * 100):02d}"
        if q_col not in sub_har.columns or q_col not in sub_harx.columns:
            continue

        ql_har  = quantile_loss(sub_har["actual"].values,  sub_har[q_col].values,  tau)
        ql_harx = quantile_loss(sub_harx["actual"].values, sub_harx[q_col].values, tau)

        rows.append({
            "end_date":    pd.Timestamp(end_date),
            "n_obs":       n,
            "ql_har":      ql_har,
            "ql_harx":     ql_harx,
            "delta_ql":    ql_harx - ql_har,       # negative = ENSO helps
            "delta_ql_e4": (ql_harx - ql_har) * 1e4,  # scaled for readability
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
results = {}
for h in HORIZONS:
    print(f"Computing expanding-window Delta_QL for horizon {h} ...")
    forecasts = load_forecasts(h)
    df_delta  = expanding_delta(
        forecasts["HAR-QR"], forecasts["HAR-X-QR"],
        tau=TAU, min_obs=MIN_OBS, step=STEP_DAYS
    )
    results[h] = df_delta
    n_pts = len(df_delta)
    final = df_delta["delta_ql_e4"].iloc[-1]
    first_neg = df_delta[df_delta["delta_ql"] < 0]["end_date"].min()
    print(f"  {n_pts} evaluation points | "
          f"final Delta_QL = {final:+.4f} x10^-4 | "
          f"first negative: {first_neg.date() if pd.notna(first_neg) else 'never'}")

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
rows_out = []
for h, df in results.items():
    for _, row in df.iterrows():
        rows_out.append({"horizon": h, **row})
csv_out = pd.DataFrame(rows_out)
csv_path = os.path.join(TBL_DIR, "expanding_window_enso_gain.csv")
csv_out.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 2.8))

# --- zero reference line ---
ax.axhline(0, color=C["mid_gray"], lw=0.8, ls="--", zorder=1)

# --- light shading from 2022-01-01 to signal the supply-crisis period ---
# Decorative only — for reader orientation. NOT a formal cutoff.
crisis_start = pd.Timestamp("2022-01-01")
x_max = max(df["end_date"].max() for df in results.values())
ax.axvspan(crisis_start, x_max,
           color=C["band_outer"], alpha=1.0, zorder=0,
           label="__nolegend__")

# --- 12m line (primary) ---
d12 = results["12m"]
ax.plot(d12["end_date"], d12["delta_ql_e4"],
        color=C["black"], lw=1.0, ls="-",
        label=r"12-month horizon ($\tau = 0.95$)", zorder=3)

# --- 6m line (secondary) ---
d6 = results["6m"]
ax.plot(d6["end_date"], d6["delta_ql_e4"],
        color=C["dark_gray"], lw=0.9, ls="--",
        label=r"6-month horizon ($\tau = 0.95$)", zorder=3)

# --- axes ---
ax.set_ylabel(r"$\Delta$QL $(\times 10^{-4})$", fontsize=9)
ax.set_xlabel("")

ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.YearLocator(1))


# --- crisis region label: placed inside the shaded band near the top ---
ax.legend(loc="lower left", handlelength=1.5, handletextpad=0.5, labelspacing=0.3)

# Place the crisis label after limits are set by the legend
ylims_tmp = ax.get_ylim()
ax.text(
    crisis_start + pd.Timedelta(days=45),
    ylims_tmp[1] - 0.04 * (ylims_tmp[1] - ylims_tmp[0]),
    "supply\ncrisis",
    fontsize=7, color=C["mid_gray"], va="top", ha="left",
)

fig.tight_layout()

png_path = os.path.join(FIG_DIR, "36_expanding_window_enso_gain.png")
pdf_path = os.path.join(FIG_DIR, "36_expanding_window_enso_gain.pdf")
fig.savefig(png_path)
fig.savefig(pdf_path)
plt.close(fig)
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

# ---------------------------------------------------------------------------
# Figure 36b: same plot + cocoa price on secondary y-axis
# ---------------------------------------------------------------------------
price_path = os.path.join(PROC_DIR, "daily_with_volatility.csv")
price_raw  = pd.read_csv(price_path, usecols=["Date", "Close"], parse_dates=["Date"])
price_raw  = price_raw.dropna(subset=["Close"]).sort_values("Date")
# Monthly average price (last trading day of each month)
price_monthly = (
    price_raw.set_index("Date")["Close"]
    .resample("ME").last()
    .dropna()
    .reset_index()
)
price_monthly.columns = ["Date", "price"]

fig2, ax2 = plt.subplots(figsize=(6.5, 2.8))

ax2_r = ax2.twinx()

# Price on the right axis — drawn first so ΔQL lines sit on top
ax2_r.plot(
    price_monthly["Date"], price_monthly["price"],
    color=C["light_gray"], lw=0.8, ls="-", zorder=1,
    label="Cocoa price (GBP)",
)
ax2_r.set_ylabel("Cocoa price (GBP)", fontsize=9, color=C["light_gray"])
ax2_r.tick_params(axis="y", labelcolor=C["light_gray"], labelsize=8)
ax2_r.spines["right"].set_visible(True)
ax2_r.spines["right"].set_color(C["light_gray"])
ax2_r.spines["right"].set_linewidth(0.8)
ax2_r.spines["top"].set_visible(False)

# Zero reference
ax2.axhline(0, color=C["mid_gray"], lw=0.8, ls="--", zorder=2)

# Supply-crisis shading
ax2.axvspan(crisis_start, x_max,
            color=C["band_outer"], alpha=1.0, zorder=0,
            label="__nolegend__")

# ΔQL lines
ax2.plot(d12["end_date"], d12["delta_ql_e4"],
         color=C["black"], lw=1.0, ls="-",
         label=r"12-month horizon ($\tau = 0.95$)", zorder=3)
ax2.plot(d6["end_date"], d6["delta_ql_e4"],
         color=C["dark_gray"], lw=0.9, ls="--",
         label=r"6-month horizon ($\tau = 0.95$)", zorder=3)

ax2.set_ylabel(r"$\Delta$QL $(\times 10^{-4})$", fontsize=9)
ax2.set_xlabel("")

ax2.xaxis.set_major_locator(mdates.YearLocator(5))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_minor_locator(mdates.YearLocator(1))

# Combine legend entries from both axes
lines_l, labels_l = ax2.get_legend_handles_labels()
lines_r, labels_r = ax2_r.get_legend_handles_labels()
ax2.legend(lines_l + lines_r, labels_l + labels_r,
           loc="lower left", handlelength=1.5,
           handletextpad=0.5, labelspacing=0.3)

ylims2 = ax2.get_ylim()
ax2.text(
    crisis_start + pd.Timedelta(days=45),
    ylims2[1] - 0.04 * (ylims2[1] - ylims2[0]),
    "supply\ncrisis",
    fontsize=7, color=C["mid_gray"], va="top", ha="left",
)

fig2.tight_layout()

png2 = os.path.join(FIG_DIR, "36b_expanding_window_enso_gain_price.png")
pdf2 = os.path.join(FIG_DIR, "36b_expanding_window_enso_gain_price.pdf")
fig2.savefig(png2)
fig2.savefig(pdf2)
plt.close(fig2)
print(f"Saved: {png2}")
print(f"Saved: {pdf2}")
print("\nDone.")
