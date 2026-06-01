"""
create_thesis_figures.py
========================
Generate the three figures required by 04_Data.tex:
  1. fig:price-series       — Daily closing price time series
  2. fig:volatility-series  — Yang-Zhang annualized volatility (no regime highlight)
  3. fig:enso-regimes       — Niño 3.4 index with El Niño / La Niña shading

Style: monochrome academic (plot-style skill). Transparent background.
Saves PNG + PDF to tex/figures/.
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR    = os.path.join(BASE_DIR, "00_Data", "processed")
TEX_FIG_DIR = os.path.join(os.path.dirname(BASE_DIR), "tex", "figures")
os.makedirs(TEX_FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Academic style (plot-style skill)
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
    "legend.fontsize":      8,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.minor.visible":  False,
    "ytick.minor.visible":  False,
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
    "light_gray": "#AAAAAA",
    "band_inner": "#DDDDDD",
    "band_outer": "#EEEEEE",
}

# ---------------------------------------------------------------------------
# Helper: save PNG + PDF
# ---------------------------------------------------------------------------
def save(fig, stem):
    fig.savefig(stem + ".png")
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    print(f"Saved: {os.path.basename(stem)}.png / .pdf")

# ---------------------------------------------------------------------------
# Load cocoa daily data
# ---------------------------------------------------------------------------
proc_path  = os.path.join(PROC_DIR, "daily_with_volatility.csv")
clean_path = os.path.join(PROC_DIR, "daily_clean.csv")
raw_path   = os.path.join(BASE_DIR, "00_Data", "raw", "Daily.csv")

if os.path.exists(proc_path):
    df = pd.read_csv(proc_path)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"Loaded processed data: {len(df)} rows")
elif os.path.exists(clean_path):
    df = pd.read_csv(clean_path)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"Loaded clean data: {len(df)} rows")
else:
    print("Loading from raw Daily.csv ...")
    df = pd.read_csv(raw_path, sep=";", skiprows=1, encoding="utf-8-sig")
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="all")
    df = df[df["High"] >= df["Low"]].reset_index(drop=True)
    print(f"Loaded raw data: {len(df)} rows")

    window = 22
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    log_oc = np.log(df["Open"] / df["Close"].shift(1))
    log_co = np.log(df["Close"] / df["Open"])
    log_ho = np.log(df["High"] / df["Open"])
    log_lo = np.log(df["Low"] / df["Open"])
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    sigma_o  = log_oc.rolling(window).var()
    sigma_c  = log_co.rolling(window).var()
    sigma_rs = rs.rolling(window).mean()
    df["vol_yz"] = np.sqrt(np.abs(sigma_o + k * sigma_c + (1 - k) * sigma_rs))
    print("Computed Yang-Zhang volatility (22-day rolling)")

# ---------------------------------------------------------------------------
# Figure 1: Price series
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 2.2))
ax.plot(df["Date"], df["Close"], color=C["black"], lw=0.6)
ax.set_ylabel("Price (GBP/tonne)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
save(fig, os.path.join(TEX_FIG_DIR, "price_series"))

# ---------------------------------------------------------------------------
# Figure 2: Yang-Zhang volatility (single panel, no regime highlight)
# ---------------------------------------------------------------------------
vol_col = next(
    (c for c in ["vol_yz", "yang_zhang", "YZ_vol", "sigma_yz", "yz_volatility"]
     if c in df.columns),
    None
)
if vol_col is None:
    raise ValueError(f"Cannot find YZ volatility column. Available: {list(df.columns)}")

vol_ann = df[vol_col] * np.sqrt(252) * 100

fig, ax = plt.subplots(figsize=(6.5, 2.2))
ax.plot(df["Date"], vol_ann, color=C["black"], lw=0.4)
ax.set_ylabel("Volatility (ann., %)")
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
save(fig, os.path.join(TEX_FIG_DIR, "volatility_timeseries"))

# ---------------------------------------------------------------------------
# Figure 3: ENSO regimes — Niño 3.4 with El Niño (red) and La Niña (blue)
# ---------------------------------------------------------------------------
enso_path = os.path.join(PROC_DIR, "nino34_clean.csv")
enso = pd.read_csv(enso_path)
enso["Date"] = pd.to_datetime(enso["Date"])
enso = enso.sort_values("Date").reset_index(drop=True)

# Clip ENSO to cocoa sample period (month-aligned on both ends)
enso_start = df["Date"].min().replace(day=1)
enso_end   = df["Date"].max().replace(day=1)
enso = enso[(enso["Date"] >= enso_start) & (enso["Date"] <= enso_end)].reset_index(drop=True)

# Upsample monthly → daily by linear interpolation for precise threshold fills
enso_daily = (
    enso.set_index("Date")["nino34"]
    .resample("D")
    .interpolate(method="linear")
    .reset_index()
)
enso_daily.columns = ["Date", "nino34"]

fig, ax = plt.subplots(figsize=(6.5, 2.2))

# Regime fills — fully opaque, clipped between threshold and curve
ax.fill_between(enso_daily["Date"], enso_daily["nino34"], 0.5,
                where=(enso_daily["nino34"] > 0.5),
                color="#C0392B", alpha=0.65, linewidth=0,
                interpolate=True, label="El Niño (> +0.5 °C)")
ax.fill_between(enso_daily["Date"], enso_daily["nino34"], -0.5,
                where=(enso_daily["nino34"] < -0.5),
                color="#2471A3", alpha=0.65, linewidth=0,
                interpolate=True, label="La Niña (< −0.5 °C)")

# Index line
ax.plot(enso_daily["Date"], enso_daily["nino34"], color=C["black"], lw=0.6)

# Threshold lines
ax.axhline( 0.5, color=C["mid_gray"], lw=0.7, ls="--", zorder=0)
ax.axhline(-0.5, color=C["mid_gray"], lw=0.7, ls="--", zorder=0)
ax.axhline( 0.0, color=C["light_gray"], lw=0.5, ls=":",  zorder=0)

ax.set_ylabel("Niño 3.4 anomaly (°C)")
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(loc="upper left", handlelength=1.2, handletextpad=0.4,
          labelspacing=0.3)

save(fig, os.path.join(TEX_FIG_DIR, "enso_regimes"))

print("\nDone. All figures saved to:", TEX_FIG_DIR)
