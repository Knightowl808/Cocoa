"""
create_thesis_figures.py
========================
Generate the two figures required by 04_Data.tex:
  1. fig:price-series  — Daily closing price time series
  2. fig:volatility-series — Yang-Zhang annualized volatility time series

Uses the monochrome academic style from the plot-style skill.
Saves directly to tex/figures/ for LaTeX inclusion.
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
TEX_FIG_DIR = os.path.join(os.path.dirname(BASE_DIR), "tex", "figures")
os.makedirs(TEX_FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Academic style (from plot-style skill)
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
    "savefig.facecolor":    "white",
    "pdf.fonttype":         42,
})

C = {
    "black":       "#000000",
    "dark_gray":   "#333333",
    "mid_gray":    "#777777",
    "light_gray":  "#AAAAAA",
    "band_inner":  "#DDDDDD",
    "band_outer":  "#EEEEEE",
    "accent":      "#2B6CB0",
}

# ---------------------------------------------------------------------------
# Load data (from raw CSV — processed files may not exist)
# ---------------------------------------------------------------------------
proc_path = os.path.join(PROC_DIR, "daily_with_volatility.csv")
clean_path = os.path.join(PROC_DIR, "daily_clean.csv")
raw_path = os.path.join(BASE_DIR, "00_Data", "raw", "Daily.csv")

if os.path.exists(proc_path):
    df = pd.read_csv(proc_path)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"Loaded processed data: {len(df)} rows")
elif os.path.exists(clean_path):
    df = pd.read_csv(clean_path)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"Loaded clean data: {len(df)} rows")
else:
    # Load directly from raw
    print("Loading from raw Daily.csv ...")
    df = pd.read_csv(raw_path, sep=";", skiprows=1, encoding="utf-8-sig")
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Volume"] = (
        df["Volume"].astype(str).str.strip()
        .str.replace(" K", "e3", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="all")
    bad = df["High"] < df["Low"]
    if bad.any():
        df = df[~bad].reset_index(drop=True)
    print(f"Loaded raw data: {len(df)} rows")

    # Compute Yang-Zhang volatility inline
    n = len(df)
    log_oc = np.log(df["Open"] / df["Close"].shift(1))  # overnight
    log_co = np.log(df["Close"] / df["Open"])            # close-to-open
    log_ho = np.log(df["High"] / df["Open"])
    log_lo = np.log(df["Low"] / df["Open"])
    # Rogers-Satchell
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    df["vol_yz"] = np.sqrt(np.abs(rs))  # single-day RS approximation
    # Better: use a rolling window for smoothing
    # For the figure we just need a reasonable daily vol estimate
    # Use a proper YZ with rolling components
    window = 22  # monthly
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    sigma_o = log_oc.rolling(window).var()
    sigma_c = log_co.rolling(window).var()
    sigma_rs = rs.rolling(window).mean()
    df["vol_yz"] = np.sqrt(np.abs(sigma_o + k * sigma_c + (1 - k) * sigma_rs))
    print(f"Computed Yang-Zhang volatility (22-day rolling)")

# ---------------------------------------------------------------------------
# Figure 1: Price series
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 2.2))

ax.plot(df["Date"], df["Close"], color=C["black"], lw=0.6)
ax.set_ylabel("Price (GBP/tonne)")
ax.set_xlabel("")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

fig.savefig(os.path.join(TEX_FIG_DIR, "price_series.png"))
fig.savefig(os.path.join(TEX_FIG_DIR, "price_series.pdf"))
plt.close(fig)
print("Saved: price_series.png / .pdf")

# ---------------------------------------------------------------------------
# Figure 2: Yang-Zhang volatility time series
# ---------------------------------------------------------------------------
# Check which column name is available for YZ volatility
vol_col = None
for candidate in ["vol_yz", "yang_zhang", "YZ_vol", "sigma_yz", "yz_volatility"]:
    if candidate in df.columns:
        vol_col = candidate
        break

if vol_col is None:
    print(f"Available columns: {list(df.columns)}")
    raise ValueError("Cannot find Yang-Zhang volatility column in data")

# Annualize: multiply daily vol by sqrt(252) and convert to percent
vol_ann = df[vol_col] * np.sqrt(252) * 100

fig, ax = plt.subplots(figsize=(6.5, 2.2))

ax.plot(df["Date"], vol_ann, color=C["black"], lw=0.4)
ax.set_ylabel("Volatility (ann., %)")
ax.set_xlabel("")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.savefig(os.path.join(TEX_FIG_DIR, "volatility_timeseries.png"))
fig.savefig(os.path.join(TEX_FIG_DIR, "volatility_timeseries.pdf"))
plt.close(fig)
print("Saved: volatility_timeseries.png / .pdf")

print("\nDone. All figures saved to:", TEX_FIG_DIR)
