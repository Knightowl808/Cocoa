"""
05c_proxy_comparison_all.py
============================
Compare all four daily OHLC volatility proxies against Realized Volatility
computed from 5-minute intraday data, to establish that Yang-Zhang is the
most informative daily estimator.

Proxies: Yang-Zhang, Garman-Klass, Parkinson, Close-to-Close
Evaluation: Pearson correlation (appropriate because both RV and daily
proxies are imperfect — see meeting notes 17.02.2026 on microstructure noise).

Note: 5-min RV is subject to microstructure noise (bid-ask bounce, order impact).
Daily OHLC proxies miss intraday variance. Neither is the "true" volatility.
Correlation, not distance, is the appropriate comparison metric.

Outputs:
    python/02_Output/figures/05c_proxy_scatter_vs_rv.png   — 2x2 scatter panels
    python/02_Output/figures/05c_proxy_correlation_matrix.png — 5x5 correlation heatmap
    python/02_Output/tables/05c_proxy_correlations.csv    — summary table
    tex/figures/proxy_scatter_vs_rv.png
    tex/figures/proxy_correlation_matrix.png
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]   # python/
PROC = ROOT / "00_Data" / "processed"
FIG  = ROOT / "02_Output" / "figures"
TBL  = ROOT / "02_Output" / "tables"
TEX  = ROOT.parent / "tex" / "figures"

FIG.mkdir(parents=True, exist_ok=True)
TBL.mkdir(parents=True, exist_ok=True)
TEX.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Thesis monochrome style (matches 10b_crisis_zoom.py)
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
    "black":      "#000000",
    "dark_gray":  "#333333",
    "mid_gray":   "#777777",
    "light_gray": "#BBBBBB",
}


# ============================================================
# 1. Realized Volatility from 5-min data
# ============================================================
def compute_rv(df_5min: pd.DataFrame) -> pd.DataFrame:
    """Compute daily RV = sqrt(sum of squared 5-min log returns)."""
    df = df_5min.copy()
    df["log_close"] = np.log(df["Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["intraday_ret"] = df.groupby("Date")["log_close"].diff()

    rv = (
        df.groupby("Date")["intraday_ret"]
        .apply(lambda x: (x ** 2).sum())
        .reset_index()
    )
    rv.columns = ["Date", "RV"]
    rv["sigma_rv"] = np.sqrt(rv["RV"])

    n_obs = df.groupby("Date")["intraday_ret"].count().reset_index()
    n_obs.columns = ["Date", "n_obs"]
    rv = rv.merge(n_obs, on="Date")

    # Filter thin trading days
    rv = rv[rv["n_obs"] >= 20].copy()
    return rv[["Date", "sigma_rv"]]


# ============================================================
# 2. Single-day OHLC proxies
# ============================================================
def compute_daily_proxies(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute single-observation daily proxies for each estimator.

    Yang-Zhang  : already in sigma_yz (single-day YZ, used in HAR model)
    Garman-Klass: sqrt(0.5*HL^2 - (2ln2-1)*CO^2) per day
    Parkinson   : sqrt(HL^2 / (4ln2)) per day
    Close-to-Close: |log_ret| (absolute daily return as single-day vol proxy)
    """
    d = daily[["Date", "log_ret", "log_high", "log_low", "oc_ret", "sigma_yz"]].copy()

    hl = d["log_high"] - d["log_low"]

    # Parkinson (single day)
    d["sigma_park"] = np.sqrt((hl ** 2) / (4.0 * np.log(2)))

    # Garman-Klass (single day)
    gk = 0.5 * (hl ** 2) - (2.0 * np.log(2) - 1.0) * (d["oc_ret"] ** 2)
    d["sigma_gk"] = np.sqrt(gk.clip(lower=0))

    # Close-to-Close (single day: absolute log return)
    d["sigma_cc"] = d["log_ret"].abs()

    return d[["Date", "sigma_yz", "sigma_gk", "sigma_park", "sigma_cc"]]


# ============================================================
# 3. 2x2 scatter plot: each proxy vs RV
# ============================================================
def plot_scatter_matrix(merged: pd.DataFrame) -> None:
    proxies = [
        ("sigma_yz",   "Yang-Zhang"),
        ("sigma_gk",   "Garman-Klass"),
        ("sigma_park", "Parkinson"),
        ("sigma_cc",   "Close-to-Close"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5))
    axes = axes.flatten()

    for i, (col, name) in enumerate(proxies):
        ax = axes[i]
        x = merged[col].values
        y = merged["sigma_rv"].values

        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        xm, ym = x[mask], y[mask]

        r = np.corrcoef(xm, ym)[0, 1]

        ax.scatter(xm, ym, s=8, alpha=0.4, color=C["mid_gray"], rasterized=True,
                   linewidths=0)

        # 45-degree reference line
        upper = max(xm.max(), ym.max()) * 1.05
        ax.plot([0, upper], [0, upper], color=C["light_gray"], lw=0.8, ls="--",
                zorder=1, label="45° line")

        # OLS fit line
        coeffs = np.polyfit(xm, ym, 1)
        xr = np.linspace(0, xm.max(), 100)
        ax.plot(xr, np.polyval(coeffs, xr), color=C["dark_gray"], lw=1.0,
                ls="-", zorder=2)

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(name, labelpad=3)

        # y-label only on left column
        if i % 2 == 0:
            ax.set_ylabel("Realized Vol (5-min)", labelpad=3)

        # Pearson r and panel label
        panel = chr(ord("a") + i)
        ax.set_title(f"({panel}) {name}   $r = {r:.3f}$", loc="left", fontsize=8.5)

    fig.suptitle(
        "Proxy Validation: Daily OHLC Estimators vs Realized Volatility (5-min)",
        fontsize=9, y=1.01
    )
    fig.tight_layout()

    fig.savefig(FIG / "05c_proxy_scatter_vs_rv.png")
    fig.savefig(TEX / "proxy_scatter_vs_rv.png")
    plt.close(fig)
    print("Saved: 05c_proxy_scatter_vs_rv.png")


# ============================================================
# 4. Correlation matrix heatmap (5x5, grayscale)
# ============================================================
def plot_correlation_matrix(merged: pd.DataFrame) -> None:
    cols   = ["sigma_rv",   "sigma_yz",    "sigma_gk",    "sigma_park", "sigma_cc"]
    labels = ["RV (5-min)", "Yang-Zhang", "Garman-Klass", "Parkinson",  "Close-to-Close"]

    corr = merged[cols].corr()

    fig, ax = plt.subplots(figsize=(4.8, 4.0))

    # Grayscale: high correlation = dark gray
    im = ax.imshow(corr.values, cmap="Greys", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)

    # Annotate cells with Pearson r values
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            txt_color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Pearson $r$", fontsize=8)

    ax.set_title("Correlation Matrix: Volatility Proxies", fontsize=9, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG / "05c_proxy_correlation_matrix.png")
    fig.savefig(TEX / "proxy_correlation_matrix.png")
    plt.close(fig)
    print("Saved: 05c_proxy_correlation_matrix.png")


# ============================================================
# 5. Correlation by aggregation window (bar chart)
# ============================================================
def plot_rolling_correlations(merged: pd.DataFrame) -> None:
    """
    Compute Pearson r between each proxy and 5-min RV at different
    aggregation windows (1d, 5d, 22d rolling means).

    At longer windows, noise averages out and correlations improve.
    This mirrors the HAR structure: daily, weekly, monthly components.
    """
    WINDOWS = {"1-day (raw)": 1, "5-day (weekly)": 5, "22-day (monthly)": 22}
    proxy_cols  = ["sigma_yz",    "sigma_gk",    "sigma_park", "sigma_cc"]
    proxy_names = ["Yang-Zhang", "Garman-Klass", "Parkinson",  "Close-to-Close"]

    m = merged.sort_values("Date").copy()

    results = {}  # window_label -> list of r values
    for label, w in WINDOWS.items():
        if w == 1:
            rv_w  = m["sigma_rv"]
            cols_w = {col: m[col] for col in proxy_cols}
        else:
            rv_w  = m["sigma_rv"].rolling(w, min_periods=w).mean()
            cols_w = {col: m[col].rolling(w, min_periods=w).mean() for col in proxy_cols}

        mask = rv_w.notna()
        rs = []
        for col in proxy_cols:
            x = cols_w[col][mask]
            y = rv_w[mask]
            valid = x.notna() & y.notna()
            rs.append(round(x[valid].corr(y[valid]), 4))
        results[label] = rs

    # Print summary table
    print("\n=== Pearson r by Aggregation Window ===")
    header = f"{'Estimator':22s}" + "".join(f"{lbl:>22s}" for lbl in WINDOWS)
    print(header)
    for i, name in enumerate(proxy_names):
        row = f"  {name:20s}" + "".join(f"{results[lbl][i]:>22.4f}" for lbl in WINDOWS)
        print(row)

    # Save table
    rows = []
    for i, name in enumerate(proxy_names):
        row = {"Estimator": name}
        for lbl in WINDOWS:
            row[lbl] = results[lbl][i]
        rows.append(row)
    pd.DataFrame(rows).to_csv(TBL / "05c_proxy_correlations_by_window.csv", index=False)
    print("Saved: 05c_proxy_correlations_by_window.csv")

    # --- Grouped bar chart ---
    x = np.arange(len(proxy_names))
    n_windows = len(WINDOWS)
    bar_w = 0.22
    offsets = np.linspace(-(n_windows - 1) / 2, (n_windows - 1) / 2, n_windows) * bar_w

    grays = [C["dark_gray"], C["mid_gray"], C["light_gray"]]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    for j, (label, offset) in enumerate(zip(WINDOWS, offsets)):
        bars = ax.bar(x + offset, results[label], width=bar_w,
                      color=grays[j], label=label, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(proxy_names, fontsize=8.5)
    ax.set_ylabel("Pearson $r$ with Realized Volatility")
    ax.set_ylim(0, 1.0)
    ax.set_title("Proxy–RV Correlation by Aggregation Window", loc="left", fontsize=9)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.axhline(0, color=C["dark_gray"], lw=0.5)

    # Add value labels on bars
    for j, (label, offset) in enumerate(zip(WINDOWS, offsets)):
        for xi, val in zip(x, results[label]):
            ax.text(xi + offset, val + 0.015, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG / "05c_proxy_correlations_by_window.png")
    fig.savefig(TEX / "proxy_correlations_by_window.png")
    plt.close(fig)
    print("Saved: 05c_proxy_correlations_by_window.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # --- Realized Volatility ---
    print("Loading 5-min data and computing RV...")
    df5 = pd.read_csv(PROC / "5min_clean.csv")
    df5["Date"] = pd.to_datetime(df5["Date"])
    rv = compute_rv(df5)
    print(f"  RV computed for {len(rv)} trading days: "
          f"{rv['Date'].min().date()} to {rv['Date'].max().date()}")

    # --- Daily OHLC proxies ---
    print("Loading daily OHLC and computing proxies...")
    daily = pd.read_csv(PROC / "daily_with_volatility.csv")
    daily["Date"] = pd.to_datetime(daily["Date"])
    proxies = compute_daily_proxies(daily)

    # --- Merge on overlapping dates ---
    merged = rv.merge(proxies, on="Date", how="inner").dropna()
    print(f"  Overlapping dates: {len(merged)}")

    if len(merged) < 20:
        print("WARNING: Too few overlapping dates. Check date alignment.")
        print(f"  RV range:      {rv['Date'].min().date()} – {rv['Date'].max().date()}")
        print(f"  Proxies range: {proxies['Date'].min().date()} – {proxies['Date'].max().date()}")
    else:
        proxy_cols  = ["sigma_yz",    "sigma_gk",    "sigma_park", "sigma_cc"]
        proxy_names = ["Yang-Zhang", "Garman-Klass", "Parkinson",  "Close-to-Close"]

        print("\n=== Pearson Correlations with Realized Volatility ===")
        rows = []
        for col, name in zip(proxy_cols, proxy_names):
            r = merged["sigma_rv"].corr(merged[col])
            print(f"  {name:20s}: r = {r:.4f}")
            rows.append({"Estimator": name, "Pearson_r": round(r, 4)})

        pd.DataFrame(rows).to_csv(TBL / "05c_proxy_correlations.csv", index=False)
        print(f"\nSaved: 05c_proxy_correlations.csv")

        plot_scatter_matrix(merged)
        plot_correlation_matrix(merged)
        plot_rolling_correlations(merged)

    print("\nDone.")
