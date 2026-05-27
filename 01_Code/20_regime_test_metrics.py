"""
20_regime_test_metrics.py
=========================
Multi-metric robustness check on the ENSO regime test (HAR-QR / HAR-D-QR /
HAR-X-QR / HAR-XD-QR), per Action Item 4 of the 17.03.2026 meeting with
Dr. Bleher: "Verify results across QLIKE, Check-Loss, MSE - ensure story is
consistent across metrics."

Rationale
---------
Johannes asked for robustness across loss functions on the SAME models, not
across additional model variants. This script:

  1. Loads forecast CSVs for the four regime-test models (already produced
     by 07, 07b, 07c, 07d).
  2. Recomputes four loss measures on the median (q50) forecast and the
     0.95 quantile forecast:
       - Quantile loss at tau=0.95  (Check-Loss)
       - QLIKE on the median (Patton 2011 robust variance-loss)
       - MSE on the median
       - Empirical exceedance rate at tau=0.95  (target = 5%)
  3. Builds one combined comparison table.
  4. Renders ONE 2x2 figure for the slides: each panel = one metric vs
     horizon, four lines = four models.

Output:
  02_Output/tables/regime_test_multi_metrics.csv
  02_Output/figures/31_regime_test_multi_metrics.png
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
TBL_DIR  = os.path.join(BASE_DIR, "02_Output", "tables")
FIG_DIR  = os.path.join(BASE_DIR, "02_Output", "figures")
TEX_FIG  = os.path.join(os.path.dirname(BASE_DIR), "tex", "figures")
os.makedirs(TBL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Thesis plot style (matches 18_enso_regime_test.py)
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
    "lines.linewidth":     1.0,
    "legend.frameon":      False,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.facecolor":   "white",
    "pdf.fonttype":        42,
})

C = {
    "black":     "#000000",
    "dark_gray": "#333333",
    "mid_gray":  "#777777",
    "light_gray":"#AAAAAA",
}

HORIZONS = ["1d", "1w", "1m", "3m", "6m", "12m"]

MODELS = {
    "HAR-QR":    {"prefix": "har_qr",    "color": C["black"],     "ls": "-",  "marker": "o"},
    "HAR-D-QR":  {"prefix": "har_d_qr",  "color": C["dark_gray"], "ls": "--", "marker": "s"},
    "HAR-X-QR":  {"prefix": "har_x_qr",  "color": C["mid_gray"],  "ls": "-",  "marker": "^"},
    "HAR-XD-QR": {"prefix": "har_xd_qr", "color": C["light_gray"],"ls": "--", "marker": "D"},
}


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def quantile_loss(actual, forecast, tau):
    """Check function (Koenker & Bassett 1978)."""
    u = actual - forecast
    return np.mean(u * (tau - (u < 0).astype(float)))


def qlike(actual, forecast):
    """
    Patton (2011) QLIKE applied to variance forecasts.
        L(s2, s2hat) = s2/s2hat - log(s2/s2hat) - 1
    Both inputs are volatilities (sigma); we square them before applying.
    QLIKE penalises under-prediction of variance more than over-prediction,
    which matches the upper-tail framing of the thesis.
    """
    a2 = np.asarray(actual, dtype=float) ** 2
    f2 = np.asarray(forecast, dtype=float) ** 2
    f2 = np.clip(f2, 1e-12, None)
    a2 = np.clip(a2, 1e-12, None)
    r  = a2 / f2
    return np.mean(r - np.log(r) - 1.0)


def mse(actual, forecast):
    return np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)


def coverage_rate(actual, q95_forecast):
    """Empirical exceedance rate at tau=0.95.  Target = 5%."""
    return float((np.asarray(actual) > np.asarray(q95_forecast)).mean()) * 100.0


# ---------------------------------------------------------------------------
# Load forecasts
# ---------------------------------------------------------------------------
def load_forecasts():
    """
    Returns dict[model_name][horizon] -> DataFrame(Date, actual, q05..q95)
    """
    out = {}
    for model, cfg in MODELS.items():
        per_h = {}
        for h in HORIZONS:
            path = os.path.join(PROC_DIR, f"{cfg['prefix']}_forecasts_{h}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing forecast file: {path}")
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"])
            per_h[h] = df
        out[model] = per_h
    return out


# ---------------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------------
def build_metrics_table(forecasts):
    rows = []
    for model in MODELS:
        for h in HORIZONS:
            df = forecasts[model][h]
            rows.append({
                "Model":      model,
                "Horizon":    h,
                "QL_q95":     quantile_loss(df["actual"], df["q95"], 0.95),
                "QLIKE":      qlike(df["actual"], df["q50"]),
                "MSE":        mse(df["actual"], df["q50"]),
                "Coverage":   coverage_rate(df["actual"], df["q95"]),
                "Miscover":   abs(coverage_rate(df["actual"], df["q95"]) - 5.0),
                "N":          len(df),
            })
    tbl = pd.DataFrame(rows)
    cat = pd.CategoricalDtype(categories=HORIZONS, ordered=True)
    tbl["Horizon"] = tbl["Horizon"].astype(cat)
    return tbl.sort_values(["Model", "Horizon"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure: 2x2 panel, one metric each
# ---------------------------------------------------------------------------
def plot_multi_metrics(tbl):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.6), sharex=True)

    panels = [
        ("QL_q95",   axes[0, 0], "Quantile loss at $\\tau=0.95$",
         "QL  (lower is better)",  None),
        ("QLIKE",    axes[0, 1], "QLIKE on median (Patton 2011)",
         "QLIKE  (lower is better)", None),
        ("MSE",      axes[1, 0], "MSE on median forecast",
         "MSE  (lower is better)",  None),
        ("Coverage", axes[1, 1], "Empirical exceedance rate at $\\tau=0.95$",
         "Coverage rate (%)  -  target = 5%", 5.0),
    ]

    x = np.arange(len(HORIZONS))

    for col, ax, title, ylab, hline in panels:
        for model, cfg in MODELS.items():
            sub = tbl[tbl["Model"] == model].sort_values("Horizon")
            ax.plot(x, sub[col].values,
                    color=cfg["color"], linestyle=cfg["ls"],
                    linewidth=1.2, marker=cfg["marker"], markersize=4,
                    label=model)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        ax.set_xticklabels(HORIZONS)
        if hline is not None:
            ax.axhline(hline, color=C["black"], linewidth=0.6,
                       linestyle=":", alpha=0.7)
        if col in ("QL_q95", "MSE"):
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.4f"))
        elif col == "QLIKE":
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
        else:
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))

    for ax in axes[1, :]:
        ax.set_xlabel("Forecast horizon")

    # Single legend on top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=4, bbox_to_anchor=(0.5, 1.005), frameon=False)

    fig.suptitle("ENSO regime test - consistency across loss functions",
                 y=1.06, fontsize=10)
    fig.tight_layout()

    out = os.path.join(FIG_DIR, "31_regime_test_multi_metrics.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: 31_regime_test_multi_metrics.png")
    if os.path.isdir(TEX_FIG):
        shutil.copy(out, os.path.join(TEX_FIG, "31_regime_test_multi_metrics.png"))
        print("  Copied to tex/figures/")


# ---------------------------------------------------------------------------
# Figure: bar chart at 12m horizon (one bar per model x metric)
# ---------------------------------------------------------------------------
def plot_12m_bars(tbl):
    """
    Four side-by-side panels, each a bar chart at the 12m horizon.
    Bars are ordered HAR-QR, HAR-D-QR, HAR-X-QR, HAR-XD-QR.
    For Coverage, a horizontal line marks the 5% target.
    """
    sub = tbl[tbl["Horizon"] == "12m"].set_index("Model").loc[list(MODELS.keys())]

    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.0))

    panels = [
        ("QL_q95",   "Quantile loss  ($\\tau=0.95$)", "QL  (lower = better)",  "%.5f", None),
        ("QLIKE",    "QLIKE on median",                "QLIKE  (lower = better)", "%.3f", None),
        ("MSE",      "MSE on median",                  "MSE  (lower = better)", "%.2e", None),
        ("Coverage", "Exceedance rate ($\\tau=0.95$)", "Rate  (target = 5%)",  "%.1f", 5.0),
    ]

    colors = [MODELS[m]["color"] for m in MODELS]
    labels = list(MODELS.keys())
    x = np.arange(len(labels))

    for (col, title, ylab, fmt, hline), ax in zip(panels, axes):
        vals = sub[col].values
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")

        if hline is not None:
            ax.axhline(hline, color=C["black"], linewidth=0.6,
                       linestyle=":", alpha=0.7)

        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(fmt))

        # Highlight best (lowest, except for coverage where closest-to-5)
        if col == "Coverage":
            best_idx = int(np.argmin(np.abs(vals - 5.0)))
        else:
            best_idx = int(np.argmin(vals))
        bars[best_idx].set_edgecolor(C["black"])
        bars[best_idx].set_linewidth(1.4)

    fig.suptitle("Regime test at 12-month horizon  -  consistency across metrics",
                 y=1.02, fontsize=10)
    fig.tight_layout()

    out = os.path.join(FIG_DIR, "32_regime_test_12m_bars.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: 32_regime_test_12m_bars.png")
    if os.path.isdir(TEX_FIG):
        shutil.copy(out, os.path.join(TEX_FIG, "32_regime_test_12m_bars.png"))
        print("  Copied to tex/figures/")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary(tbl):
    print()
    print("=" * 88)
    print("  REGIME TEST  --  consistency across loss functions")
    print("=" * 88)

    for h in HORIZONS:
        sub = tbl[tbl["Horizon"] == h].set_index("Model")
        print(f"\n  Horizon: {h}")
        print(f"  {'Model':<10}  {'QL@.95':>10}  {'QLIKE':>10}  "
              f"{'MSE':>12}  {'Cov(%)':>8}")
        print("  " + "-" * 60)
        for m in MODELS:
            r = sub.loc[m]
            print(f"  {m:<10}  {r['QL_q95']:>10.6f}  {r['QLIKE']:>10.4f}  "
                  f"{r['MSE']:>12.2e}  {r['Coverage']:>8.2f}")

    # Long-horizon ranking
    print("\n" + "=" * 88)
    print("  Long-horizon ranking (12m) - lower = better, except Coverage (target 5%)")
    print("=" * 88)
    sub = tbl[tbl["Horizon"] == "12m"].set_index("Model")
    for col in ["QL_q95", "QLIKE", "MSE", "Miscover"]:
        order = sub[col].sort_values().index.tolist()
        ranked = "  >  ".join(order)
        print(f"  {col:<10} :  {ranked}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Regime test - multi-metric robustness")
    print("=" * 60)

    forecasts = load_forecasts()
    tbl       = build_metrics_table(forecasts)

    out_csv = os.path.join(TBL_DIR, "regime_test_multi_metrics.csv")
    tbl.to_csv(out_csv, index=False)
    print(f"\n  Saved: regime_test_multi_metrics.csv ({len(tbl)} rows)")

    plot_multi_metrics(tbl)
    plot_12m_bars(tbl)
    print_summary(tbl)

    print("Done.")
