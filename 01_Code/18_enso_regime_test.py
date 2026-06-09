"""
18_enso_regime_test.py
======================
ENSO regime test: isolates ENSO's incremental contribution over and above
the 2022–2024 cocoa supply crisis regime effect.

Background (17.03.2026 meeting, Dr. Johannes Bleher):
  Johannes raised the concern that ENSO improvements at long horizons might
  be spurious — driven by coincidental overlap with the 2022–2024 cocoa
  crisis (CSSVD virus + weather shocks) rather than by ENSO's direct causal
  channel (West African rainfall → harvest quality → supply disruption →
  price volatility). He proposed:

    1. HAR-QR          — baseline (no ENSO, no dummy)
    2. HAR-D-QR        — crisis dummy only (script 07c)
    3. HAR-X-QR        — ENSO only (script 07b)
    4. HAR-XD-QR       — ENSO + crisis dummy (script 07d)

  If HAR-D-QR ≈ HAR-X-QR     → dummy and ENSO capture the same signal
                                 → ENSO benefit may be spurious
  If HAR-XD-QR << HAR-D-QR   → ENSO has incremental value beyond the dummy
                                 → ENSO is genuinely informative
  If HAR-XD-QR ≈  HAR-D-QR   → regime dummy absorbs ENSO's signal

This script:
  1. Loads the four quantile-loss tables from 02_Output/tables/
  2. Builds a wide comparison table (rows: Horizon×Quantile, cols: 4 models)
  3. Computes incremental gains over HAR-QR baseline
  4. Saves enso_regime_test_comparison.csv
  5. Produces two thesis-style figures:
       25_enso_regime_test_ql95.png    — QL at τ=0.95 for all 4 models × horizons
       26_enso_regime_test_incremental.png — incremental gain over HAR-QR at τ=0.95
"""

import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TBL_DIR  = os.path.join(BASE_DIR, "02_Output", "tables")
FIG_DIR  = os.path.join(BASE_DIR, "02_Output", "figures")
TEX_FIG  = os.path.join(os.path.dirname(BASE_DIR), "tex", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Thesis plot style (consistent with 09_comparison_plots.py)
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
    "black":     "#000000",
    "dark_gray": "#333333",
    "mid_gray":  "#777777",
    "light_gray":"#AAAAAA",
}

HORIZONS_ORDERED = ["1d", "1w", "1m", "3m", "6m", "12m"]
HORIZON_LABELS   = ["1d", "1w", "1m", "3m", "6m", "12m"]  # x-axis labels
QUANTILES        = [0.05, 0.25, 0.50, 0.75, 0.95]

# Model display names and line styles
MODELS = {
    "HAR-QR":   {"file": "har_qr_quantile_loss.csv",   "ls": "-",  "color": C["black"],     "lw": 1.2},
    "HAR-D-QR": {"file": "har_d_qr_quantile_loss.csv", "ls": "--", "color": C["dark_gray"], "lw": 1.2},
    "HAR-X-QR": {"file": "har_x_qr_quantile_loss.csv", "ls": "-",  "color": C["mid_gray"],  "lw": 1.2},
    "HAR-XD-QR":{"file": "har_xd_qr_quantile_loss.csv","ls": "--", "color": C["light_gray"],"lw": 1.2},
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_losses():
    """
    Load quantile-loss tables for all four models.
    Returns a dict: model_name -> DataFrame(Horizon, Quantile, QL).
    """
    losses = {}
    for name, cfg in MODELS.items():
        path = os.path.join(TBL_DIR, cfg["file"])
        if not os.path.exists(path):
            print(f"  WARNING: {cfg['file']} not found — run 07, 07b, 07c, 07d first")
            return None
        df = pd.read_csv(path)
        losses[name] = df
    return losses


def build_comparison_table(losses):
    """
    Build wide comparison table.
    Rows: (Horizon, Quantile)
    Columns: HAR-QR, HAR-D-QR, HAR-X-QR, HAR-XD-QR
    Plus incremental columns relative to baseline HAR-QR.
    """
    base = losses["HAR-QR"].copy().rename(columns={"QL": "HAR-QR"})
    merged = base[["Horizon", "Quantile", "HAR-QR"]].copy()

    for name in ["HAR-D-QR", "HAR-X-QR", "HAR-XD-QR"]:
        col = losses[name][["Horizon", "Quantile", "QL"]].rename(columns={"QL": name})
        merged = merged.merge(col, on=["Horizon", "Quantile"], how="left")

    # Incremental gain over HAR-QR baseline (negative = improvement)
    for name in ["HAR-D-QR", "HAR-X-QR", "HAR-XD-QR"]:
        merged[f"Delta_{name}"] = merged[name] - merged["HAR-QR"]

    # Additional: incremental gain of XD over D (ENSO on top of dummy)
    merged["Delta_ENSO_over_D"] = merged["HAR-XD-QR"] - merged["HAR-D-QR"]

    # Reorder horizons
    cat = pd.CategoricalDtype(categories=HORIZONS_ORDERED, ordered=True)
    merged["Horizon"] = merged["Horizon"].astype(cat)
    merged = merged.sort_values(["Horizon", "Quantile"]).reset_index(drop=True)

    return merged


# ---------------------------------------------------------------------------
# Figure A: QL at τ=0.95 for all 4 models × 6 horizons
# ---------------------------------------------------------------------------
def plot_ql95_comparison(losses):
    """
    Line chart: quantile loss at τ=0.95 for all four models across horizons.
    Lower is better.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    x = np.arange(len(HORIZONS_ORDERED))

    for name, cfg in MODELS.items():
        df = losses[name]
        ql = df[df["Quantile"] == 0.95].copy()
        cat = pd.CategoricalDtype(categories=HORIZONS_ORDERED, ordered=True)
        ql["Horizon"] = ql["Horizon"].astype(cat)
        ql = ql.sort_values("Horizon")
        y = ql["QL"].values
        ax.plot(x, y, linestyle=cfg["ls"], color=cfg["color"],
                linewidth=cfg["lw"], marker="o", markersize=3.5, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(HORIZON_LABELS)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Quantile loss (τ = 0.95)")
    ax.set_title("ENSO regime test — QL at τ = 0.95")
    ax.legend(loc="upper right", ncol=2)

    # Minimal tick marks on y-axis
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.4f"))

    out = os.path.join(FIG_DIR, "25_enso_regime_test_ql95.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: 25_enso_regime_test_ql95.png")

    # Copy to tex/figures if directory exists
    if os.path.isdir(TEX_FIG):
        import shutil
        shutil.copy(out, os.path.join(TEX_FIG, "25_enso_regime_test_ql95.png"))
        print(f"  Copied to tex/figures/")


# ---------------------------------------------------------------------------
# Figure B: Incremental QL improvement over HAR-QR baseline at τ=0.95
# ---------------------------------------------------------------------------
def plot_incremental_gains(comp_table):
    """
    Grouped bar chart: incremental QL gain at τ=0.95 relative to HAR-QR.

    A negative bar = improvement over baseline.
    Three bars per horizon: Δ HAR-D-QR, Δ HAR-X-QR, Δ HAR-XD-QR.

    The key comparison is at long horizons (6m, 12m) where ENSO is expected
    to contribute. If Δ HAR-D-QR ≈ Δ HAR-X-QR, the regime and ENSO signals
    overlap and ENSO's contribution may be coincidental.
    """
    tau95 = comp_table[comp_table["Quantile"] == 0.95].copy()
    cat = pd.CategoricalDtype(categories=HORIZONS_ORDERED, ordered=True)
    tau95["Horizon"] = tau95["Horizon"].astype(cat)
    tau95 = tau95.sort_values("Horizon").reset_index(drop=True)

    x     = np.arange(len(HORIZONS_ORDERED))
    width = 0.22

    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    bar_specs = [
        ("Delta_HAR-D-QR",  "HAR-D-QR (dummy only)",   C["dark_gray"],  -width),
        ("Delta_HAR-X-QR",  "HAR-X-QR (ENSO only)",    C["mid_gray"],    0.0),
        ("Delta_HAR-XD-QR", "HAR-XD-QR (ENSO + dummy)",C["light_gray"],  width),
    ]

    for col, label, color, offset in bar_specs:
        ax.bar(x + offset, tau95[col].values, width=width,
               color=color, label=label, edgecolor="white", linewidth=0.3)

    ax.axhline(0, color=C["black"], linewidth=0.7, linestyle="-")

    ax.set_xticks(x)
    ax.set_xticklabels(HORIZON_LABELS)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("ΔQL vs. HAR-QR (τ = 0.95)")
    ax.set_title("Incremental gain over baseline HAR-QR\n(negative = improvement)")
    ax.legend(loc="lower left", ncol=1)

    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.5f"))

    out = os.path.join(FIG_DIR, "26_enso_regime_test_incremental.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: 26_enso_regime_test_incremental.png")

    if os.path.isdir(TEX_FIG):
        import shutil
        shutil.copy(out, os.path.join(TEX_FIG, "26_enso_regime_test_incremental.png"))
        print(f"  Copied to tex/figures/")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary(comp_table):
    """Print readable summary table: QL at τ=0.95 for all four models."""
    tau95 = comp_table[comp_table["Quantile"] == 0.95].copy()

    print("\n" + "=" * 78)
    print("ENSO REGIME TEST -- Quantile Loss at tau = 0.95")
    print("=" * 78)
    print(f"{'Horizon':<8}  {'HAR-QR':>10}  {'HAR-D-QR':>10}  "
          f"{'HAR-X-QR':>10}  {'HAR-XD-QR':>10}")
    print("-" * 78)

    for _, row in tau95.iterrows():
        print(f"{row['Horizon']:<8}  "
              f"{row['HAR-QR']:>10.6f}  "
              f"{row['HAR-D-QR']:>10.6f}  "
              f"{row['HAR-X-QR']:>10.6f}  "
              f"{row['HAR-XD-QR']:>10.6f}")

    print("\n" + "=" * 78)
    print("INCREMENTAL GAIN over HAR-QR baseline (negative = improvement)")
    print("=" * 78)
    print(f"{'Horizon':<8}  {'D D-QR':>10}  {'D X-QR':>10}  "
          f"{'D XD-QR':>10}  {'D ENSO|D':>10}")
    print("-" * 78)

    for _, row in tau95.iterrows():
        print(f"{row['Horizon']:<8}  "
              f"{row['Delta_HAR-D-QR']:>10.6f}  "
              f"{row['Delta_HAR-X-QR']:>10.6f}  "
              f"{row['Delta_HAR-XD-QR']:>10.6f}  "
              f"{row['Delta_ENSO_over_D']:>10.6f}")

    print()
    print("  D ENSO|D = HAR-XD-QR minus HAR-D-QR")
    print("  -> measures ENSO's incremental contribution *over* the regime dummy")
    print()

    # Interpretation hints
    long_horizons = tau95[tau95["Horizon"].isin(["6m", "12m"])]
    if len(long_horizons) == 2:
        d_vs_x  = (long_horizons["Delta_HAR-D-QR"].values -
                   long_horizons["Delta_HAR-X-QR"].values)
        enso_over_d = long_horizons["Delta_ENSO_over_D"].values

        print("  Interpretation (6m + 12m):")
        if np.all(np.abs(d_vs_x) < 0.00005):
            print("  -> HAR-D-QR ~ HAR-X-QR: dummy and ENSO capture similar signal")
            print("     ENSO benefit at long horizons may be partly spurious")
        elif np.all(long_horizons["Delta_HAR-X-QR"].values <
                    long_horizons["Delta_HAR-D-QR"].values):
            print("  -> HAR-X-QR < HAR-D-QR: ENSO outperforms the regime dummy")
            print("     ENSO captures a distinct, informative channel")
        else:
            print("  -> Mixed: ENSO and dummy have varying relative performance")

        if np.all(enso_over_d < -0.00001):
            print("  -> D ENSO|D < 0: ENSO adds incremental value beyond the dummy")
        elif np.all(enso_over_d > 0.00001):
            print("  -> D ENSO|D > 0: dummy absorbs ENSO signal; no incremental gain")
        else:
            print("  -> D ENSO|D ~ 0: marginal ENSO contribution over regime dummy")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ENSO Regime Test — Four-Way Model Comparison")
    print("=" * 60)
    print()
    print("Models compared:")
    print("  HAR-QR    -- baseline (sigma_d, sigma_w, sigma_m)")
    print("  HAR-D-QR  -- + crisis dummy D_t  (07c)")
    print("  HAR-X-QR  -- + ENSO anomaly      (07b)")
    print("  HAR-XD-QR -- + ENSO + dummy      (07d)")
    print()

    # Load
    losses = load_losses()
    if losses is None:
        print("ERROR: Missing input files. Run 07, 07b, 07c, 07d first.")
        raise SystemExit(1)

    # Build comparison table
    comp_table = build_comparison_table(losses)
    comp_table.to_csv(
        os.path.join(TBL_DIR, "enso_regime_test_comparison.csv"), index=False
    )
    print(f"  Saved: enso_regime_test_comparison.csv "
          f"({len(comp_table)} rows)")

    # Figures
    print()
    plot_ql95_comparison(losses)
    plot_incremental_gains(comp_table)

    # Summary
    print_summary(comp_table)

    print("Done!")
