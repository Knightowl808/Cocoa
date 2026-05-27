"""
07i_enso_cocoa_ccf.py
=====================
Cross-correlogram: Niño 3.4 vs. cocoa OHLC volatility.

Tests Path B (cross-correlogram branch) from the 28.04.2026 meeting.

Under the physical-channel hypothesis (ENSO → rainfall → harvest → supply
→ volatility), the CCF should peak at positive lags around +6 to +12
months — the agronomic transmission window.

Under the regime-indicator reading (ENSO correlates with current regime
state), the CCF should peak near lag 0.

The figure is purely descriptive (no model, no look-ahead) and can be
placed in the Data section. Two panels: full sample and pre-crisis
subsample (Date < 2024-02-08, Bai-Perron sup-Wald break), allowing direct
comparison with the pre-crisis QL results from 07e.

Outputs:
  02_Output/figures/35_enso_cocoa_ccf.png
  02_Output/tables/enso_cocoa_ccf.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

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
    "black":      "#000000",
    "dark_gray":  "#333333",
    "mid_gray":   "#777777",
    "light_gray": "#AAAAAA",
    "band_inner": "#DDDDDD",
}

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(BASE_DIR, "00_Data", "processed")
FIG_DIR   = os.path.join(BASE_DIR, "02_Output", "figures")
TBL_DIR   = os.path.join(BASE_DIR, "02_Output", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

CRISIS_CUTOFF = pd.Timestamp("2024-02-08")
MAX_LAG       = 24   # months each direction


# -----------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------
def load_monthly_data(pre_crisis_only=False):
    """
    Monthly-aggregate sigma_d from har_dataset, merged with Niño 3.4.
    Returns aligned (enso, vol) arrays and the sample label.
    """
    har = pd.read_csv(os.path.join(PROC_DIR, "har_dataset.csv"))
    har["Date"] = pd.to_datetime(har["Date"])

    if pre_crisis_only:
        har = har[har["Date"] < CRISIS_CUTOFF].copy()

    har["ym"] = har["Date"].dt.to_period("M")
    monthly_vol = (
        har.groupby("ym")["sigma_d"]
        .mean()
        .reset_index()
        .rename(columns={"sigma_d": "vol"})
    )

    nino = pd.read_csv(os.path.join(PROC_DIR, "nino34_clean.csv"))
    nino["Date"] = pd.to_datetime(nino["Date"])
    nino["ym"]   = nino["Date"].dt.to_period("M")
    nino = nino[["ym", "nino34"]].rename(columns={"nino34": "enso"})

    merged = monthly_vol.merge(nino, on="ym", how="inner").dropna()
    return merged["enso"].values, merged["vol"].values


# -----------------------------------------------------------------------
# 2. Compute CCF at lags -MAX_LAG to +MAX_LAG
# -----------------------------------------------------------------------
def compute_ccf(enso, vol, max_lag):
    """
    Returns lags array and corresponding correlations.

    Positive lag k: corr(enso_t, vol_{t+k})  →  ENSO leads vol
    Negative lag k: corr(vol_t, enso_{t+k})  →  vol leads ENSO

    Manual computation to avoid statsmodels version differences.
    """
    T = len(enso)
    e = (enso - enso.mean()) / enso.std(ddof=1)
    v = (vol   - vol.mean())  / vol.std(ddof=1)

    lags   = np.arange(-max_lag, max_lag + 1)
    corrs  = np.empty(len(lags))
    for i, k in enumerate(lags):
        if k >= 0:
            # ENSO leads vol by k: corr(e[0:T-k], v[k:T])
            corrs[i] = np.corrcoef(e[:T - k] if k > 0 else e,
                                   v[k:]     if k > 0 else v)[0, 1]
        else:
            # vol leads ENSO by |k|: corr(e[|k|:T], v[0:T-|k|])
            ak = abs(k)
            corrs[i] = np.corrcoef(e[ak:], v[:T - ak])[0, 1]

    ci = 1.96 / np.sqrt(T)
    return lags, corrs, ci


# -----------------------------------------------------------------------
# 3. Plot
# -----------------------------------------------------------------------
def plot_ccf(ax, lags, corrs, ci, panel_label, n_obs):
    bars = ax.bar(lags, corrs, color=C["light_gray"], width=0.8,
                  edgecolor=C["dark_gray"], linewidth=0.3)

    for bar, c in zip(bars, corrs):
        if abs(c) > ci:
            bar.set_facecolor(C["dark_gray"])

    ax.axhline(y=0,    color=C["black"],     linewidth=0.8)
    ax.axhline(y= ci,  color=C["dark_gray"], linewidth=0.8, linestyle="--",
               label=f"95% CI (±{ci:.3f})")
    ax.axhline(y=-ci,  color=C["dark_gray"], linewidth=0.8, linestyle="--")

    ax.axvspan(6, 12, alpha=0.07, color=C["black"],
               label="Physical-channel window (+6 to +12m)")
    ax.axvline(x=0, color=C["mid_gray"], linewidth=0.6, linestyle=":")

    ax.set_title(panel_label, loc="left")
    ax.set_ylabel("Cross-correlation")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(6))
    ax.set_xlim(-MAX_LAG - 0.5, MAX_LAG + 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.text(0.98, 0.95, f"N = {n_obs} months", transform=ax.transAxes,
            ha="right", va="top", fontsize=8)
    ax.legend(loc="upper left", handlelength=1.5, handletextpad=0.5,
              labelspacing=0.3)


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ENSO–Cocoa CCF: full sample and pre-crisis")
    print("=" * 60)

    enso_full, vol_full   = load_monthly_data(pre_crisis_only=False)
    enso_pre,  vol_pre    = load_monthly_data(pre_crisis_only=True)

    lags_f, corrs_f, ci_f = compute_ccf(enso_full, vol_full, MAX_LAG)
    lags_p, corrs_p, ci_p = compute_ccf(enso_pre,  vol_pre,  MAX_LAG)

    # Print summary
    peak_idx_f = np.argmax(np.abs(corrs_f))
    peak_idx_p = np.argmax(np.abs(corrs_p))
    print(f"\nFull sample  (N={len(enso_full)}m): "
          f"peak |r|={abs(corrs_f[peak_idx_f]):.3f} at lag={lags_f[peak_idx_f]}")
    print(f"Pre-crisis   (N={len(enso_pre)}m):  "
          f"peak |r|={abs(corrs_p[peak_idx_p]):.3f} at lag={lags_p[peak_idx_p]}")
    print(f"\n95% CI full={ci_f:.3f}, pre={ci_p:.3f}")
    n_sig_f = (np.abs(corrs_f) > ci_f).sum()
    n_sig_p = (np.abs(corrs_p) > ci_p).sum()
    print(f"Significant lags (|r| > CI): full={n_sig_f}, pre-crisis={n_sig_p}")

    # Lag-by-lag table — merge on lag to handle any length differences
    df_full = pd.DataFrame({
        "lag":           lags_f,
        "r_full":        corrs_f,
        "sig_full":      np.abs(corrs_f) > ci_f,
    })
    df_pre = pd.DataFrame({
        "lag":              lags_p,
        "r_precrisis":      corrs_p,
        "sig_precrisis":    np.abs(corrs_p) > ci_p,
    })
    df_out = df_full.merge(df_pre, on="lag", how="outer").sort_values("lag")
    out_csv = os.path.join(TBL_DIR, "enso_cocoa_ccf.csv")
    df_out.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\nSaved: {out_csv}")

    # Print table for positive lags only (physical-channel window)
    print("\n--- Positive lags (ENSO leads vol): ---")
    print(df_out[df_out["lag"] >= 0].to_string(index=False))

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.2), sharex=True)

    plot_ccf(axes[0], lags_f, corrs_f, ci_f,
             "(a) Full sample (1990–2026)", len(enso_full))
    plot_ccf(axes[1], lags_p, corrs_p, ci_p,
             "(b) Pre-crisis subsample (1990–Feb 2024)", len(enso_pre))

    axes[1].set_xlabel("Lag (months) — positive: ENSO leads volatility")

    fig.text(0.5, -0.01,
             "Dark bars significant at 5% level. Shaded region: physical-channel "
             "prediction (+6 to +12 months).",
             ha="center", fontsize=7, style="italic")

    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, "35_enso_cocoa_ccf.png")
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Saved: {out_fig}")
    print("\nDone.")
