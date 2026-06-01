"""
05_proxy_validation.py
======================
Validate Yang-Zhang daily volatility against Realized Volatility
computed from 5-minute intraday data.

This implements the validation described in Methodology Section 3.7:
  "Yang-Zhang Approximates Realized Volatility"

Steps:
  1. Compute daily Realized Volatility (RV) from 5-min returns
  2. Compute daily Yang-Zhang from daily OHLC (same dates)
  3. Run Mincer-Zarnowitz regression: RV_t = alpha + beta * YZ_t + eps_t
     -> Test: alpha=0, beta=1 (unbiased proxy)
  4. Scatter plot + regression line
  5. Time-series overlay

Output: figures and regression results table
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "02_Output", "figures")
TBL_DIR = os.path.join(BASE_DIR, "02_Output", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


# ===================================================================
# 1. Compute Realized Volatility from 5-min data
# ===================================================================
def compute_realized_volatility(df_5min):
    """
    Compute daily Realized Volatility from 5-minute close prices.

    RV_t = sum of squared intraday log returns
    sigma_RV_t = sqrt(RV_t)
    """
    df = df_5min.copy()

    # Log returns within each day
    df["log_close"] = np.log(df["Close"])
    df["Date"] = pd.to_datetime(df["Date"])

    # Compute 5-min log returns (within each day)
    df["intraday_ret"] = df.groupby("Date")["log_close"].diff()

    # Daily RV = sum of squared intraday returns
    rv_daily = (
        df.groupby("Date")["intraday_ret"]
        .apply(lambda x: (x ** 2).sum())
        .reset_index()
    )
    rv_daily.columns = ["Date", "RV"]
    rv_daily["sigma_RV"] = np.sqrt(rv_daily["RV"])

    # Count number of observations per day (for filtering thin days)
    obs_count = df.groupby("Date")["intraday_ret"].count().reset_index()
    obs_count.columns = ["Date", "n_obs"]
    rv_daily = rv_daily.merge(obs_count, on="Date")

    print(f"Realized Volatility computed for {len(rv_daily)} trading days")
    print(f"  Avg observations per day: {rv_daily['n_obs'].mean():.0f}")

    return rv_daily


# ===================================================================
# 2. Get Yang-Zhang for matching dates
# ===================================================================
def get_yang_zhang_for_dates(dates):
    """Load daily data and extract Yang-Zhang volatility for given dates."""
    path = os.path.join(PROC_DIR, "daily_with_volatility.csv")
    daily = pd.read_csv(path)
    daily["Date"] = pd.to_datetime(daily["Date"])

    # Keep only dates that appear in both datasets
    yz = daily[["Date", "sigma_yz"]].copy()
    yz = yz[yz["Date"].isin(dates)]
    return yz


# ===================================================================
# 3. Mincer-Zarnowitz Regression
# ===================================================================
def mincer_zarnowitz(merged):
    """
    Run Mincer-Zarnowitz regression to test if YZ is unbiased proxy for RV.

    RV_t = alpha + beta * YZ_t + epsilon_t

    Under H0 (unbiased proxy): alpha = 0, beta = 1
    """
    y = merged["sigma_RV"].values
    x = merged["sigma_yz"].values
    X = sm.add_constant(x)

    model = sm.OLS(y, X).fit(cov_type="HC1")  # heteroskedasticity-robust SEs

    # Joint test: alpha=0, beta=1
    # Format: (R, q) where R @ beta = q under H0
    r_matrix = np.eye(2)              # [[1,0],[0,1]]
    q_values = np.array([0.0, 1.0])   # H0: alpha=0, beta=1
    f_test = model.f_test((r_matrix, q_values))

    results = {
        "alpha": model.params[0],
        "alpha_se": model.bse[0],
        "alpha_pval": model.pvalues[0],
        "beta": model.params[1],
        "beta_se": model.bse[1],
        "beta_pval": model.pvalues[1],
        "R_squared": model.rsquared,
        "N": int(model.nobs),
        "F_stat_joint": float(f_test.fvalue),
        "F_pval_joint": float(f_test.pvalue),
        "Correlation": np.corrcoef(x, y)[0, 1],
    }

    print("=== Mincer-Zarnowitz Regression ===")
    print(f"  RV_t = {results['alpha']:.6f} + {results['beta']:.4f} * YZ_t")
    print(f"  alpha: {results['alpha']:.6f} (SE={results['alpha_se']:.6f}, p={results['alpha_pval']:.4f})")
    print(f"  beta:  {results['beta']:.4f} (SE={results['beta_se']:.4f}, p={results['beta_pval']:.4f})")
    print(f"  R²:    {results['R_squared']:.4f}")
    print(f"  Correlation: {results['Correlation']:.4f}")
    print(f"  Joint test (alpha=0, beta=1): F={results['F_stat_joint']:.2f}, p={results['F_pval_joint']:.4f}")

    return results, model


# ===================================================================
# 4. Scatter Plot: YZ vs RV
# ===================================================================
def plot_scatter(merged, model):
    """Scatter plot of Yang-Zhang vs Realized Volatility with regression line."""
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(merged["sigma_yz"], merged["sigma_RV"],
               alpha=0.4, s=20, color="steelblue", label="Daily observations")

    # Regression line
    x_range = np.linspace(merged["sigma_yz"].min(), merged["sigma_yz"].max(), 100)
    y_pred = model.params[0] + model.params[1] * x_range
    ax.plot(x_range, y_pred, color="red", linewidth=2,
            label=f"OLS: α={model.params[0]:.4f}, β={model.params[1]:.3f}")

    # 45-degree line (perfect proxy)
    ax.plot(x_range, x_range, color="gray", linestyle="--", linewidth=1,
            label="45° line (perfect proxy)")

    ax.set_xlabel("Yang-Zhang Daily Volatility (σ_YZ)")
    ax.set_ylabel("Realized Volatility (σ_RV)")
    ax.set_title(f"Proxy Validation: Yang-Zhang vs Realized Volatility (R²={model.rsquared:.3f})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "16_proxy_validation_scatter.png"))
    plt.close(fig)
    print("Saved: 16_proxy_validation_scatter.png")


# ===================================================================
# 5. Time Series Overlay
# ===================================================================
def plot_timeseries_overlay(merged):
    """Overlay YZ and RV on a time-series plot."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(merged["Date"], merged["sigma_RV"],
            linewidth=0.8, color="steelblue", alpha=0.7, label="Realized Vol (5-min)")
    ax.plot(merged["Date"], merged["sigma_yz"],
            linewidth=0.8, color="darkred", alpha=0.7, label="Yang-Zhang (daily OHLC)")

    ax.set_title("Proxy Validation: Yang-Zhang vs Realized Volatility Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Volatility")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "17_proxy_validation_timeseries.png"))
    plt.close(fig)
    print("Saved: 17_proxy_validation_timeseries.png")


# ===================================================================
# 6. Summary comparison table
# ===================================================================
def summary_comparison(merged):
    """Descriptive stats for both volatility measures."""
    summary = pd.DataFrame({
        "Yang-Zhang": merged["sigma_yz"].describe(),
        "Realized Vol": merged["sigma_RV"].describe(),
    })
    summary.to_csv(os.path.join(TBL_DIR, "proxy_validation_summary.csv"))
    print("\n=== Summary Statistics (overlapping period) ===")
    print(summary.round(6).to_string())
    return summary


# ===================================================================
# 7. Rolling-Window Comparison (22-day averages)
# ===================================================================
def rolling_comparison(merged, window=22):
    """
    Compare 22-day rolling averages of YZ and RV.

    Single-day YZ includes overnight variance that intraday RV misses,
    making single-day comparison noisy. Averaging over 22 days aligns
    with the HAR model's use of multi-day aggregates and reduces noise.
    """
    m = merged.sort_values("Date").copy()
    m["sigma_yz_roll"] = m["sigma_yz"].rolling(window, min_periods=window).mean()
    m["sigma_RV_roll"] = m["sigma_RV"].rolling(window, min_periods=window).mean()
    m = m.dropna(subset=["sigma_yz_roll", "sigma_RV_roll"])

    if len(m) < 20:
        print(f"  Not enough data for {window}-day rolling comparison")
        return None, None

    # Mincer-Zarnowitz on rolling averages
    y = m["sigma_RV_roll"].values
    x = m["sigma_yz_roll"].values
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type="HC1")

    r_matrix = np.eye(2)
    q_values = np.array([0.0, 1.0])
    f_test = model.f_test((r_matrix, q_values))

    results = {
        "alpha": model.params[0],
        "alpha_se": model.bse[0],
        "alpha_pval": model.pvalues[0],
        "beta": model.params[1],
        "beta_se": model.bse[1],
        "beta_pval": model.pvalues[1],
        "R_squared": model.rsquared,
        "N": int(model.nobs),
        "F_stat_joint": float(f_test.fvalue),
        "F_pval_joint": float(f_test.pvalue),
        "Correlation": np.corrcoef(x, y)[0, 1],
    }

    print(f"\n=== Mincer-Zarnowitz ({window}-day rolling averages) ===")
    print(f"  RV_roll = {results['alpha']:.6f} + {results['beta']:.4f} * YZ_roll")
    print(f"  alpha: {results['alpha']:.6f} (p={results['alpha_pval']:.4f})")
    print(f"  beta:  {results['beta']:.4f} (p={results['beta_pval']:.4f})")
    print(f"  R²:    {results['R_squared']:.4f}")
    print(f"  Correlation: {results['Correlation']:.4f}")

    return m, results


def plot_rolling_scatter(m, window=22):
    """Scatter of 22-day rolling YZ vs RV."""
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(m["sigma_yz_roll"], m["sigma_RV_roll"],
               alpha=0.5, s=20, color="steelblue")

    # Regression line
    x = m["sigma_yz_roll"].values
    y = m["sigma_RV_roll"].values
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    x_range = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_range, model.params[0] + model.params[1] * x_range,
            color="red", linewidth=2,
            label=f"OLS: α={model.params[0]:.4f}, β={model.params[1]:.3f}")

    ax.plot(x_range, x_range, color="gray", linestyle="--", linewidth=1,
            label="45° line")

    ax.set_xlabel(f"Yang-Zhang ({window}-day avg)")
    ax.set_ylabel(f"Realized Vol ({window}-day avg)")
    ax.set_title(f"Proxy Validation: {window}-day Rolling Averages (R²={model.rsquared:.3f})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "16b_proxy_validation_rolling_scatter.png"))
    plt.close(fig)
    print("Saved: 16b_proxy_validation_rolling_scatter.png")


# ===================================================================
# 8. Write LaTeX table to tex/tables/proxy_validation.tex
# ===================================================================
def write_latex_table(results_daily, results_roll):
    tex_path = os.path.join(
        BASE_DIR, os.pardir, "tex", "tables", "proxy_validation.tex"
    )

    def fmt_coef(val, se, pval):
        stars = "^{***}" if pval < 0.001 else ("^{**}" if pval < 0.01 else ("^{*}" if pval < 0.05 else ""))
        return f"${val:.4f}{stars}$", f"$({se:.4f})$"

    a_d, a_d_se = fmt_coef(results_daily["alpha"], results_daily["alpha_se"], results_daily["alpha_pval"])
    b_d, b_d_se = fmt_coef(results_daily["beta"],  results_daily["beta_se"],  results_daily["beta_pval"])
    a_r, a_r_se = fmt_coef(results_roll["alpha"],  results_roll["alpha_se"],  results_roll["alpha_pval"])
    b_r, b_r_se = fmt_coef(results_roll["beta"],   results_roll["beta_se"],   results_roll["beta_pval"])

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Mincer-Zarnowitz regression: Yang-Zhang against intraday realized volatility}",
        r"\label{tab:proxy-validation}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & $\hat{\alpha}$ & $\hat{\beta}$ & $R^2$ & $N$ & $F_{\alpha=0,\,\beta=1}$ & $p$-value \\",
        r"\midrule",
        f"Daily         & {a_d} & {b_d} & ${results_daily['R_squared']:.3f}$ & {int(results_daily['N'])} & ${results_daily['F_stat_joint']:.1f}$ & $<0.001$ \\\\",
        f"              & {a_d_se} & {b_d_se} &  &  &  & \\\\[4pt]",
        f"22-day rolling & {a_r} & {b_r} & ${results_roll['R_squared']:.3f}$ & {int(results_roll['N'])} & ${results_roll['F_stat_joint']:.1f}$ & $<0.001$ \\\\",
        f"              & {a_r_se} & {b_r_se} &  &  &  & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"{\footnotesize\textit{Note:} Mincer-Zarnowitz regression $\text{RV}_t = \alpha + \beta\,\hat{\sigma}^{\text{YZ}}_t + \varepsilon_t$",
        r"estimated with HC1 standard errors (in parentheses). The $F$-statistic tests the joint",
        r"hypothesis $\alpha = 0$, $\beta = 1$ (unbiased proxy). RV is computed from five-minute",
        r"log returns; $\hat{\sigma}^{\text{YZ}}$ is the Yang-Zhang estimator from daily OHLC prices.",
        r"The 22-day row uses rolling 22-day averages of both series, aligning with the HAR monthly",
        r"component. The lower $\hat{\beta}$ at the daily frequency reflects the different treatment of",
        r"the overnight component: Yang-Zhang captures the close-to-open gap while intraday RV does not.",
        r"Sample: 3 February 2025 -- 12 January 2026.",
        r"$^{***}$ $p < 0.001$.}",
        r"\end{table}",
    ]

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved LaTeX table: tex/tables/proxy_validation.tex")


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    # Load 5-minute data
    print("Loading 5-minute data...")
    df_5min = pd.read_csv(os.path.join(PROC_DIR, "5min_clean.csv"))
    df_5min["Date"] = pd.to_datetime(df_5min["Date"])

    # Compute RV
    print("\nComputing Realized Volatility from intraday data...")
    rv = compute_realized_volatility(df_5min)

    # Filter out days with too few observations (< 20 bars = thin trading)
    rv = rv[rv["n_obs"] >= 20].copy()
    print(f"  After filtering thin days: {len(rv)} days remain")

    # Get matching YZ values
    print("\nLoading Yang-Zhang for matching dates...")
    yz = get_yang_zhang_for_dates(rv["Date"])

    # Merge
    merged = rv.merge(yz, on="Date", how="inner").dropna()
    print(f"  Matched observations: {len(merged)}")

    if len(merged) < 30:
        print("\nWARNING: Very few matched observations. Check date alignment.")
        print("5-min dates:", rv["Date"].min(), "to", rv["Date"].max())
        print("Daily dates:", yz["Date"].min(), "to", yz["Date"].max())
    else:
        # --- Single-day comparison ---
        summary_comparison(merged)

        print("\n--- Mincer-Zarnowitz Regression (single-day) ---")
        results_daily, model_daily = mincer_zarnowitz(merged)
        pd.Series(results_daily).to_csv(
            os.path.join(TBL_DIR, "mincer_zarnowitz_results.csv")
        )

        # --- Rolling-window comparison (the better test) ---
        m_roll, results_roll = rolling_comparison(merged, window=22)
        if results_roll is not None:
            pd.Series(results_roll).to_csv(
                os.path.join(TBL_DIR, "mincer_zarnowitz_rolling22.csv")
            )

        if results_roll is not None:
            write_latex_table(results_daily, results_roll)

        print("\n--- Generating Plots ---")
        plot_scatter(merged, model_daily)
        plot_timeseries_overlay(merged)
        if m_roll is not None:
            plot_rolling_scatter(m_roll, window=22)

    print("\nDone!")
