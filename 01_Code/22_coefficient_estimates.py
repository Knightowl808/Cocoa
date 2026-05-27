"""
22_coefficient_estimates.py
===========================
Full-sample HAR-QR and HAR-X-QR coefficient estimates for the appendix.

The forecasting pipeline (07_*, 07b_*) estimates the models on rolling
windows, so coefficients vary over time. For a static reference table we
re-estimate each model once on the FULL sample, separately per horizon and
quantile. This gives the reader a sense of the sign and magnitude of the
HAR loadings and the ENSO term.

Quantiles: tau in {0.50, 0.75, 0.95}  (the set reported in the thesis)
Horizons:  h in {1, 5, 22, 66, 132, 264}

Output: tex/tables/har_qr_coefficients.tex
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Maximum number of iterations")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
TEX_TBL_DIR = os.path.join(os.path.dirname(BASE_DIR), "tex", "tables")

HORIZONS = {1: "1d", 5: "1w", 22: "1m", 66: "3m", 132: "6m", 264: "12m"}
HORIZON_LABEL = {"1d": "1 day", "1w": "1 week", "1m": "1 month",
                 "3m": "3 months", "6m": "6 months", "12m": "12 months"}
QUANTILES = [0.50, 0.75, 0.95]


def load_data():
    df = pd.read_csv(os.path.join(PROC_DIR, "har_dataset.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def estimate(df, feature_cols, horizon_label, tau):
    """Full-sample quantile regression. Returns (params, pvalues, n)."""
    target_col = f"target_{horizon_label}"
    cols = ["Date"] + feature_cols + [target_col]
    valid = df[cols].dropna().reset_index(drop=True)
    X = sm.add_constant(valid[feature_cols].values)
    y = valid[target_col].values
    res = QuantReg(y, X).fit(q=tau, max_iter=2000)
    return res.params, res.pvalues, len(valid)


def fmt(coef, p):
    # Point estimates only. Significance stars are intentionally omitted:
    # the targets are overlapping h-day-ahead averages, so naive QuantReg
    # standard errors are invalid and would overstate significance.
    return f"{coef:.4f}"


def build_table(df):
    """Build the LaTeX table: one panel per quantile, rows = horizons."""
    base_feats = ["sigma_d", "sigma_w", "sigma_m"]
    x_feats = ["sigma_d", "sigma_w", "sigma_m", "enso"]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Full-sample HAR-QR and HAR-X-QR coefficient "
                 r"estimates}")
    lines.append(r"\label{tab:har-qr-coefficients}")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Horizon & Model & $\beta_0$ & $\beta_d$ & $\beta_w$ "
                 r"& $\beta_m$ & $\beta_{\text{ENSO}}$ \\")

    for tau in QUANTILES:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{l}{\textit{Panel: } $\tau = "
                     f"{tau:.2f}$" + r"} \\")
        lines.append(r"\midrule")
        for h, label in HORIZONS.items():
            # HAR-QR
            p, pv, n = estimate(df, base_feats, label, tau)
            row = (f"{HORIZON_LABEL[label]} & HAR-QR & "
                   f"{fmt(p[0], pv[0])} & {fmt(p[1], pv[1])} & "
                   f"{fmt(p[2], pv[2])} & {fmt(p[3], pv[3])} & --- \\\\")
            lines.append(row)
            # HAR-X-QR
            p, pv, n = estimate(df, x_feats, label, tau)
            row = (f" & HAR-X-QR & "
                   f"{fmt(p[0], pv[0])} & {fmt(p[1], pv[1])} & "
                   f"{fmt(p[2], pv[2])} & {fmt(p[3], pv[3])} & "
                   f"{fmt(p[4], pv[4])} \\\\")
            lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\par\smallskip")
    lines.append(r"{\footnotesize\textit{Note:} Quantile regression "
                 r"coefficients estimated once on the full sample for each "
                 r"horizon and quantile $\tau$. $\beta_0$ is the intercept; "
                 r"$\beta_d$, $\beta_w$, $\beta_m$ are the daily, weekly and "
                 r"monthly HAR components; $\beta_{\text{ENSO}}$ is the "
                 r"Ni\~no~3.4 loading (HAR-X-QR only). The table is "
                 r"descriptive: it reports the sign and magnitude of the "
                 r"loadings only. Significance tests are omitted because the "
                 r"forecast targets are overlapping $h$-day-ahead averages, "
                 r"which violates the independence assumption underlying the "
                 r"standard errors. The forecasting results in "
                 r"Section~\ref{sec:results} use rolling-window "
                 r"re-estimation; these full-sample estimates are reported "
                 r"for reference only.}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print("Loading HAR dataset...")
    df = load_data()
    print("Estimating full-sample quantile regressions...")
    table = build_table(df)
    out_path = os.path.join(TEX_TBL_DIR, "har_qr_coefficients.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Saved: {out_path}")
