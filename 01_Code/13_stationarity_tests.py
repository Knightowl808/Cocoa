"""
13_stationarity_tests.py
Run ADF and Phillips-Perron stationarity tests on:
  - Daily log returns
  - Yang-Zhang daily volatility (sigma_yz)
  - Nino 3.4 ENSO anomaly index
Output: CSV table + LaTeX-ready table for the Data section.
"""

import os
import sys
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
OUT_DIR = os.path.join(BASE_DIR, "02_Output", "tables")


def phillips_perron(series, regression="c", nlags=None):
    """
    Phillips-Perron unit root test (Z(t) statistic).

    Uses ADF(0) as the base regression, then applies the PP correction
    for serial correlation using a Newey-West long-run variance estimate.
    """
    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)

    # Build regression: Dy_t = alpha + rho * y_{t-1} + e_t
    dy = np.diff(y)
    y_lag = y[:-1]

    if regression == "c":
        X = np.column_stack([np.ones(n - 1), y_lag])
    elif regression == "ct":
        X = np.column_stack([np.ones(n - 1), np.arange(1, n), y_lag])
    else:  # "n"
        X = y_lag.reshape(-1, 1)

    # OLS
    beta = np.linalg.lstsq(X, dy, rcond=None)[0]
    resid = dy - X @ beta
    rho_hat = beta[-1]

    # Short-run variance
    s2 = np.sum(resid**2) / (n - 1 - X.shape[1])

    # Long-run variance (Newey-West)
    if nlags is None:
        nlags = int(np.floor(4 * (n / 100) ** (2/9)))

    gamma0 = np.sum(resid**2) / (n - 1)
    lrv = gamma0
    for j in range(1, nlags + 1):
        w = 1 - j / (nlags + 1)  # Bartlett kernel
        gamma_j = np.sum(resid[j:] * resid[:-j]) / (n - 1)
        lrv += 2 * w * gamma_j

    # PP correction
    se_rho = np.sqrt(s2 * np.linalg.inv(X.T @ X)[-1, -1])
    t_adf = rho_hat / se_rho

    # Z(t) statistic
    correction = (lrv - s2) / (2 * lrv)
    z_t = np.sqrt(s2 / lrv) * t_adf - correction * np.sqrt(lrv) * (n - 1) * se_rho / np.sqrt(s2)

    # Use ADF critical values (same asymptotic distribution)
    from statsmodels.tsa.stattools import adfuller
    # Get critical values from a dummy ADF call
    dummy_result = adfuller(y, maxlag=0, regression=regression, autolag=None)
    crit_values = dummy_result[4]

    # Approximate p-value using MacKinnon via ADF infrastructure
    from statsmodels.tsa.adfvalues import mackinnonp
    nobs_eff = n - 1
    if regression == "c":
        p_value = mackinnonp(z_t, regression="c", N=1)
    elif regression == "ct":
        p_value = mackinnonp(z_t, regression="ct", N=1)
    else:
        p_value = mackinnonp(z_t, regression="nc", N=1)

    return z_t, p_value, nlags, crit_values


def run_tests():
    # Load data
    vol_df = pd.read_csv(os.path.join(PROC_DIR, "daily_with_volatility.csv"))
    vol_df["Date"] = pd.to_datetime(vol_df["Date"])

    nino_df = pd.read_csv(os.path.join(PROC_DIR, "nino34_clean.csv"))
    nino_df["Date"] = pd.to_datetime(nino_df["Date"])

    # Series to test
    log_returns = vol_df["log_ret"].dropna().values
    sigma_yz = vol_df["sigma_yz"].dropna().values
    enso = nino_df["nino34"].dropna().values

    series_dict = {
        "Log returns": log_returns,
        "Yang-Zhang volatility": sigma_yz,
        "Niño 3.4 anomaly": enso,
    }

    results = []

    for name, data in series_dict.items():
        n_obs = len(data)

        # ADF test (constant, automatic lag selection via AIC)
        adf_result = adfuller(data, regression="c", autolag="AIC")
        adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, adf_ic = adf_result

        # Phillips-Perron test (constant)
        pp_stat, pp_p, pp_nlags, pp_crit = phillips_perron(
            data, regression="c"
        )

        results.append({
            "Series": name,
            "N": n_obs,
            "ADF stat": adf_stat,
            "ADF p-value": adf_p,
            "ADF lags": adf_lags,
            "ADF 1%": adf_crit["1%"],
            "ADF 5%": adf_crit["5%"],
            "PP stat": pp_stat,
            "PP p-value": pp_p,
            "PP bandwidth": pp_nlags,
        })

        print(f"\n{'='*60}")
        print(f"  {name}  (N = {n_obs})")
        print(f"{'='*60}")
        print(f"  ADF:  stat = {adf_stat:.4f},  p = {adf_p:.4f},  lags = {adf_lags}")
        print(f"        1% cv = {adf_crit['1%']:.3f},  5% cv = {adf_crit['5%']:.3f}")
        print(f"  PP:   stat = {pp_stat:.4f},  p = {pp_p:.4f},  bandwidth = {pp_nlags}")
        print(f"        1% cv = {pp_crit['1%']:.3f},  5% cv = {pp_crit['5%']:.3f}")

    # Save results
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "stationarity_tests.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Generate LaTeX table
    generate_latex_table(results)

    return df_results


def generate_latex_table(results):
    """Print a LaTeX-ready table for the Data section."""
    tex_path = os.path.join(
        BASE_DIR, os.pardir, "tex", "tables", "stationarity_tests.tex"
    )

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Unit root tests for stationarity}")
    lines.append(r"\label{tab:stationarity}")
    lines.append(r"\begin{tabular}{lrccrcc}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{2}{c}{ADF} & & \multicolumn{2}{c}{PP} \\")
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}")
    lines.append(r"Series & $N$ & Statistic & $p$-value & & Statistic & $p$-value \\")
    lines.append(r"\midrule")

    for r in results:
        p_adf = f"$<0.001$" if r["ADF p-value"] < 0.001 else f"${r['ADF p-value']:.3f}$"
        p_pp = f"$<0.001$" if r["PP p-value"] < 0.001 else f"${r['PP p-value']:.3f}$"

        lines.append(
            f"{r['Series']} & {r['N']:,} & ${r['ADF stat']:.2f}$ & {p_adf} "
            f"& & ${r['PP stat']:.2f}$ & {p_pp} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\par\smallskip")
    lines.append(r"{\footnotesize\textit{Note:} ADF = Augmented Dickey--Fuller test")
    lines.append(r"\parencite{dickey1979distribution}; PP = Phillips--Perron test")
    lines.append(r"\parencite{phillips1988testing}. Both tests include a constant.")
    lines.append(r"ADF lag length selected by AIC. The null hypothesis is a unit root;")
    lines.append(r"rejection indicates stationarity.}")
    lines.append(r"\end{table}")

    tex_content = "\n".join(lines)

    os.makedirs(os.path.dirname(tex_path), exist_ok=True)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    print(f"\nLaTeX table saved: {tex_path}")
    print("\n" + tex_content)


if __name__ == "__main__":
    run_tests()