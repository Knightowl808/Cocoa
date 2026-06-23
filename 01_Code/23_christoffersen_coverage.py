"""
23_christoffersen_coverage.py
==============================
Christoffersen (1998) conditional coverage tests for HAR-QR and HAR-X-QR
at tau = 0.95 across all six horizons.

Tests computed:
  LR_uc  -- unconditional coverage (chi^2, df=1)
  LR_ind -- independence of violations (chi^2, df=1)
  LR_cc  -- conditional coverage = LR_uc + LR_ind (chi^2, df=2)

Overlap handling: at horizons h > 1 the daily forecasts target OVERLAPPING
h-day average windows, so consecutive violations are serially dependent by
construction and the iid assumption behind the LR tests fails. The empirical
violation rate (VR) is therefore reported on the full daily sequence
(descriptive), while all LR tests are computed on the NON-OVERLAPPING
subsequence of hits (every h-th forecast). At long horizons this leaves few
observations; cells where a test is not computable are shown as "---".

Output: tables/christoffersen_coverage.tex
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
TBL_DIR  = os.path.join(BASE_DIR, "02_Output", "tables")
os.makedirs(TBL_DIR, exist_ok=True)

HORIZONS = ["1d", "1w", "1m", "3m", "6m", "12m"]
H_DAYS   = {"1d": 1, "1w": 5, "1m": 22, "3m": 66, "6m": 132, "12m": 264}
TAU = 0.95
MODELS = {
    "HAR-QR":   "har_qr",
    "HAR-X-QR": "har_x_qr",
}


def christoffersen_tests(hits, step=1):
    """
    Given a binary hit sequence (1 = violation), return dict with:
      n, n_viol, vr, LR_uc, p_uc, LR_ind, p_ind, LR_cc, p_cc

    The violation rate (VR) is computed on the FULL daily sequence. The LR
    tests are computed on the non-overlapping subsequence hits[::step]
    (step = h) because overlapping h-day targets make consecutive daily
    hits dependent by construction, invalidating the iid likelihood.
    """
    hits_full = np.asarray(hits, dtype=float)
    T_full = len(hits_full)
    vr_full = hits_full.sum() / T_full if T_full > 0 else np.nan

    # Non-overlapping subsequence used for all LR tests
    hits = hits_full[::step]
    T = len(hits)
    n_viol = int(hits.sum())
    pi = n_viol / T if T > 0 else np.nan
    alpha = 1.0 - TAU  # nominal = 0.05

    # --- Unconditional coverage ---
    if 0 < pi < 1:
        lr_uc = 2 * (n_viol * np.log(pi / alpha) +
                     (T - n_viol) * np.log((1 - pi) / (1 - alpha)))
    else:
        lr_uc = np.nan
    p_uc = 1 - stats.chi2.cdf(lr_uc, df=1) if not np.isnan(lr_uc) else np.nan

    # --- Independence (Markov transition) ---
    n00 = int(((hits[:-1] == 0) & (hits[1:] == 0)).sum())
    n01 = int(((hits[:-1] == 0) & (hits[1:] == 1)).sum())
    n10 = int(((hits[:-1] == 1) & (hits[1:] == 0)).sum())
    n11 = int(((hits[:-1] == 1) & (hits[1:] == 1)).sum())

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else np.nan
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else np.nan

    if (0 < pi0 < 1) and (0 < pi1 < 1) and 0 < pi < 1:
        ll_ind = (n00 * np.log(1 - pi0) + n01 * np.log(pi0) +
                  n10 * np.log(1 - pi1) + n11 * np.log(pi1))
        ll_null = ((n00 + n10) * np.log(1 - pi) +
                   (n01 + n11) * np.log(pi))
        lr_ind = 2 * (ll_ind - ll_null)
    else:
        lr_ind = np.nan
    p_ind = 1 - stats.chi2.cdf(lr_ind, df=1) if not np.isnan(lr_ind) else np.nan

    lr_cc = lr_uc + lr_ind if not (np.isnan(lr_uc) or np.isnan(lr_ind)) else np.nan
    p_cc  = 1 - stats.chi2.cdf(lr_cc, df=2) if not np.isnan(lr_cc) else np.nan

    return {
        "N": T_full, "N_no": T, "n_viol": n_viol,
        "VR (%)": round(100 * vr_full, 2),
        "VR_no (%)": round(100 * pi, 2) if T > 0 else np.nan,
        "LR_uc":  round(lr_uc,  3) if not np.isnan(lr_uc)  else "---",
        "p_uc":   round(p_uc,   4) if not np.isnan(p_uc)   else "---",
        "LR_ind": round(lr_ind, 3) if not np.isnan(lr_ind) else "---",
        "p_ind":  round(p_ind,  4) if not np.isnan(p_ind)  else "---",
        "LR_cc":  round(lr_cc,  3) if not np.isnan(lr_cc)  else "---",
        "p_cc":   round(p_cc,   4) if not np.isnan(p_cc)   else "---",
        "pi0": round(pi0, 4) if (pi0 is not None and not np.isnan(pi0)) else "---",
        "pi1": round(pi1, 4) if (pi1 is not None and not np.isnan(pi1)) else "---",
    }


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
rows = []
for h in HORIZONS:
    for label, tag in MODELS.items():
        path = os.path.join(PROC_DIR, f"{tag}_forecasts_{h}.csv")
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        if "q95" not in df.columns:
            print(f"  No q95 in {path}")
            continue
        hits = (df["actual"] > df["q95"]).astype(int)
        res = christoffersen_tests(hits.values, step=H_DAYS[h])
        rows.append({"Horizon": h, "Model": label, **res})
        print(f"{label:12s}  {h:4s}  VR={res['VR (%)']:5.2f}%  N_no={res['N_no']}  "
              f"LR_uc={res['LR_uc']}(p={res['p_uc']})  "
              f"LR_ind={res['LR_ind']}(p={res['p_ind']})  "
              f"LR_cc={res['LR_cc']}(p={res['p_cc']})")

df_out = pd.DataFrame(rows)
df_out.to_csv(os.path.join(TBL_DIR, "christoffersen_coverage.csv"), index=False)

# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
HORIZON_LABELS = {
    "1d": "1 day", "1w": "1 week", "1m": "1 month",
    "3m": "3 months", "6m": "6 months", "12m": "12 months"
}

def fmt_stat(val, pval):
    """Format test statistic with significance stars."""
    if val == "---":
        return "---"
    stars = ""
    if pval != "---":
        if pval < 0.01:
            stars = "^{***}"
        elif pval < 0.05:
            stars = "^{**}"
        elif pval < 0.10:
            stars = "^{*}"
    return f"${val}{stars}$"

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\caption{Christoffersen (1998) conditional coverage tests, $\tau = 0.95$}")
lines.append(r"\label{tab:christoffersen}")
lines.append(r"\begin{tabular}{llrrrccc}")
lines.append(r"\toprule")
lines.append(r"Horizon & Model & $N$ & $N_{no}$ & VR\,(\%) & $LR_{uc}$ & $LR_{ind}$ & $LR_{cc}$ \\")
lines.append(r"\midrule")

prev_h = None
for _, row in df_out.iterrows():
    h_lab = HORIZON_LABELS[row["Horizon"]]
    if row["Horizon"] != prev_h and prev_h is not None:
        lines.append(r"\addlinespace")
    prev_h = row["Horizon"]
    lr_uc  = fmt_stat(row["LR_uc"],  row["p_uc"])
    lr_ind = fmt_stat(row["LR_ind"], row["p_ind"])
    lr_cc  = fmt_stat(row["LR_cc"],  row["p_cc"])
    lines.append(
        f"{h_lab} & {row['Model']} & {row['N']} & {row['N_no']} & {row['VR (%)']:.2f} "
        f"& {lr_uc} & {lr_ind} & {lr_cc} \\\\"
    )

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\par\smallskip")
lines.append(
    r"{\footnotesize "
    r"$LR_{uc}$: unconditional coverage test ($\chi^2_1$); "
    r"$LR_{ind}$: independence of violations ($\chi^2_1$); "
    r"$LR_{cc}$: conditional coverage ($\chi^2_2$). "
    r"VR: empirical violation rate ($\%$) over all $N$ daily forecasts; "
    r"nominal target $= 5\%$. Because daily forecasts at $h>1$ target "
    r"overlapping $h$-day windows, consecutive violations are dependent by "
    r"construction; the LR tests are therefore computed on the "
    r"non-overlapping subsequence of every $h$-th forecast ($N_{no}$ "
    r"observations). Cells marked --- are not computable (no violations or "
    r"too few observations in the non-overlapping subsequence). "
    r"$^{*}p<0.10$; $^{**}p<0.05$; $^{***}p<0.01$.}"
)
lines.append(r"\end{table}")

tex_path = os.path.join(TBL_DIR, "christoffersen_coverage.tex")
with open(tex_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nWrote {tex_path}")
print(f"Wrote {os.path.join(TBL_DIR, 'christoffersen_coverage.csv')}")
