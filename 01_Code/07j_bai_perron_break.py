"""
07j_bai_perron_break.py
=======================
Sup-Wald (QLR) structural break test on the out-of-sample Yang-Zhang
volatility series, following Andrews (1993).

The algorithm fits a mean-shift regression at each candidate break date
and records the F-statistic; the global maximum is the Andrews sup-Wald
estimator.  A preliminary inspection of the series shows that the cocoa
price crisis manifests as a volatility jump in 2024 (not 2022 as assumed
from press knowledge).  The candidate window is therefore restricted to
2022-01-01 to 2025-01-01, which keeps both sub-segments large enough
(>= 300 obs post-break) and focuses the test on the plausible onset range.

Critical values from Andrews (1993), Table 1, k = 1 (mean-shift):
  1%: 10.17    5%: 7.17    10%: 5.74

Output:
  python/02_Output/tables/bai_perron_break.csv
"""

import pandas as pd
import numpy as np
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "00_Data", "processed", "har_dataset.csv")
TBL_DIR   = os.path.join(BASE_DIR, "02_Output", "tables")
os.makedirs(TBL_DIR, exist_ok=True)

ROLLING_WINDOW = 2520

ANDREWS_CV = {0.01: 10.17, 0.05: 7.17, 0.10: 5.74}


def sup_wald_range(y, dates, date_lo, date_hi):
    """
    Find the sup-Wald break in the candidate window [date_lo, date_hi).
    Returns (best_index_into_y, best_date, best_f_stat).
    """
    T = len(y)
    rss_r = np.sum((y - y.mean()) ** 2)
    mask = (dates >= date_lo) & (dates < date_hi)
    candidates = np.where(mask)[0]

    best_stat, best_idx = -np.inf, None
    for k in candidates:
        if k < 2 or k > T - 2:
            continue
        seg1, seg2 = y[:k], y[k:]
        rss_u = (np.sum((seg1 - seg1.mean()) ** 2) +
                 np.sum((seg2 - seg2.mean()) ** 2))
        f = ((rss_r - rss_u) / 1) / (rss_u / (T - 2))
        if f > best_stat:
            best_stat, best_idx = f, k

    return best_idx, dates.iloc[best_idx], best_stat


# ---------------------------------------------------------------------------
# Load out-of-sample series
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

oos   = df.iloc[ROLLING_WINDOW:][["Date", "sigma_yz"]].reset_index(drop=True)
y     = oos["sigma_yz"].values
dates = oos["Date"]
T     = len(oos)

print("=" * 65)
print("Sup-Wald Structural Break Test (Andrews 1993)")
print(f"Series   : sigma_yz (Yang-Zhang volatility)")
print(f"OOS range: {dates.iloc[0].date()} – {dates.iloc[-1].date()}"
      f"  (T = {T:,})")
print("=" * 65)

# Preliminary: print quarterly mean vol to motivate the candidate window
print("\nQuarterly mean sigma_yz (recent period):")
tmp = oos[oos["Date"] >= "2020-01-01"].copy()
tmp["Q"] = tmp["Date"].dt.to_period("Q")
for q, v in tmp.groupby("Q")["sigma_yz"].mean().items():
    print(f"  {q}: {v:.6f}")

# ---------------------------------------------------------------------------
# Primary: focused candidate window 2022-01-01 to 2025-01-01
# ---------------------------------------------------------------------------
CAND_LO = pd.Timestamp("2022-01-01")
CAND_HI = pd.Timestamp("2025-01-01")

best_idx, best_date, best_f = sup_wald_range(y, dates, CAND_LO, CAND_HI)

n_pre  = best_idx
n_post = T - best_idx
mu_pre  = y[:n_pre].mean()
mu_post = y[n_pre:].mean()

print(f"\n{'='*65}")
print(f"Focused candidate window: {CAND_LO.date()} – {CAND_HI.date()}")
print(f"Estimated break date    : {best_date.date()}")
print(f"Sup-Wald F-statistic    : {best_f:.4f}")
print(f"{'='*65}")

print("\nAndrews (1993) critical values (k = 1):")
for sig, cv in ANDREWS_CV.items():
    verdict = "REJECT" if best_f > cv else "fail to reject"
    print(f"  {int(sig*100):2d}%  CV = {cv:5.2f}  ->  {verdict} H0 (no break)")

print(f"\nPre-break segment : {n_pre:,} obs "
      f"({dates.iloc[0].date()} – {dates.iloc[n_pre-1].date()})")
print(f"Post-break segment: {n_post:,} obs "
      f"({best_date.date()} – {dates.iloc[-1].date()})")
print(f"\nMean sigma_yz  pre-break : {mu_pre:.6f}")
print(f"Mean sigma_yz post-break : {mu_post:.6f}")
print(f"Ratio (post / pre)       : {mu_post / mu_pre:.3f}x")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
results = pd.DataFrame([{
    "break_date":       best_date.strftime("%Y-%m-%d"),
    "sup_wald_stat":    round(best_f, 4),
    "cv_10pct":         ANDREWS_CV[0.10],
    "cv_05pct":         ANDREWS_CV[0.05],
    "cv_01pct":         ANDREWS_CV[0.01],
    "n_pre":            n_pre,
    "n_post":           n_post,
    "mu_pre":           round(mu_pre,  6),
    "mu_post":          round(mu_post, 6),
    "cand_window_lo":   CAND_LO.strftime("%Y-%m-%d"),
    "cand_window_hi":   CAND_HI.strftime("%Y-%m-%d"),
    "rolling_window":   ROLLING_WINDOW,
    "T_oos":            T,
}])

out_path = os.path.join(TBL_DIR, "bai_perron_break.csv")
results.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print("\nDone.")
