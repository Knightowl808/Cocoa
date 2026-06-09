import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
TBL_DIR  = os.path.join(BASE_DIR, "02_Output", "tables")

HORIZONS  = ["1d", "1w", "1m", "3m", "6m", "12m"]
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

def quantile_loss(actual, forecast, tau):
    u = actual - forecast
    return np.mean(u * (tau - (u < 0).astype(float)))

rows = []
for h in HORIZONS:
    path = os.path.join(PROC_DIR, f"har_ols_forecasts_{h}.csv")
    df = pd.read_csv(path)
    for tau in QUANTILES:
        col = f"q{int(tau*100):02d}"
        ql = quantile_loss(df["actual"].values, df[col].values, tau)
        rows.append({"Horizon": h, "Quantile": tau, "QL": ql})

result = pd.DataFrame(rows)
result.to_csv(os.path.join(TBL_DIR, "har_ols_quantile_loss.csv"), index=False)
print("Saved: har_ols_quantile_loss.csv")

print("\nHAR-OLS pseudo-quantile QL at tau=0.95:")
print(result[result["Quantile"] == 0.95].to_string(index=False))

print("\nFull pivot (rows=tau, cols=horizon):")
pivot = result.pivot(index="Quantile", columns="Horizon", values="QL")[HORIZONS]
print(pivot.round(7).to_string())
