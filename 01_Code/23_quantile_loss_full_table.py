"""
23_quantile_loss_full_table.py
------------------------------
Emit the full five-model quantile-loss comparison table for the three
quantile levels used in the thesis (tau in {0.50, 0.75, 0.95}) across all
six horizons. The tau = 0.95 slice already appears in the Results body
(Table tab:upper-qlike-comparison); this table generalises that comparison
to the median and tau = 0.75 so RQ1 can be read across the distribution.

Source CSVs (python/02_Output/tables/):
    benchmark_quantile_loss.csv   -> Historical Vol, GARCH(1,1)
    har_ols_quantile_loss.csv     -> HAR-OLS (pseudo-quantile)
    har_qr_quantile_loss.csv      -> HAR-QR
    har_x_qr_quantile_loss.csv    -> HAR-X-QR

Output:
    tex/tables/quantile_loss_all.tex   (\\input-ed by Appendix_A.tex)
"""

import pathlib
import pandas as pd

ROOT   = pathlib.Path(__file__).resolve().parents[2]
TABLES = ROOT / "python" / "02_Output" / "tables"
OUT    = ROOT / "tex" / "tables" / "quantile_loss_all.tex"

HORIZONS  = ["1d", "1w", "1m", "3m", "6m", "12m"]
QUANTILES = [0.50, 0.75, 0.95]
MODELS    = ["Historical Vol", "GARCH(1,1)", "HAR-OLS", "HAR-QR", "HAR-X-QR"]

# ---------------------------------------------------------------------------
# Load all losses into a single (Model, Horizon, Quantile) -> QL lookup
# ---------------------------------------------------------------------------
def load_single(fname, model):
    df = pd.read_csv(TABLES / fname)
    df["Model"] = model
    return df[["Model", "Horizon", "Quantile", "QL"]]

frames = [
    pd.read_csv(TABLES / "benchmark_quantile_loss.csv")[
        ["Model", "Horizon", "Quantile", "QL"]
    ],
    load_single("har_ols_quantile_loss.csv",  "HAR-OLS"),
    load_single("har_qr_quantile_loss.csv",   "HAR-QR"),
    load_single("har_x_qr_quantile_loss.csv", "HAR-X-QR"),
]
data = pd.concat(frames, ignore_index=True)
# normalise GARCH label to the form used in the thesis
data["Model"] = data["Model"].str.replace(r"\s+", "", regex=True).where(
    data["Model"].str.startswith("GARCH"), data["Model"]
)
data.loc[data["Model"] == "GARCH(1,1)", "Model"] = "GARCH(1,1)"

lookup = {
    (r.Model, r.Horizon, round(r.Quantile, 2)): r.QL
    for r in data.itertuples(index=False)
}

# LaTeX label per model (escape the GARCH parentheses-free form is fine)
LABEL = {
    "Historical Vol": "Historical Vol",
    "GARCH(1,1)":     "GARCH(1,1)",
    "HAR-OLS":        "HAR-OLS",
    "HAR-QR":         "HAR-QR",
    "HAR-X-QR":       "HAR-X-QR",
}

# ---------------------------------------------------------------------------
# Build the table body
# ---------------------------------------------------------------------------
def fmt(v, best):
    s = f"{v:.6f}"
    return rf"$\mathbf{{{s}}}$" if best else f"${s}$"

lines = []
lines += [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\small",
    r"\caption{Quantile loss by model and horizon across quantile levels}",
    r"\label{tab:quantile-loss-all}",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r"Model & 1d & 1w & 1m & 3m & 6m & 12m \\",
]

for qi, q in enumerate(QUANTILES):
    # best (lowest) QL per horizon at this quantile
    best_per_h = {}
    for h in HORIZONS:
        vals = {m: lookup[(m, h, round(q, 2))] for m in MODELS}
        best_per_h[h] = min(vals, key=vals.get)

    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{7}}{{l}}{{\textit{{Panel {chr(65+qi)}: "
        rf"$\tau = {q:.2f}$}}}} \\"
    )
    lines.append(r"\addlinespace")
    for m in MODELS:
        cells = []
        for h in HORIZONS:
            v = lookup[(m, h, round(q, 2))]
            cells.append(fmt(v, best=(best_per_h[h] == m)))
        lines.append(rf"{LABEL[m]:<17}& " + " & ".join(cells) + r" \\")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\smallskip",
    r"{\footnotesize Quantile loss (Equation~\eqref{eq:quantile-loss}); lower "
    r"values indicate better forecasts. Bold: best model at each "
    r"horizon within a panel. The $\tau = 0.95$ panel reproduces "
    r"Table~\ref{tab:upper-qlike-comparison} from the main text so the "
    r"three panels can be read together. HAR-OLS quantiles are "
    r"pseudo-quantiles from its in-sample residual distribution; HAR-QR "
    r"and HAR-X-QR estimate the quantile directly.}",
    r"\end{table}",
    "",
]

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}")
