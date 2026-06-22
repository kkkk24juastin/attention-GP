from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_dir, latex_table, save_tex, prettify_method

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
OUTPUT = CASE_DIR / "appendix_tables.tex"

METHODS = ["D_opt", "EI", "G_opt", "IMSE", "K_means", "LHS", "PM", "UCB"]

def load_all(n_added):
    extra = pd.read_excel(ROOT / f"2d+{n_added}" / "refactored_results" / "all_methods_results.xlsx")
    extra["n_total"] = extra["n_initial"] + extra["n_added"]
    if n_added == 10:
        base = pd.read_excel(ROOT / "2d+10" / "initial_history_结果.xlsx")
    else:
        base = pd.read_excel(ROOT / "2d+20" / "initial_history_data.xlsx")
    base["n_total"] = base["n_initial"] + n_added
    base["n_added"] = n_added
    data = pd.concat([base, extra], ignore_index=True, sort=False)
    return data

def build():
    blocks = []
    for n_added in [10, 20]:
        data = load_all(n_added)
        for n_total in sorted(data["n_total"].unique()):
            subset = data[data["n_total"] == n_total]
            rows = []
            for m in METHODS:
                vals = subset.loc[subset["method"] == m, "RMSE_target"].dropna()
                rows.append({
                    "method": prettify_method(m),
                    "median": round(vals.median(), 4),
                    "min": round(vals.min(), 4),
                    "max": round(vals.max(), 4),
                    "q1": round(vals.quantile(0.25), 4),
                    "q3": round(vals.quantile(0.75), 4),
                })
            df = pd.DataFrame(rows).set_index("method")
            blocks.append(latex_table(
                df,
                f"RMSE statistics in the potential optimum region ({n_added} active learning points, n={int(n_total)}). Median, min, max, Q1, Q3.",
                f"tab:appendix_rmse_{n_added}_n{int(n_total)}",
                float_digits=4,
            ))
    return blocks

def main():
    ensure_dir(CASE_DIR)
    save_tex(OUTPUT, build())
    print(f"saved: {OUTPUT}")

if __name__ == "__main__":
    main()
