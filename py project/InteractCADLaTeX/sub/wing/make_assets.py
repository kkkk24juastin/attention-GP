from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    configure_publication_style,
    draw_colored_boxplot,
    ensure_dir,
    METRIC_LABELS,
    prettify_method,
    project_root,
    save_figure,
    save_tex,
    style_axes,
)


ROOT = project_root(Path(__file__))
CASE_DIR = Path(__file__).resolve().parent
RESULT_FILE = ROOT / "wing_weight_10d" / "refactored_results" / "ga_optimization_results.xlsx"
OUTPUT_TEX = CASE_DIR / "tables.tex"
OUTPUT_FIG_TEX = CASE_DIR / "figures.tex"


def raw() -> pd.DataFrame:
    return pd.read_excel(RESULT_FILE)


def method_order() -> list[str]:
    return ["D_opt", "EI", "G_opt", "IMSE", "K_means", "LHS", "PM", "UCB"]


def build_tables() -> list[str]:
    data = raw()
    row_order = method_order()
    metrics = [
        "RMSE_target",
        "RMSE_all",
        "ga_true_response",
        "ga_true_quality_loss",
    ]
    summary = (
        data.groupby("method")[metrics]
        .agg(["mean", "median"])
        .reindex(row_order)
    )
    summary = summary.rename(
        columns={
            "RMSE_target": "RMSE in the target region",
            "RMSE_all": "RMSE over the full surface",
            "ga_true_response": "GA true response",
            "ga_true_quality_loss": "True quality loss",
            "mean": "Mean",
            "median": "Median",
        }
    )
    summary.index = [prettify_method(m) for m in summary.index]
    summary.index.name = "Method"
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Summary of the 10-dimensional Wing Weight case.}",
        "\\label{tab:sub_wing_all}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        " & \\multicolumn{2}{c}{RMSE in the target region} & \\multicolumn{2}{c}{RMSE over the full surface} & \\multicolumn{2}{c}{GA true response} & \\multicolumn{2}{c}{True quality loss} \\\\",
        "Method & Mean & Median & Mean & Median & Mean & Median & Mean & Median \\\\",
        "\\midrule",
    ]
    for method, row in summary.iterrows():
        values = " & ".join(f"{float(value):.3f}" for value in row.to_numpy())
        lines.append(f"{method} & {values} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ]
    )
    return ["\n".join(lines) + "\n"]


def build_figure() -> None:
    configure_publication_style()
    data = raw()
    methods = method_order()
    colors = {
        "D_opt": "#4e79a7",
        "EI": "#f28e2b",
        "G_opt": "#59a14f",
        "IMSE": "#e15759",
        "K_means": "#b07aa1",
        "LHS": "#9c755f",
        "PM": "#ff9da7",
        "UCB": "#bab0ab",
    }
    metrics = [
        ("RMSE_target", "wing_rmse_target.pdf"),
        ("ga_true_quality_loss", "wing_true_quality_loss.pdf"),
        ("ga_true_response", "wing_true_response.pdf"),
    ]
    for metric, filename in metrics:
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        grouped = [data.loc[data["method"] == method, metric].dropna().to_numpy() for method in methods]
        draw_colored_boxplot(ax, grouped, [prettify_method(m) for m in methods], [colors[m] for m in methods])
        style_axes(ax, "Method", METRIC_LABELS.get(metric, metric.replace("_", " ")))
        save_figure(fig, CASE_DIR / filename)
        plt.close(fig)


def main() -> None:
    ensure_dir(CASE_DIR)
    build_figure()
    save_tex(OUTPUT_TEX, build_tables())
    save_tex(
        OUTPUT_FIG_TEX,
        [
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            "\\includegraphics[width=\\linewidth]{wing_rmse_target.pdf}\n"
            "\\caption{Target-region RMSE for the wing-weight case.}\n"
            "\\label{fig:sub_wing_rmse_target}\n"
            "\\end{figure}\n",
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            "\\includegraphics[width=\\linewidth]{wing_true_quality_loss.pdf}\n"
            "\\caption{True quality loss for the wing-weight case.}\n"
            "\\label{fig:sub_wing_quality_loss}\n"
            "\\end{figure}\n",
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            "\\includegraphics[width=\\linewidth]{wing_true_response.pdf}\n"
            "\\caption{GA true response for the wing-weight case.}\n"
            "\\label{fig:sub_wing_true_response}\n"
            "\\end{figure}\n",
        ],
    )
    print(f"saved: {CASE_DIR / 'wing_rmse_target.pdf'}")
    print(f"saved: {CASE_DIR / 'wing_true_quality_loss.pdf'}")
    print(f"saved: {CASE_DIR / 'wing_true_response.pdf'}")
    print(f"saved: {OUTPUT_TEX}")
    print(f"saved: {OUTPUT_FIG_TEX}")


if __name__ == "__main__":
    main()
