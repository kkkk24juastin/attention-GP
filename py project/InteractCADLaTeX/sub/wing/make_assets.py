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
    latex_table,
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
        "RMSE_all",
        "RMSE_target",
        "best_true_target_error",
        "best_true_response",
        "ga_true_quality_loss",
    ]
    summary = (
        data.groupby("method")[metrics]
        .agg(["mean", "median"])
        .reindex(row_order)
    )
    summary.index = [prettify_method(m) for m in summary.index]
    summary.index.name = "method"
    return [
        latex_table(
            summary,
            "Wing-weight summary for all methods.",
            "tab:sub_wing_all",
            float_digits=3,
        ),
    ]


def _metric_frame(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        data.groupby("method")[metric]
        .apply(list)
        .reindex(method_order())
        .to_frame(name="values")
    )


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
        ("RMSE_target", "wing_rmse_target.pdf", "Target-region RMSE"),
        ("best_true_target_error", "wing_true_target_error.pdf", "True target error"),
        ("ga_true_quality_loss", "wing_true_quality_loss.pdf", "True quality loss"),
    ]
    for metric, filename, title in metrics:
        fig, ax = plt.subplots(figsize=(3.35, 2.65))
        grouped = [data.loc[data["method"] == method, metric].dropna().to_numpy() for method in methods]
        draw_colored_boxplot(ax, grouped, [prettify_method(m) for m in methods], [colors[m] for m in methods])
        ax.set_title(title)
        style_axes(ax, "Method", metric.replace("_", " "))
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
            "\\includegraphics[width=\\linewidth]{wing_true_target_error.pdf}\n"
            "\\caption{True target error for the wing-weight case.}\n"
            "\\label{fig:sub_wing_target_error}\n"
            "\\end{figure}\n",
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            "\\includegraphics[width=\\linewidth]{wing_true_quality_loss.pdf}\n"
            "\\caption{True quality loss for the wing-weight case.}\n"
            "\\label{fig:sub_wing_quality_loss}\n"
            "\\end{figure}\n",
        ],
    )
    print(f"saved: {CASE_DIR / 'wing_rmse_target.pdf'}")
    print(f"saved: {CASE_DIR / 'wing_true_target_error.pdf'}")
    print(f"saved: {CASE_DIR / 'wing_true_quality_loss.pdf'}")
    print(f"saved: {OUTPUT_TEX}")
    print(f"saved: {OUTPUT_FIG_TEX}")


if __name__ == "__main__":
    main()
