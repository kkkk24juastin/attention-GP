from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    configure_publication_style,
    ensure_dir,
    latex_table,
    plot_line_bands,
    prettify_method,
    project_root,
    save_figure,
    save_tex,
    style_axes,
)


ROOT = project_root(Path(__file__))
CASE_DIR = Path(__file__).resolve().parent
RESULT_FILES = {
    10: ROOT / "2d+10" / "initial_history_结果.xlsx",
    20: ROOT / "2d+20" / "initial_history_data.xlsx",
}
ALL_METHOD_FILES = {
    10: ROOT / "2d+10" / "refactored_results" / "all_methods_results.xlsx",
    20: ROOT / "2d+20" / "refactored_results" / "all_methods_results.xlsx",
}
METHODS = ["D_opt", "EI", "G_opt", "IMSE", "K_means", "LHS", "PM", "UCB"]
PALETTE = {
    "D_opt": "#4e79a7",
    "EI": "#f28e2b",
    "G_opt": "#59a14f",
    "IMSE": "#e15759",
    "K_means": "#b07aa1",
    "LHS": "#9c755f",
    "PM": "#ff9da7",
    "UCB": "#bab0ab",
}


def load_case(n_added: int) -> pd.DataFrame:
    base = pd.read_excel(RESULT_FILES[n_added]).copy()
    base["n_added"] = n_added
    base["n_total"] = base["n_initial"] + base["n_added"]

    extra = pd.read_excel(ALL_METHOD_FILES[n_added]).copy()
    if "case" in extra.columns:
        extra = extra[extra["case"].astype(str).str.contains(f"2d+{n_added}", regex=False)].copy()
    extra["n_added"] = n_added
    extra["n_total"] = extra["n_initial"] + extra["n_added"]

    data = pd.concat([base, extra], ignore_index=True, sort=False)
    return data


def summary_case(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = (
        data.pivot_table(index="n_total", columns="method", values=metric, aggfunc="mean")
        .reindex(columns=METHODS)
        .sort_index()
        .reset_index()
    )
    return table


def grouped_summary(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        data.groupby(["n_total", "method"])[metric]
        .agg(mean="mean", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75))
        .reset_index()
    )


def build_tables() -> list[str]:
    blocks: list[str] = []
    for n_added in (10, 20):
        data = load_case(n_added)
        target = summary_case(data, "RMSE_target")
        overall = summary_case(data, "RMSE_all")
        target.columns = [str(c) if c == "n_total" else prettify_method(str(c)) for c in target.columns]
        overall.columns = [str(c) if c == "n_total" else prettify_method(str(c)) for c in overall.columns]
        blocks.append(
            latex_table(
                target,
                f"RMSE results for the potential optimum region in the 2D example with {n_added} active learning points.",
                f"tab:sub_2d_rmse_target_{n_added}",
                float_digits=4,
            )
        )
        blocks.append(
            latex_table(
                overall,
                f"RMSE results for the overall response surface in the 2D example with {n_added} active learning points.",
                f"tab:sub_2d_rmse_all_{n_added}",
                float_digits=4,
            )
        )
    return blocks


def build_figure() -> None:
    configure_publication_style()
    for n_added in (10, 20):
        data = load_case(n_added)
        x_values = sorted(data["n_total"].unique())
        target_summary = grouped_summary(data, "RMSE_target")
        overall_summary = grouped_summary(data, "RMSE_all")

        for metric, summary, suffix in [
            ("RMSE_target", target_summary, "target"),
            ("RMSE_all", overall_summary, "overall"),
        ]:
            fig, ax = plt.subplots(figsize=(4.35, 2.8))
            for method in METHODS:
                subset = summary.loc[summary["method"] == method].set_index("n_total").reindex(x_values)
                plot_line_bands(
                    ax,
                    subset.index.to_numpy(),
                    subset["mean"].to_numpy(),
                    subset["q1"].to_numpy(),
                    subset["q3"].to_numpy(),
                    prettify_method(method),
                    PALETTE[method],
                )
            ax.set_xticks(x_values)
            style_axes(ax, "Total samples", "RMSE")
            ax.legend(frameon=False, ncol=2, loc="best", columnspacing=0.8, handlelength=1.8)
            save_figure(fig, CASE_DIR / f"2d_{suffix}_{n_added}.pdf")
            plt.close(fig)


def build_figure_tex() -> list[str]:
    blocks: list[str] = []
    for n_added in (10, 20):
        blocks.append(
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{2d_target_{n_added}.pdf}}\n"
            f"\\caption{{RMSE in the target region for the 2D example with {n_added} active learning points.}}\n"
            f"\\label{{fig:sub_2d_target_{n_added}}}\n"
            "\\end{figure}\n"
        )
        blocks.append(
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{2d_overall_{n_added}.pdf}}\n"
            f"\\caption{{RMSE over the full surface for the 2D example with {n_added} active learning points.}}\n"
            f"\\label{{fig:sub_2d_overall_{n_added}}}\n"
            "\\end{figure}\n"
        )
    return blocks


def main() -> None:
    ensure_dir(CASE_DIR)
    build_figure()
    save_tex(OUTPUT_TEX := CASE_DIR / "tables.tex", build_tables())
    save_tex(OUTPUT_FIG_TEX := CASE_DIR / "figures.tex", build_figure_tex())
    print(f"saved: {OUTPUT_TEX}")
    print(f"saved: {OUTPUT_FIG_TEX}")


if __name__ == "__main__":
    main()
