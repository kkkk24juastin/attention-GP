from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    METRIC_LABELS,
    configure_publication_style,
    ensure_dir,
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
BASELINE_METHODS = [method for method in METHODS if method != "PM"]
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
    table = table.rename(columns={"n_total": "Number"})
    return table


def grouped_summary(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        data.groupby(["n_total", "method"])[metric]
        .agg(mean="mean", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75))
        .reset_index()
    )


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.2f}\\%"


def _combined_rmse_table(
    data: pd.DataFrame,
    metric: str,
    rate_label_mean: str,
    rate_label_median: str,
    rate_direction: str,
) -> pd.DataFrame:
    grouped = (
        data.groupby(["n_total", "method"])[metric]
        .agg(mean="mean", median="median")
        .reset_index()
    )
    mean = (
        grouped.pivot(index="n_total", columns="method", values="mean")
        .reindex(columns=METHODS)
        .sort_index()
    )
    median = (
        grouped.pivot(index="n_total", columns="method", values="median")
        .reindex(columns=METHODS)
        .sort_index()
    )
    rows = []
    for n_total in mean.index:
        row = {"Number": int(n_total)}
        for method in METHODS:
            row[("RMSE (Mean)", prettify_method(method))] = mean.loc[n_total, method]
        for method in METHODS:
            row[("RMSE (Median)", prettify_method(method))] = median.loc[n_total, method]
        for method in BASELINE_METHODS:
            if rate_direction == "improvement":
                value = (mean.loc[n_total, method] - mean.loc[n_total, "PM"]) / mean.loc[n_total, method] * 100
            else:
                value = (mean.loc[n_total, "PM"] - mean.loc[n_total, method]) / mean.loc[n_total, method] * 100
            row[(rate_label_mean, prettify_method(method))] = _format_pct(value)
        row[(rate_label_mean, "PM")] = "--"
        for method in BASELINE_METHODS:
            if rate_direction == "improvement":
                value = (
                    (median.loc[n_total, method] - median.loc[n_total, "PM"])
                    / median.loc[n_total, method]
                    * 100
                )
            else:
                value = (
                    (median.loc[n_total, "PM"] - median.loc[n_total, method])
                    / median.loc[n_total, method]
                    * 100
                )
            row[(rate_label_median, prettify_method(method))] = _format_pct(value)
        row[(rate_label_median, "PM")] = "--"
        rows.append(row)

    table = pd.DataFrame(rows)
    ordered_columns = ["Number"]
    for group in [
        "RMSE (Mean)",
        "RMSE (Median)",
        rate_label_mean,
        rate_label_median,
    ]:
        for method in METHODS:
            ordered_columns.append((group, prettify_method(method)))
    return table[ordered_columns]


def combined_rmse_latex_table(
    data: pd.DataFrame,
    metric: str,
    caption: str,
    label: str,
    rate_label_mean: str,
    rate_label_median: str,
    rate_direction: str,
    response_tag: str,
    table_pos: str = "h!",
) -> str:
    table = _combined_rmse_table(data, metric, rate_label_mean, rate_label_median, rate_direction)
    metric_count = len(METHODS)
    column_format = "c " + "r" * metric_count + " c " + "r" * metric_count
    first_start, first_end = 2, 1 + metric_count
    second_start, second_end = 3 + metric_count, 2 + metric_count * 2

    def add_pair_block(lines: list[str], mean_group: str, median_group: str) -> None:
        mean_label = mean_group.replace("%", r"\%")
        median_label = median_group.replace("%", r"\%")
        lines.append(
            f"\\multirow{{2}}{{*}}{{Number}} & "
            f"\\multicolumn{{{metric_count}}}{{c}}{{{mean_label}}} & & "
            f"\\multicolumn{{{metric_count}}}{{c}}{{{median_label}}} \\\\"
        )
        lines.append(f"\\cmidrule(lr){{{first_start}-{first_end}}} \\cmidrule(lr){{{second_start}-{second_end}}}")
        method_header = " & ".join(prettify_method(method) for method in METHODS)
        lines.append(f"& {method_header} & & {method_header} \\\\")
        lines.append("\\midrule")
        for _, record in table.iterrows():
            values = [str(int(record["Number"]))]
            for method in METHODS:
                value = record[(mean_group, prettify_method(method))]
                values.append(value if isinstance(value, str) else f"{value:.4f}")
            values.append("")
            for method in METHODS:
                value = record[(median_group, prettify_method(method))]
                values.append(value if isinstance(value, str) else f"{value:.4f}")
            lines.append(" & ".join(values) + r" \\")

    lines = [
        f"\\begin{{table}}[{table_pos}]",
        "\\centering",
        f"\\caption{{\\reviewadd{{{caption}}}{{{response_tag}}}}}",
        f"\\label{{{label}}}",
        "\\begingroup\\color{red}",
        "\\resizebox{\\linewidth}{!}{%",
        f"\\begin{{tabular}}{{{column_format}}}",
        "\\toprule",
    ]
    add_pair_block(lines, "RMSE (Mean)", "RMSE (Median)")
    lines.append("\\midrule")
    add_pair_block(lines, rate_label_mean, rate_label_median)
    lines.append("\\bottomrule")
    lines.extend(["\\end{tabular}%", "}", "\\endgroup", "\\end{table}", ""])
    return "\n".join(lines)


def build_tables() -> list[str]:
    blocks: list[str] = []
    for n_added in (10, 20):
        data = load_case(n_added)
        blocks.append(
            combined_rmse_latex_table(
                data,
                "RMSE_target",
                f"新增 {n_added} 个主动学习点时目标区域 RMSE 及 PM 改善率。",
                f"tab:rmse_target_{n_added}",
                "PM Improvement (Mean, %)",
                "PM Improvement (Median, %)",
                "improvement",
                "R1-2, R2-4",
                table_pos="h!",
            )
        )
        blocks.append(
            combined_rmse_latex_table(
                data,
                "RMSE_all",
                f"新增 {n_added} 个主动学习点时整体响应面 RMSE 及 PM 损失率。",
                f"tab:rmse_all_{n_added}",
                "PM Loss Rate (Mean, %)",
                "PM Loss Rate (Median, %)",
                "loss",
                "R1-3",
                table_pos="h!",
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
            fig, ax = plt.subplots(figsize=(6.0, 5.0))
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
            style_axes(ax, "Total samples", METRIC_LABELS[metric])
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
