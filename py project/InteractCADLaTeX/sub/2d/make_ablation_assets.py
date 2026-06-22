from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    METRIC_LABELS,
    configure_publication_style,
    draw_colored_boxplot,
    ensure_dir,
    latex_table,
    save_figure,
    save_tex,
    style_axes,
)

ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
RESULT_FILE = ROOT / "2d+20" / "refactored_results" / "4_1_2_pm_ablation.xlsx"
OUTPUT_TEX = CASE_DIR / "ablation_tables.tex"
OUTPUT_FIG_TEX = CASE_DIR / "ablation_figures.tex"

VARIANT_LABELS = {
    "two_stage_softmax": "PM (full)",
    "quality_loss_only": "QL only",
    "mdsf_only": "MDSF only",
    "raw_rank_two_stage": "Two-stage (no softmax)",
    "lambda=0.25": r"Linear ($\lambda{=}0.25$)",
    "lambda=0.50": r"Linear ($\lambda{=}0.50$)",
    "lambda=0.75": r"Linear ($\lambda{=}0.75$)",
}

VARIANT_SHORT_LABELS = {
    "two_stage_softmax": "PM",
    "quality_loss_only": "QL",
    "mdsf_only": "MDSF",
    "raw_rank_two_stage": "No SM",
    "lambda=0.25": "L25",
    "lambda=0.50": "L50",
    "lambda=0.75": "L75",
}

VARIANT_ORDER = [
    "two_stage_softmax",
    "quality_loss_only",
    "mdsf_only",
    "raw_rank_two_stage",
    "lambda=0.25",
    "lambda=0.50",
    "lambda=0.75",
]

VARIANT_COLORS = {
    "two_stage_softmax": "#ff9da7",
    "quality_loss_only": "#4e79a7",
    "mdsf_only": "#59a14f",
    "raw_rank_two_stage": "#f28e2b",
    "lambda=0.25": "#b07aa1",
    "lambda=0.50": "#9c755f",
    "lambda=0.75": "#e15759",
}


def raw() -> pd.DataFrame:
    return pd.read_excel(RESULT_FILE, sheet_name="raw")


def _variant_label(pl: str) -> str:
    return VARIANT_LABELS.get(pl, pl)


def _variant_short_label(pl: str) -> str:
    return VARIANT_SHORT_LABELS.get(pl, pl)


def build_tables() -> list[str]:
    data = raw().copy()
    data["n_total"] = data["n_initial"] + data["n_added"]
    summary = (
        data.groupby(["parameter_label", "n_total"])[["RMSE_target", "RMSE_all"]]
        .agg(["mean", "median", "min", "max", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        .round(4)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]

    blocks = []
    for n_total in sorted(data["n_total"].unique()):
        subset = summary.xs(n_total, level="n_total")
        subset = subset.reindex(VARIANT_ORDER)
        subset.index = [_variant_label(idx) for idx in subset.index]
        subset.index.name = "Variant"
        display = subset[
            ["RMSE_target_mean", "RMSE_target_median", "RMSE_all_mean", "RMSE_all_median"]
        ].reset_index()
        display.columns = [
            "Variant",
            "Target mean",
            "Target median",
            "Overall mean",
            "Overall median",
        ]
        blocks.append(
            latex_table(
                display,
                f"Ablation results at $n={int(n_total)}$.",
                f"tab:sub_2d_ablation_n{int(n_total)}",
                float_digits=4,
                table_pos="h!",
                index=False,
                column_format="lrrrr",
            )
        )
    overall = (
        data.groupby("parameter_label")[["RMSE_target", "RMSE_all"]]
        .agg(["mean", "median", "min", "max"])
        .reindex(VARIANT_ORDER)
        .round(4)
    )
    overall.index = [_variant_label(idx) for idx in overall.index]
    overall.index.name = "Variant"
    flat_overall = overall.copy()
    flat_overall.columns = [f"{a}_{b}" for a, b in flat_overall.columns]
    flat_overall = flat_overall.reset_index()
    flat_overall.columns = [
        "Variant",
        "Target mean",
        "Target median",
        "Target min",
        "Target max",
        "Overall mean",
        "Overall median",
        "Overall min",
        "Overall max",
    ]
    blocks.append(
        latex_table(
            flat_overall,
            "Ablation summary (all training sizes pooled).",
            "tab:sub_2d_ablation_overall",
            float_digits=4,
            table_pos="h!",
            index=False,
            column_format="lrrrrrrrr",
        )
    )
    return blocks


def build_figure() -> None:
    configure_publication_style()
    data = raw().copy()
    data["n_total"] = data["n_initial"] + data["n_added"]
    order = VARIANT_ORDER
    labels = [_variant_short_label(p) for p in order]
    colors = [VARIANT_COLORS[p] for p in order]

    # Boxplot: RMSE_target by variant --- match battery/wing style: figsize=(3.35, 2.65), no rotation
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    grouped = [
        data.loc[data["parameter_label"] == param, "RMSE_target"].dropna().to_numpy()
        for param in order
    ]
    draw_colored_boxplot(ax, grouped, labels, colors)
    style_axes(ax, "Method", METRIC_LABELS["RMSE_target"])
    save_figure(fig, CASE_DIR / "2d_ablation_rmse_target.pdf")
    plt.close(fig)

    # Boxplot: RMSE_all by variant
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    grouped_all = [
        data.loc[data["parameter_label"] == param, "RMSE_all"].dropna().to_numpy()
        for param in order
    ]
    draw_colored_boxplot(ax, grouped_all, labels, colors)
    style_axes(ax, "Method", METRIC_LABELS["RMSE_all"])
    save_figure(fig, CASE_DIR / "2d_ablation_rmse_all.pdf")
    plt.close(fig)

    # Line plot: RMSE_target mean +/- Q1/Q3 vs n_total --- match 2d main style: figsize=(4.35, 2.8)
    n_total_values = sorted(data["n_total"].unique())
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for param in order:
        sub = data[data["parameter_label"] == param]
        means, q1s, q3s = [], [], []
        for nt in n_total_values:
            vals = sub.loc[sub["n_total"] == nt, "RMSE_target"]
            if len(vals):
                means.append(vals.mean())
                q1s.append(vals.quantile(0.25))
                q3s.append(vals.quantile(0.75))
            else:
                means.append(np.nan)
                q1s.append(np.nan)
                q3s.append(np.nan)
        ax.plot(n_total_values, means, marker="o", linewidth=1.6,
                label=_variant_short_label(param), color=VARIANT_COLORS[param])
        ax.fill_between(n_total_values, q1s, q3s,
                        color=VARIANT_COLORS[param], alpha=0.12, linewidth=0)
    ax.set_xticks(n_total_values)
    style_axes(ax, "Total samples", METRIC_LABELS["RMSE_target"])
    ax.legend(frameon=False, ncol=4, fontsize=12.0, columnspacing=0.55, handlelength=1.4)
    save_figure(fig, CASE_DIR / "2d_ablation_rmse_target_vs_n.pdf")
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
            "\\includegraphics[width=\\linewidth]{2d_ablation_rmse_target.pdf}\n"
            "\\caption{Ablation -- RMSE in the target region across variants.}\n"
            "\\label{fig:sub_2d_ablation_target}\n"
            "\\end{figure}\n",
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            "\\includegraphics[width=\\linewidth]{2d_ablation_rmse_all.pdf}\n"
            "\\caption{Ablation -- RMSE over the full response surface across variants.}\n"
            "\\label{fig:sub_2d_ablation_all}\n"
            "\\end{figure}\n",
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            "\\includegraphics[width=\\linewidth]{2d_ablation_rmse_target_vs_n.pdf}\n"
            "\\caption{Ablation -- target-region RMSE versus total training samples.}\n"
            "\\label{fig:sub_2d_ablation_vs_n}\n"
            "\\end{figure}\n",
        ],
    )
    print(f"saved: {OUTPUT_TEX}")
    print(f"saved: {OUTPUT_FIG_TEX}")
    for stem in ["2d_ablation_rmse_target", "2d_ablation_rmse_all", "2d_ablation_rmse_target_vs_n"]:
        print(f"saved: {CASE_DIR / f'{stem}.pdf'}")


if __name__ == "__main__":
    main()
