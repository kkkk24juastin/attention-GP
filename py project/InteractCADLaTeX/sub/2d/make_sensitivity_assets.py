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
    ensure_dir,
    latex_table,
    save_figure,
    save_tex,
    style_axes,
)

ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
RESULT_FILE = ROOT / "2d+20" / "refactored_results" / "4_1_4_pm_target_sensitivity.xlsx"
OUTPUT_TEX = CASE_DIR / "sensitivity_tables.tex"
OUTPUT_FIG_TEX = CASE_DIR / "sensitivity_figures.tex"

T_COLORS = {
    0.0: "#4e79a7",
    1.2: "#f28e2b",
    1.3: "#e15759",
    1.5: "#ff9da7",
    1.6: "#59a14f",
    1.7: "#b07aa1",
}


def raw() -> pd.DataFrame:
    return pd.read_excel(RESULT_FILE, sheet_name="raw")


def build_tables() -> list[str]:
    data = raw().copy()
    data["n_total"] = data["n_initial"] + data["n_added"]
    t_vals = sorted(data["sampling_target_value"].unique())

    blocks = []
    for n_total in sorted(data["n_total"].unique()):
        subset = data[data["n_total"] == n_total]
        summary = (
            subset.groupby("sampling_target_value")[["RMSE_target", "RMSE_all", "best_true_target_error"]]
            .agg(["mean", "median"])
            .round(4)
            .reindex(t_vals)
        )
        summary.index.name = "Sampling $T$"
        flat = summary.copy()
        flat.columns = [f"{a}_{b}" for a, b in flat.columns]
        display = flat[
            ["RMSE_target_mean", "RMSE_target_median", "RMSE_all_mean",
             "RMSE_all_median", "best_true_target_error_mean", "best_true_target_error_median"]
        ].reset_index()
        display.columns = [
            "Sampling $T$",
            "Target mean",
            "Target median",
            "Overall mean",
            "Overall median",
            "Best error mean",
            "Best error median",
        ]
        blocks.append(
            latex_table(
                display,
                f"Target sensitivity (n_total={int(n_total)}). Evaluation target fixed at $T=1.5$.",
                f"tab:sub_2d_sensitivity_n{int(n_total)}",
                float_digits=4,
                table_pos="h!",
                index=False,
                column_format="crrrrrr",
            )
        )

    overall = (
        data.groupby("sampling_target_value")[["RMSE_target", "RMSE_all", "best_true_target_error"]]
        .agg(["mean", "median", "min", "max"])
        .reindex(t_vals)
        .round(4)
    )
    overall.index.name = "Sampling $T$"
    flat_overall = overall.copy()
    flat_overall.columns = [f"{a}_{b}" for a, b in flat_overall.columns]
    flat_overall = flat_overall.reset_index()
    flat_overall.columns = [
        "Sampling $T$",
        "Target mean",
        "Target median",
        "Target min",
        "Target max",
        "Overall mean",
        "Overall median",
        "Overall min",
        "Overall max",
        "Best error mean",
        "Best error median",
        "Best error min",
        "Best error max",
    ]
    blocks.append(
        latex_table(
            flat_overall,
            "Target sensitivity summary (all training sizes pooled). Evaluation target fixed at $T=1.5$.",
            "tab:sub_2d_sensitivity_overall",
            float_digits=4,
            table_pos="h!",
            index=False,
            column_format="crrrrrrrrrrrr",
        )
    )
    return blocks


def build_figure() -> None:
    configure_publication_style()
    data = raw().copy()
    data["n_total"] = data["n_initial"] + data["n_added"]
    t_vals = sorted(data["sampling_target_value"].unique())
    n_total_values = sorted(data["n_total"].unique())

    # Figure 1: RMSE_target vs n_total --- match 2d main style: figsize=(4.35, 2.8)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for t_val in t_vals:
        sub = data[data["sampling_target_value"] == t_val]
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
        color = T_COLORS.get(t_val, "#999999")
        ax.plot(n_total_values, means, marker="o", linewidth=1.6,
                label=f"Sampling $T={t_val}$", color=color)
        ax.fill_between(n_total_values, q1s, q3s,
                        color=color, alpha=0.10, linewidth=0)
    ax.set_xticks(n_total_values)
    style_axes(ax, "Total samples", METRIC_LABELS["RMSE_target"])
    ax.legend(frameon=False, ncol=2, fontsize=12.0, columnspacing=0.5)
    save_figure(fig, CASE_DIR / "2d_sensitivity_rmse_target.pdf")
    plt.close(fig)

    # Figure 2: Bar chart --- match battery/wing boxplot proportions: (3.35, 2.65)
    max_nt = max(n_total_values)
    bar_data = (
        data[data["n_total"] == max_nt]
        .groupby("sampling_target_value")["RMSE_target"]
        .agg(["mean", "median"])
        .reindex(t_vals)
    )
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    x = np.arange(len(t_vals))
    width = 0.35
    ax.bar(x - width/2, bar_data["mean"], width,
           label="Mean", color="#4e79a7", alpha=0.8)
    ax.bar(x + width/2, bar_data["median"], width,
           label="Median", color="#f28e2b", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$T={tv}$" for tv in t_vals])
    ax.legend(frameon=False, fontsize=12.0)
    style_axes(ax, "Sampling target", METRIC_LABELS["RMSE_target"])
    save_figure(fig, CASE_DIR / "2d_sensitivity_bar.pdf")
    plt.close(fig)

    # Figure 3: best_true_target_error vs n_total --- match 2d main style
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for t_val in t_vals:
        sub = data[data["sampling_target_value"] == t_val]
        means, q1s, q3s = [], [], []
        for nt in n_total_values:
            vals = sub.loc[sub["n_total"] == nt, "best_true_target_error"]
            if len(vals):
                means.append(vals.mean())
                q1s.append(vals.quantile(0.25))
                q3s.append(vals.quantile(0.75))
            else:
                means.append(np.nan)
                q1s.append(np.nan)
                q3s.append(np.nan)
        color = T_COLORS.get(t_val, "#999999")
        ax.plot(n_total_values, means, marker="s", linewidth=1.6,
                label=f"Sampling $T={t_val}$", color=color)
        ax.fill_between(n_total_values, q1s, q3s,
                        color=color, alpha=0.10, linewidth=0)
    ax.set_xticks(n_total_values)
    style_axes(ax, "Total samples", METRIC_LABELS["best_true_target_error"])
    ax.legend(frameon=False, ncol=2, fontsize=12.0, columnspacing=0.5)
    save_figure(fig, CASE_DIR / "2d_sensitivity_true_error.pdf")
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
            "\\includegraphics[width=\\linewidth]{2d_sensitivity_rmse_target.pdf}\n"
            "\\caption{Target sensitivity -- target-region RMSE versus total training samples for different sampling targets $T$. Evaluation target fixed at $T=1.5$.}\n"
            "\\label{fig:sub_2d_sensitivity_rmse}\n"
            "\\end{figure}\n",
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            "\\includegraphics[width=\\linewidth]{2d_sensitivity_bar.pdf}\n"
            "\\caption{Target sensitivity -- mean and median target-region RMSE at the largest training size ($n=100$) across sampling targets.}\n"
            "\\label{fig:sub_2d_sensitivity_bar}\n"
            "\\end{figure}\n",
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            "\\includegraphics[width=\\linewidth]{2d_sensitivity_true_error.pdf}\n"
            "\\caption{Target sensitivity -- Best true target error versus total training samples.}\n"
            "\\label{fig:sub_2d_sensitivity_true_error}\n"
            "\\end{figure}\n",
        ],
    )
    print(f"saved: {OUTPUT_TEX}")
    print(f"saved: {OUTPUT_FIG_TEX}")
    for stem in ["2d_sensitivity_rmse_target", "2d_sensitivity_bar", "2d_sensitivity_true_error"]:
        print(f"saved: {CASE_DIR / f'{stem}.pdf'}")


if __name__ == "__main__":
    main()
