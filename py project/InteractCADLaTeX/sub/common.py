from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["D_opt", "EI", "G_opt", "IMSE", "K_means", "LHS", "PM", "UCB"]
METHOD_LABELS = {
    "D_opt": "DO",
    "G_opt": "GO",
    "K_means": "CVT",
}


def project_root(path: Path) -> Path:
    return path.resolve().parents[3]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.style": "normal",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "mathtext.default": "regular",
            "lines.linewidth": 1.4,
            "lines.markersize": 4.2,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def method_sort_key(method: str) -> tuple[int, str]:
    return (METHOD_ORDER.index(method), method) if method in METHOD_ORDER else (len(METHOD_ORDER), method)


def prettify_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def float_fmt(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def grouped_summary(
    data: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
    stats: Sequence[str] = ("mean", "median"),
) -> pd.DataFrame:
    agg = data.groupby(list(group_cols))[value_col].agg(list(stats)).reset_index()
    return agg


def summary_matrix(
    data: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    stats: Sequence[str] = ("mean", "median"),
    row_order: Iterable | None = None,
    col_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    row_order = list(row_order) if row_order is not None else list(pd.unique(data[row_col]))
    if col_order is None:
        col_order = sorted(pd.unique(data[col_col]), key=method_sort_key)

    pieces = []
    for col_name in col_order:
        block = (
            data.loc[data[col_col] == col_name]
            .groupby(row_col)[value_col]
            .agg(list(stats))
            .reindex(row_order)
        )
        block.columns = pd.MultiIndex.from_product([[col_name], list(stats)])
        pieces.append(block)

    matrix = pd.concat(pieces, axis=1)
    matrix.index.name = row_col
    return matrix


def method_matrix(
    data: pd.DataFrame,
    row_col: str,
    value_cols: Sequence[str],
    stats: Sequence[str] = ("mean", "median"),
    row_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    row_order = list(row_order) if row_order is not None else sorted(pd.unique(data[row_col]), key=method_sort_key)
    pieces = []
    for value_col in value_cols:
        block = (
            data.groupby(row_col)[value_col]
            .agg(list(stats))
            .reindex(row_order)
        )
        block.columns = pd.MultiIndex.from_product([[value_col], list(stats)])
        pieces.append(block)
    matrix = pd.concat(pieces, axis=1)
    matrix.index.name = row_col
    return matrix


def latex_table(
    body: pd.DataFrame,
    caption: str,
    label: str,
    float_digits: int = 4,
    table_pos: str = "htbp",
) -> str:
    table = body.to_latex(
        escape=False,
        multicolumn=True,
        multirow=True,
        float_format=lambda x: float_fmt(x, float_digits),
        index=True,
        bold_rows=False,
    )
    return (
        f"\\begin{{table}}[{table_pos}]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        f"{table}"
        "}\n"
        "\\end{table}\n"
    )


def save_tex(path: Path, blocks: Sequence[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(blocks), encoding="utf-8")


def plot_line_bands(
    ax,
    x: Sequence[float],
    mean: Sequence[float],
    q1: Sequence[float],
    q3: Sequence[float],
    label: str,
    color: str,
) -> None:
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q3 = np.asarray(q3, dtype=float)
    ax.plot(x, mean, marker="o", linewidth=2.0, label=label, color=color)
    ax.fill_between(x, q1, q3, color=color, alpha=0.14, linewidth=0)


def style_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_fontstyle("normal")
    ax.yaxis.label.set_fontstyle("normal")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontstyle("normal")


def style_boxplot(ax, xlabel: str, ylabel: str) -> None:
    style_axes(ax, xlabel, ylabel)
    ax.tick_params(axis="x", rotation=0)


def draw_colored_boxplot(ax, data_by_method, labels, colors) -> None:
    bp = ax.boxplot(
        data_by_method,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.56,
        medianprops={"color": "#222222", "linewidth": 1.1},
        whiskerprops={"linewidth": 0.9},
        capprops={"linewidth": 0.9},
        boxprops={"linewidth": 0.9},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)


def save_figure(fig, path: Path) -> None:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
