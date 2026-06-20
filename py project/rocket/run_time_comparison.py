import argparse
import importlib
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as robjects

from config import (
    CASE_NAME,
    DATA_FILE,
    FEATURE_NAMES,
    INITIAL_SET_DIR,
    METHOD_MODULES,
    N_POOL,
    PM_EVAL_SEED,
    REPEAT_SEED_START,
    TARGET_VALUE,
    TEST_SEED,
)
from core import frame_to_xy, generate_candidates, quality_loss_scores, tgp_predict
from generate_data import initial_set_filename
from run_ga_optimization import direct_btgp_ga_optimize


DEFAULT_OUTPUT = Path("time.txt")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one n50 timing comparison for each rocket sampling method: "
            "active-learning selection plus one archive-style direct-btgp GA."
        )
    )
    parser.add_argument(
        "--initial-file",
        type=Path,
        default=INITIAL_SET_DIR / initial_set_filename(50),
        help="Initial set workbook. Default: generated_data/initial_sets/rocket_initial_n50.xlsx.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Timing report path. Default: time.txt in the current directory.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_MODULES.keys()),
        help="Methods to run. Default: all configured methods.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=N_POOL,
        help=f"LHS pool size used by active learning. Default: {N_POOL}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=REPEAT_SEED_START,
        help=f"R and LHS seed for this timing run. Default: {REPEAT_SEED_START}.",
    )
    parser.add_argument(
        "--pm-eval-seed",
        type=int,
        default=PM_EVAL_SEED,
        help=f"R seed for grouped PM evaluation. Default: {PM_EVAL_SEED}.",
    )
    return parser.parse_args()


def load_initial_set(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"初始集不存在: {path}")
    frame = pd.read_excel(path)
    x, y = frame_to_xy(frame)
    n_initial = int(frame["n_initial"].iloc[0]) if "n_initial" in frame.columns else len(frame)
    repeat = int(frame["repeat"].iloc[0]) if "repeat" in frame.columns else 1
    return x, y, n_initial, repeat


def load_pool(pool_size, seed):
    if DATA_FILE.exists():
        shared = pd.read_excel(DATA_FILE, sheet_name=None)
        if "pool" in shared:
            pool = shared["pool"]
            if "repeat" in pool.columns:
                pool = pool[pool["repeat"] == 1]
            pool_x = pool[list(FEATURE_NAMES)].to_numpy(dtype=float)
            if len(pool_x) >= int(pool_size):
                return pool_x[: int(pool_size)], np.full(int(pool_size), np.nan)
    return generate_candidates(int(pool_size), int(seed)), np.full(int(pool_size), np.nan)


def run_method_selection(method, module_name, initial_x, initial_y, pool_x, pool_y, seed):
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=f"time_{CASE_NAME}_{method}_select_") as tmp_dir:
        os.chdir(tmp_dir)
        robjects.r.assign("tgp_worker_dir", tmp_dir)
        robjects.r("setwd(tgp_worker_dir)")
        robjects.r(f"set.seed({int(seed)})")
        try:
            module = importlib.import_module(module_name)
            return module.run(initial_x, initial_y, pool_x, pool_y, seed=seed)
        finally:
            os.chdir(original_cwd)
            robjects.r.assign("tgp_worker_dir", str(original_cwd))
            robjects.r("setwd(tgp_worker_dir)")


def run_one_ga(method, train_x, train_y):
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=f"time_{CASE_NAME}_{method}_ga_") as tmp_dir:
        os.chdir(tmp_dir)
        robjects.r.assign("tgp_worker_dir", tmp_dir)
        robjects.r("setwd(tgp_worker_dir)")
        try:
            return direct_btgp_ga_optimize(train_x, train_y)
        finally:
            os.chdir(original_cwd)
            robjects.r.assign("tgp_worker_dir", str(original_cwd))
            robjects.r("setwd(tgp_worker_dir)")


def evaluate_with_pm(pm_x, pm_y, ga_x_rows, seed):
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=f"time_{CASE_NAME}_pm_eval_") as tmp_dir:
        os.chdir(tmp_dir)
        robjects.r.assign("tgp_worker_dir", tmp_dir)
        robjects.r("setwd(tgp_worker_dir)")
        try:
            robjects.r(f"set.seed({int(seed)})")
            mean, variance = tgp_predict(pm_x, pm_y, np.asarray(ga_x_rows, dtype=float))
            ql = quality_loss_scores(mean, variance, TARGET_VALUE)
        finally:
            os.chdir(original_cwd)
            robjects.r.assign("tgp_worker_dir", str(original_cwd))
            robjects.r("setwd(tgp_worker_dir)")
    return np.asarray(mean, dtype=float), np.asarray(variance, dtype=float), np.asarray(ql, dtype=float)


def seconds(value):
    return f"{float(value):.6f}"


def write_report(path, rows, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# rocket runtime comparison",
        *(f"# {key}: {value}" for key, value in metadata.items()),
        "",
        "\t".join(
            [
                "method",
                "selection_seconds",
                "ga_seconds",
                "pm_eval_seconds",
                "total_seconds",
                "n_initial",
                "n_total",
                "ga_self_pred_quality_loss",
                "pm_eval_QL",
                "pm_eval_mean",
                "pm_eval_variance",
            ]
        ),
    ]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row["method"],
                    seconds(row["selection_seconds"]),
                    seconds(row["ga_seconds"]),
                    seconds(row["pm_eval_seconds"]),
                    seconds(row["total_seconds"]),
                    str(row["n_initial"]),
                    str(row["n_total"]),
                    f"{row['ga_self_pred_quality_loss']:.10g}",
                    f"{row['pm_eval_QL']:.10g}",
                    f"{row['pm_eval_mean']:.10g}",
                    f"{row['pm_eval_variance']:.10g}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    initial_x, initial_y, n_initial, repeat = load_initial_set(args.initial_file)
    pool_x, pool_y = load_pool(args.pool_size, args.seed)
    methods = list(args.methods)
    unknown = sorted(set(methods).difference(METHOD_MODULES))
    if unknown:
        raise ValueError(f"未知方法: {unknown}")

    rows = []
    pm_train = None
    for method in methods:
        print(f"{CASE_NAME}: timing selection+GA for {method}", flush=True)
        start_total = time.perf_counter()
        start = time.perf_counter()
        final_x, final_y = run_method_selection(
            method,
            METHOD_MODULES[method],
            initial_x,
            initial_y,
            pool_x,
            pool_y,
            seed=args.seed,
        )
        selection_seconds = time.perf_counter() - start
        if method == "PM":
            pm_train = (final_x.copy(), final_y.copy())

        start = time.perf_counter()
        ga_metrics = run_one_ga(method, final_x, final_y)
        ga_seconds = time.perf_counter() - start
        ga_x = np.asarray(
            [ga_metrics[f"ga_x{idx}"] for idx in range(1, len(FEATURE_NAMES) + 1)],
            dtype=float,
        )
        rows.append(
            {
                "method": method,
                "selection_seconds": selection_seconds,
                "ga_seconds": ga_seconds,
                "pm_eval_seconds": np.nan,
                "total_seconds": time.perf_counter() - start_total,
                "n_initial": n_initial,
                "n_total": len(final_x),
                "ga_self_pred_quality_loss": ga_metrics["ga_self_pred_quality_loss"],
                "pm_eval_QL": np.nan,
                "pm_eval_mean": np.nan,
                "pm_eval_variance": np.nan,
                "ga_x": ga_x,
            }
        )
        print(
            f"{CASE_NAME}: {method} selection={selection_seconds:.2f}s, "
            f"GA={ga_seconds:.2f}s",
            flush=True,
        )

    if pm_train is None:
        print(f"{CASE_NAME}: PM was not in --methods; running PM selection for evaluation framework.", flush=True)
        pm_x, pm_y = run_method_selection(
            "PM",
            METHOD_MODULES["PM"],
            initial_x,
            initial_y,
            pool_x,
            pool_y,
            seed=args.seed,
        )
    else:
        pm_x, pm_y = pm_train

    start = time.perf_counter()
    mean, variance, ql = evaluate_with_pm(pm_x, pm_y, [row["ga_x"] for row in rows], args.pm_eval_seed)
    pm_eval_seconds = time.perf_counter() - start
    for idx, row in enumerate(rows):
        row["pm_eval_seconds"] = pm_eval_seconds
        row["total_seconds"] += pm_eval_seconds
        row["pm_eval_mean"] = float(mean[idx])
        row["pm_eval_variance"] = float(variance[idx])
        row["pm_eval_QL"] = float(ql[idx])
        row.pop("ga_x", None)

    metadata = {
        "initial_file": Path(args.initial_file),
        "n_initial": n_initial,
        "repeat": repeat,
        "pool_size": args.pool_size,
        "seed": args.seed,
        "pm_eval_seed": args.pm_eval_seed,
        "methods": ",".join(methods),
    }
    write_report(args.output, rows, metadata)
    print(f"{CASE_NAME}: timing report saved to {Path(args.output)}")


if __name__ == "__main__":
    main()
