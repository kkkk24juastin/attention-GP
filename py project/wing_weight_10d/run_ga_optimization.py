import argparse
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as robjects

from config import (
    CASE_NAME,
    DATASET_DIR,
    DATASET_INDEX_FILE,
    DATA_FILE,
    FEATURE_NAMES,
    GA_GENERATIONS,
    GA_POP_SIZE,
    GA_REPEATS,
    GA_STALL_GENERATIONS,
    LOWER_BOUNDS,
    PM_EVAL_SEED,
    RESULT_DIR,
    TARGET_BAND,
    TARGET_VALUE,
    UPPER_BOUNDS,
)
from core import (
    EPS,
    NUMPY_CONVERTER,
    completed_keys,
    frame_to_xy,
    load_shared_data,
    load_training_set,
    non_test_function,
    quality_loss_scores,
    save_result_row,
    summarize_results,
    tgp_predict,
)


DIRECT_GA_RESULT_FILE = RESULT_DIR / "ga_optimization_results.xlsx"
DIRECT_SUMMARY_FILE = RESULT_DIR / "summary_statistics.xlsx"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run archive-style direct btgp(X, Z, XX=x) GA, then report the main "
            "QL by evaluating all GA points jointly under the PM framework."
        )
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Only run these methods, for example: --methods PM UCB",
    )
    parser.add_argument(
        "--ga-repeats",
        type=int,
        default=GA_REPEATS,
        help=f"GA repeats per dataset. Default: {GA_REPEATS}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("DIRECT_BTGP_WORKERS", "12")),
        help="Parallel worker count. Default: 12, or DIRECT_BTGP_WORKERS.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun matching tasks even if they already exist in the formal result file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DIRECT_GA_RESULT_FILE,
        help=f"Result xlsx path. Default: {DIRECT_GA_RESULT_FILE}",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DIRECT_SUMMARY_FILE,
        help=f"Summary xlsx path. Default: {DIRECT_SUMMARY_FILE}",
    )
    parser.add_argument(
        "--pm-eval-seed",
        type=int,
        default=PM_EVAL_SEED,
        help=f"R seed used before grouped PM btgp evaluation. Default: {PM_EVAL_SEED}.",
    )
    return parser.parse_args()


def load_dataset_index(methods=None):
    if not DATASET_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"训练集索引不存在: {DATASET_INDEX_FILE}，请先运行 run_experiments.py"
        )
    data = pd.read_excel(DATASET_INDEX_FILE)
    if methods:
        requested = set(methods)
        data = data[data["method"].isin(requested)].copy()
        missing = sorted(requested.difference(set(data["method"])))
        if missing:
            raise ValueError(f"训练集索引中找不到这些方法: {missing}")
    if data.empty:
        raise ValueError("没有可运行的训练集。")
    return data


def resolve_dataset_path(row):
    dataset_path = Path(row.dataset_path)
    if dataset_path.exists():
        return dataset_path
    local_path = DATASET_DIR / row.dataset_file
    if local_path.exists():
        return local_path
    raise FileNotFoundError(
        f"训练集文件不存在: {dataset_path}，本地回退路径也不存在: {local_path}"
    )


def build_tasks(done, dataset_index, shared_data, ga_repeats, force=False):
    test_x, test_y = frame_to_xy(shared_data["test"])
    pm_rows = dataset_index[dataset_index["method"] == "PM"]
    if pm_rows.empty:
        raise ValueError("PM 训练集不存在，无法按 PM 框架评价 direct GA 结果。")
    pm_paths = {
        (int(row.n_initial), int(row.repeat)): str(resolve_dataset_path(row))
        for row in pm_rows.itertuples(index=False)
    }
    tasks = []
    for row in dataset_index.itertuples(index=False):
        dataset_path = resolve_dataset_path(row)
        pm_key = (int(row.n_initial), int(row.repeat))
        if pm_key not in pm_paths:
            raise ValueError(f"找不到对应 PM 评价训练集: n_initial={pm_key[0]}, repeat={pm_key[1]}")
        for ga_repeat in range(1, int(ga_repeats) + 1):
            key = (row.method, int(row.n_initial), int(row.repeat), int(ga_repeat))
            if not force and key in done:
                continue
            tasks.append(
                {
                    "method": row.method,
                    "n_initial": int(row.n_initial),
                    "n_added": int(row.n_added),
                    "n_total": int(row.n_total),
                    "repeat": int(row.repeat),
                    "seed": int(row.seed),
                    "ga_repeat": int(ga_repeat),
                    "dataset_file": row.dataset_file,
                    "dataset_path": str(dataset_path),
                    "pm_dataset_path": pm_paths[pm_key],
                    "test_x": test_x,
                    "test_y": test_y,
                }
            )
    return tasks


def reference_metrics(X_train, y_train, test_x, test_y):
    pred_y, pred_var = tgp_predict(X_train, y_train, test_x)
    pred_y = np.asarray(pred_y, dtype=float).ravel()
    pred_var = np.maximum(np.asarray(pred_var, dtype=float).ravel(), EPS)
    test_y = np.asarray(test_y, dtype=float).ravel()

    rmse_all = float(np.sqrt(np.mean((pred_y - test_y) ** 2)))
    mask = (test_y >= TARGET_VALUE - TARGET_BAND) & (
        test_y <= TARGET_VALUE + TARGET_BAND
    )
    rmse_target = (
        float(np.sqrt(np.mean((pred_y[mask] - test_y[mask]) ** 2)))
        if mask.any()
        else np.nan
    )
    quality_loss = quality_loss_scores(pred_y, pred_var, TARGET_VALUE)
    best_idx = int(np.nanargmin(quality_loss))
    best_x = test_x[best_idx]
    result = {
        "RMSE_all": rmse_all,
        "RMSE_target": rmse_target,
        "best_pred_quality_loss": float(quality_loss[best_idx]),
        "best_true_target_error": float((test_y[best_idx] - TARGET_VALUE) ** 2),
        "best_true_response": float(test_y[best_idx]),
    }
    for idx, value in enumerate(best_x, start=1):
        result[f"best_x{idx}"] = float(value)
    return result


def pm_frame_evaluate_ga_points(ga_x, pm_x, pm_y, seed):
    robjects.r(f"set.seed({int(seed)})")
    pm_mu, pm_var = tgp_predict(pm_x, pm_y, np.asarray(ga_x, dtype=float))
    pm_mu = np.asarray(pm_mu, dtype=float).ravel()
    pm_var = np.maximum(np.asarray(pm_var, dtype=float).ravel(), EPS)
    pm_quality_loss = quality_loss_scores(pm_mu, pm_var, TARGET_VALUE)
    return {
        "pm_eval_mean": pm_mu,
        "pm_eval_variance": pm_var,
        "pm_eval_QL": pm_quality_loss,
    }


def direct_btgp_ga_optimize(X_train, y_train):
    r = robjects.r
    lower = np.asarray(LOWER_BOUNDS, dtype=float)
    upper = np.asarray(UPPER_BOUNDS, dtype=float)

    with NUMPY_CONVERTER.context():
        r.assign("X_train", np.asarray(X_train, dtype=float))
        r.assign("y_train", np.asarray(y_train, dtype=float))
        r.assign("target_value", float(TARGET_VALUE))
        r.assign("lower_bounds", lower)
        r.assign("upper_bounds", upper)
        r.assign("eps_value", float(EPS))
        r.assign("ga_pop_size", int(GA_POP_SIZE))
        r.assign("ga_generations", int(GA_GENERATIONS))
        r.assign("ga_stall_generations", int(GA_STALL_GENERATIONS))
        r(
            """
            suppressPackageStartupMessages(library(GA))
            suppressPackageStartupMessages(library(tgp))

            direct_qualityfun <- function(x) {
                x <- matrix(as.numeric(x), nrow = 1)
                btgpforopt <- tryCatch(
                    btgp(X = X_train, Z = y_train, XX = x, verb = 0),
                    error = function(e) NULL
                )
                if (is.null(btgpforopt) ||
                    is.null(btgpforopt$ZZ.mean) ||
                    is.null(btgpforopt$ZZ.s2)) {
                    return(.Machine$double.xmax / 100)
                }
                mu <- as.numeric(btgpforopt$ZZ.mean)[1]
                var <- as.numeric(btgpforopt$ZZ.s2)[1]
                if (!is.finite(mu) || !is.finite(var)) {
                    return(.Machine$double.xmax / 100)
                }
                var <- max(var, eps_value)
                return((mu - target_value)^2 + var)
            }

            direct_eval <- function(x) {
                x <- matrix(as.numeric(x), nrow = 1)
                btgpforopt <- btgp(X = X_train, Z = y_train, XX = x, verb = 0)
                mu <- as.numeric(btgpforopt$ZZ.mean)[1]
                var <- max(as.numeric(btgpforopt$ZZ.s2)[1], eps_value)
                return(c(mu = mu, var = var, ql = (mu - target_value)^2 + var))
            }

            direct_ga_model <- ga(
                type = "real-valued",
                fitness = function(x) -direct_qualityfun(x),
                lower = lower_bounds,
                upper = upper_bounds,
                popSize = ga_pop_size,
                maxiter = ga_generations,
                run = ga_stall_generations,
                parallel = FALSE
            )

            direct_solutions <- direct_ga_model@solution
            if (is.null(dim(direct_solutions))) {
                direct_solutions <- matrix(direct_solutions, nrow = 1)
            }
            direct_values <- apply(direct_solutions, 1, direct_qualityfun)
            direct_best_idx <- which.min(direct_values)
            direct_best_solution <- matrix(
                as.numeric(direct_solutions[direct_best_idx, ]),
                nrow = 1
            )
            direct_best_eval <- direct_eval(direct_best_solution)
            direct_best_generation <- tryCatch(
                direct_ga_model@iter,
                error = function(e) NA_integer_
            )
            """
        )

        best_x = np.asarray(r("direct_best_solution"), dtype=float).reshape(1, -1)
        best_eval = np.asarray(r("direct_best_eval"), dtype=float).ravel()
        best_generation = np.asarray(r("direct_best_generation"), dtype=float).ravel()

    best_x = best_x.ravel()
    pred_mean = float(best_eval[0])
    pred_var = float(max(best_eval[1], EPS))
    pred_quality_loss = float(best_eval[2])
    true_response = float(non_test_function(best_x.reshape(1, -1))[0])
    true_target_error = float((true_response - TARGET_VALUE) ** 2)

    result = {
        "ga_engine": "R_GA_direct_btgp",
        "ga_param_source": "wing_weight_10d/config.py",
        "ga_pop_size": int(GA_POP_SIZE),
        "ga_generations": int(GA_GENERATIONS),
        "ga_stall_generations": int(GA_STALL_GENERATIONS),
        "ga_search_quality_loss": pred_quality_loss,
        "ga_search_mean": pred_mean,
        "ga_search_variance": pred_var,
        "ga_pred_quality_loss": pred_quality_loss,
        "ga_pred_mean": pred_mean,
        "ga_pred_variance": pred_var,
        "ga_true_response": true_response,
        "ga_true_target_error": true_target_error,
        "ga_true_quality_loss": true_target_error,
        "ga_best_generation": (
            int(best_generation[0]) if best_generation.size and np.isfinite(best_generation[0]) else np.nan
        ),
    }
    for idx, value in enumerate(best_x, start=1):
        result[f"ga_x{idx}"] = float(value)
    return result


def run_task(task):
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(
        prefix=(
            f"tgp_direct_ga_{CASE_NAME}_{task['method']}_{task['n_initial']}_"
            f"{task['repeat']}_{task['ga_repeat']}_"
        )
    ) as tmp_dir:
        os.chdir(tmp_dir)
        robjects.r.assign("tgp_worker_dir", tmp_dir)
        robjects.r("setwd(tgp_worker_dir)")
        try:
            train_x, train_y = load_training_set(task["dataset_path"])
            metrics = reference_metrics(
                train_x,
                train_y,
                task["test_x"],
                task["test_y"],
            )
            direct_metrics = direct_btgp_ga_optimize(train_x, train_y)
            direct_metrics.update(
                {
                    "ga_self_pred_quality_loss": direct_metrics["ga_pred_quality_loss"],
                    "ga_self_pred_mean": direct_metrics["ga_pred_mean"],
                    "ga_self_pred_variance": direct_metrics["ga_pred_variance"],
                }
            )
            metrics.update(direct_metrics)
        finally:
            os.chdir(original_cwd)
            robjects.r.assign("tgp_worker_dir", str(original_cwd))
            robjects.r("setwd(tgp_worker_dir)")

    result = {
        "case": CASE_NAME,
        "method": task["method"],
        "n_initial": task["n_initial"],
        "n_added": task["n_added"],
        "n_total": task["n_total"],
        "repeat": task["repeat"],
        "seed": task["seed"],
        "ga_repeat": task["ga_repeat"],
        "dataset_file": task["dataset_file"],
        "dataset_path": task["dataset_path"],
        "pm_dataset_path": task["pm_dataset_path"],
    }
    result.update(metrics)
    return result


def run_tasks(tasks, workers, result_file):
    if int(workers) <= 1:
        for task in tasks:
            row = run_task(task)
            save_result_row(result_file, row)
            print(
                f"Saved direct-btgp GA {CASE_NAME}: method={task['method']}, "
                f"repeat={task['repeat']}, ga_repeat={task['ga_repeat']}, "
                f"self_QL={row['ga_self_pred_quality_loss']:.4f}, "
                f"true_y={row['ga_true_response']:.4f}"
            )
        return

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            row = future.result()
            save_result_row(result_file, row)
            print(
                f"Saved direct-btgp GA {CASE_NAME}: method={task['method']}, "
                f"repeat={task['repeat']}, ga_repeat={task['ga_repeat']}, "
                f"self_QL={row['ga_self_pred_quality_loss']:.4f}, "
                f"true_y={row['ga_true_response']:.4f}"
            )


def preserve_self_model_columns(data):
    mappings = {
        "ga_self_pred_quality_loss": "ga_pred_quality_loss",
        "ga_self_pred_mean": "ga_pred_mean",
        "ga_self_pred_variance": "ga_pred_variance",
    }
    for target, source in mappings.items():
        if target not in data.columns and source in data.columns:
            data[target] = data[source]
    return data


def evaluate_pm_group(task, seed):
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=f"pm_eval_{CASE_NAME}_{task['label']}_") as tmp_dir:
        os.chdir(tmp_dir)
        robjects.r.assign("tgp_worker_dir", tmp_dir)
        robjects.r("setwd(tgp_worker_dir)")
        try:
            pm_x, pm_y = load_training_set(task["pm_dataset_path"])
            pm_metrics = pm_frame_evaluate_ga_points(task["ga_x"], pm_x, pm_y, seed)
        finally:
            os.chdir(original_cwd)
            robjects.r.assign("tgp_worker_dir", str(original_cwd))
            robjects.r("setwd(tgp_worker_dir)")
    pm_metrics["indices"] = task["indices"]
    pm_metrics["pm_dataset_path"] = task["pm_dataset_path"]
    return pm_metrics


def apply_grouped_pm_framework(result_file, summary_file, pm_eval_seed):
    result_path = Path(result_file)
    if not result_path.exists():
        print(f"结果文件不存在: {result_path}")
        return

    data = pd.read_excel(result_path)
    if data.empty:
        print(f"{CASE_NAME}: result file is empty, skip PM grouped evaluation.")
        summarize_results(result_path, summary_file)
        return

    data = preserve_self_model_columns(data.copy())
    x_columns = [f"ga_x{idx}" for idx in range(1, len(FEATURE_NAMES) + 1)]
    grouped = {}
    for row in data.itertuples(index=True):
        pm_key = (int(row.n_initial), int(row.repeat))
        if not hasattr(row, "pm_dataset_path") or pd.isna(row.pm_dataset_path):
            raise ValueError(
                f"缺少 PM 评价训练集路径: n_initial={pm_key[0]}, repeat={pm_key[1]}"
            )
        grouped.setdefault(pm_key, {"pm_dataset_path": row.pm_dataset_path, "indices": []})
        grouped[pm_key]["indices"].append(row.Index)

    data["pm_dataset_path"] = data["pm_dataset_path"].astype(str)
    data["pm_eval_mean"] = np.nan
    data["pm_eval_variance"] = np.nan
    data["pm_eval_QL"] = np.nan

    tasks = []
    for position, (pm_key, group) in enumerate(sorted(grouped.items()), start=1):
        indices = group["indices"]
        tasks.append(
            {
                "position": position,
                "label": f"n{pm_key[0]}_r{pm_key[1]}",
                "indices": indices,
                "pm_dataset_path": str(group["pm_dataset_path"]),
                "ga_x": data.loc[indices, x_columns].to_numpy(dtype=float),
            }
        )

    print(
        f"{CASE_NAME}: PM-framework grouped evaluation for {len(data)} GA rows "
        f"using {len(tasks)} btgp fit(s), R seed={int(pm_eval_seed)}."
    )
    for task in tasks:
        print(
            f"{CASE_NAME}: PM-evaluate group {task['position']}/{len(tasks)} "
            f"({task['label']}), rows={len(task['indices'])}",
            flush=True,
        )
        result = evaluate_pm_group(task, pm_eval_seed)
        indices = result["indices"]
        data.loc[indices, "pm_dataset_path"] = result["pm_dataset_path"]
        data.loc[indices, "pm_eval_mean"] = result["pm_eval_mean"]
        data.loc[indices, "pm_eval_variance"] = result["pm_eval_variance"]
        data.loc[indices, "pm_eval_QL"] = result["pm_eval_QL"]

    data["ga_pred_quality_loss"] = data["pm_eval_QL"]
    data["ga_pred_mean"] = data["pm_eval_mean"]
    data["ga_pred_variance"] = data["pm_eval_variance"]
    data["ga_true_quality_loss_with_pred_variance"] = (
        data["ga_true_target_error"].astype(float)
        + data["pm_eval_variance"].astype(float)
    )
    data["pm_eval_seed"] = int(pm_eval_seed)
    data["pm_eval_mode"] = "grouped_by_pm_dataset"
    data = data.sort_values(["method", "n_initial", "repeat", "ga_repeat"]).reset_index(drop=True)
    data.to_excel(result_path, index=False)
    summarize_results(result_path, summary_file)

    duplicate_x = int(data.duplicated(x_columns).sum())
    print(f"{CASE_NAME}: formal PM-framework result saved to {result_path}")
    print(f"{CASE_NAME}: rows={len(data)}, duplicate_ga_x_rows={duplicate_x}")
    print(f"{CASE_NAME}: ga_pred_quality_loss equals pm_eval_QL.")


def main():
    args = parse_args()
    dataset_index = load_dataset_index(args.methods)
    shared_data = load_shared_data(DATA_FILE)
    done = set() if args.force else completed_keys(args.output)
    tasks = build_tasks(
        done,
        dataset_index,
        shared_data,
        ga_repeats=args.ga_repeats,
        force=args.force,
    )
    if not tasks:
        print(f"{CASE_NAME}: no pending direct-btgp GA tasks.")
        apply_grouped_pm_framework(args.output, args.summary, args.pm_eval_seed)
        return

    methods = sorted(dataset_index["method"].unique())
    print(
        f"{CASE_NAME}: run {len(tasks)} archive-style direct-btgp GA tasks "
        f"with {int(args.workers)} worker(s); methods={methods}; "
        f"ga_repeats={int(args.ga_repeats)}; "
        f"pop_size={GA_POP_SIZE}; generations={GA_GENERATIONS}; "
        f"stall_generations={GA_STALL_GENERATIONS}; output={args.output}"
    )
    run_tasks(tasks, args.workers, args.output)
    apply_grouped_pm_framework(args.output, args.summary, args.pm_eval_seed)
    print(f"{CASE_NAME}: direct-btgp GA results saved to {args.output}")


if __name__ == "__main__":
    main()
