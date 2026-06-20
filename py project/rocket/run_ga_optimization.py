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
    DATASET_INDEX_FILE,
    DATASET_DIR,
    FEATURE_NAMES,
    GA_CANDIDATE_FILE,
    GA_GENERATIONS,
    GA_POP_SIZE,
    GA_REPEATS,
    GA_STALL_GENERATIONS,
    LOWER_BOUNDS,
    NOSE_SHAPE_PARAMETER_COLUMN,
    OPENROCKET_TEMPLATE_FILE,
    PM_EVAL_SEED,
    RAW_FEATURE_COLUMNS,
    SUMMARY_FILE,
    TARGET_VALUE,
    UPPER_BOUNDS,
    WORKERS,
    nose_shape_parameter_from_length,
)
from core import (
    EPS,
    NUMPY_CONVERTER,
    load_training_set,
    quality_loss_scores,
    summarize_results,
    tgp_predict,
)


GA_CANDIDATE_COLUMNS = (
    "case",
    "method",
    "n_initial",
    "n_added",
    "n_total",
    "repeat",
    "seed",
    "ga_repeat",
    "dataset_file",
    "dataset_path",
    "ga_engine",
    "ga_param_source",
    "ga_pop_size",
    "ga_generations",
    "ga_stall_generations",
    "ga_search_quality_loss",
    "ga_search_mean",
    "ga_search_variance",
    "ga_self_pred_quality_loss",
    "ga_self_pred_mean",
    "ga_self_pred_variance",
    "ga_pred_quality_loss",
    "ga_pred_mean",
    "ga_pred_variance",
    "pm_framework_dataset_file",
    "pm_framework_dataset_path",
    "pm_framework_pred_mean",
    "pm_framework_pred_variance",
    "QL_value",
    "pm_eval_QL",
    "pm_eval_mean",
    "pm_eval_variance",
    "pm_eval_seed",
    "pm_eval_mode",
    "ga_best_generation",
    "openrocket_status",
    *(f"ga_x{idx}" for idx in range(1, len(FEATURE_NAMES) + 1)),
    NOSE_SHAPE_PARAMETER_COLUMN,
    *RAW_FEATURE_COLUMNS,
)


def load_dataset_index():
    if not DATASET_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"训练集索引不存在: {DATASET_INDEX_FILE}，请先运行 run_experiments.py"
        )
    return pd.read_excel(DATASET_INDEX_FILE)


def resolve_dataset_path(dataset_file, dataset_path=None):
    candidates = []
    if dataset_path and not pd.isna(dataset_path):
        candidates.append(Path(str(dataset_path)))
    candidates.append(DATASET_DIR / str(dataset_file))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"训练集文件不存在: {dataset_file}; checked={candidates}"
    )


def build_pm_framework_lookup(dataset_index):
    pm_rows = dataset_index[dataset_index["method"] == "PM"]
    if pm_rows.empty:
        raise ValueError("PM 框架训练集不存在。请先在 run_experiments.py 中生成 PM 方法训练集。")
    lookup = {}
    for row in pm_rows.itertuples(index=False):
        key = (int(row.n_initial), int(row.repeat))
        pm_path = resolve_dataset_path(row.dataset_file, row.dataset_path)
        lookup[key] = {
            "pm_framework_dataset_file": row.dataset_file,
            "pm_framework_dataset_path": str(pm_path),
        }
    return lookup


def build_tasks(dataset_index):
    tasks = []
    pm_lookup = build_pm_framework_lookup(dataset_index)
    for row in dataset_index.itertuples(index=False):
        dataset_path = resolve_dataset_path(row.dataset_file, row.dataset_path)
        pm_key = (int(row.n_initial), int(row.repeat))
        if pm_key not in pm_lookup:
            raise FileNotFoundError(
                f"缺少 n_initial={row.n_initial}, repeat={row.repeat} 的 PM 框架训练集"
            )
        for ga_repeat in range(1, GA_REPEATS + 1):
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
                    **pm_lookup[pm_key],
                }
            )
    return tasks


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
                monitor = FALSE,
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
    result = {
        "ga_engine": "R_GA_direct_btgp",
        "ga_param_source": "rocket/config.py",
        "ga_pop_size": int(GA_POP_SIZE),
        "ga_generations": int(GA_GENERATIONS),
        "ga_stall_generations": int(GA_STALL_GENERATIONS),
        "ga_search_quality_loss": pred_quality_loss,
        "ga_search_mean": pred_mean,
        "ga_search_variance": pred_var,
        "ga_self_pred_quality_loss": pred_quality_loss,
        "ga_self_pred_mean": pred_mean,
        "ga_self_pred_variance": pred_var,
        "ga_pred_quality_loss": pred_quality_loss,
        "ga_pred_mean": pred_mean,
        "ga_pred_variance": pred_var,
        "ga_best_generation": (
            int(best_generation[0])
            if best_generation.size and np.isfinite(best_generation[0])
            else np.nan
        ),
        "openrocket_status": "pending",
    }
    for idx, value in enumerate(best_x, start=1):
        result[f"ga_x{idx}"] = float(value)
    result[NOSE_SHAPE_PARAMETER_COLUMN] = nose_shape_parameter_from_length(best_x[0])
    for raw_name, value in zip(RAW_FEATURE_COLUMNS, best_x):
        result[raw_name] = float(value)
    return result


def run_task(task):
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(
        prefix=(
            f"tgp_ga_{CASE_NAME}_{task['method']}_{task['n_initial']}_"
            f"{task['repeat']}_{task['ga_repeat']}_"
        )
    ) as tmp_dir:
        os.chdir(tmp_dir)
        robjects.r.assign("tgp_worker_dir", tmp_dir)
        robjects.r("setwd(tgp_worker_dir)")
        try:
            train_x, train_y = load_training_set(task["dataset_path"])
            metrics = direct_btgp_ga_optimize(train_x, train_y)
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
        "pm_framework_dataset_file": task["pm_framework_dataset_file"],
        "pm_framework_dataset_path": task["pm_framework_dataset_path"],
    }
    result.update(metrics)
    return result


def reorder_candidate_columns(data):
    ordered = [column for column in GA_CANDIDATE_COLUMNS if column in data.columns]
    remaining = [column for column in data.columns if column not in ordered]
    return data[ordered + remaining]


def write_candidate_rows(rows):
    GA_CANDIDATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    if data.empty:
        data = pd.DataFrame(columns=GA_CANDIDATE_COLUMNS)
    if not data.empty:
        data = data.sort_values(["method", "n_initial", "repeat", "ga_repeat"])
        data = reorder_candidate_columns(data)
    data.to_excel(GA_CANDIDATE_FILE, index=False)


def write_openrocket_template(candidate_file=GA_CANDIDATE_FILE):
    if not candidate_file.exists():
        return
    candidates = pd.read_excel(candidate_file)
    template = candidates.copy()
    for column in (
        "最大飞行高度",
        "QL_value",
        "flight_time",
        "time_to_apogee",
        "max_velocity",
        "max_acceleration",
        "simulation_note",
    ):
        if column not in template.columns:
            template[column] = pd.NA
    OPENROCKET_TEMPLATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    template.to_excel(OPENROCKET_TEMPLATE_FILE, index=False)


def evaluate_pm_group(task, pm_eval_seed):
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=f"pm_eval_{CASE_NAME}_{task['label']}_") as tmp_dir:
        os.chdir(tmp_dir)
        robjects.r.assign("tgp_worker_dir", tmp_dir)
        robjects.r("setwd(tgp_worker_dir)")
        try:
            robjects.r(f"set.seed({int(pm_eval_seed)})")
            pm_x, pm_y = load_training_set(task["pm_framework_dataset_path"])
            pm_mean, pm_variance = tgp_predict(
                pm_x,
                pm_y,
                np.asarray(task["ga_x"], dtype=float),
            )
            pm_mean = np.asarray(pm_mean, dtype=float).ravel()
            pm_variance = np.maximum(np.asarray(pm_variance, dtype=float).ravel(), EPS)
            pm_ql = quality_loss_scores(pm_mean, pm_variance)
        finally:
            os.chdir(original_cwd)
            robjects.r.assign("tgp_worker_dir", str(original_cwd))
            robjects.r("setwd(tgp_worker_dir)")
    return {
        "indices": task["indices"],
        "pm_framework_dataset_path": task["pm_framework_dataset_path"],
        "pm_eval_mean": pm_mean,
        "pm_eval_variance": pm_variance,
        "pm_eval_QL": pm_ql,
    }


def apply_grouped_pm_framework(candidate_file=GA_CANDIDATE_FILE, pm_eval_seed=PM_EVAL_SEED):
    candidate_path = Path(candidate_file)
    if not candidate_path.exists():
        print(f"GA 候选文件不存在: {candidate_path}")
        return

    data = pd.read_excel(candidate_path)
    if data.empty:
        summarize_results(candidate_path, SUMMARY_FILE)
        return

    x_columns = [f"ga_x{idx}" for idx in range(1, len(FEATURE_NAMES) + 1)]
    grouped = {}
    for row in data.itertuples(index=True):
        pm_key = (int(row.n_initial), int(row.repeat))
        if not hasattr(row, "pm_framework_dataset_path") or pd.isna(row.pm_framework_dataset_path):
            raise ValueError(
                f"缺少 PM 框架训练集路径: n_initial={pm_key[0]}, repeat={pm_key[1]}"
            )
        grouped.setdefault(
            pm_key,
            {
                "pm_framework_dataset_file": row.pm_framework_dataset_file,
                "pm_framework_dataset_path": row.pm_framework_dataset_path,
                "indices": [],
            },
        )
        grouped[pm_key]["indices"].append(row.Index)

    tasks = []
    for position, (pm_key, group) in enumerate(sorted(grouped.items()), start=1):
        indices = group["indices"]
        tasks.append(
            {
                "position": position,
                "label": f"n{pm_key[0]}_r{pm_key[1]}",
                "indices": indices,
                "pm_framework_dataset_path": str(group["pm_framework_dataset_path"]),
                "ga_x": data.loc[indices, x_columns].to_numpy(dtype=float),
            }
        )

    print(
        f"{CASE_NAME}: PM-framework grouped evaluation for {len(data)} GA rows "
        f"using {len(tasks)} btgp fit(s), R seed={int(pm_eval_seed)}."
    )
    data["pm_eval_mean"] = np.nan
    data["pm_eval_variance"] = np.nan
    data["pm_eval_QL"] = np.nan
    for task in tasks:
        print(
            f"{CASE_NAME}: PM-evaluate group {task['position']}/{len(tasks)} "
            f"({task['label']}), rows={len(task['indices'])}",
            flush=True,
        )
        result = evaluate_pm_group(task, pm_eval_seed)
        indices = result["indices"]
        data.loc[indices, "pm_framework_dataset_path"] = result["pm_framework_dataset_path"]
        data.loc[indices, "pm_eval_mean"] = result["pm_eval_mean"]
        data.loc[indices, "pm_eval_variance"] = result["pm_eval_variance"]
        data.loc[indices, "pm_eval_QL"] = result["pm_eval_QL"]

    data["pm_framework_pred_mean"] = data["pm_eval_mean"]
    data["pm_framework_pred_variance"] = data["pm_eval_variance"]
    data["QL_value"] = data["pm_eval_QL"]
    data["ga_pred_quality_loss"] = data["pm_eval_QL"]
    data["ga_pred_mean"] = data["pm_eval_mean"]
    data["ga_pred_variance"] = data["pm_eval_variance"]
    data["pm_eval_seed"] = int(pm_eval_seed)
    data["pm_eval_mode"] = "grouped_by_pm_dataset"
    data = data.sort_values(["method", "n_initial", "repeat", "ga_repeat"]).reset_index(drop=True)
    data = reorder_candidate_columns(data)
    data.to_excel(candidate_path, index=False)
    summarize_results(candidate_path, SUMMARY_FILE)

    duplicate_x = int(data.duplicated(x_columns).sum())
    print(f"{CASE_NAME}: PM-framework GA candidates saved to {candidate_path}")
    print(f"{CASE_NAME}: rows={len(data)}, duplicate_ga_x_rows={duplicate_x}")
    print(f"{CASE_NAME}: ga_pred_quality_loss and QL_value now equal pm_eval_QL.")


def main():
    dataset_index = load_dataset_index()
    tasks = build_tasks(dataset_index)
    if not tasks:
        print(f"{CASE_NAME}: no surrogate GA candidate tasks.")
        write_candidate_rows([])
        return

    total_methods = dataset_index["method"].nunique()
    repeats_per_method = dataset_index.groupby("method")["repeat"].nunique().to_dict()
    print(
        f"{CASE_NAME}: regenerate {len(tasks)} surrogate GA tasks with {WORKERS} workers "
        f"({total_methods} methods; active-learning repeats per method: {repeats_per_method}; "
        f"GA repeats per dataset: {GA_REPEATS}; PM eval seed: {PM_EVAL_SEED})"
    )
    context = mp.get_context("spawn")
    rows = []
    write_candidate_rows(rows)
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            row = future.result()
            rows.append(row)
            write_candidate_rows(rows)
            print(
                f"Saved GA candidate {CASE_NAME}: method={task['method']}, "
                f"n_initial={task['n_initial']}, repeat={task['repeat']}, "
                f"ga_repeat={task['ga_repeat']}/{GA_REPEATS}, "
                f"self_QL={row['ga_self_pred_quality_loss']:.4f}, "
                f"self_MH={row['ga_self_pred_mean']:.4f}"
            )

    apply_grouped_pm_framework(GA_CANDIDATE_FILE, PM_EVAL_SEED)
    write_openrocket_template()
    print(f"{CASE_NAME}: GA candidates saved to {GA_CANDIDATE_FILE}")
    print(f"{CASE_NAME}: OpenRocket simulation template saved to {OPENROCKET_TEMPLATE_FILE}")


if __name__ == "__main__":
    main()
