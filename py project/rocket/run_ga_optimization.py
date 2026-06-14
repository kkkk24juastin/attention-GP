import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import rpy2.robjects as robjects

from config import (
    CASE_NAME,
    DATASET_INDEX_FILE,
    FEATURE_NAMES,
    GA_CANDIDATE_FILE,
    GA_REPEATS,
    GA_SEED_START,
    OPENROCKET_TEMPLATE_FILE,
    RAW_FEATURE_COLUMNS,
    SUMMARY_FILE,
    WORKERS,
)
from core import (
    evaluate_surrogate_optimization,
    load_training_set,
    summarize_results,
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
    "ga_seed",
    "dataset_file",
    "dataset_path",
    "ga_pred_quality_loss",
    "ga_pred_mean",
    "ga_pred_variance",
    "ga_best_generation",
    "openrocket_status",
    *(f"ga_x{idx}" for idx in range(1, len(FEATURE_NAMES) + 1)),
    *RAW_FEATURE_COLUMNS,
)


def load_dataset_index():
    if not DATASET_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"训练集索引不存在: {DATASET_INDEX_FILE}，请先运行 run_experiments.py"
        )
    return pd.read_excel(DATASET_INDEX_FILE)


def build_tasks(dataset_index):
    tasks = []
    for row in dataset_index.itertuples(index=False):
        dataset_path = Path(row.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"训练集文件不存在: {dataset_path}")
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
                    "ga_seed": int(GA_SEED_START + ga_repeat - 1),
                    "dataset_file": row.dataset_file,
                    "dataset_path": str(dataset_path),
                }
            )
    return tasks


def run_task(task):
    ga_seed = int(task["ga_seed"])
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
        robjects.r(f"set.seed({ga_seed})")
        try:
            train_x, train_y = load_training_set(task["dataset_path"])
            metrics = evaluate_surrogate_optimization(
                train_x,
                train_y,
                seed=ga_seed,
            )
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
        "ga_seed": ga_seed,
        "dataset_file": task["dataset_file"],
        "dataset_path": task["dataset_path"],
    }
    result.update(metrics)
    return result


def write_candidate_rows(rows):
    GA_CANDIDATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    if data.empty:
        data = pd.DataFrame(columns=GA_CANDIDATE_COLUMNS)
    if not data.empty:
        data = data.sort_values(["method", "n_initial", "repeat", "ga_repeat"])
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
        f"GA repeats per dataset: {GA_REPEATS})"
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
                f"pred_QL={row['ga_pred_quality_loss']:.4f}, "
                f"pred_MH={row['ga_pred_mean']:.4f}"
            )

    summarize_results(GA_CANDIDATE_FILE, SUMMARY_FILE)
    write_openrocket_template()
    print(f"{CASE_NAME}: GA candidates saved to {GA_CANDIDATE_FILE}")
    print(f"{CASE_NAME}: OpenRocket simulation template saved to {OPENROCKET_TEMPLATE_FILE}")


if __name__ == "__main__":
    main()
