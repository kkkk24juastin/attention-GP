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
    DATA_FILE,
    GA_RESULT_FILE,
    REPEAT_SEED_START,
    SUMMARY_FILE,
    WORKERS,
)
from core import (
    completed_keys,
    evaluate_model,
    frame_to_xy,
    load_shared_data,
    load_training_set,
    save_result_row,
    summarize_results,
)


def load_dataset_index():
    if not DATASET_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"训练集索引不存在: {DATASET_INDEX_FILE}，请先运行 run_experiments.py"
        )
    return pd.read_excel(DATASET_INDEX_FILE)


def build_tasks(done, dataset_index, shared_data):
    test_x, test_y = frame_to_xy(shared_data["test"])
    tasks = []
    for row in dataset_index.itertuples(index=False):
        key = (row.method, int(row.n_initial), int(row.repeat))
        if key in done:
            continue
        dataset_path = Path(row.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"训练集文件不存在: {dataset_path}")
        tasks.append(
            {
                "method": row.method,
                "n_initial": int(row.n_initial),
                "n_added": int(row.n_added),
                "n_total": int(row.n_total),
                "repeat": int(row.repeat),
                "seed": int(row.seed),
                "dataset_file": row.dataset_file,
                "dataset_path": str(dataset_path),
                "test_x": test_x,
                "test_y": test_y,
            }
        )
    return tasks


def run_task(task):
    seed = int(task.get("seed", REPEAT_SEED_START + int(task["repeat"]) - 1))
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(
        prefix=f"tgp_ga_{CASE_NAME}_{task['method']}_{task['n_initial']}_{task['repeat']}_"
    ) as tmp_dir:
        os.chdir(tmp_dir)
        robjects.r.assign("tgp_worker_dir", tmp_dir)
        robjects.r("setwd(tgp_worker_dir)")
        robjects.r(f"set.seed({seed})")
        try:
            train_x, train_y = load_training_set(task["dataset_path"])
            metrics = evaluate_model(
                train_x,
                train_y,
                task["test_x"],
                task["test_y"],
                seed=seed,
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
        "seed": seed,
        "dataset_file": task["dataset_file"],
        "dataset_path": task["dataset_path"],
    }
    result.update(metrics)
    return result


def main():
    dataset_index = load_dataset_index()
    shared_data = load_shared_data(DATA_FILE)
    done = completed_keys(GA_RESULT_FILE)
    tasks = build_tasks(done, dataset_index, shared_data)
    if not tasks:
        print(f"{CASE_NAME}: no pending GA optimization tasks.")
        summarize_results(GA_RESULT_FILE, SUMMARY_FILE)
        return

    total_methods = dataset_index["method"].nunique()
    repeats_per_method = dataset_index.groupby("method")["repeat"].nunique().to_dict()
    print(
        f"{CASE_NAME}: run {len(tasks)} pending GA optimization tasks with {WORKERS} workers "
        f"({total_methods} methods; repeats per method: {repeats_per_method})"
    )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            row = future.result()
            save_result_row(GA_RESULT_FILE, row)
            print(
                f"Saved GA {CASE_NAME}: method={task['method']}, "
                f"repeat={task['repeat']}, QL={row['ga_pred_quality_loss']:.4f}, "
                f"true_QL={row['ga_true_quality_loss']:.4f}, "
                f"true_y={row['ga_true_response']:.4f}"
            )

    summarize_results(GA_RESULT_FILE, SUMMARY_FILE)
    print(f"{CASE_NAME}: GA results saved to {GA_RESULT_FILE}")


if __name__ == "__main__":
    main()
