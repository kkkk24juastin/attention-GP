import importlib
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import rpy2.robjects as robjects

from config import (
    CASE_NAME,
    DATA_FILE,
    METHOD_MODULES,
    N_ADDED,
    REPEAT_SEED_START,
    RESET_RESULTS_ON_RUN,
    RESULT_FILE,
    RUN_METHODS,
    RUN_N_INITIAL_VALUES,
    RUN_REPEATS,
    SUMMARY_FILE,
    TRAINING_SET_INDEX_FILE,
    WORKERS,
)
from core import (
    completed_keys,
    evaluate_model,
    get_split,
    load_shared_data,
    save_result_row,
    save_training_set,
    save_training_set_index_row,
    summarize_results,
)
from generate_data import main as generate_data


def build_tasks(done, shared_data):
    tasks = []
    for n_initial in RUN_N_INITIAL_VALUES:
        for repeat in range(1, RUN_REPEATS + 1):
            split = get_split(shared_data, n_initial, repeat)
            for method in RUN_METHODS:
                if method not in METHOD_MODULES:
                    raise ValueError(f"Unknown method in RUN_METHODS: {method}")
                key = (method, n_initial, repeat)
                if key not in done:
                    tasks.append((method, METHOD_MODULES[method], n_initial, repeat, split))
    return tasks


def run_task(task):
    method, module_name, n_initial, repeat, split = task
    initial_x, initial_y, pool_x, pool_y, test_x, test_y = split
    seed = REPEAT_SEED_START + int(repeat) - 1

    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=f"tgp_{CASE_NAME}_{method}_{n_initial}_{repeat}_") as tmp_dir:
        os.chdir(tmp_dir)
        robjects.r.assign("tgp_worker_dir", tmp_dir)
        robjects.r("setwd(tgp_worker_dir)")
        robjects.r(f"set.seed({int(seed)})")
        try:
            method_module = importlib.import_module(module_name)
            final_x, final_y = method_module.run(
                initial_x, initial_y, pool_x, pool_y, seed=seed, repeat=repeat
            )
            metrics = evaluate_model(final_x, final_y, test_x, test_y)
            training_set_row = save_training_set(
                method, n_initial, repeat, seed, final_x, final_y
            )
        finally:
            os.chdir(original_cwd)
            robjects.r.assign("tgp_worker_dir", str(original_cwd))
            robjects.r("setwd(tgp_worker_dir)")

    row = {
        "case": CASE_NAME,
        "method": method,
        "n_initial": n_initial,
        "n_added": N_ADDED,
        "repeat": repeat,
        "seed": seed,
    }
    row.update(metrics)
    row.update(
        {
            "training_set_file": training_set_row["training_set_file"],
            "training_set_path": training_set_row["training_set_path"],
        }
    )
    return row, training_set_row


def reset_results_if_requested():
    if RESET_RESULTS_ON_RUN and RESULT_FILE.exists():
        RESULT_FILE.unlink()
    if RESET_RESULTS_ON_RUN and SUMMARY_FILE.exists():
        SUMMARY_FILE.unlink()


def main():
    generate_data()
    reset_results_if_requested()
    shared_data = load_shared_data(DATA_FILE)
    done = completed_keys(RESULT_FILE)
    tasks = build_tasks(done, shared_data)
    if not tasks:
        print(f"{CASE_NAME}: no pending tasks.")
        summarize_results(RESULT_FILE, SUMMARY_FILE)
        return

    print(f"{CASE_NAME}: run {len(tasks)} pending tasks with {WORKERS} workers")
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            method, _, n_initial, repeat, _ = futures[future]
            row, training_set_row = future.result()
            save_result_row(RESULT_FILE, row)
            save_training_set_index_row(TRAINING_SET_INDEX_FILE, training_set_row)
            print(
                f"Saved {CASE_NAME}: method={method}, n_initial={n_initial}, "
                f"repeat={repeat}/{RUN_REPEATS}, RMSE_all={row['RMSE_all']:.4f}, "
                f"RMSE_target={row['RMSE_target']:.4f}, MAE_all={row['MAE_all']:.4f}, "
                f"training_set={training_set_row['training_set_file']}"
            )

    summarize_results(RESULT_FILE, SUMMARY_FILE)
    print(f"{CASE_NAME}: results saved to {RESULT_FILE}")


if __name__ == "__main__":
    main()
