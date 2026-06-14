import importlib
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import rpy2.robjects as robjects

from config import (
    CASE_NAME,
    DATASET_INDEX_FILE,
    DATA_FILE,
    METHOD_MODULES,
    REPEAT_SEED_START,
    RUN_METHODS,
    RUN_N_INITIAL_VALUES,
    RUN_REPEATS,
    WORKERS,
)
from core import (
    completed_dataset_keys,
    get_split,
    load_shared_data,
    save_dataset_index_row,
    save_training_set,
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
    initial_x, initial_y, pool_x, pool_y, _test_x, _test_y = split
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
                initial_x, initial_y, pool_x, pool_y, seed=seed
            )
        finally:
            os.chdir(original_cwd)
            robjects.r.assign("tgp_worker_dir", str(original_cwd))
            robjects.r("setwd(tgp_worker_dir)")

    return save_training_set(method, n_initial, repeat, seed, final_x, final_y)


def main():
    generate_data()
    shared_data = load_shared_data(DATA_FILE)
    done = completed_dataset_keys(DATASET_INDEX_FILE)
    tasks = build_tasks(done, shared_data)
    if not tasks:
        print(f"{CASE_NAME}: no pending dataset-generation tasks.")
        return

    print(f"{CASE_NAME}: generate {len(tasks)} pending training sets with {WORKERS} workers")
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            method, _, n_initial, repeat, _ = futures[future]
            row = future.result()
            save_dataset_index_row(DATASET_INDEX_FILE, row)
            print(
                f"Saved training set {CASE_NAME}: method={method}, "
                f"n_initial={n_initial}, repeat={repeat}/{RUN_REPEATS}, "
                f"n_total={row['n_total']}, file={row['dataset_file']}"
            )

    print(f"{CASE_NAME}: selected dataset index saved to {DATASET_INDEX_FILE}")


if __name__ == "__main__":
    main()
