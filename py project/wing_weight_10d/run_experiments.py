import importlib
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
    METHOD_MODULES,
    REPEAT_SEED_START,
    RUN_METHODS,
    RUN_N_INITIAL_VALUES,
    RUN_REPEATS,
    WORKERS,
)
from core import (
    get_split,
    load_shared_data,
    save_training_set,
)
from generate_data import main as generate_data


DATASET_INDEX_COLUMNS = (
    "case",
    "method",
    "n_initial",
    "n_added",
    "n_total",
    "repeat",
    "seed",
    "dataset_file",
    "dataset_path",
)


def build_tasks(shared_data):
    tasks = []
    for n_initial in RUN_N_INITIAL_VALUES:
        for repeat in range(1, RUN_REPEATS + 1):
            split = get_split(shared_data, n_initial, repeat)
            for method in RUN_METHODS:
                if method not in METHOD_MODULES:
                    raise ValueError(f"Unknown method in RUN_METHODS: {method}")
                tasks.append((method, METHOD_MODULES[method], n_initial, repeat, split))
    return tasks


def write_dataset_index(rows):
    DATASET_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    if data.empty:
        data = pd.DataFrame(columns=DATASET_INDEX_COLUMNS)
    if not data.empty:
        data = data.sort_values(["method", "n_initial", "repeat"])
    data.to_excel(DATASET_INDEX_FILE, index=False)


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
    tasks = build_tasks(shared_data)
    if not tasks:
        print(f"{CASE_NAME}: no dataset-generation tasks.")
        write_dataset_index([])
        return

    print(f"{CASE_NAME}: regenerate {len(tasks)} training sets with {WORKERS} workers")
    context = mp.get_context("spawn")
    rows = []
    write_dataset_index(rows)
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            method, _, n_initial, repeat, _ = futures[future]
            row = future.result()
            rows.append(row)
            write_dataset_index(rows)
            print(
                f"Saved training set {CASE_NAME}: method={method}, "
                f"n_initial={n_initial}, repeat={repeat}/{RUN_REPEATS}, "
                f"n_total={row['n_total']}, file={row['dataset_file']}"
            )

    print(f"{CASE_NAME}: selected dataset index saved to {DATASET_INDEX_FILE}")


if __name__ == "__main__":
    main()
