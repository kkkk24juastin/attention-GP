import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import default_converter, numpy2ri
from scipy.stats import qmc

from config import (
    CASE_NAME,
    DATA_FILE,
    LOWER_BOUNDS,
    SUMMARY_FILE,
    TARGET_BAND,
    TARGET_VALUE,
    UPPER_BOUNDS,
)


NUMPY_CONVERTER = default_converter + numpy2ri.converter
_TGP_LOADED = False


def non_test_function(X):
    x, y = X[:, 0], X[:, 1]
    condition = x > 0
    return np.where(
        condition,
        (2 + 0.5 * y) * np.sin(x) + y**2,
        x**2 + y**2 + (8 - (x**2 + y**2)) * (1 - np.exp(-x**2)),
    )


def generate_candidates(n_samples, seed):
    sampler = qmc.LatinHypercube(d=len(LOWER_BOUNDS), seed=int(seed))
    unit_lhs = sampler.random(n=n_samples)
    return qmc.scale(unit_lhs, LOWER_BOUNDS, UPPER_BOUNDS)


def points_to_frame(X, y, n_initial=None, repeat=None):
    frame = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y})
    if n_initial is not None:
        frame.insert(0, "n_initial", int(n_initial))
    if repeat is not None:
        insert_at = 1 if "n_initial" in frame.columns else 0
        frame.insert(insert_at, "repeat", int(repeat))
    return frame


def frame_to_xy(frame):
    X = frame[["x1", "x2"]].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    return X, y


def load_shared_data(data_file=DATA_FILE):
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"共享数据文件不存在: {data_path}，请先运行 generate_data.py")
    return pd.read_excel(data_path, sheet_name=None)


def get_split(shared_data, n_initial, repeat):
    initial = shared_data["initial"]
    pool = shared_data["pool"]
    initial = initial[
        (initial["n_initial"] == int(n_initial)) & (initial["repeat"] == int(repeat))
    ]
    pool = pool[
        (pool["n_initial"] == int(n_initial)) & (pool["repeat"] == int(repeat))
    ]
    initial_x, initial_y = frame_to_xy(initial)
    pool_x, pool_y = frame_to_xy(pool)
    test_x, test_y = frame_to_xy(shared_data["test"])
    return (
        initial_x,
        initial_y,
        pool_x,
        pool_y,
        test_x,
        test_y,
    )


def ensure_tgp_loaded():
    global _TGP_LOADED
    if not _TGP_LOADED:
        robjects.r("suppressPackageStartupMessages(library(tgp))")
        _TGP_LOADED = True


def tgp_predict(X_train, y_train, X_candidates):
    ensure_tgp_loaded()
    r = robjects.r
    with NUMPY_CONVERTER.context():
        r.assign("X_train", X_train)
        r.assign("y_train", y_train)
        r.assign("X_candidates", X_candidates)
        r("model <- btgp(X_train, y_train, XX = X_candidates, verb = 0)")
        r("pred <- list(mean = model$ZZ.mean, var = model$ZZ.s2)")
        mean = np.asarray(r("pred$mean"), dtype=float).ravel()
        variance = np.asarray(r("pred$var"), dtype=float).ravel()
    return mean, np.maximum(variance, 1e-12)


def evaluate_model(X_train, y_train, test_x, test_y):
    pred_y, _ = tgp_predict(X_train, y_train, test_x)
    rmse_all = np.sqrt(np.mean((pred_y - test_y) ** 2))
    mask = (test_y >= TARGET_VALUE - TARGET_BAND) & (test_y <= TARGET_VALUE + TARGET_BAND)
    rmse_target = np.sqrt(np.mean((pred_y[mask] - test_y[mask]) ** 2)) if mask.any() else np.nan
    return rmse_all, rmse_target


def append_pool_point(X_train, y_train, pool_x, pool_y, selected_idx):
    X_new = pool_x[selected_idx].reshape(1, -1)
    y_new = np.array([pool_y[selected_idx]])
    X_train = np.vstack([X_train, X_new])
    y_train = np.concatenate([y_train, y_new])
    pool_x = np.delete(pool_x, selected_idx, axis=0)
    pool_y = np.delete(pool_y, selected_idx, axis=0)
    return X_train, y_train, pool_x, pool_y


def completed_keys(result_file):
    result_path = Path(result_file)
    if not result_path.exists():
        return set()
    data = pd.read_excel(result_path)
    return {
        (row.method, int(row.n_initial), int(row.repeat))
        for row in data.itertuples(index=False)
    }


def save_result_row(result_file, row):
    result_path = Path(result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if result_path.exists():
        old_rows = pd.read_excel(result_path)
        data = pd.concat([old_rows, new_row], ignore_index=True)
        data = data.drop_duplicates(["method", "n_initial", "repeat"], keep="last")
    else:
        data = new_row
    data = data.sort_values(["method", "n_initial", "repeat"])
    data.to_excel(result_path, index=False)


def summarize_results(result_file, summary_file=SUMMARY_FILE):
    result_path = Path(result_file)
    if not result_path.exists():
        print(f"结果文件不存在: {result_path}")
        return

    data = pd.read_excel(result_path)
    summary_path = Path(summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(summary_path) as writer:
        data.to_excel(writer, sheet_name="raw", index=False)
        counts = (
            data.groupby(["case", "method", "n_initial", "n_added"])
            .size()
            .reset_index(name="n")
            .sort_values(["method", "n_initial"])
        )
        counts.to_excel(writer, sheet_name="counts", index=False)
        for metric in ["RMSE_all", "RMSE_target"]:
            summary = (
                data.groupby(["case", "method", "n_initial", "n_added"])[metric]
                .agg(
                    mean_RMSE="mean",
                    median_RMSE="median",
                    min_RMSE="min",
                    max_RMSE="max",
                    Q1_RMSE=lambda x: x.quantile(0.25),
                    Q3_RMSE=lambda x: x.quantile(0.75),
                )
                .reset_index()
                .sort_values(["method", "n_initial"])
            )
            summary.to_excel(writer, sheet_name=metric, index=False)
    print(f"{CASE_NAME} summary saved to {summary_path}")
