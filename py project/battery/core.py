import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import default_converter, numpy2ri

from config import (
    CASE_NAME,
    DATA_FILE,
    FEATURE_COLUMNS,
    RESPONSE_COLUMN,
    SOC_ERROR_THRESHOLD,
    SUMMARY_FILE,
    TARGET_BAND,
    TARGET_VALUE,
    TRAINING_SET_DIR,
)


NUMPY_CONVERTER = default_converter + numpy2ri.converter
_TGP_LOADED = False
MODEL_FEATURE_COLUMNS = [f"x{i}" for i in range(1, len(FEATURE_COLUMNS) + 1)]
SUMMARY_METRICS = (
    "RMSE_all",
    "RMSE_target",
    "MAE_all",
    "MaxAE_all",
    "SOC_error_le_5pct_ratio",
    "MaxAE_target",
)


def raw_frame_to_xy(frame):
    missing = [column for column in (*FEATURE_COLUMNS, RESPONSE_COLUMN) if column not in frame]
    if missing:
        raise ValueError(f"Battery data missing columns: {missing}")
    X = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    y = frame.loc[:, RESPONSE_COLUMN].to_numpy(dtype=float)
    return X, y


def points_to_frame(X, y, n_initial=None, repeat=None, seed=None):
    data = {column: X[:, idx] for idx, column in enumerate(MODEL_FEATURE_COLUMNS)}
    data["y"] = y
    frame = pd.DataFrame(data)
    if n_initial is not None:
        frame.insert(0, "n_initial", int(n_initial))
    if repeat is not None:
        insert_at = 1 if "n_initial" in frame.columns else 0
        frame.insert(insert_at, "repeat", int(repeat))
    if seed is not None:
        insert_at = 2 if "n_initial" in frame.columns and "repeat" in frame.columns else 1
        frame.insert(insert_at, "seed", int(seed))
    return frame


def training_set_to_frame(X, y, method, n_initial, repeat, seed):
    frame = points_to_frame(X, y)
    frame.insert(0, "point_id", np.arange(1, len(frame) + 1))
    frame.insert(1, "case", CASE_NAME)
    frame.insert(2, "method", method)
    frame.insert(3, "n_initial", int(n_initial))
    frame.insert(4, "n_added", max(0, len(frame) - int(n_initial)))
    frame.insert(5, "repeat", int(repeat))
    frame.insert(6, "seed", int(seed))
    point_source = np.where(frame["point_id"] <= int(n_initial), "initial", "active_learning")
    frame.insert(7, "point_source", point_source)
    return frame


def frame_to_xy(frame):
    X = frame[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    return X, y


def training_set_filename(method, n_initial, repeat):
    return f"{CASE_NAME}_{method}_n{int(n_initial)}_r{int(repeat)}.xlsx"


def training_set_path(method, n_initial, repeat):
    return TRAINING_SET_DIR / training_set_filename(method, n_initial, repeat)


def save_training_set(method, n_initial, repeat, seed, X, y):
    TRAINING_SET_DIR.mkdir(parents=True, exist_ok=True)
    path = training_set_path(method, n_initial, repeat)
    frame = training_set_to_frame(X, y, method, n_initial, repeat, seed)
    frame.to_excel(path, index=False)
    return {
        "case": CASE_NAME,
        "method": method,
        "n_initial": int(n_initial),
        "n_added": len(frame) - int(n_initial),
        "n_total": len(frame),
        "repeat": int(repeat),
        "seed": int(seed),
        "training_set_file": path.name,
        "training_set_path": str(path),
    }


def save_training_set_index_row(index_file, row):
    index_path = Path(index_file)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if index_path.exists():
        old_rows = pd.read_excel(index_path)
        data = pd.concat([old_rows, new_row], ignore_index=True)
        data = data.drop_duplicates(["method", "n_initial", "repeat"], keep="last")
    else:
        data = new_row
    data = data.sort_values(["method", "n_initial", "repeat"])
    data.to_excel(index_path, index=False)


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
    pool = pool[pool["repeat"] == int(repeat)]
    initial_x, initial_y = frame_to_xy(initial)
    pool_x, pool_y = frame_to_xy(pool)
    test_x, test_y = frame_to_xy(shared_data["test"])
    return initial_x, initial_y, pool_x, pool_y, test_x, test_y


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
    abs_error = np.abs(pred_y - test_y)
    rmse_all = np.sqrt(np.mean((pred_y - test_y) ** 2))
    mae_all = np.mean(abs_error)
    maxae_all = np.max(abs_error)
    soc_error_le_5pct_ratio = np.mean(abs_error <= SOC_ERROR_THRESHOLD)

    mask = (test_y >= TARGET_VALUE - TARGET_BAND) & (test_y <= TARGET_VALUE + TARGET_BAND)
    if mask.any():
        target_error = pred_y[mask] - test_y[mask]
        target_abs_error = np.abs(target_error)
        rmse_target = np.sqrt(np.mean(target_error**2))
        maxae_target = np.max(target_abs_error)
    else:
        rmse_target = np.nan
        maxae_target = np.nan

    return {
        "RMSE_all": rmse_all,
        "RMSE_target": rmse_target,
        "MAE_all": mae_all,
        "MaxAE_all": maxae_all,
        "SOC_error_le_5pct_ratio": soc_error_le_5pct_ratio,
        "MaxAE_target": maxae_target,
    }


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
        for metric in SUMMARY_METRICS:
            summary = (
                data.groupby(["case", "method", "n_initial", "n_added"])[metric]
                .agg(
                    mean="mean",
                    median="median",
                    min="min",
                    max="max",
                    Q1=lambda x: x.quantile(0.25),
                    Q3=lambda x: x.quantile(0.75),
                )
                .reset_index()
                .sort_values(["method", "n_initial"])
            )
            summary.to_excel(writer, sheet_name=metric[:31], index=False)
    print(f"{CASE_NAME} summary saved to {summary_path}")
