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
    DATASET_DIR,
    DATASET_INDEX_FILE,
    FEATURE_NAMES,
    LOWER_BOUNDS,
    SUMMARY_FILE,
    TARGET_VALUE,
    UPPER_BOUNDS,
)


NUMPY_CONVERTER = default_converter + numpy2ri.converter
_TGP_LOADED = False
EPS = 1e-12
SUMMARY_METRICS = (
    "RMSE_all",
    "RMSE_target",
    "best_pred_quality_loss",
    "best_true_target_error",
    "best_true_response",
    "ga_search_quality_loss",
    "ga_search_mean",
    "ga_search_variance",
    "ga_self_pred_quality_loss",
    "ga_self_pred_mean",
    "ga_self_pred_variance",
    "ga_pred_quality_loss",
    "ga_true_quality_loss",
    "ga_true_target_error",
    "ga_true_response",
    "ga_pred_mean",
    "ga_pred_variance",
    "ga_true_quality_loss_with_pred_variance",
    "pm_eval_QL",
    "pm_eval_mean",
    "pm_eval_variance",
)


def wing_weight_base(X):
    if X.shape[1] != 10:
        raise ValueError("X must have 10 columns for the Wing Weight function.")

    Sw = X[:, 0]
    Wfw = X[:, 1]
    A = X[:, 2]
    Lambda = np.deg2rad(X[:, 3])
    q = X[:, 4]
    taper = X[:, 5]
    tc = X[:, 6]
    Nz = X[:, 7]
    Wdg = X[:, 8]
    Wp = X[:, 9]
    cos_lam = np.cos(Lambda)
    return (
        0.036
        * Sw**0.758
        * Wfw**0.0035
        * (A / cos_lam**2) ** 0.6
        * q**0.006
        * taper**0.04
        * (100.0 * tc / cos_lam) ** (-0.3)
        * (Nz * Wdg) ** 0.49
        + Sw * Wp
    )


def nonstationary_penalty(X):
    Sw = X[:, 0]
    A = X[:, 2]
    sweep = X[:, 3]
    q = X[:, 4]
    taper = X[:, 5]
    tc = X[:, 6]
    Nz = X[:, 7]
    Wdg = X[:, 8]

    high_load = (Nz > 5.0) & (q > 36.0)
    load_excess = np.maximum(Nz - 5.0, 0.0)
    pressure_excess = np.maximum(q - 36.0, 0.0)
    reinforcement = np.where(
        high_load,
        18.0 + 9.0 * load_excess + 0.7 * pressure_excess + 0.004 * (Wdg - 2100.0),
        0.0,
    )

    sweep_band = (sweep > 2.0) & (sweep < 7.0)
    sweep_correction = np.where(sweep_band, 10.0 + 2.2 * np.sin(np.deg2rad(8.0 * sweep)), 0.0)

    thin_high_aspect = (tc < 0.11) & (A > 8.5)
    buckling_penalty = np.where(thin_high_aspect, 14.0 * (8.5 - taper) / 8.0, 0.0)

    local_modes = 4.0 * np.sin(0.20 * Sw + 2.0 * A) * np.cos(0.25 * q)
    return reinforcement + sweep_correction + buckling_penalty + local_modes


def non_test_function(X):
    return wing_weight_base(X) + nonstationary_penalty(X)


def generate_candidates(n_samples, seed):
    sampler = qmc.LatinHypercube(d=len(LOWER_BOUNDS), seed=int(seed))
    unit_lhs = sampler.random(n=n_samples)
    return qmc.scale(unit_lhs, LOWER_BOUNDS, UPPER_BOUNDS)


def points_to_frame(X, y, n_initial=None, repeat=None):
    data = {column: X[:, idx] for idx, column in enumerate(FEATURE_NAMES)}
    data["y"] = y
    frame = pd.DataFrame(data)
    if n_initial is not None:
        frame.insert(0, "n_initial", int(n_initial))
    if repeat is not None:
        insert_at = 1 if "n_initial" in frame.columns else 0
        frame.insert(insert_at, "repeat", int(repeat))
    return frame


def training_set_to_frame(X, y, method, n_initial, repeat, seed, source):
    frame = points_to_frame(X, y)
    frame.insert(0, "point_id", np.arange(1, len(frame) + 1))
    frame.insert(1, "case", CASE_NAME)
    frame.insert(2, "method", method)
    frame.insert(3, "n_initial", int(n_initial))
    frame.insert(4, "n_added", max(0, len(frame) - int(n_initial)))
    frame.insert(5, "repeat", int(repeat))
    frame.insert(6, "seed", int(seed))
    frame.insert(7, "source", source)
    point_source = np.where(frame["point_id"] <= int(n_initial), "initial", "active_learning")
    frame.insert(8, "point_source", point_source)
    return frame


def frame_to_xy(frame):
    X = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    return X, y


def training_set_filename(method, n_initial, repeat):
    return f"{CASE_NAME}_{method}_n{int(n_initial)}_r{int(repeat)}.xlsx"


def training_set_path(method, n_initial, repeat):
    return DATASET_DIR / training_set_filename(method, n_initial, repeat)


def save_training_set(method, n_initial, repeat, seed, X, y):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    path = training_set_path(method, n_initial, repeat)
    frame = training_set_to_frame(
        X,
        y,
        method=method,
        n_initial=n_initial,
        repeat=repeat,
        seed=seed,
        source="initial+active_learning",
    )
    frame.to_excel(path, index=False)
    return {
        "case": CASE_NAME,
        "method": method,
        "n_initial": int(n_initial),
        "n_added": len(frame) - int(n_initial),
        "n_total": len(frame),
        "repeat": int(repeat),
        "seed": int(seed),
        "dataset_file": path.name,
        "dataset_path": str(path),
    }


def completed_dataset_keys(index_file=DATASET_INDEX_FILE):
    index_path = Path(index_file)
    if not index_path.exists():
        return set()
    data = pd.read_excel(index_path)
    return {
        (row.method, int(row.n_initial), int(row.repeat))
        for row in data.itertuples(index=False)
    }


def save_dataset_index_row(index_file, row):
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


def load_training_set(path):
    frame = pd.read_excel(path)
    return frame_to_xy(frame)


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


def _prediction_variance_array(raw_variance, expected_size):
    variance = np.asarray(raw_variance, dtype=float)
    if variance.ndim == 2 and variance.shape[0] == variance.shape[1]:
        variance = np.diag(variance)
    variance = variance.ravel()
    if variance.size == int(expected_size) ** 2:
        variance = variance[:: int(expected_size) + 1]
    if variance.size != int(expected_size):
        raise ValueError(
            f"TGP variance size mismatch: expected {int(expected_size)}, got {variance.size}"
        )
    return np.maximum(variance, EPS)


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
        variance = _prediction_variance_array(r("pred$var"), mean.size)
    return mean, variance


def quality_loss_scores(mu, variance, target_value=TARGET_VALUE):
    mu = np.asarray(mu, dtype=float).ravel()
    variance = np.maximum(np.asarray(variance, dtype=float).ravel(), EPS)
    return (mu - float(target_value)) ** 2 + variance


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
        (row.method, int(row.n_initial), int(row.repeat), int(row.ga_repeat))
        for row in data.itertuples(index=False)
        if hasattr(row, "ga_repeat") and pd.notna(row.ga_repeat)
    }


def save_result_row(result_file, row):
    result_path = Path(result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if result_path.exists():
        old_rows = pd.read_excel(result_path)
        data = pd.concat([old_rows, new_row], ignore_index=True)
        if "ga_repeat" in data.columns:
            data = data[data["ga_repeat"].notna()].copy()
        dedupe_keys = ["method", "n_initial", "repeat"]
        if "ga_repeat" in data.columns:
            dedupe_keys.append("ga_repeat")
        data = data.drop_duplicates(dedupe_keys, keep="last")
    else:
        data = new_row
    sort_keys = ["method", "n_initial", "repeat"]
    if "ga_repeat" in data.columns:
        sort_keys.append("ga_repeat")
    data = data.sort_values(sort_keys)
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
        group_columns = ["case", "method", "n_initial", "n_added"]
        if "n_total" in data.columns:
            group_columns.append("n_total")
        counts = (
            data.groupby(group_columns)
            .size()
            .reset_index(name="n")
            .sort_values(["method", "n_initial"])
        )
        counts.to_excel(writer, sheet_name="counts", index=False)
        for metric in SUMMARY_METRICS:
            if metric not in data.columns:
                continue
            summary = (
                data.groupby(group_columns)[metric]
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
