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
    FEATURE_NAMES,
    LOWER_BOUNDS,
    NOSE_SHAPE_PARAMETER_COLUMN,
    RAW_FEATURE_COLUMNS,
    RAW_RESPONSE_COLUMN,
    SUMMARY_FILE,
    TARGET_BAND,
    TARGET_VALUE,
    UPPER_BOUNDS,
    nose_shape_parameter_from_length,
)
from openrocket_eval import simulate_design


NUMPY_CONVERTER = default_converter + numpy2ri.converter
_TGP_LOADED = False
EPS = 1e-12
SUMMARY_METRICS = (
    "RMSE_all",
    "RMSE_target",
    "best_pred_quality_loss",
    "best_reference_target_error",
    "best_reference_response",
    "ga_search_quality_loss",
    "ga_search_mean",
    "ga_search_variance",
    "ga_self_pred_quality_loss",
    "ga_self_pred_mean",
    "ga_self_pred_variance",
    "ga_pred_quality_loss",
    "ga_pred_mean",
    "ga_pred_variance",
    "pm_eval_QL",
    "pm_eval_mean",
    "pm_eval_variance",
    "pm_framework_pred_mean",
    "pm_framework_pred_variance",
    "最大飞行高度",
    "QL_value",
    "true_target_error",
    "flight_time",
    "time_to_apogee",
    "max_velocity",
    "max_acceleration",
)


def raw_frame_to_xy(frame):
    missing = [
        column
        for column in (*RAW_FEATURE_COLUMNS, RAW_RESPONSE_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Rocket data missing columns: {missing}")
    X = frame.loc[:, RAW_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = frame.loc[:, RAW_RESPONSE_COLUMN].to_numpy(dtype=float)
    return X, y


def generate_candidates(n_samples, seed):
    sampler = qmc.LatinHypercube(d=len(LOWER_BOUNDS), seed=int(seed))
    unit_lhs = sampler.random(n=n_samples)
    return qmc.scale(unit_lhs, LOWER_BOUNDS, UPPER_BOUNDS)


def points_to_frame(X, y=None, n_initial=None, repeat=None):
    data = {column: X[:, idx] for idx, column in enumerate(FEATURE_NAMES)}
    if y is not None:
        data["y"] = y
    frame = pd.DataFrame(data)
    frame.insert(
        len(FEATURE_NAMES),
        NOSE_SHAPE_PARAMETER_COLUMN,
        [nose_shape_parameter_from_length(value) for value in frame["x1"]],
    )
    if n_initial is not None:
        frame.insert(0, "n_initial", int(n_initial))
    if repeat is not None:
        insert_at = 1 if "n_initial" in frame.columns else 0
        frame.insert(insert_at, "repeat", int(repeat))
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
    if method == "LHS":
        point_source = np.full(len(frame), "lhs", dtype=object)
        response_source = np.full(len(frame), "openrocket", dtype=object)
    else:
        point_source = np.where(frame["point_id"] <= int(n_initial), "initial", "active_learning")
        response_source = np.where(
            frame["point_id"] <= int(n_initial),
            "openrocket",
            "openrocket",
        )
    frame.insert(7, "point_source", point_source)
    frame.insert(8, "response_source", response_source)
    return frame


def frame_to_xy(frame):
    X = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    if "y" in frame.columns:
        y = frame["y"].to_numpy(dtype=float)
    else:
        y = np.full(X.shape[0], np.nan, dtype=float)
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


def fit_tgp_model(X_train, y_train, model_name="ga_tgp_model"):
    ensure_tgp_loaded()
    r = robjects.r
    x_name = f"{model_name}_X_train"
    y_name = f"{model_name}_y_train"
    with NUMPY_CONVERTER.context():
        r.assign(x_name, X_train)
        r.assign(y_name, y_train)
        r(f"{model_name} <- btgp({x_name}, {y_name}, pred.n = FALSE, verb = 0)")
    return model_name


def tgp_model_predict(X_candidates, model_name="ga_tgp_model"):
    ensure_tgp_loaded()
    r = robjects.r
    x_name = f"{model_name}_X_train"
    with NUMPY_CONVERTER.context():
        r.assign("X_candidates", X_candidates)
        r(f"{model_name}$Xsplit <- rbind({x_name}, X_candidates)")
        r(
            "pred <- predict("
            f"{model_name}, XX = X_candidates, pred.n = FALSE, verb = 0"
            ")"
        )
        mean = np.asarray(r("pred$ZZ.mean"), dtype=float).ravel()
        variance = _prediction_variance_array(r("pred$ZZ.s2"), mean.size)
    return mean, variance


def quality_loss_scores(mu, variance, target_value=TARGET_VALUE):
    mu = np.asarray(mu, dtype=float).ravel()
    variance = np.maximum(np.asarray(variance, dtype=float).ravel(), EPS)
    return (mu - float(target_value)) ** 2 + variance


def evaluate_model(X_train, y_train, test_x, test_y):
    pred_y, pred_var = tgp_predict(X_train, y_train, test_x)
    rmse_all = np.sqrt(np.mean((pred_y - test_y) ** 2))
    mask = (test_y >= TARGET_VALUE - TARGET_BAND) & (test_y <= TARGET_VALUE + TARGET_BAND)
    rmse_target = np.sqrt(np.mean((pred_y[mask] - test_y[mask]) ** 2)) if mask.any() else np.nan
    ql = quality_loss_scores(pred_y, pred_var, TARGET_VALUE)
    return {
        "RMSE_all": rmse_all,
        "RMSE_target": rmse_target,
        "best_pred_quality_loss": float(np.nanmin(ql)),
    }


def append_pool_point(X_train, y_train, pool_x, pool_y, selected_idx):
    X_new = pool_x[selected_idx].reshape(1, -1)
    if pool_y is None or not np.isfinite(pool_y[selected_idx]):
        simulation_dir = Path.cwd() / "openrocket_active_learning" / f"point_{len(y_train) + 1:04d}"
        simulation_result = simulate_design(X_new.ravel(), simulation_dir)
        y_value = float(simulation_result["最大飞行高度"])
    else:
        y_value = float(pool_y[selected_idx])
    y_new = np.array([y_value])
    X_train = np.vstack([X_train, X_new])
    y_train = np.concatenate([y_train, y_new])
    pool_x = np.delete(pool_x, selected_idx, axis=0)
    pool_y = None if pool_y is None else np.delete(pool_y, selected_idx, axis=0)
    return X_train, y_train, pool_x, pool_y


def training_set_filename(method, n_initial, repeat):
    return f"{CASE_NAME}_{method}_n{int(n_initial)}_r{int(repeat)}.xlsx"


def training_set_path(method, n_initial, repeat):
    return DATASET_DIR / training_set_filename(method, n_initial, repeat)


def load_training_set(path):
    frame = pd.read_excel(path)
    return frame_to_xy(frame)


def save_training_set(method, n_initial, repeat, seed, X, y):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
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
        "dataset_file": path.name,
        "dataset_path": str(path),
    }


def summarize_results(result_file, summary_file=SUMMARY_FILE):
    result_path = Path(result_file)
    if not result_path.exists():
        print(f"结果文件不存在: {result_path}")
        return
    data = pd.read_excel(result_path)
    summary_path = Path(summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    group_columns = ["case", "method", "n_initial", "n_added"]
    if "n_total" in data.columns:
        group_columns.append("n_total")
    with pd.ExcelWriter(summary_path) as writer:
        data.to_excel(writer, sheet_name="raw", index=False)
        counts = data.groupby(group_columns).size().reset_index(name="n")
        counts.to_excel(writer, sheet_name="counts", index=False)
        for metric in SUMMARY_METRICS:
            if metric not in data.columns:
                continue
            numeric_metric = pd.to_numeric(data[metric], errors="coerce")
            if numeric_metric.notna().sum() == 0:
                continue
            data_for_metric = data.copy()
            data_for_metric[metric] = numeric_metric
            summary = (
                data_for_metric.groupby(group_columns)[metric]
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
