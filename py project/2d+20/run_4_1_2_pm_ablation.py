import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import default_converter, numpy2ri
from scipy.spatial.distance import cdist
from scipy.stats import qmc


SECTION = "4.1.2"
CASE_NAME = "2d+20"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "generated_data"
DATA_FILE = DATA_DIR / "shared_data.xlsx"
RESULT_DIR = SCRIPT_DIR / "refactored_results"
RESULT_FILE = RESULT_DIR / "4_1_2_pm_ablation.xlsx"

TARGET_VALUE = 1.5
TARGET_BAND = 0.2
LOWER_BOUNDS = (-2.0, -2.0)
UPPER_BOUNDS = (2.0, 2.0)
N_ADDED = 20
N_PRESELECT = 25
N_POOL = 2000
TEST_SIZE = 2000
N_INITIAL_VALUES = tuple(range(20, 81, 10))
REPEATS = 30
WORKERS = 12
TEST_SEED = 42
REPEAT_SEED_START = 42
TAU1 = 1.0
TAU2 = 1.0
EPS = 1e-12

VARIANTS = (
    {"variant": "PM_full", "parameter_label": "two_stage_softmax", "lambda_quality": np.nan},
    {"variant": "PM_stage1_QL_only", "parameter_label": "quality_loss_only", "lambda_quality": np.nan},
    {"variant": "PM_stage2_MDSF_only", "parameter_label": "mdsf_only", "lambda_quality": np.nan},
    {"variant": "PM_no_softmax_rank", "parameter_label": "raw_rank_two_stage", "lambda_quality": np.nan},
    {"variant": "PM_linear_weighted", "parameter_label": "lambda=0.25", "lambda_quality": 0.25},
    {"variant": "PM_linear_weighted", "parameter_label": "lambda=0.50", "lambda_quality": 0.50},
    {"variant": "PM_linear_weighted", "parameter_label": "lambda=0.75", "lambda_quality": 0.75},
)

KEY_COLUMNS = ["task_id"]
METRICS = [
    "RMSE_all",
    "RMSE_target",
    "best_pred_quality_loss",
    "best_true_target_error",
    "runtime_seconds",
]

NUMPY_CONVERTER = default_converter + numpy2ri.converter
_TGP_LOADED = False


def non_test_function(x):
    x1, x2 = x[:, 0], x[:, 1]
    condition = x1 > 0
    return np.where(
        condition,
        (2 + 0.5 * x2) * np.sin(x1) + x2**2,
        x1**2 + x2**2 + (8 - (x1**2 + x2**2)) * (1 - np.exp(-x1**2)),
    )


def generate_candidates(n_samples, seed):
    sampler = qmc.LatinHypercube(d=len(LOWER_BOUNDS), seed=int(seed))
    unit_lhs = sampler.random(n=n_samples)
    return qmc.scale(unit_lhs, LOWER_BOUNDS, UPPER_BOUNDS)


def points_to_frame(x, y, n_initial=None, repeat=None):
    frame = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y})
    if n_initial is not None:
        frame.insert(0, "n_initial", int(n_initial))
    if repeat is not None:
        insert_at = 1 if "n_initial" in frame.columns else 0
        frame.insert(insert_at, "repeat", int(repeat))
    return frame


def frame_to_xy(frame):
    x = frame[["x1", "x2"]].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    return x, y


def load_shared_data(data_file=DATA_FILE):
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Shared data file not found: {data_path}")
    return pd.read_excel(data_path, sheet_name=None)


def ensure_tgp_loaded():
    global _TGP_LOADED
    if not _TGP_LOADED:
        robjects.r("suppressPackageStartupMessages(library(tgp))")
        _TGP_LOADED = True


def tgp_predict(x_train, y_train, x_candidates):
    ensure_tgp_loaded()
    r = robjects.r
    with NUMPY_CONVERTER.context():
        r.assign("X_train", x_train)
        r.assign("y_train", y_train)
        r.assign("X_candidates", x_candidates)
        r("model <- btgp(X_train, y_train, XX = X_candidates, verb = 0)")
        r("pred <- list(mean = model$ZZ.mean, var = model$ZZ.s2)")
        mean = np.asarray(r("pred$mean"), dtype=float).ravel()
        variance = np.asarray(r("pred$var"), dtype=float).ravel()
    return mean, np.maximum(variance, EPS)


def repeat_seed(repeat):
    return REPEAT_SEED_START + int(repeat) - 1


def ensure_shared_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        return

    test_x = generate_candidates(TEST_SIZE, TEST_SEED)
    metadata = pd.DataFrame(
        [
            {"key": "case_name", "value": CASE_NAME},
            {"key": "test_seed", "value": TEST_SEED},
            {"key": "repeat_seed_start", "value": REPEAT_SEED_START},
            {"key": "repeat_seed_rule", "value": "seed = repeat_seed_start + repeat - 1"},
            {"key": "n_initial_values", "value": ",".join(map(str, N_INITIAL_VALUES))},
            {"key": "repeats", "value": REPEATS},
            {"key": "n_pool", "value": N_POOL},
            {"key": "test_size", "value": TEST_SIZE},
        ]
    )

    with pd.ExcelWriter(DATA_FILE) as writer:
        metadata.to_excel(writer, sheet_name="metadata", index=False)
        points_to_frame(test_x, non_test_function(test_x)).to_excel(
            writer, sheet_name="test", index=False
        )

        initial_frames = []
        pool_frames = []
        for repeat in range(1, REPEATS + 1):
            seed = repeat_seed(repeat)
            pool_x = generate_candidates(N_POOL, seed)
            pool_frame = points_to_frame(
                pool_x,
                non_test_function(pool_x),
                repeat=repeat,
            )
            pool_frame.insert(1, "seed", seed)
            pool_frames.append(pool_frame)

        for n_initial in N_INITIAL_VALUES:
            for repeat in range(1, REPEATS + 1):
                seed = repeat_seed(repeat)
                initial_x = generate_candidates(n_initial, seed)
                initial_frame = points_to_frame(
                    initial_x,
                    non_test_function(initial_x),
                    n_initial=n_initial,
                    repeat=repeat,
                )
                initial_frame.insert(2, "seed", seed)
                initial_frames.append(initial_frame)

        pd.concat(initial_frames, ignore_index=True).to_excel(
            writer, sheet_name="initial", index=False
        )
        pd.concat(pool_frames, ignore_index=True).to_excel(
            writer, sheet_name="pool", index=False
        )


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


def softmax_from_log_scores(log_scores, tau=1.0):
    values = np.asarray(log_scores, dtype=float) / max(float(tau), EPS)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(values.shape, 1.0 / len(values))
    floor = np.min(values[finite]) - 50.0
    values = np.where(finite, values, floor)
    values = values - np.max(values)
    weights = np.exp(values)
    total = np.sum(weights)
    return weights / total if total > 0 else np.full(values.shape, 1.0 / len(values))


def minmax_score(values, reverse=False):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.ones(values.shape)
    safe_values = np.where(finite, values, np.nanmedian(values[finite]))
    low = np.min(safe_values)
    high = np.max(safe_values)
    if high - low <= EPS:
        score = np.ones(values.shape)
    else:
        score = (safe_values - low) / (high - low)
    return 1.0 - score if reverse else score


def quality_loss_scores(mu, variance, target_value):
    mu = np.asarray(mu, dtype=float).ravel()
    variance = np.maximum(np.asarray(variance, dtype=float).ravel(), EPS)
    return (mu - float(target_value)) ** 2 + variance


def response_attention_weights(quality_loss, tau=TAU1):
    normalized_loss = quality_loss / max(float(np.nanmax(quality_loss)), EPS)
    return softmax_from_log_scores(-np.log(np.maximum(normalized_loss, EPS)), tau=tau)


def space_filling_scores(candidates, x_train):
    distances = cdist(candidates, x_train)
    return np.min(distances, axis=1)


def space_attention_weights(space_scores, tau=TAU2):
    normalized_space = space_scores / max(float(np.nanmax(space_scores)), EPS)
    return softmax_from_log_scores(np.log(np.maximum(normalized_space, EPS)), tau=tau)


def append_pool_point(x_train, y_train, x_pool, y_pool, selected_idx):
    x_new = x_pool[selected_idx].reshape(1, -1)
    y_new = np.array([y_pool[selected_idx]])
    x_train = np.vstack([x_train, x_new])
    y_train = np.concatenate([y_train, y_new])
    x_pool = np.delete(x_pool, selected_idx, axis=0)
    y_pool = np.delete(y_pool, selected_idx, axis=0)
    return x_train, y_train, x_pool, y_pool


def select_next_index(x_train, y_train, x_pool, variant, lambda_quality):
    mu, variance = tgp_predict(x_train, y_train, x_pool)
    ql = quality_loss_scores(mu, variance, TARGET_VALUE)
    space = space_filling_scores(x_pool, x_train)

    if variant == "PM_full":
        response_weights = response_attention_weights(ql, tau=TAU1)
        n_preselect = min(N_PRESELECT, len(x_pool))
        top_indices = np.argsort(-response_weights)[:n_preselect]
        space_weights = space_attention_weights(space[top_indices], tau=TAU2)
        return int(top_indices[np.argmax(space_weights)])

    if variant == "PM_stage1_QL_only":
        response_weights = response_attention_weights(ql, tau=TAU1)
        return int(np.nanargmax(response_weights))

    if variant == "PM_stage2_MDSF_only":
        return int(np.nanargmax(space))

    if variant == "PM_no_softmax_rank":
        n_preselect = min(N_PRESELECT, len(x_pool))
        top_indices = np.argsort(ql)[:n_preselect]
        return int(top_indices[np.nanargmax(space[top_indices])])

    if variant == "PM_linear_weighted":
        quality_score = minmax_score(ql, reverse=True)
        space_score = minmax_score(space, reverse=False)
        lam = float(lambda_quality)
        combined = lam * quality_score + (1.0 - lam) * space_score
        return int(np.nanargmax(combined))

    raise ValueError(f"Unknown PM ablation variant: {variant}")


def run_pm_variant(initial_x, initial_y, pool_x, pool_y, variant, lambda_quality):
    x_train = initial_x.copy()
    y_train = initial_y.copy()
    x_pool = pool_x.copy()
    y_pool = pool_y.copy()

    for _ in range(N_ADDED):
        selected_idx = select_next_index(x_train, y_train, x_pool, variant, lambda_quality)
        x_train, y_train, x_pool, y_pool = append_pool_point(
            x_train, y_train, x_pool, y_pool, selected_idx
        )

    return x_train, y_train


def evaluate_final(x_train, y_train, test_x, test_y):
    pred_y, pred_var = tgp_predict(x_train, y_train, test_x)
    pred_y = np.asarray(pred_y, dtype=float).ravel()
    pred_var = np.maximum(np.asarray(pred_var, dtype=float).ravel(), EPS)
    rmse_all = np.sqrt(np.mean((pred_y - test_y) ** 2))
    mask = (test_y >= TARGET_VALUE - TARGET_BAND) & (test_y <= TARGET_VALUE + TARGET_BAND)
    rmse_target = (
        np.sqrt(np.mean((pred_y[mask] - test_y[mask]) ** 2)) if mask.any() else np.nan
    )
    final_ql = quality_loss_scores(pred_y, pred_var, TARGET_VALUE)
    best_idx = int(np.nanargmin(final_ql))
    return {
        "RMSE_all": rmse_all,
        "RMSE_target": rmse_target,
        "best_pred_quality_loss": float(final_ql[best_idx]),
        "best_true_target_error": float((test_y[best_idx] - TARGET_VALUE) ** 2),
    }


def task_id(variant_spec, n_initial, repeat):
    return (
        f"{SECTION}|{variant_spec['variant']}|{variant_spec['parameter_label']}|"
        f"n{int(n_initial)}|r{int(repeat)}"
    )


def completed_task_ids():
    if not RESULT_FILE.exists():
        return set()
    raw = pd.read_excel(RESULT_FILE, sheet_name="raw")
    return set(raw["task_id"].astype(str))


def build_tasks(done, shared_data):
    tasks = []
    for variant_spec in VARIANTS:
        for n_initial in N_INITIAL_VALUES:
            for repeat in range(1, REPEATS + 1):
                tid = task_id(variant_spec, n_initial, repeat)
                if tid in done:
                    continue
                split = get_split(shared_data, n_initial, repeat)
                tasks.append((tid, variant_spec, n_initial, repeat, split))
    return tasks


def run_task(task):
    tid, variant_spec, n_initial, repeat, split = task
    initial_x, initial_y, pool_x, pool_y, test_x, test_y = split
    seed = repeat_seed(repeat)
    started = time.perf_counter()

    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(
        prefix=f"tgp_{CASE_NAME}_{SECTION}_{variant_spec['variant']}_{n_initial}_{repeat}_"
    ) as tmp_dir:
        os.chdir(tmp_dir)
        robjects.r.assign("tgp_worker_dir", tmp_dir)
        robjects.r("setwd(tgp_worker_dir)")
        try:
            final_x, final_y = run_pm_variant(
                initial_x,
                initial_y,
                pool_x,
                pool_y,
                variant_spec["variant"],
                variant_spec["lambda_quality"],
            )
            metrics = evaluate_final(final_x, final_y, test_x, test_y)
        finally:
            os.chdir(original_cwd)
            robjects.r.assign("tgp_worker_dir", str(original_cwd))
            robjects.r("setwd(tgp_worker_dir)")

    return {
        "task_id": tid,
        "section": SECTION,
        "case": CASE_NAME,
        "variant": variant_spec["variant"],
        "parameter_label": variant_spec["parameter_label"],
        "target_value": TARGET_VALUE,
        "target_band": TARGET_BAND,
        "n_initial": int(n_initial),
        "n_added": N_ADDED,
        "n_preselect": N_PRESELECT,
        "tau1": TAU1,
        "tau2": TAU2,
        "lambda_quality": variant_spec["lambda_quality"],
        "repeat": int(repeat),
        "seed": int(seed),
        **metrics,
        "runtime_seconds": time.perf_counter() - started,
    }


def q1(series):
    return series.quantile(0.25)


def q3(series):
    return series.quantile(0.75)


def metric_summary(data, group_cols):
    summary = data.groupby(group_cols, dropna=False)[METRICS].agg(
        ["mean", "median", "min", "max", q1, q3]
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns.to_flat_index()]
    return summary.reset_index()


def metadata_frame():
    return pd.DataFrame(
        [
            {"key": "section", "value": SECTION},
            {"key": "case", "value": CASE_NAME},
            {"key": "result_file", "value": str(RESULT_FILE)},
            {"key": "target_value", "value": TARGET_VALUE},
            {"key": "target_band", "value": TARGET_BAND},
            {"key": "n_added", "value": N_ADDED},
            {"key": "n_preselect", "value": N_PRESELECT},
            {"key": "n_initial_values", "value": ",".join(map(str, N_INITIAL_VALUES))},
            {"key": "repeats", "value": REPEATS},
            {"key": "workers", "value": WORKERS},
            {"key": "tau1", "value": TAU1},
            {"key": "tau2", "value": TAU2},
            {"key": "variants", "value": ",".join(v["parameter_label"] for v in VARIANTS)},
            {"key": "cli_args", "value": "none; edit constants in this file only"},
        ]
    )


def write_workbook(raw):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    raw = raw.drop_duplicates(KEY_COLUMNS, keep="last")
    raw = raw.sort_values(["variant", "parameter_label", "n_initial", "repeat"])
    counts = (
        raw.groupby(["section", "case", "variant", "parameter_label", "n_initial"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["variant", "parameter_label", "n_initial"])
    )
    by_n = metric_summary(
        raw,
        ["section", "case", "variant", "parameter_label", "target_value", "n_initial"],
    )
    overall = metric_summary(
        raw,
        ["section", "case", "variant", "parameter_label", "target_value"],
    )

    with pd.ExcelWriter(RESULT_FILE) as writer:
        metadata_frame().to_excel(writer, sheet_name="metadata", index=False)
        raw.to_excel(writer, sheet_name="raw", index=False)
        counts.to_excel(writer, sheet_name="counts", index=False)
        by_n.to_excel(writer, sheet_name="summary_by_n", index=False)
        overall.to_excel(writer, sheet_name="summary_overall", index=False)


def save_result_row(row):
    if RESULT_FILE.exists():
        raw = pd.read_excel(RESULT_FILE, sheet_name="raw")
        raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True)
    else:
        raw = pd.DataFrame([row])
    write_workbook(raw)


def main():
    ensure_shared_data()
    shared_data = load_shared_data(DATA_FILE)
    done = completed_task_ids()
    tasks = build_tasks(done, shared_data)
    if not tasks:
        print(f"{SECTION} {CASE_NAME}: no pending PM ablation tasks.")
        if RESULT_FILE.exists():
            write_workbook(pd.read_excel(RESULT_FILE, sheet_name="raw"))
        return

    print(f"{SECTION} {CASE_NAME}: run {len(tasks)} pending PM ablation tasks.")
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            save_result_row(row)
            print(
                f"Saved {row['task_id']}: RMSE_all={row['RMSE_all']:.4f}, "
                f"RMSE_target={row['RMSE_target']:.4f}"
            )

    print(f"{SECTION} {CASE_NAME}: results saved to {RESULT_FILE}")


if __name__ == "__main__":
    main()
