import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import default_converter, numpy2ri
from scipy.spatial.distance import cdist
from scipy.stats import norm, qmc


NUMPY_CONVERTER = default_converter + numpy2ri.converter

CASE_NAME = "2d+20"
TARGET_VALUE = 1.5
LOWER_BOUNDS = [-2, -2]
UPPER_BOUNDS = [2, 2]
N_ITERATIONS = 20
N_INITIAL_VALUES = range(20, 81, 10)
DEFAULT_METHODS = ["EI", "UCB", "IMSE"]
OUTPUT_FILE = Path("active_learning_baselines_results.xlsx")
SUMMARY_FILE = Path("active_learning_baselines_summary.xlsx")


def non_test_function(X):
    x, y = X[:, 0], X[:, 1]
    condition = x > 0
    return np.where(
        condition,
        (2 + 0.5 * y) * np.sin(x) + y**2,
        x**2 + y**2 + (8 - (x**2 + y**2)) * (1 - np.exp(-x**2)),
    )


def generate_candidates(n_samples, lower_bounds, upper_bounds):
    sampler = qmc.LatinHypercube(d=len(lower_bounds))
    unit_lhs = sampler.random(n=n_samples)
    return qmc.scale(unit_lhs, lower_bounds, upper_bounds)


def tgp_predict(X_train, u_train, X_candidates):
    r = robjects.r
    with NUMPY_CONVERTER.context():
        r.assign("X_train", X_train)
        r.assign("u_train", u_train)
        r.assign("X_candidates", X_candidates)
        r("library(tgp)")
        r("model <- btgp(X_train, u_train, XX = X_candidates)")
        r("pred <- list(mean = model$ZZ.mean, var = model$ZZ.s2)")
        mean = np.asarray(r("pred$mean"), dtype=float).ravel()
        variance = np.asarray(r("pred$var"), dtype=float).ravel()
    return mean, np.maximum(variance, 1e-12)


def evaluate_model(X_train, u_train, test_x, test_y):
    predict_y, _ = tgp_predict(X_train, u_train, test_x)
    rmse_all = np.sqrt(np.mean((predict_y - test_y) ** 2))
    mask = (test_y >= TARGET_VALUE - 0.2) & (test_y <= TARGET_VALUE + 0.2)
    rmse_target = np.sqrt(np.mean((predict_y[mask] - test_y[mask]) ** 2)) if mask.any() else np.nan
    return rmse_all, rmse_target


def expected_improvement_scores(mu, variance, u_train, xi=0.01):
    std = np.sqrt(variance)
    best_error = np.min(np.abs(u_train - TARGET_VALUE))
    improvement = best_error - np.abs(mu - TARGET_VALUE) - xi
    z = improvement / std
    return improvement * norm.cdf(z) + std * norm.pdf(z)


def targeted_ucb_scores(mu, variance, kappa=2.0):
    std = np.sqrt(variance)
    return -(np.abs(mu - TARGET_VALUE) - kappa * std)


def estimate_length_scale(X_train, X_candidates):
    if X_train.shape[0] > 1:
        train_distances = cdist(X_train, X_train)
        train_distances[train_distances == 0] = np.inf
        nearest_distances = np.min(train_distances, axis=1)
        positive_distances = nearest_distances[
            np.isfinite(nearest_distances) & (nearest_distances > 0)
        ]
        if positive_distances.size > 0:
            return max(np.median(positive_distances), 1e-6)

    sample = X_candidates[: min(200, len(X_candidates))]
    pool_distances = cdist(sample, sample)
    positive_distances = pool_distances[pool_distances > 0]
    if positive_distances.size == 0:
        return 1.0
    return max(np.median(positive_distances), 1e-6)


def approximate_imse_scores(X_train, X_candidates, variance):
    length_scale = estimate_length_scale(X_train, X_candidates)
    sq_distances = cdist(X_candidates, X_candidates, metric="sqeuclidean")
    corr_squared = np.exp(-sq_distances / (length_scale**2))
    noise = max(np.mean(variance) * 1e-6, 1e-12)
    return (variance / (variance + noise)) * (corr_squared @ variance)


def select_acquisition_index(method, X_train, u_train, X_candidates):
    mu, variance = tgp_predict(X_train, u_train, X_candidates)
    if method == "EI":
        scores = expected_improvement_scores(mu, variance, u_train)
    elif method == "UCB":
        scores = targeted_ucb_scores(mu, variance)
    elif method == "IMSE":
        scores = approximate_imse_scores(X_train, X_candidates, variance)
    else:
        raise ValueError(f"Unknown method: {method}")
    return int(np.nanargmax(scores))


def run_baseline(method, n_initial, n_pool):
    X_train = generate_candidates(n_initial, LOWER_BOUNDS, UPPER_BOUNDS)
    u_train = non_test_function(X_train)
    X_candidates = generate_candidates(n_pool, LOWER_BOUNDS, UPPER_BOUNDS)

    for _ in range(N_ITERATIONS):
        selected_idx = select_acquisition_index(method, X_train, u_train, X_candidates)
        X_new = X_candidates[selected_idx].reshape(1, -1)
        u_new = non_test_function(X_new)
        X_train = np.vstack([X_train, X_new])
        u_train = np.concatenate([u_train, u_new])
        X_candidates = np.delete(X_candidates, selected_idx, axis=0)

    return X_train, u_train


def completed_keys():
    if not OUTPUT_FILE.exists():
        return set()
    data = pd.read_excel(OUTPUT_FILE)
    return {
        (row.method, int(row.n_initial), int(row.repeat))
        for row in data.itertuples(index=False)
    }


def append_result(row):
    new_row = pd.DataFrame([row])
    if OUTPUT_FILE.exists():
        old_rows = pd.read_excel(OUTPUT_FILE)
        result = pd.concat([old_rows, new_row], ignore_index=True)
    else:
        result = new_row
    result.to_excel(OUTPUT_FILE, index=False)


def summarize_results():
    if not OUTPUT_FILE.exists():
        print(f"No result file found: {OUTPUT_FILE}")
        return

    data = pd.read_excel(OUTPUT_FILE)
    with pd.ExcelWriter(SUMMARY_FILE) as writer:
        for metric in ["RMSE_all", "RMSE_target"]:
            summary = (
                data
                .groupby(["case", "method", "n_initial", "n_added"])[metric]
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
    print(f"Summary saved to {SUMMARY_FILE}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EI, UCB, and IMSE baselines for the 2d+20 case."
    )
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--n-pool", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.summary_only:
        done = completed_keys()
        test_x = generate_candidates(args.test_size, LOWER_BOUNDS, UPPER_BOUNDS)
        test_y = non_test_function(test_x)

        for n_initial in N_INITIAL_VALUES:
            for repeat in range(1, args.repeats + 1):
                for method in args.methods:
                    key = (method, n_initial, repeat)
                    if key in done:
                        print(f"Skip completed: {key}")
                        continue

                    print(
                        f"Run {CASE_NAME}: method={method}, "
                        f"n_initial={n_initial}, repeat={repeat}/{args.repeats}"
                    )
                    X_train, u_train = run_baseline(method, n_initial, args.n_pool)
                    rmse_all, rmse_target = evaluate_model(X_train, u_train, test_x, test_y)
                    append_result({
                        "case": CASE_NAME,
                        "method": method,
                        "n_initial": n_initial,
                        "n_added": N_ITERATIONS,
                        "repeat": repeat,
                        "RMSE_all": rmse_all,
                        "RMSE_target": rmse_target,
                    })
                    done.add(key)
                    print(
                        f"Saved to {OUTPUT_FILE}: "
                        f"RMSE_all={rmse_all:.4f}, RMSE_target={rmse_target:.4f}"
                    )

    summarize_results()


if __name__ == "__main__":
    main()
