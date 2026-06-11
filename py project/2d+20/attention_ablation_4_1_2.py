import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import default_converter, numpy2ri
from scipy.spatial.distance import cdist
from scipy.stats import qmc


NUMPY_CONVERTER = default_converter + numpy2ri.converter

CASE_NAME = "2d+20"
TARGET_VALUE = 1.5
LOWER_BOUNDS = [-2, -2]
UPPER_BOUNDS = [2, 2]
N_ADDED = 20
N_INITIAL_VALUES = range(20, 81, 10)
DEFAULT_METHODS = [
    "PM_full",
    "Stage1_QL_only",
    "Stage2_MDSF_only",
    "NoSoftmax_QL_rank",
    "Linear_QL_MDSF_lam025",
    "Linear_QL_MDSF_lam050",
    "Linear_QL_MDSF_lam075",
]
LINEAR_LAMBDAS = {
    "Linear_QL_MDSF_lam025": 0.25,
    "Linear_QL_MDSF_lam050": 0.50,
    "Linear_QL_MDSF_lam075": 0.75,
}
OUTPUT_FILE = Path("attention_ablation_4_1_2_results.xlsx")


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


def softmax(values):
    values = np.asarray(values, dtype=float)
    values = values - np.nanmax(values)
    exp_values = np.exp(values)
    total = np.sum(exp_values)
    if total <= 0 or not np.isfinite(total):
        return np.ones_like(values) / len(values)
    return exp_values / total


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


def response_penalty_tgp(X_train, u_train, X_candidates):
    mu, variance = tgp_predict(X_train, u_train, X_candidates)
    return (mu - TARGET_VALUE) ** 2 + variance


def min_distance_scores(X_candidates, X_train):
    distances = cdist(X_candidates, X_train)
    scores = np.min(distances, axis=1)
    max_score = np.max(scores)
    if max_score <= 0:
        return scores
    return scores / max_score


def quality_scores_from_penalty(penalty):
    penalty = np.asarray(penalty, dtype=float)
    min_penalty = np.min(penalty)
    max_penalty = np.max(penalty)
    if np.isclose(max_penalty, min_penalty):
        return np.ones_like(penalty)
    return (max_penalty - penalty) / (max_penalty - min_penalty)


def select_pm_full(X_train, u_train, X_candidates, n_preselect):
    penalty = response_penalty_tgp(X_train, u_train, X_candidates)
    normalized_penalty = penalty / max(np.max(penalty), 1e-12)
    first_stage_weights = softmax(-np.log(np.clip(normalized_penalty, 1e-12, None)))
    top_indices = np.argsort(-first_stage_weights)[:n_preselect]
    X_preselected = X_candidates[top_indices]
    space_scores = min_distance_scores(X_preselected, X_train)
    second_stage_weights = softmax(np.log(np.clip(space_scores, 1e-12, None)))
    return int(top_indices[np.argmax(second_stage_weights)])


def select_stage1_quality_only(X_train, u_train, X_candidates, n_preselect):
    del n_preselect
    penalty = response_penalty_tgp(X_train, u_train, X_candidates)
    return int(np.argmin(penalty))


def select_stage2_mdsf_only(X_train, u_train, X_candidates, n_preselect):
    del u_train, n_preselect
    space_scores = min_distance_scores(X_candidates, X_train)
    return int(np.argmax(space_scores))


def select_no_softmax_quality_rank(X_train, u_train, X_candidates, n_preselect):
    penalty = response_penalty_tgp(X_train, u_train, X_candidates)
    top_indices = np.argsort(penalty)[:n_preselect]
    X_preselected = X_candidates[top_indices]
    space_scores = min_distance_scores(X_preselected, X_train)
    return int(top_indices[np.argmax(space_scores)])


def select_linear_weighted(X_train, u_train, X_candidates, n_preselect, lambda_quality):
    del n_preselect
    penalty = response_penalty_tgp(X_train, u_train, X_candidates)
    quality_scores = quality_scores_from_penalty(penalty)
    space_scores = min_distance_scores(X_candidates, X_train)
    scores = lambda_quality * quality_scores + (1 - lambda_quality) * space_scores
    return int(np.argmax(scores))


def select_next_index(method, X_train, u_train, X_candidates, n_preselect):
    if method == "PM_full":
        return select_pm_full(X_train, u_train, X_candidates, n_preselect)
    if method == "Stage1_QL_only":
        return select_stage1_quality_only(X_train, u_train, X_candidates, n_preselect)
    if method == "Stage2_MDSF_only":
        return select_stage2_mdsf_only(X_train, u_train, X_candidates, n_preselect)
    if method == "NoSoftmax_QL_rank":
        return select_no_softmax_quality_rank(X_train, u_train, X_candidates, n_preselect)
    if method.startswith("Linear_QL_MDSF"):
        return select_linear_weighted(
            X_train,
            u_train,
            X_candidates,
            n_preselect,
            LINEAR_LAMBDAS[method],
        )
    raise ValueError(f"Unknown ablation method: {method}")


def run_ablation(method, n_initial, n_pool, n_preselect):
    X_train = generate_candidates(n_initial, LOWER_BOUNDS, UPPER_BOUNDS)
    u_train = non_test_function(X_train)
    X_candidates = generate_candidates(n_pool, LOWER_BOUNDS, UPPER_BOUNDS)

    for _ in range(N_ADDED):
        selected_idx = select_next_index(method, X_train, u_train, X_candidates, n_preselect)
        X_new = X_candidates[selected_idx].reshape(1, -1)
        u_new = non_test_function(X_new)
        X_train = np.vstack([X_train, X_new])
        u_train = np.concatenate([u_train, u_new])
        X_candidates = np.delete(X_candidates, selected_idx, axis=0)

    return X_train, u_train


def read_raw_results():
    if not OUTPUT_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(OUTPUT_FILE, sheet_name="raw_results")
    except ValueError:
        return pd.read_excel(OUTPUT_FILE)


def completed_keys(raw_results):
    if raw_results.empty:
        return set()
    return {
        (row.method, int(row.n_initial), int(row.repeat))
        for row in raw_results.itertuples(index=False)
    }


def build_summary(raw_results, metric):
    if raw_results.empty:
        return pd.DataFrame()
    return (
        raw_results
        .groupby(["method", "n_initial", "n_added"])[metric]
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


def save_results(raw_results, config_rows):
    summary_all = build_summary(raw_results, "RMSE_all")
    summary_target = build_summary(raw_results, "RMSE_target")
    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        raw_results.to_excel(writer, sheet_name="raw_results", index=False)
        summary_all.to_excel(writer, sheet_name="summary_RMSE_all", index=False)
        summary_target.to_excel(writer, sheet_name="summary_RMSE_target", index=False)
        pd.DataFrame(config_rows).to_excel(writer, sheet_name="config", index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 4.1.2 two-stage attention ablation for the 2d+20 case."
    )
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--n-pool", type=int, default=2000)
    parser.add_argument("--n-preselect", type=int, default=25)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_results = read_raw_results()
    done = completed_keys(raw_results)
    rows = raw_results.to_dict("records") if not raw_results.empty else []
    pending_since_save = 0
    config_rows = [{
        "case": CASE_NAME,
        "n_added": N_ADDED,
        "n_pool": args.n_pool,
        "n_preselect": args.n_preselect,
        "test_size": args.test_size,
        "target_value": TARGET_VALUE,
        "methods": ",".join(args.methods),
        "repeats": args.repeats,
    }]

    if args.summary_only:
        save_results(pd.DataFrame(rows), config_rows)
        print(f"Summary refreshed in {OUTPUT_FILE}")
        return

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
                    f"Run {CASE_NAME}: method={method}, n_initial={n_initial}, "
                    f"n_added={N_ADDED}, repeat={repeat}/{args.repeats}"
                )
                X_train, u_train = run_ablation(
                    method,
                    n_initial,
                    args.n_pool,
                    args.n_preselect,
                )
                rmse_all, rmse_target = evaluate_model(X_train, u_train, test_x, test_y)
                row = {
                    "case": CASE_NAME,
                    "method": method,
                    "n_initial": n_initial,
                    "n_added": N_ADDED,
                    "repeat": repeat,
                    "n_pool": args.n_pool,
                    "n_preselect": args.n_preselect,
                    "lambda_quality": LINEAR_LAMBDAS.get(method, np.nan),
                    "RMSE_all": rmse_all,
                    "RMSE_target": rmse_target,
                }
                rows.append(row)
                done.add(key)
                pending_since_save += 1
                print(
                    f"Result: RMSE_all={rmse_all:.4f}, "
                    f"RMSE_target={rmse_target:.4f}"
                )

                if pending_since_save >= args.save_every:
                    save_results(pd.DataFrame(rows), config_rows)
                    pending_since_save = 0
                    print(f"Saved checkpoint to {OUTPUT_FILE}")

    save_results(pd.DataFrame(rows), config_rows)
    print(f"Experiment completed. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
