import tempfile

import numpy as np
import pandas as pd

from config import (
    CASE_NAME,
    FEATURE_NAMES,
    GA_CANDIDATE_FILE,
    NOSE_SHAPE_PARAMETER_COLUMN,
    OPENROCKET_RESULT_FILE,
    nose_shape_parameter_from_length,
)
from openrocket_eval import simulate_design_row


KEY_COLUMNS = ("case", "method", "n_initial", "repeat", "ga_repeat")
PASSTHROUGH_COLUMNS = (
    "case",
    "method",
    "n_initial",
    "n_added",
    "n_total",
    "repeat",
    "seed",
    "ga_repeat",
    "dataset_file",
    "dataset_path",
    "ga_engine",
    "ga_param_source",
    "ga_pop_size",
    "ga_generations",
    "ga_stall_generations",
    "ga_search_quality_loss",
    "ga_search_mean",
    "ga_search_variance",
    "ga_self_pred_quality_loss",
    "ga_self_pred_mean",
    "ga_self_pred_variance",
    "pm_framework_dataset_file",
    "pm_framework_dataset_path",
    "ga_best_generation",
    "pm_framework_pred_mean",
    "pm_framework_pred_variance",
    "QL_value",
    "pm_eval_QL",
    "pm_eval_mean",
    "pm_eval_variance",
    "pm_eval_seed",
    "pm_eval_mode",
    NOSE_SHAPE_PARAMETER_COLUMN,
)
RESULT_COLUMNS = (
    *PASSTHROUGH_COLUMNS,
    "ga_pred_quality_loss",
    "ga_pred_mean",
    "ga_pred_variance",
    "ga_search_quality_loss",
    "ga_search_mean",
    "ga_search_variance",
    "ga_self_pred_quality_loss",
    "ga_self_pred_mean",
    "ga_self_pred_variance",
    "pm_eval_QL",
    "pm_eval_mean",
    "pm_eval_variance",
    *(f"ga_x{idx}" for idx in range(1, len(FEATURE_NAMES) + 1)),
    "最大飞行高度",
    "flight_time",
    "time_to_apogee",
    "max_velocity",
    "max_acceleration",
    "simulation_note",
)


def _write_results(rows):
    OPENROCKET_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    if data.empty:
        data = pd.DataFrame(columns=RESULT_COLUMNS)
        data.to_excel(OPENROCKET_RESULT_FILE, index=False)
        return
    data = data.sort_values(["method", "n_initial", "repeat", "ga_repeat"])
    data.to_excel(OPENROCKET_RESULT_FILE, index=False)


def main():
    if not GA_CANDIDATE_FILE.exists():
        raise FileNotFoundError(f"GA 候选文件不存在: {GA_CANDIDATE_FILE}，请先运行 run_ga_optimization.py")

    candidates = pd.read_excel(GA_CANDIDATE_FILE)
    missing_keys = [column for column in KEY_COLUMNS if column not in candidates.columns]
    if missing_keys:
        raise ValueError(f"GA 候选文件缺少必要键列: {missing_keys}")

    rows = []
    print(f"{CASE_NAME}: regenerate {len(candidates)} OpenRocket simulations")
    _write_results(rows)

    for index, (_, row) in enumerate(candidates.iterrows(), start=1):
        result_row = {
            column: row[column]
            for column in candidates.columns
            if (
                column in PASSTHROUGH_COLUMNS
                or column.startswith("ga_x")
                or column.startswith("ga_pred")
                or column.startswith("ga_search")
                or column.startswith("ga_self")
                or column.startswith("pm_eval")
            )
        }
        result_row.setdefault("case", CASE_NAME)
        result_row[NOSE_SHAPE_PARAMETER_COLUMN] = nose_shape_parameter_from_length(row["ga_x1"])
        try:
            with tempfile.TemporaryDirectory(prefix="openrocket_rocket_") as tmp_dir:
                result_row.update(simulate_design_row(row, tmp_dir, prefix="ga_x"))
        except Exception as exc:
            result_row.update(
                {
                    "最大飞行高度": np.nan,
                    "flight_time": np.nan,
                    "time_to_apogee": np.nan,
                    "max_velocity": np.nan,
                    "max_acceleration": np.nan,
                    "simulation_note": f"failed: {exc}",
                }
            )
        rows.append(result_row)
        _write_results(rows)
        print(
            f"OpenRocket {index}/{len(candidates)}: method={row['method']}, "
            f"n_initial={row['n_initial']}, repeat={row['repeat']}, "
            f"ga_repeat={row['ga_repeat']}, MH={result_row['最大飞行高度']}"
        )

    print(f"{CASE_NAME}: OpenRocket results saved to {OPENROCKET_RESULT_FILE}")


if __name__ == "__main__":
    main()
