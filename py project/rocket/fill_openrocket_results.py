from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    GA_CANDIDATE_FILE,
    GA_RESULT_FILE,
    NOSE_SHAPE_PARAMETER_COLUMN,
    OPENROCKET_RESULT_FILE,
    SUMMARY_FILE,
    TARGET_VALUE,
)
from core import summarize_results


SIMULATION_COLUMNS = (
    "最大飞行高度",
    "flight_time",
    "time_to_apogee",
    "max_velocity",
    "max_acceleration",
    "simulation_note",
)

def _key_columns(frame):
    preferred = ["case", "method", "n_initial", "repeat", "ga_repeat"]
    if all(column in frame.columns for column in preferred):
        return preferred
    missing = [column for column in preferred if column not in frame.columns]
    raise ValueError(f"OpenRocket 结果缺少新实验必要键列: {missing}")


def _coerce_numeric(frame, columns):
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _validate_unique_keys(frame, key_columns, label):
    duplicated = frame.duplicated(key_columns, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, key_columns].head(5).to_dict("records")
        raise ValueError(f"{label} 存在重复的新实验键列组合，示例: {examples}")


def build_final_results(candidate_file=GA_CANDIDATE_FILE, simulation_file=OPENROCKET_RESULT_FILE):
    candidate_path = Path(candidate_file)
    simulation_path = Path(simulation_file)
    if not candidate_path.exists():
        raise FileNotFoundError(f"GA 候选文件不存在: {candidate_path}，请先运行 run_ga_optimization.py")
    if not simulation_path.exists():
        raise FileNotFoundError(
            f"OpenRocket 仿真结果文件不存在: {simulation_path}。"
            "请先运行外部 OpenRocket 仿真，并把结果写入该文件。"
        )

    candidates = pd.read_excel(candidate_path)
    simulations = pd.read_excel(simulation_path)
    key_columns = _key_columns(simulations)
    candidates = _coerce_numeric(candidates, ["n_initial", "repeat", "ga_repeat"])
    simulations = _coerce_numeric(
        simulations,
        [
            "n_initial",
            "repeat",
            "ga_repeat",
            "最大飞行高度",
            "flight_time",
            "time_to_apogee",
            "max_velocity",
            "max_acceleration",
        ],
    )
    missing_candidate_keys = [column for column in key_columns if column not in candidates.columns]
    if missing_candidate_keys:
        raise ValueError(f"GA 候选文件缺少新实验必要键列: {missing_candidate_keys}")
    _validate_unique_keys(candidates, key_columns, "GA 候选文件")
    _validate_unique_keys(simulations, key_columns, "OpenRocket 仿真结果")

    keep_columns = [column for column in key_columns if column in simulations.columns]
    if NOSE_SHAPE_PARAMETER_COLUMN in simulations.columns:
        keep_columns.append(NOSE_SHAPE_PARAMETER_COLUMN)
    keep_columns += [column for column in SIMULATION_COLUMNS if column in simulations.columns]
    simulations = simulations[keep_columns]
    data = candidates.merge(simulations, on=key_columns, how="left", suffixes=("", "_openrocket"))
    openrocket_shape_column = f"{NOSE_SHAPE_PARAMETER_COLUMN}_openrocket"
    if openrocket_shape_column in data.columns:
        if NOSE_SHAPE_PARAMETER_COLUMN not in data.columns:
            data[NOSE_SHAPE_PARAMETER_COLUMN] = data[openrocket_shape_column]
        data = data.drop(columns=[openrocket_shape_column])

    if "最大飞行高度" not in data.columns:
        data["最大飞行高度"] = np.nan
    data["true_target_error"] = (
        pd.to_numeric(data["最大飞行高度"], errors="coerce") - TARGET_VALUE
    ) ** 2
    if "QL_value" not in data.columns:
        data["QL_value"] = np.nan
    data["openrocket_status"] = np.where(data["最大飞行高度"].notna(), "completed", "pending")
    return data


def main():
    final_results = build_final_results()
    GA_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_results.to_excel(GA_RESULT_FILE, index=False)
    summarize_results(GA_RESULT_FILE, SUMMARY_FILE)
    completed = int(final_results["最大飞行高度"].notna().sum())
    print(f"OpenRocket results merged: {completed}/{len(final_results)} completed")
    print(f"Final GA results saved to {GA_RESULT_FILE}")


if __name__ == "__main__":
    main()
