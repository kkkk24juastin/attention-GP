import pandas as pd

from config import (
    CASE_NAME,
    DATA_DIR,
    DATA_FILE,
    FEATURE_NAMES,
    GA_REPEATS,
    INITIAL_LHS_SEED_START,
    INITIAL_SET_DIR,
    INITIAL_SET_INDEX_FILE,
    N_INITIAL_VALUES,
    N_POOL,
    NOSE_SHAPE_PARAMETER_COLUMN,
    REPEATS,
    REPEAT_SEED_START,
    TEST_SEED,
)
from core import generate_candidates, points_to_frame
from openrocket_eval import simulate_design_matrix


def repeat_seed(repeat):
    return REPEAT_SEED_START + int(repeat) - 1


def initial_set_filename(n_initial):
    return f"{CASE_NAME}_initial_n{int(n_initial)}.xlsx"


def initial_set_path(n_initial):
    return INITIAL_SET_DIR / initial_set_filename(n_initial)


def initial_lhs_seed(n_initial, repeat=1):
    return INITIAL_LHS_SEED_START + int(n_initial) * 100 + int(repeat) - 1


def build_initial_frame(n_initial, repeat):
    seed = initial_lhs_seed(n_initial, repeat)
    initial_x = generate_candidates(int(n_initial), seed)
    work_dir = DATA_DIR / "initial_openrocket" / f"n{int(n_initial)}_r{int(repeat)}"
    initial_y = simulate_design_matrix(initial_x, work_dir)
    frame = points_to_frame(initial_x, initial_y, n_initial=n_initial, repeat=repeat)
    frame.insert(0, "point_id", range(1, len(frame) + 1))
    frame.insert(1, "case", CASE_NAME)
    frame.insert(4, "seed", seed)
    frame.insert(5, "point_source", "initial_lhs")
    frame.insert(6, "response_source", "openrocket")
    return frame


def write_initial_set_files(initial_by_key):
    INITIAL_SET_DIR.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for n_initial in N_INITIAL_VALUES:
        frame = initial_by_key[(int(n_initial), 1)]
        seed = int(frame["seed"].iloc[0])

        path = initial_set_path(n_initial)
        frame.to_excel(path, index=False)
        index_rows.append(
            {
                "case": CASE_NAME,
                "n_initial": int(n_initial),
                "repeat": 1,
                "seed": int(seed),
                "n_total": int(n_initial),
                "initial_file": path.name,
                "initial_path": str(path),
            }
        )

    INITIAL_SET_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(index_rows).sort_values("n_initial").to_excel(
        INITIAL_SET_INDEX_FILE, index=False
    )


def write_shared_data():
    initial_by_key = {}
    for n_initial in N_INITIAL_VALUES:
        for repeat in range(1, REPEATS + 1):
            initial_by_key[(int(n_initial), int(repeat))] = build_initial_frame(
                n_initial, repeat
            )

    write_initial_set_files(initial_by_key)
    test_x = generate_candidates(N_POOL, TEST_SEED)
    metadata = pd.DataFrame(
        [
            {"key": "case_name", "value": CASE_NAME},
            {"key": "initial_design_source", "value": "lhs_independent_by_n_initial"},
            {"key": "initial_response_source", "value": "openrocket"},
            {"key": "pool_response_source", "value": "openrocket_on_selection"},
            {"key": "test_response_source", "value": "design_only"},
            {"key": "test_seed", "value": TEST_SEED},
            {"key": "initial_lhs_seed_start", "value": INITIAL_LHS_SEED_START},
            {"key": "initial_lhs_seed_rule", "value": "seed = initial_lhs_seed_start + n_initial * 100 + repeat - 1"},
            {"key": "repeat_seed_start", "value": REPEAT_SEED_START},
            {"key": "repeat_seed_rule", "value": "seed = repeat_seed_start + repeat - 1"},
            {"key": "n_initial_values", "value": ",".join(map(str, N_INITIAL_VALUES))},
            {"key": "repeats", "value": REPEATS},
            {"key": "ga_repeats", "value": GA_REPEATS},
            {"key": "n_pool", "value": N_POOL},
            {"key": "initial_set_dir", "value": str(INITIAL_SET_DIR.relative_to(DATA_DIR))},
            {"key": "initial_set_index", "value": INITIAL_SET_INDEX_FILE.name},
        ]
    )
    variables = pd.DataFrame({"name": FEATURE_NAMES})
    variables.loc[len(variables)] = NOSE_SHAPE_PARAMETER_COLUMN

    with pd.ExcelWriter(DATA_FILE) as writer:
        metadata.to_excel(writer, sheet_name="metadata", index=False)
        variables.to_excel(writer, sheet_name="variables", index=False)
        points_to_frame(test_x).to_excel(writer, sheet_name="test", index=False)

        initial_frames = []
        pool_frames = []
        for repeat in range(1, REPEATS + 1):
            seed = repeat_seed(repeat)
            pool_x = generate_candidates(N_POOL, seed)
            pool_frame = points_to_frame(pool_x, repeat=repeat)
            pool_frame.insert(1, "seed", seed)
            pool_frames.append(pool_frame)

        for n_initial in N_INITIAL_VALUES:
            for repeat in range(1, REPEATS + 1):
                initial_frame = initial_by_key[(int(n_initial), int(repeat))].drop(
                    columns=["point_id", "case", "point_source"]
                )
                initial_frames.append(initial_frame)

        pd.concat(initial_frames, ignore_index=True).to_excel(
            writer, sheet_name="initial", index=False
        )
        pd.concat(pool_frames, ignore_index=True).to_excel(
            writer, sheet_name="pool", index=False
        )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_shared_data()
    print(f"{CASE_NAME} shared data saved to {DATA_FILE}")


if __name__ == "__main__":
    main()
