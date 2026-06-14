import pandas as pd

from config import (
    CASE_NAME,
    DATA_DIR,
    DATA_FILE,
    N_ADDED,
    N_INITIAL_VALUES,
    OVERWRITE_DATA,
    RAW_INITIAL_FILE,
    RAW_LHS_FILE,
    RAW_POOL_FILE,
    RAW_TEST_FILE,
    REPEATS,
    REPEAT_SEED_START,
    TEST_SEED,
)
from core import points_to_frame, raw_frame_to_xy


def repeat_seed(repeat):
    return REPEAT_SEED_START + int(repeat) - 1


def read_xy(path):
    frame = pd.read_excel(path)
    return raw_frame_to_xy(frame)


def write_shared_data():
    test_x, test_y = read_xy(RAW_TEST_FILE)
    initial_x, initial_y = read_xy(RAW_INITIAL_FILE)
    pool_x, pool_y = read_xy(RAW_POOL_FILE)
    lhs_x, lhs_y = read_xy(RAW_LHS_FILE)

    metadata = pd.DataFrame(
        [
            {"key": "case_name", "value": CASE_NAME},
            {"key": "source_test_file", "value": RAW_TEST_FILE.name},
            {"key": "source_initial_file", "value": RAW_INITIAL_FILE.name},
            {"key": "source_pool_file", "value": RAW_POOL_FILE.name},
            {"key": "source_lhs_file", "value": RAW_LHS_FILE.name},
            {"key": "test_seed", "value": TEST_SEED},
            {"key": "repeat_seed_start", "value": REPEAT_SEED_START},
            {"key": "repeat_seed_rule", "value": "seed = repeat_seed_start + repeat - 1"},
            {"key": "n_initial_values", "value": ",".join(map(str, N_INITIAL_VALUES))},
            {"key": "repeats", "value": REPEATS},
            {"key": "n_added", "value": N_ADDED},
            {"key": "n_pool", "value": len(pool_x)},
            {"key": "test_size", "value": len(test_x)},
        ]
    )

    with pd.ExcelWriter(DATA_FILE) as writer:
        metadata.to_excel(writer, sheet_name="metadata", index=False)
        points_to_frame(test_x, test_y).to_excel(writer, sheet_name="test", index=False)

        initial_frames = []
        pool_frames = []
        lhs_frames = []
        for n_initial in N_INITIAL_VALUES:
            if n_initial != len(initial_x):
                raise ValueError(
                    f"Battery fixed initial set has {len(initial_x)} rows, "
                    f"but N_INITIAL_VALUES contains {n_initial}."
                )
            for repeat in range(1, REPEATS + 1):
                seed = repeat_seed(repeat)
                initial_frames.append(
                    points_to_frame(
                        initial_x,
                        initial_y,
                        n_initial=n_initial,
                        repeat=repeat,
                        seed=seed,
                    )
                )
                pool_frames.append(
                    points_to_frame(pool_x, pool_y, repeat=repeat, seed=seed)
                )
                lhs_frames.append(
                    points_to_frame(
                        lhs_x,
                        lhs_y,
                        n_initial=n_initial,
                        repeat=repeat,
                        seed=seed,
                    )
                )

        pd.concat(initial_frames, ignore_index=True).to_excel(
            writer, sheet_name="initial", index=False
        )
        pd.concat(pool_frames, ignore_index=True).to_excel(
            writer, sheet_name="pool", index=False
        )
        pd.concat(lhs_frames, ignore_index=True).to_excel(
            writer, sheet_name="lhs_final", index=False
        )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists() and not OVERWRITE_DATA:
        print(f"{CASE_NAME} shared data already exists: {DATA_FILE}")
        return

    write_shared_data()
    print(f"{CASE_NAME} shared data saved to {DATA_FILE}")
    print(
        f"n_initial={N_INITIAL_VALUES}, repeats={REPEATS}, "
        f"n_added={N_ADDED}, test_seed={TEST_SEED}, "
        f"repeat_seed_start={REPEAT_SEED_START}"
    )


if __name__ == "__main__":
    main()
