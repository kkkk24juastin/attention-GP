import pandas as pd

from config import (
    CASE_NAME,
    DATA_FILE,
    DATA_DIR,
    N_INITIAL_VALUES,
    N_POOL,
    OVERWRITE_DATA,
    REPEATS,
    REPEAT_SEED_START,
    TEST_SEED,
    TEST_SIZE,
)
from core import generate_candidates, non_test_function, points_to_frame


def repeat_seed(repeat):
    return REPEAT_SEED_START + int(repeat) - 1


def write_shared_data():
    test_x = generate_candidates(TEST_SIZE, TEST_SEED)
    metadata = pd.DataFrame(
        [
            {"key": "case_name", "value": CASE_NAME},
            {"key": "test_seed", "value": TEST_SEED},
            {"key": "repeat_seed_start", "value": REPEAT_SEED_START},
            {"key": "repeat_seed_rule", "value": "seed = repeat_seed_start + repeat - 1"},
            {"key": "initial_seed_rule", "value": "initial_seed = seed"},
            {"key": "pool_seed_rule", "value": "pool_seed = seed"},
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
                initial_frames.append(
                    points_to_frame(
                        initial_x,
                        non_test_function(initial_x),
                        n_initial=n_initial,
                        repeat=repeat,
                    )
                )
                initial_frames[-1].insert(2, "seed", seed)

        pd.concat(initial_frames, ignore_index=True).to_excel(
            writer, sheet_name="initial", index=False
        )
        pd.concat(pool_frames, ignore_index=True).to_excel(
            writer, sheet_name="pool", index=False
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
        f"n_pool={N_POOL}, test_size={TEST_SIZE}, "
        f"test_seed={TEST_SEED}, repeat_seed_start={REPEAT_SEED_START}"
    )


if __name__ == "__main__":
    main()
