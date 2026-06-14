from config import DATA_FILE, N_ADDED
from core import append_pool_point, frame_to_xy, load_shared_data


METHOD_NAME = "LHS"


def _load_lhs_final(repeat):
    shared_data = load_shared_data(DATA_FILE)
    lhs_final = shared_data["lhs_final"]
    lhs_final = lhs_final[lhs_final["repeat"] == int(repeat)]
    return frame_to_xy(lhs_final)


def run(initial_x, initial_y, pool_x, pool_y, seed=None, repeat=None):
    if repeat is not None:
        return _load_lhs_final(repeat)

    X_train = initial_x.copy()
    y_train = initial_y.copy()
    X_pool = pool_x.copy()
    y_pool = pool_y.copy()

    for _ in range(N_ADDED):
        X_train, y_train, X_pool, y_pool = append_pool_point(
            X_train, y_train, X_pool, y_pool, 0
        )

    return X_train, y_train
