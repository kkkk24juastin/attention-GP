from config import N_ADDED
from core import append_pool_point


METHOD_NAME = "LHS"


def run(initial_x, initial_y, pool_x, pool_y, seed=None):
    X_train = initial_x.copy()
    y_train = initial_y.copy()
    X_pool = pool_x.copy()
    y_pool = pool_y.copy()

    for _ in range(N_ADDED):
        X_train, y_train, X_pool, y_pool = append_pool_point(
            X_train, y_train, X_pool, y_pool, 0
        )

    return X_train, y_train
