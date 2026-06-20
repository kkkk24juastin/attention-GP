import numpy as np

from config import N_ADDED
from core import append_pool_point, generate_candidates


METHOD_NAME = "LHS"


def run(initial_x, initial_y, pool_x, pool_y, seed=None):
    X_train = np.empty((0, initial_x.shape[1]), dtype=float)
    y_train = np.empty(0, dtype=float)
    sample_seed = 0 if seed is None else seed
    n_total = len(initial_x) + N_ADDED
    X_pool = generate_candidates(n_total, sample_seed)
    y_pool = np.full(n_total, np.nan)

    while len(X_pool):
        X_train, y_train, X_pool, y_pool = append_pool_point(
            X_train, y_train, X_pool, y_pool, 0
        )

    return X_train, y_train
