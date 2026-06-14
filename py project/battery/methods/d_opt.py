import numpy as np

from config import N_ADDED
from core import append_pool_point


METHOD_NAME = "D_opt"


def _log_det(design_matrix):
    sign, logdet = np.linalg.slogdet(design_matrix.T @ design_matrix)
    return logdet if sign > 0 else -np.inf


def run(initial_x, initial_y, pool_x, pool_y, seed=None, repeat=None):
    X_train = initial_x.copy()
    y_train = initial_y.copy()
    X_pool = pool_x.copy()
    y_pool = pool_y.copy()

    for _ in range(N_ADDED):
        current_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        best_score = -np.inf
        selected_idx = 0
        for idx in range(X_pool.shape[0]):
            candidate = np.hstack([1.0, X_pool[idx]]).reshape(1, -1)
            score = _log_det(np.vstack([current_design, candidate]))
            if score > best_score:
                best_score = score
                selected_idx = idx
        X_train, y_train, X_pool, y_pool = append_pool_point(
            X_train, y_train, X_pool, y_pool, selected_idx
        )

    return X_train, y_train
