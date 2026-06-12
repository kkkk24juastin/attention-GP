import numpy as np
from scipy.spatial.distance import cdist

from config import N_ADDED, N_PRESELECT, TARGET_VALUE
from core import append_pool_point, tgp_predict


METHOD_NAME = "PM"


def run(initial_x, initial_y, pool_x, pool_y, seed=None):
    X_train = initial_x.copy()
    y_train = initial_y.copy()
    X_pool = pool_x.copy()
    y_pool = pool_y.copy()

    for _ in range(N_ADDED):
        mu, variance = tgp_predict(X_train, y_train, X_pool)
        penalty = (mu - TARGET_VALUE) ** 2 + variance
        n_preselect = min(N_PRESELECT, len(X_pool))
        top_indices = np.argsort(penalty)[:n_preselect]
        distances = cdist(X_pool[top_indices], X_train)
        space_filling = np.min(distances, axis=1)
        selected_idx = int(top_indices[np.argmax(space_filling)])
        X_train, y_train, X_pool, y_pool = append_pool_point(
            X_train, y_train, X_pool, y_pool, selected_idx
        )

    return X_train, y_train
