import numpy as np

from config import N_ADDED
from core import append_pool_point, tgp_predict


METHOD_NAME = "UCB"


def _ucb_scores(mu, variance, kappa=2.0):
    std = np.sqrt(np.maximum(variance, 1e-12))
    return mu + kappa * std


def run(initial_x, initial_y, pool_x, pool_y, seed=None, repeat=None):
    X_train = initial_x.copy()
    y_train = initial_y.copy()
    X_pool = pool_x.copy()
    y_pool = pool_y.copy()

    for _ in range(N_ADDED):
        mu, variance = tgp_predict(X_train, y_train, X_pool)
        selected_idx = int(np.nanargmax(_ucb_scores(mu, variance)))
        X_train, y_train, X_pool, y_pool = append_pool_point(
            X_train, y_train, X_pool, y_pool, selected_idx
        )

    return X_train, y_train
