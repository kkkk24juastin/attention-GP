import numpy as np
from scipy.stats import norm

from config import N_ADDED
from core import append_pool_point, tgp_predict


METHOD_NAME = "EI"


def _ei_scores(mu, variance, y_train, xi=0.01):
    std = np.sqrt(np.maximum(variance, 1e-12))
    best_y = np.max(y_train)
    improvement = mu - best_y - xi
    z = improvement / std
    return improvement * norm.cdf(z) + std * norm.pdf(z)


def run(initial_x, initial_y, pool_x, pool_y, seed=None, repeat=None):
    X_train = initial_x.copy()
    y_train = initial_y.copy()
    X_pool = pool_x.copy()
    y_pool = pool_y.copy()

    for _ in range(N_ADDED):
        mu, variance = tgp_predict(X_train, y_train, X_pool)
        selected_idx = int(np.nanargmax(_ei_scores(mu, variance, y_train)))
        X_train, y_train, X_pool, y_pool = append_pool_point(
            X_train, y_train, X_pool, y_pool, selected_idx
        )

    return X_train, y_train
