import numpy as np
from scipy.spatial.distance import cdist

from config import N_ADDED
from core import append_pool_point, tgp_predict


METHOD_NAME = "IMSE"
MAX_ANCHORS = 800


def _estimate_length_scale(X_train, X_pool):
    if X_train.shape[0] > 1:
        train_distances = cdist(X_train, X_train)
        train_distances[train_distances == 0] = np.inf
        nearest_distances = np.min(train_distances, axis=1)
        positive = nearest_distances[np.isfinite(nearest_distances) & (nearest_distances > 0)]
        if positive.size > 0:
            return max(np.median(positive), 1e-6)

    sample = X_pool[: min(200, len(X_pool))]
    pool_distances = cdist(sample, sample)
    positive = pool_distances[pool_distances > 0]
    return max(np.median(positive), 1e-6) if positive.size > 0 else 1.0


def _anchor_indices(n_pool):
    if n_pool <= MAX_ANCHORS:
        return np.arange(n_pool)
    return np.linspace(0, n_pool - 1, MAX_ANCHORS, dtype=int)


def _imse_scores(X_train, X_pool, variance):
    length_scale = _estimate_length_scale(X_train, X_pool)
    anchors = _anchor_indices(len(X_pool))
    anchor_x = X_pool[anchors]
    anchor_variance = variance[anchors]
    sq_distances = cdist(X_pool, anchor_x, metric="sqeuclidean")
    corr_squared = np.exp(-sq_distances / (length_scale**2))
    noise = max(np.mean(variance) * 1e-6, 1e-12)
    return (variance / (variance + noise)) * (corr_squared @ anchor_variance)


def run(initial_x, initial_y, pool_x, pool_y, seed=None, repeat=None):
    X_train = initial_x.copy()
    y_train = initial_y.copy()
    X_pool = pool_x.copy()
    y_pool = pool_y.copy()

    for _ in range(N_ADDED):
        _, variance = tgp_predict(X_train, y_train, X_pool)
        selected_idx = int(np.nanargmax(_imse_scores(X_train, X_pool, variance)))
        X_train, y_train, X_pool, y_pool = append_pool_point(
            X_train, y_train, X_pool, y_pool, selected_idx
        )

    return X_train, y_train
