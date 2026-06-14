import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from config import N_ADDED
from core import append_pool_point


METHOD_NAME = "K_means"


def run(initial_x, initial_y, pool_x, pool_y, seed=None, repeat=None):
    X_train = initial_x.copy()
    y_train = initial_y.copy()
    X_pool = pool_x.copy()
    y_pool = pool_y.copy()

    kmeans = KMeans(n_clusters=N_ADDED, n_init=10, random_state=seed)
    kmeans.fit(X_pool)
    centers = kmeans.cluster_centers_
    selected = []
    distances = cdist(centers, X_pool)
    for row in distances:
        for idx in np.argsort(row):
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
                break

    for selected_idx in sorted(selected, reverse=True):
        X_train, y_train, X_pool, y_pool = append_pool_point(
            X_train, y_train, X_pool, y_pool, selected_idx
        )

    return X_train, y_train
