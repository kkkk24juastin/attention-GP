import numpy as np
from sklearn.cluster import KMeans

from config import N_ADDED
from core import non_test_function


METHOD_NAME = "K_means"


def run(initial_x, initial_y, pool_x, pool_y, seed=None):
    X_train = initial_x.copy()
    y_train = initial_y.copy()
    X_pool = pool_x.copy()

    kmeans = KMeans(n_clusters=N_ADDED)
    kmeans.fit(X_pool)
    centers = kmeans.cluster_centers_
    center_y = non_test_function(centers)
    X_train = np.vstack([X_train, centers])
    y_train = np.concatenate([y_train, center_y])

    return X_train, y_train
