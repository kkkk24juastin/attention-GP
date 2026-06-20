import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from config import N_ADDED


METHOD_NAME = "K_means"


def run(initial_x, initial_y, pool_x, pool_y, seed=None, repeat=None):
    X_train = initial_x.copy()
    y_train = initial_y.copy()
    X_pool = pool_x.copy()
    y_pool = pool_y.copy()

    kmeans = KMeans(n_clusters=N_ADDED)
    kmeans.fit(X_pool)
    centers = kmeans.cluster_centers_
    distances = cdist(centers, X_pool)
    closest_indices = np.argmin(distances, axis=1)
    X_train = np.vstack([X_train, X_pool[closest_indices]])
    y_train = np.concatenate([y_train, y_pool[closest_indices]])

    return X_train, y_train
