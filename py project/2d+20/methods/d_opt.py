import numpy as np

from config import N_ADDED


METHOD_NAME = "D_opt"


def _d_optimality(design_matrix):
    return np.linalg.det(design_matrix.T @ design_matrix)


def _select_d_optimal_indices(X_initial, X_candidates, n_additional):
    X_initial_design = np.hstack([np.ones((X_initial.shape[0], 1)), X_initial])
    X_candidates_design = np.hstack([np.ones((X_candidates.shape[0], 1)), X_candidates])
    current_X = X_initial_design
    current_det = _d_optimality(current_X)
    selected_indices = []

    for _ in range(n_additional):
        max_increase = -np.inf
        best_index = -1
        for idx in range(X_candidates.shape[0]):
            if idx in selected_indices:
                continue
            new_X = np.vstack([current_X, X_candidates_design[idx]])
            new_det = _d_optimality(new_X)
            increase = new_det - current_det
            if increase > max_increase:
                max_increase = increase
                best_index = idx
        if best_index != -1:
            selected_indices.append(best_index)
            current_X = np.vstack([current_X, X_candidates_design[best_index]])
            current_det = _d_optimality(current_X)

    return selected_indices


def run(initial_x, initial_y, pool_x, pool_y, seed=None):
    X_train = initial_x.copy()
    y_train = initial_y.copy()
    X_pool = pool_x.copy()
    y_pool = pool_y.copy()

    selected = _select_d_optimal_indices(X_train, X_pool, N_ADDED)
    X_train = np.vstack([X_train, X_pool[selected]])
    y_train = np.concatenate([y_train, y_pool[selected]])

    return X_train, y_train
