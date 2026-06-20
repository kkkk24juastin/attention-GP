from config import N_ADDED
from core import generate_candidates, non_test_function


METHOD_NAME = "LHS"


def run(initial_x, initial_y, pool_x, pool_y, seed=None):
    sample_seed = 0 if seed is None else seed
    X_train = generate_candidates(len(initial_x) + N_ADDED, sample_seed)
    y_train = non_test_function(X_train)
    return X_train, y_train
