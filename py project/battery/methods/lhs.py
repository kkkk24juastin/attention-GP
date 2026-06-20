from config import DATA_FILE
from core import frame_to_xy, load_shared_data


METHOD_NAME = "LHS"


def _load_lhs_final(repeat, n_initial):
    shared_data = load_shared_data(DATA_FILE)
    lhs_final = shared_data["lhs_final"]
    lhs_final = lhs_final[lhs_final["repeat"] == int(repeat)]
    if "n_initial" in lhs_final.columns:
        lhs_final = lhs_final[lhs_final["n_initial"] == int(n_initial)]
    if lhs_final.empty:
        raise ValueError(
            f"No one-shot LHS training set found for repeat={repeat}, "
            f"n_initial={n_initial}."
        )
    return frame_to_xy(lhs_final)


def run(initial_x, initial_y, pool_x, pool_y, seed=None, repeat=None):
    if repeat is None:
        raise ValueError("Battery LHS requires repeat to load the one-shot lhs_final set.")
    return _load_lhs_final(repeat, len(initial_x))
