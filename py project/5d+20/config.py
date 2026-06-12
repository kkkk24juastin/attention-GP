from pathlib import Path


CASE_NAME = "5d+20"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "generated_data"
RESULT_DIR = SCRIPT_DIR / "refactored_results"
DATA_FILE = DATA_DIR / "shared_data.xlsx"
RESULT_FILE = RESULT_DIR / "all_methods_results.xlsx"
SUMMARY_FILE = RESULT_DIR / "summary_statistics.xlsx"

TARGET_VALUE = 5.5
TARGET_BAND = 0.5
LOWER_BOUNDS = (0.0, 0.0, 0.0, 0.0, 0.0)
UPPER_BOUNDS = (1.0, 1.0, 1.0, 1.0, 1.0)

N_ADDED = 20
N_PRESELECT = 10
N_POOL = 6000
TEST_SIZE = 5000
N_INITIAL_VALUES = tuple(range(80, 141, 10))
REPEATS = 30
WORKERS = 12
TEST_SEED = 42
REPEAT_SEED_START = 42

OVERWRITE_DATA = False
RESET_RESULTS_ON_RUN = False

METHOD_MODULES = {
    "PM": "methods.pm",
    "LHS": "methods.lhs",
    "G_opt": "methods.g_opt",
    "D_opt": "methods.d_opt",
    "K_means": "methods.k_means",
    "EI": "methods.ei",
    "UCB": "methods.ucb",
    "IMSE": "methods.imse",
}

METHODS = tuple(METHOD_MODULES.keys())

# Scheduler scope. Edit these values to run a subset without changing data generation.
RUN_METHODS = METHODS
RUN_N_INITIAL_VALUES = N_INITIAL_VALUES
RUN_REPEATS = REPEATS
