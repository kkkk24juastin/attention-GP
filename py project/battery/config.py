from pathlib import Path


CASE_NAME = "battery"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "generated_data"
RESULT_DIR = SCRIPT_DIR / "refactored_results"
DATA_FILE = DATA_DIR / "shared_data.xlsx"
RESULT_FILE = RESULT_DIR / "all_methods_results.xlsx"
SUMMARY_FILE = RESULT_DIR / "summary_statistics.xlsx"
TRAINING_SET_DIR = RESULT_DIR / "training_sets"
TRAINING_SET_INDEX_FILE = RESULT_DIR / "training_sets_index.xlsx"

RAW_TEST_FILE = SCRIPT_DIR / "data.xlsx"
RAW_INITIAL_FILE = SCRIPT_DIR / "acttrain.xlsx"
RAW_POOL_FILE = SCRIPT_DIR / "actpool.xlsx"
RAW_LHS_FILE = SCRIPT_DIR / "lhstrain.xlsx"

FEATURE_COLUMNS = ("tem", "Current", "Voltage")
RESPONSE_COLUMN = "Capacity"

TARGET_VALUE = 5.0
TARGET_BAND = 1.0
SOC_ERROR_THRESHOLD = 5.0

N_ADDED = 30
N_PRESELECT = 50
N_INITIAL_VALUES = (180,)
REPEATS = 30
WORKERS = 4
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
