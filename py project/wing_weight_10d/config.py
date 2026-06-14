from pathlib import Path


CASE_NAME = "wing_weight_10d"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "generated_data"
RESULT_DIR = SCRIPT_DIR / "refactored_results"
DATA_FILE = DATA_DIR / "shared_data.xlsx"
DATASET_DIR = RESULT_DIR / "selected_datasets"
DATASET_INDEX_FILE = RESULT_DIR / "selected_datasets_index.xlsx"
GA_RESULT_FILE = RESULT_DIR / "ga_optimization_results.xlsx"
RESULT_FILE = GA_RESULT_FILE
SUMMARY_FILE = RESULT_DIR / "summary_statistics.xlsx"

FEATURE_NAMES = (
    "Sw",
    "Wfw",
    "A",
    "Lambda",
    "q",
    "lambda",
    "tc",
    "Nz",
    "Wdg",
    "Wp",
)
FEATURE_DESCRIPTIONS = (
    "wing area [ft^2]",
    "weight of fuel in the wing [lb]",
    "aspect ratio [-]",
    "quarter-chord sweep [deg]",
    "dynamic pressure at cruise [lb/ft^2]",
    "taper ratio [-]",
    "airfoil thickness-to-chord ratio [-]",
    "ultimate load factor [-]",
    "flight design gross weight [lb]",
    "paint weight [lb/ft^2]",
)
LOWER_BOUNDS = (150.0, 220.0, 6.0, -10.0, 16.0, 0.5, 0.08, 2.5, 1700.0, 0.025)
UPPER_BOUNDS = (200.0, 300.0, 10.0, 10.0, 45.0, 1.0, 0.18, 6.0, 2500.0, 0.08)

TARGET_VALUE = 300.0
TARGET_BAND = 20.0

N_ADDED = 30
N_PRESELECT = 50
N_POOL = 10000
TEST_SIZE = 3000
N_INITIAL_VALUES = (120,)
REPEATS = 20
WORKERS = 8
TEST_SEED = 42
REPEAT_SEED_START = 42

GA_POP_SIZE = 50
GA_GENERATIONS = 40
GA_ELITE_FRACTION = 0.15
GA_TOURNAMENT_SIZE = 3
GA_MUTATION_RATE = 0.12
GA_MUTATION_SCALE = 0.08

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
