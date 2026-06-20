from pathlib import Path


CASE_NAME = "rocket"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "generated_data"
RESULT_DIR = SCRIPT_DIR / "refactored_results"
DATA_FILE = DATA_DIR / "shared_data.xlsx"
INITIAL_SET_DIR = DATA_DIR / "initial_sets"
INITIAL_SET_INDEX_FILE = DATA_DIR / "initial_sets_index.xlsx"
DATASET_DIR = RESULT_DIR / "selected_datasets"
DATASET_INDEX_FILE = RESULT_DIR / "selected_datasets_index.xlsx"
GA_CANDIDATE_FILE = RESULT_DIR / "ga_optimization_candidates.xlsx"
OPENROCKET_TEMPLATE_FILE = RESULT_DIR / "openrocket_simulation_template.xlsx"
OPENROCKET_RESULT_FILE = RESULT_DIR / "openrocket_simulation_results.xlsx"
GA_RESULT_FILE = RESULT_DIR / "ga_optimization_results.xlsx"
RESULT_FILE = GA_RESULT_FILE
SUMMARY_FILE = RESULT_DIR / "summary_statistics.xlsx"

OPENROCKET_JAR_FILE = SCRIPT_DIR / "OpenRocket-24.12.jar"
OPENROCKET_ORK_TEMPLATE = SCRIPT_DIR / "backup.ork"
OPENROCKET_RUNNER_FILE = SCRIPT_DIR / "OpenRocketBatchRunner.java"
OPENROCKET_SIMULATION_INDEX = 3
OPENROCKET_TIMEOUT_SECONDS = 120

FEATURE_NAMES = ("x1", "x2", "x3", "x4", "x5")
NOSE_SHAPE_PARAMETER_COLUMN = "头锥外形参数"
RAW_FEATURE_COLUMNS = (
    "头锥长度(0-20)",
    "头锥底座直径/箭体外直径(2-5)",
    "头锥壁厚(0-1)",
    "箭体长度(30-50)",
    "箭体内直径(1-2)",
)
RAW_RESPONSE_COLUMN = "最大飞行高度"

LOWER_BOUNDS = (0.0, 2.0, 0.0, 30.0, 1.0)
UPPER_BOUNDS = (20.0, 5.0, 1.0, 50.0, 2.0)
TARGET_VALUE = 400.0
TARGET_BAND = 30.0

# OpenRocket stores SI units. The original design table uses cm for lengths,
# diameters, and wall thicknesses.
NOSE_SHAPE_NAME = "ogive"
NOSE_LENGTH_SCALE = 0.01
OUTER_DIAMETER_SCALE = 0.01
NOSE_THICKNESS_SCALE = 0.01
BODY_LENGTH_SCALE = 0.01
INNER_DIAMETER_SCALE = 0.01
NOSE_SHAPE_THRESHOLD_CM = 10.0
NOSE_SHAPE_SHORT_PARAMETER = 0.0
NOSE_SHAPE_LONG_PARAMETER = 1.0


def nose_shape_parameter_from_length(nose_length_cm):
    if float(nose_length_cm) > NOSE_SHAPE_THRESHOLD_CM:
        return float(NOSE_SHAPE_LONG_PARAMETER)
    return float(NOSE_SHAPE_SHORT_PARAMETER)

N_INITIAL_VALUES = (50, 60, 70, 80, 90, 100, 110, 120)
N_ADDED = 20
N_PRESELECT = 5
N_POOL = 10000
REPEATS = 1
WORKERS = 12
TEST_SEED = 42
REPEAT_SEED_START = 42
INITIAL_LHS_SEED_START = 2026

GA_REPEATS = 20
GA_POP_SIZE = 50
GA_GENERATIONS = 200
GA_STALL_GENERATIONS = 10
PM_EVAL_SEED = 42

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
