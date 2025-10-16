from .environments import EnvironmentStore
from .metrics import compute_standard_metrics, compute_standard_metrics_by_group
from .paths import PathDataset

# Global constants fixed throughout the paper

LEVELS = [1, 2, 6, 8, 11]
METADATA_COLS = ["age", "gender"]
AGE_RANGE = {'min': 24, 'max': 80}
BOUNDARY_PARAMs = {
    1: {"rin": 1, "rout": 2},
    2: {"rin": 1, "rout": 2},
    6: {"rin": 1.5, "rout": 4},
    8: {"rin": 1.5, "rout": 4},
    11: {"rin": 1, "rout": 2}}
ODMAT_WINDOW_SIZE = 5
ODMAT_WEIGHT_SCALE = 2.0

METADATA_FILENAME = "users_gb_1998.csv"
PATHS_NORMATIVE_FILENAMES = {lvl: f"level{lvl:02}_gb_1998_first-attempts.csv" for lvl in LEVELS}


__all__ = [
    "EnvironmentStore",
    "PathDataset",
    "compute_standard_metrics",
    "compute_standard_metrics_by_group",
]
