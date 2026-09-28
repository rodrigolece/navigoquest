import pathlib


# Global constants fixed throughout the paper
LEVELS = [1, 2, 6, 8, 11]

FOLDER_NAME_DATA = "../data"
FOLDER_NAME_PATHS = FOLDER_NAME_DATA + "/raw_paths"
FOLDER_NAME_COHORT_ENVS = FOLDER_NAME_DATA + "/env_cohort"
FOLDER_NAME_BOUNDARY_ENVS = FOLDER_NAME_DATA + "/env_boundary"
FOLDER_NAME_METRICS = FOLDER_NAME_DATA + "/metrics"

METADATA_FILENAME = "users_gb_1998.csv"
PATHS_NORMATIVE_FILENAMES = {
    lvl: f"level{lvl:02}_gb_1998_first-attempts.csv" for lvl in LEVELS
}
PATHS_CLINICAL_FILENAME = "clinical_paths.feather"

METADATA_COLS = ["age", "gender"]
AGE_RANGE = {"min": 24, "max": 80}
BOUNDARY_PARAMs = {
    1: {"rin": 1, "rout": 2},
    2: {"rin": 1, "rout": 2},
    6: {"rin": 1.5, "rout": 4},
    8: {"rin": 1.5, "rout": 4},
    11: {"rin": 1, "rout": 2},
}
ODMAT_WINDOW_SIZE = 5
ODMAT_WEIGHT_SCALE = 2.0

## Plotting specific constants
PLOT_CONFIG = {
    "feat_types": [
        "duration",
        "path_length",
        "average_curvature",
        "boundary_affinity",
        "frobenius_deviation",
        "supremum_deviation",
        "conformity",
        "vector_conformity",
    ],
    "title": {
        "duration": "Duration",
        "path_length": "Length",
        "average_curvature": "Curvature",
        "boundary_affinity": "Boundary",
        "frobenius_deviation": "F-Deviation",
        "supremum_deviation": "S-Deviation",
        "conformity": "M-Conformity",
        "vector_conformity": "V-Conformity",
    },
    "short_title": {
        "duration": "Dur",
        "path_length": "Len",
        "average_curvature": "Curv",
        "boundary_affinity": "Bdry",
        "frobenius_deviation": "FDev",
        "supremum_deviation": "SDev",
        "conformity": "MCon",
        "vector_conformity": "VCon",
    },
}


# Standard directories used over the notebooks
def standard_dirs():
    cwd = pathlib.Path.cwd()
    data_dir = cwd / FOLDER_NAME_DATA
    paths_dir = cwd / FOLDER_NAME_PATHS
    cohort_env_dir = cwd / FOLDER_NAME_COHORT_ENVS
    boundary_env_dir = cwd / FOLDER_NAME_BOUNDARY_ENVS
    output_dir = cwd / FOLDER_NAME_METRICS

    output = {
        "cwd": cwd,
        "data": data_dir,
        "paths": paths_dir,
        "env_cohort": cohort_env_dir,
        "env_boundary": boundary_env_dir,
        "output": output_dir,
    }
    return output
