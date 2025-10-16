"""
Pipeline functions that simplify bundled tasks
"""

import time

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from .environments import EnvironmentStore, UserODMatrix
from .paths import PathDataset
from .utils_paper import (
    LEVELS,
    METADATA_COLS,
    METADATA_FILENAME,
    PATHS_CLINICAL_FILENAME,
    PATHS_NORMATIVE_FILENAMES,
)


def load_paths_normative(
    lvl,
    dirs,
    age_range=None,
    metadata_cols=METADATA_COLS,
    filenames_normative=PATHS_NORMATIVE_FILENAMES,
    metadata_filename=METADATA_FILENAME,
):
    """
    Load path data for normative data
    Currently retrieves the list of paths
    """
    if age_range is None:
        age_range = {"min": -np.inf, "max": np.inf}
    paths_dir = dirs["paths"]

    # Load path data
    print("Loading path data...")
    dataset = PathDataset.from_level_csv(
        paths_filename=paths_dir / filenames_normative[lvl],
        metadata_filename=paths_dir / metadata_filename,
        metadata_keep_cols=metadata_cols,
    )

    # Filter by age
    df_metadata = dataset.user_metadata
    idx_drop = (df_metadata.age < age_range["min"]) | (df_metadata.age > age_range["max"])
    idx_include = ~idx_drop  # Indices to use
    dataset.user_metadata = df_metadata.loc[idx_include].reset_index(drop=True)
    filtered_paths = [dataset.paths[i] for i in df_metadata.loc[idx_include].index]
    dataset.paths = filtered_paths  # Only use paths in the index range
    print("done. \n")
    return dataset.paths


def load_paths_clinical(lvl, dirs, filename=PATHS_CLINICAL_FILENAME):
    if lvl == "all":
        levels = LEVELS
    elif isinstance(lvl, int):
        levels = [lvl]
    else:
        assert isinstance(lvl, list)
        levels = lvl
    paths_dir = dirs["paths"]

    # Load paths
    df_all = pd.read_feather(paths_dir / filename)
    paths_split = []
    for lvl in levels:
        grp_lvl = df_all.loc[df_all["level"] == lvl]
        dataset = PathDataset.from_clinical_dataset(grp_lvl, path_col="trajectory_data")
        paths_split.append(dataset.paths)

    if isinstance(lvl, int):
        assert len(paths_split) == 1
        return paths_split[0]
    else:
        return paths_split


def load_environment(dirs, levels=LEVELS):
    """
    Load environment store for given levels
    """
    print("Loading environment data...")
    env_store = EnvironmentStore(dirs["env_cohort"], dirs["env_boundary"], levels)
    print("done. \n")
    return env_store


def pipeline_compute_features(paths, env, task_list, early_stop=np.inf, use_sparse_norm=True):
    """
    General pipeline for computing features of paths fed, as named in the task_list variable
    """
    key_list = [item[0] for item in task_list]
    cohort_env, boundary_env = env
    time_counters = {key: [] for key in key_list}  # Computation time counter (unnecessary)
    output = []
    early_stop_counter = 0
    for item_path in tqdm(paths):
        item_odmat = UserODMatrix.from_path(item_path, cohort_env)  # OD Mat for particular path
        # For replacing placeholders in task_list
        arg_dict = {
            "path": item_path,
            "odmat": item_odmat,
            "env_cohort": cohort_env,
            "env_boundary": boundary_env,
            "use_sparse_norm": use_sparse_norm,
        }

        # Main computation: Metrics per path
        item_output = dict(**item_path.metadata)
        for key, func, arg_pre in task_list:
            tic = time.time()
            arg = [arg_dict[item_argkey] for item_argkey in arg_pre]
            item_output[key] = func(*arg)  # Main computation
            toc = time.time()
            time_counters[key].append(toc - tic)  # Computation time
        output.append(item_output)

        # Early stop (optional)
        early_stop_counter += 1
        if early_stop_counter > early_stop:
            break
    return output
