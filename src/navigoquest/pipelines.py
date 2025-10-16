import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import time

from .environments import EnvironmentStore, UserODMatrix
from .metrics import VisitingOrderMetric, PathLengthMetric, AverageCurvatureMetric, BoundaryAffinityMetric, FrobeniusDeviationMetric, SupremumDeviationMetric, ConformityMetric, VectorConformityMetric
from .__init__ import EnvironmentStore, PathDataset, compute_standard_metrics, compute_standard_metrics_by_group
from .__init__ import LEVELS, METADATA_COLS, AGE_RANGE, METADATA_FILENAME, PATHS_NORMATIVE_FILENAMES, PATHS_CLINICAL_FILENAME


def load_paths_normative(lvl, paths_dir, metadata_cols, age_range=None):
    """
    Load path data for normative data
    Currently retrieves the list of paths
    """
    if age_range is None:
        age_range = {'min': -np.inf, 'max': np.inf}

    # Load path data
    print(f'Loading path data...')
    dataset = PathDataset.from_level_csv(
        paths_filename=paths_dir / PATHS_NORMATIVE_FILENAMES[lvl],
        metadata_filename=paths_dir / METADATA_FILENAME,
        metadata_keep_cols=metadata_cols,)
    
    # Filter by age
    df_metadata = dataset.user_metadata
    idx_drop = (df_metadata.age < age_range['min']) | (df_metadata.age > age_range['max'])
    idx_include = ~idx_drop  # Indices to use
    dataset.user_metadata = df_metadata.loc[idx_include].reset_index(drop=True)
    filtered_paths = [dataset.paths[i] for i in df_metadata.loc[idx_include].index]
    dataset.paths = filtered_paths # Only use paths in the index range
    print(f'done. \n')
    return dataset.paths


def load_paths_clinical(lvl, paths_dir):
    if lvl == 'all':
        levels = LEVELS
    elif type(lvl) == int:
        levels = [lvl]
    else:
        assert type(lvl) == list
        levels = lvl

    # Load paths
    df_all = pd.read_feather(paths_dir / PATHS_CLINICAL_FILENAME)
    paths_split = []
    for lvl in levels:
        grp_lvl = df_all.loc[df_all["level"] == lvl]
        dataset = PathDataset.from_clinical_dataset(grp_lvl, path_col="trajectory_data")
        paths_split.append(dataset.paths)

    if type(lvl) == int:
        assert len(paths_split) == 1
        return paths_split[0]
    else:
        return paths_split


def pipeline_compute_features(paths, env, task_list, early_stop=np.inf, use_sparse_norm=True):
    """
    General pipeline for computing features of paths fed, as named in the task_list variable
    """
    key_list = [item[0] for item in task_list]
    cohort_env, boundary_env = env
    time_counters = {key: [] for key in key_list} # Computation time counter (unnecessary)
    output = []
    early_stop_counter = 0
    for item_path in tqdm(paths):
        item_odmat = UserODMatrix.from_path(item_path, cohort_env) # OD Mat for particular path
        # For replacing placeholders in task_list
        arg_dict = {'path': item_path, 'odmat': item_odmat,
                    'env_cohort': cohort_env, 'env_boundary': boundary_env,
                    'use_sparse_norm': use_sparse_norm}

        # Main computation: Metrics per path
        item_output = dict(**item_path.metadata)
        for key, func, arg_pre in task_list:
            tic = time.time()
            arg = [arg_dict[item_argkey] for item_argkey in arg_pre]
            item_output[key] = func(*arg)  # Main computation
            toc = time.time()
            time_counters[key].append(toc-tic)  # Computation time
        output.append(item_output)

        # Early stop (optional)
        early_stop_counter += 1
        if early_stop_counter > early_stop:
            break
    return output

