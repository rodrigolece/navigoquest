import os
from multiprocessing import Pool
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse as sp
from tqdm import tqdm

from .config import BOUNDARY_RADII
from .environments import (
    AggregateODMatrix,
    BoundaryEnvironment,
    CohortEnvironment,
    GridLike,
    GroupKeyT,
    MinLevelEnvironment,
    SupportsBoundaryDistance,
    UserODMatrix,
)
from .paths import Path, PathDataset


class MetricProtocol(Protocol):
    def __call__(
        self, path: Path, env: GridLike | None = None, params: dict | None = None
    ) -> float:
        pass


def VisitingOrderMetric(path: Path, env: GridLike) -> int:
    return int(env.visiting_order_correctness(path))


def PathLengthMetric(path: Path | NDArray) -> float:
    # Adding support for arrays so the smooth path can be passed in BoundaryAffinityMetric
    arr = path.xy if isinstance(path, Path) else path
    displacements = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
    return float(displacements.sum())


def AverageCurvatureMetric(path: Path) -> float:
    diff = path.smooth_xy[1:] - path.smooth_xy[:-1]
    displacements = np.linalg.norm(diff, axis=1)
    length = displacements.sum()
    if length == 0:
        return 0

    # Remove stationary points
    idx = displacements > 0
    diff = diff[idx]
    displacements = displacements[idx]

    diff = diff / displacements[:, np.newaxis]
    curv = np.linalg.norm(diff[1:] - diff[:-1], axis=1)

    return float(np.sum(curv) / length)


def BoundaryAffinityMetric(path: Path, env: SupportsBoundaryDistance) -> float:
    length = PathLengthMetric(path.smooth_xy)
    ds = env.distances_to_boundary(path)
    ds_rescaled = -2 * env.scale * (ds - (env.rout + env.rin) / 2) / (env.rout - env.rin)
    if length == 0:
        return 0
    # NB: ds_rescaled is negative. This is in the original code but might be a mistake
    # sigmoid: f(x) = 1 / (1 + exp(-x))
    return float(np.sum(1 / (1 + np.exp(-ds_rescaled))) / length)


def FrobeniusDeviationMetric(
    mat: UserODMatrix,
    env: CohortEnvironment | None = None,
    reference_mat: AggregateODMatrix | None = None,
) -> float:
    if reference_mat is None:
        if env is None:
            raise ValueError("Must provide either env or reference_mat")
        key = mat.metadata["age"], mat.metadata["gender"]
        reference_mat = env.od_matrices[key]

    mat_diff = reference_mat.norm_mat - mat.norm_mat
    return float(sp.linalg.norm(mat_diff, "fro"))


def SupremumDeviationMetric(
    mat: UserODMatrix,
    env: CohortEnvironment | None = None,
    reference_mat: AggregateODMatrix | None = None,
) -> float:
    if reference_mat is None:
        if env is None:
            raise ValueError("Must provide either env or reference_mat")
        key = mat.metadata["age"], mat.metadata["gender"]
        reference_mat = env.od_matrices[key]

    mat_diff = reference_mat.norm_mat - mat.norm_mat
    return float(sp.linalg.norm((mat_diff), np.inf))


def ConformityMetric(
    mat: UserODMatrix,
    env: CohortEnvironment | None = None,
    reference_mat: AggregateODMatrix | None = None,
) -> float:
    if reference_mat is None:
        if env is None:
            raise ValueError("Must provide either env or reference_mat")
        key = mat.metadata["age"], mat.metadata["gender"]
        reference_mat = env.od_matrices[key]

    r, s = mat.norm_mat.nonzero()
    matching = reference_mat.norm_mat[r, s].sum() / len(r)

    # minus sign to reverse order
    return float(-matching)


def VectorConformityMetric(
    path: Path,
    env: CohortEnvironment | None = None,
    reference_field: dict[GroupKeyT, NDArray[np.int32]] | None = None,
) -> float:
    if reference_field is None:
        if env is None:
            raise ValueError("Must provide either env or reference_field")
        key = path.metadata["age"], path.metadata["gender"]
        reference_field = env.mobility_fields[key]

    T = len(path.xy)  # proxy for duration
    out = 0.0

    diff = path.xy[1:] - path.xy[:-1]

    for k, el in enumerate(path.xy[:-1]):
        Fi = reference_field.get(tuple(el), np.zeros(2))
        out += np.dot(Fi, diff[k])

    # minus sign to reverse order
    return float(-out / T)


def compute_standard_metrics(
    dataset: PathDataset,
    cohort_env: CohortEnvironment,
    boundary_env: BoundaryEnvironment,
) -> pd.DataFrame:
    """Compute all the standard metrics for paths in the dataset.

    Parameters
    ----------
    dataset : PathDataset
        The dataset containing paths to analyze
    cohort_env : CohortEnvironment
        The cohort environment for reference data
    boundary_env : BoundaryEnvironment
        The boundary environment for boundary-related metrics

    Returns
    -------
    pd.DataFrame
        DataFrame containing computed metrics for each path.
    """
    records: list[dict] = []

    for path in dataset.paths:
        mat = UserODMatrix.from_path(path, cohort_env)
        records.append(
            {
                **path.metadata,
                "voc": VisitingOrderMetric(path, cohort_env),
                "path_length": PathLengthMetric(path),
                "average_curvature": AverageCurvatureMetric(path),
                "boundary_affinity": BoundaryAffinityMetric(path, boundary_env),
                "frobenius_deviation": FrobeniusDeviationMetric(mat, cohort_env),
                "supremum_deviation": SupremumDeviationMetric(mat, cohort_env),
                "conformity": ConformityMetric(mat, cohort_env),
                "vector_conformity": VectorConformityMetric(path, cohort_env),
            }
        )

    return pd.DataFrame.from_records(records)


# Module-level worker state for multiprocessing
_worker_level_env: MinLevelEnvironment
_worker_boundary_env: BoundaryEnvironment


def _init_metrics_worker(level: int) -> None:
    """Initialize worker process with environments."""
    global _worker_level_env, _worker_boundary_env
    _worker_level_env = MinLevelEnvironment(level=level)
    _worker_boundary_env = BoundaryEnvironment(level=level, **BOUNDARY_RADII[level])


def _helper_standard_metrics_for_group(args: tuple) -> list[dict]:
    """Helper function to process a group of paths using the worker's environments.

    This function is called by worker processes and uses the module-level
    environments initialized by _init_metrics_worker.

    """
    paths_group, reference_mat, reference_field = args

    records: list[dict] = []

    for path in paths_group:
        mat = UserODMatrix.from_path(path, _worker_level_env)
        records.append(
            {
                **path.metadata,
                "voc": VisitingOrderMetric(path, _worker_level_env),
                "path_length": PathLengthMetric(path),
                "average_curvature": AverageCurvatureMetric(path),
                "boundary_affinity": BoundaryAffinityMetric(path, _worker_boundary_env),
                "frobenius_deviation": FrobeniusDeviationMetric(mat, reference_mat=reference_mat),
                "supremum_deviation": SupremumDeviationMetric(mat, reference_mat=reference_mat),
                "conformity": ConformityMetric(mat, reference_mat=reference_mat),
                "vector_conformity": VectorConformityMetric(path, reference_field=reference_field),
            }
        )

    return records


def compute_standard_metrics_by_group(
    dataset: PathDataset,
    cohort_env: CohortEnvironment,
    n_processes: int | None = None,
) -> pd.DataFrame:
    """Compute in parallel all the standard metrics for paths in the dataset.

    Parameters
    ----------
    dataset : PathDataset
        The dataset containing paths to analyze
    cohort_env : CohortEnvironment
        The cohort environment for reference data
    n_processes : int | None, optional
        Number of processes to use for parallel computation.
        If None, uses os.cpu_count()

    Returns
    -------
    pd.DataFrame
        DataFrame containing computed metrics for each path.
    """
    if n_processes is None:
        n_processes = os.cpu_count()

    group_args = []

    for key, (_, paths_iterator) in dataset.group_by("age", "gender"):
        paths_list = list(paths_iterator)  # list to avoid serialization issues
        reference_mat = cohort_env.od_matrices[key]
        reference_field = cohort_env.mobility_fields[key]
        group_args.append((paths_list, reference_mat, reference_field))

    # calling Pool with initializer to set up environments once per worker
    with Pool(
        processes=n_processes,
        initializer=_init_metrics_worker,
        initargs=[cohort_env.level],
    ) as pool:
        group_records = list(
            tqdm(
                pool.imap(_helper_standard_metrics_for_group, group_args),
                total=len(group_args),
            )
        )

    records = [r for group in group_records for r in group]

    return pd.DataFrame.from_records(records)


dict_metric_function = {
    "voc": VisitingOrderMetric,
    "path_length": PathLengthMetric,
    "average_curvature": AverageCurvatureMetric,
    "boundary_affinity": BoundaryAffinityMetric,
    "frobenius_deviation": FrobeniusDeviationMetric,
    "supremum_deviation": SupremumDeviationMetric,
    "conformity": ConformityMetric,
    "vector_conformity": VectorConformityMetric,
}

dict_metric_argname = {
    "voc": ("path", "env_cohort"),
    "path_length": ("path",),
    "average_curvature": ("path",),
    "boundary_affinity": ("path", "env_boundary"),
    "frobenius_deviation": ("odmat", "env_cohort"),
    "supremum_deviation": ("odmat", "env_cohort"),
    "conformity": ("odmat", "env_cohort"),
    "vector_conformity": ("path", "env_cohort"),
}

def format_metric_task_data(task_list):
    functions = [dict_metric_function[task] for task in task_list]
    argnames = [dict_metric_argname[task] for task in task_list]
    return list(zip(task_list, functions, argnames))
