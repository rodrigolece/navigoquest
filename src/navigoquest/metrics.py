import os
from multiprocessing import Pool
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from scipy.sparse.linalg import norm as spnorm

from .environments import (
    BoundaryEnvironment,
    CohortEnvironment,
    GridLike,
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
    # NB: ds_rescaled is negative. This is in the original code but might be a mistake
    # sigmoid: f(x) = 1 / (1 + exp(-x))
    return float(np.sum(1 / (1 + np.exp(-ds_rescaled))) / length)


# For handling FrobeniusDeviationMetric and SupremumDeviationMetric
def MatrixDeviation(mat: UserODMatrix, env: CohortEnvironment, ord: float | str, use_sparse: bool = True) -> float:
    key = mat.metadata["age"], mat.metadata["gender"]
    reference_mat = env.od_matrices[key].norm_mat
    mat_diff = reference_mat - mat.norm_mat

    # Matrix norm
    if use_sparse:
        output = spnorm(mat_diff, ord)
    else:
        output = float(np.linalg.norm(mat_diff.toarray(), ord))
    return output


def FrobeniusDeviationMetric(mat: UserODMatrix, env: CohortEnvironment, use_sparse: bool = True) -> float:
    return MatrixDeviation(mat, env, 'fro', use_sparse)


def SupremumDeviationMetric(mat: UserODMatrix, env: CohortEnvironment, use_sparse: bool = True) -> float:
    return MatrixDeviation(mat, env, np.inf, use_sparse)


def ConformityMetric(mat: UserODMatrix, env: CohortEnvironment) -> float:
    key = mat.metadata["age"], mat.metadata["gender"]
    reference_mat = env.od_matrices[key].norm_mat

    r, s = mat.norm_mat.nonzero()
    matching = reference_mat[r, s].sum() / len(r)

    # minus sign to reverse order
    return float(-matching)


def VectorConformityMetric(path: Path, env: CohortEnvironment) -> float:
    key = path.metadata["age"], path.metadata["gender"]
    field = env.mobility_fields[key]

    T = len(path.xy)  # proxy for duration
    out = 0.0

    diff = path.xy[1:] - path.xy[:-1]

    for k, el in enumerate(path.xy[:-1]):
        Fi = field.get(tuple(el), np.zeros(2))
        out += np.dot(Fi, diff[k])

    # minus sign to reverse order
    return float(-out / T)


def compute_standard_metrics(
    dataset: PathDataset,
    cohort_env: CohortEnvironment,
    boundary_env: BoundaryEnvironment,
    use_sparse: bool = False,
) -> list[dict]:
    records: list[dict] = []

    for path in dataset.paths:
        mat = UserODMatrix.from_path(path, cohort_env)

        if use_sparse:
            FrobFunc = FrobeniusDeviationMetricSparse
            SupFunc = SupremumDeviationMetricSparse
        else:
            FrobFunc = FrobeniusDeviationMetric
            SupFunc = SupremumDeviationMetric


        if use_sparse:
            item_output = {
                    **path.metadata,
                    "voc": VisitingOrderMetric(path, cohort_env),
                    "path_length": PathLengthMetric(path),
                    "average_curvature": AverageCurvatureMetric(path),
                    "boundary_affinity": BoundaryAffinityMetric(path, boundary_env),
                    "frobenius_deviation": FrobeniusDeviationMetricSparse(mat, cohort_env),
                    "supremum_deviation": SupremumDeviationMetricSparse(mat, cohort_env),
                    "conformity": ConformityMetric(mat, cohort_env),
                    "vector_conformity": VectorConformityMetric(path, cohort_env),
                }
        else:
            item_output = {
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

        records.append(item_output)

    return records


def _helper_standard_metrics_for_group(
    args: tuple[list[Path], CohortEnvironment, BoundaryEnvironment],
) -> list[dict]:
    """
    Helper function for a group of paths to process in parallel.
    """
    paths_group, cohort_env, boundary_env = args

    records: list[dict] = []
    for path in paths_group:
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

    return records


def compute_standard_metrics_by_group(
    dataset: PathDataset,
    cohort_env: CohortEnvironment,
    boundary_env: BoundaryEnvironment,
    n_processes: int | None = None,
) -> list[dict]:
    """
    Compute standard metrics for all paths in the dataset.

    Parameters
    ----------
    dataset : PathDataset
        The dataset containing paths to analyze
    cohort_env : CohortEnvironment
        The cohort environment for reference data
    boundary_env : BoundaryEnvironment
        The boundary environment for boundary-related metrics
    n_processes : int | None, optional
        Number of processes to use for parallel computation.
        If None, uses os.cpu_count()

    Returns
    -------
    list[dict]
        List of dictionaries containing computed metrics for each path,
        in the same order as the input dataset
    """
    if n_processes is None:
        n_processes = os.cpu_count()

    # Group paths by age and prepare chunks for parallel processing
    groups = []

    for _, (_, paths_iterator) in dataset.group_by("age", "gender"):
        # Convert iterator to list to avoid serialization issues
        paths_list = list(paths_iterator)
        groups.append((paths_list, cohort_env, boundary_env))

    with Pool(processes=n_processes) as pool:
        group_records = list(
            tqdm(
                pool.imap(_helper_standard_metrics_for_group, groups),
                total=len(groups),
            )
        )

    records: list[dict] = []

    for r in group_records:
        records.extend(r)

    return records
