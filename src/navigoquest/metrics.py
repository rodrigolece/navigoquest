from typing import Protocol

import numpy as np

from .environments import (
    CohortEnvironment,
    GridLike,
    SupportsBoundaryDistance,
    UserODMatrix,
)
from .paths import Path


class MetricProtocol(Protocol):
    def __call__(
        self, path: Path, env: GridLike | None = None, params: dict | None = None
    ) -> float:
        pass


def VisitingOrderMetric(path: Path, env: GridLike) -> int:
    return int(env.visiting_order_correctness(path))


def PathLengthMetric(path: Path) -> float:
    displacements = np.linalg.norm(path.xy[1:] - path.xy[:-1], axis=1)
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
    length = PathLengthMetric(path)
    ds = env.distances_to_boundary(path)
    ds_rescaled = 2 * env.scale * (ds - (env.rout + env.rin) / 2) / (env.rout - env.rin)
    return float(np.sum(1 / (1 + np.exp(-ds_rescaled))) / length)


def FrobeniusDeviationMetric(mat: UserODMatrix, env: CohortEnvironment) -> float:
    key = mat.metadata["age"], mat.metadata["gender"]
    reference_mat = env.od_matrices[key].norm_mat
    return float(np.linalg.norm((reference_mat - mat.norm_mat).toarray(), "fro"))


def SupremumDeviationMetric(mat: UserODMatrix, env: CohortEnvironment) -> float:
    key = mat.metadata["age"], mat.metadata["gender"]
    reference_mat = env.od_matrices[key].norm_mat
    return float(np.linalg.norm((reference_mat - mat.norm_mat).toarray(), np.inf))


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
