from __future__ import annotations

import itertools
import json
import pathlib
import pickle
import re
from abc import ABC
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.spatial import distance_matrix
from sklearn.neighbors import KDTree

from .paths import Path, PathDataset, smooth_path


DEFAULT_FLAG_RADIUS = 3
EXPECTED_VISITING_ORDERS = {
    1: [0],
    2: [0],
    6: [0, 1, 2],
    8: [0, 1, 2],
    11: [1, 0, 1, 2],
}

GroupKey = tuple[Any, ...]


# class Environment(ABC):
#     pass

# --- Contracts ----------------------------------------------------------------


class GridLike(Protocol):
    level: int
    grid: NDArray[np.int32]
    grid_width: int
    grid_length: int
    flags: NDArray[np.int32]
    flag_radius: int  # not used

    def visiting_order(self, path: Path) -> list[int]: ...

    def visiting_order_correctness(self, path: Path) -> bool: ...


class SupportsODMatrix(Protocol):
    def od_matrix(self, paths: Iterable[Path], remove_diag: bool = False) -> sp.csr_matrix: ...


class SupportsBoundaryDistance(Protocol):
    innery_bdry_kdtree: KDTree
    rin: float
    rout: float
    scale: float

    def distances_to_boundary(self, path: Path) -> float: ...


# --- Independent functionality that depends on the grid ------------------------


class LevelGridBase(ABC):
    level: int
    grid: NDArray[np.int32]
    grid_width: int
    grid_length: int
    flags: NDArray[np.int32]
    flag_radius: int = DEFAULT_FLAG_RADIUS

    def __init__(self, grid_filename: str | pathlib.Path) -> None:
        """
        Load the grid and flags.

        Parameters
        ----------
        grid_filename : str | pathlib.Path
            Path to the level JSON file. The code expects the level to be in the filename.
        """
        p = pathlib.Path(grid_filename)

        # Checks
        if not p.exists():
            raise FileNotFoundError(f"{p} does not exist")
        if p.suffix != ".json":
            raise ValueError(f"{p} is not a JSON file")

        # Extract level from filename
        level_match = re.search(r"level(\d+)", p.stem)
        if not level_match:
            raise ValueError(f"Cannot extract level number from filename: {p}")
        self.level = int(level_match.group(1))

        # Initialize
        with p.open("r") as f:
            data = json.load(f)["fixed"]

        self.grid_width = int(data["grid_width"])
        self.grid_length = int(data["grid_length"])
        self.grid = np.array(data["grid_data"]).reshape(
            (self.grid_width, self.grid_length), order="F"
        )
        flags = np.array([(d["x"], d["y"]) for d in data["flags"]], dtype=np.int32)
        if self.level in [6]:  # TODO: check the other levels
            flags = flags[::-1]
        self.flags = flags

    def visiting_order(self, path: Path) -> list[int]:
        """Check if the path visits the flags in the correct order."""
        if self.level not in EXPECTED_VISITING_ORDERS:
            raise NotImplementedError(f"Expected order not defined for level {self.level}")

        dmat = distance_matrix(path.smooth_xy, self.flags)
        _, j = (dmat < self.flag_radius).nonzero()

        return [x[0] for x in itertools.groupby(j)]

    def visiting_order_correctness(self, path: Path) -> bool:
        vo = self.visiting_order(path)
        return vo == EXPECTED_VISITING_ORDERS[self.level]


class ODMatrixMixin:
    def od_matrix(
        self: GridLike, paths: Iterable["Path"], remove_diag: bool = False
    ) -> sp.csr_matrix:
        """
        Compute the origin-destination matrix from a set of paths.

        Parameters
        ----------
        paths : Iterable[Path]
            Paths to aggregate.
        remove_diag : bool
            If True, zero the diagonal.

        Returns
        -------
        sp.csr_matrix
            OD matrix.
        """
        width, height = self.grid_width, self.grid_length
        N = width * height
        out = sp.lil_matrix((N, N), dtype=int)

        for path in paths:
            lex = width * path.xy[:, 1] + path.xy[:, 0]
            for i, j in zip(lex[:-1], lex[1:]):
                out[i, j] += 1

        if remove_diag:
            out[np.diag_indices_from(out)] = 0

        return out.tocsr()


class MobilityFieldMixin:
    def mobility_field(
        self: GridLike, od_mat: sp.csr_matrix
    ) -> dict[tuple[int, int], NDArray[np.int32]]:
        """
        Compute the field at each location (origin) where it is non-zero.

        Parameters
        ----------
        od_mat : sp.csr_matrix
            The orgin-destination (OD) matrix to use as input.

        Returns
        -------
        dict[tuple[int, int], np.ndarray]
            Map (x, y) -> vector field.
        """
        if not sp.issparse(od_mat):
            raise ValueError("od_mat must be sparse")

        width = self.grid_width

        i, j = od_mat.nonzero()
        r_origin = np.vstack((i % width, i // width)).T
        r_dest = np.vstack((j % width, j // width)).T
        u_vec = (r_dest - r_origin).astype(float)

        # unit vectors
        norm = np.linalg.norm(u_vec, axis=1)
        idx = norm > 0
        u_vec[idx] = u_vec[idx] / norm[idx, np.newaxis]

        # unit vectors weighted by the number of trips
        weighted_vec = u_vec * np.asarray(od_mat[i, j]).T

        # sum the vectors grouped by origin using counts of unique elements
        _, idx_uniq, counts = np.unique(i, return_index=True, return_counts=True)
        c = np.cumsum(counts)

        nb_locations = len(idx_uniq)
        Xs = r_origin[idx_uniq]
        Fs = np.zeros((nb_locations, 2))

        start = 0
        for k, end in enumerate(c):
            Fs[k] += weighted_vec[start:end].sum(axis=0)
            start = end

        out: dict[tuple[int, int], NDArray[np.int32]] = {}
        for k in range(len(Xs)):
            out[tuple(Xs[k])] = Fs[k]

        return out


# --- Concrete types -----------------------------------------------------------


@dataclass
class UserODMatrix:
    """OD matrix for a single user/path with associated metadata."""

    norm_mat: sp.csr_matrix
    metadata: dict

    @classmethod
    def from_path(cls, path: Path, level_grid: SupportsODMatrix) -> UserODMatrix:
        mat = level_grid.od_matrix([path])
        return cls(mat, path.metadata)


@dataclass
class AggregateODMatrix:
    mat: sp.csr_matrix
    N: int

    @property
    def norm_mat(self) -> sp.csr_matrix:
        """Matrix normalized by the number of contributing paths."""
        return self.mat / self.N


class CohortEnvironment(ODMatrixMixin, MobilityFieldMixin, LevelGridBase):
    _od_matrices: dict[GroupKey, AggregateODMatrix] | None = None
    _mobility_fields: dict[GroupKey, dict[tuple[int, int], NDArray[np.int32]]] | None = None

    def __init__(self, grid_filename: str | pathlib.Path):
        super().__init__(grid_filename=grid_filename)

    def to_pickle(self, filename: str | pathlib.Path) -> None:
        if self._od_matrices is None or self._mobility_fields is None:
            raise ValueError("call set_od_matrices() and/or set_mobility_fields()")

        if isinstance(filename, str):
            filename = pathlib.Path(filename)

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @property
    def od_matrices(self) -> dict[GroupKey, AggregateODMatrix]:
        if self._od_matrices is None:
            raise ValueError("od_matrices not  initialized; call set_od_matrices() first")
        return self._od_matrices

    @property
    def mobility_fields(self) -> dict[GroupKey, dict[tuple[int, int], NDArray[np.int32]]]:
        if self._mobility_fields is None:
            raise ValueError("mobility_fields not initialized; call set_mobility_fields() first")
        return self._mobility_fields

    def set_od_matrices(self, path_dataset: PathDataset, *attributes: str) -> None:
        if self._od_matrices is not None:
            raise ValueError("matrices already set")
        self._od_matrices = {}

        for key, (N, paths) in path_dataset.group_by(*attributes):
            mat = self.od_matrix(paths)
            self._od_matrices[key] = AggregateODMatrix(mat, N)

        return None

    def set_mobility_fields(
        self,
        window: int = 1,
    ) -> None:
        if self._od_matrices is None:
            raise ValueError("call `set_od_matrices` first")
        if self._mobility_fields is not None:
            raise ValueError("mobility fields already set")
        self._mobility_fields = {}

        if window != 1:
            # See _od_matrix_windowed in od_loader.py

            # for key in self._od_matrices.keys():
            # age_centre is in key
            # ages = range(age_centre - window, age_centre + window + 1)
            # weights = st.distributions.norm(age_centre, scale=scale).pdf(ages)
            # weights /= weights.sum()

            # N = self.grid_width * self.grid_length
            # od_mat = sp.csr_matrix((N, N))

            # for i, age in enumerate(ages):
            #     od_mat += weights[i] * self._od_matrices[key].norm_mat  # need changed age in key

            raise NotImplementedError("age windows not implemented yet")

        for key, agg in self._od_matrices.items():
            self._mobility_fields[key] = self.mobility_field(agg.mat)

        return None


class BoundaryEnvironment:
    innery_bdry_kdtree: KDTree
    rin: float
    rout: float
    scale: float

    def __init__(self, filename: str | pathlib.Path, rin: float, rout: float, scale: float) -> None:
        p = pathlib.Path(filename)
        if not p.exists():
            raise FileNotFoundError(f"{p} does not exist")
        if p.suffix != ".npy":  # TODO: change the format
            raise ValueError

        inner_bdry = smooth_path(np.load(p))
        self.inner_bdry_kdtree = KDTree(inner_bdry)
        self.rin = rin
        self.rout = rout
        self.scale = scale

    def to_pickle(self, filename: str | pathlib.Path) -> None:
        if isinstance(filename, str):
            filename = pathlib.Path(filename)
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def distances_to_boundary(self, path: Path) -> float:
        ds, _ = self.inner_bdry_kdtree.query(path.smooth_xy, k=1)  # second out arg is indices
        return ds.flatten()  # KDTree returns a column vector


# def breakup_by_flags(path: Path, flags: NDArray[np.int32], R: float = 3) -> list[int]:
#     """
#     Find the last index of the first passage by each flag.

#     Parameters
#     ----------
#     path : Path
#         The path to be split.
#     flags : np.ndarray
#         The coordinates of the flags (ordered).
#     R : float
#         The radius to consider a checkpoint visited.

#     Returns
#     -------
#     List[int]
#         The list with the indices that can be used to split `path`.

#     """
#     out = []
#     offset = 0

#     for f in flags:
#         idx = np.where(np.linalg.norm(path.xy[offset:] - f, 2, axis=1) <= R)[0] + offset
#         max_sets = [
#             max(map(itemgetter(1), g))
#             for _, g in groupby(enumerate(idx), lambda ix: ix[0] - ix[1])
#         ]
#         offset = min(max_sets)
#         out.append(offset)

#     return out

# def od_matrix_brokenup(
#     self, paths: list[Path], R: float = FLAG_RADIUS, remove_diag: bool = False
# ) -> float:
#     """
#     Calculate the OD matrices broken up by the visits to the checkpoints.

#     Parameters
#     ----------
#     paths : list[Path]
#         The paths to be used in counting the number of trips between locations.
#     R : float, optional
#         The radius to consider a checkpoint visited.
#     remove_diag : bool, optional
#         Delete the entries in the diagonal (default is False).

#     Returns
#     -------
#     List[sp.csr_matrix]
#         List of origin-destination (OD) matrices.

#     """
#     width, height = self.grid_width, self.grid_length
#     N = width * height
#     nb_flags = len(self.flags)

#     out = [sp.lil_matrix((N, N), dtype=int) for _ in range(nb_flags)]

#     for path in paths:
#         idx = self.breakup_by_flags(path.xy, self.flags, R=R)

#         for k, chunk in enumerate(np.split(path.xy, idx[:-1])):
#             lex = self.grid_width * chunk[:, 1] + chunk[:, 0]

#             for i, j in zip(lex[:-1], lex[1:]):
#                 out[k][i, j] += 1

#     for k in range(nb_flags):
#         out[k] = out[k].tocsr()

#         if remove_diag:
#             out[k][np.diag_indices_from(out[k])] = 0

#     return out
