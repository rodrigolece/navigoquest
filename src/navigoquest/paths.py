from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d


def smooth_path(
    arr: NDArray[np.int32], spline_res: int = 3, bandwidth: float = 1.67
) -> NDArray[np.float64]:
    N = len(arr)
    xs = np.arange(1 + (N - 1) * spline_res) / spline_res
    interpolation = np.vstack([np.interp(xs, range(N), pt) for pt in arr.T]).T
    return gaussian_filter1d(interpolation, spline_res * bandwidth, axis=0)


@dataclass
class Path:
    xy: NDArray[np.int32]
    metadata: dict
    _smooth_xy: NDArray[np.float64] | None = None

    @property
    def smooth_xy(self) -> NDArray[np.float64]:
        if self._smooth_xy is None:
            # self._smooth_xy = self._initialize_smooth_xy()
            self._smooth_xy = smooth_path(self.xy)
        return self._smooth_xy

    # def _initialize_smooth_xy(self) -> NDArray[np.float64]:
    #     N = len(self.xy)
    #     xs = np.arange(1 + (N - 1) * spline_res) / spline_res
    #     interpolation = np.vstack([np.interp(xs, range(N), pt) for pt in self.xy.T]).T
    #     return gaussian_filter1d(interpolation, spline_res * bandwidth, axis=0)


class PathDataset:
    paths: list[Path]
    user_metadata: pd.DataFrame
    metadata_cols: list[str]

    def __init__(
        self,
        paths: list[Path],
        user_metadata: pd.DataFrame,
        metadata_cols: list[str],
    ):
        self.paths = paths
        self.user_metadata = user_metadata
        self.metadata_cols = metadata_cols

    @classmethod
    def from_csv(
        cls,
        paths_filename: str | pathlib.Path,
        metadata_filename: str | pathlib.Path,
        metadata_cols: list[str],
        paths_id_col: str = "user_id",
        metadata_id_col: str = "id",
        na_policy: str = "drop",
    ) -> PathDataset:
        paths, user_ids = cls._paths_from_csv(paths_filename)

        user_metadata_df = pd.read_csv(metadata_filename)  # .rename(columns={"id": paths_id_col})
        merged_df = pd.merge(
            user_ids, user_metadata_df, left_on=paths_id_col, right_on=metadata_id_col, how="left"
        )

        na_rows = merged_df[metadata_id_col].isna()

        if na_policy == "drop":
            print(f"dropping {na_rows.sum()} unmatched path(s)")
            merged_df = merged_df.loc[~na_rows]

            # filter out paths with unmatched metadata, using the original index
            paths = [paths[i] for i in merged_df.index]

            # Re-align the paths to the metadata
            merged_df = merged_df.reset_index(drop=True)

        else:
            raise NotImplementedError(f"na_policy {na_policy} not implemented")

        if len(paths) != len(merged_df):
            raise ValueError(
                f"number of paths ({len(paths)}) does not match number of metadata rows ({len(merged_df)})"
            )

        for i, path in enumerate(paths):
            subject_metadata = merged_df.iloc[i][metadata_cols].to_dict()
            path.metadata.update(subject_metadata)

        return cls(paths, merged_df, metadata_cols)

    @staticmethod
    def _paths_from_csv(
        filename: str | pathlib.Path,
        id_col: str = "user_id",
        path_col: str = "trajectory_data",
    ) -> tuple[list[Path], pd.Series]:
        if isinstance(filename, str):
            filename = pathlib.Path(filename)

        assert filename.exists(), f"file {filename} does not exist"

        df = pd.read_csv(filename)
        # TODO: this is where we should deal with instance_id and duplicated entries

        for col in [id_col, path_col]:
            assert col in df.columns, f"column {col} not found in {filename}"

        # metadata_cols = [col for col in df.columns if col not in [id_col, path_col]]

        paths = []
        user_ids = []

        for i, row in df.iterrows():
            path = path_json2array(row[path_col])

            if path is None:
                print(f"corrupted data for entry: {i}")
                continue

            user_ids.append(row[id_col])
            metadata = {col: row[col] for col in ["instance_id", "user_id", "duration"]}
            paths.append(Path(path, metadata))

        return paths, pd.Series(user_ids).rename(id_col)

    def filter_by_attributes(
        self,
        age: int,
        gender: str,
        #  country: str = "GB",
    ) -> Iterator[Path]:
        age_idx = self.user_metadata["age"] == age
        gender_idx = self.user_metadata["gender"] == gender
        # country_idx = self.user_metadata["country"] == country

        idx = age_idx & gender_idx  # & country_idx

        return (self.paths[i] for i in self.user_metadata.loc[idx].index)

    def group_by(self, *attributes: str):  # -> dict[tuple, tuple[int, Iterator[Path]]]:
        """Group paths by one or more user metadata attributes.

        Parameters
        ----------
        *attributes : str
            One or more column names from user_metadata to group by

        Returns
        -------
        dict[tuple, tuple[int, Iterator[Path]]]
            Dictionary where keys are tuples of attribute values and
            values are tuples of (group_size, iterator_over_matching_paths)
        """
        # Group the dataframe by the specified attributes
        grouped = self.user_metadata.groupby(list(attributes))
        group_sizes = grouped.size()

        for group_key, group_df in grouped:
            N = group_sizes[group_key]
            paths = (self.paths[i] for i in group_df.index)
            yield group_key, (N, paths)


def path_json2array(json_str: str) -> NDArray[np.int32] | None:
    """
    Convert a JSON string to an Nx2 numpy array.

    Parameters
    ----------
    json_str : str
        The JSON string to convert.

    Returns
    -------
    NDArray[np.int32] | None
        The numpy array. If the path is corrupted, returns None.§
    """
    player_data = json.loads(json_str)["player"]

    if isinstance(player_data, dict):
        # The path is corrupted, the path is a single point (and usually x is None)
        out = None
    else:
        out = np.array([(pt["x"], pt["y"]) for pt in player_data])

    return out
