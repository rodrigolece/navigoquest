import pathlib

import pandas as pd


def normalize_clinical_results(
    df: pd.DataFrame,
    feat_types: list[str],
    idx_cols: list[str] = ["id", "group"],
) -> pd.DataFrame:
    gby = df.groupby("level")

    # compute the correction terms
    one = gby.get_group(1).set_index(idx_cols)
    two = gby.get_group(2).set_index(idx_cols)
    norm_factor = (one["path_length"] + two["path_length"]).abs()  # keep only magnitude

    out = []

    for lvl in [6, 8, 11]:
        lvl_df = gby.get_group(lvl).set_index(idx_cols)
        lvl_df.loc[:, feat_types] = lvl_df[feat_types].divide(norm_factor, axis=0)
        out.append(lvl_df.reset_index())

    return pd.concat(out, ignore_index=True)


def normalize_level_results(
    metrics_dir: pathlib.Path,
    level: int,
    feat_types: list[str],
    id_col: str = "user_id",
):
    df = pd.read_csv(metrics_dir / f"metrics_level{level:02}_gb_2480.csv")
    ref_ids = df[id_col]

    one = (
        pd.read_csv(metrics_dir / "metrics_level01_gb_2480.csv").set_index(id_col).reindex(ref_ids)
    )
    two = (
        pd.read_csv(metrics_dir / "metrics_level01_gb_2480.csv").set_index(id_col).reindex(ref_ids)
    )

    norm_factor = (one["path_length"] + two["path_length"]).abs()

    df.set_index(id_col, inplace=True)
    df.loc[:, feat_types] = df[feat_types].divide(norm_factor, axis=0)

    return df.reset_index().dropna(subset="path_length")
