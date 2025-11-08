import numpy as np
import pandas as pd
import scipy.stats as st

from .auc import delong_roc_variance


def compute_pvalues(
    lvl_df: pd.DataFrame,
    feat_types: list[str],
    equal_var: bool = False,
) -> pd.Series:
    label = lvl_df["label"]
    gs = label.unique()
    assert len(gs) == 2

    idx = label == gs[0]
    first = lvl_df.loc[idx]
    second = lvl_df.loc[~idx]

    pvals = [
        st.ttest_ind(first[feat], second[feat], equal_var=equal_var).pvalue for feat in feat_types
    ]

    idx = pd.Index(feat_types, name="metric")

    return pd.Series(pvals, index=idx, name="pvals")


def confidence_interval(auc, std, alpha=0.95):
    offset = (1 - alpha) / 2
    percentiles = np.array([offset, 1 - offset])

    out = st.norm.ppf(percentiles, loc=auc, scale=std)
    out[out > 1] = 1

    return out


def _ci_wrapper(row, alpha=0.95):
    return confidence_interval(row["auc"], row["std"], alpha=alpha)


def compute_auc(lvl_df: pd.DataFrame, feat_types: list[str]) -> pd.DataFrame:
    label = lvl_df["label"].values  # delong_roc_var takes np.array
    vals = []

    for feat in feat_types:
        score = lvl_df[feat].values
        vals.append(delong_roc_variance(label, score))

    idx = pd.Index(feat_types, name="metric")
    out = pd.DataFrame(vals, columns=["auc", "var"], index=idx)

    out["std"] = out["var"].apply(np.sqrt)
    out[["CI_low", "CI_high"]] = out.apply(_ci_wrapper, axis=1, result_type="expand")

    return out.drop(columns="var")
