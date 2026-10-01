import pandas as pd
import numpy as np
import scipy.stats as st
from sklearn import metrics

from .config import PLOT_CONFIG
from .stat_report import auc_test, tests_frame, welch_ttest
from .. import (
    compute_pvalues,
    compute_auc,
    normalize_clinical_results,
    normalize_level_results,
)

feat_types = PLOT_CONFIG["feat_types"]


def load_level_data(metrics_dir, lvl, norm=False):
    if norm:
        df = normalize_level_results(metrics_dir, lvl, feat_types)
    else:
        df = pd.read_csv(metrics_dir / f"metrics_level{lvl:02}_gb_2480.csv")

    return df


def roc_curves(metrics_dir, feat_types, norm=False):
    roc_xy_dict = {}

    for lvl in [6, 8, 11]:
        df = load_level_data(metrics_dir, lvl, norm)
        age_filter = df.age >= 50
        df = df.loc[age_filter].copy()

        label = (~df["voc"].astype(bool)).astype(int)

        for feat in feat_types:
            score = df[feat].values

            fpr, tpr, _ = metrics.roc_curve(label, score)  # 3rd argument is thresholds
            roc_xy_dict[(lvl, feat)] = (fpr, tpr)

    return roc_xy_dict


def clinical_roc_curves(
    df, other, ref="e3e3", feat_types=feat_types, levels=(6, 8, 11)
):
    roc_xy_dict = {}

    groups = df.loc[df.group.isin([ref, other])].copy()
    groups["label"] = (groups["group"] == other).astype(int)

    # gby = groups.groupby("level")

    for lvl in levels:
        df = groups.loc[groups.level == lvl]

        label = df["label"]

        for feat in feat_types:
            score = df[feat].values

            fpr, tpr, _ = metrics.roc_curve(label, score)  # 3rd argument is thresholds
            roc_xy_dict[(lvl, feat)] = (fpr, tpr)

    return roc_xy_dict


def aucs(metrics_dir, feat_types, norm=False):
    out = []

    for lvl in [6, 8, 11]:
        df = load_level_data(metrics_dir, lvl, norm)
        age_filter = df.age >= 50
        df = df.loc[age_filter].copy()

        df["label"] = (~df["voc"].astype(bool)).astype(int)

        aucs_level = compute_auc(df, feat_types).reset_index()
        aucs_level["level"] = lvl
        out.append(aucs_level)

    return pd.concat(out).set_index(["level", "metric"])


def pvalues(metrics_dir, feat_types, norm=False):
    out = []

    for lvl in [6, 8, 11]:
        df = load_level_data(metrics_dir, lvl, norm)
        age_filter = df.age >= 50
        df = df.loc[age_filter].copy()

        df["label"] = (~df["voc"].astype(bool)).astype(int)

        pvals = compute_pvalues(df, feat_types).reset_index()
        pvals["level"] = lvl
        out.append(pvals)

    return pd.concat(out).set_index(["level", "metric"])


def voc_tests(metrics_dir, feat_types, norm=False, levels=(6, 8, 11)):
    """Statistical tests of incorrect vs. correct visiting order (players aged 50+).

    For each level and metric: Welch's t-test (the p-values of ``pvalues``) and the AUC
    against chance (the AUCs and CIs of ``aucs``), with degrees of freedom, effect sizes
    and confidence intervals. Returns a table for ``stat_report.save_tests``.
    """
    rows = []

    for lvl in levels:
        df = load_level_data(metrics_dir, lvl, norm)
        df = df.loc[df.age >= 50].copy()
        incorrect = ~df["voc"].astype(bool)

        for feat in feat_types:
            labels = {"normalised": norm, "level": lvl, "metric": feat}
            rows.append(
                welch_ttest(
                    df.loc[incorrect, feat],
                    df.loc[~incorrect, feat],
                    group1="incorrect VOC",
                    group2="correct VOC",
                    **labels,
                )
            )
            rows.append(
                auc_test(
                    incorrect.astype(int),
                    df[feat],
                    positive="incorrect VOC",
                    negative="correct VOC",
                    **labels,
                )
            )

    return tests_frame(rows)


def _fill_group(group_df, feat_types, ref_lvl=None):
    gby = group_df.groupby("level")
    keys = list(gby.groups.keys())

    if ref_lvl is None:
        # Select the lowest level which we assume holds the most ids
        ref_lvl = min(keys)

    # The reference data
    base_df = gby.get_group(ref_lvl).set_index("id")
    demo_cols = list(set(base_df.columns).difference(feat_types))
    demo_cols.remove("level")

    out = [base_df]

    keys.remove(ref_lvl)
    for k in keys:
        lvl = gby.get_group(k).set_index("id")

        extended = base_df[demo_cols].join(lvl[["level"] + feat_types])
        extended.loc[extended.level.isna(), "level"] = k  # fix the level
        extended.level = extended.level.astype(int)

        out.append(extended)

    out_df = pd.concat(out).reset_index()[group_df.columns]

    return out_df.sort_values(["level", "id"]).reset_index(drop=True)


def fill_missing_ad_attempts(df, feat_types, ref_lvl=None):
    gby = df.groupby("group")
    group_df = gby.get_group("ad")

    processed_df = _fill_group(group_df, feat_types, ref_lvl=ref_lvl)
    out = [processed_df]

    for key, group in gby:
        if key != "ad":
            out.append(group)

    return pd.concat(out).reset_index(drop=True)


def percentiles(metrics_dir, norm=False):
    df = pd.read_csv(metrics_dir / "clinical_metrics.csv")
    if norm:
        df = normalize_clinical_results(df, feat_types)
    else:
        df = df.loc[~df.level.isin([1, 2])]
    df = fill_missing_ad_attempts(df, feat_types).fillna(np.inf)

    out = df.copy()

    # percentile computation for each level and gender
    for lvl in [6, 8, 11]:
        ref_df = load_level_data(metrics_dir, lvl, norm=norm)

        for g in ["m", "f"]:
            idx = (out.level == lvl) & (out.gender == g)
            idx_ref = ref_df.gender == g  # & (ref_df.voc == 1)

            for i, row in out.loc[idx].iterrows():
                ref = ref_df.loc[idx_ref & (ref_df.age == row.age)]

                for col in feat_types:
                    scores, val = ref[col], row[col]
                    out.loc[i, col] = st.percentileofscore(scores, val, kind="weak")
                    # weak corresponds to the CDF definition

    return out


def clinical_aucs(df, other, ref="e3e3", feat_types=feat_types, levels=(6, 8, 11)):
    groups = df.loc[df.group.isin([ref, other])].copy()
    groups["label"] = (groups["group"] == other).astype(int)

    gby = groups.groupby("level")

    out = []

    for lvl in levels:
        lvl_df = gby.get_group(lvl).dropna(subset=feat_types)

        auc = compute_auc(lvl_df, feat_types).reset_index()
        auc["level"] = lvl
        out.append(auc)

    return pd.concat(out).set_index(["level", "metric"])


def clinical_auc_tests(df, other, ref="e3e3", feat_types=feat_types, levels=(6, 8, 11), **labels):
    """AUC of ``other`` vs. ``ref`` against chance, per level and metric.

    Same data and AUCs as ``clinical_aucs``, with the z statistic, p-value and CI.
    Extra keyword arguments are added as labels. Returns a table for
    ``stat_report.save_tests``.
    """
    groups = df.loc[df.group.isin([ref, other])].copy()
    groups["label"] = (groups["group"] == other).astype(int)

    gby = groups.groupby("level")

    rows = []

    for lvl in levels:
        lvl_df = gby.get_group(lvl).dropna(subset=feat_types)

        for feat in feat_types:
            rows.append(
                auc_test(
                    lvl_df["label"],
                    lvl_df[feat],
                    positive=other,
                    negative=ref,
                    **labels,
                    level=lvl,
                    metric=feat,
                )
            )

    return tests_frame(rows)
