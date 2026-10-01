"""Report frequentist tests in the format requested by the journal, and save them.

The editor asked for every frequentist inferential claim to be reported as

    Statistic (degrees of freedom) = value, p = value,
    effect size statistic = value, % confidence interval = values

for example ``t(212.4) = -8.31, p = 3.02 × 10⁻¹⁴, Cohen's d = -1.86, 95% CI = [-2.21, -1.51]``.

Each test function below returns one row (a dict) with every number of the test.
``save_tests`` turns a list of rows into a table and writes

* ``statistical_tests/<name>.csv``: one row per test, all numbers at full precision;
* ``statistical_tests/<name>.txt``: one formatted sentence per test, after a header
  saying which test, effect size and confidence intervals were used.

The folder is set in ``notebooks/plot_config.toml`` (section ``[statistical_tests]``).
By default it is ``statistical_tests/`` at the repository root, next to ``figures/``.

Tests
-----
welch_ttest
    Welch's two-sample t-test (unequal variances). Effect size: Cohen's d (pooled SD)
    with its confidence interval. The difference in means and its Welch confidence
    interval are also reported.
auc_test
    Area under the ROC curve against chance (AUC = 0.5): z-test with the DeLong
    standard error. Effect size: the AUC, with its DeLong confidence interval.
delong_paired_test
    Paired DeLong test comparing two correlated AUCs, as computed by
    ``pROC::roc.test(..., method = "delong", paired = TRUE)``. Effect size: the
    difference in AUC, with its confidence interval.

``adjust_pvalues`` adds Bonferroni or Benjamini-Hochberg adjusted p-values, which are
then quoted in the report next to the unadjusted p-value.
"""

from __future__ import annotations

import math
import pathlib
import tomllib
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import special

from ..stats.auc import delong_roc_variance
from .figures import find_config


DEFAULT_DIR = "../statistical_tests"  # relative to the folder of plot_config.toml
DEFAULT_CI_LEVEL = 0.95

# Columns written by the test functions, in order. Any other column of a row is a label
# (analysis, level, metric, ...) and is written first.
RESULT_COLUMNS = [
    "test",
    "group1",
    "group2",
    "n1",
    "n2",
    "mean1",
    "sd1",
    "mean2",
    "sd2",
    "statistic_name",
    "df",
    "statistic",
    "p",
    "p_log10",
    "effect_size_name",
    "effect_size",
    "effect_size_ci_low",
    "effect_size_ci_high",
    "estimate_name",
    "estimate",
    "estimate_ci_low",
    "estimate_ci_high",
    "auc1",
    "auc1_ci_low",
    "auc1_ci_high",
    "auc2",
    "auc2_ci_low",
    "auc2_ci_high",
    "ci_level",
]

_ADJUSTMENTS = {
    "bonferroni": "Bonferroni-adjusted p",
    "bh": "Benjamini-Hochberg-adjusted p",
}
_ADJUSTMENT_COLUMNS = [
    f"{prefix}{key}{suffix}"
    for key in _ADJUSTMENTS
    for prefix, suffix in [("p_", ""), ("p_", "_log10"), ("n_tests_", "")]
]

_METHOD_NOTES = {
    "Welch's t-test": (
        "Welch's t-test: two-sided, unequal variances; df from the Welch-Satterthwaite "
        "equation. Effect size: Cohen's d = (mean1 - mean2) / pooled SD, with a CI from "
        "the large-sample standard error of d (Hedges & Olkin, 1985). The difference in "
        "means is also given, with its Welch CI."
    ),
    "AUC vs. chance (DeLong)": (
        "AUC vs. chance: two-sided z-test of AUC = 0.5, z = (AUC - 0.5) / SE, with the "
        "DeLong standard error (no degrees of freedom: the reference distribution is the "
        "standard normal). Effect size: the AUC, with its DeLong CI (capped to [0, 1]). "
        "group1 is the positive class, whose scores are expected to be higher."
    ),
    "Paired DeLong test": (
        "Paired DeLong test of two correlated AUCs (pROC::roc.test, method = 'delong', "
        "paired = TRUE): two-sided z-test, no degrees of freedom. Effect size: the "
        "difference in AUC, with the CI difference ± z_crit × SE, where SE = difference / z."
    ),
}

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _config() -> tuple[dict, pathlib.Path]:
    """The ``[statistical_tests]`` section of plot_config.toml and the folder it is in."""
    config_path = find_config()
    if config_path is None:
        return {}, pathlib.Path.cwd()
    with open(config_path, "rb") as f:
        section = tomllib.load(f).get("statistical_tests", {})
    return section, config_path.parent


def output_dir() -> pathlib.Path | None:
    """Folder the test results are written to, or None if writing is disabled."""
    section, base = _config()
    if not section.get("enabled", True):
        return None
    folder = pathlib.Path(section.get("dir", DEFAULT_DIR)).expanduser()
    return (folder if folder.is_absolute() else base / folder).resolve()


def default_ci_level() -> float:
    return float(_config()[0].get("ci_level", DEFAULT_CI_LEVEL))


def _ci_level(ci_level: float | None) -> float:
    level = default_ci_level() if ci_level is None else float(ci_level)
    if not 0 < level < 1:
        raise ValueError(f"ci_level must be between 0 and 1, got {level}")
    return level


def _z_crit(level: float) -> float:
    return float(st.norm.ppf(0.5 + level / 2))


# ---------------------------------------------------------------------------
# p-values that underflow
# ---------------------------------------------------------------------------


def _log10_p_normal(z: float) -> float:
    """log10 of the two-sided p-value of a standard normal statistic (no underflow)."""
    if not np.isfinite(z):
        return np.nan
    return float((math.log(2) + st.norm.logsf(abs(z))) / math.log(10))


def _log10_p_t(t: float, df: float) -> float:
    """log10 of the two-sided p-value of a t statistic, also when it underflows to 0.

    The two-sided p-value is the regularised incomplete beta function I_x(df/2, 1/2),
    with x = df / (df + t^2). When it underflows it is computed in log space from
    I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * 2F1(a + b, 1; a + 1; x) (DLMF 8.17.8).
    """
    if not (np.isfinite(t) and np.isfinite(df)):
        return np.nan
    p = 2 * st.t.sf(abs(t), df)
    if p > 0:
        return float(math.log10(p))
    a, b = df / 2, 0.5
    x = df / (df + t * t)
    log_p = (
        a * math.log(x)
        + b * math.log1p(-x)
        - math.log(a)
        - special.betaln(a, b)
        + math.log(special.hyp2f1(a + b, 1, a + 1, x))
    )
    return float(log_p / math.log(10))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _clean(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def welch_ttest(
    a,
    b,
    *,
    group1: str = "group 1",
    group2: str = "group 2",
    ci_level: float | None = None,
    **labels,
) -> dict:
    """Welch's two-sample t-test of ``a`` vs. ``b``, with Cohen's d and CIs.

    Missing values (NaN) are dropped. The mean difference and Cohen's d are
    ``a - b``, as is the sign of t. Keyword arguments other than the ones listed
    (e.g. ``level=6, metric="duration"``) are stored as labels of the row.
    """
    level = _ci_level(ci_level)
    a, b = _clean(a), _clean(b)
    n1, n2 = len(a), len(b)

    res = st.ttest_ind(a, b, equal_var=False)
    ci = res.confidence_interval(confidence_level=level)

    mean1, mean2 = float(np.mean(a)), float(np.mean(b))
    sd1, sd2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    pooled_sd = math.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    d = (mean1 - mean2) / pooled_sd
    se_d = math.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    z = _z_crit(level)

    return _with_report(
        {
            **labels,
            "test": "Welch's t-test",
            "group1": group1,
            "group2": group2,
            "n1": n1,
            "n2": n2,
            "mean1": mean1,
            "sd1": sd1,
            "mean2": mean2,
            "sd2": sd2,
            "statistic_name": "t",
            "df": float(res.df),
            "statistic": float(res.statistic),
            "p": float(res.pvalue),
            "p_log10": _log10_p_t(float(res.statistic), float(res.df)),
            "effect_size_name": "Cohen's d",
            "effect_size": d,
            "effect_size_ci_low": d - z * se_d,
            "effect_size_ci_high": d + z * se_d,
            "estimate_name": "mean difference",
            "estimate": mean1 - mean2,
            "estimate_ci_low": float(ci.low),
            "estimate_ci_high": float(ci.high),
            "ci_level": level,
        }
    )


def auc_test(
    label,
    score,
    *,
    positive: str = "label 1",
    negative: str = "label 0",
    ci_level: float | None = None,
    **labels,
) -> dict:
    """AUC of ``score`` for separating ``label == 1`` (positive) from ``label == 0``,
    tested against chance (0.5) with the DeLong standard error.

    Uses the same DeLong implementation (``navigoquest.stats``) as ``compute_auc``,
    so the AUC and its CI are the ones plotted in the figures.
    """
    level = _ci_level(ci_level)
    label = np.asarray(label).astype(int).ravel()
    score = np.asarray(score, dtype=float).ravel()
    keep = ~np.isnan(score)
    label, score = label[keep], score[keep]

    auc, var = delong_roc_variance(label, score)
    auc, se = float(auc), float(np.sqrt(np.squeeze(var)))
    z_stat = (auc - 0.5) / se if se > 0 else np.nan
    z = _z_crit(level)

    return _with_report(
        {
            **labels,
            "test": "AUC vs. chance (DeLong)",
            "group1": positive,
            "group2": negative,
            "n1": int(np.sum(label == 1)),
            "n2": int(np.sum(label == 0)),
            "statistic_name": "z",
            "df": np.nan,
            "statistic": z_stat,
            "p": float(2 * st.norm.sf(abs(z_stat))) if np.isfinite(z_stat) else np.nan,
            "p_log10": _log10_p_normal(z_stat),
            "effect_size_name": "AUC",
            "effect_size": auc,
            "effect_size_ci_low": max(0.0, auc - z * se),
            "effect_size_ci_high": min(1.0, auc + z * se),
            "estimate_name": "AUC standard error",
            "estimate": se,
            "ci_level": level,
        }
    )


def delong_paired_test(
    auc1: float,
    auc2: float,
    z: float,
    p: float | None = None,
    *,
    name1: str = "AUC 1",
    name2: str = "AUC 2",
    auc1_se: float | None = None,
    auc2_se: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    ci_level: float | None = None,
    **labels,
) -> dict:
    """Paired DeLong comparison of two AUCs from their values and the z statistic.

    ``z`` is the DeLong statistic (auc1 - auc2) / SE, e.g. ``roc.test(...)$statistic``
    in pROC; ``p`` its p-value (recomputed from z when not given). ``n1`` and ``n2``
    are the numbers of positive and negative cases, if known.
    """
    level = _ci_level(ci_level)
    zc = _z_crit(level)
    auc1, auc2, z = float(auc1), float(auc2), float(z)
    diff = auc1 - auc2
    se = diff / z if z != 0 else np.nan
    p_from_z = float(2 * st.norm.sf(abs(z)))
    p = p_from_z if p is None or not np.isfinite(p) else float(p)

    row = {
        **labels,
        "test": "Paired DeLong test",
        "group1": name1,
        "group2": name2,
        "n1": n1 if n1 is not None else np.nan,
        "n2": n2 if n2 is not None else np.nan,
        "statistic_name": "z",
        "df": np.nan,
        "statistic": z,
        "p": p,
        "p_log10": math.log10(p) if p > 0 else _log10_p_normal(z),
        "effect_size_name": f"ΔAUC ({name1} − {name2})",
        "effect_size": diff,
        "effect_size_ci_low": diff - zc * abs(se),
        "effect_size_ci_high": diff + zc * abs(se),
        "auc1": auc1,
        "auc2": auc2,
        "ci_level": level,
    }
    for key, auc, auc_se in [("auc1", auc1, auc1_se), ("auc2", auc2, auc2_se)]:
        if auc_se is not None:
            row[f"{key}_ci_low"] = max(0.0, auc - zc * float(auc_se))
            row[f"{key}_ci_high"] = min(1.0, auc + zc * float(auc_se))
    return _with_report(row)


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------


def adjust_pvalues(
    tests: pd.DataFrame | Iterable[Mapping],
    method: str = "bonferroni",
    *,
    n_tests: int | None = None,
    where=None,
) -> pd.DataFrame:
    """Add adjusted p-values (columns ``p_<method>``) and refresh the reports.

    method: ``"bonferroni"`` (p × n_tests, capped at 1) or ``"bh"`` (Benjamini-Hochberg
    step-up adjusted p-values, i.e. q-values).
    n_tests: size of the family. Defaults to the number of tests adjusted.
    where: boolean mask of the rows forming the family; the other rows get NaN.
    """
    if method not in _ADJUSTMENTS:
        raise ValueError(f"method must be one of {list(_ADJUSTMENTS)}, got {method!r}")
    df = tests_frame(tests)
    mask = np.ones(len(df), dtype=bool) if where is None else np.asarray(where, dtype=bool)

    log_p = df.loc[mask, "p_log10"].to_numpy(dtype=float)
    m = int(mask.sum()) if n_tests is None else int(n_tests)

    if method == "bonferroni":
        log_adj = log_p + math.log10(m)
    else:  # Benjamini-Hochberg, in log space so that underflowed p-values are kept
        order = np.argsort(log_p)
        ranks = np.arange(1, len(log_p) + 1)
        scaled = log_p[order] + np.log10(m / ranks)
        stepped = np.minimum.accumulate(scaled[::-1])[::-1]
        log_adj = np.empty_like(stepped)
        log_adj[order] = stepped
    log_adj = np.minimum(log_adj, 0.0)

    df[f"p_{method}"] = np.nan
    df[f"p_{method}_log10"] = np.nan
    df[f"n_tests_{method}"] = np.nan
    df.loc[mask, f"p_{method}"] = 10.0**log_adj
    df.loc[mask, f"p_{method}_log10"] = log_adj
    df.loc[mask, f"n_tests_{method}"] = m
    return add_reports(df)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def format_p(p: float, log10_p: float | None = None) -> str:
    """p-value with 3 significant figures; scientific notation below 0.001.

    Uses ``log10_p`` for p-values that underflow to 0 in double precision.
    """
    if _isnan(p) and (log10_p is None or _isnan(log10_p)):
        return "NA"
    if not _isnan(p) and p >= 0.001:
        return f"{min(p, 1.0):.3g}"
    if log10_p is None or _isnan(log10_p) or not np.isfinite(log10_p):
        log10_p = math.log10(p) if p > 0 else -np.inf
    if not np.isfinite(log10_p):
        return "< 1 × 10⁻³⁰⁰"
    exponent = math.floor(log10_p)
    mantissa = 10 ** (log10_p - exponent)
    if round(mantissa, 2) >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    return f"{mantissa:.2f} × 10{str(exponent).translate(_SUPERSCRIPT)}"


def _p_clause(p_text: str) -> str:
    return f"p {p_text}" if p_text.startswith("<") else f"p = {p_text}"


def _decimals(*values, sig: int = 3) -> int:
    """Decimals giving ``sig`` significant figures to the largest of ``values``."""
    finite = [abs(v) for v in values if not _isnan(v) and np.isfinite(v) and v != 0]
    if not finite:
        return 2
    return max(0, sig - 1 - math.floor(math.log10(max(finite))))


def _num(x, decimals: int) -> str:
    if _isnan(x):
        return "NA"
    return f"{x:.{decimals}f}".replace("-", "−")


def _ci(low, high, decimals: int) -> str:
    return f"[{_num(low, decimals)}, {_num(high, decimals)}]"


def format_report(row: Mapping) -> str:
    """The sentence reporting one test, in the journal's format."""
    pct = f"{100 * row['ci_level']:g}%"

    # Statistic (degrees of freedom) = value
    stat_dec = 2 if abs(row["statistic"]) >= 0.01 or _isnan(row["statistic"]) else 3
    stat = f"{row['statistic_name']}"
    if not _isnan(row.get("df")):
        df = row["df"]
        stat += f"({df:.0f})" if float(df).is_integer() else f"({df:.1f})"
    stat += f" = {_num(row['statistic'], stat_dec)}"

    # p = value (and adjusted p-values)
    p_text = _p_clause(format_p(row["p"], row.get("p_log10")))
    adjusted = []
    for key, label in _ADJUSTMENTS.items():
        value = row.get(f"p_{key}")
        if _isnan(value):
            continue
        text = format_p(value, row.get(f"p_{key}_log10"))
        adjusted.append(f"{label} {text}" if text.startswith("<") else f"{label} = {text}")
    if adjusted:
        p_text += " (" + "; ".join(adjusted) + ")"

    # effect size statistic = value, % CI = values
    es_dec = 3 if row["effect_size_name"].startswith(("AUC", "ΔAUC")) else 2
    effect = (
        f"{row['effect_size_name']} = {_num(row['effect_size'], es_dec)}, "
        f"{pct} CI = {_ci(row['effect_size_ci_low'], row['effect_size_ci_high'], es_dec)}"
    )

    report = f"{stat}, {p_text}, {effect}"

    # Supporting estimates
    if row["test"] == "Welch's t-test":
        dec = _decimals(row["estimate"], row["estimate_ci_low"], row["estimate_ci_high"])
        report += (
            f"; mean difference ({row['group1']} − {row['group2']}) = "
            f"{_num(row['estimate'], dec)}, {pct} CI = "
            f"{_ci(row['estimate_ci_low'], row['estimate_ci_high'], dec)}"
            f"; n = {row['n1']:.0f} vs. {row['n2']:.0f}"
        )
    elif row["test"] == "AUC vs. chance (DeLong)":
        report += (
            f"; {row['group1']} (n = {row['n1']:.0f}) vs. {row['group2']} (n = {row['n2']:.0f})"
        )
    elif row["test"] == "Paired DeLong test":
        parts = []
        for key, name in [("auc1", row["group1"]), ("auc2", row["group2"])]:
            part = f"AUC {name} = {_num(row[key], 3)}"
            if not _isnan(row.get(f"{key}_ci_low")):
                part += f", {pct} CI = {_ci(row[f'{key}_ci_low'], row[f'{key}_ci_high'], 3)}"
            parts.append(part)
        report += "; " + "; ".join(parts)
    return report


def _with_report(row: dict) -> dict:
    """Add the formatted ``report`` to a test row (and make it the last entry)."""
    row.pop("report", None)
    full = {col: np.nan for col in RESULT_COLUMNS} | row
    row["report"] = format_report(full)
    return row


def _label_columns(df: pd.DataFrame) -> list[str]:
    known = set(RESULT_COLUMNS) | set(_ADJUSTMENT_COLUMNS) | {"report"}
    return [c for c in df.columns if c not in known]


# Columns left out of the table when no test in it uses them
_OPTIONAL_COLUMNS = [
    "mean1",
    "sd1",
    "mean2",
    "sd2",
    "estimate_name",
    "estimate",
    "estimate_ci_low",
    "estimate_ci_high",
    "auc1",
    "auc1_ci_low",
    "auc1_ci_high",
    "auc2",
    "auc2_ci_low",
    "auc2_ci_high",
]


def add_reports(df: pd.DataFrame) -> pd.DataFrame:
    """(Re)compute the ``report`` column and put the columns in a standard order."""
    df = df.copy()
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["report"] = [format_report(row) for row in df.to_dict("records")]
    unused = [c for c in _OPTIONAL_COLUMNS if df[c].isna().all()]
    labels = _label_columns(df)
    results = [
        c for c in RESULT_COLUMNS + _ADJUSTMENT_COLUMNS if c in df.columns and c not in unused
    ]
    return df[labels + results + ["report"]]


def tests_frame(tests: pd.DataFrame | Iterable[Mapping]) -> pd.DataFrame:
    """Table of tests (one row per test, from a list of test rows) with a ``report`` column."""
    if isinstance(tests, pd.DataFrame):
        df = tests.reset_index(drop=True)
    else:
        df = pd.DataFrame(list(tests))
    return add_reports(df)


def _prefix(row: Mapping, labels: list[str]) -> str:
    parts = []
    for col in labels:
        value = row[col]
        if _isnan(value):
            continue
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        parts.append(f"{col} = {value}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_tests(
    tests: pd.DataFrame | Iterable[Mapping],
    name: str,
    *,
    title: str | None = None,
    description: str | None = None,
    show: bool = True,
) -> pd.DataFrame:
    """Write ``<name>.csv`` and ``<name>.txt`` to the statistical_tests folder.

    Returns the table of tests (with the ``report`` column). With ``show=True`` the
    formatted reports are also printed.
    """
    df = tests_frame(tests)
    labels = _label_columns(df)
    lines = [f"[{_prefix(row, labels)}] {row['report']}" for row in df.to_dict("records")]

    header = [title or name, "=" * len(title or name)]
    if description:
        header += ["", description.strip()]
    header += [""]
    for test in df["test"].dropna().unique():
        if test in _METHOD_NOTES:
            header.append(_METHOD_NOTES[test])
    notes = []
    for key, label in _ADJUSTMENTS.items():
        col = f"n_tests_{key}"
        if col in df.columns and df[col].notna().any():
            sizes = ", ".join(f"{n:.0f}" for n in sorted(df[col].dropna().unique()))
            notes.append(f"{label}: family of {sizes} tests.")
    if notes:
        header.append("Multiple comparisons. " + " ".join(notes))
    header.append(
        "p-values are two-sided and unadjusted unless stated. Signs follow "
        "group1 − group2. Full-precision values are in the .csv file with the same name."
    )
    text = "\n".join(header) + "\n\n" + "\n".join(lines) + "\n"

    folder = output_dir()
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(folder / f"{name}.csv", index=False)
        (folder / f"{name}.txt").write_text(text, encoding="utf-8")

    if show:
        where = f" -> {folder / name}.csv/.txt" if folder is not None else " (not saved)"
        print(f"{title or name}{where}")
        print("\n".join(lines))
    return df
