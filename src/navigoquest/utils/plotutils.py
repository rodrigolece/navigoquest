import itertools
import pickle
from copy import copy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import lines
from matplotlib.patches import Circle
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
import numpy as np
import pandas as pd

from .config import PLOT_CONFIG
from .figures import attach_source_data

title = PLOT_CONFIG["title"]
feat_types = PLOT_CONFIG["feat_types"]
short_title = PLOT_CONFIG["short_title"]


def _get_labels(style):
    labels = "abcdefghijklmnopqrstuvwxyz"
    if style == "lowercase":
        return labels
    elif style == "uppercase":
        return labels.upper()
    else:
        raise ValueError(f"Unknown style '{style}'")


class Background:
    """
    Background axes for Matplotlib figures, which can be used for layouting (when visible=True)
    and for plotting various annotations and markers that do not belong to any specific subplot.
    """

    def __init__(
        self, fig=None, visible=False, spacing=0.1, linecolor="0.5", linewidth=1
    ):
        """
        Args:
            fig:             Matplotlib figure. If None, the current one is used.
            visible (bool):  Show the grid if True.
            spacing:         Spacing of the background grid. Irrelevant if visible=False.
            linecolor:       Default color of added lines.
            linewidth:       Default width of added lines.
        """

        if fig is not None:
            plt.scf(fig)
        ax = plt.axes([0, 0, 1, 1], facecolor=None, zorder=-1000)
        plt.xticks(np.arange(0, 1 + spacing / 2.0, spacing))
        plt.yticks(np.arange(0, 1 + spacing / 2.0, spacing))
        plt.grid()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        ax.autoscale(False)
        if not visible:
            plt.axis("off")
        self.axes = ax
        self.linecolor = linecolor
        self.linewidth = linewidth

    def vline(self, x, y0=0, y1=1, **args):
        "Place a vertical line at position x spanning between y0 and y1."

        defargs = dict(color=self.linecolor, linewidth=self.linewidth)
        defargs.update(args)
        self.axes.add_line(lines.Line2D([x, x], [y0, y1], **defargs))

    def hline(self, y, x0=0, x1=1, **args):
        "Place a horizontal line at position y spanning between x0 and x1."

        defargs = dict(color=self.linecolor, linewidth=self.linewidth)
        defargs.update(args)
        self.axes.add_line(lines.Line2D([x0, x1], [y, y], **defargs))

    def box(self, pos, title=None, titlestyle=None, pad=0.0, **args):
        """Draw a box with optional title.

        Args:
            pos:         (left, right, bottom, top) axes coordinates.
            title:       Optional box title.
            titlestyle:  Dict with arguments passed to plt.text().
            pad:         Padding size in axes coordinates.
        """

        plt.sca(self.axes)
        width = pos[1] - pos[0]
        height = pos[3] - pos[2]

        defargs = dict(ec=self.linecolor, linewidth=self.linewidth, fc="none")
        defargs.update(args)

        fancy = matplotlib.patches.FancyBboxPatch(
            (pos[0], pos[2]), width, height, boxstyle=f"round,pad={pad}", **defargs
        )
        self.axes.add_patch(fancy)

        if title:
            titleargs = dict(
                ha="left",
                va="center",
                backgroundcolor="w",
                color=self.linecolor,
                fontsize=12,
            )
            if titlestyle is not None:
                titleargs.update(titlestyle)

            plt.text(pos[0] + 0.02, pos[3] + pad, title, **titleargs)

    def add_labels(self, xs, ys, fontsize=18, style="uppercase", labels=None):
        """Place panel labels at positions given by x-coordinates and y-coordinates.

        Args:
            xs:       x-coordinates of lower left corner of the labels.
            ys:       y-coordinates of lower left corner of the labels.
            fontsize: Font size.
            style:    Either 'uppercase' or 'lowercase'.
            labels:   If provided, these labels are used. Overrides style.
        """

        if labels is None:
            labels = _get_labels(style)

        assert len(xs) == len(ys)
        for x, y, label in zip(xs, ys, labels):
            self.axes.text(
                x,
                y,
                label,
                transform=self.axes.transAxes,
                size=fontsize,
                weight="bold",
                ha="left",
                va="bottom",
            )


def add_panel_labels(
    fig=None, axes=None, fontsize=18, xs=-0.05, ys=1.05, style="uppercase", labels=None
):
    """Place panel labels to given (or all) axes, relative to their upper left corner.

    Args:
        fig:       Matplotlib figure. If None, the current one is used.
        axes:      Axes to which place the labels. If None, all axes in fig are used.
        fontsize:  Font size.
        xs:        x-coordinate(s) of the label lower left corners. In axes coordinates.
                   Can be either float (then used for all axes), or list of floats, one for every axes.
        ys:        y-coordinate(s) of the label lower left corners. In axes coordinates.
                   Can be either float (then used for all axes), or list of floats, one for every axes.
        style:    Either 'uppercase' or 'lowercase'.
        labels:   If provided, these labels are used. Overrides style.
    """

    if labels is None:
        labels = _get_labels(style)

    if fig is None:
        fig = plt.gcf()

    if axes is None:
        axes = fig.get_axes()

    if not hasattr(xs, "__iter__"):
        xs = itertools.repeat(xs)
    if not hasattr(ys, "__iter__"):
        ys = itertools.repeat(ys)

    for i, (ax, x, y) in enumerate(zip(axes, xs, ys)):
        ax.text(
            x,
            y,
            labels[i],
            transform=ax.transAxes,
            size=fontsize,
            weight="bold",
            ha="left",
            va="bottom",
        )


def axtext(ax, text, **args):
    """Place text in the middle of given axes, and remove everything else.

    Occasionaly useful for placing labels of subplot grids.
    """

    defargs = {"fontsize": 12, "ha": "center", "va": "center"}
    defargs.update(args)
    plt.sca(ax)
    plt.text(0.5, 0.5, text, **defargs)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.axis("off")


def bottomleft_spines(ax):
    """Hide the right and top spines."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


## ------------------------------------------------------
## Project specific functions
## ------------------------------------------------------


def plot_quartiles(ax, df, col, label=None):
    data = df.groupby("age")[col].describe()
    ln = ax.plot(data["50%"], "-", label=label)
    # TODO: this is where I need to fix the legend so that it includes shaded areas
    poly = ax.fill_between(data.index, data["25%"], data["75%"], alpha=0.1)

    ax.set_xlabel("Age")
    ax.set_ylabel(title[col])

    return ln[0], poly


def plot_quartiles_by_vo(
    ax,
    filename,
    col,
    legend=True,
    loc="upper left",
    verbose=False,
):
    if verbose:
        print("Loading: ", filename)

    with open(filename, "rb") as f:
        data = pickle.load(f)

    df, vo_idx = data["df"], data["idx"]
    correct, incorrect = df.loc[vo_idx], df.loc[~vo_idx]

    art1 = plot_quartiles(ax, correct, col)
    art2 = plot_quartiles(ax, incorrect, col)

    # Source data: the plotted median and interquartile range per age
    tables = []
    for group, sub in (("Correct", correct), ("Incorrect", incorrect)):
        stats = sub.groupby("age")[col].describe()[["count", "25%", "50%", "75%"]]
        stats.columns = ["n", "q25", "median", "q75"]
        stats.insert(0, "visiting_order", group)
        tables.append(stats.reset_index())
    table = pd.concat(tables, ignore_index=True)
    table.insert(0, "metric", col)
    attach_source_data(ax, table)

    if legend:
        handles = [art1, art2]
        labels = ["Correct", "Incorrect"]
        ax.legend(handles, labels, loc=loc)


def plot_roc_curves(
    ax,
    roc_xy,
    col,
    levels=[6, 8, 11],
    legend=True,
    loc="lower right",
    verbose=False,
):

    tables = []
    for lvl in levels:
        # xy = roc_xy[lvl][col]
        xy = roc_xy[(lvl, col)]
        ax.plot(*xy, label=f"Level {lvl}", lw=2.5)
        fpr, tpr = xy
        tables.append(pd.DataFrame({"metric": col, "level": lvl, "fpr": fpr, "tpr": tpr}))

    # Source data: the plotted ROC curves
    attach_source_data(ax, pd.concat(tables, ignore_index=True))

    ax.set_xlim((-0.1, 1.1))
    ax.set_ylim((-0.1, 1.1))
    ax.set_aspect("equal")
    x, X, y, Y = ax.axis()
    mM = max(x, y), min(X, Y)
    ax.plot(mM, mM, ls="dotted", c=".3", alpha=0.5)  # , label="y=x (diagonal)")

    ax.set_yticks([0, 0.5, 1.0])
    ax.set_title(title[col])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")

    if legend:
        ax.legend(loc=loc)  # fontsize=18


def plot_auc_bars(
    ax,
    df,
    title=None,
    levels=[6, 8, 11],
    legend=True,
    leg_offset=0.5,
):
    # Source data: AUC values and confidence intervals of the plotted bars
    attach_source_data(ax, df.loc[df.level.isin(levels)].reset_index(drop=True))

    x = np.arange(len(feat_types))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    leg_handles = []

    for lvl in levels:
        metric_values = df.loc[df.level == lvl].set_index("metric")

        auc = metric_values["auc"].values

        # err = metric_values["std"].values
        low = metric_values["auc"] - metric_values["CI_low"]
        high = metric_values["CI_high"] - metric_values["auc"]
        err = np.vstack((low.values, high.values))

        offset = width * multiplier
        rects = ax.bar(x + offset, auc, width)
        leg_handles.append(rects[0])

        ax.errorbar(x + offset, auc, yerr=err, fmt="none", color="k")
        multiplier += 1

    ax.set_xticks(x + width)
    ax.set_xticklabels(short_title.values())
    ax.set_ylabel("AUC")
    ax.set_title(title)

    if legend:
        x, X = ax.get_xlim()
        ax.set_xlim(x, X + leg_offset)
        ax.legend(leg_handles, levels, loc="upper right")


def radar_factory(num_vars, frame="circle"):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = "radar"
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")
            self.set_theta_direction("clockwise")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill_between so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def radar(
    ax,
    data,
    theta,
    groups,
    colors,
    lw=1,
    alpha=0.18,
    radii=None,
    labels=None,
    legend=False,
):
    # Source data: the plotted values per group (rows) and metric (columns)
    attach_source_data(ax, data.loc[groups])

    for gp, c in zip(groups, colors):
        ax.plot(theta, data.loc[gp], color=c, lw=lw)
        ax.fill(theta, data.loc[gp], color=c, alpha=alpha, label=gp)
        # ax.set_rmax(rmax)
        ax.set_rgrids(radii, labels)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        hs = []
        for h in handles:
            h = copy(h)
            h.set_alpha(0.5)
            hs.append(h)
            # handles[0].set_linestyle("-")
        ax.legend(hs, labels, loc=(0.9, 1.1))
