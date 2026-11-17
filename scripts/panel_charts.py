"""How `fig:agreement`'s five panels are drawn, and how a panel is written out.

One chart vocabulary for the whole figure: blue points, a dashed grey ``y = x``,
a solid firebrick regression, grey median dashes. A reader who has learned to
read one panel can read the next, which is the point of putting them in one
figure.

The statistics are not here. Everything this module draws arrives as a frame
from ``scripts/panels.py``, so a number in a panel's corner is read from the
table the panel's CSV is written from rather than recomputed beside it.

Altair is in the ``dev`` dependency group, beside ``vl-convert-python``,
``marimo`` and ``snakemake``: the panels are produced by the pipeline, and the
pipeline runs only where ``snakemake`` is installed.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import FULL_DEPTH, assert_dasm2_is_this_repo, cap_of

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import math

import altair as alt
import matplotlib.figure
import numpy as np
import pandas as pd
from panels import figure_dir

# Re-exported, like the statistics through `scripts/panels.py`: a notebook then
# imports its charts from one module and never names `dasm2` itself, which is
# what keeps `tests/test_import_order.py`'s two orderings a property of this
# layer rather than of every notebook that draws a panel.
# `noqa`, not a removal: nothing in this module draws with it any more, but
# `notebooks/agreement_mutations.py` and `tests/test_panels.py` reach it as
# `charts.mutation_lookup_line`, and `--fix` would delete the re-export and
# break both.
from dasm2.analysis.charts import mutation_lookup_line  # noqa: F401

#: Everything is written at one resolution, spelled once. A matplotlib point
#: and a Vega-Lite pixel both land at ``size * ppi / 72`` pixels, so equal font
#: numbers across the panels are equal type on the composed page -- and a panel
#: written at the wrong resolution cannot be fixed by composing it.
VEGA_WRITE_PPI = 300

#: The three roles every scatter in this figure draws. Separating the diagonal
#: from the regression by hue *and* by dash means neither reading depends on
#: the other being noticed first: a reader looking for agreement finds the
#: diagonal, one looking for trend finds the fit.
POINT_BLUE = "#1f77b4"
REFERENCE_GREY = "#555555"
FIT_FIREBRICK = "#b22222"

#: Panel A's two series are the two *approaches* rather than data and a fit, so
#: neither reuses a scatter colour -- borrowing those hues would imply one of
#: them was a reference line.
COUNTS_GREY = "#4d4d4d"
DASM_PURPLE = "#8172b3"

#: The name on a mutation `fig:within_clade_shifts`' panel A calls out.
#: Near-black rather than a fourth hue: the mark it names is already
#: `POINT_BLUE`, and a coloured label would read as a second measurement of the
#: same mutation instead of an annotation on the one beneath it.
LABEL_INK = "#111111"

#: ...except for the mutations that define the background panels B and D split
#: on, which *are* a second thing being said about the clade and are named in
#: the text as such. They are the clade's own sweep: their counts-based
#: estimates are understated because most of the clade already carries them, so
#: they sit in the same corner of the panel as the shifted mutations for an
#: unrelated reason. Orange separates the two readings; near-black would merge
#: them.
BACKGROUND_INK = "#e07b00"

#: The ink each labelled group is drawn in, keyed by the candidate status
#: `data/shift_candidates.csv` records. A Vega colour *scale* rather than a
#: mark constant, because one frame carries both groups: the label column is
#: stacked over all of them at once, and splitting it into two frames would
#: stack each group blind to the other's rows.
CANDIDATE_INK = {"figure": LABEL_INK, "background": BACKGROUND_INK}

#: Roughly how tall one line of label text is, as a multiple of its font size.
#: It sets the minimum separation between two labels and nothing else, so an
#: approximation is enough: generous costs empty space, short costs a
#: collision.
LABEL_LINE_HEIGHT = 1.3

#: Gap in pixels between a mark and the start of its leader, and between the
#: panel edge and the label column.
LABEL_LEAD_PX = 10

#: One type size for every panel. A figure that mixes two text scales reads as
#: two figures.
PANEL_FONT_SIZE = 18

#: What the two axes of every scatter are. What they measure -- an estimated
#: fitness effect, or its shift between clades -- is in each panel's title, so
#: the axis only has to name the approach that produced the number.
X_AXIS_TITLE = "DASM approach"
Y_AXIS_TITLE = "counts approach"


def save_panel(
    chart: alt.TopLevelMixin, stem: str, *, selection: str
) -> alt.TopLevelMixin:
    """Write one chart to this selection's figure directory as PNG and SVG.

    Returns the chart, so a call can wrap an expression without restructuring
    it.

    **Failures raise.** The upstream dashboard swallowed them, because it was a
    reading tool first and a figure source second; here the call sites are
    pipeline rules, and a swallowed failure leaves a rule "complete" with no
    panel written.

    PNG *and* SVG. The composition is raster, but a vector copy costs nothing
    here and is what a journal asks for if the figure is ever rebuilt.
    """
    directory = figure_dir(selection)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        chart.save(str(directory / f"{stem}.{extension}"), ppi=VEGA_WRITE_PPI)
    return chart


def stat_label(row: pd.Series | None, n_plotted: int) -> str:
    """``r`` and ``n`` for a panel's corner, read from the metrics frame.

    Read rather than recomputed, so the picture and the CSV cannot drift. NaN
    prints as ``n/a``: r is genuinely undefined on fewer than three mutations or
    a constant axis, and ``nan`` on a chart reads like a bug.

    Two significant figures. Three decimals would imply a precision the
    statistic does not have -- its third decimal moves with the clade floor and
    the clip, neither of which is reported beside it.
    """
    if row is None:
        return f"n={n_plotted:,}"
    value = row["pearson_r"]
    text = "n/a" if pd.isna(value) else f"{value:.2g}"
    return f"r={text}\nn={int(row['n_mutations']):,}"


def _corner_label(stat: str, font_size: int = PANEL_FONT_SIZE) -> alt.Chart:
    """The statistic, pixel-positioned in the panel's top-left corner.

    Positioned with ``alt.value`` rather than at a data coordinate, so it lands
    in the corner whatever the domain turns out to be. ``lineBreak`` is what
    makes the newline in ``stat`` a line break: Vega-Lite only breaks a text
    mark where it says to.
    """
    return (
        alt.Chart(pd.DataFrame({"stat": [stat]}))
        .mark_text(
            align="left",
            baseline="top",
            dx=4,
            dy=3,
            fontSize=font_size,
            color="#333",
            lineBreak="\n",
        )
        .encode(x=alt.value(0), y=alt.value(0), text="stat:N")
    )


def stacked_label_y(
    values: Sequence[float], *, min_gap: float, ceiling: float
) -> list[float]:
    """``values`` pushed apart downward to ``min_gap``, starting at the top.

    Each label keeps its own height where it can and is pushed down only where
    it cannot, which preserves the marks' vertical order and so keeps every
    label beside the mark it names. A fixed ladder does neither: it sends the
    whole column to the bottom of the panel however close the points were.

    Args:
        values: The marks' y values, highest first.
        min_gap: Minimum separation between two labels, in data units.
        ceiling: y the topmost label may not exceed.

    Returns:
        The label y values, in the order given.
    """
    placed: list[float] = []
    for value in values:
        limit = ceiling if not placed else placed[-1] - min_gap
        placed.append(min(value, limit))
    return placed


def _candidate_labels(
    labelled: pd.DataFrame,
    *,
    metric: str,
    x_enc: alt.X,
    y_enc: alt.Y,
    scale: alt.Scale,
    axis: alt.Axis,
    side: str,
    size: int,
) -> alt.LayerChart:
    """A leader and a name on each labelled mutation, in a column of its own.

    **The column sits outside the data, not in an empty corner of it.** On the
    right it is drawn past the panel's edge, which is what ``clamp=False`` on
    the scale is for; on the left it stops just short of the leftmost labelled
    mark. Widening the domain to park text inside it would invite a reader to
    look for data in the empty half.

    **Which side is a property of where a clade's candidates sit.** They are
    outliers below the diagonal, so they lie to the lower right of the cloud:
    in 22D that is the cloud's right-hand end, where the right gutter is the
    nearest empty space, but in 20B they are high above the middle of the
    diagonal and reaching the right edge would drag a leader across the whole
    cloud, while directly to their left there is nothing at all.

    **No ring on the marks.** One drawn large enough to read at this panel's
    size on the page covers its own neighbours, and the leader already says
    which mark a label belongs to -- a ring marks points that are not in doubt
    at the cost of hiding points that are. The marks are instead redrawn
    opaque, in their group's own ink: that reads at panel size without taking
    up more room than the mark already does.

    **Two groups, one ink scale.** ``candidate_group`` names which, and
    :data:`CANDIDATE_INK` maps it to a colour. The scale's domain is the set
    the frame carries rather than the constant's keys, so a panel with one
    group does not reserve a slot for the other.
    """
    groups = [
        status for status in CANDIDATE_INK if status in set(labelled["candidate_group"])
    ]
    ink = alt.Color(
        "candidate_group:N",
        scale=alt.Scale(domain=groups, range=[CANDIDATE_INK[g] for g in groups]),
        legend=None,
    )
    low, high = float(scale.domain[0]), float(scale.domain[1])
    units_per_px = (high - low) / size
    ordered = labelled.sort_values(metric, ascending=False).copy()
    if side == "right":
        align = "left"
        ordered["_label_x"] = high + units_per_px * LABEL_LEAD_PX
    else:
        align = "right"
        ordered["_label_x"] = (
            float(ordered["log_selection_factor"].min()) - units_per_px * LABEL_LEAD_PX
        )
    ordered["_label_y"] = stacked_label_y(
        ordered[metric].astype(float).tolist(),
        min_gap=units_per_px * PANEL_FONT_SIZE * LABEL_LINE_HEIGHT,
        ceiling=high,
    )

    marks = (
        alt.Chart(ordered)
        .mark_point(size=64, opacity=1.0, filled=True)
        .encode(x=x_enc, y=y_enc, color=ink)
    )
    leaders = (
        alt.Chart(ordered)
        .mark_rule(strokeWidth=0.8, opacity=0.55)
        .encode(
            x=x_enc,
            y=y_enc,
            x2=alt.X2("_label_x:Q"),
            y2=alt.Y2("_label_y:Q"),
            color=ink,
        )
    )
    # Spelled from `scale` and `axis` rather than read off `x_enc`: an
    # encoding's `.scale` is a property setter, not the value it was built
    # with, and reaching for it raises at render time rather than here.
    text = (
        alt.Chart(ordered)
        .mark_text(
            align=align,
            baseline="middle",
            fontSize=PANEL_FONT_SIZE,
            fontWeight="bold",
        )
        .encode(
            x=alt.X("_label_x:Q", title=X_AXIS_TITLE, scale=scale, axis=axis),
            y=alt.Y("_label_y:Q", title=Y_AXIS_TITLE, scale=scale, axis=axis),
            text="aa_mutation:N",
            color=ink,
        )
    )
    return marks + leaders + text


def scatter_panel(
    frame: pd.DataFrame,
    *,
    metric: str,
    clade: str,
    clip_lower: float,
    stat: str | None,
    size: int = 225,
    labelled: pd.DataFrame | None = None,
    label_side: str = "right",
    title: Sequence[str] | None = None,
) -> alt.LayerChart:
    """Panel B: one clade's mutations, log SF against the counts-based effect.

    **One domain for both axes**, so the diagonal is a true 45 degrees and a
    point's distance from it is the size of the disagreement. The clip is where
    the values pile up and the data's own maximum is whichever approach reaches
    further, so both sit *inside* the domain: a mark is a disc of some radius,
    and a point exactly on a bound is drawn half outside the plot.

    Every layer states the same x and y definition. Vega-Lite merges a layer's
    axes, and a single ``axis=None`` among them disables the merged axis
    outright, ticks and labels included.

    ``labelled`` and ``title`` are what ``fig:within_clade_shifts``' panel A
    adds, and both default to the agreement figure's behaviour: that panel is
    this one over a different clade with a few mutations named, which is why it
    is a parameter here rather than a second builder drifting from this one.
    The labelled rows are drawn from ``frame``'s own values, so a label cannot
    sit anywhere but on its mark.

    **Labelling changes the scale and the ticks, and only then.** The label
    column is drawn past the panel's edge, which needs ``clamp=False`` or Vega
    pulls it back onto the boundary, and the ticks are stated rather than left
    to Vega, which would otherwise tick out across the gutter -- an axis that
    keeps going past the last point invites a reader to look for one. Neither
    applies without ``labelled``, so the agreement figure's panel is unchanged.

    ``stat=None`` draws no corner statistic. `fig:agreement`'s panel reports
    an ``r`` there, which is what that figure is about; these panels would
    report only an ``n``, and the count of mutations passing a cutoff is not
    a result -- it is a property of the cutoff, which the caption already
    states.
    """
    high = float(max(frame["log_selection_factor"].max(), frame[metric].max()))
    padding = 0.04 * ((high - clip_lower) or 1)
    domain = [clip_lower - padding, high + padding]
    axis_options = {
        "labelFontSize": PANEL_FONT_SIZE,
        "titleFontSize": PANEL_FONT_SIZE,
    }
    if labelled is not None:
        axis_options["values"] = [
            value for value in range(-20, 21, 2) if domain[0] <= value <= domain[1]
        ]
    axis = alt.Axis(**axis_options)
    scale_options: dict[str, object] = {"domain": domain, "nice": False}
    if labelled is not None:
        scale_options["clamp"] = False
    scale = alt.Scale(**scale_options)
    x_enc = alt.X("log_selection_factor:Q", title=X_AXIS_TITLE, scale=scale, axis=axis)
    y_enc = alt.Y(f"{metric}:Q", title=Y_AXIS_TITLE, scale=scale, axis=axis)
    base = (
        alt.Chart(frame)
        .mark_point(size=64, opacity=0.35, filled=True, color=POINT_BLUE)
        .encode(
            x=x_enc,
            y=y_enc,
            tooltip=[
                alt.Tooltip("aa_mutation:N", title="mutation"),
                alt.Tooltip("aa_site:Q", title="site"),
                alt.Tooltip("mutant_aa:N", title="mutant aa"),
                alt.Tooltip(
                    "log_selection_factor:Q", title="log SF (clipped)", format=".3f"
                ),
                alt.Tooltip(f"{metric}:Q", title="fitness (clipped)", format=".3f"),
                # Pre-clip values, so a point sitting on the bound still reveals
                # what it actually was.
                alt.Tooltip(
                    "log_selection_factor_raw:Q", title="log SF (raw)", format=".3f"
                ),
                alt.Tooltip(f"{metric}_raw:Q", title="fitness (raw)", format=".3f"),
                alt.Tooltip("actual_count:Q", title="actual count", format=".0f"),
                alt.Tooltip("predicted_count:Q", title="predicted count", format=".1f"),
            ],
        )
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"diagonal": [scale.domain[0], scale.domain[1]]}))
        .mark_line(color=REFERENCE_GREY, strokeDash=[4, 3], strokeWidth=1.5)
        .encode(
            x=alt.X("diagonal:Q", title=X_AXIS_TITLE, scale=scale, axis=axis),
            y=alt.Y("diagonal:Q", title=Y_AXIS_TITLE, scale=scale, axis=axis),
        )
    )
    fit = base.transform_regression("log_selection_factor", metric).mark_line(
        strokeWidth=2, color=FIT_FIREBRICK
    )
    # The cloud first, then the lines over it: a reference line under a dense
    # scatter is buried wherever the data is thickest, which is exactly where a
    # reader looks for it. The labels last of all, over everything.
    # `None` draws no corner at all, which is not the same as an empty
    # string: an empty text mark still reserves its line, and the two panels
    # that pass `None` are read as a column with the background rows beside
    # them, where a blank top-left corner is visible as a ragged edge.
    layered = base + diagonal + fit
    if stat is not None:
        layered = layered + _corner_label(stat)
    if labelled is not None and len(labelled):
        layered = layered + _candidate_labels(
            labelled,
            metric=metric,
            x_enc=x_enc,
            y_enc=y_enc,
            scale=scale,
            axis=axis,
            side=label_side,
            size=size,
        )
    return layered.properties(
        width=size,
        height=size,
        title=alt.TitleParams(
            # A list is two title lines; a "\n" in a title string is inert in
            # Vega-Lite, which governs text marks rather than titles. Broken
            # before the noun phrase rather than inside it.
            list(title)
            if title is not None
            else [
                "estimated fitness effects to the",
                f"{clade} clade-founder sequence",
            ],
            fontSize=PANEL_FONT_SIZE,
            offset=3,
            anchor="middle",
        ),
    )


# ---------------------------------------------------------------------------
# `fig:dms`
# ---------------------------------------------------------------------------
#
# Ported from `dms_comparison.py` in `dasm2-experiments`'
# `experiments/2026-07-20-spike-dms-comparison`, which is the notebook that
# drew the figure `main.tex` includes today -- its `interactive_scatter`, its
# section 3 scatter grid and its bar figure, in that order. The geometry, the
# type sizes, the two row colours and the titling are that notebook's, kept
# rather than re-derived: this is the published figure, and a panel that
# merely resembled it would be a different figure with the same caption.
#
# Three things are not carried over. The marimo wrapper, because a rule has no
# session to put a chart in; the dropdowns and sliders, because a rule's
# population is `config.yaml`'s rather than a reader's; and the tooltip-only
# defaults of the interactive sections, since section 3 overrides every one of
# them and section 3 is what the paper prints.

#: Panel geometry, from the paper block of `dms_comparison.py`'s section 3:
#: "these six panels are a figure, so the type is at reading size and the
#: panels are sized to carry it".
DMS_PANEL_WIDTH = 200
DMS_PANEL_HEIGHT = 188

#: `PAPER_FONT_SIZE`, upstream. Every axis, title and label in both panels is
#: set to it -- the bar figure's `BAR_FONT_SIZE` is this same number -- which
#: is what lets the composition solve one dpi for the whole page.
DMS_FONT_SIZE = 15

#: Width of a y axis -- rotated title, tick labels and ticks -- at
#: :data:`DMS_FONT_SIZE`, measured upstream by rendering one panel with and
#: one without: 359 px against 304 px. Only the leftmost panel of a row
#: carries an axis, so the row is this much wider on its left than on its
#: right, and a centred title sits half of it to the left of the middle
#: panel's midpoint. `dx` moves it back.
DMS_AXIS_BAND = 55

#: One hue per row, carried by that row's dots and by its heading so the two
#: read as one unit. Okabe-Ito blue and vermillion: far apart in lightness and
#: in hue, and distinguishable under every common form of colour blindness.
#: The pair of hues the DMS figures are drawn in, named once. Okabe-Ito blue
#: and vermillion: far apart in lightness and in hue, and distinguishable under
#: every common form of colour blindness. Two dicts below key them to different
#: meanings, which is why they are a constant rather than a literal in each.
OKABE_ITO_BLUE = "#0072B2"
OKABE_ITO_VERMILLION = "#D55E00"

DMS_ROW_COLORS: dict[int, str] = {0: OKABE_ITO_BLUE, 10: OKABE_ITO_VERMILLION}

#: The bars' own geometry. Half the scatter panels' width, because three bars
#: need far less room than a point cloud and the vertical tick labels mean the
#: width does not have to carry the method names; and sized in height so that
#: the whole bar figure is as tall as the scatter grid beside it.
DMS_BAR_WIDTH = 100
DMS_BAR_HEIGHT = 85

#: Spacing between panels within a row, upstream's `spacing=8`.
DMS_ROW_SPACING = 8

#: What a point is. `point_size` and the opacity are section 3's, not the
#: interactive sections' smaller default.
DMS_POINT_SIZE = 36
DMS_POINT_OPACITY = 0.3

#: Headroom added at the top of the measured axis only, so the in-plot `r`
#: sits over empty canvas instead of over the dense band of near-neutral
#: mutations at DMS ~ 0.
DMS_Y_HEADROOM = 0.12

#: The measurement's axis title, shared by all three columns.
DMS_Y_TITLE = "DMS effect (cell entry)"

#: The same string, named for the axis it labels on `fig:dms_strata`, which
#: puts the measurement on x. Borrowing :data:`DMS_Y_TITLE` there would leave a
#: constant named for one axis labelling the other -- and `28-dms-predictors`
#: gives `DMS_Y_TITLE` a different string, so the borrow would silently change
#: this figure's x title when that branch merges.
DMS_STRATA_X_TITLE = DMS_Y_TITLE

#: Each predictor's axis title, keyed by the name the figure gives it. Two
#: lines even where one would fit, so all three titles sit on the same
#: baseline and the panel bottoms line up.
DMS_AXIS_TITLES: dict[str, list[str]] = {
    "DASM": ["DASM", "fitness effect"],
    "per-clade counts": ["per-clade counts", "fitness effect"],
    "full-tree counts": ["full-tree counts", "fitness effect"],
}

#: How `r` is printed in a scatter's corner.
DMS_R_FORMAT = ".2f"

#: The bar axis's headroom above the tallest bar, so a value label or a tall
#: bar does not touch the frame. Shared across both rows, so a bar in one is
#: directly comparable to the bar above it in the other.
DMS_BAR_HEADROOM = 1.12


def dms_count_label(min_count: int) -> str:
    """Short name for a threshold, shared by the scatters and the bars.

    At 0 the threshold is not really a threshold -- it admits every mutation
    carrying a counts-based estimate -- so naming it ">= 0" would invite the
    reader to look for a cut that is not there.
    """
    return "all mutations" if min_count == 0 else f"≥ {min_count} counts"


def dms_homolog_heading(homolog: str, clade: str, n: int) -> str:
    """`fig:dms` panel A's heading: which assay the clouds are drawn against.

    What the figure's row needs said, now that it is the only row. The
    default heading -- :func:`dms_scatter_row`'s -- names the count floor,
    which is there to tell a row from the one under it; with one row it
    answers a question nobody asked and leaves the one a reader does ask,
    "which homolog is this?", to the caption. `n` stays because a panel
    should be readable on its own terms.
    """
    return f"{homolog} (clade {clade}), n = {n:,}"


def dms_padded_domain(values: Sequence[float] | pd.Series) -> list[float]:
    """``[min, max]`` padded by 5%, as upstream's `_padded_domain` pads it."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    low, high = float(array.min()), float(array.max())
    pad = 0.05 * (high - low) if high > low else (abs(high) or 1.0) * 0.05
    return [low - pad, high + pad]


def dms_group_domain(frame: pd.DataFrame, columns: Sequence[str]) -> list[float]:
    """One domain over several columns at once, so a row shares its axes."""
    return dms_padded_domain(frame[list(columns)].to_numpy().ravel())


def dms_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    x_domain: Sequence[float] | None = None,
    y_domain: Sequence[float] | None = None,
    *,
    axis_xlabel: Sequence[str] | str | None = None,
    width: int = DMS_PANEL_WIDTH,
    height: int = DMS_PANEL_HEIGHT,
    font_size: int = DMS_FONT_SIZE,
    r_format: str = DMS_R_FORMAT,
    show_y_title: bool = True,
    show_y_labels: bool = True,
    point_color: str | None = None,
    point_size: int = DMS_POINT_SIZE,
    r: float | None = None,
) -> alt.LayerChart:
    """One predictor against the measurement: upstream's `interactive_scatter`.

    No diagonal and no regression, as there. The axes are a log selection
    factor and a cell-entry measurement -- different units on different scales
    -- so a line at ``y = x`` would invite a reading the data does not
    support, and the number the panel reports is the correlation, printed in
    the corner.

    ``r`` is **read, not recomputed**, when the caller has it: the figure's
    coefficient is `dms.dms_correlations`' and it is what the bars beside this
    panel and the CSV under it report, so computing a second one here is a
    second chance to drift -- the same reason :func:`stat_label` reads its
    number. Recomputing is kept for a standalone call, where there is no
    correlations frame to read from.

    The frame is projected to the four columns the chart and its tooltip use
    before it is handed to Altair, which serializes whatever it is given: the
    joined frame carries about twenty columns, and passing it whole is what
    made the source notebook's first scatter emit 18 MB of Arrow.
    """
    drawn = frame.dropna(subset=[x, y])[[x, y, "mutation", "site"]]
    if r is None:
        r = (
            float(np.corrcoef(drawn[x], drawn[y])[0, 1])
            if len(drawn) > 1
            else float("nan")
        )
    xdom = list(x_domain) if x_domain is not None else dms_padded_domain(drawn[x])
    ydom = list(y_domain) if y_domain is not None else dms_padded_domain(drawn[y])
    axis = alt.Axis(
        titleFontSize=font_size, labelFontSize=font_size, titlePadding=6, tickCount=5
    )
    blind_axis = alt.Axis(
        titleFontSize=font_size,
        labelFontSize=font_size,
        titlePadding=6,
        tickCount=5,
        labels=False,
        ticks=False,
    )
    cloud = (
        alt.Chart(drawn)
        .mark_circle(
            size=point_size,
            opacity=DMS_POINT_OPACITY,
            **({} if point_color is None else {"color": point_color}),
        )
        .encode(
            x=alt.X(
                f"{x}:Q",
                title=xlabel if axis_xlabel is None else axis_xlabel,
                scale=alt.Scale(zero=False, domain=xdom),
                axis=axis,
            ),
            y=alt.Y(
                f"{y}:Q",
                title=ylabel if show_y_title else None,
                scale=alt.Scale(zero=False, domain=ydom),
                axis=axis if show_y_labels else blind_axis,
            ),
            tooltip=[
                alt.Tooltip("mutation:N", title="mutation"),
                alt.Tooltip("site:Q", title="site"),
                alt.Tooltip(f"{x}:Q", title=xlabel, format=".3f"),
                alt.Tooltip(f"{y}:Q", title=ylabel, format=".3f"),
            ],
        )
        .properties(
            width=width,
            height=height,
            # The variable names live on the axes, so nothing is left for the
            # title: `r` goes inside the plot instead, where it cannot collide
            # with the neighbouring panel's.
            title=alt.TitleParams(
                text="",
                fontSize=font_size,
                fontWeight="normal",
                anchor="start",
                offset=3,
            ),
        )
    )
    # Pixel-anchored rather than placed in data coordinates: the row shares one
    # domain across three panels, so a data-space position would drift with the
    # padding and could land on a point.
    label = (
        alt.Chart(pd.DataFrame({"_label": [f"r = {r:{r_format}}"]}))
        .mark_text(align="left", baseline="top", fontSize=font_size, dx=7, dy=6)
        .encode(x=alt.value(0), y=alt.value(0), text="_label:N")
    )
    return alt.layer(cloud, label).properties(width=width, height=height)


def dms_scatter_row(
    frame: pd.DataFrame,
    *,
    min_count: int,
    clade: str,
    methods: Sequence[tuple[str, str]],
    x_domain: Sequence[float],
    y_domain: Sequence[float],
    coefficients: Mapping[str, float] | None = None,
    heading: str | None = None,
    heading_color: str | None = None,
) -> alt.HConcatChart:
    """One row of panel A: three predictors against one homolog's assay.

    ``coefficients`` is ``{column: pearson r}`` for this homolog and rung,
    taken from the frame the bars and the CSVs are drawn from, so that the
    number in a panel's corner is the same object as the bar beside it rather
    than a second computation of it. Omitted, each panel computes its own.

    ``heading`` and ``heading_color`` are what the composed figure needs and
    a standalone panel does not. Left at ``None`` the row heads itself with
    its count floor, in the row's own colour, which is what tells two stacked
    rows apart; `fig:dms` draws one row, so there is nothing to tell apart
    and it passes :func:`dms_homolog_heading` in black instead. ``""``
    suppresses the heading altogether. See :data:`panel_spec.DMS_FIGURE_ROWS`.

    The default heading names the clade the count floor was applied in -- the
    floor is on this homolog's *own* clade -- rather than leaving a reader to
    infer which counts were thresholded.

    Titles live inside the chart rather than in markdown above it, as
    upstream: a markdown caption is positioned by the browser against the
    chart element, while a Vega title is positioned against the panels
    themselves. `anchor` centres it on the whole row, and `dx` then moves it
    the half-axis-width that separates the row's midpoint from the middle
    panel's.
    """
    colour = DMS_ROW_COLORS[min_count]
    panels = [
        dms_scatter(
            frame,
            column,
            "dms_cell_entry",
            label,
            DMS_Y_TITLE,
            x_domain=x_domain,
            y_domain=y_domain,
            axis_xlabel=DMS_AXIS_TITLES[label],
            point_color=colour,
            show_y_title=index == 0,
            show_y_labels=index == 0,
            r=None if coefficients is None else coefficients[column],
        )
        for index, (column, label) in enumerate(methods)
    ]
    if heading is None:
        heading = (
            f"all mutations with estimated fitness effects in clade {clade} "
            f"(n = {len(frame):,})"
            if min_count == 0
            else (
                f"only mutations with ≥ {min_count} actual or expected counts "
                f"in clade {clade} (n = {len(frame):,})"
            )
        )
    row = alt.hconcat(*panels, spacing=DMS_ROW_SPACING)
    if not heading:
        return row
    return row.properties(
        title=alt.TitleParams(
            text=heading,
            color=colour if heading_color is None else heading_color,
            fontSize=DMS_FONT_SIZE,
            fontWeight="bold",
            anchor="middle",
            dx=DMS_AXIS_BAND / 2,
            offset=6,
        )
    )


#: What each stratum is called on a chart. Ported from `28-dms-predictors`,
#: where the reasoning is recorded: spelled here rather than in `scripts/dms.py`
#: because the strings in that module are join keys and CSV values, and changing
#: one to read better on a chart would change what a panel's table says.
#:
#: **Each says "mutations", and says what kind.** The stratum is a property of
#: amino-acid mutations and of the founder codon that reaches them, and a bare
#: "single-nt" leaves a reader to guess whether the nucleotide or the residue
#: is the unit being counted. Two lines because the long forms are wider than
#: a :data:`DMS_PANEL_WIDTH` cell, and Vega-Lite neither wraps nor elides a
#: title: adjacent ones simply overlap.
DMS_STRATUM_LABELS: dict[str, tuple[str, ...]] = {
    "all": ("all amino-acid", "mutations"),
    "single_nt": ("mutations reachable by a", "single-nt codon mutation"),
    "multi_nt": ("mutations requiring a", "multi-nt codon mutation"),
}

#: Hue by stratum, for `fig:dms_strata`. The same Okabe-Ito blue and vermillion
#: as :data:`DMS_ROW_COLORS`, reused deliberately: they are the DMS figures'
#: pair, far apart in lightness and in hue, and distinguishable under every
#: common form of colour blindness.
#:
#: **The two constants mean different things, and the caption has to say which.**
#: On `fig:dms` the hue is a count threshold; here it is codon accessibility. A
#: third blue would be worse -- two near-identical blues on facing pages read as
#: a failed attempt at the same encoding -- but a reader given no caption would
#: carry the first meaning onto the second figure.
DMS_STRATUM_COLORS: dict[str, str] = {
    "single_nt": OKABE_ITO_BLUE,
    "multi_nt": OKABE_ITO_VERMILLION,
}


#: How many characters of y tick label each row of `fig:dms_strata` reserves.
#:
#: **This is what aligns the columns, and it has to happen in the chart.** The
#: rows draw identical cells, so their only width difference is the y axis on
#: the left -- antiGen's labels run to ``-16`` where the DASM's run to ``-5``,
#: and Vega-Lite sizes the axis to the widest label it draws. Left justified,
#: the narrower row's plots then sit a band left of the other's, and a column
#: whose plots do not share an x is not a column.
#:
#: Padding to a common width in the *composition* fixes the composed figure and
#: nothing else: the panel PNGs are still misaligned, and so is the notebook,
#: which draws the rows as two charts. Reserving the width here fixes all three
#: at once, which is why it is done here and not there.
#:
#: Three because that is the widest label either row draws. ``pad`` only ever
#: adds, so a row that one day needs four would simply stop aligning rather
#: than truncate -- `tests/test_dms_strata.py` is what notices.
DMS_Y_LABEL_CHARS = 3

#: The character the padding is made of: U+2007 FIGURE SPACE, which is defined
#: to be the width of a digit. An ordinary space is narrower than a digit in a
#: proportional font, so padding with one would leave the bands unequal by a
#: fraction of a character -- small, and exactly the kind of misalignment that
#: survives review.
DMS_LABEL_PAD = " "


def dms_stratum_label(stratum: str) -> list[str]:
    """One stratum's name, as the figure prints it, a line per element.

    Raises:
        KeyError: On a stratum with no label. Returning the raw key instead
            would put ``single_nt`` on a paper figure, which is the one failure
            here that looks deliberate.
    """
    if stratum not in DMS_STRATUM_LABELS:
        raise KeyError(
            f"{stratum!r} has no label; expected one of {sorted(DMS_STRATUM_LABELS)}"
        )
    return list(DMS_STRATUM_LABELS[stratum])


def dms_stratum_row(
    frame: pd.DataFrame,
    *,
    score_column: str,
    score_title: Sequence[str] | str,
    stratum_order: Sequence[str],
    x_domain: Sequence[float],
    stats: pd.DataFrame | None = None,
    show_column_titles: bool = True,
    show_x_title: bool = True,
    width: int = DMS_PANEL_WIDTH,
    height: int = DMS_PANEL_HEIGHT,
    font_size: int = DMS_FONT_SIZE,
) -> alt.HConcatChart:
    """One model's row of `fig:dms_strata`: the same scores, cut three ways.

    **The measurement is on x and the model on y**, which is the transpose of
    `fig:dms` and of the `28-dms-predictors` panel this is ported from. The
    reason is the grid: the DMS is the one quantity common to both rows, so
    putting it on x makes a column readable down the page -- the same mutations,
    two models. The cost is that `DMS_Y_TITLE` and `DMS_AXIS_TITLES` are written
    for the other orientation and are not reused here; the titles arrive as
    arguments instead.

    **Points and the corner statistic, nothing else.** No diagonal, because the
    axes are a measurement and a model score in different units and a 45-degree
    line would invite a reading the data does not support. No regression and no
    binned median either, although the ported panel drew both: `fig:dms`'s own
    scatters are a cloud and an ``r``, and this figure sits beside them.

    **Each cell is layered by stratum rather than coloured by a column**, which
    is what keeps every point. Altair refuses a *dataset* over 5,000 rows, and
    the refusal is asymmetric in a way that hides it -- ``chart.save()`` writes
    the PNG through ``vl-convert`` regardless while ``chart.to_dict()`` raises,
    so a rule fails after its panel is already on disk. The ``all`` cell is over
    that ceiling at every homolog; split into its two strata it is not, and the
    split is one the figure wanted anyway. Nothing is thinned, which
    `scripts/panel_spec.py` records as this figure's own decision.

    Args:
        frame: One homolog's cells, carrying ``dms_cell_entry``, ``stratum``
            and ``score_column``.
        score_column: The model's column, which becomes y.
        score_title: The y axis title, a string or the lines of one.
        stratum_order: Which cells to draw, left to right. ``"all"`` draws the
            whole frame; any other value draws that stratum alone.
        x_domain: The measurement's domain, shared across every row of the
            figure so a column can be read down the page. **Within one figure
            only.** `fig:dms_strata` is one figure per assay and each computes
            this from its own homolog's cells, so a column may not be read
            across the three.
        stats: ``predictor_correlations.csv`` filtered to this row's homolog and
            predictor. ``r`` and ``n`` are read from it rather than recomputed,
            so the picture and the table cannot drift.
    """
    axis = alt.Axis(labelFontSize=font_size, titleFontSize=font_size, titlePadding=6)
    x_scale = alt.Scale(domain=list(x_domain), nice=False)
    # One y domain across the row, computed from every cell rather than per
    # cell: the strata are subsets of one population and the question the row
    # answers is whether they differ, which two y ranges would hide.
    y_scale = alt.Scale(domain=dms_padded_domain(frame[score_column]), nice=False)

    cells = []
    for stratum in stratum_order:
        cell_frame = frame if stratum == "all" else frame[frame["stratum"] == stratum]
        row = None
        if stats is not None:
            matching = stats[stats["stratum"] == stratum]
            row = None if matching.empty else matching.iloc[0]

        # The column heading and the x title belong to the *column*, not to
        # each cell in it: every row is the same measurement over the same
        # domain, so printing either once per row prints this figure's two
        # longest strings three times each. The heading goes on the top row and
        # the title on the bottom, where a reader looks for them. Tick labels
        # stay on both: the rows are separated by a panel break, so a cell has
        # to be readable without tracing down the page.
        x_enc = alt.X(
            "dms_cell_entry:Q",
            title=DMS_STRATA_X_TITLE if show_x_title else None,
            axis=axis,
            scale=x_scale,
        )
        # The y scale is shared across the row, so only the leftmost cell
        # carries the axis: repeating it prints this figure's longest string
        # once per stratum.
        y_axis = (
            alt.Axis(
                labelFontSize=font_size,
                titleFontSize=font_size,
                titlePadding=6,
                # Reserve a fixed label width, so both rows' axes are the same
                # band and their plots start at the same x. See
                # `DMS_Y_LABEL_CHARS`.
                labelExpr=(
                    f"pad(datum.label, {DMS_Y_LABEL_CHARS}, '{DMS_LABEL_PAD}', 'left')"
                ),
            )
            if not cells
            else alt.Axis(labels=False, ticks=False, domain=True, title=None)
        )
        y_enc = alt.Y(
            f"{score_column}:Q",
            title=score_title if not cells else None,
            axis=y_axis,
            scale=y_scale,
        )

        # One layer per stratum present, each its own dataset at its own fixed
        # colour. This is the row-limit escape described above, and it is also
        # why no `alt.Color` scale and no `legend=None` appear anywhere here.
        layers = [
            alt.Chart(
                cell_frame[cell_frame["stratum"] == present][
                    ["dms_cell_entry", score_column]
                ]
            )
            .mark_point(
                size=DMS_POINT_SIZE,
                opacity=DMS_POINT_OPACITY,
                filled=True,
                color=DMS_STRATUM_COLORS[present],
            )
            .encode(x=x_enc, y=y_enc)
            for present in DMS_STRATUM_COLORS
            if not cell_frame[cell_frame["stratum"] == present].empty
        ]
        # `n` is the cell's own count, not the layer's.
        labelled = alt.layer(
            *layers, _corner_label(stat_label(row, len(cell_frame)), font_size)
        )
        properties = {"width": width, "height": height}
        if show_column_titles:
            properties["title"] = alt.TitleParams(
                dms_stratum_label(stratum),
                fontSize=font_size,
                offset=3,
                anchor="middle",
            )
        cells.append(labelled.properties(**properties))
    # Stated rather than changed: every cell was given the same explicit
    # domains above. Without it Vega-Lite resolves concatenated scales
    # independently and the shared domain becomes a coincidence of the data.
    return alt.hconcat(*cells, spacing=DMS_ROW_SPACING).resolve_scale(
        x="shared", y="shared"
    )


def _dms_bar_group_header(title: str, n: int) -> alt.Chart:
    """The ``n =`` strip above one group of bars, with the group's name on it.

    Shared by `fig:dms`'s bar row and `fig:probabilities`' bars: both stack a
    header over a bar chart and the geometry has to match, or the two figures
    set the same furniture at different sizes. Extracted when the second
    caller arrived rather than copied a third time.
    """
    return (
        alt.Chart(pd.DataFrame({"_label": [f"n = {n:,}"]}))
        .mark_text(fontSize=DMS_FONT_SIZE, align="center")
        .encode(x=alt.value(DMS_BAR_WIDTH / 2), y=alt.value(10), text="_label:N")
        .properties(
            width=DMS_BAR_WIDTH,
            height=24,
            view=alt.ViewBackground(stroke=None),
            title=alt.TitleParams(
                text=title,
                fontSize=DMS_FONT_SIZE,
                fontWeight="bold",
                anchor="middle",
            ),
        )
    )


def dms_bar_row(
    correlations: pd.DataFrame,
    *,
    min_count: int,
    homologs: Sequence[str],
    clades: dict[str, str],
    methods: Sequence[tuple[str, str]],
    y_domain: Sequence[float],
    heading: str | None = None,
) -> alt.HConcatChart:
    """One row of panel B: the same coefficient, for every homolog.

    The scatter row reports `r` in each panel for one homolog; this collects
    them all, so the comparison that matters -- how much each method
    loses when the threshold drops -- is one figure rather than three dropdown
    settings held in the reader's head.

    Built one homolog at a time and `hconcat`'d rather than with `.facet()`,
    because a Vega-Lite facet header is a single string: it cannot carry the
    homolog's name and that panel's `n` on separate lines.

    **Every bar carries its value**, as `fig:probabilities`' bars do and for
    the same reason: a reader comparing three methods across four datasets
    should not have to measure against an axis, and the differences that
    carry the argument here are small -- Delta's DASM and per-clade bars
    differ by 0.03, which is where eye-matching fails. The axis stays, since
    it is what makes a bar in one group comparable to the bar beside it in
    another; the numbers are what make a pair readable without it.
    """
    colour = DMS_ROW_COLORS[min_count]
    order = [label for _column, label in methods]
    rows = correlations[correlations["min_count"] == min_count]

    def panel(homolog: str, first: bool) -> alt.VConcatChart:
        one = rows[rows["homolog"] == homolog].copy()
        # Formatted here rather than by a Vega `format`, so the number drawn
        # over a bar is produced by the same expression as the one in the
        # scatter's corner and the two cannot drift apart.
        one["_value"] = [f"{r:{DMS_R_FORMAT}}" for r in one["pearson"]]
        header = _dms_bar_group_header(
            f"{homolog} ({clades[homolog]})", int(one["n"].iloc[0])
        )
        base = alt.Chart(one).encode(
            x=alt.X(
                "method:N",
                title=None,
                sort=order,
                axis=alt.Axis(labelAngle=-90, labelFontSize=DMS_FONT_SIZE),
            ),
            y=alt.Y(
                "pearson:Q",
                # Axis on the leftmost panel only, as in the scatter row,
                # so the columns sit a character apart instead of a whole
                # axis apart. The heading's `dx` compensates.
                title="Pearson r" if first else None,
                scale=alt.Scale(domain=list(y_domain)),
                axis=alt.Axis(
                    titleFontSize=DMS_FONT_SIZE,
                    labelFontSize=DMS_FONT_SIZE,
                    labels=first,
                    ticks=first,
                ),
            ),
        )
        bars = base.mark_bar(color=colour).encode(
            tooltip=[
                alt.Tooltip("homolog:N", title="homolog"),
                alt.Tooltip("method:N", title="method"),
                alt.Tooltip("pearson:Q", title="Pearson r", format=".2f"),
                alt.Tooltip("n:Q", title="n", format=","),
            ],
        )
        # `dy` sits the value just clear of the bar it belongs to, at
        # `fig:probabilities`' offset and for its reason: these panels render
        # at `VEGA_WRITE_PPI`, so a chart-unit offset is multiplied by about
        # four on the written PNG.
        values = base.mark_text(
            dy=-2, fontSize=DMS_FONT_SIZE, baseline="bottom"
        ).encode(text="_value:N")
        bar = alt.layer(bars, values).properties(
            width=DMS_BAR_WIDTH, height=DMS_BAR_HEIGHT
        )
        return alt.vconcat(header, bar, spacing=2)

    if heading is None:
        heading = (
            "all mutations with estimated fitness effects"
            if min_count == 0
            else f"only mutations with ≥ {min_count} actual or expected counts"
        )
    row = alt.hconcat(
        *[panel(homolog, homolog == homologs[0]) for homolog in homologs],
        spacing=DMS_ROW_SPACING,
    )
    # Each group already carries its homolog and its `n`, so on the composed
    # figure -- one row, nothing to tell it apart from -- the row heading
    # restates the population the caption states. `""` drops it; the
    # thresholded rung, which is written but not laid out, keeps its own.
    if not heading:
        return row
    return row.properties(
        title=alt.TitleParams(
            text=heading,
            color=colour,
            fontSize=DMS_FONT_SIZE,
            fontWeight="bold",
            anchor="middle",
            dx=DMS_AXIS_BAND / 2,
            offset=6,
        )
    )


#: Panel D's per-cell columns. The statistics are computed upstream on the full
#: data, so thinning a dense cell changes what is drawn and never what is
#: reported.
SHIFT_PLOT_COLUMNS = ["mutation_label", "sf_shift", "fitness_shift"]

#: A ceiling on the **whole grid**, not a per-panel quota, so a grid that fits
#: under it is drawn in full. It exists to keep a notebook's output under
#: marimo's 5 MB per-output limit; at three columns a point costs on the order
#: of 60 bytes of Vega-Lite JSON.
SHIFT_POINT_BUDGET = 40_000

#: Round tick values rather than a tick count: shifts are read in units of log
#: fitness, so labelling the same values in every cell lets a cloud's width be
#: compared across cells without reading each axis. Values outside the shared
#: domain do not render, so the set is safe whatever the selection spans.
SHIFT_TICKS = [-6, -3, 0, 3, 6]


def shift_panel(
    frame: pd.DataFrame,
    *,
    title: str,
    stat: str,
    domain: list[float],
    size: int,
    x_title: str | None,
    y_title: str | None,
) -> alt.LayerChart:
    """One clade pair's shift scatter, on axes shared with every other cell.

    Every point is drawn the same. Where a point sits relative to the dashed
    diagonal says more than a hue could -- an encoding would spend the colour
    channel on what position already shows, and force a legend into every cell
    of an already dense grid.

    **The clip bound does not bound these axes.** A shift is a difference of two
    clipped effects, so it legitimately reaches twice the bound, and bounding at
    the clip would pile the largest shifts -- the ones the panel exists to show
    -- onto the edge. The domain comes from the pooled shift data instead.
    """
    axis = alt.Axis(
        labelFontSize=PANEL_FONT_SIZE,
        titleFontSize=PANEL_FONT_SIZE,
        values=SHIFT_TICKS,
        labelOverlap="greedy",
    )
    scale = alt.Scale(domain=domain, nice=False, clamp=True)
    points = (
        alt.Chart(frame[SHIFT_PLOT_COLUMNS])
        # Fainter than a standalone scatter: the cells are compact, so a dense
        # pair goes solid at full opacity. Opacity is what carries density here.
        .mark_point(size=64, opacity=0.35, filled=True, color=POINT_BLUE)
        .encode(
            x=alt.X("sf_shift:Q", title=x_title, scale=scale, axis=axis),
            y=alt.Y("fitness_shift:Q", title=y_title, scale=scale, axis=axis),
            tooltip=[
                alt.Tooltip("mutation_label:N", title="mutation"),
                alt.Tooltip("sf_shift:Q", title="Δ log SF", format=".3f"),
                alt.Tooltip("fitness_shift:Q", title="Δ fitness", format=".3f"),
            ],
        )
    )
    # y = x, not the two zero rules: agreement between the two approaches *is*
    # this diagonal, so a point's distance from it is the size of the
    # disagreement, magnitude included, where the zero rules showed only sign.
    diagonal = (
        alt.Chart(pd.DataFrame({"diagonal": [domain[0], domain[1]]}))
        .mark_line(color=REFERENCE_GREY, strokeDash=[4, 3], strokeWidth=1.5)
        .encode(
            x=alt.X("diagonal:Q", title=x_title, scale=scale, axis=axis),
            y=alt.Y("diagonal:Q", title=y_title, scale=scale, axis=axis),
        )
    )
    # Fitted on the points the panel draws, which on a thinned pair is the
    # sample; the r beside it is computed on the full set either way, so a
    # visible gap between the two is the thinning, not a disagreement.
    fit = points.transform_regression("sf_shift", "fitness_shift").mark_line(
        strokeWidth=2, color=FIT_FIREBRICK
    )
    # The pair, over its own cell. Only the bottom row carries an x title and
    # only the first column a y title, so without this a reader has to count
    # rows and columns against the grid's axes to learn which pair a cell is
    # -- and the grid is lower-triangular, so the counting is not the obvious
    # kind. `offset=3` keeps the label tight to the cell it names rather than
    # drifting toward the one above.
    return (points + diagonal + fit + _corner_label(stat)).properties(
        width=size,
        height=size,
        title=alt.TitleParams(title, fontSize=PANEL_FONT_SIZE, offset=3),
    )


def blank_cell(size: int) -> alt.Chart:
    """A same-sized hole in the grid.

    A pair can be missing -- identical founders, or fewer shared mutations than
    a correlation needs -- and dropping its cell instead of blanking it would
    slide every later column in that row one place left, putting a panel under
    the wrong clade.
    """
    return (
        alt.Chart(pd.DataFrame({"zero": [0.0]}))
        .mark_point(opacity=0)
        .properties(width=size, height=size)
    )


def shift_grid(
    shift_df: pd.DataFrame,
    metrics: pd.DataFrame,
    clades: list[str],
) -> alt.VConcatChart:
    """Panel D: a lower-triangular grid of clade pairs, one cell per pair.

    Laid out like a pairplot -- the **column is the earlier clade**, the **row
    is the later one** -- so the cell at row 23A, column 21J holds the 21J→23A
    shift. Only the lower triangle: the upper half is the same pair with both
    axes negated, and the diagonal is a clade against itself.

    Built by hand rather than with ``.facet()`` so each cell can carry that
    pair's own r and n. A facet header gives the pair label only, and the
    per-pair numbers are the reason to separate the pairs rather than pool them.

    **Shared, explicit domains over the whole selection**, one for both axes.
    The cells are square, so equal ranges make a degree of slope on the screen a
    degree of slope in the data: with separate domains a pair whose
    counts-based shifts are twice the model's is stretched back to a
    45-degree cloud and reads as agreement.
    """
    low = float(min(shift_df["sf_shift"].min(), shift_df["fitness_shift"].min()))
    high = float(max(shift_df["sf_shift"].max(), shift_df["fitness_shift"].max()))
    padding = 0.02 * ((high - low) or 1)
    domain = [low - padding, high + padding]

    pairs = sorted(shift_df["pair"].unique())
    # Thin only when the grid as a whole is over budget. Dividing the budget
    # evenly regardless caps a dense cell while the whole grid is well under
    # the ceiling -- points dropped for no reason.
    per_panel_cap = (
        len(shift_df)
        if len(shift_df) <= SHIFT_POINT_BUDGET
        else max(600, SHIFT_POINT_BUDGET // max(len(pairs), 1))
    )
    order = sorted(clades)
    # Cells shrink as the triangle grows: five clades is ten cells, six is
    # fifteen, and a fixed large cell turns those into a page of scrolling.
    cell = 165 if len(order) <= 4 else 120
    by_pair = {pair: shift_df[shift_df["pair"] == pair] for pair in pairs}

    rows = []
    for row_index, later in enumerate(order[1:], start=1):
        is_bottom_row = row_index == len(order) - 1
        cells = []
        for column_index, earlier in enumerate(order[:row_index]):
            pair = f"{earlier}→{later}"
            frame = by_pair.get(pair)
            if frame is None or frame.empty:
                cells.append(blank_cell(cell))
                continue
            # Fixed seed: a rerun should not reshuffle the picture.
            drawn = (
                frame
                if len(frame) <= per_panel_cap
                else frame.sample(n=per_panel_cap, random_state=0)
            )
            matched = (
                metrics[metrics["pair_label"] == pair] if len(metrics) else metrics
            )
            row = None if matched.empty else matched.iloc[0]
            stat = stat_label(row, len(drawn))
            if row is not None and bool(row["sf_shift_is_constant"]):
                stat += "\nΔSF constant"
            if len(drawn) < len(frame):
                stat += f"\nplotting {len(drawn):,}"
            cells.append(
                shift_panel(
                    drawn,
                    title=pair,
                    stat=stat,
                    domain=domain,
                    size=cell,
                    x_title=X_AXIS_TITLE if is_bottom_row else None,
                    y_title=Y_AXIS_TITLE if column_index == 0 else None,
                )
            )
        rows.append(alt.hconcat(*cells, spacing=4))
    return alt.vconcat(*rows, spacing=4).properties(
        title=alt.TitleParams(
            "estimated shifts in fitness effects between clade-founder sequences",
            fontSize=PANEL_FONT_SIZE,
            # A concat spec's title anchors to "start" by default, so it would
            # sit over the leftmost column rather than over the grid.
            anchor="middle",
        )
    )


#: Negative r is clipped to zero **in the plot**, dots and medians alike. An
#: r below zero here is a unit whose shifts were noise in both columns -- the
#: sign of a correlation between two noise processes carries no information --
#: and left unclipped those dots stretch the y axis over a region nothing is
#: being claimed about. The CSVs keep the raw value.
LADDER_R_FLOOR = 0.0
LADDER_MEDIAN_COLOR = "#8c8c8c"

#: Both axis titles are wrapped by hand: Vega-Lite will not wrap an axis title
#: on its own, and at these panel widths a single line of either runs well past
#: the panel.
LADDER_X_TITLE = ["minimum actual or", "expected count"]
LADDER_Y_TITLE = ["Pearson correlation", "coefficient (r)"]

#: Narrow and tall on purpose: the x axis has three rungs, so width buys
#: nothing but space between them, while height is what resolves dots that
#: differ in the second decimal of r.
LADDER_PANEL_WIDTH = 150
LADDER_PANEL_HEIGHT = 187


def ladder_row(
    long_df: pd.DataFrame,
    *,
    unit_field: str,
    unit_title: str,
    title: list[str],
    width: int = LADDER_PANEL_WIDTH,
    height: int = LADDER_PANEL_HEIGHT,
    point_size: int = 80,
) -> alt.LayerChart:
    """Panels C and E: every unit as a blue dot, the medians as grey dashes.

    **No jitter.** The x axis is ordinal over the rungs, so coincident dots are
    a real statement -- two units with the same r at the same cutoff -- and
    displacing them sideways to reveal both implies a precision in x the rungs
    do not have.

    The median is a **tick**: a short thick horizontal dash spanning part of the
    rung's band, drawn behind the dots and joined rung to rung by a grey line. A
    dash reads as an aggregate of the column it sits in rather than as one more
    observation, and unlike a point it never hides a dot underneath it.

    The dots matter as much as the dashes: a median that rises because every
    unit rose is a different result from one that rises because two units moved
    a long way, and only the spread distinguishes them.
    """
    # Clipped before the median, not after: the median of the drawn dots is the
    # line the panel should show, and a median taken on the raw column could sit
    # below every dot it is supposed to summarise.
    long_df = long_df.assign(pearson_r=long_df["pearson_r"].clip(lower=LADDER_R_FLOOR))
    medians = (
        long_df.groupby("min_count", as_index=False)["pearson_r"]
        .median()
        .rename(columns={"pearson_r": "median_r"})
    )
    x_enc = alt.X(
        "min_count:O",
        title=LADDER_X_TITLE,
        axis=alt.Axis(labelFontSize=PANEL_FONT_SIZE, titleFontSize=PANEL_FONT_SIZE),
    )
    y_axis = alt.Axis(
        labelFontSize=PANEL_FONT_SIZE, titleFontSize=PANEL_FONT_SIZE, tickCount=5
    )
    y_enc = alt.Y("median_r:Q", title=LADDER_Y_TITLE, axis=y_axis)
    median_line = (
        alt.Chart(medians)
        .mark_line(color=LADDER_MEDIAN_COLOR, strokeWidth=1.5)
        .encode(x=x_enc, y=y_enc)
    )
    median_dash = (
        alt.Chart(medians)
        .mark_tick(color=LADDER_MEDIAN_COLOR, thickness=4, size=22, opacity=1.0)
        .encode(
            x=x_enc,
            y=y_enc,
            tooltip=[
                alt.Tooltip("min_count:O", title="Min count"),
                alt.Tooltip("median_r:Q", title="Median r", format=".4f"),
            ],
        )
    )
    units = (
        alt.Chart(long_df)
        .mark_point(size=point_size, filled=True, opacity=0.7, color=POINT_BLUE)
        .encode(
            x=x_enc,
            y=alt.Y("pearson_r:Q", title=LADDER_Y_TITLE, axis=y_axis),
            tooltip=[
                alt.Tooltip(f"{unit_field}:N", title=unit_title),
                alt.Tooltip("min_count:O", title="Min count"),
                alt.Tooltip("pearson_r:Q", title="Pearson r", format=".4f"),
                alt.Tooltip("n_mutations:Q", title="Mutations", format=","),
            ],
        )
    )
    # The title goes on the *panel*, not on a concat around it: a layer spec's
    # "middle" anchor centres over the plot area, while a concat's centres over
    # the whole group including its axis titles and labels, which at these
    # widths reads as noticeably off-centre.
    return (median_line + median_dash + units).properties(
        width=width,
        height=height,
        title=alt.TitleParams(
            title, fontSize=PANEL_FONT_SIZE, offset=4, anchor="middle"
        ),
    )


def ladder_clade_count(long_df: pd.DataFrame, unit_field: str) -> int:
    """How many clades a ladder panel is drawn over.

    Counted from the frame rather than from a constant, so the number in a title
    cannot drift from the dots beneath it. A within-clade panel's unit *is* a
    clade; a shift panel's unit is a pair, so its clades are the two endpoints
    of every pair label.
    """
    if unit_field == "pair_label":
        clades: set[str] = set()
        for label in long_df["pair_label"].unique():
            clades.update(str(label).split("→"))
        return len(clades)
    return int(long_df[unit_field].nunique())


#: Panel A comes down to 80% in both directions. Bigger type in a smaller
#: frame is the trade this panel wants: one mutation across a few dozen clades
#: needs legible axes far more than it needs area, and the two series are far
#: enough apart that the extra pixels only bought whitespace.
LOOKUP_PANEL_SCALE = 0.8

#: Height separately from width, because the two are constrained by different
#: things. The width has to hold a few dozen rotated clade labels; the height
#: only has to resolve two series that are far apart. **In the composed figure
#: panel A shares a row with B and C**, so a taller panel A does not make
#: itself bigger -- it pushes the whole row down, forces `compose` to solve a
#: smaller dpi for the page, and shrinks the type on every other panel while
#: leaving B and C floating in whitespace. This brings it level with them.
LOOKUP_PANEL_HEIGHT = 290


def lookup_panel_sizing(n_clades: int) -> dict[str, int]:
    """Panel A's width and height, from how many clades it has to place."""
    return {
        "width": max(
            round(LOOKUP_PANEL_SCALE * 420),
            round(LOOKUP_PANEL_SCALE * 20) * n_clades,
        ),
        "height": round(LOOKUP_PANEL_SCALE * LOOKUP_PANEL_HEIGHT),
    }


#: The background rows are matplotlib, not Vega-Lite, and that is deliberate.
#: They are the one part of this figure that has to line up *across three
#: separately written images*: B, D and F stack left-justified, and a reader
#: can only compare their slopes if each clade's sub-panels sit at the same x
#: and the same width in every row. A Vega-Lite ``hconcat`` sizes itself to
#: its own content, so 20B's two cells and 22D's five come out at different
#: left edges and different widths; matplotlib can be handed one slot count
#: and one set of margins and asked to honour them. The published panels were
#: drawn this way for the same reason, in
#: ``2026-09-15-within-clade-shift-v12/notebooks/background_effects.py``, and
#: this is a port of that renderer.
#:
#: The mixed-renderer figure costs nothing at composition time:
#: ``figure_layout`` already solves one dpi for the whole page, and both
#: renderers put a type size of *N* at ``N * 300/72`` pixels on disk.

#: A sub-panel's width and a row's height, **in ems of the row's own type**
#: rather than in inches. Narrow and tall on purpose: the panel's subject is
#: the *slope* between two x positions, which a wide short cell flattens --
#: at a square aspect a 2-nat shift and a 5-nat one look alike. The width
#: leaves room for the three-line background labels and the legend slot and
#: no more, because up to five of these plus a scatter share one row of the
#: composed figure.
#:
#: Ems, because the published panels were tuned as a unit -- 1.76 x 4.5
#: inches at 13 pt -- and this figure draws them at 17 pt. In inches the
#: three numbers are independent and the type outgrows the slot: at 1.76 the
#: legend ran off the right edge and the value axis thinned from five ticks
#: to two, because matplotlib fits ticks to the space the labels leave. In
#: ems they move together, so these panels have the same type-to-panel ratio
#: as the published ones whatever :data:`BACKGROUND_FONT_PT` is set to. The
#: absolute size does not matter: ``figure_layout.compose`` solves one dpi
#: from the pixels it is handed.
BACKGROUND_CELL_W_EM = 1.76 / (13 / 72)

#: The published row is 4.5 inches tall at 13 pt. Sixty per cent of that is
#: plot area and the rest is the title, the three-line tick labels and the
#: axis label, which do not scale with the value axis. Withholding the
#: p-value takes the axis from 1.30 to 1.16 times the data span, so the plot
#: area needs 1.16/1.30 of its height to draw the same slopes and the
#: furniture is unchanged: 4.5 - 0.60 x 4.5 x (1 - 1.16/1.30) = 4.2 inches.
#: That is the whole of the reduction; the panels are not otherwise made
#: smaller. Restoring the p-value restores this and
#: :data:`BACKGROUND_HEADROOM` together.
BACKGROUND_ROW_H_EM = 4.2 / (13 / 72)

#: Type size for the background rows, in points. 17 rather than
#: :data:`PANEL_FONT_SIZE` because the two renderers do not put the same ink
#: inside the same em: a matplotlib point and a Vega-Lite pixel both land at
#: ``size * 300/72`` pixels on disk, but matplotlib's glyphs fill more of
#: that box. The upstream notebook measured the ratio on a tick digit -- the
#: one cap-height glyph both renderers draw -- and found 13 pt matching 14 px;
#: 18 px scales to 16.7, and 17 is that to the nearest point.
BACKGROUND_FONT_PT = 17

#: Headroom above and below the drawn range, as fractions of it. Equal:
#: the top used to carry 0.22 because the randomization test's p-value was
#: drawn inside the axes, and that panel is withheld while the multiple-
#: testing question is open -- see `main.tex`, where the sentences reporting
#: it are commented rather than deleted. Restoring the p-value means
#: restoring the headroom with it.
BACKGROUND_HEADROOM = 0.08
BACKGROUND_FOOTROOM = 0.08

#: The two approaches, their colours and their matplotlib markers, in the
#: order they are drawn. One spelling, because both the lines and the legend
#: are built from it: a legend drawn from its own copy of the mapping is a
#: legend that can come to disagree with the marks it names. The colours are
#: the figure's own, so a reader who learned grey-circle/purple-triangle in
#: `fig:agreement`'s lookup panels does not have to re-learn it here.
BACKGROUND_SERIES = (
    ("counts", "counts_effect", COUNTS_GREY, "o"),
    ("DASM", "dasm_effect", DASM_PURPLE, "^"),
)

#: What the x axis of a row is called, and the gloss under it saying which
#: side is which. Two strings rather than one, because they are drawn at two
#: sizes and two weights; see :func:`background_panel`.
BACKGROUND_X_TITLE = "amino-acid identities defining each subset"
BACKGROUND_X_GLOSS = "(left: clade-founder identities; right: derived identities)"


def _em_in() -> float:
    """One em of the background rows' type, in inches."""
    return BACKGROUND_FONT_PT / 72.0


def background_y_domain(frame: pd.DataFrame) -> tuple[float, float]:
    """The value axis every sub-panel of every background row shares.

    One domain across all three clades, so a steep line means a large shift
    and not a narrow axis -- which is the whole claim these panels make.

    Computed here rather than in the notebook so that the padding and the
    renderer stay together: :func:`background_panel` draws the p-value inside
    the axes, and the headroom that leaves it room is a fact about the
    drawing.

    Args:
        frame: Every row that will be drawn, by every clade.

    Returns:
        ``(low, high)``, the data range padded by :data:`BACKGROUND_FOOTROOM`
        below and :data:`BACKGROUND_HEADROOM` above.
    """
    columns = [column for _, column, _, _ in BACKGROUND_SERIES]
    drawn = frame[columns].to_numpy(dtype=float)
    drawn = drawn[np.isfinite(drawn)]
    if drawn.size == 0:
        raise ValueError(
            "background_y_domain was handed no finite value to scale against."
        )
    low, high = float(drawn.min()), float(drawn.max())
    span = (high - low) or 1.0
    return (low - BACKGROUND_FOOTROOM * span, high + BACKGROUND_HEADROOM * span)


def background_panel(
    frame: pd.DataFrame,
    clade: str,
    *,
    y_domain: tuple[float, float],
    columns: int,
    h_params: tuple[float, float, float] | None = None,
) -> matplotlib.figure.Figure:
    """One clade's row of `fig:within_clade_shifts`: panel B, D or F.

    One sub-panel per candidate mutation, each plotting that mutation's
    fitness effect on the clade's two backgrounds by both approaches. The x
    axis is the two backgrounds named by their residues -- ``R346, K356,
    F490`` against ``T346, T356, S490`` -- so the two ticks are a
    like-for-like pair rather than "none" against a list of substitutions.
    Each is set one residue per line and horizontal, because a sub-panel is
    1.76 inches wide and a one-line ``R346, K356, F490`` fits there only
    rotated, which is three labels a reader has to turn the page to read.

    Every row carries its own legend and its own x label: each is a
    separately lettered panel of the figure, and a panel that borrows its key
    from the one above it cannot be read on its own.

    **Every row is the same number of slots wide**, whatever its candidate
    count, and the surplus slots are left blank with the first of them
    holding the legend. That, plus ``h_params``, is what makes the three rows
    line up when they are stacked left-justified.

    Args:
        frame: Columns ``candidate``, ``background``, ``background_order``,
            and one per entry of :data:`BACKGROUND_SERIES`. A ``p_label``
            column may ride along and is not drawn.
        clade: The clade this row draws, for the value-axis title.
        y_domain: ``(low, high)`` shared by every sub-panel of every row,
            from :func:`background_y_domain`.
        columns: How many slots wide every row is -- the widest clade's
            candidate count plus one for the legend.
        h_params: ``(left, right, wspace)`` taken from the first row drawn
            and imposed on the rest, after ``tight_layout`` has run. Without
            it the slots do not line up even at equal width, because
            ``tight_layout`` sizes each figure's margins from its own tick
            labels, and a row whose empty right-hand slots carry none comes
            out with wider panels than the first.

    Returns:
        The row, as a matplotlib figure, ordered left to right by decreasing
        absolute counts shift.
    """
    import matplotlib.pyplot as plt

    required = {"candidate", "background", "background_order"} | {
        column for _, column, _, _ in BACKGROUND_SERIES
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"background_panel needs column(s) {missing}.")

    order = (
        frame.sort_values("background_order")["background"].drop_duplicates().tolist()
    )
    if len(order) != 2:
        raise ValueError(
            f"{clade} has {len(order)} backgrounds, not two: {order}. The panel "
            f"draws a contrast, so a third would be plotted with nothing "
            f"saying which pair the slope belongs to."
        )

    # Steepest first, measured on counts rather than on the DASM: counts is
    # the observation the model is being checked against, so the order is set
    # by the data and not by the thing under test.
    def shift_of(candidate: str) -> float:
        part = frame[frame["candidate"] == candidate].sort_values("background_order")
        return abs(
            float(part.iloc[-1]["counts_effect"]) - float(part.iloc[0]["counts_effect"])
        )

    members = sorted(frame["candidate"].unique(), key=shift_of, reverse=True)
    if len(members) > columns - 1:
        raise ValueError(
            f"{clade} has {len(members)} candidates and the row is {columns} "
            f"slots wide, which leaves no slot for the legend."
        )

    figure, axes = plt.subplots(
        1,
        columns,
        figsize=(
            BACKGROUND_CELL_W_EM * _em_in() * columns,
            BACKGROUND_ROW_H_EM * _em_in(),
        ),
        squeeze=False,
        sharey=True,
    )
    for empty in axes[0][len(members) :]:
        empty.set_axis_off()

    for slot, candidate in enumerate(members):
        axis = axes[0][slot]
        part = frame[frame["candidate"] == candidate].sort_values("background_order")
        x = list(range(len(part)))
        for label, column, colour, marker in BACKGROUND_SERIES:
            axis.plot(
                x,
                part[column],
                marker=marker,
                color=colour,
                markersize=7,
                linewidth=1.4,
                label=label,
                zorder=3,
            )
        axis.axhline(0.0, color="#cccccc", linewidth=1, zorder=1)
        # No p-value above the mutation. `rule pool_counts` still computes the
        # randomization test and still writes it to `pool_shifts.parquet` and
        # to this panel's CSV; only the label is withheld, while how to
        # correct it for multiple testing is undecided.
        axis.set_xticks(x)
        axis.set_xticklabels(
            [label.replace(", ", "\n") for label in part["background"]],
            fontsize=BACKGROUND_FONT_PT,
        )
        axis.set_xlim(-0.45, len(part) - 0.55)
        axis.set_ylim(*y_domain)
        # The mutation alone. Every sub-panel of a row is one clade's, so
        # "in clade 20B" would be the same five words on every one of them --
        # and on a two-candidate row they are wide enough to collide. The
        # clade is on the value-axis label instead, once.
        axis.set_title(candidate, fontsize=BACKGROUND_FONT_PT, fontweight="bold")
        axis.tick_params(axis="y", labelsize=BACKGROUND_FONT_PT)
        # On every sub-panel, not only the one carrying the ticks: the grid is
        # how a reader carries a level across a row, and `sharey` hides the
        # labels, not the lines.
        axis.grid(axis="y", color="#eeeeee", zorder=0)
        axis.set_axisbelow(True)

    # Both lines go in one label so that `tight_layout` reserves the height of
    # both; the gloss is then lifted out of it below, because a label carries
    # one size and one weight, and the gloss should not read as heavy as the
    # thing it glosses.
    #
    # "defining", not "describing": a parent in the R346, K356, F490 subset
    # carries those three and whatever else it happens to have, so the label
    # names what membership requires rather than the parent's whole
    # background.
    placed_xlabel = figure.supxlabel(
        f"{BACKGROUND_X_TITLE}\n{BACKGROUND_X_GLOSS}",
        fontsize=BACKGROUND_FONT_PT,
        fontweight="bold",
        ma="center",
    )
    # "fitness effect", the paper's name for this quantity everywhere else,
    # over two lines: one line of it is taller than the panels and would set
    # the figure's height rather than read as its label. Bold to match the
    # sub-panel titles and the x label, which is what the Vega-Lite panel
    # beside this one in the composed figure does by default. `ma="center"`
    # centres the two lines on each other -- rotated text left-justifies by
    # default, which for a vertical label means the lines align along their
    # lower edge and the shorter one hangs off the top.
    placed_ylabel = figure.supylabel(
        f"fitness effect in\nclade {clade} subset",
        fontsize=BACKGROUND_FONT_PT,
        fontweight="bold",
        ma="center",
    )
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 1.0))
    if h_params is not None:
        left, right, wspace = h_params
        figure.subplots_adjust(left=left, right=right, wspace=wspace)

    # The labels and the legend are placed against the panels, not against the
    # figure, and only after `tight_layout` has given the panels their final
    # geometry. The value label and its ticks take up the left margin, so
    # anything centred at 0.5 sits visibly left of the block of plots it
    # belongs to. Nothing placed here claims any space of its own -- the
    # legend goes in a slot the padding already left empty, and the labels
    # only slide along the layout they were measured into.
    renderer = _measuring_renderer(figure)
    panel_middle = (
        min(axis.get_position().y0 for axis in axes.ravel())
        + max(axis.get_position().y1 for axis in axes.ravel())
    ) / 2
    placed_ylabel.set_y(panel_middle)

    # On this clade's own sub-panels rather than on the row's slots: the label
    # belongs under the plots it describes, not under the middle of a padding
    # that exists only to make the rows line up.
    occupied = [axes[0][slot].get_position() for slot in range(len(members))]
    label_x = (
        min(span.x0 for span in occupied) + max(span.x1 for span in occupied)
    ) / 2
    placed_xlabel.set_x(label_x)

    # Split the reserved block in two: the label keeps its top, the gloss
    # takes its bottom at one point smaller and regular weight. Measured
    # rather than offset by a guessed line height, so the pair occupies
    # exactly the space the two-line label was measured into.
    block = placed_xlabel.get_window_extent(renderer).transformed(
        figure.transFigure.inverted()
    )
    # `tight_layout` leaves the label a full line of air under the tick
    # labels, which on a row this short reads as a gap rather than as spacing.
    # Half of it is taken back by sliding the whole two-line block up,
    # measured against the lowest tick label rather than offset by a guessed
    # amount, so the label keeps the same clearance whatever the tick labels
    # turn out to be.
    ticks_bottom = min(
        axis.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for axis in axes.ravel()
    )
    lift = max(ticks_bottom - block.y1, 0.0) / 2
    placed_xlabel.set_text(BACKGROUND_X_TITLE)
    placed_xlabel.set_va("top")
    placed_xlabel.set_y(block.y1 + lift)
    figure.text(
        label_x,
        block.y0 + lift,
        BACKGROUND_X_GLOSS,
        fontsize=BACKGROUND_FONT_PT - 1,
        ha="center",
        va="bottom",
    )

    # In the first slot past this clade's sub-panels, level with them: a key
    # beside the plots rather than a band above them. Every row reserves that
    # slot, which is what the surplus column is for.
    handles, labels = axes[0][0].get_legend_handles_labels()
    spare = axes[0][len(members)] if columns > len(members) else axes[0][-1]
    drawn = figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(spare.get_position().x0, panel_middle),
        ncol=1,
        frameon=False,
        fontsize=BACKGROUND_FONT_PT,
        title="approach",
    )
    drawn.get_title().set_fontsize(BACKGROUND_FONT_PT)
    drawn.get_title().set_fontweight("bold")
    return figure


def _measuring_renderer(figure: matplotlib.figure.Figure):
    """A renderer to measure this figure's text with, whatever canvas it has.

    Under ``marimo export`` the figure already has an Agg canvas and carries
    its own renderer; in the running app it has the base canvas, which has
    none, and asking it for one is an ``AttributeError``. Attaching Agg is
    what ``savefig`` does anyway.
    """
    try:
        return figure.canvas.get_renderer()
    except AttributeError:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        return FigureCanvasAgg(figure).get_renderer()


def save_mpl_panel(
    figure: matplotlib.figure.Figure, stem: str, *, selection: str
) -> matplotlib.figure.Figure:
    """Write one matplotlib panel to this selection's figure directory.

    The matplotlib counterpart of :func:`save_panel`, writing the same two
    formats to the same place at the same resolution, so the composition can
    read a row without knowing which renderer drew it.

    ``bbox_inches="tight"`` is deliberately *not* passed. The composition
    places panels by their pixel size, and a tight box crops each one by
    however much white space it happens to carry -- so three rows drawn at
    one slot geometry would arrive at three different ones, which is the
    alignment this renderer exists to get right.

    Returns the figure, so a call can wrap an expression without
    restructuring it.
    """
    directory = figure_dir(selection)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(directory / f"{stem}.{extension}", dpi=VEGA_WRITE_PPI)
    return figure


# ---------------------------------------------------------------------------
# `fig:subsampling` panels A and B
# ---------------------------------------------------------------------------

#: Full depth against the three caps. One ink for the undisturbed tree and a
#: light-to-dark blue ramp for the caps, so depth reads as one ordered channel:
#: a reader follows the curves down as the cap tightens without consulting the
#: legend. :data:`POINT_BLUE` and friends are the scatter's
#: data-reference-fit roles, so a cap arm does not borrow one.
#:
#: One colour per cap arm, and :func:`check_cap_colours` enforces it: Vega
#: recycles a ``range`` shorter than its ``domain`` without raising, so a
#: fourth cap in `config.yaml` would silently give two arms one blue.
DEPTH_INK = "#1a1a1a"
CAP_BLUES = ("#6baed6", "#3182bd", "#08519c")

#: The accuracy threshold's rule, in the one hue neither the depth ramp nor the
#: scatter roles use.
THRESHOLD_ACCENT = "#d95f02"

#: The type every panel of `fig:subsampling` is set in, and the one place in
#: this module that is not :data:`PANEL_FONT_SIZE`. `panel_spec`'s
#: ``SUBSAMPLING_TITLE_PX`` gives the reason -- this figure puts two panels
#: across a row, so the same spec size lands smaller on the page here than
#: anywhere else -- and `tests/test_panels.py` pins the two files together.
#: Both sources of the published figure drew it at these sizes.
SUBSAMPLING_TITLE_PX = 24
SUBSAMPLING_LABEL_PX = 22

#: Panels A and B's geometry, the published figure's. A cap is drawn as a rule
#: across the whole of panel A rather than as a per-clade mark, because what it
#: does is set one height for every clade above it. Panel B's width is set so
#: that the whole panel -- plots, axes and legend -- comes out as wide as panel
#: C, which it stacks on top of in the composed figure.
CLADE_SIZE_WIDTH = 1190
CLADE_SIZE_HEIGHT = 380
ECDF_PANEL_WIDTH = 285
ECDF_PANEL_HEIGHT = 270

#: Panel A's 56 clade ticks, smaller than the rest of its type: three
#: characters rotated 90 degrees, which collide at the label size.
CLADE_TICK_PX = 19

#: Panel A's y title, and panel B's. Wrapped by hand, because Vega-Lite does
#: not wrap an axis title and either runs past its plot at these widths.
CLADE_SIZE_Y_TITLE = ["number of sequences", "per clade"]
ECDF_Y_TITLE = [
    "fraction of mutations",
    "with ≥ X actual or",
    "expected counts",
]

#: Panel A's date ramp, warm to dark with time, and the stems under its dots.
#: The five stops are `DATE_CMAP`'s own, from the published panel's notebook,
#: given as stops rather than as a scheme name because a five-stop *temporal*
#: scale comes back from vl-convert as a two-colour interpolation.
DATE_STOPS: tuple[str, ...] = (
    "#f89540",
    "#da5a6a",
    "#aa2395",
    "#6600a7",
    "#0d0887",
)
STEM_GREY = "#d9d8d1"

#: The ink panel B's full-depth curve and panel A's arrows are drawn in, and
#: the neutral its cap rules take. Where the caps are colour-coded, as in panel
#: B, they keep the single-hue :data:`BLUE_ORDINAL`, so the two scales cannot
#: be confused; on the panel that carries the dates the caps are neutral.
INK = "#0b0b0b"
INK_2 = "#52514e"

#: The single-hue ramp panel B's cap curves are drawn on, light to dark.
BLUE_ORDINAL: tuple[str, ...] = (
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)

#: Panel A's log domain, as multiples of the smallest and largest clade. A mark
#: sitting exactly on a bound is drawn half outside the frame and an unclipped
#: rule at the bound lands on the axis line, so neither end is tight; the top is
#: the looser of the two because it carries the whole arrow over the tallest
#: clade, head included, and not just that clade's dot.
DOMAIN_FLOOR_PAD = 0.7
DOMAIN_TOP_PAD = 3.0

#: The arrow over a highlighted clade, in screen pixels above its dot's centre:
#: the head's centre, then the staff's far end. The staff starts at the head's
#: centre so its end is hidden under the head rather than floating past it.
#:
#: Pixels, not multiples, because a head of a fixed size needs a fixed gap --
#: and on a log axis a multiplicative offset is a different number of pixels at
#: every height. :func:`_raised` converts.
#: These are the published panel's numbers: the dot's own radius is about 9 px,
#: so a tip at 21 leaves white space between the two.
ARROW_HEAD_PX = 21.0
ARROW_TAIL_PX = 35.0
ARROW_HEAD_SIZE = 165


def check_cap_colours(caps: Sequence[object]) -> None:
    """Refuse to draw more cap arms than :data:`CAP_BLUES` has colours."""
    if len(caps) > len(CAP_BLUES):
        raise ValueError(
            f"{len(caps)} cap arms against {len(CAP_BLUES)} colours in "
            f"CAP_BLUES. Vega recycles a short range without raising, so two "
            f"arms would share a blue in panels A and B both."
        )


def _raised(values: pd.Series, pixels: float, *, per_decade: float) -> pd.Series:
    """``values``, raised by ``pixels`` screen pixels on a log axis."""
    return values * 10 ** (pixels / per_decade)


def depth_label(arm: str) -> str:
    """One sampling depth's label, in the one spelling both panels use.

    Panel A labels its cap rules and panel B its curves, and a reader matches
    the two by that label, so they are built here rather than in each.

    The published figure's spelling, carried across from `arm_label` in
    `dasm2-experiments`, `experiments/2026-09-02-v12-sampling-depth-ladder`,
    `notebooks/ladder.py`: `cap_50000_0` is the manifest's name, whose
    trailing `_0` is the subsample seed that every arm here shares, so it
    carries nothing in a tick or a column heading.
    """
    return "full depth" if arm == FULL_DEPTH.name else f"{cap_of(arm):,} per clade"


def clade_size_panel(
    frame: pd.DataFrame,
    caps: Sequence[int],
    *,
    title: list[str] | None = None,
    highlight: Sequence[str] = (),
    width: int = CLADE_SIZE_WIDTH,
    height: int = CLADE_SIZE_HEIGHT,
) -> alt.LayerChart:
    """Panel A: sequences per clade, coloured by when the clade emerged.

    **Ported from the published figure's own producer**, the
    `panel_a_sequences_per_clade` cell of `notebooks/cap_arm_scatters.py` in
    the mat-pcp run this paper is built on, which `main.tex`'s provenance
    comment names.

    A dot per clade on a log y axis, a stem down to the floor, one neutral
    rule per downsampling cap labelled on the line, and an arrow over each
    clade in ``highlight`` -- panel B's two, so a reader arriving from its
    titles finds them here without counting along 56 ticks.

    **The colour is the clade's emergence date**, and it is what turns a
    ranked list of 56 counts into a picture of the pandemic: the clades with
    millions of sequences are the early ones and the modern clades sit at the
    bottom, which the heights alone do not say. It is encoded as a **number**,
    the date as a decimal year, rather than as a date: a five-stop range on a
    temporal scale comes back from vl-convert as a two-colour interpolation,
    where on a quantitative scale the same five stops interpolate the way the
    matplotlib colormap does, which is the point of giving them.

    **A log y axis**, because the clades span four orders of magnitude. On a
    linear axis every clade but the largest five is a flat line against the
    axis, which is the opposite of the panel's point.

    ``title`` is accepted and unused: this panel carries no title in the
    published figure, where the caption and the axis titles say what it is.

    Raises:
        ValueError: If ``frame`` carries no ``year`` column. The dots cannot
            be coloured by a date the caller did not measure, and a panel that
            silently fell back to one ink would be the pre-port drawing again.
    """
    if "year" not in frame.columns:
        raise ValueError(
            "clade_size_frame must carry a `year` column -- the clade's "
            "emergence as a decimal year, which `rule clade_emergence` "
            "measures. Panel A colours its dots by it."
        )
    order = list(frame["clade"])
    smallest = float(frame["n_sequences"].min())
    largest = float(frame["n_sequences"].max())
    floor = smallest * DOMAIN_FLOOR_PAD
    # The top of the domain carries the whole arrow over the tallest clade,
    # not just its dot: at a tighter bound the head is half off the panel.
    top = largest * DOMAIN_TOP_PAD
    y_scale = alt.Scale(type="log", domain=[floor, top], nice=False)
    # Pixels per decade, so the arrow can be drawn to a pixel spec -- a head of
    # a fixed size needs a fixed gap, and a multiplicative offset would mean
    # something different at every height.
    px_per_decade = height / math.log10(top / floor)

    # One scale and one axis object, shared by every layer: a layer that set
    # `title=None` on the shared y would blank the axis title for all of them.
    y_axis = alt.Axis(
        labelFontSize=SUBSAMPLING_LABEL_PX,
        titleFontSize=SUBSAMPLING_TITLE_PX,
        values=[1e3, 1e4, 1e5, 1e6, 1e7],
        format="~s",
    )
    x_enc = alt.X(
        "clade:N",
        sort=order,
        title="clade",
        axis=alt.Axis(
            labelAngle=-90,
            labelFontSize=CLADE_TICK_PX,
            titleFontSize=SUBSAMPLING_TITLE_PX,
        ),
    )
    y_enc = alt.Y("n_sequences:Q", scale=y_scale, title=CLADE_SIZE_Y_TITLE, axis=y_axis)

    rows = frame.assign(floor=floor)
    known = [value for value in rows["year"] if pd.notna(value)]
    # The ramp starts at the turn of the earliest clade's year, so that year
    # gets a tick: a tick outside the colour domain is not drawn, and rounding
    # the domain itself up leaves the bottom of the bar unlabelled.
    low_year = float(np.floor(min(known)))
    high_year = float(max(known))
    year_ticks = list(range(int(low_year), int(np.floor(high_year)) + 1))

    cap_rows = pd.DataFrame({"cap": sorted(caps)})
    cap_rows["clade"] = order[-1]
    cap_rows["label"] = [f"{cap:,}" for cap in cap_rows["cap"]]

    # The stem is a guide to the axis, not a magnitude: a log axis has no zero
    # for a bar to stand on.
    stems = (
        alt.Chart(rows)
        .mark_rule(color=STEM_GREY, strokeWidth=1)
        .encode(
            x=x_enc,
            y=alt.Y("floor:Q", scale=y_scale, title=CLADE_SIZE_Y_TITLE, axis=y_axis),
            y2="n_sequences:Q",
        )
    )
    dots = (
        alt.Chart(rows)
        .mark_point(filled=True, size=255, stroke="white", strokeWidth=1.2)
        .encode(
            x=x_enc,
            y=y_enc,
            color=alt.Color(
                "year:Q",
                scale=alt.Scale(
                    domain=list(np.linspace(low_year, high_year, len(DATE_STOPS))),
                    range=list(DATE_STOPS),
                ),
                # A tick per calendar year the clades span, including the
                # first. The title runs up the side rather than across the top,
                # which costs one line of width instead of the title's length.
                legend=alt.Legend(
                    title="clade emergence date",
                    values=year_ticks,
                    format="d",
                    labelFontSize=SUBSAMPLING_LABEL_PX,
                    titleFontSize=SUBSAMPLING_TITLE_PX,
                    titleLimit=0,
                    titleOrient="right",
                    # The full plot height, which is also what centres it: a
                    # shorter bar is drawn from the top of the plotting area
                    # down, leaving the remainder at the bottom.
                    gradientLength=height,
                ),
            ),
            tooltip=["clade", "n_sequences"],
        )
    )
    # The caps are reference lines, not data: neutral, and labelled on the line
    # rather than in a legend, so the colour channel is the dates alone.
    cap_rules = (
        alt.Chart(cap_rows)
        .mark_rule(strokeDash=[6, 4], color=INK_2, strokeWidth=1.2)
        .encode(y=alt.Y("cap:Q", scale=y_scale, title=CLADE_SIZE_Y_TITLE, axis=y_axis))
    )
    cap_text = (
        alt.Chart(cap_rows)
        .mark_text(
            align="right",
            baseline="bottom",
            dx=-3,
            dy=-3,
            fontSize=SUBSAMPLING_LABEL_PX,
            color=INK_2,
        )
        .encode(
            x=x_enc,
            y=alt.Y("cap:Q", scale=y_scale, title=CLADE_SIZE_Y_TITLE, axis=y_axis),
            text="label:N",
        )
    )
    marks = rows[rows["clade"].isin(list(highlight))].copy()
    # Head, then staff, measured up from the dot's centre: the dot's own radius
    # is about 9 px, so a tip at 21 leaves white space between the two. The
    # staff starts at the head's centre, so its end is hidden under the head
    # rather than floating above it.
    marks["y_head"] = _raised(
        marks["n_sequences"], ARROW_HEAD_PX, per_decade=px_per_decade
    )
    marks["y_tail"] = marks["y_head"]
    marks["y_tail_end"] = _raised(
        marks["n_sequences"], ARROW_TAIL_PX, per_decade=px_per_decade
    )
    staff = (
        alt.Chart(marks)
        .mark_rule(color=INK, strokeWidth=1.8, opacity=1)
        .encode(
            x=x_enc,
            y=alt.Y("y_tail:Q", scale=y_scale, title=CLADE_SIZE_Y_TITLE, axis=y_axis),
            y2="y_tail_end:Q",
        )
    )
    head = (
        alt.Chart(marks)
        .mark_point(
            shape="triangle-down",
            filled=True,
            size=ARROW_HEAD_SIZE,
            color=INK,
            opacity=1,
        )
        .encode(
            x=x_enc,
            y=alt.Y("y_head:Q", scale=y_scale, title=CLADE_SIZE_Y_TITLE, axis=y_axis),
        )
    )
    return (stems + cap_rules + dots + cap_text + staff + head).properties(
        width=width, height=height
    )


def count_ecdf_panel(
    frame: pd.DataFrame,
    *,
    clade_titles: dict[str, str],
    threshold: float,
    width: int = ECDF_PANEL_WIDTH,
    height: int = ECDF_PANEL_HEIGHT,
) -> alt.FacetChart:
    """Panel B: one complementary eCDF per clade, one curve per sampling depth.

    **Ported from the published figure's own producer**, the
    `panel_b_count_ecdf` cell of `notebooks/cap_arm_scatters.py` in the mat-pcp
    run this paper is built on.

    Each curve is the share of a clade's mutations carrying **at least** *x*
    counts, so it falls from 1.0 rather than rising to it. A labelled rule
    marks the accuracy threshold, and a curve's height where it crosses is the
    share that depth still puts at or above it.

    One frame with a ``role`` column and each layer filtering to its own role,
    because Vega will not facet a layered chart whose layers carry their own
    data -- the same construction panel C uses, and for the same reason.

    Full depth in ink, the caps on a single-hue ramp light to dark: bigger cap,
    more data kept, darker line. Width rides on the same field as colour, so
    the baseline is heavier as well as darker and the two channels merge into
    one legend whose swatches carry both.
    """
    arms = list(frame["arm"].unique())
    caps = [arm for arm in arms if arm != FULL_DEPTH.name]
    check_cap_colours(caps)
    depth_order = [depth_label(FULL_DEPTH.name)] + [depth_label(arm) for arm in caps]
    indices = np.linspace(len(BLUE_ORDINAL) - 1, 0, len(caps)).round().astype(int)
    colour_range = [INK] + [BLUE_ORDINAL[index] for index in indices]

    parts = []
    for clade, clade_title in clade_titles.items():
        panel = f"clade {clade_title}"
        rows = frame[frame["clade"] == clade]
        parts.append(
            pd.DataFrame(
                {
                    "panel": panel,
                    "role": "curve",
                    "depth": [depth_label(arm) for arm in rows["arm"]],
                    "x": rows["x"].to_numpy(),
                    "share": rows["share"].to_numpy(),
                }
            )
        )
        parts.append(
            pd.DataFrame(
                {
                    "panel": [panel],
                    "role": ["threshold"],
                    "depth": [depth_label(FULL_DEPTH.name)],
                    "x": [threshold],
                    "share": [np.nan],
                }
            )
        )
    long = pd.concat(parts, ignore_index=True)

    x_enc = alt.X(
        "x:Q",
        scale=alt.Scale(domain=[0, float(frame["x"].max())], nice=False),
        title="X counts",
        axis=alt.Axis(
            values=[0, 5, 10, 15, 20],
            labelFontSize=SUBSAMPLING_LABEL_PX,
            titleFontSize=SUBSAMPLING_TITLE_PX,
        ),
    )
    y_enc = alt.Y(
        "share:Q",
        scale=alt.Scale(domain=[0, 1.02], nice=False),
        title=ECDF_Y_TITLE,
        axis=alt.Axis(
            labelFontSize=SUBSAMPLING_LABEL_PX, titleFontSize=SUBSAMPLING_TITLE_PX
        ),
    )
    # One legend object for both the colour and the width channel: Vega merges
    # two channels on the same field only when their legends agree.
    legend = alt.Legend(
        title="sampling depth",
        labelFontSize=SUBSAMPLING_LABEL_PX,
        titleFontSize=SUBSAMPLING_TITLE_PX,
        labelLimit=0,
        titleLimit=0,
    )
    base = alt.Chart()
    layers = [
        base.mark_line()
        .transform_filter(alt.datum.role == "curve")
        .encode(
            x=x_enc,
            y=y_enc,
            color=alt.Color(
                "depth:N",
                sort=depth_order,
                scale=alt.Scale(domain=depth_order, range=colour_range),
                legend=legend,
            ),
            strokeWidth=alt.StrokeWidth(
                "depth:N",
                sort=depth_order,
                scale=alt.Scale(domain=depth_order, range=[3.4] + [2.0] * len(caps)),
                legend=legend,
            ),
        ),
        base.mark_rule(strokeDash=[2, 3], color=FIT_FIREBRICK, strokeWidth=1.8)
        .transform_filter(alt.datum.role == "threshold")
        .encode(x=x_enc),
        base.mark_text(
            align="left",
            baseline="top",
            dx=5,
            dy=4,
            lineBreak="|",
            fontSize=SUBSAMPLING_LABEL_PX,
            color=FIT_FIREBRICK,
        )
        .transform_filter(alt.datum.role == "threshold")
        .transform_calculate(note='"accuracy|threshold"')
        .encode(x=x_enc, y=alt.value(4), text="note:N"),
    ]
    # Narrow views with the legend to their right. The width is set so that the
    # whole panel -- plots, axes and legend -- comes out as wide as panel C,
    # which it stacks on top of in the composed figure.
    return (
        alt.layer(*layers, data=long)
        .properties(width=width, height=height)
        .facet(
            column=alt.Column(
                "panel:N",
                sort=[f"clade {title}" for title in clade_titles.values()],
                title=None,
                header=alt.Header(
                    labelFontSize=SUBSAMPLING_TITLE_PX, labelFontWeight="bold"
                ),
            ),
            spacing=24,
        )
    )


# ---------------------------------------------------------------------------
# `fig:subsampling` panels C and D
# ---------------------------------------------------------------------------

# Every constant below is the published figure's, carried across from the
# producers `main.tex`'s provenance comment names: the `panel_c_arm_grid` and
# `panel_d_clade_ladder` cells of `notebooks/ladder.py` in `dasm2-experiments`,
# `experiments/2026-09-02-v12-sampling-depth-ladder`. The panels have to come
# out the shapes `SUBSAMPLING_FIGURE_ROWS` places, so these are measurements of
# that figure rather than choices to make again.

#: Panel C's grid titles. The **target is x**, which is what
#: `within_clade_metrics` regresses on, so the drawn slope and the reported one
#: are the same number read two ways. Each is said once for the whole grid --
#: the x title under it, the y title rotated down its left -- rather than once
#: per cell, which is most of why eight cells fit a short, wide panel.
ARM_GRID_X_TITLE = "counts-based fitness effect computed at full depth"
ARM_GRID_Y_TITLE = ["fitness effect computed", "at a given sampling depth"]

#: What labels panel C's rows, which is the approach rather than the bare
#: method name the tables are keyed by.
ARM_METHOD_ROW_LABELS = {"counts": "counts approach", "DASM": "DASM approach"}

#: Panel C's cell. 190 rather than something smaller because the row label is
#: rotated: a vertical label has to fit inside its row's height, and
#: `ARM_GRID_Y_TITLE`'s longer line needs about this much at this type size.
ARM_SCATTER_SIZE = 190

#: Above this many points a cell of panel C is drawn as density rather than as
#: dots, in `ARM_SCATTER_BINS` bins a side. 21J's cells hold about 4,900
#: mutations each: a raw scatter there is an ink blob whose shape is set by
#: draw order, and the spec it embeds is large enough that marimo drops the
#: output entirely -- the chart disappears rather than rendering badly.
ARM_SCATTER_MAX_N = 2000
ARM_SCATTER_BINS = 60

#: Panel D's cell. Narrow and short, because ten of them stack five rows deep
#: in the tall, narrow panel that stands beside the B-over-C column.
ARM_LADDER_WIDTH = 130
ARM_LADDER_HEIGHT = 82

#: Panel D's x axis, legend and slope ticks. Vertical labels, not slanted: a
#: slanted label is harder to read and its bounding box is wider than the band
#: it belongs to. The slope ticks are a quarter, no compression and four times
#: -- the two ends the same distance from 1 on a log2 axis, so a line's
#: distance from the middle tick reads the same whichever way it is off.
ARM_X_TITLE = "sampling depth"
ARM_LEGEND_TITLE = "approach"
ARM_X_AXIS = alt.Axis(labelAngle=-90, labelAlign="right", labelBaseline="middle")
ARM_SLOPE_TICKS = (0.25, 1.0, 4.0)
#: Wide enough to hold both the data and the labelled ticks with room to spare
#: on each side, which is what keeps every label interior -- see
#: :func:`arm_ladder_grid` on why the two columns' rows align only then.
ARM_TICK_MARGIN = 1.4

#: The colour each approach is drawn in, shared with panel B of
#: `fig:within_clade_shifts`: one approach is one colour across the paper.
#: Shape carries the same split, so the DASM series is the triangle here that
#: it is in `fig:agreement`, in print and in black and white alike.
ARM_METHOD_COLOURS = {"counts": COUNTS_GREY, "DASM": DASM_PURPLE}
ARM_METHOD_SHAPES = {"counts": "circle", "DASM": "triangle"}


def arm_fit_line(
    frame: pd.DataFrame,
    *,
    x_field: str,
    y_field: str,
    slope: float,
    domain: Sequence[float],
) -> pd.DataFrame:
    """The two endpoints of the reported regression line.

    **Drawn from the statistic, not fitted to what is plotted.** Vega's
    ``transform_regression`` fits the rows it is handed, so under any point
    budget the line would come from a sample while the corner label came from
    the whole cloud, and the two would disagree by an amount nobody could see.
    The OLS line passes through the means, so the slope the CSV reports plus
    the full frame's means fix the line completely.
    """
    if frame.empty or pd.isna(slope):
        return pd.DataFrame({x_field: [], y_field: []})
    intercept = float(frame[y_field].mean()) - slope * float(frame[x_field].mean())
    x = [float(domain[0]), float(domain[1])]
    return pd.DataFrame({x_field: x, y_field: [intercept + slope * v for v in x]})


def arm_scatter_grid(
    frames: dict[tuple[str, str], pd.DataFrame],
    metrics: pd.DataFrame,
    *,
    arms: Sequence[str],
    methods: Sequence[str],
    x_field: str,
    y_field: str,
    clip_lower: float,
    clade: str,
    point_budget: int | None = None,
) -> alt.FacetChart:
    """Panel C: one row per approach, one column per sampling depth.

    **Ported from the published figure's own producer**, the `panel_c_arm_grid`
    cell of `notebooks/ladder.py` in `dasm2-experiments`,
    `experiments/2026-09-02-v12-sampling-depth-ladder`, which `main.tex`'s
    provenance comment names. The shape is load bearing twice over: it is what
    the caption describes, and `SUBSAMPLING_FIGURE_ROWS` places this panel as
    the short, wide one in a column under B.

    One frame for every mark, with a ``role`` column, then **one faceted grid**
    rather than eight concatenated panels. A single grid is what makes the rows
    align, the arm heading sit centred over its column, and the method name
    appear once per row instead of in all eight titles. Altair will not facet a
    layered chart whose layers carry their own frames, so the points, the fit
    line and its label all live in that one frame and each layer filters to its
    own role.

    Full depth is the leftmost column, so reading right is reading the thinning
    get harsher. The full-depth counts cell is the target against itself and
    sits at r = 1 by construction, which the caption says.
    """
    sizes = {key: len(frame) for key, frame in frames.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(
            f"cells do not score the same mutations: {sizes}. Every cell of "
            f"this grid must carry the same n, or one of them is answering a "
            f"different question than the others."
        )
    n_per_cell = next(iter(sizes.values()))
    # Above this many points a cell is drawn as density rather than as dots.
    # A raw scatter of 5,000 mutations is an ink blob whose shape is set by
    # draw order, and the spec it embeds is large enough that marimo drops the
    # output entirely. Binning happens in pandas, not in Vega, because a
    # Vega-side `bin` still ships every row to the browser.
    dense = n_per_cell > ARM_SCATTER_MAX_N

    low = clip_lower
    high = max(
        float(max(frame[x_field].max(), frame[y_field].max()))
        for frame in frames.values()
        if not frame.empty
    )
    span = alt.Scale(domain=[low, high])
    edges = np.linspace(low, high, ARM_SCATTER_BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    parts: list[pd.DataFrame] = []
    for method in methods:
        for arm in arms:
            cell = frames[(arm, method)]
            keys = {
                "method": ARM_METHOD_ROW_LABELS[method],
                "arm_label": depth_label(arm),
            }
            row = metrics[(metrics["arm"] == arm) & (metrics["method"] == method)]
            if row.empty:
                raise ValueError(
                    f"no statistics for arm {arm!r}, method {method!r}: the "
                    f"corner label would disagree with the cloud beside it."
                )
            stats_row = row.iloc[0]
            slope = float(stats_row["regression_slope"])
            if dense:
                binned = cell.assign(
                    xb=centres[
                        np.clip(
                            np.digitize(cell[x_field], edges) - 1,
                            0,
                            ARM_SCATTER_BINS - 1,
                        )
                    ],
                    yb=centres[
                        np.clip(
                            np.digitize(cell[y_field], edges) - 1,
                            0,
                            ARM_SCATTER_BINS - 1,
                        )
                    ],
                )
                cells = (
                    binned.groupby(["xb", "yb"], observed=True)
                    .size()
                    .reset_index(name="n")
                )
                parts.append(
                    pd.DataFrame(
                        {
                            **keys,
                            "role": "bin",
                            "x": cells["xb"].to_numpy(),
                            "y": cells["yb"].to_numpy(),
                            "n": cells["n"].to_numpy(),
                        }
                    )
                )
            else:
                drawn = cell
                if point_budget is not None and len(cell) > point_budget:
                    # Sampled with a fixed seed so a rebuild redraws the same
                    # cloud, and after the fit, which is why thinning cannot
                    # move the line.
                    drawn = cell.sample(n=point_budget, random_state=0)
                parts.append(
                    pd.DataFrame(
                        {
                            **keys,
                            "role": "point",
                            "x": drawn[x_field].to_numpy(),
                            "y": drawn[y_field].to_numpy(),
                        }
                    )
                )
            # The fit line spans **this cell's observed x range**, not the
            # shared axis. Extrapolated to the axis ends a steep slope leaves
            # the cell entirely, and Vega then pads the view to contain it,
            # which stretches every cell in the row. Its endpoints come from
            # the slope that is printed, and the OLS line passes through the
            # means, so the line and the number cannot disagree.
            intercept = float(cell[y_field].mean()) - slope * float(
                cell[x_field].mean()
            )
            x_lo, x_hi = float(cell[x_field].min()), float(cell[x_field].max())
            parts.append(
                pd.DataFrame(
                    {
                        **keys,
                        "role": "fit",
                        "x": [x_lo, x_hi],
                        "y": [slope * x_lo + intercept, slope * x_hi + intercept],
                    }
                )
            )
            # The y = x reference does span the whole axis -- it is a reference
            # rather than a fit, and clipping it to the data would make it look
            # like a second fitted line.
            parts.append(pd.DataFrame({**keys, "role": "diagonal", "x": [low, high]}))
            # Two lines of annotation, positioned in data space rather than by
            # a mark offset: `dy` is a mark property and cannot vary per datum,
            # and at this type size one line is wider than the cell.
            parts.append(
                pd.DataFrame(
                    {
                        **keys,
                        "role": "label",
                        "x": low,
                        "y": [high, high - 0.09 * (high - low)],
                        "text": [
                            f"r = {stats_row['pearson_r']:.2g}",
                            f"slope = {slope:.2f}",
                        ],
                    }
                )
            )
    frame = pd.concat(parts, ignore_index=True)

    base = alt.Chart()
    x_enc = alt.X("x:Q", scale=span, title=None)
    y_enc = alt.Y("y:Q", scale=span, title=None)
    layers = [
        base.mark_rect(
            width=ARM_SCATTER_SIZE / ARM_SCATTER_BINS + 0.6,
            height=ARM_SCATTER_SIZE / ARM_SCATTER_BINS + 0.6,
        )
        .transform_filter(alt.datum.role == "bin")
        .encode(
            x=x_enc,
            y=y_enc,
            # `extent` lifts the pale end of the ramp off the page: at the
            # scheme's own 0 a singleton bin is nearly white, and the cloud's
            # outline -- which is most of what the eye reads here -- is the
            # part made of singletons.
            color=alt.Color(
                "n:Q",
                scale=alt.Scale(
                    type="log", scheme=alt.SchemeParams("blues", extent=[0.2, 1.0])
                ),
                legend=None,
            ),
            tooltip=["x", "y", "n"],
        )
        if dense
        else base.mark_circle(size=14, opacity=0.4, color=POINT_BLUE)
        .transform_filter(alt.datum.role == "point")
        .encode(x=x_enc, y=y_enc),
        base.mark_line(
            strokeDash=[4, 3], color=REFERENCE_GREY, strokeWidth=1.5, clip=True
        )
        .transform_filter(alt.datum.role == "diagonal")
        .encode(x=x_enc, y=alt.Y("x:Q", scale=span, title=None)),
        # `clip` so that a fit whose ends still fall outside the shared y span
        # is cut at the cell edge rather than drawn past it.
        base.mark_line(color=FIT_FIREBRICK, strokeWidth=2, clip=True)
        .transform_filter(alt.datum.role == "fit")
        .encode(x=x_enc, y=y_enc),
        base.mark_text(
            align="left", baseline="top", dx=4, fontSize=SUBSAMPLING_LABEL_PX
        )
        .transform_filter(alt.datum.role == "label")
        .encode(x=x_enc, y=y_enc, text="text:N"),
    ]

    return (
        alt.layer(*layers, data=frame)
        .properties(width=ARM_SCATTER_SIZE, height=ARM_SCATTER_SIZE)
        .facet(
            # The method labels the row, on the left, once. The arm heads the
            # column, centred: a column facet header is drawn over the plotting
            # area only, excluding the row-label band and the y axis.
            row=alt.Row(
                "method:N",
                sort=[ARM_METHOD_ROW_LABELS[m] for m in methods],
                # The header labels name the approach, so the quantity the two
                # share is said once here, down the left -- the mirror of the x
                # title under the grid.
                title=ARM_GRID_Y_TITLE,
                # Rotated, like a y-axis title, and for the same reason: a
                # horizontal label of this length reserves a header band as
                # wide as the text and leaves the plots stranded to the right
                # of it. Vertical costs one line height instead.
                header=alt.Header(
                    labelOrient="left",
                    labelAngle=-90,
                    labelAlign="center",
                    labelBaseline="middle",
                    labelPadding=2,
                    labelFontSize=SUBSAMPLING_TITLE_PX,
                    labelFontWeight="bold",
                    titleFontSize=SUBSAMPLING_TITLE_PX,
                    titleFontWeight="bold",
                ),
            ),
            column=alt.Column(
                "arm_label:N",
                sort=[depth_label(arm) for arm in arms],
                # The x label once, under the whole grid. A facet draws one x
                # axis per column, so putting it on the encoding would print it
                # four times.
                #
                # It rides on the column facet rather than on the chart, and
                # that is the whole point: a chart title anchors to the middle
                # of the *view*, which includes the rotated row-label band and
                # the y axis down the left, so it came out left of the plots it
                # names. A facet title anchors to the column band, the same
                # span the arm headers above are centred on.
                title=ARM_GRID_X_TITLE,
                header=alt.Header(
                    labelFontSize=SUBSAMPLING_TITLE_PX,
                    labelFontWeight="bold",
                    titleOrient="bottom",
                    titleAnchor="middle",
                    titleFontSize=SUBSAMPLING_TITLE_PX,
                    titleFontWeight="bold",
                ),
            ),
            spacing={"row": 34, "column": 14},
        )
        .configure_axis(
            labelFontSize=SUBSAMPLING_LABEL_PX, titleFontSize=SUBSAMPLING_LABEL_PX
        )
        .configure_header(labelLimit=0)
    )


def arm_ladder_grid(
    metrics: pd.DataFrame,
    *,
    clades: Sequence[str],
    arms: Sequence[str],
    methods: Sequence[str],
    r_floor: float | None = None,
) -> alt.HConcatChart:
    """Panel D: r and the OLS slope, per clade, across the sampling depths.

    **Ported from the published figure's own producer**, the
    `panel_d_clade_ladder` cell of `notebooks/ladder.py` in `dasm2-experiments`,
    `experiments/2026-09-02-v12-sampling-depth-ladder`.

    **Clades down, the two statistics across.** That is what makes this the
    tall, narrow panel `SUBSAMPLING_FIGURE_ROWS` stands beside the column of B
    over C; laid out the other way it is a wide panel with nowhere to go.

    Two faceted charts side by side rather than one grid faceted on the
    statistic: a scale *type* cannot vary by facet column, so the slope column
    could not be logarithmic. **The rows align only because no tick label sits
    on a panel edge.** Vega pads a view to fit whatever its outermost label
    overflows, and two concatenated grids padded by different amounts have
    their rows offset against each other. Both domains below are therefore
    padded so that every labelled tick is interior; changing one without
    keeping its ticks interior will pull the two columns out of step again.

    **The slope column is on log2**, labelled in ratios: a quarter, no
    compression, four times. A slope is a ratio, so halving and doubling the
    dynamic range are the same size of error in opposite directions, and a
    linear axis makes the halving look smaller.

    ``r_floor`` is ``None`` here, unlike :data:`LADDER_R_FLOOR`, which clamps at
    zero: how far the counts side falls at the shallowest cap is this panel's
    claim, and a floor would hide the bottom of it.
    """
    long = metrics[
        metrics["method"].isin(list(methods)) & metrics["clade"].isin(list(clades))
    ].copy()
    if r_floor is not None:
        long["pearson_r"] = long["pearson_r"].clip(lower=r_floor)
    arm_names = [depth_label(arm) for arm in arms]

    # r is a correlation, so its axis is the whole of its range and nothing is
    # gained by cropping. The domain is padded a hair past 0 and 1 for the
    # layout reason above, not a statistical one.
    r_scale = alt.Scale(domain=[-0.04, 1.04], nice=False)
    r_axis = alt.Axis(values=[0.0, 0.5, 1.0], labelOverlap="greedy")

    # A non-positive slope cannot be placed on a log axis at all. None occur in
    # this readout, so this is a guard rather than a code path: it says so
    # loudly instead of dropping points that would silently vanish.
    dropped = int((long["regression_slope"] <= 0).sum())
    if dropped:
        raise ValueError(
            f"{dropped} non-positive slopes have no logarithm and cannot be "
            f"placed on this panel's slope axis."
        )
    reach = float(np.abs(np.log2(long["regression_slope"])).max()) * 1.05
    slope_domain = [
        min(2**-reach, ARM_SLOPE_TICKS[0] / ARM_TICK_MARGIN),
        max(2**reach, ARM_SLOPE_TICKS[-1] * ARM_TICK_MARGIN),
    ]
    slope_scale = alt.Scale(
        type="log", base=2, domain=slope_domain, nice=False, clamp=False
    )
    slope_axis = alt.Axis(
        values=list(ARM_SLOPE_TICKS), format=".3~g", labelOverlap="greedy"
    )

    colours = alt.Scale(
        domain=list(methods), range=[ARM_METHOD_COLOURS[m] for m in methods]
    )
    shapes = alt.Scale(
        domain=list(methods), range=[ARM_METHOD_SHAPES[m] for m in methods]
    )

    def column(field: str, title: str, scale, axis, row_labels: bool):
        # The axis reads in plain English, so the plotted field is the label
        # rather than the manifest's arm name.
        rows = (
            long.rename(columns={field: "value"})[["arm", "clade", "method", "value"]]
            .assign(role="data")
            .assign(arm_name=lambda d: d["arm"].map(depth_label))
        )
        # One rule per cell, at the trailing edge of the full-depth band, which
        # is the boundary between the baseline and the downsampled arms.
        seps = pd.DataFrame(
            [
                {
                    "arm": arms[0],
                    "arm_name": depth_label(arms[0]),
                    "clade": clade,
                    "method": methods[0],
                    "value": np.nan,
                    "role": "separator",
                }
                for clade in clades
            ]
        )
        # `statistic` is a constant column carrying the column heading: a
        # single-valued column facet is what centres the title over the plots.
        frame = pd.concat([rows, seps], ignore_index=True).assign(statistic=title)
        x_enc = alt.X("arm_name:N", sort=arm_names, title=ARM_X_TITLE, axis=ARM_X_AXIS)
        y_enc = alt.Y("value:Q", title=None, scale=scale, axis=axis)
        base = alt.Chart()
        layers = [
            base.mark_rule(strokeDash=[4, 4], color="grey", opacity=0.8)
            .transform_filter(alt.datum.role == "separator")
            # The x definition here has to match the line layer's exactly, apart
            # from `bandPosition`. Layers share one x scale and Vega merges
            # identical axis definitions into one -- but `axis=None` on any
            # layer suppresses the shared axis, which is what left this figure
            # with no ticks and no x label at all.
            .encode(
                x=alt.X(
                    "arm_name:N",
                    sort=arm_names,
                    title=ARM_X_TITLE,
                    axis=ARM_X_AXIS,
                    bandPosition=1.0,
                )
            ),
            base.mark_line(point=alt.OverlayMarkDef(size=180))
            .transform_filter(alt.datum.role == "data")
            .encode(
                x=x_enc,
                y=y_enc,
                color=alt.Color("method:N", title=ARM_LEGEND_TITLE, scale=colours),
                # Shape on the same field as colour, so Vega-Lite merges the two
                # into one legend -- and so the DASM series is the triangle here
                # that it is in `fig:agreement`, in print and in black and white
                # alike. The same title as colour: Vega merges two channels on
                # one field into a single legend only when their definitions
                # agree.
                shape=alt.Shape("method:N", title=ARM_LEGEND_TITLE, scale=shapes),
                tooltip=["clade", "arm_name", "method", "value"],
            ),
        ]
        return (
            alt.layer(*layers, data=frame)
            .properties(width=ARM_LADDER_WIDTH, height=ARM_LADDER_HEIGHT)
            .facet(
                # Clade labels once, to the left of the left column. The right
                # column plots the same five rows in the same order, so a second
                # copy would only be a second thing to read. The statistic heads
                # its column, centred: a column facet header is drawn over the
                # plotting area only, excluding the row-label band and the axis.
                row=alt.Row(
                    "clade:N",
                    sort=list(clades),
                    # What the rows are, said once down the left, the way the
                    # scatter grid says its quantity. The labels themselves are
                    # rotated and pulled in against the axis so the title fits
                    # beside them rather than pushing the plots right.
                    title="clade" if row_labels else None,
                    header=alt.Header(
                        labelOrient="left",
                        labelAngle=-90,
                        labelAlign="center",
                        labelBaseline="middle",
                        labelPadding=2,
                        labelFontSize=SUBSAMPLING_LABEL_PX,
                        titleFontSize=SUBSAMPLING_LABEL_PX,
                        titleFontWeight="bold",
                    )
                    if row_labels
                    else None,
                ),
                column=alt.Column(
                    "statistic:N",
                    title=None,
                    header=alt.Header(
                        labelFontSize=SUBSAMPLING_TITLE_PX, labelFontWeight="bold"
                    ),
                ),
                spacing={"row": 14, "column": 20},
            )
        )

    return (
        alt.hconcat(
            column("pearson_r", "Pearson r", r_scale, r_axis, True),
            column("regression_slope", "slope", slope_scale, slope_axis, False),
        )
        .resolve_scale(color="shared")
        .configure_axis(
            labelFontSize=SUBSAMPLING_LABEL_PX, titleFontSize=SUBSAMPLING_LABEL_PX
        )
        .configure_legend(
            labelFontSize=SUBSAMPLING_LABEL_PX, titleFontSize=SUBSAMPLING_LABEL_PX
        )
    )


# ---------------------------------------------------------------------------
# `fig:probabilities`
# ---------------------------------------------------------------------------

#: One colour for every bar, and it is `fig:dms`'s: these two bar panels are
#: read on facing pages and a hue that groups there and not here would read as
#: an encoding. Nothing is lost by dropping the per-experiment hues, because
#: the arms are named under every bar and the experiment heads its own group,
#: so the channel was spending itself on distinctions the labels already make.
PROBABILITIES_BAR_COLOR = DMS_ROW_COLORS[0]

#: How each arm is named under its bar. Short, because they are set vertically.
PROBABILITIES_ARM_LABELS: dict[str, str] = {
    "dasm": "DASM fitness effect",
    "dasm_prob": "DASM probability",
    "antigen": "antiGen probability",
}

#: The scatter's side. Square, and that is the whole point: one nat on x is one
#: nat on y, so the cloud's slope-1 structure reads at 45 degrees and the
#: iso-colour contours run parallel to the diagonal. An aspect ratio that
#: distorted the slope would make that claim unreadable.
PROBABILITIES_SCATTER_SIDE = 235

#: How a bar's value is printed above it, which is `r`'s format everywhere
#: else in these figures. Two significant figures while ``0.1 <= r < 1``, which
#: `tests/test_probabilities.py` asserts the drawn values stay inside: the
#: differences this figure is about are in the second decimal, and more digits
#: would imply a precision these correlations do not carry.
PROBABILITIES_BAR_FORMAT = DMS_R_FORMAT

#: The padding the equal-span domains carry, as a multiple of the wider span.
PROBABILITIES_DOMAIN_PAD = 1.08


def probabilities_equal_span_domains(
    frame: pd.DataFrame, x: str, y: str
) -> dict[str, list[float]]:
    """Domains of equal width on both axes, each centred on its own column.

    Equal data-units-per-pixel is what makes the slope readable, and the panel
    is already square, so the two domains need only equal *spans*. Each is
    centred on its own midrange rather than shared, because ``log p`` runs near
    -11 while ``log f`` runs near 0: a shared domain would put one cloud off
    the panel entirely.

    **Not** :func:`dms_padded_domain`, which pads each axis independently by
    5% and therefore returns unequal spans by construction. That helper is the
    repository's usual one and is the wrong one here.

    Returns:
        ``{x: [low, high], y: [low, high]}``, the two spans equal.
    """
    spans = [float(frame[column].max() - frame[column].min()) for column in (x, y)]
    half = max(spans) * PROBABILITIES_DOMAIN_PAD / 2
    out: dict[str, list[float]] = {}
    for column in (x, y):
        midrange = (float(frame[column].max()) + float(frame[column].min())) / 2
        out[column] = [midrange - half, midrange + half]
    return out


#: Panel A's tick step, in nats, on both axes.
PROBABILITIES_TICK_STEP = 2

#: Panel B's tick step on the `r` axis. The domain is rounded up to a multiple
#: of it, so the axis always ends on a labelled tick rather than a bare rule:
#: at `DMS_BAR_HEADROOM` the tallest bar leaves the top of the scale just shy
#: of one, and an axis that stopped at 0.6 would read as if nothing went
#: higher.
PROBABILITIES_BAR_TICK_STEP = 0.2


def _even_ticks(domain: Sequence[float], step: float) -> list[float]:
    """Multiples of ``step`` inside ``domain``, which is what a reader counts.

    Rounded at each tick rather than accumulated, because a fractional step
    otherwise drifts: ten additions of 0.2 do not land on 2.0.
    """
    low, high = float(domain[0]), float(domain[1])
    first = math.ceil(low / step) * step
    count = math.floor((high - first) / step + 1e-9) + 1
    return [round(first + index * step, 6) for index in range(count)]


def probability_against_fitness(
    frame: pd.DataFrame,
    x: str,
    y: str,
    colour: str,
    homolog: str,
    r: float,
) -> alt.Chart:
    """Panel A: the probability as the fitness effect, displaced by the rate.

    ``log p = log f + log lambda - log Z`` with ``log Z`` one number per
    founder, so every point sits on a line of slope 1 through its own
    ``log f``, displaced vertically by its own ``log lambda``. Colouring by
    that rate makes the structure visible: the correlation between the vertical
    residual and ``log lambda`` is exactly 1, so the iso-colour contours *are*
    the slope-1 lines.

    The panel is therefore tautological -- it cannot come out otherwise. It
    earns its place by making an identity legible, not by testing one.

    ``nice=False`` is load-bearing. Without it Vega re-rounds the domains, the
    two spans stop being equal, and the 45-degree reading becomes false.

    Args:
        frame: One founder's cells, carrying ``x``, ``y`` and ``colour``.
        x, y, colour: The fitness, probability and neutral-rate columns.
        homolog: Named in the panel title.
        side: The square's side in pixels.
    """
    domains = probabilities_equal_span_domains(frame, x, y)

    def _axis(title: list[str], domain: list[float]) -> alt.Axis:
        return alt.Axis(
            title=title,
            titleFontSize=DMS_FONT_SIZE,
            labelFontSize=DMS_FONT_SIZE,
            # Every two nats, stated rather than left to Vega's tick count: the
            # two axes carry the same quantity in the same units, so a reader
            # checking the cloud's slope against the diagonal needs the two
            # grids to have the same step. A `tickCount` gives Vega licence to
            # pick a different one per axis as the domains move.
            values=_even_ticks(domain, PROBABILITIES_TICK_STEP),
        )

    points = (
        alt.Chart(frame)
        .mark_circle(size=DMS_POINT_SIZE, opacity=DMS_POINT_OPACITY)
        .encode(
            x=alt.X(
                f"{x}:Q",
                scale=alt.Scale(domain=domains[x], zero=False, nice=False),
                axis=_axis(["DASM fitness effect"], domains[x]),
            ),
            y=alt.Y(
                f"{y}:Q",
                scale=alt.Scale(domain=domains[y], zero=False, nice=False),
                axis=_axis(["log DASM mutation probability"], domains[y]),
            ),
            color=alt.Color(
                f"{colour}:Q",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(
                    title=["log neutral", "mutation rate"],
                    titleFontSize=DMS_FONT_SIZE,
                    labelFontSize=DMS_FONT_SIZE,
                    # Vega labels a gradient's ends only by default, which
                    # gives two numbers and no sense of where the middle of the
                    # scale sits on the cloud.
                    tickCount=4,
                    gradientLength=PROBABILITIES_SCATTER_SIDE * 0.6,
                    offset=4,
                ),
            ),
            tooltip=[
                alt.Tooltip(f"{x}:Q", title="log f", format=".2f"),
                alt.Tooltip(f"{y}:Q", title="log p", format=".2f"),
                alt.Tooltip(f"{colour}:Q", title="log lambda", format=".2f"),
            ],
        )
    )
    # This `r` is the two drawn quantities against each other, not either
    # against an assay -- unlike every other `r` these figures print in this
    # corner. The caption carries that; the panel just says `r`.
    label = _corner_label(
        f"r = {r:{DMS_R_FORMAT}}\nn = {len(frame):,}", font_size=DMS_FONT_SIZE
    )
    return (
        alt.layer(points, label)
        .properties(
            width=PROBABILITIES_SCATTER_SIDE,
            height=PROBABILITIES_SCATTER_SIDE,
            title=alt.TitleParams(
                text=homolog,
                fontSize=DMS_FONT_SIZE,
                fontWeight="bold",
                anchor="middle",
            ),
        )
        .resolve_scale(color="shared")
    )


def dms_correlation_bars(
    frame: pd.DataFrame,
    homologs: Sequence[str],
    arms: Sequence[str],
    bar_height: int,
) -> alt.Chart:
    """Panel B: each arm's agreement with the assay, per dataset.

    Grouped by homolog, three bars a group, every bar carrying its value to
    two significant figures. A reader comparing three arms across three
    datasets should not have to measure against an axis -- and one pair here
    differs by 0.011, which is exactly where eye-matching fails.

    **Anchored at zero, with one domain across all three groups.** Bars encode
    length, so a nonzero baseline misstates every ratio; and a shared domain is
    what makes a bar in one dataset comparable to the bar beside it in another.
    The axis is drawn on the leftmost group only, as `fig:dms`'s bars do, so
    the groups sit a character apart rather than a whole axis apart.

    Args:
        frame: Rows of ``(homolog, predictor, pearson_r, n_mutations)``.
        homologs: Group order, left to right.
        arms: Bar order within a group.
        bar_height: Measured so the panel comes out level with panel A; see
            `notebooks/probabilities.py`.
    """
    headroom = float(frame["pearson_r"].max()) * DMS_BAR_HEADROOM
    top = (
        math.ceil(headroom / PROBABILITIES_BAR_TICK_STEP) * PROBABILITIES_BAR_TICK_STEP
    )
    ticks = _even_ticks([0.0, top], PROBABILITIES_BAR_TICK_STEP)
    order = [PROBABILITIES_ARM_LABELS[arm] for arm in arms]

    def group(homolog: str, first: bool) -> alt.Chart:
        one = frame[frame["homolog"] == homolog].copy()
        one["_arm"] = [PROBABILITIES_ARM_LABELS[a] for a in one["predictor"]]
        one["_value"] = [f"{r:{PROBABILITIES_BAR_FORMAT}}" for r in one["pearson_r"]]
        header = _dms_bar_group_header(homolog, int(one["n_mutations"].iloc[0]))
        base = alt.Chart(one).encode(
            x=alt.X(
                "_arm:N",
                title=None,
                sort=order,
                axis=alt.Axis(labelAngle=-90, labelFontSize=DMS_FONT_SIZE),
            ),
            y=alt.Y(
                "pearson_r:Q",
                title="Pearson r" if first else None,
                scale=alt.Scale(domain=[0.0, top]),
                # **Laid out on every group, drawn on the first.** Setting
                # `labels=False` on the others removes the label boxes from
                # their layout, so the leftmost group alone reserves room for
                # a tick label's overhang and its baseline lands ~20px lower
                # than the rest -- which breaks exactly the eye comparison the
                # shared domain exists to allow. Zero opacity keeps every
                # group's geometry identical and hides the repetition.
                axis=alt.Axis(
                    titleFontSize=DMS_FONT_SIZE,
                    labelFontSize=DMS_FONT_SIZE,
                    values=ticks,
                    labelOpacity=1.0 if first else 0.0,
                    tickOpacity=1.0 if first else 0.0,
                    domainOpacity=1.0 if first else 0.0,
                ),
            ),
        )
        # One flat colour for every bar, as on `fig:dms`: the arms are
        # distinguished by their labels and the group by its heading.
        bars = base.mark_bar(color=PROBABILITIES_BAR_COLOR).encode(
            tooltip=[
                alt.Tooltip("homolog:N", title="homolog"),
                alt.Tooltip("_arm:N", title="arm"),
                alt.Tooltip("pearson_r:Q", title="Pearson r", format=".4f"),
                alt.Tooltip("n_mutations:Q", title="n", format=","),
            ],
        )
        # `dy` sits the value just clear of the bar it belongs to. Small:
        # these panels render at `VEGA_WRITE_PPI`, so a chart-unit offset is
        # multiplied by about four on the written PNG, and -6 left a visible
        # band of white between every bar and its number.
        values = base.mark_text(
            dy=-2, fontSize=DMS_FONT_SIZE, baseline="bottom"
        ).encode(text="_value:N")
        return alt.vconcat(
            header,
            alt.layer(bars, values).properties(width=DMS_BAR_WIDTH, height=bar_height),
            spacing=2,
        )

    return alt.hconcat(
        *[group(homolog, homolog == homologs[0]) for homolog in homologs],
        spacing=DMS_ROW_SPACING,
    )
