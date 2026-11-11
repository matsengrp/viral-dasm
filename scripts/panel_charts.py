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
from collections.abc import Sequence
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

#: Panel A's geometry. A cap is drawn as a rule across the whole panel rather
#: than as a per-clade mark, because what it does is set one height for every
#: clade above it.
CLADE_SIZE_WIDTH = 760
CLADE_SIZE_HEIGHT = 230
ECDF_PANEL_WIDTH = 300
ECDF_PANEL_HEIGHT = 230

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
ARROW_HEAD_PX = 15.0
ARROW_TAIL_PX = 29.0
ARROW_HEAD_SIZE = 110


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
    """
    return "full depth" if arm == FULL_DEPTH.name else f"cap {cap_of(arm):,}"


def clade_size_panel(
    frame: pd.DataFrame,
    caps: Sequence[int],
    *,
    title: list[str],
    highlight: Sequence[str] = (),
    width: int = CLADE_SIZE_WIDTH,
    height: int = CLADE_SIZE_HEIGHT,
) -> alt.LayerChart:
    """Panel A: sequences per clade, with one rule per downsampling cap.

    A dot per clade on a log y axis, and an arrow over each clade in
    ``highlight`` -- panel B's two, so a reader arriving from its titles finds
    them here without counting along 56 ticks.

    **A log y axis**, because the clades span four orders of magnitude. On a
    linear axis every clade but the largest five is a flat line against the
    axis, which is the opposite of the panel's point: the imbalance is the
    finding, and a reader has to be able to see where a cap falls among the
    clades it does *not* touch as well as the ones it cuts.

    The clades are ordered by size rather than by name, so each cap rule meets
    the curve of dots at one place and the reader can count what it reaches.
    """
    order = list(frame["clade"])
    x_enc = alt.X(
        "clade:N",
        sort=order,
        title="Nextstrain clade",
        axis=alt.Axis(
            labelFontSize=PANEL_FONT_SIZE - 6,
            titleFontSize=PANEL_FONT_SIZE,
            labelAngle=-90,
        ),
    )
    # One scale object and one field name for both layers. A layered chart
    # shares a channel's scale, so a rule encoded on its own field, or on a
    # linear scale beside a log one, resolves to a domain that fits neither
    # layer and draws both somewhere other than where the code reads.
    #
    # The domain spans the caps as well as the clades: a rule is not clipped to
    # the plot, so a cap outside the clade range would be drawn outside the
    # frame or not at all, with nothing raising.
    check_cap_colours(caps)
    bounds = [int(value) for value in frame["n_sequences"]] + [int(cap) for cap in caps]
    floor = max(1.0, min(bounds) * DOMAIN_FLOOR_PAD)
    ceiling = max(bounds) * DOMAIN_TOP_PAD
    y_scale = alt.Scale(type="log", domain=[floor, ceiling], nice=False)
    y_axis = alt.Axis(
        labelFontSize=PANEL_FONT_SIZE,
        titleFontSize=PANEL_FONT_SIZE,
        format="~s",
    )

    def y_of(field: str) -> alt.Y:
        """One field on panel A's one y scale.

        Every layer repeats the scale, the title and the axis. The arrow sits
        at its own field rather than at the clade's count, and a layer that
        named a field without restating the domain would make Vega resolve one
        scale across two of them.
        """
        return alt.Y(f"{field}:Q", title="sequences", scale=y_scale, axis=y_axis)

    y_enc = y_of("n_sequences")
    # A dot per clade, not a bar. A bar extends from its scale's zero and a log
    # scale has none, so a bar mark here draws nothing at all while the chart
    # still builds, saves, and satisfies every assertion about it.
    dots = (
        alt.Chart(frame)
        .mark_point(color=DEPTH_INK, filled=True, size=70, opacity=0.9)
        .encode(
            x=x_enc,
            y=y_enc,
            tooltip=[
                alt.Tooltip("clade:N", title="Clade"),
                alt.Tooltip("n_sequences:Q", title="Sequences", format=","),
            ],
        )
    )
    cap_frame = pd.DataFrame(
        {
            "n_sequences": [float(cap) for cap in caps],
            "label": [f"cap {cap:,}" for cap in caps],
        }
    )

    rules = (
        alt.Chart(cap_frame)
        .mark_rule(strokeDash=[6, 4], strokeWidth=1.8)
        .encode(
            y=y_enc,
            color=alt.Color(
                "label:N",
                title="downsampling cap",
                sort=cap_frame["label"].tolist(),
                scale=alt.Scale(
                    domain=cap_frame["label"].tolist(), range=list(CAP_BLUES)
                ),
                legend=alt.Legend(
                    labelFontSize=PANEL_FONT_SIZE, titleFontSize=PANEL_FONT_SIZE
                ),
            ),
            tooltip=[alt.Tooltip("label:N", title="Cap")],
        )
    )
    layers = [dots, rules]
    marked = frame[frame["clade"].isin(list(highlight))]
    if len(marked):
        # Pixels per decade, which is what lets the arrow be drawn to a pixel
        # spec on a log axis.
        per_decade = height / math.log10(ceiling / floor)
        marked = marked.assign(
            arrow_head=_raised(
                marked["n_sequences"], ARROW_HEAD_PX, per_decade=per_decade
            ),
            arrow_tail=_raised(
                marked["n_sequences"], ARROW_HEAD_PX, per_decade=per_decade
            ),
            arrow_end=_raised(
                marked["n_sequences"], ARROW_TAIL_PX, per_decade=per_decade
            ),
        )
        layers.append(
            alt.Chart(marked)
            .mark_rule(color=DEPTH_INK, strokeWidth=1.8)
            .encode(x=x_enc, y=y_of("arrow_tail"), y2="arrow_end:Q")
        )
        layers.append(
            alt.Chart(marked)
            .mark_point(
                shape="triangle-down",
                filled=True,
                size=ARROW_HEAD_SIZE,
                color=DEPTH_INK,
                opacity=1,
            )
            .encode(x=x_enc, y=y_of("arrow_head"))
        )
    return alt.layer(*layers).properties(
        width=width,
        height=height,
        title=alt.TitleParams(
            title, fontSize=PANEL_FONT_SIZE, offset=4, anchor="middle"
        ),
    )


def count_ecdf_panel(
    frame: pd.DataFrame,
    *,
    clade_titles: dict[str, str],
    threshold: float,
    width: int = ECDF_PANEL_WIDTH,
    height: int = ECDF_PANEL_HEIGHT,
) -> alt.HConcatChart:
    """Panel B: one complementary eCDF per clade, one curve per sampling depth.

    Each curve is the share of a clade's mutations carrying **at least** *x*
    counts, so it falls from 1.0 rather than rising to it. A labelled rule
    marks the accuracy threshold, and the curve's height where it crosses is
    the share that depth still puts at or above the threshold.

    Concatenated, so each view carries its clade's name in the paper's own
    spelling -- `21J (Delta)`, `25C (XFG)` -- where a facet header would show
    the bare label this pipeline keys on.
    """
    arms = list(frame["arm"].unique())
    caps = [arm for arm in arms if arm != FULL_DEPTH.name]
    check_cap_colours(caps)
    domain = [depth_label(FULL_DEPTH.name)] + [depth_label(arm) for arm in caps]
    labelled = frame.assign(depth=[depth_label(arm) for arm in frame["arm"]])
    threshold_frame = pd.DataFrame({"x": [threshold]})

    # One x scale for both views and for the rule. `hconcat` leaves position
    # scales independent, so two clades whose grids ended at different counts
    # would be drawn over different ranges side by side; and a rule with no
    # scale of its own resolves against whichever field shares its name.
    x_max = float(frame["x"].max())
    if threshold > x_max:
        raise ValueError(
            f"the accuracy threshold {threshold:g} is past the end of the "
            f"count grid ({x_max:g}), so the rule a reader takes the share off "
            f"would be drawn outside the plot."
        )
    x_scale = alt.Scale(domain=[0, x_max], nice=False)
    x_axis = alt.Axis(
        labelFontSize=PANEL_FONT_SIZE,
        titleFontSize=PANEL_FONT_SIZE,
        # The threshold among the ticks: it is the one count on this axis a
        # reader reads a number off, and the default run skips it.
        values=sorted({0, int(threshold), *range(4, int(x_max) + 1, 4)}),
    )

    panels = []
    for index, (clade, clade_title) in enumerate(clade_titles.items()):
        rows = labelled[labelled["clade"] == clade]
        curves = (
            alt.Chart(rows)
            .mark_line(strokeWidth=2.2)
            .encode(
                x=alt.X("x:Q", title=LADDER_X_TITLE, scale=x_scale, axis=x_axis),
                y=alt.Y(
                    "share:Q",
                    # Two lines: a one-line title is taller than the plot it
                    # labels at this panel size, and in an hconcat that pushes
                    # the left view's plot down while its title stays at the top
                    # of its box, leaving the two titles at different heights.
                    title=(
                        ["fraction of mutations", "at or above that count"]
                        if index == 0
                        else None
                    ),
                    scale=alt.Scale(domain=[0, 1], nice=False),
                    axis=alt.Axis(
                        labelFontSize=PANEL_FONT_SIZE,
                        titleFontSize=PANEL_FONT_SIZE,
                        labels=index == 0,
                    ),
                ),
                color=alt.Color(
                    "depth:N",
                    title="sampling depth",
                    sort=domain,
                    scale=alt.Scale(domain=domain, range=[DEPTH_INK] + list(CAP_BLUES)),
                    legend=alt.Legend(
                        labelFontSize=PANEL_FONT_SIZE, titleFontSize=PANEL_FONT_SIZE
                    ),
                ),
                tooltip=[
                    alt.Tooltip("depth:N", title="Depth"),
                    alt.Tooltip("x:Q", title="Count", format=".1f"),
                    alt.Tooltip("share:Q", title="Fraction", format=".4f"),
                ],
            )
        )
        rule = (
            alt.Chart(threshold_frame)
            .mark_rule(color=THRESHOLD_ACCENT, strokeDash=[2, 3], strokeWidth=1.6)
            .encode(x=alt.X("x:Q", scale=x_scale, axis=x_axis))
        )
        # The rule says nothing about itself otherwise: it takes no legend
        # entry, so a reader who has not read the caption sees an unexplained
        # dotted line across the one panel they read a number off.
        rule_label = (
            alt.Chart(threshold_frame)
            .mark_text(
                color=THRESHOLD_ACCENT,
                fontSize=PANEL_FONT_SIZE - 4,
                align="left",
                baseline="top",
                dx=4,
            )
            .encode(
                x=alt.X("x:Q", scale=x_scale, axis=x_axis),
                y=alt.value(0.0),
                text=alt.value(f"threshold {threshold:g}"),
            )
        )
        panels.append(
            (curves + rule + rule_label).properties(
                width=width,
                height=height,
                title=alt.TitleParams(
                    [f"clade {clade_title}"],
                    fontSize=PANEL_FONT_SIZE,
                    offset=4,
                    anchor="middle",
                ),
            )
        )
    return alt.hconcat(*panels)
