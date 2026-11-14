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

from lib import assert_dasm2_is_this_repo

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import altair as alt
import pandas as pd
from panels import figure_dir

# Re-exported, like the statistics through `scripts/panels.py`: a notebook then
# imports its charts from one module and never names `dasm2` itself, which is
# what keeps `tests/test_import_order.py`'s two orderings a property of this
# layer rather than of every notebook that draws a panel.
from dasm2.analysis.charts import mutation_lookup_line

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
    stat: str,
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
    layered = base + diagonal + fit + _corner_label(stat)
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


#: One sub-panel of a background row, in pixels. Narrow and tall on purpose:
#: a row carries up to seven of them side by side, and the panel's subject is
#: the *slope* between two x positions, which a wide short cell flattens --
#: at a square aspect a 2-nat shift and a 5-nat one look alike.
BACKGROUND_CELL_W = 125
BACKGROUND_CELL_H = 320

#: Headroom above the drawn range, as a fraction of it, for the p-value that
#: sits at the top of each sub-panel.
BACKGROUND_HEADROOM = 0.18

#: The two approaches, their colours and their marks, in the order
#: `mutation_lookup_line` takes them -- ``delta_fitness`` first, ``log_sf``
#: second. One spelling, because both the sub-panels and the legend beside
#: them are built from it: a legend drawn from its own copy of the mapping is
#: a legend that can come to disagree with the marks it names.
BACKGROUND_SERIES = (
    ("counts", COUNTS_GREY, "circle"),
    ("DASM", DASM_PURPLE, "triangle"),
)


def background_panel(
    frame: pd.DataFrame,
    clade: str,
    *,
    y_domain: tuple[float, float] | None = None,
) -> alt.HConcatChart:
    """One clade's row of `fig:within_clade_shifts`: panel B, D or F.

    One sub-panel per candidate mutation, each plotting that mutation's
    fitness effect on the clade's two backgrounds by both approaches. The x
    axis is the two backgrounds named by their residues -- ``R346, K356,
    F490`` against ``T346, T356, S490`` -- so the two ticks are a
    like-for-like pair rather than "none" against a list of substitutions.

    Assembled from :func:`dasm2.analysis.charts.mutation_lookup_line` rather
    than drawn here. That builder already puts two series over a categorical
    axis with the counts/DASM colours and shapes this figure uses elsewhere,
    and reusing it is what makes the two approaches look the same in this
    figure as in `fig:agreement`.

    **One y domain across every sub-panel of every row.** Passed in rather
    than computed per clade: the three rows are read against each other, so a
    steep line has to mean a large shift and not a narrow axis.

    Args:
        frame: Columns ``candidate``, ``background``, ``counts_effect``,
            ``dasm_effect``, ``p_label`` and ``background_order``.
        clade: The clade this row draws, for the y-axis title.
        y_domain: ``(low, high)`` shared by every sub-panel.

    Returns:
        The row, left to right in order of decreasing absolute counts shift.
    """
    required = {
        "candidate",
        "background",
        "counts_effect",
        "dasm_effect",
        "p_label",
        "background_order",
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

    candidates = sorted(frame["candidate"].unique(), key=shift_of, reverse=True)

    cells = []
    for index, candidate in enumerate(candidates):
        part = frame[frame["candidate"] == candidate]
        lookup = pd.DataFrame(
            {
                "clade": part["background"],
                "delta_fitness": part["counts_effect"],
                "log_sf": part["dasm_effect"],
            }
        )
        cell = mutation_lookup_line(
            lookup,
            candidate,
            clade_order=order,
            width=BACKGROUND_CELL_W,
            height=BACKGROUND_CELL_H,
            # The sub-panel's axes are titled once for the row, below, so a
            # seven-cell row does not repeat one label seven times.
            x_title="",
            y_title=(f"fitness effect in clade {clade} subset" if index == 0 else ""),
            color_title="approach",
            series_labels=dict(
                zip(("delta_fitness", "log_sf"), [s[0] for s in BACKGROUND_SERIES])
            ),
            series_colors=tuple(s[1] for s in BACKGROUND_SERIES),
            series_shapes=tuple(s[2] for s in BACKGROUND_SERIES),
            title=candidate,
            y_domain=y_domain,
            # No per-cell legend. Vega-Lite drops a legend declared on a
            # layer nested inside an hconcat -- which is what every cell is,
            # since the p-value is layered over the lines -- so the row's key
            # is drawn as its own cell by `_background_legend` instead.
            show_legend=False,
            # One value axis, on the first cell. Every cell shares
            # `y_domain`, so the rest would repeat the same ticks.
            show_y_axis=index == 0,
        )
        label = part["p_label"].iloc[0]
        annotation = (
            alt.Chart(pd.DataFrame({"text": [label]}))
            .mark_text(
                align="center",
                baseline="top",
                dy=4,
                fontSize=PANEL_FONT_SIZE - 4,
                color="#444444",
            )
            .encode(x=alt.value(BACKGROUND_CELL_W / 2), y=alt.value(0), text="text:N")
        )
        # No `.properties` on the layer: `mutation_lookup_line` has already
        # set the size and the title, and restating them on the wrapper is
        # what drops the legend off the cell that is supposed to carry it.
        cells.append(alt.layer(cell, annotation))

    return alt.hconcat(*cells, _background_legend()).properties(
        title=alt.TitleParams(
            "amino-acid identities defining each subset",
            subtitle="left: clade-founder identities; right: derived identities",
            fontSize=PANEL_FONT_SIZE - 2,
            subtitleFontSize=PANEL_FONT_SIZE - 5,
            anchor="middle",
            orient="bottom",
            offset=6,
        )
    )


def _background_legend() -> alt.LayerChart:
    """The row's key, as its own cell.

    Drawn by hand rather than declared on one of the sub-panels: each
    sub-panel is a layer (the two lines, plus the p-value over them) nested
    inside an hconcat, and Vega-Lite drops a legend declared there -- the
    spec keeps it and the rendered SVG carries no `role-legend` group at all.
    A hand-drawn key renders wherever it is put.

    Built from :data:`BACKGROUND_SERIES`, the same tuple the marks are built
    from, so the swatch and the mark it names cannot drift apart.

    Positioned on a unit scale with the axis off, rather than on a band
    scale: a band scale spreads two entries over the full height of a cell
    320 pixels tall, which reads as two unrelated labels rather than as one
    key. ``stroke=None`` drops the cell's frame, which a key has no use for
    and which makes it look like a panel with nothing plotted in it.
    """
    names = [name for name, _, _ in BACKGROUND_SERIES]
    frame = pd.DataFrame(
        {
            "approach": names,
            # Just above and just below the middle of the cell, so the key
            # reads as a block against the panels beside it.
            "position": [0.54, 0.46][: len(names)],
        }
    )
    shared = {
        "y": alt.Y("position:Q", scale=alt.Scale(domain=[0, 1]), axis=None, title=None)
    }
    marks = (
        alt.Chart(frame)
        .mark_point(filled=True, size=110, opacity=1)
        .encode(
            x=alt.value(10),
            color=alt.Color(
                "approach:N",
                scale=alt.Scale(
                    domain=names, range=[colour for _, colour, _ in BACKGROUND_SERIES]
                ),
                legend=None,
            ),
            shape=alt.Shape(
                "approach:N",
                scale=alt.Scale(
                    domain=names, range=[shape for _, _, shape in BACKGROUND_SERIES]
                ),
                legend=None,
            ),
            **shared,
        )
    )
    labels = (
        alt.Chart(frame)
        .mark_text(align="left", fontSize=PANEL_FONT_SIZE - 4, color="#222222")
        .encode(x=alt.value(24), text="approach:N", **shared)
    )
    return (
        alt.layer(marks, labels)
        .properties(
            width=76,
            height=BACKGROUND_CELL_H,
            view=alt.ViewConfig(stroke=None),
            title=alt.TitleParams(
                "approach", fontSize=PANEL_FONT_SIZE - 3, anchor="start", dx=6
            ),
        )
        .resolve_scale(color="independent", shape="independent")
    )
