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


def scatter_panel(
    frame: pd.DataFrame,
    *,
    metric: str,
    clade: str,
    clip_lower: float,
    stat: str,
    size: int = 225,
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
    """
    axis = alt.Axis(labelFontSize=PANEL_FONT_SIZE, titleFontSize=PANEL_FONT_SIZE)
    high = float(max(frame["log_selection_factor"].max(), frame[metric].max()))
    padding = 0.04 * ((high - clip_lower) or 1)
    scale = alt.Scale(domain=[clip_lower - padding, high + padding], nice=False)
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
    # reader looks for it.
    return (base + diagonal + fit + _corner_label(stat)).properties(
        width=size,
        height=size,
        title=alt.TitleParams(
            # A list is two title lines; a "\n" in a title string is inert in
            # Vega-Lite, which governs text marks rather than titles. Broken
            # before the noun phrase rather than inside it.
            [
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
    return (points + diagonal + fit + _corner_label(stat)).properties(
        width=size, height=size
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


def lookup_panel_sizing(n_clades: int) -> dict[str, int]:
    """Panel A's width and height, from how many clades it has to place."""
    return {
        "width": max(420, 20 * n_clades),
        "height": 350,
    }
