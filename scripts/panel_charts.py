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

from lib import FULL_DEPTH, assert_dasm2_is_this_repo, cap_of

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import math
from collections.abc import Sequence

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


# ---------------------------------------------------------------------------
# `fig:dms`
# ---------------------------------------------------------------------------

#: The three predictors panel B sets against each other, in the order it draws
#: them, with the colour each carries.
#:
#: `DASM_PURPLE` and `COUNTS_GREY` are `fig:agreement`'s own two approaches and
#: mean the same thing here. The pooled estimate takes a third hue rather than
#: a lighter grey: it is a different predictor, not a weaker version of the
#: per-clade one, and a reader should not have to notice a shade to see that.
DMS_PREDICTOR_COLORS: dict[str, str] = {
    "DASM": DASM_PURPLE,
    "per-clade counts": COUNTS_GREY,
    "pooled counts": "#b0975f",
}

#: What panel A's axes measure. The DMS is the measurement and the model is the
#: prediction, so the measurement goes on y -- the convention a reader brings to
#: a predicted-versus-measured plot.
DMS_X_TITLE = "DASM fitness effect (log SF)"
DMS_Y_TITLE = "measured effect on cell entry"

#: Panel B's axis titles. Two lines, as the ladder panels' are, because a long
#: single-line axis title sets the panel's width.
DMS_BAR_Y_TITLE = ["Pearson correlation", "with cell entry (r)"]

#: Panel sizes. The scatter is square because its axes are not comparable and
#: nothing is read off a diagonal, so the aspect carries no claim; the bars are
#: wider than tall because the categorical axis has nine positions.
DMS_SCATTER_SIZE = 260
DMS_BAR_WIDTH = 150
DMS_BAR_HEIGHT = 220

#: How many equal-count bins the binned median in panel A is taken over. The
#: cloud saturates where the assay's dynamic range ends, and a trend through
#: the saturation is what shows the shape a correlation coefficient hides.
DMS_TREND_BINS = 20

#: The columns panel A draws, so the chart embeds those and not the frame.
#: `scripts/panels.py` hands over a frame with the join's full width, and
#: Altair serializes whatever it is given.
DMS_PLOT_COLUMNS = ["log_selection_factor", "dms_cell_entry"]

#: Points drawn per stratum cell. **Altair refuses a dataset over 5,000 rows**,
#: and the refusal is asymmetric in a way that hides it: `chart.save()` goes
#: through `vl-convert` and writes the PNG regardless, while `chart.to_dict()`
#: raises `MaxRowsError` -- so a rule writes the panel and the notebook cell
#: that displays it fails. The real strata are larger than that ceiling (5,432
#: single-nucleotide mutations on KP.3.1.1 alone), so the cloud is thinned.
#:
#: Drawing only. Every statistic in the corner label is computed upstream on
#: the full data, as `shift_panel`'s are, so thinning a dense cell changes what
#: is *drawn* and never what is reported.
DMS_POINT_BUDGET = 4_000

#: The sampling seed, fixed so a rebuild draws the same cloud. Shared with
#: `shift_grid`'s thinning, which uses the same value for the same reason.
DMS_SAMPLE_SEED = 0


def _padded_domain(values: pd.Series, pad_fraction: float = 0.04) -> list[float]:
    """``[min, max]`` of ``values``, padded so a mark on a bound is not clipped.

    A point is a disc of some radius, so one sitting exactly on a bound is
    drawn half outside the plot -- the reason `scatter_panel` pads its own
    domain the same way.
    """
    low = float(values.min())
    high = float(values.max())
    padding = pad_fraction * ((high - low) or 1.0)
    return [low - padding, high + padding]


def binned_median(
    frame: pd.DataFrame, *, x: str, y: str, bins: int = DMS_TREND_BINS
) -> pd.DataFrame:
    """The median of ``y`` in equal-count bins of ``x``.

    Equal-count rather than equal-width: the predictor's distribution has a
    long deleterious tail, and equal-width bins would put most of the data in
    two of them and draw a trend from single points everywhere else.

    Returns a frame of ``bins`` rows at most, so it costs the same whatever
    ``n`` is -- which is what makes it affordable in a notebook cell beside the
    cloud itself.
    """
    if frame.empty:
        return pd.DataFrame({x: [], y: []})
    ranked = frame[[x, y]].dropna().sort_values(x)
    if ranked.empty:
        return pd.DataFrame({x: [], y: []})
    labels = pd.qcut(
        ranked[x].rank(method="first"), q=min(bins, len(ranked)), labels=False
    )
    return (
        ranked.assign(_bin=labels)
        .groupby("_bin", as_index=False)
        .agg({x: "median", y: "median"})
        .drop(columns="_bin", errors="ignore")
    )


#: What each stratum is called on a chart, long form and short.
#:
#: Spelled here rather than in `scripts/dms.py` because the strings in that
#: module are join keys and CSV values: changing one to read better on a chart
#: would change what a panel's table says.
#:
#: Two forms because the two panels have different room. Panel A gives each
#: stratum a 260 px cell and a two-line title; panel B gives it a facet header
#: over a 150 px column, where the long form is wider than the column and
#: Vega-Lite neither wraps nor elides a header label -- adjacent headers simply
#: overlap, and the row becomes unreadable without anything raising.
STRATUM_LABELS: dict[str, tuple[str, str]] = {
    "single_nt": ("single-nt codon mutations", "single-nt"),
    "multi_nt": ("multi-nt codon mutations", "multi-nt"),
    "all": ("all measured mutations", "all"),
}


def stratum_label(stratum: str, *, short: bool = False) -> str:
    """One stratum's name, as the figure prints it.

    Raises:
        KeyError: On a stratum with no label. Returning the raw key instead
            would put ``single_nt`` on a paper figure, which is the one failure
            here that looks deliberate.
    """
    if stratum not in STRATUM_LABELS:
        raise KeyError(
            f"{stratum!r} has no label; expected one of {sorted(STRATUM_LABELS)}"
        )
    long_form, short_form = STRATUM_LABELS[stratum]
    return short_form if short else long_form


def dms_scatter_panel(
    frame: pd.DataFrame,
    *,
    homolog: str,
    stratum_order: Sequence[str],
    stats: pd.DataFrame | None = None,
    size: int = DMS_SCATTER_SIZE,
) -> alt.VConcatChart | alt.HConcatChart:
    """Panel A: one homolog's measured effect against the DASM's, per stratum.

    **No diagonal, and no shared domain.** `fig:agreement`'s scatter compares
    two estimates of the same quantity, so a point's distance from ``y = x`` is
    the disagreement. Here the axes are a log selection factor and a cell-entry
    measurement: different units on different scales, with no reason for a
    point to sit anywhere in particular. Drawing a 45-degree line would invite
    a reading the data does not support, so what is drawn instead is the
    regression and the binned median -- which say how the prediction *orders*
    the measurement, which is what the correlation reports.

    Built as one cell per stratum rather than through ``.facet()``, as
    ``shift_grid`` is, so each panel carries its own ``r`` and ``n``: the whole
    point of the split is that the strata differ, and a shared corner label
    would hide it.
    """
    axis = alt.Axis(labelFontSize=PANEL_FONT_SIZE, titleFontSize=PANEL_FONT_SIZE)
    # One domain per axis across the whole row, computed from the data and
    # padded. Two things need it. The y scale is shared, so a cell left to
    # infer its own bounds would widen the row; and the regression is an
    # unbounded line, so without an explicit domain Vega sizes the axis to
    # where the *fit* reaches rather than to where the assay measured -- which
    # on these data means an axis running up to +3.5 cell entry against a
    # measured ceiling of +0.3, and a third of the panel empty.
    x_domain = _padded_domain(frame["log_selection_factor"])
    y_domain = _padded_domain(frame["dms_cell_entry"])
    x_scale = alt.Scale(domain=x_domain, nice=False)
    y_scale = alt.Scale(domain=y_domain, nice=False)
    cells = []
    for stratum in stratum_order:
        cell_frame = frame[frame["stratum"] == stratum]
        row = None
        if stats is not None:
            matching = stats[stats["stratum"] == stratum]
            row = None if matching.empty else matching.iloc[0]
        drawn = cell_frame[DMS_PLOT_COLUMNS]
        if len(drawn) > DMS_POINT_BUDGET:
            drawn = drawn.sample(DMS_POINT_BUDGET, random_state=DMS_SAMPLE_SEED)
        x_enc = alt.X(
            "log_selection_factor:Q",
            title=DMS_X_TITLE,
            axis=axis,
            scale=x_scale,
        )
        # The y scale is shared across the row, so only the leftmost cell
        # carries the axis. Repeating it would print the same title and the
        # same ticks beside every cell, which on the composed page is the
        # figure's longest string drawn once per stratum.
        y_axis = (
            axis
            if not cells
            else alt.Axis(labels=False, ticks=False, domain=True, title=None)
        )
        y_enc = alt.Y(
            "dms_cell_entry:Q",
            title=DMS_Y_TITLE if not cells else None,
            axis=y_axis,
            scale=y_scale,
        )
        cloud = (
            alt.Chart(drawn)
            .mark_point(size=56, opacity=0.3, filled=True, color=POINT_BLUE)
            .encode(x=x_enc, y=y_enc)
        )
        # `clip=True` on both lines: a layered mark is not clipped to the plot
        # area by default, so with an explicit domain the regression would run
        # out over the axis and the neighbouring cell instead of stopping.
        fit = cloud.transform_regression(
            "log_selection_factor", "dms_cell_entry"
        ).mark_line(strokeWidth=2, color=FIT_FIREBRICK, clip=True)
        # The median is computed on the *full* cell, not the thinned draw, and
        # is 20 rows whatever `n` is. Solid rather than dashed: it has to read
        # over a saturated cloud, and `ladder_row`'s median is solid for the
        # same reason.
        trend_frame = binned_median(
            cell_frame, x="log_selection_factor", y="dms_cell_entry"
        )
        trend = (
            alt.Chart(trend_frame)
            .mark_line(
                strokeWidth=2.2,
                color=LADDER_MEDIAN_COLOR,
                point=alt.OverlayMarkDef(
                    color=LADDER_MEDIAN_COLOR, stroke="white", strokeWidth=1, size=45
                ),
                clip=True,
            )
            .encode(x=x_enc, y=y_enc)
        )
        cells.append(
            (
                cloud + fit + trend + _corner_label(stat_label(row, len(cell_frame)))
            ).properties(
                width=size,
                height=size,
                title=alt.TitleParams(
                    [homolog, stratum_label(stratum)],
                    fontSize=PANEL_FONT_SIZE,
                    offset=3,
                    anchor="middle",
                ),
            )
        )
    # Both scales shared. Every cell is given the same explicit domain above,
    # so this states what is already true rather than changing it -- and it is
    # what makes the strata comparable: the split is the point of the panel,
    # and two cells on different x ranges put the fit and the median at slopes
    # a reader cannot set against each other.
    return alt.hconcat(*cells).resolve_scale(x="shared", y="shared")


def homolog_correlation_panel(
    frame: pd.DataFrame,
    *,
    homolog_order: Sequence[str],
    stratum_order: Sequence[str],
    predictor_order: Sequence[str],
    width: int = DMS_BAR_WIDTH,
    height: int = DMS_BAR_HEIGHT,
) -> alt.Chart:
    """Panel B: every predictor's correlation, every homolog, every stratum.

    Bars rather than dots, and from zero rather than from the data's floor. A
    correlation has a meaningful zero -- no ordering at all -- so a bar's
    length is the quantity, which is the one case in this figure where a bar
    is the honest mark.

    **One row of grouped bars, not a grid.** Homolog is the x axis and
    predictor the group within it, so the whole panel carries one y axis and
    one legend. Faceting on homolog as well as stratum draws nine cells, nine
    y axes and the axis title nine times, and the comparison a reader wants --
    the three predictors at one homolog -- then spans three cells instead of
    sitting side by side.

    **No ``LADDER_R_FLOOR`` here.** The ladder panels clip negative r to zero,
    because there the x axis is a count cutoff and a negative r is a rung with
    too little data rather than a result. Here a negative r is a real statement
    about a predictor, so it is drawn.

    Raises:
        ValueError: If ``frame`` names a predictor with no colour, which Vega
            would otherwise recycle from the start of the range and give two
            predictors one hue; or if a homolog or stratum in ``frame`` is
            absent from the order it is sorted by, which Vega drops silently.
    """
    missing_r = frame["pearson_r"].isna()
    if missing_r.any():
        rows = frame.loc[missing_r, ["homolog", "stratum", "predictor"]]
        raise ValueError(
            f"{int(missing_r.sum())} rows carry no `pearson_r`, which draws no "
            f"bar and leaves a gap the legend still promises:\n"
            f"{rows.to_string(index=False)}\n"
            f"`within_clade_metrics` returns NaN below its `min_mutations`, so "
            f"decide whether those cells are reported as absent or drawn as "
            f"`n/a` rather than left to render as nothing."
        )
    unknown = sorted(set(frame["predictor"]) - set(DMS_PREDICTOR_COLORS))
    if unknown:
        raise ValueError(
            f"{unknown} have no entry in DMS_PREDICTOR_COLORS, whose keys are "
            f"{sorted(DMS_PREDICTOR_COLORS)}. Vega recycles a colour range "
            f"shorter than its domain without raising, so two predictors would "
            f"be drawn in one hue."
        )
    for column, order in (
        ("homolog", homolog_order),
        ("stratum", stratum_order),
        ("predictor", predictor_order),
    ):
        missing = sorted(set(frame[column]) - set(order))
        if missing:
            raise ValueError(
                f"{column}={missing} is in the frame but not in the order it "
                f"is sorted by, {list(order)}. Vega drops a value absent from "
                f"an explicit sort without raising, so those bars would not be "
                f"drawn and the panel would look complete."
            )
    present = set(frame["predictor"])
    predictors = [name for name in predictor_order if name in present]
    axis = alt.Axis(labelFontSize=PANEL_FONT_SIZE, titleFontSize=PANEL_FONT_SIZE)
    labelled = frame.assign(
        stratum_label=frame["stratum"].map(
            lambda name: stratum_label(name, short=True)
        ),
    )
    return (
        alt.Chart(labelled)
        .mark_bar()
        .encode(
            x=alt.X(
                "homolog:N",
                title=None,
                sort=list(homolog_order),
                axis=alt.Axis(labelFontSize=PANEL_FONT_SIZE, labelAngle=-40),
            ),
            xOffset=alt.XOffset("predictor:N", sort=predictors),
            y=alt.Y("pearson_r:Q", title=DMS_BAR_Y_TITLE, axis=axis),
            color=alt.Color(
                "predictor:N",
                sort=predictors,
                scale=alt.Scale(
                    domain=predictors,
                    range=[DMS_PREDICTOR_COLORS[name] for name in predictors],
                ),
                legend=alt.Legend(
                    title=None,
                    labelFontSize=PANEL_FONT_SIZE,
                    orient="top",
                    direction="horizontal",
                ),
            ),
            column=alt.Column(
                "stratum_label:N",
                title=None,
                sort=[stratum_label(name, short=True) for name in stratum_order],
                header=alt.Header(labelFontSize=PANEL_FONT_SIZE),
            ),
        )
        .properties(width=width, height=height)
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
