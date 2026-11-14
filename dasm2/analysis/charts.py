"""The one Altair chart builder `fig:agreement` takes from the library.

A fork of the upstream module of the same name, carrying 1 of its 15 builders.
The figure's five panels make five chart calls, and only panel A's is a library
chart: the rest are built inline in the notebooks that lay them out, because a
scatter with a diagonal and a fit is layout rather than a reusable statistic.

`mutation_effect_heatmap` is the builder most obviously missing, and it is
missing on purpose: it draws no panel of this figure, and it needs
`dasm2/neutral/`'s tabular rates model, which this tree prunes.

Altair is in the `dev` dependency group, with `vl-convert-python`, `marimo` and
`snakemake`. Everything that imports this module is run by the pipeline, and
the pipeline only runs where `snakemake` is installed, so the group that has
snakemake is the group that needs altair.
"""

from typing import Optional

try:
    import altair as alt
except ImportError as exc:
    raise ImportError(
        "dasm2.analysis.charts requires altair, which is in this "
        "repository's `dev` dependency group. Install it with "
        "`uv sync --group dev`."
    ) from exc
import pandas as pd


def mutation_lookup_line(
    lookup_df: pd.DataFrame,
    mutation_str: str,
    clade_order: Optional[list[str]] = None,
    *,
    width: int = 500,
    height: int = 350,
    x_title: str = "Clade",
    y_title: str = "Value",
    color_title: str = "Metric",
    series_labels: Optional[dict[str, str]] = None,
    title: Optional[str] = None,
    interval_columns: Optional[tuple[str, str]] = None,
    interval_label: str = "95% interval",
    extra_tooltips: Optional[dict[str, str]] = None,
    series_colors: tuple[str, str] = ("steelblue", "coral"),
    series_shapes: tuple[str, str] = ("circle", "circle"),
) -> alt.LayerChart:
    """Build a line chart tracking a mutation's metrics across clades.

    The two series are the two ways of estimating one mutation's effect, so
    they share a y axis: ``series_labels`` renames them for the legend without
    touching the column names the caller assembled, which keeps the naming a
    presentation choice rather than a change to the data contract.

    Args:
        lookup_df: DataFrame with columns ``clade``, ``delta_fitness``,
            ``log_sf``, and optionally ``expected_count`` and
            ``actual_count``, both of which ride along into the tooltip.
        mutation_str: Mutation label (e.g. ``"D614G"``) for the chart title.
        clade_order: Optional list of clade names for x-axis ordering.
        width: Chart width in pixels.
        height: Chart height in pixels.
        x_title: Axis title for the clade axis. Also titles the clade entry in
            the tooltip, so the hover names the axis it reads off. A caller
            that wants sentence case throughout sets it here rather than
            post-editing the spec.
        y_title: Axis title for the shared value axis. Also titles the value
            entry in the tooltip, so the hover names the same quantity the
            axis does.
        color_title: Legend title for the series.
        series_labels: Optional ``{column: display name}`` for
            ``delta_fitness`` and ``log_sf``. Missing keys keep the column
            name.
        title: Optional chart title, defaulting to
            ``"Mutation <mutation> across clades"``.
        interval_columns: Optional ``(lower, upper)`` column names to shade as
            a grey band behind the series -- an uncertainty region for one of
            the two estimates. Rows missing either bound are skipped. A
            non-finite bound is the **caller's** to replace before calling:
            Vega-Lite extends the y scale to it and takes both series with it,
            which is why `scripts/panels.py` redraws an open-below bound at the
            floor of what the panel already shows.
        interval_label: What the band is, for its tooltip.
        extra_tooltips: Optional ``{column: title}`` for per-clade columns to
            carry through the melt and append to the series tooltip. Absent
            columns are ignored, so a caller can offer a column it only
            sometimes computes.
        series_colors: ``(delta_fitness, log_sf)`` colours, in that order --
            the same order as the legend domain, so the pair cannot be read
            onto the wrong series. Parameterized rather than fixed because a
            notebook that also draws scatters wants one palette across all of
            them.
        series_shapes: ``(delta_fitness, log_sf)`` point shapes, same order.
            Shape is redundant with colour on screen and the only thing left
            when the figure is printed in black and white, so a caller
            preparing a figure distinguishes the series twice. The default
            is a circle for both.

    Returns:
        Layered chart: the optional grey interval band, a two-line overlay
        (delta_fitness, log_sf), and a zero reference line.
    """
    # Melt to long form
    id_vars = ["clade"]
    for optional in ("expected_count", "actual_count"):
        if optional in lookup_df.columns:
            id_vars.append(optional)
    extra = {
        column: title
        for column, title in (extra_tooltips or {}).items()
        if column in lookup_df.columns and column not in id_vars
    }
    id_vars.extend(extra)

    long = lookup_df.melt(
        id_vars=id_vars,
        value_vars=["delta_fitness", "log_sf"],
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])

    labels = {"delta_fitness": "delta_fitness", "log_sf": "log_sf"}
    labels.update(series_labels or {})
    long["metric"] = long["metric"].map(labels)

    sort_order = clade_order if clade_order else sorted(lookup_df["clade"].unique())

    tooltip_fields = [
        alt.Tooltip("clade:N", title=x_title),
        alt.Tooltip("metric:N", title=color_title),
        alt.Tooltip("value:Q", title=y_title, format=".4f"),
    ]
    if "actual_count" in long.columns:
        tooltip_fields.append(
            alt.Tooltip("actual_count:Q", title="Actual Count", format=".0f")
        )
    if "expected_count" in long.columns:
        tooltip_fields.append(
            alt.Tooltip("expected_count:Q", title="Expected Count", format=".1f")
        )
    for column, tip_title in extra.items():
        tooltip_fields.append(alt.Tooltip(f"{column}:Q", title=tip_title, format=".4f"))

    # Drawn first so it sits behind both series. A band is a statement about
    # one of them, not about the pair, so it carries no colour from the series
    # scale: grey, unlabelled by hue, and named in its own tooltip.
    band = None
    if interval_columns is not None:
        lower, upper = interval_columns
        missing = [c for c in interval_columns if c not in lookup_df.columns]
        if missing:
            raise KeyError(
                f"interval_columns {interval_columns!r} not in lookup_df; "
                f"{missing} absent. Columns are {sorted(lookup_df.columns)}."
            )
        band_df = lookup_df[["clade", lower, upper]].dropna()
        if len(band_df):
            band = (
                alt.Chart(band_df)
                .mark_area(color="#9e9e9e", opacity=0.25)
                .encode(
                    x=alt.X("clade:N", title=x_title, sort=sort_order),
                    y=alt.Y(f"{lower}:Q", title=y_title),
                    y2=alt.Y2(f"{upper}:Q"),
                    tooltip=[
                        alt.Tooltip("clade:N", title=x_title),
                        alt.Tooltip(
                            f"{lower}:Q", title=f"{interval_label} low", format=".4f"
                        ),
                        alt.Tooltip(
                            f"{upper}:Q", title=f"{interval_label} high", format=".4f"
                        ),
                    ],
                )
            )

    line_chart = (
        alt.Chart(long)
        .mark_line(point=alt.OverlayMarkDef(size=80))
        .encode(
            x=alt.X("clade:N", title=x_title, sort=sort_order),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(
                    domain=[labels["delta_fitness"], labels["log_sf"]],
                    range=list(series_colors),
                ),
                legend=alt.Legend(title=color_title),
            ),
            # Both channels on one field, so Vega-Lite merges them into a
            # single legend whose swatches carry the colour and the shape
            # together rather than listing the series twice.
            shape=alt.Shape(
                "metric:N",
                scale=alt.Scale(
                    domain=[labels["delta_fitness"], labels["log_sf"]],
                    range=list(series_shapes),
                ),
                legend=alt.Legend(title=color_title),
            ),
            tooltip=tooltip_fields,
        )
    )

    zero_line = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(strokeDash=[3, 3], color="gray", opacity=0.5)
        .encode(y="y:Q")
    )

    # Size and title on the *outer* layer rather than on one child: with the
    # band present the line is no longer the first child, and a title set on a
    # child that is not first has no defined place to render.
    layers = ([band] if band is not None else []) + [line_chart, zero_line]
    # `alt.layer` is typed, unlike the `a + b` overloads used elsewhere here, so
    # its `.properties` is declared to return `LayerChart | FacetChart`.
    return alt.layer(*layers).properties(  # type: ignore[no-any-return, return-value]
        width=width,
        height=height,
        title=title or f"Mutation {mutation_str} across clades",
    )
