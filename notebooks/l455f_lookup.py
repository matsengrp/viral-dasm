import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # `fig:l455f` panel B — L455F across the clades

    *One mutation's fitness effect in every clade that carries an estimate of
    it, by both approaches, against the one estimate that pools the clades.*

    The same chart as `fig:agreement`'s panel A at a different mutation, drawn
    from the same `selection_factors.parquet`. No inference and no GPU: `rule
    score` did the forward passes.

    **Why L455F and not F456L.** `fig:agreement`'s panel A is an example of a
    mutation whose effect shifts between clades. This figure is about one
    specific shift and what causes it, so the mutation it draws is the one
    panel A above has already split by background.

    **The dashed rule is the counts estimate with the clades pooled.** Not an
    average of the points above it: it is mat-pcp's own clade-pooled table,
    `actual` and `expected` summed over branches and divided once. It is drawn
    in the counts colour because it is a counts quantity, and under the series
    so it cannot hide a point.

    **What "pooled" does and does not cover.** The sum runs over the clades
    whose founder carries the residue L455F mutates away from. Every clade
    whose founder already carries 455S is outside it, and outside this panel:
    the mutation named `L455F` is not defined against an S455 founder. So the
    rule is "pooled over the clades at risk for this mutation", which is what
    the caption says, and not "the whole tree".

    **GISAID**: the table this reads is cluster-only. The CSV it writes is
    counts, fitness effects and selection factors, which are publishable, and
    carries no sequence.
    """)
    return


@app.cell
def _(mo):
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parent.parent
    for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

    # `panels` before `pandas`: it imports `lib` and then `dasm2`, which moves
    # Arrow off the bundled mimalloc, and `pandas` pulls pyarrow in
    # transitively. Whichever import lands first wins, and this notebook reads
    # a Parquet.
    import panels
    import panel_charts as charts
    import panel_spec as spec

    # Required, not defaulted. A fallback spelling of the path is a second
    # spelling of a fact `workflow/figures.smk` already states, and the rule
    # passes it in.
    _args = mo.cli_args()
    TABLE_PATH = Path(str(_args["table"]))
    SELECTION = str(_args.get("selection") or panels.DEFAULT_SELECTION)
    # A per-model picture, so it is the selection's first model -- the same
    # choice `agreement_mutations.py` makes for its own panel A.
    PANEL_MODEL = panels.selection_models(SELECTION)[0]
    return PANEL_MODEL, SELECTION, TABLE_PATH, charts, panels, spec


@app.cell
def _(PANEL_MODEL, SELECTION, TABLE_PATH, mo, panels, spec):
    SPEC = panels.BASE_SPEC
    POOLED_COLUMN = panels.pooled_metric_column(SPEC)

    # The read is widened here rather than in `panel_spec.TABLE_COLUMNS`: the
    # pooled column is wanted by this panel and by `scripts/dms.py`, and
    # carrying it in the shared tuple would put it in every panel's read and
    # every test fixture. `lookup_frame(pooled=True)` raises if it is absent,
    # so the two halves of this choice cannot drift apart silently.
    TABLE = panels.read_scored_table(
        TABLE_PATH,
        models=[PANEL_MODEL],
        columns=(*spec.TABLE_COLUMNS, POOLED_COLUMN),
    )
    LOOKUP = panels.with_count_noise_interval(
        panels.lookup_frame(
            TABLE,
            mutation=spec.L455F_MUTATION,
            model=PANEL_MODEL,
            spec=SPEC,
            pooled=True,
        ),
        spec=SPEC,
    )
    if LOOKUP.empty:
        raise ValueError(
            f"no clade carries an estimate of {spec.L455F_MUTATION} for "
            f"{PANEL_MODEL} at a cutoff of {panels.PANEL_MIN_COUNT}. The "
            f"label must match mat-pcp's `aa_mutation` spelling exactly: "
            f"wildtype residue, 1-indexed site, mutant residue."
        )

    # Constant down the frame by construction -- the pooled join key does not
    # include the clade -- so this asserts what the column means rather than
    # trusting it. A pooled value that varied by clade would be a per-clade
    # quantity drawn as one horizontal rule, which is a wrong picture that
    # looks right.
    _distinct = LOOKUP[panels.LOOKUP_POOLED_COLUMN].round(9).nunique()
    if _distinct != 1:
        raise ValueError(
            f"{panels.LOOKUP_POOLED_COLUMN} takes {_distinct} distinct values "
            f"over {len(LOOKUP)} clades. It is meant to be one clade-pooled "
            f"estimate, and one horizontal rule cannot stand for several."
        )
    POOLED = float(LOOKUP[panels.LOOKUP_POOLED_COLUMN].iloc[0])

    mo.md(
        f"Selection **{SELECTION}** → `{PANEL_MODEL}`. "
        f"**{len(LOOKUP)} clades** carry an estimate of "
        f"`{spec.L455F_MUTATION}` at a cutoff of {panels.PANEL_MIN_COUNT} "
        f"actual or expected counts. Clade-pooled counts estimate: "
        f"**{POOLED:+.3f}**."
    )
    return LOOKUP, PANEL_MODEL, POOLED, SPEC


@app.cell
def _(LOOKUP, PANEL_MODEL, POOLED, SELECTION, SPEC, charts, mo, panels, spec):
    _chart = charts.mutation_lookup_line(
        LOOKUP,
        spec.L455F_MUTATION,
        clade_order=list(LOOKUP["clade"]),
        x_title="clade",
        y_title="fitness effect",
        color_title="approach",
        series_labels={"delta_fitness": "counts", "log_sf": "DASM"},
        series_colors=(charts.COUNTS_GREY, charts.DASM_PURPLE),
        series_shapes=("circle", "triangle"),
        title=f"estimated per-clade fitness effect of {spec.L455F_MUTATION}",
        reference_value=POOLED,
        reference_label="counts, clades pooled",
        reference_font_size=charts.PANEL_FONT_SIZE,
        # **No shaded band**, for the reason `agreement_mutations.py` gives at
        # length: the interval is computed and carried into the tooltip and
        # the CSV, but drawing it widens the y domain to the thinnest clade's
        # Poisson tail and costs the two series the contrast the panel exists
        # to show. Here it would also swallow the reference rule.
        extra_tooltips={"count_noise_p": "Poisson p (DASM vs count)"},
        # Its own size rather than `lookup_panel_sizing`, which is tuned for a
        # panel sharing a three-across row. This one has the row to itself.
        width=spec.L455F_LOOKUP_WIDTH,
        height=spec.L455F_LOOKUP_HEIGHT,
    )
    # Every piece of type on the panel at one size, set through `configure_*`
    # on the outer layer rather than per-encoding, so the series, the title
    # and the reference label cannot drift apart.
    _chart = (
        _chart.configure_axis(
            labelFontSize=charts.PANEL_FONT_SIZE,
            titleFontSize=charts.PANEL_FONT_SIZE,
        )
        .configure_legend(
            labelFontSize=charts.PANEL_FONT_SIZE,
            titleFontSize=charts.PANEL_FONT_SIZE,
        )
        .configure_title(fontSize=charts.PANEL_FONT_SIZE)
    )
    charts.save_panel(_chart, spec.L455F_LOOKUP_PANEL, selection=SELECTION)
    panels.write_panel_csv(
        LOOKUP.assign(model=PANEL_MODEL, mutation=spec.L455F_MUTATION),
        spec.L455F_LOOKUP_PANEL,
        selection=SELECTION,
        spec=SPEC,
    )
    mo.vstack([mo.md(f"**{len(LOOKUP)} clades** carry an estimate."), _chart])
    return


if __name__ == "__main__":
    app.run()
