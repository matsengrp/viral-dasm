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
    # `fig:agreement` panels A, B and D

    *The DASM learns mutational effects and how they shift between clades.*
    Three per-mutation panels of that figure, drawn from
    `selection_factors.parquet`. No inference and no GPU: `rule score` did the
    forward passes.

    | panel | what the paper shows |
    |---|---|
    | A | F456L across the clades, as an example of an effect that shifts |
    | B | every mutation to the Delta (21J) founder, counts against DASM |
    | D | shifts between four clades well separated on the tree |

    Panels C and E are the count-threshold sweeps and live in
    `agreement_ladders.py`.

    **GISAID**: the table this reads is cluster-only. The CSVs it writes are
    counts, fitness effects and selection factors, which are publishable, and
    carry no sequence.
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

    import pandas as pd

    # The rule passes the table in; a second spelling in a notebook is a path
    # that can drift from `workflow/score.smk`'s.
    # Required, not defaulted. A fallback spelling of the path is a second
    # spelling of a fact `workflow/figures.smk` already states, and the rule
    # passes it in; missing it should stop here rather than quietly read a
    # different file from the one the rule declared.
    _args = mo.cli_args()
    TABLE_PATH = Path(str(_args["table"]))
    SELECTION = str(_args.get("selection") or panels.DEFAULT_SELECTION)
    MODELS = panels.selection_models(SELECTION)
    # Panels A, B and D are per-model pictures, so both the panel and its CSV
    # are the selection's first model. Only panels C and E pool a selection's
    # models, which is what the published ladder numbers do.
    PANEL_MODEL = MODELS[0]
    return (
        MODELS,
        PANEL_MODEL,
        SELECTION,
        TABLE_PATH,
        charts,
        panels,
        pd,
    )


@app.cell
def _(MODELS, SELECTION, TABLE_PATH, mo, panels):
    # Every clade these three panels need, read once: panel B's one clade and
    # the shift panel's four.
    CLADES = sorted({panels.SCATTER_CLADE, *panels.SHIFT_CLADES})
    TABLE = panels.read_scored_table(TABLE_PATH, models=MODELS, clades=CLADES)
    SPEC = panels.BASE_SPEC

    # At cutoff zero, so the count filter below is the panels' own and the
    # frames are the same ones the ladder notebook sweeps.
    BASE = panels.base_frames(
        TABLE,
        spec=SPEC,
        models=MODELS,
        clades=CLADES,
    )
    mo.md(
        f"Selection **{SELECTION}** → models `{'`, `'.join(MODELS)}`, "
        f"{len(BASE)} (model, clade) frames, "
        f"{sum(len(frame) for frame in BASE.values()):,} mutations before any "
        f"count cutoff."
    )
    return BASE, SPEC


@app.cell
def _(mo, panels):
    mo.md(f"""
    ## Panel A — F456L across the clades

    The paper's example of a mutation whose effect shifts dramatically
    between clades, and which the DASM predicts. Two series of one quantity:
    mat-pcp's counts-based effect, and the DASM's log selection factor.

    Clades with at least {panels.PANEL_MIN_COUNT} actual or expected counts of
    the mutation, as the caption states. The grey band is what the observed
    count alone can support under Poisson noise, holding the expected count
    fixed — both estimates are anchored to the same neutral model, so an error
    there moves them together.
    """)
    return


@app.cell
def _(PANEL_MODEL, SELECTION, SPEC, TABLE_PATH, charts, mo, panels):
    LOOKUP = panels.with_count_noise_interval(
        panels.lookup_frame(
            # Read again over all clades: this panel's row set is wider than
            # the correlation panels', which read a four-clade selection.
            panels.read_scored_table(TABLE_PATH, models=[PANEL_MODEL]),
            mutation=panels.LOOKUP_MUTATION,
            model=PANEL_MODEL,
            spec=SPEC,
        ),
        spec=SPEC,
    )

    if len(LOOKUP):
        _chart = charts.mutation_lookup_line(
            LOOKUP,
            panels.LOOKUP_MUTATION,
            clade_order=list(LOOKUP["clade"]),
            x_title="clade",
            y_title="fitness effect",
            color_title="approach",
            series_labels={"delta_fitness": "counts", "log_sf": "DASM"},
            series_colors=(charts.COUNTS_GREY, charts.DASM_PURPLE),
            series_shapes=("circle", "triangle"),
            title=(f"estimated per-clade fitness effect of {panels.LOOKUP_MUTATION}"),
            # **No shaded band.** The interval is computed and carried -- it
            # rides into the tooltip as `count_noise_p` and into the panel's
            # CSV as `count_noise_lo`/`_hi` -- but it is not drawn. Shading it
            # is a separate question from having it, and the published panel
            # answers no: the band widens the y domain to cover the thinnest
            # clade's Poisson tail, which costs the two series the contrast
            # the panel exists to show, and makes panel A taller than B and C
            # beside it in the composed row. Upstream's
            # `lookup_band_checkbox` defaults to False for the same reason,
            # and the figure in the manuscript was saved with it off.
            extra_tooltips={"count_noise_p": "Poisson p (DASM vs count)"},
            **charts.lookup_panel_sizing(len(LOOKUP)),
        )
        # Every piece of type on the panel at one size, set through
        # `configure_*` on the outer layer rather than per-encoding, so the
        # band, the two series and the title cannot drift apart.
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
        charts.save_panel(_chart, "panel_a_mutation_lookup", selection=SELECTION)
        panels.write_panel_csv(
            LOOKUP.assign(model=PANEL_MODEL, mutation=panels.LOOKUP_MUTATION),
            "panel_a_mutation_lookup",
            selection=SELECTION,
            spec=SPEC,
        )
        panel_a = _chart
    else:
        raise ValueError(
            f"no clade carries an estimate of {panels.LOOKUP_MUTATION} for "
            f"{PANEL_MODEL}. The label must match mat-pcp's `aa_mutation` "
            f"spelling exactly: wildtype residue, 1-indexed site, mutant "
            f"residue."
        )
    mo.vstack([mo.md(f"**{len(LOOKUP)} clades** carry an estimate."), panel_a])
    return (LOOKUP,)


@app.cell
def _(mo, panels):
    mo.md(f"""
    ## Panel B — the Delta ({panels.SCATTER_CLADE}) founder

    The clade with the most mutation counts, at a cutoff of
    {panels.PANEL_MIN_COUNT} actual or expected counts, which is the paper's
    "mutational effects that are supported by the underlying counts". Both
    estimates are clipped below at {panels.BASE_SPEC.clip_lower:g}, for the
    limited dynamic range the caption notes.

    One shared domain for both axes, so the dashed `y = x` is a true 45
    degrees and a point's distance from it is the size of the disagreement.
    """)
    return


@app.cell
def _(BASE, MODELS, PANEL_MODEL, SELECTION, SPEC, charts, panels, pd):
    _frames = {
        model: panels.apply_count_threshold(
            BASE[(model, panels.SCATTER_CLADE)], panels.PANEL_MIN_COUNT
        )
        for model in MODELS
        if (model, panels.SCATTER_CLADE) in BASE
    }
    SCATTER_METRICS = pd.concat(
        [
            panels.within_clade_metrics(
                {panels.SCATTER_CLADE: frame}, SPEC.fitness_metric
            ).assign(model=model, min_count=panels.PANEL_MIN_COUNT)
            for model, frame in _frames.items()
        ],
        ignore_index=True,
    )
    _drawn = _frames[PANEL_MODEL]
    panel_b = charts.save_panel(
        charts.scatter_panel(
            _drawn,
            metric=SPEC.fitness_metric,
            clade=panels.SCATTER_CLADE,
            clip_lower=SPEC.clip_lower,
            stat=charts.stat_label(
                SCATTER_METRICS[SCATTER_METRICS["model"] == PANEL_MODEL].iloc[0],
                len(_drawn),
            ),
        ),
        "panel_b_within_clade_scatter",
        selection=SELECTION,
    )
    panels.write_panel_csv(
        _drawn.assign(model=PANEL_MODEL, min_count=panels.PANEL_MIN_COUNT)[
            [
                "model",
                "clade",
                "min_count",
                "aa_mutation",
                "aa_site",
                "parent_aa",
                "mutant_aa",
                "log_selection_factor",
                "log_selection_factor_raw",
                SPEC.fitness_metric,
                f"{SPEC.fitness_metric}_raw",
                "actual_count",
                "predicted_count",
            ]
        ],
        "panel_b_within_clade_scatter",
        selection=SELECTION,
        spec=SPEC,
    )
    panel_b
    return (SCATTER_METRICS,)


@app.cell
def _(mo, panels):
    mo.md(f"""
    ## Panel D — shifts between clades

    The paper's four clades separated by long branch lengths —
    `{"`, `".join(panels.SHIFT_CLADES)}` — giving six pairs, laid out like a
    pairplot: the column is the earlier clade, the row is the later one. A
    mutation's shift is the difference of its effects in the two clades, over
    the mutations they share whose wildtype residue agrees, so D614G in one
    clade is never compared with E614G in the other.

    Because shifts are differences of clipped estimates, a mutation sitting at
    the clip in both clades has a shift of exactly zero. A pair whose founder
    is identical in spike is dropped rather than scored at zero: the model sees
    the same input twice, while mat-pcp's Δfitness still moves with each
    clade's own counts.
    """)
    return


@app.cell
def _(
    BASE,
    PANEL_MODEL,
    SELECTION,
    SPEC,
    charts,
    mo,
    panels,
    pd,
):
    FOUNDERS = panels.shift_pair_founders()
    _at_cutoff = {
        clade: panels.apply_count_threshold(
            BASE[(PANEL_MODEL, clade)], panels.PANEL_MIN_COUNT
        )
        for clade in panels.SHIFT_CLADES
        if (PANEL_MODEL, clade) in BASE
    }
    SHIFT_METRICS = panels.shift_pair_metrics(
        _at_cutoff,
        list(_at_cutoff),
        SPEC.fitness_metric,
        founders=FOUNDERS,
    ).assign(model=PANEL_MODEL, min_count=panels.PANEL_MIN_COUNT)
    # The per-mutation shifts the cells draw, for the pairs that survived the
    # founder check -- which is what `shift_pair_metrics` reports on.
    _drawn = [
        panels.shift_frame(
            _at_cutoff[row.clade_a],
            _at_cutoff[row.clade_b],
            row.clade_a,
            row.clade_b,
            SPEC.fitness_metric,
        )
        for row in SHIFT_METRICS.itertuples()
    ]
    SHIFTS = pd.concat(_drawn, ignore_index=True) if _drawn else pd.DataFrame()
    SKIPPED = panels.identical_founder_pairs(FOUNDERS, list(_at_cutoff))

    panel_d = charts.save_panel(
        charts.shift_grid(SHIFTS, SHIFT_METRICS, list(_at_cutoff)),
        "panel_d_shift_grid",
        selection=SELECTION,
    )
    panels.write_panel_csv(
        SHIFTS[
            [
                "pair",
                "mutation_label",
                "aa_site",
                "mutant_aa",
                "sf_shift",
                "fitness_shift",
            ]
        ].assign(model=PANEL_MODEL, min_count=panels.PANEL_MIN_COUNT),
        "panel_d_shift_grid",
        selection=SELECTION,
        spec=SPEC,
    )
    mo.vstack(
        [
            mo.md(
                f"**skipped — identical {panels.GENE} founders:** "
                + (", ".join(f"{a}↔{b}" for a, b in SKIPPED) if SKIPPED else "none")
            ),
            panel_d,
        ]
    )
    return SHIFTS, SHIFT_METRICS


if __name__ == "__main__":
    app.run()
