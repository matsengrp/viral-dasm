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
    # `fig:agreement`, panels A, B and D: the mutations behind the summary

    Three per-mutation views of one question — does a DASM's log selection
    factor agree with mat-pcp's counts-based fitness effect?

    - **Panel A** traces one mutation across every clade that carries an
      estimate of it, with the band the observed counts can support under
      Poisson noise.
    - **Panel B** is one clade's whole cloud, which is the data behind a
      single dot of panel C.
    - **Panel D** is the same comparison on *shifts*: how an effect moves
      between two clades, one cell per clade pair.

    Everything here reads `selection_factors.parquet` and runs no inference.
    The statistics come from `scripts/panels.py`, the drawing from
    `scripts/panel_charts.py`; this notebook is the layout and the
    commentary.

    **GISAID**: the table this reads is cluster-only. The panel CSVs it
    writes are counts, fitness effects and selection factors, which are
    publishable, and carry no sequence.
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
    # Required, not defaulted. A fallback spelling of either path is a second
    # spelling of a fact `workflow/figures.smk` already states, and the rule
    # passes both in; missing one should stop here rather than quietly read a
    # different file from the one the rule declared.
    _args = mo.cli_args()
    TABLE_PATH = Path(str(_args["table"]))
    AUDIT_PATH = Path(str(_args["audit"]))
    SELECTION = str(_args.get("selection") or panels.DEFAULT_SELECTION)
    MODELS = panels.selection_models(SELECTION)
    # Panels A, B and D are per-model pictures, so both the panel and its CSV
    # are the selection's first model. Only panels C and E pool a selection's
    # models, which is what the published ladder numbers do.
    PANEL_MODEL = MODELS[0]
    return (
        MODELS,
        PANEL_MODEL,
        AUDIT_PATH,
        SELECTION,
        TABLE_PATH,
        charts,
        panels,
        pd,
    )


@app.cell
def _(AUDIT_PATH, MODELS, SELECTION, TABLE_PATH, mo, panels):
    REVERTING = panels.reverting_sites_by_clade(AUDIT_PATH)
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
        reverting_by_clade=REVERTING,
        models=MODELS,
        clades=CLADES,
    )
    mo.md(
        f"Selection **{SELECTION}** → models `{'`, `'.join(MODELS)}`, "
        f"{len(BASE)} (model, clade) frames, "
        f"{sum(len(frame) for frame in BASE.values()):,} mutations before any "
        f"count cutoff."
    )
    return BASE, REVERTING, SPEC


@app.cell
def _(mo):
    mo.md(r"""
    ## Panel A — one mutation across the clades

    Two series of one quantity: mat-pcp's counts-based effect and the DASM's
    log selection factor, for a single mutation, in every clade that carries
    an estimate of it. Not restricted to the clades the correlation panels
    score — the point of this panel is to show *where* the two approaches
    disagree, which is not confined to a selection.

    The grey band is what the observed count can support under Poisson noise
    in that count alone, holding the expected count fixed, because both
    estimates are anchored to the same neutral model and an error in that
    expectation moves them together.
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
            interval_columns=("count_noise_lo", "count_noise_hi"),
            interval_label=f"{panels.COUNT_NOISE_COVERAGE:.0%} count noise",
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
    ## Panel B — one clade's cloud

    Every surviving mutation in **{panels.SCATTER_CLADE}** at a cutoff of
    {panels.PANEL_MIN_COUNT}, the DASM on x and the counts on y, on one
    shared domain so the dashed `y = x` is a true 45 degrees and a point's
    distance from it is the size of the disagreement. The firebrick line is
    the least-squares fit; r and n in the corner are read from the same
    table the CSV is written from.
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
    ## Panel D — the same question about *shifts*

    One cell per unordered clade pair drawn from
    `{"`, `".join(panels.SHIFT_CLADES)}`, laid out like a pairplot: the
    column is the earlier clade, the row is the later one. Both shifts are
    `later − earlier`, over the mutations the two clades share whose
    wildtype residue agrees — so D614G in one clade is never compared with
    E614G in the other.

    **A pair whose founder is identical in this gene is dropped**, not
    scored at zero: the model sees the same input twice, so no model can
    shift between them, while mat-pcp's Δfitness still moves with each
    clade's own tree counts.
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
                "concordant",
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


@app.cell
def _(SCATTER_METRICS, SHIFT_METRICS, mo):
    mo.vstack(
        [
            mo.md("### What the panels report"),
            mo.ui.table(SCATTER_METRICS, selection=None),
            mo.ui.table(SHIFT_METRICS, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
