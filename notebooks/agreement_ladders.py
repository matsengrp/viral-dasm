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
    # `fig:agreement`, panels C and E: agreement against how well it was measured

    Two panels on one x axis, and the x axis is the **count cutoff** — the
    minimum actual or expected count a mutation needs to enter a
    correlation. **Panel C**: one dot per clade, the within-clade Pearson r.
    **Panel E**: one dot per unordered clade pair, the r between ΔlogSF and
    Δfitness over the mutations the two clades share. The grey dashes are
    the medians.

    **What a rising curve means.** The same model, on a better-measured
    subset of the same clades, agrees better with the counts. That is
    evidence about the *yardstick*, not about the model: the distance from
    r = 1 at the lowest rung was substantially noise in mat-pcp's estimate
    rather than error in the DASM.

    **The clade set is pinned across the rungs.** Raising the floor removes
    mutations, which removes whole clades through the adequacy floor — and
    the clades it removes first are the thin, poorly-measured ones whose r
    is lowest. A sweep that let each rung keep its own clade set would show
    r rising with the threshold even if no surviving clade moved at all:
    the trend would be survivorship. `count_ladder_adequacy` restricts every
    rung to the clades that clear the floor at all of them, and reports what
    each rung would have kept on its own, so the cost of the pairing is
    visible rather than assumed.

    **A pair thins faster than a clade.** A mutation leaves a pair as soon
    as *either* side falls below the cutoff, so panel E's dots rest on fewer
    mutations than panel C's at the same rung.

    **GISAID**: reads the cluster-only scored table; writes publishable
    aggregate statistics.
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

    _args = mo.cli_args()
    TABLE_PATH = Path(
        str(
            _args.get("table")
            or panels.ARTIFACT_ROOT / "scored" / "selection_factors.parquet"
        )
    )
    SELECTION = str(_args.get("selection") or panels.DEFAULT_SELECTION)
    MODELS = panels.selection_models(SELECTION)
    return MODELS, SELECTION, TABLE_PATH, charts, panels, pd


@app.cell
def _(MODELS, SELECTION, TABLE_PATH, mo, panels):
    REVERTING = panels.reverting_sites_by_clade()
    SPEC = panels.BASE_SPEC
    # Panel C scores every clade that can be scored, so the roster is the
    # clade axis here rather than a selection; panel E draws pairs from the
    # shift clades, which are a selection because pairs are quadratic.
    TABLE = panels.read_scored_table(TABLE_PATH, models=MODELS)
    CLADES = sorted(TABLE["clade"].unique())
    BASE = panels.base_frames(
        TABLE,
        spec=SPEC,
        reverting_by_clade=REVERTING,
        models=MODELS,
        clades=CLADES,
    )
    mo.md(
        f"Selection **{SELECTION}** → models `{'`, `'.join(MODELS)}`, "
        f"{len(CLADES)} clades, {len(BASE)} (model, clade) frames at cutoff "
        f"zero."
    )
    return BASE, CLADES, REVERTING, SPEC, TABLE


@app.cell
def _(BASE, MODELS, SELECTION, SPEC, charts, mo, panels, pd):
    FOUNDERS = panels.shift_pair_founders()
    _within, _shift, _adequacy, _paired_by_model = [], [], [], {}
    for _model in MODELS:
        _base = panels.by_clade(BASE, _model)
        if not _base:
            continue
        # The paired clade set, and the per-rung adequacy it came from, kept so
        # the pairing's cost can be named rather than assumed.
        _paired, _adq = panels.count_ladder_adequacy(
            _base,
            thresholds=panels.COUNT_LADDER,
            min_mutations=SPEC.min_clade_mutations,
        )
        _adequacy.append(_adq.assign(model=_model))
        _paired_by_model[_model] = list(_paired)
        if _paired:
            _within.append(
                panels.within_clade_count_ladder(
                    _base,
                    SPEC.fitness_metric,
                    _paired,
                    thresholds=panels.COUNT_LADDER,
                ).assign(model=_model)
            )
        # A clade can be selected for the shift panel and still be too thin to
        # correlate at every rung, so the pairs are drawn from the intersection.
        _pair_clades = [c for c in panels.SHIFT_CLADES if c in _paired]
        if len(_pair_clades) >= 2:
            _shift.append(
                panels.shift_pair_count_ladder(
                    {c: _base[c] for c in _pair_clades},
                    SPEC.fitness_metric,
                    _pair_clades,
                    thresholds=panels.COUNT_LADDER,
                    founders=FOUNDERS,
                ).assign(model=_model)
            )

    WITHIN_LADDER = pd.concat(_within, ignore_index=True)
    SHIFT_LADDER = pd.concat(_shift, ignore_index=True)
    ADEQUACY = pd.concat(_adequacy, ignore_index=True)
    PAIRED_BY_MODEL = _paired_by_model
    mo.md(
        "\n".join(
            f"- `{model}`: {len(clades)} clades clear the floor of "
            f"{SPEC.min_clade_mutations} at every rung of "
            f"{panels.COUNT_LADDER}"
            for model, clades in PAIRED_BY_MODEL.items()
        )
    )
    return ADEQUACY, PAIRED_BY_MODEL, SHIFT_LADDER, WITHIN_LADDER


@app.cell
def _(SELECTION, SPEC, WITHIN_LADDER, charts, panels):
    # Pooled over the selection's models under one median, which is what the
    # published numbers do: each rung's dots are every (model, clade) pair.
    panel_c = charts.save_panel(
        charts.ladder_row(
            WITHIN_LADDER,
            unit_field="clade",
            unit_title="Clade",
            title=[
                "per-clade correlation in",
                "estimated fitness effects",
                f"(N clades = {charts.ladder_clade_count(WITHIN_LADDER, 'clade')})",
            ],
        ),
        "panel_c_within_clade_ladder",
        selection=SELECTION,
    )
    panels.write_panel_csv(
        WITHIN_LADDER,
        "panel_c_within_clade_ladder",
        selection=SELECTION,
        spec=SPEC,
    )
    panel_c
    return


@app.cell
def _(SELECTION, SHIFT_LADDER, SPEC, charts, panels):
    panel_e = charts.save_panel(
        charts.ladder_row(
            SHIFT_LADDER,
            unit_field="pair_label",
            unit_title="Clade pair",
            title=[
                "correlation in estimated shifts",
                "in fitness effects between clades",
                "(N clades = "
                + f"{charts.ladder_clade_count(SHIFT_LADDER, 'pair_label')})",
            ],
        ),
        "panel_e_shift_ladder",
        selection=SELECTION,
    )
    panels.write_panel_csv(
        SHIFT_LADDER, "panel_e_shift_ladder", selection=SELECTION, spec=SPEC
    )
    panel_e
    return


@app.cell
def _(ADEQUACY, SHIFT_LADDER, WITHIN_LADDER, mo):
    mo.vstack(
        [
            mo.md("""
            ### What the panels report

            `pearson_r` below zero is drawn on the floor and kept raw in the
            CSVs: the sign of a correlation between two noise processes
            carries no information, and left unclipped those dots stretch the
            y axis over a region nothing is being claimed about.
            """),
            mo.ui.table(
                WITHIN_LADDER.groupby(["model", "min_count"])["pearson_r"]
                .agg(["median", "count"])
                .reset_index(),
                selection=None,
            ),
            mo.ui.table(
                SHIFT_LADDER.groupby(["model", "min_count"])["pearson_r"]
                .agg(["median", "count"])
                .reset_index(),
                selection=None,
            ),
            mo.md("#### What each rung would have kept on its own"),
            mo.ui.table(
                ADEQUACY.groupby(["model", "min_count"])["adequate"]
                .agg(["sum", "count"])
                .reset_index(),
                selection=None,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
