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
    # `fig:within_clade_shifts` panels A, C and E

    *For Nextstrain clades 22D, 20B and 21K, the estimated effects strongly
    differ between the counts and DASM approaches for a few mutations.*

    One scatter per clade, the same mark and the same square domain as
    `fig:agreement`'s panel B, with the figure's candidate mutations named.
    Each panel establishes the subject of the panel beside it, which asks what
    that mutation does on the clade's two genetic backgrounds -- so the subject
    has to be a mutation the two approaches visibly disagree about.

    **Two groups of labelled mutation, in two inks.** Near-black names the
    subject: the mutation whose shift the row measures. Orange names the
    substitutions that *define* the two backgrounds, and only in 22D, where
    they sit well below the diagonal and a reader is owed the reason. The
    counts approach takes the opportunity for a mutation to be the clade's
    fitted rate scaled to the whole clade while counting it only on branches
    still carrying the founder residue, so once most of the clade carries a
    mutation its counts-based estimate is understated. That is not a shift.

    20B's and 21K's background mutations did not sweep as far, so the two
    approaches agree about them and they sit on the diagonal with everything
    else. `BACKGROUND_LABEL_CLADES` keeps them unlabelled: the panel is about
    where the approaches *disagree*, and a label on a mark that is where it
    should be points at nothing to see.

    **Two lists, one labelled set.** The subjects come from
    `data/shift_candidates.csv`, which this panel and the next both read, so a
    candidate moved off the figure leaves both in one edit. The background
    mutations are not in that file at all: they are derived from branch
    frequencies by `rule clade_backgrounds` into
    `results/backgrounds/clade_backgrounds.csv`, so no model has a say in which
    backgrounds the figure contrasts. Nothing here selects on the size of the
    disagreement; `panels.candidate_rows` raises if a named candidate is not in
    the clade's comparison frame, because a panel drawing six of seven labelled
    mutations is worse than one that stops.

    **GISAID**: reads the cluster-only scored table; writes effects and counts,
    which are publishable, and no sequence.
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

    # Required, not defaulted, as the agreement notebooks are: a fallback
    # spelling here is a second spelling of what `workflow/figures.smk`
    # already states and passes in.
    _args = mo.cli_args()
    TABLE_PATH = Path(str(_args["table"]))
    SELECTION = str(_args.get("selection") or panels.DEFAULT_SELECTION)
    MODELS = panels.selection_models(SELECTION)
    # A per-model picture, like the agreement figure's own per-mutation
    # panels: the selection's first model, not a pool.
    PANEL_MODEL = MODELS[0]
    return MODELS, PANEL_MODEL, SELECTION, TABLE_PATH, charts, panels, pd


@app.cell
def _(MODELS, SELECTION, TABLE_PATH, mo, panels):
    CLADES = list(panels.CANDIDATE_CLADES)
    CANDIDATES = panels.labelled_candidates()
    SPEC = panels.BASE_SPEC

    TABLE = panels.read_scored_table(TABLE_PATH, models=MODELS, clades=CLADES)
    # At cutoff zero, so the cutoff below is the panel's own and these are the
    # same frames every other panel of either figure is filtered from.
    BASE = panels.base_frames(TABLE, spec=SPEC, models=MODELS, clades=CLADES)
    mo.md(
        f"Selection **{SELECTION}** → models `{'`, `'.join(MODELS)}`, clades "
        f"`{'`, `'.join(CLADES)}`, {len(CANDIDATES)} candidate mutations."
    )
    return BASE, CANDIDATES, CLADES, SPEC


@app.cell
def _(BASE, CANDIDATES, CLADES, PANEL_MODEL, SELECTION, SPEC, charts, mo, panels):
    PANELS = {}
    for _clade, _stem in zip(CLADES, panels.OUTLIER_PANELS):
        _frame = panels.filter_merged_fitness(
            BASE[(PANEL_MODEL, _clade)],
            min_actual_count=panels.PANEL_MIN_COUNT,
            min_predicted_count=panels.PANEL_MIN_COUNT,
            count_filter_mode=panels.COUNT_FILTER_MODE,
        )
        _labelled = panels.candidate_rows(_frame, _clade, CANDIDATES)
        PANELS[_clade] = charts.save_panel(
            charts.scatter_panel(
                _frame,
                metric=SPEC.fitness_metric,
                clade=_clade,
                clip_lower=SPEC.clip_lower,
                stat=charts.stat_label(None, len(_frame)),
                labelled=_labelled,
                label_side=panels.CANDIDATE_LABEL_SIDES[_clade],
            ),
            _stem,
            selection=SELECTION,
        )
        # The labelled rows alone, which is what this panel's caption is
        # about. The cloud behind them is the agreement figure's own row set
        # at the same cutoff, committed with that figure rather than again
        # here.
        panels.write_panel_csv(
            _labelled.assign(
                model=PANEL_MODEL, clade=_clade, min_count=panels.PANEL_MIN_COUNT
            )[
                [
                    "model",
                    "clade",
                    "min_count",
                    "candidate_group",
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
            _stem,
            selection=SELECTION,
            spec=SPEC,
        )
    mo.vstack([item for _c in CLADES for item in (mo.md(f"### {_c}"), PANELS[_c])])
    return


if __name__ == "__main__":
    app.run()
