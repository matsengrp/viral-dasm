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
    # `fig:dms_strata` — two whole-space models, cut by codon accessibility

    One assay, two predictors that are defined on **every** amino-acid
    mutation, and three populations: all measured mutations, those a single
    nucleotide change to the founder's codon reaches, and those reachable only
    by two or three changes.

    **Why these two models.** The paper's other predictors are not defined on
    the multi-nucleotide half at all — mat-pcp estimates only what the tree
    could observe, and the DASM's mutation probability is zero off the
    single-nucleotide set because the neutral model gives no rate there. The
    DASM's *selection factor* and antiGen both answer everywhere, so the
    multi-nucleotide column is the one place methods that span the whole
    mutational space can be set against each other.

    **The split is the founder's, not the tree's and not the assay's.** It is
    read from the clade founder's own codons: a mutation mat-pcp emitted no
    count for is still single-nucleotide if its codon is one change away. It is
    deliberately *not* read from the construct each study expressed, which is
    routinely codon-optimized and so carries codons that never circulated.

    **Different units down the page.** The DASM's column is a log selection
    factor and antiGen's a log probability, and nothing converts between them,
    so the rows share the measurement axis and never a score axis.

    **Population.** Every cell here survived the training mask —
    `scripts/dms_censor.py` — so no mutation is scored that the model was
    taught to ignore. That is a different population from `fig:dms`, which
    takes its mask from mat-pcp's counts and is single-nucleotide throughout as
    a result; the two figures' *r* for the DASM are not the same number and are
    not meant to be.

    **GISAID**: reads the predictor scores and mat-pcp's founders and censor
    tables. The DMS measurements are published data. What it writes —
    selection factors, log probabilities and correlations — is publishable.
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
    # transitively. Whichever import lands first wins.
    import panels
    import dms
    import dms_summary
    import panel_charts as charts
    import panel_spec
    import pandas as pd

    SELECTION = panel_spec.DMS_SELECTION
    HOMOLOG = panel_spec.SCATTER_HOMOLOG

    # Required, not defaulted, as the other panel notebooks have it.
    _args = mo.cli_args()
    SCORE_PATHS = [Path(str(path)) for path in _args.get_all("scores")]
    return (
        HOMOLOG,
        SCORE_PATHS,
        SELECTION,
        charts,
        dms,
        dms_summary,
        pd,
        panel_spec,
        panels,
    )


@app.cell
def _(HOMOLOG, SCORE_PATHS, SELECTION, dms_summary, pd, panel_spec):
    # The same join the correlation table is computed from, called rather than
    # respelled: two spellings of it would be two populations.
    _merged, _ = dms_summary.merged_frames(SCORE_PATHS)
    frame = _merged[HOMOLOG]

    # `r` and `n` are read from the table rather than recomputed here, so the
    # picture and the CSV beside it cannot drift. This is why the rule declares
    # the correlation table as an input.
    _all_stats = pd.read_csv(
        panel_spec.panel_csv_path(SELECTION, panel_spec.DMS_PREDICTOR_CORRELATIONS)
    )
    stats = _all_stats[_all_stats["homolog"] == HOMOLOG]
    return frame, stats


@app.cell
def _(HOMOLOG, charts, dms, frame, mo, panel_spec, stats):
    mo.md(
        f"**{HOMOLOG}**, {len(frame):,} measured cells: "
        + ", ".join(
            f"{' '.join(charts.dms_stratum_label(_s))} "
            f"{len(frame) if _s == dms.ALL else int((frame['stratum'] == _s).sum()):,}"
            for _s in panel_spec.DMS_STRATA_COLUMNS
        )
        + ". The left cell is the union of the other two, so its *n* is their "
        "sum. Pearson *r*, read from `predictor_correlations.csv`:\n\n"
        + stats.to_markdown(index=False)
    )
    return


@app.cell
def _(SELECTION, charts, dms, frame, panel_spec, panels, stats):
    # One x domain for the whole figure. Computed once and passed into both
    # rows: the measurement is the same quantity in all six cells, and a column
    # that cannot be read down the page is the one thing this layout is for.
    x_domain = charts.dms_padded_domain(frame["dms_cell_entry"])

    _last = len(panel_spec.DMS_STRATA_MODELS) - 1

    strata_panels = []
    for _index, (_stem, (_model, _title)) in enumerate(
        zip(panel_spec.DMS_STRATA_PANELS, panel_spec.DMS_STRATA_MODELS)
    ):
        _row = charts.save_panel(
            charts.dms_stratum_row(
                frame,
                score_column=_model,
                score_title=list(_title),
                stratum_order=panel_spec.DMS_STRATA_COLUMNS,
                x_domain=x_domain,
                stats=stats[stats["predictor"] == _model],
                # The column heading and the x title belong to the column, so
                # each is drawn once: the heading on the top row, the title on
                # the bottom.
                show_column_titles=_index == 0,
                show_x_title=_index == _last,
            ),
            _stem,
            selection=SELECTION,
        )
        strata_panels.append(_row)
        # `spec=None`: these panels applied no count floor and no clip, and
        # `dms.dms_spec` would stamp both onto the CSV's provenance.
        panels.write_panel_csv(
            frame[["aa_site", "parent_aa", "mutant_aa", "stratum", "dms_cell_entry", _model]],
            _stem,
            selection=SELECTION,
            spec=None,
        )
    return (strata_panels,)


@app.cell
def _(strata_panels):
    strata_panels[0]
    return


@app.cell
def _(strata_panels):
    strata_panels[1]
    return


if __name__ == "__main__":
    app.run()
