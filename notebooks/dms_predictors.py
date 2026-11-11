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
    # `fig:dms` panel C

    Every predictor's correlation with the three spike deep mutational scans,
    per homolog and per stratum, as one row of grouped bars.

    Two predictors so far, both references: **BLOSUM62**, a general substitution
    matrix that knows nothing about SARS-CoV-2, and the **leaf frequency**, how
    conserved each site is among the clade's own sampled sequences. The DASM's
    own bar is a filter on `selection_factors.parquet` and joins when that table
    exists; the chart takes however many predictors the table carries.

    The bars are drawn from zero because a correlation has a meaningful zero,
    and nothing here clips a negative one.

    **GISAID**: reads a publishable correlation table only; carries no sequence,
    no site and no sample.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parent.parent
    for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

    # `panel_charts` before `pandas`, for the reason `subsampling_depth` gives
    # for `panels`: it imports `lib` and then `dasm2`, which moves Arrow off the
    # bundled mimalloc, and `pandas` pulls pyarrow in transitively.
    import panel_charts as charts
    import panel_spec

    from lib import dms_homologs

    import dms

    import pandas as pd

    return charts, dms, dms_homologs, panel_spec, pd


@app.cell
def _(mo, panel_spec):
    mo.md(f"""
    Reading `{
        panel_spec.panel_csv_path(
            panel_spec.DMS_SELECTION, panel_spec.DMS_PREDICTOR_CORRELATIONS
        )
    }`.
    """)
    return


@app.cell
def _(charts, panel_spec, pd):
    table = pd.read_csv(
        panel_spec.panel_csv_path(
            panel_spec.DMS_SELECTION, panel_spec.DMS_PREDICTOR_CORRELATIONS
        )
    )
    # The CSV carries predictor *ids*, which are join keys; a legend prints
    # labels. `predictor_label` raises on an id it has no label for, so a new
    # predictor fails here rather than being drawn as a bare id.
    plotted = table.assign(
        predictor=[charts.predictor_label(name) for name in table["predictor"]]
    )
    plotted
    return (plotted,)


@app.cell
def _(charts, dms, dms_homologs, panel_spec, plotted):
    panel_c = charts.save_panel(
        charts.homolog_correlation_panel(
            plotted,
            # Chronological, as the paper names them, which is `config.yaml`'s
            # own order rather than an alphabetical one.
            homolog_order=dms_homologs(),
            stratum_order=list(dms.STRATA),
            predictor_order=[
                charts.predictor_label(name) for name in panel_spec.DMS_PREDICTORS
            ],
            # The rank correlation, not Pearson. The leaf predictor puts every
            # substitution its clade has never shown at one floor, so its scale
            # is bimodal and its Pearson says nothing -- on KP.3.1.1's multi-nt
            # cells +0.04 against a Spearman of +0.58.
            metric="spearman_rho",
        ),
        panel_spec.DMS_PREDICTOR_CORRELATIONS,
        selection=panel_spec.DMS_SELECTION,
    )
    # No `write_panel_csv`: `scripts/dms_summary.py` wrote the table this panel
    # is a picture of, at the same stem. A second writer would be a second
    # definition of the same file.
    panel_c
    return


if __name__ == "__main__":
    app.run()
