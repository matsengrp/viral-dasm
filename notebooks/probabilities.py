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
    # `fig:probabilities` — what a mutation probability is, and what it costs

    Two panels. **A** draws the identity the model is built on; **B** draws
    what that identity does to agreement with an assay, beside the published
    model that also reports a probability.

    **The identity.** The DASM defines `p = lambda * f / Z`, so

        log p = log f + log lambda - log Z

    with `log Z` one number per founder. Every mutation therefore sits on a
    line of slope 1 through its own `log f`, displaced vertically by its own
    neutral rate. Panel A colours by that rate, which makes the displacement
    legible: the correlation between the vertical residual and `log lambda` is
    exactly 1, so the iso-colour contours **are** the slope-1 lines.

    **Panel A is tautological, and that is what it is for.** It cannot come out
    any other way — it is a picture of an algebraic identity, not a test of
    one. It earns its place by making the identity readable, and the caption
    should not imply otherwise. What it is *not* is a thin ribbon: `log lambda`
    spans about 8.6 nats against `log f`'s 14.5, so the cloud is a fat diagonal
    ellipse carrying a 45-degree colour gradient.

    **Single-nucleotide only, which is a definition rather than a filter.**
    `lambda` is exactly zero on a multi-nucleotide codon mutation, so `log p`
    is undefined there — not small, undefined. Panel A has no y value to plot
    and panel B no correlation to report.

    **Panel B reads, it does not compute.** Every bar is a row of
    `predictor_correlations.csv`, so the picture and the table beside it cannot
    drift. Panel A's `r` is the one exception in this figure: it is the
    correlation of the two plotted model quantities, not of either against an
    assay, so that table holds no row for it. All three arms are scored on one population through one join, so a
    difference between bars is the method rather than the cell set.

    **GISAID**: reads scored tables and a correlation CSV. No sequence and no
    per-site count passes through this notebook.
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

    # `panels` before `pandas`, for `notebooks/dms_strata.py`'s reason: it
    # imports `lib` and then `dasm2`, which moves Arrow off the bundled
    # mimalloc, and `pandas` pulls pyarrow in transitively.
    import panels
    import dms_prob
    import dms_summary
    import lib
    import panel_charts as charts
    import panel_spec
    import pandas as pd
    import altair as alt

    # Panel A draws every single-nucleotide mutation on the founder, which is
    # over Altair's default 5,000-row ceiling. The ceiling is worth disabling
    # rather than working around: `chart.save()` writes the PNG through
    # `vl-convert` regardless while `chart.to_dict()` raises, so a rule that
    # tripped it would fail *after* its panel was already on disk.
    alt.data_transformers.disable_max_rows()

    SELECTION = panel_spec.DMS_SELECTION
    HOMOLOG = panel_spec.PROBABILITIES_SCATTER_HOMOLOG

    # Required, not defaulted, as the other panel notebooks have it.
    _args = mo.cli_args()
    SCORE_PATHS = [Path(str(path)) for path in _args.get_all("scores")]
    # Panel A is founder-wide, so it needs the scored grid and the era's
    # neutral archive rather than the per-cell score files panel B reads.
    SCORED_TABLE = Path(str(_args.get("table")))
    NEUTRAL_ARCHIVE = Path(str(_args.get("neutral-archive")))
    return (
        HOMOLOG,
        NEUTRAL_ARCHIVE,
        SCORED_TABLE,
        SCORE_PATHS,
        SELECTION,
        alt,
        charts,
        dms_prob,
        dms_summary,
        lib,
        pd,
        panel_spec,
        panels,
    )


@app.cell
def _(
    HOMOLOG,
    NEUTRAL_ARCHIVE,
    SCORED_TABLE,
    SCORE_PATHS,
    SELECTION,
    dms_prob,
    dms_summary,
    lib,
    pd,
    panel_spec,
):
    # **Panel A is founder-wide; panel B is assay-wide.** They answer different
    # questions and so draw different populations. Panel A asks how a mutation
    # probability sits relative to its fitness effect, which is a property of
    # the model on its founder -- so it draws every single-nucleotide mutation
    # the model was trained on, whether or not any assay measured it.
    # Restricting it to measured cells would answer a question about the model
    # on a subset chosen by a different method.
    _clade = lib.dms_dataset(HOMOLOG).clade
    _founder = dms_summary.dms_models.homolog_cells(HOMOLOG)[1]
    scatter = dms_prob.founder_frame(_clade, _founder, SCORED_TABLE, NEUTRAL_ARCHIVE)

    # Panel B is the effect of that relationship on agreement with an assay,
    # so it is restricted to what the assay measured -- and read from the
    # table rather than recomputed.
    _all_stats = pd.read_csv(
        panel_spec.panel_csv_path(SELECTION, panel_spec.DMS_PREDICTOR_CORRELATIONS)
    )
    bars = _all_stats[
        (_all_stats["stratum"] == "single_nt")
        & (_all_stats["predictor"].isin(panel_spec.PROBABILITIES_BAR_ARMS))
    ]
    return bars, scatter


@app.cell
def _(HOMOLOG, bars, mo, scatter):
    _residual = scatter["dasm_prob"] - scatter["dasm"] - scatter["neutral_rate"]
    mo.md(
        f"**Panel A**, {HOMOLOG}: {len(scatter):,} single-nucleotide mutations "
        f"on the founder -- the whole protein, not the measured part of it. "
        f"`log p - log f - log lambda` is constant at {_residual.mean():.6f} "
        f"over a range of {_residual.max() - _residual.min():.2g}, which is "
        f"`-log Z` and float noise. The correlation between the vertical "
        f"residual and `log lambda` is "
        f"{scatter['dasm_prob'].sub(scatter['dasm']).corr(scatter['neutral_rate']):.10f}.\n\n"
        "**Panel B**, over the cells the assays measured, read from "
        "`predictor_correlations.csv`:\n\n" + bars.to_markdown(index=False)
    )
    return


@app.cell
def _(HOMOLOG, SELECTION, charts, panel_spec, panels, scatter):
    _stem_a = panel_spec.PROBABILITIES_PANELS[0]
    panel_a = charts.save_panel(
        charts.probability_against_fitness(
            scatter,
            x="dasm",
            y="dasm_prob",
            colour="neutral_rate",
            homolog=HOMOLOG,
            # Computed here, not read: this panel sets two model quantities
            # against each other with no assay on either axis, so
            # `predictor_correlations.csv` has no row for it.
            r=float(scatter["dasm"].corr(scatter["dasm_prob"])),
        ),
        _stem_a,
        selection=SELECTION,
    )
    # `spec=None`: this panel applied no count floor and no clip, and
    # `dms.dms_spec` would stamp both onto the CSV's provenance.
    #
    # No `dms_cell_entry` column: this panel has no assay on it. The CSV
    # carries what is drawn, and what is drawn is the model against itself
    # over the whole protein.
    panels.write_panel_csv(
        scatter[
            [
                "aa_site",
                "parent_aa",
                "mutant_aa",
                "dasm",
                "dasm_prob",
                "neutral_rate",
            ]
        ],
        _stem_a,
        selection=SELECTION,
        spec=None,
    )
    return (panel_a,)


@app.cell
def _(SELECTION, bars, charts, lib, panel_spec, panels):
    # **Measured, not guessed.** The two panels sit side by side, so they must
    # come out level in height -- the mirror of the width levelling
    # `fig:dms_strata` needs. Read off the two written PNGs at the ppi
    # `save_panel` writes, and re-measure whenever either panel's geometry
    # changes, the arm labels included: their rotated text sets the bar
    # panel's axis height. `notebooks/probabilities_figure.py` raises if the
    # two stop matching, and its message carries the correction to apply.
    BAR_HEIGHT = 102

    _stem_b = panel_spec.PROBABILITIES_PANELS[1]
    panel_b = charts.save_panel(
        charts.dms_correlation_bars(
            bars,
            # `config.yaml`'s order, which is chronological by clade --
            # not alphabetical, which would read BA.1, KP.3.1.1, XBB.1.5
            # and put the newest founder in the middle.
            homologs=[h for h in lib.dms_homologs() if h in set(bars["homolog"])],
            arms=panel_spec.PROBABILITIES_BAR_ARMS,
            bar_height=BAR_HEIGHT,
        ),
        _stem_b,
        selection=SELECTION,
    )
    panels.write_panel_csv(
        bars[["homolog", "predictor", "stratum", "pearson_r", "n_mutations"]],
        _stem_b,
        selection=SELECTION,
        spec=None,
    )
    return (panel_b,)


@app.cell
def _(alt, panel_a, panel_b):
    # Previewed as one chart for `notebooks/dms_strata.py`'s reason: two
    # independently placed charts do not line up however well their specs
    # agree, and this figure's whole layout question is whether the two come
    # out level.
    alt.hconcat(panel_a, panel_b, spacing=24).resolve_scale(color="independent")
    return


if __name__ == "__main__":
    app.run()
