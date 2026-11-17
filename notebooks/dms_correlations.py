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
    # `fig:dms` — estimated fitness effects against deep mutational scanning

    Four full-spike cell-entry assays, each on the clade whose founder the
    model is scored on, against three predictors of the same quantity: the
    **DASM**, mat-pcp's **per-clade** counts estimate, and its **full-tree**
    (all-clade pooled) one.

    **Panels A and B** are every mutation the tree could estimate at all;
    **panels C and D** are the subset with at least ten actual or expected
    counts. A and C are one homolog's clouds, B and D the same coefficient for
    all four at once. Reading across answers "does this hold for the other
    homologs", reading down answers "does it survive thresholding".

    **`fig:dms` lays out A and B only.** All four are written, with their
    CSVs; `panel_spec.DMS_FIGURE_ROWS` is what puts two of them on the page.
    That is also why the two rows head themselves differently — the pair the
    figure carries names the homolog, in black, and the pair it does not
    keeps the count floor in its row colour.

    **A count floor of 0 is already a restriction.** A mutation mat-pcp never
    estimated carries no counts and no fitness effect, so it is absent from
    the top row too — which, as the source experiment establishes, makes that
    row the single-nucleotide-reachable set.

    Ported from `dms_comparison.py` in `dasm2-experiments`'
    `experiments/2026-07-20-spike-dms-comparison`, which drew the figure the
    manuscript carries today. The scoring half of that notebook is gone:
    `rule score` has already evaluated the checkpoint on every clade founder
    and joined both counts estimates onto it, so what is left here is the
    assay and the drawing.

    **GISAID**: reads the scored table and mat-pcp's founders and censor
    tables. The DMS measurements are published data. What it writes —
    fitness effects, selection factors and correlations — is publishable.
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
    import panel_charts as charts
    import panel_spec

    from lib import dms_dataset, dms_homologs

    SELECTION = panel_spec.DMS_SELECTION
    HOMOLOGS = tuple(dms_homologs())
    CLADES = {homolog: dms_dataset(homolog).clade for homolog in HOMOLOGS}

    # Required, not defaulted, as the other panel notebooks have it: a
    # fallback spelling of the path is a second spelling of a fact
    # `workflow/figures.smk` already states.
    _args = mo.cli_args()
    TABLE_PATH = Path(str(_args["table"]))
    return (
        CLADES,
        HOMOLOGS,
        SELECTION,
        TABLE_PATH,
        charts,
        dms,
        panel_spec,
        panels,
    )


@app.cell
def _(CLADES, HOMOLOGS, TABLE_PATH, dms, panel_spec, panels):
    table = panels.read_scored_table(
        TABLE_PATH,
        models=[panel_spec.FIGURE_MODEL],
        clades=sorted(CLADES.values()),
        columns=dms.TABLE_COLUMNS,
    )
    frames = dms.dms_frames(table, HOMOLOGS)
    correlations = dms.dms_correlations(frames)
    return correlations, frames


@app.cell
def _(HOMOLOGS, correlations, dms, mo, panel_spec):
    mo.md(
        f"Scored with **{panel_spec.FIGURE_MODEL}**, and **unclipped**: every "
        f"predictor is plotted and correlated at the value the model or the "
        f"tree reports, and the assay is never bounded either. The source "
        f"experiment drew this figure inside [-8, 6]; see `dms.CLIP_LOWER`. "
        f"Pearson *r* against cell entry:\n\n"
        + correlations.set_index(["min_count", "homolog", "method"])
        .loc[:, ["pearson", "spearman", "n"]]
        .round(3)
        .reindex(
            [
                (min_count, homolog, method)
                for min_count in dms.MIN_COUNTS
                for homolog in HOMOLOGS
                for _column, method in dms.METHODS
            ]
        )
        .to_markdown()
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Panels A and C — one homolog's clouds

    Three predictors side by side against the same assay. Both rows share one
    pair of axis domains, taken from the wider top-row subset, so a point sits
    in the same place in both and the only visible difference is which points
    survive the threshold.

    No diagonal and no fitted line, as upstream: the axes are a log selection
    factor and a cell-entry measurement, so there is no reason for a point to
    sit anywhere in particular, and the number the panel reports is the
    correlation in its corner.
    """)
    return


@app.cell
def _(CLADES, SELECTION, charts, correlations, dms, frames, panel_spec, panels):
    SCATTER_HOMOLOG = panel_spec.SCATTER_HOMOLOG
    _widest = frames[(SCATTER_HOMOLOG, min(dms.MIN_COUNTS))]
    X_DOMAIN = charts.dms_group_domain(
        _widest, [column for column, _label in dms.METHODS]
    )
    _y = charts.dms_padded_domain(_widest["dms_cell_entry"])
    # Extra headroom at the top only, so the in-plot `r` sits over empty
    # canvas instead of over the dense band of near-neutral mutations at
    # DMS ~ 0.
    Y_DOMAIN = [_y[0], _y[1] + charts.DMS_Y_HEADROOM * (_y[1] - _y[0])]

    scatter_panels = []
    for _stem, _min_count in zip(panel_spec.DMS_SCATTER_PANELS, dms.MIN_COUNTS):
        _frame = frames[(SCATTER_HOMOLOG, _min_count)]
        # The `r` each panel prints, read from the frame the bars and the CSVs
        # are drawn from rather than computed a second time inside the chart.
        _rows = correlations[
            (correlations["homolog"] == SCATTER_HOMOLOG)
            & (correlations["min_count"] == _min_count)
        ]
        _coefficients = dict(zip(_rows["column"], _rows["pearson"]))
        # The rung the figure lays out heads itself with the homolog, in
        # black; the other keeps the count floor in its row colour, which is
        # what a panel read on its own needs. `panel_spec.DMS_FIGURE_ROWS`
        # is where the split is recorded.
        _in_figure = _min_count == panel_spec.DMS_FIGURE_MIN_COUNT
        scatter_panels.append(
            charts.save_panel(
                charts.dms_scatter_row(
                    _frame,
                    min_count=_min_count,
                    clade=CLADES[SCATTER_HOMOLOG],
                    methods=dms.METHODS,
                    x_domain=X_DOMAIN,
                    y_domain=Y_DOMAIN,
                    coefficients=_coefficients,
                    heading=(
                        charts.dms_homolog_heading(
                            SCATTER_HOMOLOG, CLADES[SCATTER_HOMOLOG], len(_frame)
                        )
                        if _in_figure
                        else None
                    ),
                    heading_color=charts.INK if _in_figure else None,
                ),
                _stem,
                selection=SELECTION,
            )
        )
        panels.write_panel_csv(
            _frame[
                [
                    "homolog",
                    "clade",
                    "aa_mutation",
                    "aa_site",
                    "parent_aa",
                    "mutant_aa",
                    "dms_cell_entry",
                    *[column for column, _label in dms.METHODS],
                ]
            ].round(4),
            _stem,
            selection=SELECTION,
            spec=dms.dms_spec(_min_count),
        )
    return (scatter_panels,)


@app.cell
def _(scatter_panels):
    scatter_panels[0]
    return


@app.cell
def _(scatter_panels):
    scatter_panels[1]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Panels B and D — the same coefficient, for all four homologs

    The clouds report *r* in each panel for one homolog; these collect it for
    all four, so the comparison that matters — how much each method loses
    when the threshold drops — is one figure rather than four dropdown
    settings held in the reader's head.

    Both rows share one bar axis, so a bar in the lower row is directly
    comparable to the bar above it.
    """)
    return


@app.cell
def _(CLADES, HOMOLOGS, SELECTION, charts, correlations, dms, panel_spec, panels):
    BAR_DOMAIN = [0.0, float(correlations["pearson"].max()) * charts.DMS_BAR_HEADROOM]

    bar_panels = []
    for _stem, _min_count in zip(panel_spec.DMS_BAR_PANELS, dms.MIN_COUNTS):
        bar_panels.append(
            charts.save_panel(
                charts.dms_bar_row(
                    correlations,
                    min_count=_min_count,
                    homologs=HOMOLOGS,
                    clades=CLADES,
                    methods=dms.METHODS,
                    y_domain=BAR_DOMAIN,
                    # Each group already carries its homolog and its `n`, and
                    # the panel the figure lays out has no second row to be
                    # told apart from, so it goes bare.
                    heading=(
                        "" if _min_count == panel_spec.DMS_FIGURE_MIN_COUNT else None
                    ),
                ),
                _stem,
                selection=SELECTION,
            )
        )
        panels.write_panel_csv(
            correlations[correlations["min_count"] == _min_count].round(4),
            _stem,
            selection=SELECTION,
            spec=dms.dms_spec(_min_count),
        )
    return (bar_panels,)


@app.cell
def _(bar_panels):
    bar_panels[0]
    return


@app.cell
def _(bar_panels):
    bar_panels[1]
    return


if __name__ == "__main__":
    app.run()
