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
    # `fig:within_clade_shifts` panels B, D and F

    *Each labelled mutation's effect in each of two genetic backgrounds within
    the clade, estimated by both approaches.*

    One row per clade, one sub-panel per candidate mutation, two x positions:
    the clade's two genetic backgrounds. The row sits beside that clade's
    scatter from `notebooks/within_clade_outliers.py`, which establishes the
    subjects -- the mutations the two approaches disagree about on the clade
    founder. This panel asks what those mutations do once the clade is
    subdivided.

    **The backgrounds come from the tree, not from the model.** `rule
    clade_backgrounds` ranks substitutions by the share of the clade's branch
    mass carrying them and splits on the most abundant one plus whatever
    travels with it. Nothing the DASM produces reaches that choice, the pools
    drawn from it, or the p-value, which is what makes this panel a test of
    the model rather than a restatement of it.

    **Backgrounds are named by residue, not by substitution.** `R346, K356,
    F490` against `T346, T356, S490`. Both sides of a contrast are defined by
    their residues at the same sites, so naming them this way makes the two
    ticks a like-for-like pair; naming them by substitution leaves the founder
    side, which substitutes nothing, with no name beyond "none".

    **Two estimates per background.** The counts estimate is
    `log(actual / expected)` over the branches in that background, with
    `expected` built from a `T` refitted on that background's own synonymous
    events. The DASM estimate is the **pool average**: the mean log selection
    factor over the distinct parent haplotypes the background actually
    contains, weighted by each one's synonymous count. That weighting is what
    makes the two comparable -- the counts side divides a sum of events by a
    sum of expected counts, and matching it needs an arithmetic mean of the
    selection factors over the same weights, not a mean of their logs.

    **The pseudocount is withheld.** In a sub-pool the expected counts run
    0.01 to 1, so `log((a + 1/2)/(e + 1/2))` collapses onto a function of the
    event count alone with the expected count edited out. It is applied only
    where a mutation is never observed on a background, which is the one case
    the plain ratio cannot express.

    **The p-value is two-sided.** It asks whether a counts shift this large is
    likely to arise by chance -- not whether the two approaches agree. Branches
    are reassigned between the two backgrounds at random with each side's size
    held fixed, both counts are recomputed from the branches drawn, and the
    statistic is ranked on its distance from zero. It is floored at
    `1 / (draws + 1)`, so a run that reached the floor is reported as an
    inequality rather than as a number the draw count cannot support.

    **GISAID**: reads pooled counts and scores from the artifact root; writes
    effects, shifts and p-values, which are publishable, and names a
    background by its residues rather than by any sequence.
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

    # Required, not defaulted, as the other panel notebooks are: a fallback
    # spelling here is a second spelling of what `workflow/figures.smk`
    # already states and passes in.
    _args = mo.cli_args()
    CELLS_PATH = Path(str(_args["cells"]))
    SHIFTS_PATH = Path(str(_args["shifts"]))
    SELECTION = str(_args.get("selection") or panels.DEFAULT_SELECTION)
    return CELLS_PATH, SELECTION, SHIFTS_PATH, charts, panels, pd


@app.cell
def _(CELLS_PATH, SELECTION, SHIFTS_PATH, mo, panels, pd):
    CLADES = list(panels.CANDIDATE_CLADES)
    #: Only the mutations the figure draws. The designed null and the
    #: held-back nomination are measured by `rule pool_counts` and kept off
    #: the panels, which is what `off_figure_candidates` is for -- they are
    #: excluded here rather than never computed, so the null stays a null.
    DRAWN = {(clade, mutation) for clade, mutation in panels.figure_candidates()}

    CELLS = pd.read_parquet(CELLS_PATH)
    SHIFTS = pd.read_parquet(SHIFTS_PATH)

    # The focal cell of each candidate: the row whose site and residue are the
    # candidate's own. Every other cell of the pool rides along in the
    # producer's table for the shift statistics, and none of them is drawn.
    _site = CELLS["candidate"].map(lambda m: panels.parse_mutation(m)[1])
    _mutant = CELLS["candidate"].map(lambda m: panels.parse_mutation(m)[2])
    _drawn = pd.Series(
        [
            (clade, mutation) in DRAWN
            for clade, mutation in zip(CELLS["cluster"], CELLS["candidate"])
        ],
        index=CELLS.index,
    )
    FOCAL = CELLS[
        (CELLS["aa_site"] == _site) & (CELLS["mutant_aa"] == _mutant) & _drawn
    ].copy()

    # `pool` is 0 for the founder side and 1 for the derived side, which is
    # the order the x axis reads and the order the shift is signed in.
    FOCAL["background_order"] = FOCAL["pool"].astype(int)

    _p = SHIFTS.drop_duplicates(["candidate", "p_value"]).set_index("candidate")
    FOCAL["p_label"] = [
        (
            f"p < {1 / 10001:.0e}".replace("e-0", "e-")
            if bool(_p.loc[candidate, "p_at_floor"])
            else f"p = {float(_p.loc[candidate, 'p_value']):.2g}"
        )
        for candidate in FOCAL["candidate"]
    ]
    mo.md(
        f"Selection **{SELECTION}** → clades `{'`, `'.join(CLADES)}`, "
        f"{FOCAL['candidate'].nunique()} candidates drawn of "
        f"{CELLS['candidate'].nunique()} measured."
    )
    return CLADES, FOCAL


@app.cell
def _(CLADES, FOCAL, charts):
    # One y domain for every sub-panel of every row, so a steep line means a
    # large shift rather than a narrow axis. Computed over all three clades
    # for that reason, with headroom at the top for the p-value.
    Y_DOMAIN = charts.background_y_domain(FOCAL)

    # Every row is the same number of slots wide: the widest clade's
    # candidate count, plus one slot for the legend. Rows are stacked
    # left-justified in the composed figure, so a row that sized itself to
    # its own two candidates would put its sub-panels at a different x and a
    # different width from the row above it, and three rows that do not share
    # a column grid do not read as one exhibit.
    _width = {_c: FOCAL[FOCAL["cluster"] == _c]["candidate"].nunique() for _c in CLADES}
    SLOTS = max(_width.values()) + 1

    # The order the rows are *drawn* in, which is not the order they are
    # stacked in: `h_params` is measured from the first row drawn, and only
    # the widest clade's margins are measured from a full set of tick labels.
    # Stacking order stays `CLADES`, since that is what pairs a row with the
    # scatter beside it.
    DRAW_ORDER = sorted(CLADES, key=lambda clade: -_width[clade])
    return DRAW_ORDER, SLOTS, Y_DOMAIN


@app.cell
def _(CLADES, DRAW_ORDER, FOCAL, SELECTION, SLOTS, Y_DOMAIN, charts, mo, panels):
    PANELS = {}
    # `h_params` is taken from the first row drawn and imposed on the rest.
    # `tight_layout` sizes a figure's margins from its own tick labels, so a
    # row whose surplus slots carry none comes out with wider sub-panels than
    # the first -- equal slot counts alone do not line the rows up. The first
    # row is the widest clade's, so it is the only one whose margins are
    # measured from a full set of labels.
    _h_params = None
    _stems = dict(zip(CLADES, panels.BACKGROUND_PANELS))
    for _clade in DRAW_ORDER:
        _stem = _stems[_clade]
        _row = FOCAL[FOCAL["cluster"] == _clade]
        if _row.empty:
            raise ValueError(
                f"{_clade} has no drawn candidate, so its row of the figure "
                f"would be blank while the scatter beside it names subjects."
            )
        _figure = charts.background_panel(
            _row,
            _clade,
            y_domain=Y_DOMAIN,
            columns=SLOTS,
            h_params=_h_params,
        )
        if _h_params is None:
            _pars = _figure.subplotpars
            _h_params = (_pars.left, _pars.right, _pars.wspace)
        PANELS[_clade] = charts.save_mpl_panel(_figure, _stem, selection=SELECTION)
        panels.write_pool_csv(
            _row[
                [
                    "cluster",
                    "candidate",
                    "background",
                    "background_order",
                    "aa_site",
                    "clade_founder_aa",
                    "mutant_aa",
                    "n_pcps",
                    "synonymous_observed",
                    "actual",
                    "expected",
                    "counts_effect",
                    "se",
                    "dasm_effect",
                    "p_label",
                ]
            ].rename(columns={"cluster": "clade"}),
            _stem,
            selection=SELECTION,
        )
    mo.vstack([item for _c in CLADES for item in (mo.md(f"### {_c}"), PANELS[_c])])
    return


if __name__ == "__main__":
    app.run()
