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
    # `fig:l455f` — L455F on two backgrounds, clade by clade

    *L455F's fitness effect on the parents carrying the clade founder's 456F
    and on the parents carrying 456L, estimated by both approaches, in each
    XBB clade where both pools carry evidence.*

    The same renderer as `fig:within_clade_shifts`' panels B, D and F, turned
    ninety degrees: there a row is one clade and a sub-panel is one mutation,
    here the row is one mutation and a sub-panel is one clade. That is the
    `group_column` argument, and it is an argument rather than a relabelled
    column because the default would draw every clade's row as a single
    sub-panel of points alternating between two backgrounds -- a wrong
    picture that looks right.

    **The split is specified by hand, at site 456 and nothing else.** This is
    a deviation from `fig:within_clade_shifts`, whose splits come from branch
    frequencies through `rule clade_backgrounds`, and it is the reason this
    figure has its own measuring script. Measured rather than assumed, by
    running that rule's chooser unaided on the four clades that have two
    pools:

    | clade | what the unaided chooser picks | what it drags in |
    |---|---|---|
    | 22F | **S486P** — not site 456 at all | — |
    | 23A | F456L | **L455F itself (59%)**, A475V (53%) |
    | 23B | F456L | nothing |
    | 23D | F456L | K478R (52%), A701V (57%) |

    So for 23A the frequency-derived background would contain the very
    mutation under test, which leaves the cell unscoreable; for 22F it would
    not be a 456 contrast at all; and for 23B the hand specification confirms
    what the chooser already finds rather than overriding it.

    **Two clades of the roster are measured and not drawn, for two
    different reasons.** 23F has no founder-side pool -- EG.5.1's founder
    already carries 456L, so a founder pool and a derived pool would be the
    same branches -- so `scripts/l455f_pools.py`, upstream of this notebook,
    measures it on its derived pool alone, and a sub-panel for it would be a
    single point with nothing to compare it to. 23E has two pools and its
    456L pool holds one event against an expected 0.06, below the cutoff the
    rest of the figure is drawn at. Both are in the panel CSV.

    **The whiskers.** The purple triangle is the pool average, and the
    whisker on it is the 5th to 95th percentile of the same haplotypes'
    effects under the same weights -- how far the model's estimate moves
    between the parents of one pool. It is *not* an uncertainty: it does not
    shrink as the pool grows, and a pool of a thousand parents that agree
    will show a shorter whisker than a pool of fifty that do not.

    **The three arms, and what makes them comparable.** The per-subset
    counts estimate is `log(actual / expected)` over the branches in that
    pool. The DASM estimate is the pool average: the mean log selection
    factor over the distinct parent haplotypes the pool contains, weighted by
    each one's synonymous count. Matching the counts side -- a sum of events
    over a sum of expected counts -- needs an arithmetic mean of the selection
    factors, not a mean of their logs.

    The **full-tree counts** arm is the same statistic over every branch of
    every clade, read from the clade-pooled grid: one number, drawn flat
    across both backgrounds of every sub-panel. It is what the counts
    approach says about this mutation when it is handed no split at all, and
    it is in the panel because a reader is entitled to ask that -- the
    per-subset arm is given the contrast the figure is about, and this one
    shows what is there without it.

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
    import panel_spec as spec

    import pandas as pd

    # Required, not defaulted, as the other panel notebooks are: a fallback
    # spelling here is a second spelling of what `workflow/figures.smk`
    # already states and passes in.
    _args = mo.cli_args()
    CELLS_PATH = Path(str(_args["cells"]))
    SHIFTS_PATH = Path(str(_args["shifts"]))
    POOLED_PATH = Path(str(_args["pooled"]))
    SELECTION = str(_args.get("selection") or panels.DEFAULT_SELECTION)
    return (
        CELLS_PATH,
        POOLED_PATH,
        SELECTION,
        SHIFTS_PATH,
        charts,
        panels,
        pd,
        spec,
    )


@app.cell
def _(CELLS_PATH, POOLED_PATH, SELECTION, SHIFTS_PATH, mo, panels, pd, spec):
    CELLS = pd.read_parquet(CELLS_PATH)
    SHIFTS = pd.read_parquet(SHIFTS_PATH)

    # The focal cell of the candidate: the row whose site and residue are the
    # mutation's own. Every other cell of the pool rides along in the
    # producer's table for the shift statistics, and none of them is drawn.
    _, _site, _mutant = panels.parse_mutation(spec.L455F_MUTATION)
    FOCAL = CELLS[
        (CELLS["candidate"] == spec.L455F_MUTATION)
        & (CELLS["aa_site"] == _site)
        & (CELLS["mutant_aa"] == _mutant)
    ].copy()

    # `pool` is 0 for the founder side and 1 for the derived side, which is
    # the order the x axis reads and the order the shift is signed in.
    FOCAL["background_order"] = FOCAL["pool"].astype(int)

    # The panel's own evidence gate, at the cutoff the rest of the figure is
    # drawn at rather than at `MIN_POOL_EVIDENCE`, which is 1. Both pools of a
    # clade must clear it: a contrast is only as good as its thinner side.
    # Taken from the producer's function rather than written as a comparison
    # here, so the panel and the pipeline cannot come to mean different things
    # by "enough evidence".
    FOCAL["panel_gate"] = panels.passes_evidence_gate(
        FOCAL["actual"], FOCAL["expected"], minimum=panels.PANEL_MIN_COUNT
    )
    # Two conditions, and both are about the contrast rather than about a
    # point. Every pool a clade has must clear the cutoff -- a contrast is
    # only as good as its thinner side -- and the clade must have both pools.
    # 23F is the clade that fails the second: EG.5.1's founder already carries
    # 456L, so a founder pool and a derived pool would be the same branches.
    # It was drawn as a named tick with no point on the F456 side, which asked
    # the reader to carry an absence along a row whose every other sub-panel
    # is a slope. It is still measured and still in this notebook's CSV; it is
    # not drawn.
    _clears = FOCAL.groupby("cluster")["panel_gate"].all()
    _pools = FOCAL.groupby("cluster")["pool"].nunique()
    DRAWN = sorted(
        clade
        for clade in spec.L455F_CLADES
        if _clears.get(clade, False) and int(_pools.get(clade, 0)) == 2
    )

    def _gated_why(clade: str) -> str:
        if clade not in _clears.index:
            return "not measured -- see the pools script"
        if not bool(_clears[clade]):
            return "below the cutoff"
        return (
            f"{int(_pools[clade])} background measured, not two -- "
            f"no contrast to draw"
        )

    GATED = {
        clade: _gated_why(clade)
        for clade in spec.L455F_CLADES
        if clade not in DRAWN
    }
    if not DRAWN:
        raise ValueError(
            f"no clade of {list(spec.L455F_CLADES)} has two pools that both "
            f"clear {panels.PANEL_MIN_COUNT} actual or expected counts, so "
            f"the panel would be blank."
        )
    FOCAL = FOCAL[FOCAL["cluster"].isin(DRAWN)]

    # The counts approach over the whole tree: one number, joined onto every
    # row, so it is drawn flat across both backgrounds of every sub-panel.
    # What it is there for is the question a reader of the per-subset arm is
    # entitled to ask -- what does this approach say when it is not handed the
    # split? -- and the answer is a level, not a slope.
    FOCAL["full_tree_effect"] = panels.pooled_counts_estimate(
        POOLED_PATH, spec.L455F_MUTATION
    )

    # One p-value per clade, not per candidate: the candidate is the same
    # mutation in every sub-panel here, and keying on it would collapse the
    # clades onto one row. A clade with one pool has no shift and so no test;
    # the panel draws no p-value in any case, and the CSV says which it is.
    _p = SHIFTS[SHIFTS["candidate"] == spec.L455F_MUTATION]
    _p = _p.drop_duplicates(["cluster", "p_value"]).set_index("cluster")

    def _p_label(clade: str) -> str:
        if clade not in _p.index:
            return "n.a. -- one pool, no shift to test"
        if bool(_p.loc[clade, "p_at_floor"]):
            return f"p < {1 / (panels.DEFAULT_PERMUTATIONS + 1):.0e}".replace(
                "e-0", "e-"
            )
        return f"p = {float(_p.loc[clade, 'p_value']):.2g}"

    FOCAL["p_label"] = [_p_label(clade) for clade in FOCAL["cluster"]]
    mo.md(
        f"Selection **{SELECTION}**. Drawn: `{'`, `'.join(DRAWN)}`. "
        + (
            "Not drawn: "
            + ", ".join(f"**{clade}** ({why})" for clade, why in GATED.items())
            if GATED
            else "Every clade of the roster is drawn."
        )
    )
    return DRAWN, FOCAL


@app.cell
def _(DRAWN, FOCAL, SELECTION, charts, mo, panels, spec):
    # One y domain across every sub-panel, so a steep line means a large shift
    # rather than a narrow axis -- the claim the panel makes.
    # The whiskers' ends as well as the points, so an interval reaching past
    # its dot is drawn whole rather than clipped at the axis.
    _y_domain = charts.background_y_domain(
        FOCAL,
        extra_columns=("dasm_low", "dasm_high"),
        series=charts.L455F_BACKGROUND_SERIES,
    )

    # One slot per clade, plus one for the legend, exactly as a background row
    # is sized in `fig:within_clade_shifts`. There is only one row here, so
    # nothing has to line up against a row above it and no `h_params` are
    # carried over.
    _figure = charts.background_panel(
        FOCAL,
        # The sub-panels are clades, so no one clade names the row. What the
        # row has in common is the mutation, which is what this argument says
        # and what `value_label` below then puts on the value axis.
        spec.L455F_MUTATION,
        y_domain=_y_domain,
        # Two spare slots rather than one. The legend keys here name what
        # each approach was applied to, which is several words rather than
        # one, and a key wider than its slot runs off the panel's right edge.
        columns=len(DRAWN) + 2,
        # What the sub-panel titles cannot say: that these are XBB lineages,
        # and that the estimates come from the tree -- which is the contrast
        # with the measurements in panel A above.
        figure_title=spec.L455F_BACKGROUND_TITLE,
        # Taller by the height of that heading, so the plot boxes stay the
        # height of panel A's.
        row_h_in=spec.L455F_BACKGROUND_ROW_H_IN,
        group_column="cluster",
        # Three arms rather than the two the within-clade rows draw, in the
        # order the key reads them: the DASM, then the counts approach at its
        # two resolutions. See `charts.L455F_BACKGROUND_SERIES` on why this is
        # its own tuple and not a third entry in the shared one.
        series=charts.L455F_BACKGROUND_SERIES,
        # Clade order, not steepest-first. These sub-panels are clades, which
        # have an order of their own that also runs roughly in time, and the
        # one-pool clade has no shift to be sorted by. Titled as the tree's
        # own metadata names them, so the panel and the paper's prose agree.
        members=[(clade, spec.L455F_CLADE_NAMES[clade]) for clade in DRAWN],
        # Twice the diameter of the published rows' dots. `markersize` is a
        # diameter in points, so a factor of two here is a factor of two on
        # the page.
        marker_size=14,
        # The spread of the DASM's estimate over the pool's own parent
        # haplotypes, weighted as the pool average is. Only the DASM side has
        # one: the counts estimate is a ratio of sums over the whole pool and
        # has no per-parent value to spread.
        spread_columns={"DASM": ("dasm_low", "dasm_high")},
        # One line. The two-line form exists because
        # `fig:within_clade_shifts`' label names a clade as well as a
        # quantity and is too long for its panels; this one is short enough
        # to set on one line, which reads better beside a single row.
        value_label=f"fitness effect of {spec.L455F_MUTATION}",
        # Named by residue alone rather than as "the clade founder's F456".
        # Every clade now drawn does carry F456 at its founder, 23F having
        # been dropped for having no second background, but the x axis names
        # what membership of a pool requires and not what the founder happens
        # to be -- which is also what the shared `BACKGROUND_X_TITLE` says.
        x_gloss=(
            f"(left: parents carrying F{spec.L455F_MARKER_SITE}; "
            f"right: parents carrying "
            f"{spec.L455F_MARKER_AA}{spec.L455F_MARKER_SITE})"
        ),
    )
    PANEL = charts.save_mpl_panel(
        _figure, spec.L455F_BACKGROUND_PANEL, selection=SELECTION
    )
    panels.write_pool_csv(
        FOCAL[
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
                "full_tree_effect",
                "se",
                "dasm_effect",
                "dasm_low",
                "dasm_high",
                "p_label",
            ]
        ].rename(columns={"cluster": "clade"}),
        spec.L455F_BACKGROUND_PANEL,
        selection=SELECTION,
    )
    mo.vstack([mo.md(f"### {spec.L455F_MUTATION} — {len(DRAWN)} clades"), PANEL])
    return


if __name__ == "__main__":
    app.run()
