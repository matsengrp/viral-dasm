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
    # `fig:subsampling` panels C and D

    What thinning the tree costs each approach. For every clade, the
    full-depth counts estimate is the target; each arm's model — `dasm-85k`
    trained on that arm's corpus — and each arm's own counts refit are the two
    predictors. **Panel C** is the clouds, for Delta. **Panel D** is the two
    statistics, for the five deepest clades.

    **The mutation set is fixed on the full-depth arm.** Filtering each arm on
    its own counts would score a thinned arm only over the mutations it still
    happened to observe, which is the subset it is best at, and every curve
    would be drawn over a different population. The count filter runs once, at
    full depth, and every arm is scored over exactly those mutations — so
    `n` is identical down each column of the table below.

    **The full-depth counts cell is the target against itself**, so it sits at
    r = 1 and slope = 1 by construction. It is the baseline, not a result.

    **GISAID**: reads the scored table and mat-pcp's per-clade fitness tables;
    writes publishable fitness effects, selection factors and correlations.
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
    import panel_charts as charts

    import pandas as pd

    from lib import CORPUS_NAMES, mat_pcp_path

    # Named here, as literals, rather than reached through the frame builders:
    # `tests/test_rule_inputs.py` reads the keys a rule's entry point spells
    # out and checks the rule declares them, and a key reached only through a
    # helper is one that check cannot see.
    AAMUT_DIR = mat_pcp_path("aamut_fitness_dir")
    SELECTION = panels.SUBSAMPLING_SELECTION
    ARMS = tuple(CORPUS_NAMES)

    # Required, not defaulted, as `agreement_ladders.py` has it: a fallback
    # spelling of either path is a second spelling of a fact
    # `workflow/figures.smk` already states.
    _args = mo.cli_args()
    TABLE_PATH = Path(str(_args["table"]))
    AUDIT_PATH = Path(str(_args["audit"]))
    return AAMUT_DIR, ARMS, AUDIT_PATH, SELECTION, TABLE_PATH, charts, panels, pd


@app.cell
def _(AAMUT_DIR, AUDIT_PATH, TABLE_PATH, panels):
    frames = panels.ladder_frames(
        TABLE_PATH,
        AAMUT_DIR,
        reverting_by_clade=panels.reverting_sites_by_clade(AUDIT_PATH),
    )
    metrics = panels.ladder_metrics(frames)
    return frames, metrics


@app.cell
def _(metrics, mo, panels):
    _ranked = (
        metrics[
            (metrics["method"] == "DASM") & (metrics["arm"] == panels.FULL_DEPTH.name)
        ]
        .sort_values("n_mutations", ascending=False)["clade"]
        .tolist()
    )
    assert tuple(_ranked) == tuple(panels.SUBSAMPLING_LADDER_CLADES), (
        f"ranked on this figure's own mutation counts the clades are "
        f"{_ranked}, not {list(panels.SUBSAMPLING_LADDER_CLADES)}, which is "
        f"the order `main.tex` names from the sequence counts. The two "
        f"rankings have come apart; say which the panel uses."
    )
    mo.md(
        "Clades ranked on this figure's own `n_mutations`, which is the order "
        f"the panel draws them in: {', '.join(_ranked)}."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Panel C — the clouds, for Delta

    One row per approach, one column per sampling depth, full depth leftmost
    so that reading right is reading the thinning get harsher. The target is
    on x, which is the axis `within_clade_metrics` regresses on, so the drawn
    slope and the labelled one are the same number. The red line is drawn from
    that reported slope rather than fitted to what is plotted, so no point
    budget can make the line and the label disagree.
    """)
    return


@app.cell
def _(ARMS, SELECTION, charts, frames, metrics, panels, pd):
    CLADE = panels.SUBSAMPLING_SCATTER_CLADE

    panel_c_frames = {
        (arm, method): frames[(arm, CLADE, method)]
        for arm in ARMS
        for method in panels.LADDER_METHODS
    }
    panel_c = charts.save_panel(
        charts.arm_scatter_grid(
            panel_c_frames,
            metrics[metrics["clade"] == CLADE],
            arms=ARMS,
            methods=panels.LADDER_METHODS,
            x_field=panels.SUBSAMPLING_SPEC.fitness_metric,
            y_field="log_selection_factor",
            clip_lower=panels.SUBSAMPLING_CLIP_LOWER,
            clade=CLADE,
            point_budget=charts.SHIFT_POINT_BUDGET // 16,
        ),
        panels.LADDER_ARM_PANELS[0],
        selection=SELECTION,
    )

    panel_c_table = pd.concat(
        [
            frame.assign(clade=CLADE, arm=arm, method=method)
            for (arm, method), frame in panel_c_frames.items()
        ],
        ignore_index=True,
    )[
        [
            "clade",
            "arm",
            "method",
            "aa_mutation",
            panels.SUBSAMPLING_SPEC.fitness_metric,
            "log_selection_factor",
        ]
    ]
    # Rounded for the same reason panel B's shares are: a log fitness effect
    # is not meaningful past the fourth decimal, and the float's own sixteen
    # digits record how a pandas version chose to print it. 2.5 MB of tracked
    # diff-on-noise becomes a few hundred KB.
    panels.write_panel_csv(
        panel_c_table.round(4),
        panels.LADDER_ARM_PANELS[0],
        selection=SELECTION,
        spec=None,
    )
    return (panel_c,)


@app.cell
def _(panel_c):
    panel_c
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Panel D — the two statistics, across the five deepest clades

    One row per clade, r on the left and the OLS slope on the right. The slope
    column is on log2, because a slope is a ratio and a linear axis makes
    halving look smaller than doubling.

    r is **not** floored at zero here, unlike `fig:agreement`'s ladder: how far
    the counts side falls at the shallowest cap is this panel's claim.
    """)
    return


@app.cell
def _(ARMS, SELECTION, charts, metrics, mo, panels):
    panel_d = charts.save_panel(
        charts.arm_ladder_grid(
            metrics,
            clades=panels.SUBSAMPLING_LADDER_CLADES,
            arms=ARMS,
            methods=panels.LADDER_METHODS,
        ),
        panels.LADDER_ARM_PANELS[1],
        selection=SELECTION,
    )
    panels.write_panel_csv(
        metrics.round(4), panels.LADDER_ARM_PANELS[1], selection=SELECTION, spec=None
    )

    mo.md(
        "Pearson r and OLS slope by arm, clade and approach:\n\n"
        + metrics.round(3).to_markdown(index=False)
    )
    return (panel_d,)


@app.cell
def _(panel_d):
    panel_d
    return


if __name__ == "__main__":
    app.run()
