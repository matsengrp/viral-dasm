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
    # Table S1: the mutations the two approaches most disagree about

    *Mutations to a clade-founder spike sequence for which the absolute
    difference between the counts-based and DASM estimates exceeds 2, among
    mutations whose actual or expected count in the clade reached 50, ordered
    by that difference.*

    **Clipped at $-4$, as the agreement panels are.** Below that bound the
    counts-based estimate is reporting its own limit of detection rather than
    an effect, so a difference measured there is a difference between a
    measurement and a floor. Clipping both estimates first is what makes this
    ranking the same comparison the figure draws, at a higher count cutoff.

    **The adjusted difference.** The counts approach makes two approximations
    that are not assumptions the DASM makes, and both can open a gap on their
    own.

    It adds a pseudocount $P = 0.5$ to the actual and expected counts alike.
    Where the expected count is far below one, that $P$ dominates the
    denominator and pins the estimate near $\log(2a)$ however beneficial the
    mutation is.

    And it takes the opportunity for a mutation to be the clade's fitted rate
    scaled to the whole clade, while counting the mutation only on branches
    whose parent still carries the founder residue. For a mutation that has
    risen to high frequency those are different branch sets: most of the clade
    already carries it and cannot acquire it again, so the opportunity left is
    a fraction $\phi$ of what the expectation assumes.

    *Adjusted* recomputes the difference against $\log(a / e\phi)$, dropping
    the pseudocount and scaling the expectation to the opportunity that was
    actually available. What remains is a disagreement neither approximation
    explains.

    **GISAID**: reads the cluster-only scored table; writes counts and
    estimated effects, which are publishable, and no sequence.
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

    import panels

    import numpy as np
    import pandas as pd

    _args = mo.cli_args()
    TABLE_PATH = Path(str(_args["table"]))
    SELECTION = str(_args.get("selection") or panels.DEFAULT_SELECTION)
    MODELS = panels.selection_models(SELECTION)
    TABLE_MODEL = MODELS[0]
    ARM = panels.ARM
    return ARM, MODELS, SELECTION, TABLE_MODEL, TABLE_PATH, np, panels, pd


@app.cell
def _(ARM, MODELS, TABLE_MODEL, TABLE_PATH, mo, np, panels, pd):
    # The clades the agreement figure scores, read from its own tracked
    # table rather than restated: this ranking and that figure are the same
    # comparison at a higher cutoff, and a second roster could disagree.
    ROSTER = sorted(
        pd.read_csv(
            panels.panel_csv_path(
                panels.DEFAULT_SELECTION, "panel_c_within_clade_ladder"
            )
        )["clade"].unique()
    )
    SPEC = panels.FilterSpec(min_count=panels.OUTLIER_MIN_COUNT)
    _table = panels.read_scored_table(TABLE_PATH, models=MODELS, clades=ROSTER)
    _frames = panels.base_frames(_table, spec=SPEC, models=MODELS, clades=ROSTER)
    COMPARED = pd.concat(
        [frame.assign(clade=clade) for (_, clade), frame in _frames.items()],
        ignore_index=True,
    )
    COMPARED = COMPARED[COMPARED["model"] == TABLE_MODEL]

    _difference = COMPARED["log_selection_factor"] - COMPARED["delta_fitness_ratio"]
    RANKED = (
        COMPARED.assign(difference=_difference.abs())
        .sort_values("difference", ascending=False)
        .reset_index(drop=True)
    )
    OVER_CUT = RANKED[RANKED["difference"] > panels.OUTLIER_MIN_DIFFERENCE].copy()

    # Only for the rows the table prints. The corpus is 112 MB of parent
    # sequences and the question is per (clade, site), so asking it of the
    # whole comparison would cost minutes to answer for rows nobody reads.
    _wanted: dict[str, list[tuple[int, str]]] = {}
    for _, _row in OVER_CUT.iterrows():
        _wanted.setdefault(_row["clade"], []).append(
            (int(_row["aa_site"]), _row["parent_aa"])
        )
    _phi = panels.founder_background_fraction(_wanted, panels.corpus(ARM).source_pcp)
    OVER_CUT["founder_opportunity"] = [
        _phi[(clade, int(site), residue)]
        for clade, site, residue in zip(
            OVER_CUT["clade"], OVER_CUT["aa_site"], OVER_CUT["parent_aa"]
        )
    ]
    with np.errstate(divide="ignore", invalid="ignore"):
        OVER_CUT["adjusted"] = (
            OVER_CUT["log_selection_factor"]
            - np.log(
                OVER_CUT["actual_count"]
                / (OVER_CUT["predicted_count"] * OVER_CUT["founder_opportunity"])
            )
        ).abs()
    mo.md(
        f"Model **{TABLE_MODEL}**, {len(ROSTER)} clades, "
        f"**{len(COMPARED):,}** mutations at a cutoff of "
        f"{panels.OUTLIER_MIN_COUNT:.0f} actual or expected counts. "
        f"**{len(OVER_CUT)}** of them differ by more than "
        f"{panels.OUTLIER_MIN_DIFFERENCE}."
    )
    return COMPARED, OVER_CUT, RANKED, ROSTER, SPEC


@app.cell
def _(OVER_CUT, SELECTION, SPEC, mo, panels, pd):
    COLUMNS = {
        "clade": "Clade",
        "aa_mutation": "Mutation",
        "actual_count": "a",
        "predicted_count": "e",
        "delta_fitness_ratio": "Counts",
        "log_selection_factor": "DASM",
        "founder_opportunity": "phi",
        "difference": "Difference",
        "adjusted": "Adjusted",
    }
    TABLE = OVER_CUT[list(COLUMNS)].rename(columns=COLUMNS).reset_index(drop=True)
    panels.write_panel_csv(
        OVER_CUT[list(COLUMNS)],
        panels.OUTLIER_TABLE_STEM,
        selection=SELECTION,
        spec=SPEC,
    )
    mo.vstack(
        [
            mo.md("### Table S1, in rank order"),
            mo.ui.table(TABLE.round(3), page_size=25),
        ]
    )
    return (TABLE,)


if __name__ == "__main__":
    app.run()
