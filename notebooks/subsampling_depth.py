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
    # `fig:subsampling` panels A and B

    The paper's tradeoff, in two measurements of the tree. **Panel A** is how
    unevenly the clades are sampled: a few clades from early in the pandemic
    hold millions of sequences, most hold tens of thousands. **Panel B** is
    what that costs — for the deepest clade (21J, 4.3M sequences) and a much
    shallower one (25C, 104k), the fraction of spike mutations carrying at
    least X actual or expected counts, at full depth and under each cap.

    Dividing the data by clade is what makes a within-clade shift measurable,
    and it is also what leaves most mutations in most clades with too few
    counts to estimate. Panel B is that sentence as a curve.

    Neither panel takes a model: both read mat-pcp's count tables directly.
    Panels C and D — how well each approach recovers the full-tree effects when
    trained on a thinned tree — need the thinned-arm models, and are added once
    those are trained and scored.

    **GISAID**: reads cluster-only count tables; writes publishable per-clade
    counts and shares.
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

    # `panels` before `pandas`: it imports `lib` and then `dasm2`, which moves
    # Arrow off the bundled mimalloc, and `pandas` pulls pyarrow in
    # transitively. Whichever import lands first wins.
    import panels
    import panel_charts as charts

    from lib import mat_pcp_path

    # Named here, as literals, rather than reached through the frame builders:
    # `tests/test_rule_inputs.py` reads the keys a rule's entry point spells out
    # and checks the rule declares them, and a key reached only through a helper
    # is one that check cannot see.
    CLADE_COUNTS = mat_pcp_path("clade_counts_file")
    NTMUT_DIR = mat_pcp_path("ntmut_fitness_dir")
    SELECTION = panels.SUBSAMPLING_SELECTION
    return CLADE_COUNTS, NTMUT_DIR, SELECTION, charts, panels


@app.cell
def _(mo):
    mo.md(r"""
    ## Panel A — sequences per clade

    One dot per clade, one dashed rule per cap. Every clade above a rule is cut
    back **to** it; every clade below is left untouched, which is what makes a
    cap arm a different intervention from taking a fixed fraction of every
    clade. The y axis is logarithmic because the clades span four orders of
    magnitude.
    """)
    return


@app.cell
def _(CLADE_COUNTS, SELECTION, charts, mo, panels):
    sizes = panels.clade_size_frame(CLADE_COUNTS)
    caps = panels.subsampling_caps()

    panel_a = charts.save_panel(
        charts.clade_size_panel(
            sizes,
            caps,
            # Panel B's own two clades, so an arrow cannot point at a clade
            # panel B does not draw.
            highlight=tuple(panels.ECDF_CLADES),
            title=[f"Sequences per clade ({len(sizes)} clades)"],
        ),
        panels.DEPTH_PANELS[0],
        selection=SELECTION,
    )
    panels.write_panel_csv(
        sizes, panels.DEPTH_PANELS[0], selection=SELECTION, spec=None
    )

    mo.md(
        f"**{len(sizes)} clades**, {sizes['n_sequences'].sum():,} sequences. "
        f"Largest {sizes.iloc[0]['clade']} at {sizes.iloc[0]['n_sequences']:,}; "
        f"smallest {sizes.iloc[-1]['clade']} at "
        f"{sizes.iloc[-1]['n_sequences']:,}. Caps: "
        + ", ".join(f"{cap:,}" for cap in caps)
        + "."
    )
    return (panel_a,)


@app.cell
def _(panel_a):
    panel_a
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Panel B — how many mutations have enough counts

    The share of each clade's spike mutations carrying at least X actual or
    expected counts. The dotted rule is the accuracy threshold of 10, so where
    a curve crosses it is the share of that clade's mutations with at least
    that many counts — an upper bound on what is estimable, since scoring also
    drops synonymous, masked and censored mutations.

    The population is the clade's **whole** nucleotide grid — every
    single-nucleotide mutation away from the founder — not the subset that
    clears the threshold. The subset is defined by a threshold on these counts,
    so its curve would reach 1.0 at exactly that threshold and could not show
    what the threshold leaves out.
    """)
    return


@app.cell
def _(NTMUT_DIR, SELECTION, charts, mo, panels):
    ecdf = panels.count_ecdf_frame(NTMUT_DIR)

    panel_b = charts.save_panel(
        charts.count_ecdf_panel(
            ecdf,
            clade_titles=panels.ECDF_CLADES,
            threshold=panels.PANEL_MIN_COUNT,
        ),
        panels.DEPTH_PANELS[1],
        selection=SELECTION,
    )
    panels.write_panel_csv(ecdf, panels.DEPTH_PANELS[1], selection=SELECTION, spec=None)

    _at_threshold = (
        ecdf[ecdf["x"] == float(panels.PANEL_MIN_COUNT)]
        .pivot(index="clade", columns="arm", values="share")
        .reindex(columns=list(panels.CORPUS_NAMES))
    )
    mo.md(
        f"Share with at least {panels.PANEL_MIN_COUNT} actual or expected "
        f"counts, by clade and depth:\n\n" + _at_threshold.round(4).to_markdown()
    )
    return (panel_b,)


@app.cell
def _(panel_b):
    panel_b
    return


if __name__ == "__main__":
    app.run()
