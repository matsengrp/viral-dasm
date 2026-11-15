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
    # `fig:subsampling` — DASM estimates hold up in sparsely sampled clades

    Composition only. Nothing here measures the tree, reads the scored table
    or fits a line: every panel is a PNG that `subsampling_depth.py` or
    `subsampling_ladder.py` already wrote at its own settings, and this
    notebook arranges them.

    | panel | drawn by | what it shows |
    |---|---|---|
    | A | `subsampling_depth` | sequences per clade, and the three caps |
    | B | `subsampling_depth` | count eCDFs for a deep and a shallow clade |
    | C | `subsampling_ladder` | Delta recovery, counts against DASM, per arm |
    | D | `subsampling_ladder` | the same across the five deepest clades |

    **The arrangement is this pipeline's, not the published figure's.** The
    panels that figure was composed from are in `dasm2-experiments`,
    `experiments/2026-09-09-viral-dasm-figure-drafts`, under
    `notebooks/subsampling.py` -- the notebook `main.tex`'s provenance
    comment names -- and it placed them `[["A"], [["B", "C"], "D"]]`, with D
    standing tall beside a column of B over C. That worked because its panel
    D stacked the five clades down the page. This pipeline's panel D puts the
    clades across it instead, so the tall panel is now C and the column has
    nothing to hold; `SUBSAMPLING_FIGURE_ROWS` carries the rows that follow
    from the shapes these rules actually render.

    **GISAID**: reads four rendered panels and writes a figure. No sequence,
    no per-site table and no count passes through this notebook.
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

    import figure_layout as fl
    import panel_spec as spec

    # No `--selection`, unlike the other two compositions: `fig:subsampling`
    # has one panel set, and `SUBSAMPLING_SELECTION` is the directory it is
    # written to rather than a choice of models. Panels C and D are drawn for
    # `SUBSAMPLING_MODEL` alone, for the reason that constant gives.
    SELECTION = spec.SUBSAMPLING_SELECTION
    PANEL_DIR = spec.figure_dir(SELECTION)
    return PANEL_DIR, SELECTION, fl, spec


@app.cell
def _(PANEL_DIR, fl, mo, spec):
    # `mo.stop` rather than a raise, because the message is the useful
    # output: a reader who exports this notebook by hand before the panels
    # exist should be told which rule writes them, not shown a traceback.
    # Under `rule compose_subsampling` the case cannot arise -- the four PNGs
    # are declared inputs -- so this is the standalone path.
    _missing = fl.missing_panels(
        PANEL_DIR, list(spec.SUBSAMPLING_PANEL_LETTERS.values())
    )
    mo.stop(
        bool(_missing),
        mo.md(
            "**No figure.** These panels have not been written yet: "
            f"`{'`, `'.join(_missing)}`.\n\nRun `snakemake subsampling` "
            f"first, so they land in `{PANEL_DIR}`."
        ),
    )

    PANELS = fl.load_panels(PANEL_DIR, spec.SUBSAMPLING_PANEL_LETTERS)

    # The rows are placed left justified, so row 1's panel B starts wherever
    # panel A ends. Nothing sits under either of them -- C and D each take a
    # row of their own -- so there is no column to line up and no padding to
    # do, which is the one thing `fig:within_clade_shifts`' composition needs
    # and this one does not.
    mo.md(
        "At the size each rule rendered it:\n\n"
        + "\n".join(
            f"- **{_k}** — {_v.shape[1]} x {_v.shape[0]} px"
            for _k, _v in PANELS.items()
        )
    )
    return (PANELS,)


@app.cell
def _(PANELS, fl, spec):
    # All four panels are Vega-Lite from `panel_charts.save_panel`, so one
    # title size serves them: `PANEL_TITLE_PX` is what the specs set and what
    # the panel letters are sized against. The published figure passed 24 px
    # here because both of its sources drew this figure's panels two points
    # larger than the rest of that campaign's; this pipeline draws every
    # panel of every figure at `PANEL_FONT_SIZE`, and `tests/test_panels.py`
    # pins the two together.
    composed = fl.compose(
        PANELS,
        [list(_row) for _row in spec.SUBSAMPLING_FIGURE_ROWS],
        box_in=spec.SUBSAMPLING_BOX_IN,
        scale=spec.SUBSAMPLING_FIGURE_SCALE,
        title_px_on_disk=fl.vega_title_px_on_disk(title_px=spec.PANEL_TITLE_PX),
    )
    print(composed.summary())
    composed.figure
    return (composed,)


@app.cell
def _(PANEL_DIR, SELECTION, composed, fl, mo, spec):
    _tex = spec.tex_figure_dir(SELECTION)
    _written = fl.save_figure(
        composed, PANEL_DIR, spec.SUBSAMPLING_FIGURE_STEM, tex_figure_dir=_tex
    )
    _copied = _tex is not None and any(_p.parent == _tex for _p in _written)
    _note = (
        ""
        if _copied
        else (
            f"\n\nThe manuscript copy was **skipped**: `config.yaml`'s "
            f"`tex_figure_dir` is `{_tex}`, which is not checked out here. "
            f"The figure itself is written."
        )
    )
    mo.md("Wrote:\n\n" + "\n".join(f"- `{_p}`" for _p in _written) + _note)
    return


if __name__ == "__main__":
    app.run()
