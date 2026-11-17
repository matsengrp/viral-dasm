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

    Composition only, like `agreement.py`, `within_clade_shifts.py` and
    `subsampling.py`. Nothing here reads the scored table or recomputes a
    correlation: every panel is a PNG that `dms_correlations.py` already wrote
    at its own settings, and this notebook arranges them.

    | panel | what it shows |
    |---|---|
    | A | three predictors against KP.3.1.1's cell-entry DMS, one scatter each |
    | B | the same Pearson *r*, for all four homologs at once |

    **One row: the scatters and the bars over the same mutations.** Both are
    drawn on every mutation carrying a counts-based estimate, A as one
    homolog's clouds and B as the coefficient for all four, so reading across
    answers "does this hold for the other homologs".

    **`rule dms_correlations` writes four panels and this notebook lays out
    two.** The other pair is the same figure over the mutations with at least
    ten actual or expected counts; it is a check on this one rather than part
    of it, and `main.tex` reports it in prose. `panel_spec.DMS_FIGURE_ROWS`
    is the only thing that decides which panels reach the page, and the two
    that do not are still written, with their CSVs, under `figures/dms/`.

    The arrangement descends from the published figure's, from
    `experiments/2026-09-09-viral-dasm-figure-drafts/notebooks/dms.py` in
    `dasm2-experiments`. What has changed under it is the model and the run:
    that figure is v11, scored from the `2026-08-14-censored-whichmut-v11`
    checkpoints against the `s2SpikeGisaidAll-2026-08-06-v11` mat-pcp run,
    while this one is `dasm-1077k` on v12, the same model and run as every
    other figure in the paper.

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

    # No `--selection`: `fig:dms` has one panel set, and `DMS_SELECTION` is
    # the directory it is written to rather than a choice of models.
    SELECTION = spec.DMS_SELECTION
    PANEL_DIR = spec.figure_dir(SELECTION)
    return PANEL_DIR, SELECTION, fl, spec


@app.cell
def _(PANEL_DIR, fl, mo, spec):
    # `mo.stop` rather than a raise, because the message is the useful output:
    # a reader who exports this notebook by hand before the panels exist
    # should be told which rule writes them, not shown a traceback. Under
    # `rule compose_dms` the case cannot arise -- the four PNGs are declared
    # inputs -- so this is the standalone path.
    _missing = fl.missing_panels(PANEL_DIR, list(spec.DMS_PANEL_LETTERS.values()))
    mo.stop(
        bool(_missing),
        mo.md(
            "**No figure.** These panels have not been written yet: "
            f"`{'`, `'.join(_missing)}`.\n\nRun `snakemake dms` first, so "
            f"they land in `{PANEL_DIR}`."
        ),
    )

    # Every panel, not only the laid-out ones: the sizes printed below are
    # how a reader checks that the pair left off the page rendered too.
    PANELS = fl.load_panels(PANEL_DIR, spec.DMS_PANEL_LETTERS)

    # Left justified and top aligned is all the alignment one row needs. The
    # four canvases are not all the same size -- the two rungs head themselves
    # differently and Vega-Lite lets a centred title widen the view to the
    # right -- but that is empty pixels past the plots, and only A and B are
    # laid out.
    mo.md(
        "At the size the panel rule rendered it:\n\n"
        + "\n".join(
            f"- **{_k}** — {_v.shape[1]} x {_v.shape[0]} px"
            for _k, _v in PANELS.items()
        )
    )
    return (PANELS,)


@app.cell
def _(PANELS, fl, spec):
    # Both panels are Vega-Lite written at the same type size -- the scatter
    # grid's `DMS_FONT_SIZE` is the bar figure's too -- so one title size
    # serves the page, and the panel letters are sized against it.
    composed = fl.compose(
        PANELS,
        [list(_row) for _row in spec.DMS_FIGURE_ROWS],
        box_in=spec.DMS_BOX_IN,
        scale=spec.DMS_FIGURE_SCALE,
        title_px_on_disk=fl.vega_title_px_on_disk(title_px=spec.DMS_TITLE_PX),
    )
    print(composed.summary())
    composed.figure
    return (composed,)


@app.cell
def _(PANEL_DIR, SELECTION, composed, fl, mo, spec):
    _tex = spec.tex_figure_dir(SELECTION)
    _written = fl.save_figure(
        composed, PANEL_DIR, spec.DMS_FIGURE_STEM, tex_figure_dir=_tex
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
