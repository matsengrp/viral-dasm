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
    # `fig:probabilities` — the composition

    Composition only, like `agreement.py`, `dms.py`, `dms_strata_figure.py`,
    `within_clade_shifts.py` and `subsampling.py`. Nothing here reads a score,
    a table or an assay: it loads two rendered panels and sets them side by
    side.

    **One row, two panels.** Panel A is a square scatter of the identity,
    panel B a row of grouped bars. They share no axis and no domain, so unlike
    `fig:dms_strata` there is nothing to resolve between them — the panels are
    simply placed.

    **Levelled on height, which is the mirror of the stacked figure's
    problem.** `dms_strata_figure.py` stacks two rows and checks their
    *widths*, because a width mismatch there means the two y axes sit in
    different bands and the columns stop lining up. Here the panels sit
    side by side, so what must match is their *height*: two images of
    different height placed in a row leave one floating against the other's
    top or bottom edge. `notebooks/probabilities.py` carries the measured bar
    height that makes them equal.

    **GISAID**: reads two rendered panels and writes a figure. No sequence, no
    per-site table and no count passes through this notebook.
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

    # `DMS_SELECTION`, shared with `fig:dms` and `fig:dms_strata`: it is the
    # directory these panels are written to rather than a choice of model, and
    # all three are drawn from `FIGURE_MODEL`.
    SELECTION = spec.DMS_SELECTION
    PANEL_DIR = spec.figure_dir(SELECTION)
    return PANEL_DIR, SELECTION, fl, spec


@app.cell
def _(PANEL_DIR, fl, mo, spec):
    # `mo.stop` rather than a raise, for `notebooks/dms.py`'s reason: a reader
    # exporting this by hand before the panels exist should be told which rule
    # writes them. Under `rule compose_probabilities` the case cannot arise.
    _missing = fl.missing_panels(
        PANEL_DIR, list(spec.PROBABILITIES_PANEL_LETTERS.values())
    )
    mo.stop(
        bool(_missing),
        mo.md(
            "**No figure.** These panels have not been written yet: "
            f"`{'`, `'.join(_missing)}`.\n\nRun `snakemake probabilities` "
            f"first, so they land in `{PANEL_DIR}`."
        ),
    )

    PANELS = fl.load_panels(PANEL_DIR, spec.PROBABILITIES_PANEL_LETTERS)

    # **Checked, not corrected**, for `dms_strata_figure.py`'s reason. Padding
    # the shorter panel here would level the composed figure and hide the
    # regression that made it necessary: `pad` only ever adds, so a panel that
    # grew would be silently padded back and look right while the one beside
    # it did not. The bar height in `notebooks/probabilities.py` is what makes
    # these equal, and it is measured rather than derived, so it is exactly
    # the kind of constant that goes stale without a check.
    _heights = {key: panel.shape[0] for key, panel in PANELS.items()}
    if len(set(_heights.values())) != 1:
        raise ValueError(
            f"the panels differ in height ({_heights}), so one would float "
            f"against the other's edge in a side-by-side row. "
            f"`BAR_HEIGHT` in `notebooks/probabilities.py` is measured against "
            f"panel A's square and no longer matches it; re-measure it there."
        )

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
    # Both panels are Vega-Lite written at `DMS_FONT_SIZE`, so one title size
    # serves the page and the panel letters are sized against it.
    composed = fl.compose(
        PANELS,
        [list(_row) for _row in spec.PROBABILITIES_FIGURE_ROWS],
        box_in=spec.PROBABILITIES_BOX_IN,
        scale=spec.PROBABILITIES_FIGURE_SCALE,
        title_px_on_disk=fl.vega_title_px_on_disk(title_px=spec.PROBABILITIES_TITLE_PX),
    )
    print(composed.summary())
    composed.figure
    return (composed,)


@app.cell
def _(PANEL_DIR, SELECTION, composed, fl, mo, spec):
    _tex = spec.tex_figure_dir(SELECTION)
    _written = fl.save_figure(
        composed, PANEL_DIR, spec.PROBABILITIES_FIGURE_STEM, tex_figure_dir=_tex
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
