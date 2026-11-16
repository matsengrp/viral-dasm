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
    # `fig:dms_strata` — the composition

    Composition only, like `agreement.py`, `dms.py`, `within_clade_shifts.py`
    and `subsampling.py`. Nothing here reads a score, a table or an assay: it
    loads two rendered panels and stacks them.

    **Two rows, one column.** Each panel is already a row of three cells
    sharing one y axis and one y domain, which is why the row is a single
    image: the shared domain and the single axis live inside the chart, and a
    layout that received six separate images could express neither.

    The rows are stacked left justified, so the two y axes line up and a reader
    can follow one mutation's column from the DASM down to antiGen.

    **GISAID**: reads two rendered panels and writes a figure. No sequence, no
    per-site table and no count passes through this notebook.
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

    import figure_layout as fl
    import panel_charts as charts
    import panel_spec as spec

    # `DMS_SELECTION`, shared with `fig:dms`: it is the directory these panels
    # are written to rather than a choice of model, and both figures are drawn
    # from `FIGURE_MODEL`.
    SELECTION = spec.DMS_SELECTION
    PANEL_DIR = spec.figure_dir(SELECTION)

    # **Required, not defaulted**, for `notebooks/dms_strata.py`'s reason: one
    # figure per assay, and a default here would compose one assay's panels
    # under another's filename without raising.
    HOMOLOG = str(mo.cli_args()["homolog"])
    SLUG = spec.homolog_slug(HOMOLOG)
    return HOMOLOG, PANEL_DIR, SELECTION, SLUG, charts, fl, spec


@app.cell
def _(HOMOLOG, PANEL_DIR, SLUG, charts, fl, mo, spec):
    # `mo.stop` rather than a raise, for `notebooks/dms.py`'s reason: a reader
    # exporting this by hand before the panels exist should be told which rule
    # writes them. Under `rule compose_dms_strata` the case cannot arise.
    _missing = fl.missing_panels(
        PANEL_DIR, list(spec.dms_strata_panel_letters(SLUG).values())
    )
    mo.stop(
        bool(_missing),
        mo.md(
            "**No figure.** These panels have not been written yet: "
            f"`{'`, `'.join(_missing)}`.\n\nRun `snakemake dms_strata` first, "
            f"so they land in `{PANEL_DIR}`."
        ),
    )

    PANELS = fl.load_panels(PANEL_DIR, spec.dms_strata_panel_letters(SLUG))

    # **Checked, not corrected.** The rows draw the same three cells at the
    # same width, so equal image widths mean their y axes are the same band
    # and the columns line up. That is `panel_charts.DMS_Y_LABEL_CHARS`' doing,
    # in the chart, where it also reaches the panel PNGs and the preview.
    #
    # Padding them to a common width here instead would align the composed
    # figure and hide the regression that made it necessary -- `pad` only ever
    # adds, so a row whose tick labels outgrew the reserved width would widen,
    # get silently padded back, and look right while the panels beside it did
    # not. So this raises.
    _widths = {key: panel.shape[1] for key, panel in PANELS.items()}
    if len(set(_widths.values())) != 1:
        raise ValueError(
            f"the rows differ in width ({_widths}), so their y axes are "
            f"different bands and the columns will not line up. "
            f"`panel_charts.DMS_Y_LABEL_CHARS` is "
            f"{charts.DMS_Y_LABEL_CHARS}, which no longer covers the widest "
            f"tick label either row draws."
        )

    # The assay is named here because three renders of this notebook are
    # otherwise identical on screen, and the panel stems are the only thing
    # distinguishing them.
    mo.md(
        f"**{HOMOLOG}**, at the size the panel rule rendered it:\n\n"
        + "\n".join(
            f"- **{_k}** — {_v.shape[1]} x {_v.shape[0]} px"
            for _k, _v in PANELS.items()
        )
    )
    return (PANELS,)


@app.cell
def _(PANELS, SLUG, fl, spec):
    # Both panels are Vega-Lite written at `DMS_FONT_SIZE`, so one title size
    # serves the page and the panel letters are sized against it.
    composed = fl.compose(
        PANELS,
        [list(_row) for _row in spec.dms_strata_figure_rows(SLUG)],
        box_in=spec.DMS_STRATA_BOX_IN,
        scale=spec.DMS_STRATA_FIGURE_SCALE,
        title_px_on_disk=fl.vega_title_px_on_disk(title_px=spec.DMS_STRATA_TITLE_PX),
    )
    print(composed.summary())
    composed.figure
    return (composed,)


@app.cell
def _(PANEL_DIR, SELECTION, SLUG, composed, fl, mo, spec):
    _tex = spec.tex_figure_dir(SELECTION)
    _written = fl.save_figure(
        composed, PANEL_DIR, spec.dms_strata_figure_stem(SLUG), tex_figure_dir=_tex
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
