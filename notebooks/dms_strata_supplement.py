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
    # `fig:dms_strata_supplement` — the composition

    Composition only, like `dms_strata_figure.py`, which this is the
    many-assay counterpart of. Nothing here reads a score, a table or an
    assay: it loads six rendered panels and stacks them.

    **Six rows, one column, bracketed in pairs.** `fig:dms_strata` draws one
    assay as two rows, antiGen above the DASM, and `main.tex` shows that
    figure for one assay. The others say the same thing about three more
    experiments, so they go on one page here rather than three, with a
    bracket on the right naming each pair. The panels keep the titles they
    were rendered with, so nothing in the caption has to say which pair is
    which.

    **Height binds here, where width binds in `fig:dms_strata`.** See
    `panel_spec.DMS_STRATA_SUPPLEMENT_BOX_IN` for the sizes and what the
    panels come out at.

    **GISAID**: reads six rendered panels and writes a figure. No sequence, no
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
    import panel_spec as spec
    from lib import dms_homologs

    SELECTION = spec.DMS_SELECTION
    PANEL_DIR = spec.figure_dir(SELECTION)

    # The roster is read here and passed down, for `homolog_for_slug`'s
    # reason: `panel_spec` reads no config of its own.
    HOMOLOGS = dms_homologs()
    LETTERS = spec.dms_strata_supplement_panel_letters(HOMOLOGS)
    return HOMOLOGS, LETTERS, PANEL_DIR, SELECTION, fl, spec


@app.cell
def _(HOMOLOGS, LETTERS, PANEL_DIR, fl, mo, spec):
    # `mo.stop` rather than a raise, for `notebooks/dms.py`'s reason: a reader
    # exporting this by hand before the panels exist should be told which rule
    # writes them.
    _missing = fl.missing_panels(PANEL_DIR, list(LETTERS.values()))
    mo.stop(
        bool(_missing),
        mo.md(
            "**No figure.** These panels have not been written yet: "
            f"`{'`, `'.join(_missing)}`.\n\nRun `snakemake dms_strata` first, "
            f"so they land in `{PANEL_DIR}`."
        ),
    )

    PANELS = fl.load_panels(PANEL_DIR, LETTERS)

    # **Checked, not corrected**, exactly as `dms_strata_figure.py` checks it,
    # and across assays as well as within one: equal widths mean every row's
    # y axis is the same band, so a reader can follow one column down the
    # whole page rather than only down one pair.
    _widths = {key: panel.shape[1] for key, panel in PANELS.items()}
    if len(set(_widths.values())) != 1:
        raise ValueError(
            f"the rows differ in width ({_widths}), so their y axes are "
            f"different bands and the columns will not line up down the "
            f"page. `panel_charts.DMS_Y_LABEL_CHARS` no longer covers the "
            f"widest tick label some row draws."
        )

    mo.md(
        "**"
        + ", ".join(spec.dms_strata_supplement_homologs(HOMOLOGS))
        + "**, at the size the panel rule rendered them:\n\n"
        + "\n".join(
            f"- **{_k}** — {_v.shape[1]} x {_v.shape[0]} px"
            for _k, _v in PANELS.items()
        )
    )
    return (PANELS,)


@app.cell
def _(HOMOLOGS, PANELS, fl, spec):
    composed = fl.compose(
        PANELS,
        [list(_row) for _row in spec.dms_strata_supplement_rows(HOMOLOGS)],
        box_in=spec.DMS_STRATA_SUPPLEMENT_BOX_IN,
        scale=spec.DMS_STRATA_SUPPLEMENT_SCALE,
        title_px_on_disk=fl.vega_title_px_on_disk(title_px=spec.DMS_STRATA_TITLE_PX),
        row_groups=spec.dms_strata_supplement_groups(HOMOLOGS),
    )
    print(composed.summary())
    composed.figure
    return (composed,)


@app.cell
def _(PANEL_DIR, SELECTION, composed, fl, mo, spec):
    _tex = spec.tex_figure_dir(SELECTION)
    _written = fl.save_figure(
        composed,
        PANEL_DIR,
        spec.dms_strata_supplement_figure_stem(),
        tex_figure_dir=_tex,
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
