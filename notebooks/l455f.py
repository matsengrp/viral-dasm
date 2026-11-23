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
    # `fig:l455f` — one mutation's effect on two backgrounds

    Composition only. Nothing here scores a model, tallies a pool or
    recomputes a shift: the panel is a PNG that `l455f_backgrounds.py`
    already wrote at its own settings, and this notebook places it on a page.

    Two rows of one, both lettered: the figure now makes a distinction its
    caption refers to, between what was measured and what was estimated.

    | panel | drawn by | what it shows |
    |---|---|---|
    | A | `l455f_jian` | L455F measured by Jian et al. on an F456 and an L456 XBB.1.5 background |
    | B | `l455f_backgrounds` | L455F in each XBB clade, on F456 parents against L456 parents |

    **What it is for.** Inside a single clade, where elapsed time is nearly
    fixed, the same mutation is measured on two sets of parents that differ
    at one site, and both approaches see it move -- and panel A shows an
    experiment seeing the same thing, which is what Jesse asked for.

    The two panels plot **different quantities** in the same units: a log fold
    change in a laboratory measurement is not a log selection factor. They are
    not on a shared axis, and the caption has to say why.

    **GISAID**: reads one rendered panel and writes a figure. No sequence, no
    per-site table and no count passes through this notebook.
    """)
    return


@app.cell
def _(mo):
    import dataclasses
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parent.parent
    for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

    import figure_layout as fl
    import panel_spec as spec

    # Required, not defaulted, as the panel notebooks' paths are: the rule
    # states the selection and a fallback spelling here would be a second
    # statement of it.
    SELECTION = str(mo.cli_args()["selection"])
    PANEL_DIR = spec.figure_dir(SELECTION)
    return PANEL_DIR, SELECTION, dataclasses, fl, spec


@app.cell
def _(PANEL_DIR, SELECTION, fl, mo, spec):
    # `mo.stop` rather than a raise, because the message is the useful output:
    # a reader who exports this notebook by hand before the panels exist
    # should be told which rule writes them, not shown a traceback. Under
    # `rule compose_l455f` the case cannot arise -- the panel PNG is a
    # declared input -- so this is the standalone path.
    _missing = fl.missing_panels(PANEL_DIR, list(spec.L455F_PANEL_LETTERS.values()))
    mo.stop(
        bool(_missing),
        mo.md(
            "**No figure.** These panels have not been written yet: "
            f"`{'`, `'.join(_missing)}`.\n\nRun `snakemake figures` first, so "
            f"they land in `{PANEL_DIR}`."
        ),
    )

    PANELS = fl.load_panels(PANEL_DIR, spec.L455F_PANEL_LETTERS)

    # No `pad_to_common_width` here, unlike `within_clade_shifts.py`. That
    # figure pads because each of its rows is a scatter *and* a background
    # row, and the background rows only read as a column if the scatters
    # beside them end at one x. Here there is one panel, so there is nothing
    # to line it up against.
    mo.md(
        f"Selection **{SELECTION}**, at the size each rule rendered it:\n\n"
        + "\n".join(
            f"- **{_k}** — {_v.shape[1]} x {_v.shape[0]} px"
            for _k, _v in PANELS.items()
        )
    )
    return (PANELS,)


@app.cell
def _(PANELS, dataclasses, fl, spec):
    # The panel is matplotlib, from `panel_charts.save_mpl_panel`, but the
    # title size is still given in Vega-Lite's units: both renderers write at
    # `VEGA_WRITE_PPI`, and a Vega-Lite pixel and a matplotlib point both land
    # at `size * ppi / 72` on disk -- `BACKGROUND_FONT_PT` is
    # `PANEL_FONT_SIZE` converted on that basis. It sizes the two letters and
    # the figure both, so the dpi this figure is solved at stays comparable to
    # its neighbours'.
    composed = fl.compose(
        PANELS,
        [list(_row) for _row in spec.L455F_FIGURE_ROWS],
        box_in=spec.L455F_BOX_IN,
        scale=spec.L455F_FIGURE_SCALE,
        # The default gaps, widened between the rows: each panel carries a
        # heading of its own, and at the default the two read as one block.
        spacing=dataclasses.replace(
            fl.DEFAULT_SPACING, row_gap_in=spec.L455F_ROW_GAP_IN
        ),
        labels=spec.L455F_FIGURE_LABELS,
        title_px_on_disk=fl.vega_title_px_on_disk(title_px=spec.PANEL_TITLE_PX),
    )
    print(composed.summary())
    composed.figure
    return (composed,)


@app.cell
def _(PANEL_DIR, SELECTION, composed, fl, mo, spec):
    _tex = spec.tex_figure_dir(SELECTION)
    _written = fl.save_figure(
        composed, PANEL_DIR, spec.L455F_FIGURE_STEM, tex_figure_dir=_tex
    )
    _copied = _tex is not None and any(_p.parent == _tex for _p in _written)
    if SELECTION != spec.DEFAULT_SELECTION:
        _note = (
            f"\n\nNo manuscript copy: **{SELECTION}** is not the selection the "
            f"paper publishes, and every selection composes to the same file "
            f"name. The figure itself is written."
        )
    elif _copied:
        _note = ""
    else:
        _note = (
            f"\n\nThe manuscript copy was **skipped**: `config.yaml`'s "
            f"`tex_figure_dir` is `{_tex}`, which is not checked out here. "
            f"The figure itself is written."
        )
    mo.md(
        f"Selection **{SELECTION}**. Wrote:\n\n"
        + "\n".join(f"- `{_p}`" for _p in _written)
        + _note
    )
    return


if __name__ == "__main__":
    app.run()
