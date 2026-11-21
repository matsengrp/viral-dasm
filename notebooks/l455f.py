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
    # `fig:l455f` — one mutation's effect, between clades and within them

    Composition only. Nothing here scores a model, tallies a pool or
    recomputes a shift: both panels are PNGs that `l455f_lookup.py` and
    `l455f_backgrounds.py` already wrote at their own settings, and this
    notebook arranges them.

    Two rows of one.

    | panel | drawn by | what it shows |
    |---|---|---|
    | A | `l455f_backgrounds` | L455F in each XBB clade, on 456F parents against 456L parents |
    | B | `l455f_lookup` | L455F across every clade that carries an estimate, both approaches, against the clade-pooled counts estimate |

    **What the two rows are for.** A is the claim: inside a single clade,
    where elapsed time is nearly fixed, the same mutation is measured on two
    sets of parents that differ at one site, and both approaches see it move.
    B is what that looks like when the backgrounds are *not* separated -- one
    number per clade, and one number for the whole lot, which is all the
    counts approach can give without being told where to cut.

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
    import panel_spec as spec

    # Required, not defaulted, as the panel notebooks' paths are: the rule
    # states the selection and a fallback spelling here would be a second
    # statement of it.
    SELECTION = str(mo.cli_args()["selection"])
    PANEL_DIR = spec.figure_dir(SELECTION)
    return PANEL_DIR, SELECTION, fl, spec


@app.cell
def _(PANEL_DIR, SELECTION, fl, mo, spec):
    # `mo.stop` rather than a raise, because the message is the useful output:
    # a reader who exports this notebook by hand before the panels exist
    # should be told which rule writes them, not shown a traceback. Under
    # `rule compose_l455f` the case cannot arise -- both PNGs are declared
    # inputs -- so this is the standalone path.
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
    # beside them end at one x. Here every row is one panel, so there is no
    # second column to line up and padding would only widen the narrower row
    # with whitespace.
    mo.md(
        f"Selection **{SELECTION}**, at the size each rule rendered it:\n\n"
        + "\n".join(
            f"- **{_k}** — {_v.shape[1]} x {_v.shape[0]} px"
            for _k, _v in PANELS.items()
        )
    )
    return (PANELS,)


@app.cell
def _(PANELS, fl, spec):
    # The two panels are drawn by different renderers: A is Vega-Lite from
    # `panel_charts.save_panel`, B matplotlib from `save_mpl_panel`. One title
    # size still serves both, because both are written at `VEGA_WRITE_PPI` and
    # a Vega-Lite pixel and a matplotlib point both land at `size * ppi / 72`
    # on disk -- `BACKGROUND_FONT_PT` is `PANEL_FONT_SIZE` converted on that
    # basis. That is what the panel letters are sized against.
    composed = fl.compose(
        PANELS,
        [list(_row) for _row in spec.L455F_FIGURE_ROWS],
        box_in=spec.L455F_BOX_IN,
        scale=spec.L455F_FIGURE_SCALE,
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
