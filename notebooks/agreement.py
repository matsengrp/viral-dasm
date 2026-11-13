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
    # `fig:agreement` — the DASM learns mutational effects and how they shift

    Composition only. Nothing here scores a model, reads the scored table or
    recomputes a correlation: every panel is a PNG that `agreement_mutations.py`
    or `agreement_ladders.py` already wrote at its own settings, and this
    notebook arranges them.

    That split is what makes the figure cheap to edit. The panels read a 1.4M-row
    Parquet per model and take a minute; the layout reads five PNGs and takes a
    second, so the spacing can be nudged and re-run without touching the data. It
    also means the figure cannot disagree with the panels: it *is* their output,
    not a second drawing of it.

    | panel | drawn by | what it shows |
    |---|---|---|
    | A | `agreement_mutations` | `F456L` across the clades, both approaches |
    | B | `agreement_mutations` | effects to the 21J founder, counts against DASM |
    | C | `agreement_ladders` | per-clade correlation against the counts cutoff |
    | D | `agreement_mutations` | shifts between four well-separated clades |
    | E | `agreement_ladders` | per-pair shift correlation against the cutoff |

    **GISAID**: reads five rendered panels and writes a figure. No sequence, no
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
    MODELS = spec.selection_models(SELECTION)
    PANEL_DIR = spec.figure_dir(SELECTION)
    return MODELS, PANEL_DIR, SELECTION, fl, spec


@app.cell
def _(PANEL_DIR, SELECTION, fl, mo, spec):
    # `mo.stop` rather than a raise, because the message is the useful output:
    # a reader who exports this notebook by hand before the panels exist
    # should be told which rule writes them, not shown a traceback. Under
    # `rule compose_agreement` the case cannot arise -- the five PNGs are
    # declared inputs -- so this is the standalone path.
    _missing = fl.missing_panels(PANEL_DIR, list(spec.PANEL_LETTERS.values()))
    mo.stop(
        bool(_missing),
        mo.md(
            "**No figure.** These panels have not been written yet: "
            f"`{'`, `'.join(_missing)}`.\n\nRun `snakemake panels` first, so "
            f"they land in `{PANEL_DIR}`."
        ),
    )

    PANELS = fl.load_panels(PANEL_DIR, spec.PANEL_LETTERS)
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
    # Every panel here is Vega-Lite written by `panel_charts.save_panel` at
    # `VEGA_WRITE_PPI`, with titles at `PANEL_FONT_SIZE`, which is what the
    # panel letters are sized against: `title_px_on_disk` turns that spec size
    # into pixels, and the solved dpi turns those pixels into points on the
    # page. `tests/test_panels.py` pins both numbers to `panel_charts`, so the
    # letters follow the panels whenever the type changes.
    composed = fl.compose(
        PANELS,
        [list(_row) for _row in spec.FIGURE_ROWS],
        box_in=spec.BOX_IN,
        scale=spec.FIGURE_SCALE,
        title_px_on_disk=fl.vega_title_px_on_disk(title_px=spec.PANEL_TITLE_PX),
    )
    print(composed.summary())
    composed.figure
    return (composed,)


@app.cell
def _(MODELS, PANEL_DIR, SELECTION, composed, fl, mo, spec):
    _tex = spec.tex_figure_dir(SELECTION)
    _written = fl.save_figure(
        composed, PANEL_DIR, spec.FIGURE_STEM, tex_figure_dir=_tex
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
        f"Selection **{SELECTION}** → `{'`, `'.join(MODELS)}`. Wrote:\n\n"
        + "\n".join(f"- `{_p}`" for _p in _written)
        + _note
    )
    return


if __name__ == "__main__":
    app.run()
