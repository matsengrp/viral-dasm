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
    # `fig:within_clade_shifts` — the DASM identifies shifts in mutational
    # effects within clades

    Composition only. Nothing here scores a model, tallies a pool or
    recomputes a shift: every panel is a PNG that `within_clade_outliers.py`
    or `background_effects.py` already wrote at its own settings, and this
    notebook arranges them.

    Three rows of two, one row per clade. The left panel of a row establishes
    that row's subjects -- the mutations the two approaches disagree about on
    the clade founder -- and the right panel asks what those mutations do once
    the clade is split into its two genetic backgrounds.

    | panel | drawn by | what it shows |
    |---|---|---|
    | A | `within_clade_outliers` | 22D, counts against DASM on the founder |
    | B | `background_effects` | those mutations on 22D's two backgrounds |
    | C | `within_clade_outliers` | the same scatter for 20B |
    | D | `background_effects` | the same row for 20B |
    | E | `within_clade_outliers` | the same scatter for 21K |
    | F | `background_effects` | the same row for 21K |

    **The 21K row is new.** The published figure carried 22D and 20B and then
    a fifth panel: a systematic scan over five other clades, computed by an
    approach this project no longer uses. It is retired, and 21K -- which the
    results text already describes -- takes its place as a third clade row.

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
    # `rule compose_within_clade_shifts` the case cannot arise -- the six PNGs
    # are declared inputs -- so this is the standalone path.
    _missing = fl.missing_panels(PANEL_DIR, list(spec.SHIFT_PANEL_LETTERS.values()))
    mo.stop(
        bool(_missing),
        mo.md(
            "**No figure.** These panels have not been written yet: "
            f"`{'`, `'.join(_missing)}`.\n\nRun `snakemake figures` first, so "
            f"they land in `{PANEL_DIR}`."
        ),
    )

    PANELS = fl.load_panels(PANEL_DIR, spec.SHIFT_PANEL_LETTERS)

    # The rows are placed left justified, so each row's background panel
    # starts wherever that row's scatter ends. The three scatters are not the
    # same width -- 22D's callout labels are drawn outside its plot and the
    # other two clades' inside theirs -- which started the three background
    # panels at three different x. Padding the scatters out to a common width
    # puts B, D and F in a column, which is what lets a reader read their
    # slopes against each other.
    # The first panel of every row, read from the layout rather than spelled
    # "ACE" here: the padding is a fact about what sits in the left column,
    # and a row order changed in `panel_spec` should move it.
    fl.pad_to_common_width(PANELS, [_row[0] for _row in spec.SHIFT_FIGURE_ROWS])

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
    # The two halves of this figure are drawn by different renderers: the
    # scatters are Vega-Lite from `panel_charts.save_panel`, the background
    # rows matplotlib from `save_mpl_panel`. One title size still serves all
    # six, because both are written at `VEGA_WRITE_PPI` and a Vega-Lite pixel
    # and a matplotlib point both land at `size * ppi / 72` on disk --
    # `BACKGROUND_FONT_PT` is `PANEL_FONT_SIZE` converted on that basis. That
    # is what the panel letters are sized against.
    composed = fl.compose(
        PANELS,
        [list(_row) for _row in spec.SHIFT_FIGURE_ROWS],
        box_in=spec.SHIFT_BOX_IN,
        scale=spec.SHIFT_FIGURE_SCALE,
        title_px_on_disk=fl.vega_title_px_on_disk(title_px=spec.PANEL_TITLE_PX),
    )
    print(composed.summary())
    composed.figure
    return (composed,)


@app.cell
def _(PANEL_DIR, SELECTION, composed, fl, mo, spec):
    _tex = spec.tex_figure_dir(SELECTION)
    _written = fl.save_figure(
        composed, PANEL_DIR, spec.SHIFT_FIGURE_STEM, tex_figure_dir=_tex
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
