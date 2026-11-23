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
    # `fig:l455f` panel A — the same contrast, measured

    Panel B shows two *estimates* of how L455F's effect changes with the
    residue at 456. This panel shows the *measurement*, from Jian et al., so
    the figure compares the model to an experiment and not only to another
    model. Jesse asked for this in his second round of comments.

    **Nothing is scored, tallied or read from the tree here.** The input is
    four pairs of numbers transcribed from the paper; everything below is a
    log difference of two of them.

    ## What is plotted

    Jian et al. report two assays on four XBB.1.5 variants. Each assay gives
    the effect of L455F twice: once on a parent carrying F456, once on a
    parent carrying L456 (that is, carrying F456L).

    Both assays run **opposite** to benefit for the virus -- a lower $K_D$ is
    tighter receptor binding, and a lower NT$_{50}$ is more escape from plasma
    neutralization -- so both are negated, and the plotted effect is

    $$-\Delta \ln(\text{measurement}) = \ln(\text{parent}) - \ln(\text{mutant})$$

    which puts beneficial-to-the-virus upward, the direction panel B's fitness
    effects already run in. Natural logarithms, so the axis is in the same
    units as panel B's log selection factor.

    ## The two axes

    The sub-panels measure **different quantities** -- a dissociation
    constant and a neutralization titer -- and the binding effects are far
    the larger, so one axis across both would draw escape as a flat line.
    Each gets its own, with the escape axis the binding axis divided by a
    whole number: dividing cannot move zero, so the two keep one baseline,
    and the one thing a reader carries between them is that number.

    **These are not log selection factors.** A log fold change in $K_D$ and a
    log selection factor are the same *units* and different *quantities*. The
    panel is here to corroborate the direction and rough size of the shift,
    not to claim the two numbers are the same measurement made twice -- and
    the caption has to say so.

    **GISAID**: no sequence, no per-site table and no count passes through
    this notebook.
    """)
    return


@app.cell
def _(mo):
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    REPO_ROOT = Path(__file__).resolve().parent.parent
    for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

    import panel_charts as charts
    import panel_spec as spec
    import panels

    # Required, not defaulted, as the other panel notebooks' are: the rule
    # states the selection and a fallback spelling here would restate it.
    SELECTION = str(mo.cli_args()["selection"])
    MEASUREMENTS = Path(
        str(mo.cli_args().get("measurements") or REPO_ROOT / spec.L455F_JIAN_DATA)
    )
    return MEASUREMENTS, REPO_ROOT, SELECTION, charts, np, pd, panels, spec


@app.cell
def _(MEASUREMENTS, mo, np, pd):
    # `comment="#"`: the file carries its provenance and its sign convention in
    # a header, which is the only place a reader of the CSV alone would find
    # out that both assays are negated.
    RAW = pd.read_csv(MEASUREMENTS, comment="#")

    _expected = {
        "assay",
        "background",
        "background_order",
        "parent_variant",
        "parent_value",
        "mutant_variant",
        "mutant_value",
    }
    _missing = sorted(_expected - set(RAW.columns))
    if _missing:
        raise KeyError(f"{MEASUREMENTS} is missing column(s) {_missing}.")
    if (RAW[["parent_value", "mutant_value"]] <= 0).to_numpy().any():
        raise ValueError(
            f"{MEASUREMENTS} carries a non-positive measurement, which has no "
            f"logarithm. Both assays report a concentration or a titer."
        )

    # The one derived quantity, and the whole of the panel's arithmetic.
    MEASURED = RAW.assign(
        measured_effect=np.log(RAW["parent_value"]) - np.log(RAW["mutant_value"]),
        # `background_panel` groups sub-panels by a column and requires a
        # `candidate`; here every row is the same mutation and the sub-panel is
        # the assay.
        candidate="L455F",
    )

    mo.md(
        "Transcribed measurements and the effect computed from them:\n\n"
        + MEASURED[
            [
                "assay",
                "background",
                "parent_variant",
                "parent_value",
                "mutant_variant",
                "mutant_value",
                "measured_effect",
            ]
        ]
        .round(3)
        .to_markdown(index=False)
    )
    return (MEASURED,)


@app.cell
def _(MEASURED, SELECTION, charts, mo, panels, spec):
    # Its own series rather than the two model arms, so the value column is
    # named for what it holds. See `background_panel`'s `series` argument.
    _series = (("Jian et al.", "measured_effect", charts.MEASURED_GREEN, "s"),)

    # One axis per sub-panel, because the two measure different quantities: a
    # dissociation constant and a neutralization titer. Binding sets the
    # scale and escape is nested inside it, so the two keep one zero and
    # differ by one whole number -- see `charts.proportional_y_domain`. The
    # unpacking is the check that there are exactly two.
    _reference, _nested = list(spec.L455F_JIAN_ASSAYS)
    _reference_domain = charts.background_y_domain(
        MEASURED[MEASURED["assay"] == _reference], series=_series
    )
    _nested_domain, Y_AXIS_DIVISOR = charts.proportional_y_domain(
        _reference_domain,
        MEASURED[MEASURED["assay"] == _nested],
        series=_series,
    )

    _figure = charts.background_panel(
        MEASURED,
        spec.L455F_MUTATION,
        # Unused, since every sub-panel is named in `y_domains`, and passed as
        # the reference axis so that a sub-panel added without an axis of its
        # own would raise rather than be drawn against something arbitrary.
        y_domain=_reference_domain,
        y_domains={_reference: _reference_domain, _nested: _nested_domain},
        # Named by their value axes, not by titles of their own: the quantity
        # differs between them, and `L455F_JIAN_TITLE` says what they have in
        # common once, over both.
        value_labels=spec.L455F_JIAN_ASSAYS,
        figure_title=spec.L455F_JIAN_TITLE,
        # Two assays and nothing else: no slot for a key, because there is
        # one series and the heading names it. Panel B spends two slots on
        # its own key, which has three arms to tell apart.
        columns=len(spec.L455F_JIAN_ASSAYS),
        cell_w_in=spec.L455F_JIAN_CELL_W_IN,
        legend=False,
        group_column="assay",
        members=[(assay, "") for assay in spec.L455F_JIAN_ASSAYS],
        marker_size=14,
        # No whiskers. Jian et al. report a single value per variant per assay,
        # so there is no spread to draw -- unlike panel B's DASM pools, whose
        # whiskers span their parent haplotypes.
        spread_columns=None,
        # The same two ticks as panel B, named the same way and under the
        # same words, so the reader carries one contrast down the figure.
        # No gloss. Panel B's names which side is which because its x ticks
        # are residues and its sub-panels are clades; here the heading already
        # says "in the presence and absence of F456L", and a line repeating it
        # under the ticks would be the third place one figure says so.
        x_gloss=None,
        series=_series,
    )
    PANEL = charts.save_mpl_panel(_figure, spec.L455F_JIAN_PANEL, selection=SELECTION)
    panels.write_pool_csv(
        MEASURED[
            [
                "assay",
                "candidate",
                "background",
                "background_order",
                "parent_variant",
                "parent_value",
                "mutant_variant",
                "mutant_value",
                "measured_effect",
            ]
        ],
        spec.L455F_JIAN_PANEL,
        selection=SELECTION,
    )
    mo.output.replace(PANEL)
    return (Y_AXIS_DIVISOR,)


@app.cell
def _(Y_AXIS_DIVISOR, mo):
    mo.md(
        f"""
        The escape axis spans **one {Y_AXIS_DIVISOR}th** of the binding axis,
        with both zeroes at the same height. A caption claiming the two
        sub-panels can be read against each other has to state that number;
        one that does not should say the axes differ instead.
        """
    )
    return


if __name__ == "__main__":
    app.run()
