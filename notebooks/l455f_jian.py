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

    # Panel B's domain is solved from its own estimates and this one from its
    # own measurements: the two panels plot different quantities, so a shared
    # axis would assert a comparability the caption is careful not to claim.
    _y_domain = charts.background_y_domain(MEASURED, series=_series)

    _figure = charts.background_panel(
        MEASURED,
        spec.L455F_MUTATION,
        y_domain=_y_domain,
        # Two assays plus a slot for the key, as panel B leaves slots for its
        # own. Panel B is five clades plus two, so this row is narrower and
        # `compose` scales the two to a common width.
        columns=len(spec.L455F_JIAN_ASSAYS) + 2,
        group_column="assay",
        members=list(spec.L455F_JIAN_ASSAYS.items()),
        marker_size=14,
        # No whiskers. Jian et al. report a single value per variant per assay,
        # so there is no spread to draw -- unlike panel B's DASM pools, whose
        # whiskers span their parent haplotypes.
        spread_columns=None,
        value_label=(
            f"measured effect of {spec.L455F_MUTATION}\n"
            r"($-\Delta \ln$ of the assay value)"
        ),
        # The same two ticks as panel B, named the same way, so the reader
        # carries one contrast down the figure.
        # Short, because a three-slot row is narrow and a gloss wider than
        # the figure is centred on it and clipped at both ends. The variants
        # name the backgrounds; the residues are already on the ticks.
        x_gloss=(
            "(left: XBB.1.5; right: XBB.1.5 + "
            f"F{spec.L455F_MARKER_SITE}{spec.L455F_MARKER_AA})"
        ),
        series=_series,
        # Not "approach": this panel draws a measurement, not one of the two
        # approaches panel B compares.
        legend_title="measurement",
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
    return


if __name__ == "__main__":
    app.run()
