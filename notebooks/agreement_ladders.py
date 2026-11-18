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
    # `fig:agreement` panels C and E

    The paper's two count-threshold sweeps: *the correlation increases as the
    counts threshold increases*. **Panel C** is one dot per clade, the
    within-clade Pearson r; **panel E** is one dot per clade pair, the r
    between the two approaches' shifts. The grey dashes are the medians.

    The x axis is the cutoff — the minimum actual or expected count a mutation
    needs to enter a correlation — so a rising curve says the same model agrees
    better with the counts on a better-measured subset of the same clades. That
    is evidence about the yardstick rather than the model: the distance from
    r = 1 at the lowest rung was substantially noise in mat-pcp's estimate.

    **A clade contributes a dot only where it retains enough mutations**, and
    the clade set is pinned across the rungs — every rung uses the clades that
    clear the floor at *all* of them. Otherwise the thin, low-r clades would
    drop out as the threshold rose and the trend would be survivorship rather
    than precision.

    Panels A, B and D are the per-mutation panels and live in
    `agreement_mutations.py`.

    **GISAID**: reads the cluster-only scored table; writes publishable
    aggregate statistics.
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

    # `panels` before `pandas`: it imports `lib` and then `dasm2`, which moves
    # Arrow off the bundled mimalloc, and `pandas` pulls pyarrow in
    # transitively. Whichever import lands first wins, and this notebook reads
    # a Parquet.
    import panels
    import panel_charts as charts

    import pandas as pd

    # Required, not defaulted. A fallback spelling of the path is a second
    # spelling of a fact `workflow/figures.smk` already states, and the rule
    # passes it in; missing it should stop here rather than quietly read a
    # different file from the one the rule declared.
    _args = mo.cli_args()
    TABLE_PATH = Path(str(_args["table"]))
    SELECTION = str(_args.get("selection") or panels.DEFAULT_SELECTION)
    MODELS = panels.selection_models(SELECTION)
    return MODELS, SELECTION, TABLE_PATH, charts, panels, pd


@app.cell
def _(MODELS, TABLE_PATH, mo, panels, SELECTION):
    SPEC = panels.BASE_SPEC
    # Panel C scores every clade that can be scored, so the roster is the
    # clade axis here rather than a selection; panel E draws pairs from the
    # shift clades, which are a selection because pairs are quadratic.
    TABLE = panels.read_scored_table(TABLE_PATH, models=MODELS)
    CLADES = sorted(TABLE["clade"].unique())
    BASE = panels.base_frames(
        TABLE,
        spec=SPEC,
        models=MODELS,
        clades=CLADES,
    )
    mo.md(
        f"Selection **{SELECTION}** → models `{'`, `'.join(MODELS)}`, "
        f"{len(CLADES)} clades, {len(BASE)} (model, clade) frames at cutoff "
        f"zero."
    )
    return BASE, SPEC


@app.cell
def _(BASE, MODELS, SPEC, mo, panels, pd):
    FOUNDERS = panels.shift_pair_founders()
    _within, _shift, _paired_by_model = [], [], {}
    for _model in MODELS:
        _base = panels.by_clade(BASE, _model)
        if not _base:
            continue
        # The clades that clear the mutation floor at every rung. The per-rung
        # adequacy frame is not kept: how many clades survive the pairing is
        # reported below, and nothing in the figure plots the rest of it.
        _paired, _ = panels.count_ladder_adequacy(
            _base,
            thresholds=panels.COUNT_LADDER,
            min_mutations=SPEC.min_clade_mutations,
        )
        _paired_by_model[_model] = list(_paired)
        if _paired:
            _within.append(
                panels.within_clade_count_ladder(
                    _base,
                    SPEC.fitness_metric,
                    _paired,
                    thresholds=panels.COUNT_LADDER,
                ).assign(model=_model)
            )
        # A clade can be selected for the shift panel and still be too thin to
        # correlate at every rung, so the pairs are drawn from the intersection.
        _pair_clades = [c for c in panels.SHIFT_CLADES if c in _paired]
        if len(_pair_clades) >= 2:
            _shift.append(
                panels.shift_pair_count_ladder(
                    {c: _base[c] for c in _pair_clades},
                    SPEC.fitness_metric,
                    _pair_clades,
                    thresholds=panels.COUNT_LADDER,
                    founders=FOUNDERS,
                ).assign(model=_model)
            )

    WITHIN_LADDER = pd.concat(_within, ignore_index=True)
    SHIFT_LADDER = pd.concat(_shift, ignore_index=True)
    PAIRED_BY_MODEL = _paired_by_model
    mo.md(
        "\n".join(
            f"- `{model}`: {len(clades)} clades retain at least "
            f"{SPEC.min_clade_mutations} mutations at every rung of "
            f"{panels.COUNT_LADDER}, and are the dots at all three"
            for model, clades in PAIRED_BY_MODEL.items()
        )
    )
    return PAIRED_BY_MODEL, SHIFT_LADDER, WITHIN_LADDER


@app.cell
def _(SELECTION, SPEC, WITHIN_LADDER, charts, panels):
    # Pooled over the selection's models under one median, which is what the
    # published numbers do: each rung's dots are every (model, clade) pair.
    panel_c = charts.save_panel(
        charts.ladder_row(
            WITHIN_LADDER,
            unit_field="clade",
            unit_title="Clade",
            title=[
                "per-clade correlation in",
                "estimated fitness effects",
                f"(N clades = {charts.ladder_clade_count(WITHIN_LADDER, 'clade')})",
            ],
        ),
        "panel_c_within_clade_ladder",
        selection=SELECTION,
    )
    # What the panel plots, and nothing else: one dot is (model, clade, rung)
    # at a Pearson r, over `n_mutations` mutations. `within_clade_metrics` also
    # returns `spearman_rho` and `regression_slope`, which no panel of this
    # figure draws; they are recomputed from the table when a comparison wants
    # them rather than committed beside numbers the figure does show.
    panels.write_panel_csv(
        WITHIN_LADDER[["model", "clade", "min_count", "pearson_r", "n_mutations"]],
        "panel_c_within_clade_ladder",
        selection=SELECTION,
        spec=SPEC,
    )
    panel_c
    return


@app.cell
def _(SELECTION, SHIFT_LADDER, SPEC, charts, panels):
    panel_e = charts.save_panel(
        charts.ladder_row(
            SHIFT_LADDER,
            unit_field="pair_label",
            unit_title="Clade pair",
            title=[
                "correlation in estimated shifts",
                "in fitness effects between clades",
                "(N clades = "
                + f"{charts.ladder_clade_count(SHIFT_LADDER, 'pair_label')})",
            ],
        ),
        "panel_e_shift_ladder",
        selection=SELECTION,
    )
    # As panel C: the plotted quantity and the keys that identify a dot.
    # `shift_pair_metrics`' other columns -- `spearman_rho`,
    # `directional_concordance`, `n_large_shifts`, `sf_shift_is_constant` --
    # are not drawn here, and `n_large_shifts` is the column whose gate this
    # repository does not port at all.
    panels.write_panel_csv(
        SHIFT_LADDER[["model", "pair_label", "min_count", "pearson_r", "n_mutations"]],
        "panel_e_shift_ladder",
        selection=SELECTION,
        spec=SPEC,
    )
    panel_e
    return


@app.cell
def _(BASE, MODELS, PAIRED_BY_MODEL, SPEC, panels, pd):
    # `si.tex`'s shift-error figure, and the sentence it answers: *the two
    # approaches were also in strong agreement when expanding from these four
    # clade pairs to all pairwise combinations of the 22 clades*. Panel D
    # draws six pairs; this draws every pair of the adequate set.
    #
    # The clade set is `PAIRED_BY_MODEL` — the clades that clear the mutation
    # floor at *every* rung, which is the set panels C and E are drawn over.
    # `adequate_clades` at a single cutoff would give a larger set and a
    # plausible-looking figure drawn over the wrong pairs.
    #
    # Nothing is reduced to a correlation here: the rows are kept per mutation
    # and per pair, and only the binned summary below reaches the chart.
    # Correlation would be the wrong statistic anyway — most pairs hold few
    # mutations whose effects shift appreciably, and an r over shifts that are
    # all near zero is noise.
    _frames = []
    for _model in MODELS:
        _clades = list(PAIRED_BY_MODEL.get(_model, []))
        if len(_clades) < 2:
            continue
        _base = panels.by_clade(BASE, _model)
        # With the adequate set, never bare: `shift_pair_founders` defaults to
        # the four `SHIFT_CLADES`, and `identical_founder_pairs` indexes the
        # dict it is given without a guard, so the default raises `KeyError`
        # over 22 clades.
        _founders = panels.shift_pair_founders(_clades)
        for _rung in panels.COUNT_LADDER:
            _at_cutoff = {
                _clade: panels.apply_count_threshold(_base[_clade], _rung)
                for _clade in _clades
            }
            _frames.append(
                panels.shift_error_frame(
                    _at_cutoff,
                    _clades,
                    SPEC.fitness_metric,
                    founders=_founders,
                ).assign(model=_model, min_count=_rung)
            )
    SHIFT_ERRORS = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
    return (SHIFT_ERRORS,)


@app.cell
def _(SHIFT_ERRORS, mo, panels, pd):
    # One windowed frame for the whole figure, with the rung as a column, so
    # the three panels are three filters on one grid rather than three grids.
    #
    # `min_rows=1` draws every window that holds a row: the axis then covers
    # the data, and a thin window is marked by the `n` in its tooltip and in
    # the panel's CSV rather than by an opacity a reader would have to decode.
    # `start=0.0` because the windowed quantity is a magnitude; the library's
    # own anchor is for a figure windowed along the effect axis, where the
    # clip at -4 is what the anchor keeps clear of.
    #
    # Models are pooled inside a rung, as panels C and E pool them: a
    # selection's models are two readings of one approach, not two series.
    SHIFT_ERROR_WINDOWS = (
        pd.concat(
            [
                panels.shift_error_quantiles(
                    _rows,
                    value_col="abs_fitness_shift",
                    start=0.0,
                    width=panels.SHIFT_ERROR_WINDOW_WIDTH,
                    min_rows=1,
                ).assign(min_count=_rung)
                for _rung in sorted(SHIFT_ERRORS["min_count"].unique())
                if len(_rows := SHIFT_ERRORS[SHIFT_ERRORS["min_count"] == _rung])
            ],
            ignore_index=True,
        )
        if len(SHIFT_ERRORS)
        else pd.DataFrame()
    )
    mo.md(
        f"""
        ### The shift-error windows

        {len(SHIFT_ERRORS):,} (pair, mutation) rows over
        {SHIFT_ERRORS["pair_label"].nunique() if len(SHIFT_ERRORS) else 0}
        clade pairs and {len(panels.COUNT_LADDER)} rungs, summarised into
        {len(SHIFT_ERROR_WINDOWS)} windows.
        """
    )
    return (SHIFT_ERROR_WINDOWS,)


@app.cell
def _(SELECTION, SHIFT_ERROR_WINDOWS, SPEC, charts, mo, panels):
    # The figure `si.tex` includes. It is written as a panel rather than
    # composed: `figure_layout` exists to pack several PNGs into one page box
    # and solve a shared dpi for them, and a supplement that is one chart has
    # nothing to compose. The PDF is what the manuscript includes, which is
    # the one format `save_panel` does not write by default.
    SHIFT_ERROR_PANEL = charts.save_panel(
        charts.shift_error_row(SHIFT_ERROR_WINDOWS),
        panels.SHIFT_ERROR_PANELS[0],
        selection=SELECTION,
        formats=panels.SUPPLEMENT_FORMATS,
    )
    panels.write_panel_csv(
        SHIFT_ERROR_WINDOWS,
        panels.SHIFT_ERROR_PANELS[0],
        selection=SELECTION,
        spec=SPEC,
    )
    _copied = charts.copy_panel_to_manuscript(
        panels.SHIFT_ERROR_PANELS[0], selection=SELECTION
    )
    mo.vstack(
        [
            mo.md(
                f"Manuscript copy: `{_copied}`"
                if _copied is not None
                else "Manuscript copy skipped: "
                f"`{SELECTION}` is not a published selection, or no "
                "manuscript is checked out beside this repository."
            ),
            SHIFT_ERROR_PANEL,
        ]
    )
    return


@app.cell
def _(SELECTION, SHIFT_LADDER, WITHIN_LADDER, mo, panels, pd):
    # Every median `main.tex` quotes for this figure, computed from the same
    # frames the two panels plot. The manuscript rounds to two places, so both
    # forms are here: a reader checking the paper against the pipeline should
    # be comparing 0.97 to 0.9707 rather than re-deriving it from a CSV.
    def _medians(ladder, unit_field, panel, phrase):
        return [
            {
                "panel": panel,
                "quantity": phrase,
                "cutoff": rung,
                "dots": int(_rung_dots[unit_field].nunique()),
                "median r": round(float(_rung_dots["pearson_r"].median()), 4),
                "as quoted": f"{_rung_dots['pearson_r'].median():.2f}",
            }
            for rung in panels.COUNT_LADDER
            if len(_rung_dots := ladder[ladder["min_count"] == rung])
        ]

    MANUSCRIPT_MEDIANS = pd.DataFrame(
        _medians(WITHIN_LADDER, "clade", "C", "median r across clades")
        + _medians(SHIFT_LADDER, "pair_label", "E", "median r across clade pairs")
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ### The medians the manuscript quotes

                Selection **{SELECTION}**, pooled over its models exactly as
                the panels pool them: one dot per (model, unit, rung). The
                paper's within-clade sentence cites panel C and its shift
                sentence cites the pairs drawn in panel D, whose correlations
                are panel E's dots.

                `main.tex` is checked against this table, not against a number
                carried across by hand.
                """
            ),
            MANUSCRIPT_MEDIANS,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
