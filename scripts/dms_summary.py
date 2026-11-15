"""Every predictor's correlation with the DMS, per homolog and per stratum.

One row per ``(homolog, predictor, stratum)``, in the columns
`scripts/panel_charts.py::homolog_correlation_panel` already reads. The
statistic is ``within_clade_metrics``, the same one `fig:agreement` is built on,
rather than a second implementation of Pearson and Spearman: that function also
carries ``regression_slope`` and the reasoning for reading it as a comparand
between arms rather than as a calibration, and a figure with two correlation
functions in it would eventually disagree with itself.

Unclipped. `fig:agreement`'s ladder panels clip log selection factors to a range
justified *for* selection factors, which these predictors are not -- BLOSUM62 is
a log-odds table and the leaf predictor a log frequency ratio -- so nothing here
clips and nothing here names a reference predictor.

**Two altitudes, both reported, because the predictors live at different ones.**
The pooled correlation runs over every measured cell, so it is dominated by
variance *between* sites -- and the leaf predictor is very nearly a per-site
statistic while BLOSUM62 carries no site information at all. Pooled, the leaf
predictor leads on all three homologs; centre both the score and the measurement
on their per-site means and the ordering reverses, with BLOSUM62 the better
substitution-level predictor everywhere. ``*_within_site`` is that second
readout, over sites with at least :data:`WITHIN_SITE_MIN_CELLS` measured cells.
Reporting only the pooled number would invite reading it as a statement about
which substitution at a site is tolerated, which it is not.

**Spearman is the readout.** The leaf predictor puts every cell whose mutant
residue the clade has never shown at ``-log(parent + p)``, a band a tenth of a
nat wide for two of the three clades, so its scale is bimodal and a linear
correlation unreadable: on KP.3.1.1's ``multi_nt`` cells its Spearman is +0.58
and its Pearson +0.04. Pearson is kept because a reader will look for it and
because it is readable for BLOSUM62, not because it ranks these predictors.
Nothing should plot this score on an axis.

``*_mutant_observed`` is the same correlation restricted to the cells whose
mutant residue the clade's leaves carry at all. It is a property of the cells
rather than of a predictor, so it applies to every column. It is **not** a
decomposition of the pooled number: restricting to observed cells also selects
tolerant, well-sampled sites, so a drop there is a changed cell set rather than
an isolated term. Correlations over fewer than
:data:`MUTANT_OBSERVED_MIN_CELLS` cells are left blank rather than printed,
because a coefficient over thirty cells in a tracked table reads as a
measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import assert_dasm2_is_this_repo, dms_dataset, dms_homologs

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import dms
import dms_leaf_freq
import dms_models
import pandas as pd
from panel_spec import (
    DMS_PREDICTOR_CORRELATIONS,
    DMS_PREDICTORS,
    DMS_SELECTION,
    panel_csv_path,
)

from dasm2.analysis.viral_fitness_eval import within_clade_metrics

#: The DMS column every predictor is correlated against.
DMS_COLUMN = "dms_cell_entry"

#: What ``within_clade_metrics`` requires its predictor column to be called. The
#: score is renamed to it per group rather than that function being generalised:
#: it is `fig:agreement`'s statistic and this figure borrows it, so the borrowing
#: happens here.
METRIC_COLUMN = "log_selection_factor"

#: Cells per site required before a site contributes to the within-site readout.
#: Below this the per-site mean is itself mostly noise, and centring on it adds
#: more variance than it removes.
WITHIN_SITE_MIN_CELLS = 5

#: Cells required before a ``*_mutant_observed`` correlation is printed at all.
#: KP.3.1.1 and XBB.1.5 have 30 and 29 cells with an observed mutant in their
#: ``multi_nt`` stratum; a four-decimal coefficient over those reads as a result
#: and is not one. `within_clade_metrics` only refuses below three.
MUTANT_OBSERVED_MIN_CELLS = 100

#: The CSV's columns, in order.
SUMMARY_COLUMNS = (
    "homolog",
    "predictor",
    "stratum",
    "pearson_r",
    "spearman_rho",
    "regression_slope",
    "n_mutations",
    "spearman_rho_within_site",
    "n_within_site",
    "pearson_r_mutant_observed",
    "spearman_rho_mutant_observed",
    "n_mutant_observed",
)


def centre_on_site_means(frame: pd.DataFrame) -> pd.DataFrame:
    """One stratum's cells with the score and the measurement site-mean-centred.

    What is left is the within-site comparison: given this site, does the
    predictor rank *which* substitution the assay tolerates. Sites with fewer
    than :data:`WITHIN_SITE_MIN_CELLS` cells are dropped rather than centred on a
    one- or two-cell mean.
    """
    sized = frame.groupby("aa_site")[METRIC_COLUMN].transform("size")
    kept = frame[sized >= WITHIN_SITE_MIN_CELLS]
    if kept.empty:
        return kept
    centred = {
        column: kept[column] - kept.groupby("aa_site")[column].transform("mean")
        for column in (METRIC_COLUMN, DMS_COLUMN)
    }
    return kept.assign(**centred)


def mutant_observed(cells: pd.DataFrame, counts_path: Path) -> pd.Series:
    """Whether the clade's leaves carry each cell's mutant residue at all."""
    counts = dms_leaf_freq.read_counts(counts_path)
    observed = dms_leaf_freq.counts_at(
        counts, cells["aa_site"].to_numpy(), cells["mutant_aa"]
    )
    return pd.Series(observed > 0, index=cells.index)


def _metrics(groups: dict[tuple[str, str, str], pd.DataFrame], suffix: str):
    """``within_clade_metrics`` once per group, tagged with what the group was.

    One call per group and an ``assign``, which is how
    `notebooks/agreement_mutations.py` already borrows this function. Packing the
    three fields into one string to pass as its "clade" and splitting them back
    out would break on a homolog name containing the separator, and
    `workflow/figures.smk` escapes those names for its wildcards precisely
    because they are not plain identifiers.
    """
    renames = {
        name: f"{name}{suffix}"
        for name in ("pearson_r", "spearman_rho", "regression_slope", "n_mutations")
    }
    return pd.concat(
        [
            within_clade_metrics({homolog: frame}, DMS_COLUMN)
            .drop(columns="clade")
            .rename(columns=renames)
            .assign(homolog=homolog, predictor=predictor, stratum=stratum)
            for (homolog, predictor, stratum), frame in groups.items()
        ],
        ignore_index=True,
    )


def summarize(
    merged: dict[str, pd.DataFrame],
    observed: dict[str, pd.Series],
    manifests: dict[str, dict[str, dict]],
):
    """Every ``(homolog, predictor, stratum)`` row, from the joined frames.

    ``all`` is the **union** of the two strata the ``stratum`` column carries,
    not a third value in it: ``melt_dms`` labels only ``single_nt`` and
    ``multi_nt``, and `scripts/dms.py` names ``all`` precisely because a readout
    that reports it has to say the three partition rather than nest.
    """
    missing = {
        (homolog, predictor)
        for homolog, frame in merged.items()
        for predictor in DMS_PREDICTORS
        if predictor not in frame.columns
    }
    if missing:
        raise ValueError(
            f"no scores joined for {sorted(missing)}. Every predictor is expected "
            f"for every homolog; a short table would otherwise report a figure "
            f"built from fewer arms than the panel names."
        )

    full: dict[tuple[str, str, str], pd.DataFrame] = {}
    restricted: dict[tuple[str, str, str], pd.DataFrame] = {}
    within: dict[tuple[str, str, str], pd.DataFrame] = {}
    for homolog, frame in merged.items():
        for predictor in DMS_PREDICTORS:
            scored = frame.rename(columns={predictor: METRIC_COLUMN})
            for stratum in dms.STRATA:
                subset = (
                    scored
                    if stratum == dms.ALL
                    else scored[scored["stratum"] == stratum]
                )
                key = (homolog, predictor, stratum)
                full[key] = subset
                restricted[key] = subset[observed[homolog].reindex(subset.index)]
                within[key] = centre_on_site_means(subset)

    table = _metrics(full, "").merge(
        _metrics(restricted, "_mutant_observed"),
        on=["homolog", "predictor", "stratum"],
        how="outer",
        validate="one_to_one",
    )
    table = table.merge(
        _metrics(within, "_within_site").drop(
            columns=["pearson_r_within_site", "regression_slope_within_site"]
        ),
        on=["homolog", "predictor", "stratum"],
        how="outer",
        validate="one_to_one",
    )
    table = table.rename(columns={"n_mutations_within_site": "n_within_site"})
    # The restricted pass keeps only its two correlations and its own `n`: a
    # second regression slope over a subset of the same cells says nothing the
    # first does not, and leaving it in would invite reading two slopes as a
    # comparison.
    table = table.rename(columns={"n_mutations_mutant_observed": "n_mutant_observed"})
    table = table.drop(columns=["regression_slope_mutant_observed"])

    # Blank, not printed: see MUTANT_OBSERVED_MIN_CELLS.
    thin = table["n_mutant_observed"] < MUTANT_OBSERVED_MIN_CELLS
    table.loc[thin, ["pearson_r_mutant_observed", "spearman_rho_mutant_observed"]] = (
        float("nan")
    )

    # Coverage, not the partition. `all` is the unfiltered frame and the two
    # strata are filters on it, so they sum by construction -- `melt_dms` labels
    # every cell and raises on a NaN, so the sum could not fail. What *can*
    # happen, and what this catches, is a predictor whose `n` is smaller than the
    # cells it was handed: `within_clade_metrics` drops any row where either
    # column is NaN, so an unjoined or unscored cell leaves the CSV reporting a
    # confident correlation over fewer mutations than the assay measured.
    for (homolog, predictor), rows in table.groupby(["homolog", "predictor"]):
        # Against the cells in the strata the predictor *declares*, not against
        # every measured cell. A predictor may legitimately speak on a subset --
        # `Manifest` calls that an absence rather than a zero, and `join_scores`
        # allows it -- so comparing against the whole assay would refuse the first
        # partial-coverage arm. What is still caught is the thing worth catching:
        # a cell inside a declared stratum that failed to join, or whose score
        # came back NaN, which `within_clade_metrics` drops rather than raising.
        declared = dms_models.declared_cell_strata(
            tuple(manifests[homolog][predictor]["strata"])
        )
        cells = merged[homolog]
        expected = int(cells["stratum"].isin(declared).sum())
        reported = int(rows.set_index("stratum")["n_mutations"][dms.ALL])
        if reported != expected:
            raise ValueError(
                f"{homolog}/{predictor}: reports {reported} mutations against the "
                f"{expected} cells in the strata it declares ({sorted(declared)}). "
                f"A cell that failed to join, or a score that came back NaN, is "
                f"dropped by `within_clade_metrics` rather than raising."
            )

    ordered = pd.Categorical(
        table["stratum"], categories=list(dms.STRATA), ordered=True
    )
    table = table.assign(stratum=ordered).sort_values(
        ["homolog", "predictor", "stratum"]
    )
    return table[list(SUMMARY_COLUMNS)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        required=True,
        nargs="+",
        type=Path,
        help="one predictor's scores for one homolog, repeated",
    )
    parser.add_argument(
        "--counts",
        required=True,
        nargs="+",
        type=Path,
        help="one clade's leaf tally, repeated; names the mutant-observed subset",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    by_clade = {path.stem: path for path in args.counts}
    loaded = [dms_models.read_scores(path) for path in args.scores]

    merged: dict[str, pd.DataFrame] = {}
    observed: dict[str, pd.Series] = {}
    manifests: dict[str, dict[str, dict]] = {}
    for homolog in dms_homologs():
        cells, _ = dms_models.homolog_cells(homolog)
        for_homolog = [
            (frame[frame["homolog"] == homolog], manifest)
            for frame, manifest in loaded
            if homolog in set(frame["homolog"].unique())
        ]
        if len(for_homolog) != len(DMS_PREDICTORS):
            raise SystemExit(
                f"{homolog}: {len(for_homolog)} of {len(DMS_PREDICTORS)} "
                f"predictors passed. Skipping one would shorten the table rather "
                f"than fail the rule."
            )
        merged[homolog], manifests[homolog] = dms_models.join_scores(cells, for_homolog)
        clade = dms_dataset(homolog).clade
        if clade not in by_clade:
            raise SystemExit(
                f"{homolog}: no leaf tally passed for clade {clade}; got "
                f"{sorted(by_clade)}"
            )
        observed[homolog] = mutant_observed(merged[homolog], by_clade[clade])

    table = summarize(merged, observed, manifests)
    output = args.output or panel_csv_path(DMS_SELECTION, DMS_PREDICTOR_CORRELATIONS)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    print(f"wrote {output}: {len(table)} rows")
    for row in table.itertuples():
        print(
            f"  {row.homolog:10s} {row.predictor:14s} {row.stratum:10s} "
            f"pooled={row.spearman_rho:+.4f} n={row.n_mutations:<6d} "
            f"within-site={row.spearman_rho_within_site:+.4f} "
            f"n={row.n_within_site:<6d} "
            f"mutant-observed={row.spearman_rho_mutant_observed:+.4f} "
            f"n={row.n_mutant_observed}"
        )


if __name__ == "__main__":
    main()
