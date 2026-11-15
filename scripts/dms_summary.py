"""Every predictor's correlation with the DMS, per homolog and per stratum.

One row per ``(homolog, predictor, stratum)``. No chart reads this table
yet -- the comparison has no drawn home until the DASM's own column lands -- so
what fixes its columns is the statistic that computes them rather than a panel
that consumes them. That statistic is ``within_clade_metrics``, the same one `fig:agreement` is built on,
rather than a second implementation of Pearson and Spearman: that function also
carries ``regression_slope`` and the reasoning for reading it as a comparand
between arms rather than as a calibration, and a figure with two correlation
functions in it would eventually disagree with itself.

Unclipped. `fig:agreement`'s ladder panels clip log selection factors to a range
justified *for* selection factors, which these predictors are not -- BLOSUM62 is
a log-odds table and the leaf predictor a log frequency ratio -- so nothing here
clips and nothing here names a reference predictor.

**Pearson is the readout**, because it is the coefficient the manuscript
reports. Spearman is carried beside it, in the column after, and is not a second
result: it is there because of the next paragraph.

**One warning about the leaf predictor's Pearson, which is a fact about its
scale.** It puts every cell whose mutant residue the clade has never shown at
``-log(parent + p)``, a band a tenth of a nat wide for two of the three clades,
so its values are bimodal rather than spread. A linear coefficient over that is
not reading what it looks like it is reading: on KP.3.1.1's ``multi_nt`` cells
its Pearson is +0.05 while its Spearman is +0.61, and on XBB.1.5's it is
-0.003. BLOSUM62 has no such problem -- it is a log-odds table and its Pearson
is readable. So a Pearson quoted for the leaf predictor should be quoted knowing
this, and nothing should plot that score on a linear axis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import assert_dasm2_is_this_repo, dms_homologs

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import dms
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

#: The CSV's columns, in order.
SUMMARY_COLUMNS = (
    "homolog",
    "predictor",
    "stratum",
    "pearson_r",
    "spearman_rho",
    "regression_slope",
    "n_mutations",
)


def _metrics(groups: dict[tuple[str, str, str], pd.DataFrame]):
    """``within_clade_metrics`` once per group, tagged with what the group was.

    One call per group and an ``assign``, which is how
    `notebooks/agreement_mutations.py` already borrows this function. Packing the
    three fields into one string to pass as its "clade" and splitting them back
    out would break on a homolog name containing the separator, and
    `workflow/figures.smk` escapes those names for its wildcards precisely
    because they are not plain identifiers.
    """
    return pd.concat(
        [
            within_clade_metrics({homolog: frame}, DMS_COLUMN)
            .drop(columns="clade")
            .assign(homolog=homolog, predictor=predictor, stratum=stratum)
            for (homolog, predictor, stratum), frame in groups.items()
        ],
        ignore_index=True,
    )


def summarize(
    merged: dict[str, pd.DataFrame],
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

    groups: dict[tuple[str, str, str], pd.DataFrame] = {}
    for homolog, frame in merged.items():
        for predictor in DMS_PREDICTORS:
            scored = frame.rename(columns={predictor: METRIC_COLUMN})
            for stratum in dms.STRATA:
                subset = (
                    scored
                    if stratum == dms.ALL
                    else scored[scored["stratum"] == stratum]
                )
                groups[(homolog, predictor, stratum)] = subset

    table = _metrics(groups)

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
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    loaded = [dms_models.read_scores(path) for path in args.scores]

    merged: dict[str, pd.DataFrame] = {}
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

    table = summarize(merged, manifests)
    output = args.output or panel_csv_path(DMS_SELECTION, DMS_PREDICTOR_CORRELATIONS)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    print(f"wrote {output}: {len(table)} rows")
    for row in table.itertuples():
        print(
            f"  {row.homolog:10s} {row.predictor:14s} {row.stratum:10s} "
            f"pearson={row.pearson_r:+.4f} spearman={row.spearman_rho:+.4f} "
            f"n={row.n_mutations}"
        )


if __name__ == "__main__":
    main()
