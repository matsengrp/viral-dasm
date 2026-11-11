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

**Spearman is the readout.** The leaf predictor puts every cell whose mutant
residue the clade has never shown at one site-level floor, which makes its scale
bimodal and a linear correlation unreadable: on KP.3.1.1's ``multi_nt`` cells its
Spearman is +0.58 and its Pearson +0.04. Pearson is kept because a reader will
look for it and because it is readable for BLOSUM62, not because it ranks these
predictors.

``*_mutant_observed`` is the same correlation restricted to the cells where the
clade's leaves carry the mutant residue at all. It is a property of the cells
rather than of a predictor, so it applies to every column and answers the
question the headline invites: how much of the leaf predictor's lead is the
observed/unobserved split rather than graded frequency.
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
from panel_spec import DMS_PREDICTOR_CORRELATIONS, DMS_SELECTION, panel_csv_path

from dasm2.analysis.viral_fitness_eval import within_clade_metrics
from dasm2.sequence.constants import AA_STR_SORTED

#: The DMS column every predictor is correlated against.
DMS_COLUMN = "dms_cell_entry"

#: What ``within_clade_metrics`` requires its predictor column to be called. The
#: score is renamed to it per group rather than that function being generalised:
#: it is `fig:agreement`'s statistic and this figure borrows it, so the borrowing
#: happens here.
METRIC_COLUMN = "log_selection_factor"

#: The composite key passed as ``within_clade_metrics``' "clade", split back out
#: on the way to the CSV. That function keys its rows by one string, and what
#: this figure groups by is three fields.
KEY_SEPARATOR = "|"

#: The CSV's columns, in order.
SUMMARY_COLUMNS = (
    "homolog",
    "predictor",
    "stratum",
    "pearson_r",
    "spearman_rho",
    "regression_slope",
    "n_mutations",
    "pearson_r_mutant_observed",
    "spearman_rho_mutant_observed",
    "n_mutant_observed",
)


def mutant_observed(cells: pd.DataFrame, counts_path: Path) -> pd.Series:
    """Whether the clade's leaves carry each cell's mutant residue at all."""
    counts = dms_leaf_freq.read_counts(counts_path).to_numpy()
    observed = counts[
        cells["aa_site"].to_numpy() - 1,
        [AA_STR_SORTED.index(aa) for aa in cells["mutant_aa"]],
    ]
    return pd.Series(observed > 0, index=cells.index)


def _metrics(groups: dict[str, pd.DataFrame], suffix: str) -> pd.DataFrame:
    """``within_clade_metrics`` over the composite keys, renamed for the CSV."""
    table = within_clade_metrics(groups, DMS_COLUMN)
    table = table.rename(
        columns={
            name: f"{name}{suffix}"
            for name in ("pearson_r", "spearman_rho", "regression_slope", "n_mutations")
        }
    )
    keys = table["clade"].str.split(KEY_SEPARATOR, expand=True)
    table[["homolog", "predictor", "stratum"]] = keys
    return table.drop(columns="clade")


def summarize(merged: dict[str, pd.DataFrame], observed: dict[str, pd.Series]):
    """Every ``(homolog, predictor, stratum)`` row, from the joined frames.

    ``all`` is the **union** of the two strata the ``stratum`` column carries,
    not a third value in it: ``melt_dms`` labels only ``single_nt`` and
    ``multi_nt``, and `scripts/dms.py` names ``all`` precisely because a readout
    that reports it has to say the three partition rather than nest.
    """
    full: dict[str, pd.DataFrame] = {}
    restricted: dict[str, pd.DataFrame] = {}
    for homolog, frame in merged.items():
        for predictor in dms_models.PREDICTORS:
            if predictor not in frame.columns:
                continue
            scored = frame.rename(columns={predictor: METRIC_COLUMN})
            for stratum in dms.STRATA:
                subset = (
                    scored
                    if stratum == dms.ALL
                    else scored[scored["stratum"] == stratum]
                )
                key = KEY_SEPARATOR.join((homolog, predictor, stratum))
                full[key] = subset
                restricted[key] = subset[observed[homolog].reindex(subset.index)]

    table = _metrics(full, "").merge(
        _metrics(restricted, "_mutant_observed"),
        on=["homolog", "predictor", "stratum"],
        how="outer",
        validate="one_to_one",
    )
    # The restricted pass keeps only its two correlations and its own `n`: a
    # second regression slope over a subset of the same cells says nothing the
    # first does not, and leaving it in would invite reading two slopes as a
    # comparison.
    table = table.rename(columns={"n_mutations_mutant_observed": "n_mutant_observed"})
    table = table.drop(columns=["regression_slope_mutant_observed"])

    # The partition `summarize`'s docstring asserts, checked rather than stated:
    # an `all` row that is not the two strata summed means a cell landed in
    # neither or in both.
    for (homolog, predictor), rows in table.groupby(["homolog", "predictor"]):
        counts = rows.set_index("stratum")["n_mutations"]
        if counts[dms.ALL] != counts[dms.SINGLE_NT] + counts[dms.MULTI_NT]:
            raise ValueError(
                f"{homolog}/{predictor}: {counts[dms.ALL]} cells in `all` against "
                f"{counts[dms.SINGLE_NT]} + {counts[dms.MULTI_NT]} in the two "
                f"strata, so the three do not partition"
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
    for homolog in dms_homologs():
        cells = dms_models.homolog_cells(homolog)
        for_homolog = [
            (frame[frame["homolog"] == homolog], manifest)
            for frame, manifest in loaded
            if homolog in set(frame["homolog"].unique())
        ]
        if not for_homolog:
            continue
        merged[homolog], _ = dms_models.join_scores(cells, for_homolog)
        clade = dms_dataset(homolog).clade
        if clade not in by_clade:
            raise SystemExit(
                f"{homolog}: no leaf tally passed for clade {clade}; got "
                f"{sorted(by_clade)}"
            )
        observed[homolog] = mutant_observed(merged[homolog], by_clade[clade])

    table = summarize(merged, observed)
    output = args.output or panel_csv_path(DMS_SELECTION, DMS_PREDICTOR_CORRELATIONS)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    print(f"wrote {output}: {len(table)} rows")
    for row in table.itertuples():
        print(
            f"  {row.homolog:10s} {row.predictor:14s} {row.stratum:10s} "
            f"spearman={row.spearman_rho:+.4f} pearson={row.pearson_r:+.4f} "
            f"n={row.n_mutations}  (mutant observed: "
            f"spearman={row.spearman_rho_mutant_observed:+.4f} "
            f"n={row.n_mutant_observed})"
        )


if __name__ == "__main__":
    main()
