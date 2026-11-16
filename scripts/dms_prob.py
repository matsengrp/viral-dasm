"""The DASM's own mutation probability as a predictor of the spike DMS.

A selection factor says how much a mutation is *favoured*; whichmut's
probability says how often it is actually *taken*, which is that factor
weighted by how often the neutral process offers it::

    p_{j,c->c'} = lambda_{j,c->c'} * f_{j,a->a'} / Z

So on the log scale a probability is a fitness effect plus ``log lambda`` -- a
statement about mutational opportunity, which a cell-entry assay has no reason
to track. This module scores both terms against the same assays `fig:dms` and
`fig:dms_strata` score ``log f`` against, so the cost of predicting with the
probability can be read off one table.

**Single-nucleotide only, and that is the model's definition rather than a
filter.** ``lambda`` is zero off the single-nucleotide set, so ``p`` is zero
and ``log p`` is undefined on every amino-acid mutation that needs two or three
base changes. Both predictors here declare ``strata=(dms.SINGLE_NT,)`` and emit
no row outside it: ``dms_models.validate_scores`` refuses a NaN or an infinity,
and ``join_scores`` fills an absent cell as an absence, which is what a
predictor that cannot speak is supposed to look like.

**No inference, and no forward pass.** ``selection_factors.parquet`` already
holds ``FIGURE_MODEL``'s whole ``[L, 20]`` grid per clade, gauge-fixed so that
every synonymous cell is exactly ``log f = 0``. Exponentiating it recovers the
linear factors :func:`aa_mutation_probabilities` wants, so this module claims no
GPU and loads no checkpoint -- it reads a Parquet and a neutral-rate archive.
That is also why it is safe to run in a repository whose DAG would otherwise
schedule training: nothing here depends on ``rule score``.

**The logs are natural.** ``build_probability_dataframe`` offers a base-10
column; this module does not use it. ``log_selection_factor`` is a natural log
and the identity ``log p = log f + log lambda - log Z`` holds in natural logs,
so a base-10 column here would leave a slope in a residual that is supposed to
be a constant. `tests/test_dms_prob.py` asserts that constant across the two
written files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
import lib
from lib import assert_dasm2_is_this_repo, dms_dataset, dms_homologs

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import dms
import dms_dasm
import dms_models
import numpy as np
import pandas as pd
import torch
from founders import founder_nt_sequence

from dasm2.analysis.whichmut_probs import (
    aa_mutation_probabilities,
    build_probability_dataframe,
    founder_rates,
)
from dasm2.neutral.tabular_rates import TabularRatesNeutralModel
from dasm2.sequence.constants import AA_STR_SORTED

#: The probability predictor's id, as `results/panels/dms/` carries it and
#: :data:`panel_spec.DMS_PREDICTORS` lists it.
DASM_PROB = "dasm_prob"

#: The neutral-rate predictor's id. Carried as a predictor of its own because
#: the size of the probability's penalty is only interpretable beside it: a
#: rate that predicted the assay well would make ``log p`` a blend of two
#: informative signals rather than one signal plus an irrelevance.
NEUTRAL_RATE = "neutral_rate"

#: ``build_probability_dataframe``'s own columns. :data:`LINEAR_RATE_COLUMN`
#: collides by spelling with :data:`NEUTRAL_RATE` and is a different quantity --
#: that one is a predictor id whose score is ``log lambda``, this one is a
#: column holding ``lambda`` itself. Named apart so neither can be typed where
#: the other belongs.
PROBABILITY_COLUMN = "whichmut_probability"
LINEAR_RATE_COLUMN = "neutral_rate"

#: What each predictor's ``score`` column is, in the order the two outputs are
#: written. The frame description in each manifest names the archive as well as
#: the founder, because a probability is a property of the pair.
SCORED_QUANTITIES = (
    (DASM_PROB, PROBABILITY_COLUMN),
    (NEUTRAL_RATE, LINEAR_RATE_COLUMN),
)


def linear_selection_factors(
    grid: pd.DataFrame, founder_aa: str, dtype: torch.dtype
) -> torch.Tensor:
    """``[L, 20]`` linear selection factors, from the long scored grid.

    The pivot is the one place a site or residue offset could enter unnoticed,
    and the identity test downstream cannot catch it -- ``probabilities`` is
    ``selection_factors * rate_to_aa / Z`` by construction, so an identity
    recomputed from this tensor holds however wrong this tensor is. So the
    checks are here, where the ordering is actually decided.

    Args:
        grid: :func:`dms_dasm.selection_factor_grid`'s rows for one clade.
        founder_aa: That clade's founder, whose length fixes ``L``.
        dtype: The neutral rates' dtype. ``aa_mutation_probabilities``
            multiplies the two, and torch refuses a mixed-dtype product.

    Returns:
        ``[L, 20]``, rows in site order and columns in ``AA_STR_SORTED`` order,
        in **linear** space.

    Raises:
        ValueError: If the grid is not exactly one row per (site, residue), if
            any site in ``1..L`` is missing, if any residue is missing, or if
            any factor is null. Each would otherwise reach
            ``aa_mutation_probabilities`` as a silently reindexed tensor.
    """
    expected = len(founder_aa) * len(AA_STR_SORTED)
    if len(grid) != expected:
        raise ValueError(
            f"the scored grid has {len(grid)} rows for a founder of "
            f"{len(founder_aa)} residues; expected {expected} = "
            f"{len(founder_aa)} sites x {len(AA_STR_SORTED)} amino acids. A "
            f"short grid means the table was filtered, and a long one means it "
            f"still holds more than one model, clade or arm."
        )

    pivot = grid.pivot(
        index="aa_site", columns="mutant_aa", values=dms_dasm.SCORE_COLUMN
    )
    sites = list(range(1, len(founder_aa) + 1))
    if list(pivot.index) != sites:
        raise ValueError(
            f"the scored grid's sites are not 1..{len(founder_aa)}; it runs "
            f"{pivot.index.min()}..{pivot.index.max()} over {len(pivot.index)} "
            f"distinct values. Sites are 1-indexed into the founder."
        )
    missing = sorted(set(AA_STR_SORTED) - set(pivot.columns))
    if missing:
        raise ValueError(
            f"the scored grid is missing {len(missing)} residues: {missing}. "
            f"The model scores all twenty at every site."
        )

    ordered = pivot[list(AA_STR_SORTED)]
    if ordered.isna().to_numpy().any():
        raise ValueError(
            f"{int(ordered.isna().to_numpy().sum())} cells of the scored grid "
            f"are null, so the pivot did not cover every (site, residue) pair."
        )
    return torch.tensor(np.exp(ordered.to_numpy()), dtype=dtype)


def _check_grid_founder(grid: pd.DataFrame, founder_aa: str) -> None:
    """The scored grid's parent residues are the founder's own.

    Both come from the same clade, so this agrees in every case the pipeline
    builds. It is asserted anyway because this is the first producer where a
    clade mismatch would give a wrong ``lambda`` rather than a wrong score --
    the selection factors would be the other clade's and look unremarkable,
    while the neutral rates would be this one's.
    """
    declared = [founder_aa[site - 1] for site in grid["aa_site"]]
    disagree = grid[np.asarray(declared) != grid["parent_aa"].to_numpy()]
    if len(disagree):
        row = disagree.iloc[0]
        raise ValueError(
            f"the scored grid's parent_aa disagrees with the founder at "
            f"{len(disagree)} of {len(grid)} rows, e.g. site "
            f"{int(row['aa_site'])}: the table says {row['parent_aa']!r} and "
            f"the founder says {founder_aa[int(row['aa_site']) - 1]!r}. The "
            f"table was read for a different clade than the founder FASTA."
        )


def probability_frame(
    clade: str, founder_aa: str, table_path: str | Path, archive_path: str | Path
) -> pd.DataFrame:
    """One founder's whichmut probabilities and neutral rates, one row per cell.

    Columns are ``build_probability_dataframe``'s, over the whole ``L x 20``
    grid -- the cells no assay measured, and the zero-probability cells no
    single nucleotide change reaches, both included. Narrowing happens in
    :func:`prob_scores`, against the stratum rather than against the zeros, so
    that a cell which *should* carry a probability and does not is an error
    instead of a silent drop.
    """
    grid = dms_dasm.selection_factor_grid(clade, table_path)
    _check_grid_founder(grid, founder_aa)

    neutral_model = TabularRatesNeutralModel.from_pretrained(str(archive_path))
    rates = founder_rates(neutral_model, founder_nt_sequence(clade, lib.GENE))
    linear = linear_selection_factors(grid, founder_aa, rates.codon_rates.dtype)
    probabilities, rate_to_aa = aa_mutation_probabilities(linear, rates)
    return build_probability_dataframe(probabilities, rate_to_aa, founder_aa)


def prob_scores(
    homolog: str, table_path: str | Path, archive_path: str | Path
) -> tuple[list[tuple[pd.DataFrame, dms_models.Manifest]], str]:
    """Both predictors' scores for one homolog, and the founder they read.

    One function for the pair rather than one per predictor: the archive load
    and the partition function are the expensive half and are shared, and a
    second script would run them again for the same founder.

    Returns:
        ``([(frame, manifest), ...], founder)``, in :data:`SCORED_QUANTITIES`
        order.

    Raises:
        ValueError: If a single-nucleotide cell has no probability. That cell
            is one the predictor declares itself able to speak on, so dropping
            it would leave the scored count short of the declared stratum
            without saying so.
    """
    cells, founder_aa = dms_models.homolog_cells(homolog)
    clade = dms_dataset(homolog).clade
    frame = probability_frame(clade, founder_aa, table_path, archive_path)

    reachable = cells[cells["stratum"] == dms.SINGLE_NT]
    merged = reachable.merge(
        frame[[*dms_models.JOIN_KEY, PROBABILITY_COLUMN, LINEAR_RATE_COLUMN]],
        on=list(dms_models.JOIN_KEY),
        how="left",
        validate="one_to_one",
    )

    out: list[tuple[pd.DataFrame, dms_models.Manifest]] = []
    for model, column in SCORED_QUANTITIES:
        flat = merged[merged[column].isna() | (merged[column] <= 0)]
        if len(flat):
            sample = sorted(zip(flat["aa_site"], flat["mutant_aa"]))[:5]
            raise ValueError(
                f"{model}: {len(flat)} of {len(merged)} single-nucleotide cells "
                f"have no positive {column}, so its log is undefined. Every "
                f"cell in this stratum is reachable by one base change and must "
                f"carry a positive neutral rate; a zero means the stratum and "
                f"the rates disagree about reachability. First few: {sample}."
            )
        scored = merged.assign(model=model, score=np.log(merged[column]))
        manifest = dms_models.Manifest(
            model=model,
            strata=(dms.SINGLE_NT,),
            frame=(
                f"clade founder {clade} {lib.GENE}, from mat-pcp's own "
                f"clade_founders FASTA; 1-indexed, so no site mapping. Neutral "
                f"rates from {Path(archive_path).name}"
            ),
            input_sequence_sha256=dms_models.sequence_sha256(founder_aa),
        )
        out.append((scored[list(dms_models.PREDICTOR_COLUMNS)], manifest))
    return out, founder_aa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--homolog", required=True, choices=dms_homologs())
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument(
        "--neutral-archive",
        required=True,
        type=Path,
        help="the era's TabularRates archive, resolved by the rule",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--neutral-output", required=True, type=Path)
    args = parser.parse_args()

    scored, sequence = prob_scores(args.homolog, args.table, args.neutral_archive)
    for (frame, manifest), path in zip(scored, (args.output, args.neutral_output)):
        dms_models.write_scores(frame, manifest, sequence, path)
        print(f"wrote {path}: {len(frame)} cells for {args.homolog}")


if __name__ == "__main__":
    main()
