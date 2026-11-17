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
import dms_censor
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
#: the probability's penalty is only interpretable beside it: ``log p`` is
#: ``log f`` plus ``log lambda`` at unit weight, so what the penalty measures
#: is whether unit weight is the right weight. Its *marginal* correlation with
#: the assay is weakly negative, but that is mostly induced -- the whichmut
#: loss fits ``f`` as observed-over-opportunity, so ``f`` absorbs the inverse
#: of ``lambda`` and the two correlate negatively. See `tests/test_dms_prob.py`
#: and the PR that added this for the conditional picture, which differs.
NEUTRAL_RATE = "neutral_rate"

#: Which of ``build_probability_dataframe``'s columns each predictor is the
#: natural log of. Note that ``neutral_rate`` names both a predictor here and a
#: column there, and means a different thing on each side: the predictor's score
#: is ``log lambda``, the column holds ``lambda``.
#:
#: A mapping rather than a sequence, because the output paths are looked up by
#: predictor id in :func:`main`. A pair of positional sequences zipped together
#: would write the probabilities to the neutral-rate file if either were ever
#: reordered, and nothing downstream could catch it: ``write_scores`` validates
#: the frame against its manifest, and neither knows which path it is bound for.
SCORED_COLUMNS: dict[str, str] = {
    DASM_PROB: "whichmut_probability",
    NEUTRAL_RATE: "neutral_rate",
}


def linear_selection_factors(
    grid: pd.DataFrame, founder_aa: str, dtype: torch.dtype
) -> torch.Tensor:
    """``[L, 20]`` linear selection factors, from the long scored grid.

    The pivot is the one place a site or residue offset could enter unnoticed,
    so the ordering is checked here, where it is decided. The identity
    ``log p - log f - log lambda = -log Z`` can catch an offset too, but only
    when ``log f`` is re-read and re-joined from the scored table: recomputed
    from *this* tensor it is ``selection_factors * rate_to_aa / Z`` divided
    back out, and holds however wrong this tensor is.
    `tests/test_dms_prob.py` writes it in the form that catches one.

    Args:
        grid: :func:`dms_dasm.selection_factor_grid`'s rows for one clade.
        founder_aa: That clade's founder, whose length fixes ``L``.
        dtype: The neutral rates' dtype. ``aa_mutation_probabilities``
            multiplies the two, and torch refuses a mixed-dtype product.

    Returns:
        ``[L, 20]``, rows in site order and columns in ``AA_STR_SORTED`` order,
        in **linear** space.

    Raises:
        ValueError: If the grid is not exactly one row per (site, residue), or
            if reindexing it to ``1..L`` x ``AA_STR_SORTED`` leaves a hole.
            Between them those two cover a filtered grid, a grid still holding
            more than one model or clade, 0-indexed sites, and a missing
            residue -- each of which would otherwise reach
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

    # `reindex` rather than a check per axis: a hole after it is a site outside
    # `1..L` or a residue outside the alphabet, and with the row count already
    # fixed above, either one means the grid is indexed against something other
    # than this founder. Separate messages per axis cost two more raises and
    # say the same thing.
    ordered = grid.pivot(
        index="aa_site", columns="mutant_aa", values=dms_dasm.SCORE_COLUMN
    ).reindex(index=range(1, len(founder_aa) + 1), columns=list(AA_STR_SORTED))
    holes = int(ordered.isna().to_numpy().sum())
    if holes:
        raise ValueError(
            f"{holes} of {expected} cells are absent after reindexing the "
            f"scored grid to sites 1..{len(founder_aa)} and the twenty "
            f"residues. Sites are 1-indexed into the founder, and the model "
            f"scores all twenty at every one of them."
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


def _require_loggable(merged: pd.DataFrame, column: str) -> None:
    """Every cell in the declared stratum carries a positive ``column``.

    Both columns are checked rather than only the probability, although
    ``p = lambda * f / Z`` with ``f`` and ``Z`` positive makes a positive ``p``
    imply a positive ``lambda``. The implication fails exactly where ``p``
    underflows, and a message naming the column actually at fault is worth more
    than the line it costs.

    Raises:
        ValueError: If any cell is missing, zero or negative. These are cells
            the predictor declares itself able to speak on, so dropping them
            would leave the scored count short of the declared stratum without
            saying so.
    """
    flat = merged[merged[column].isna() | (merged[column] <= 0)]
    if not len(flat):
        return
    absent = int(flat[column].isna().sum())
    sample = sorted(zip(flat["aa_site"], flat["mutant_aa"]))[:5]
    raise ValueError(
        f"{len(flat)} of {len(merged)} single-nucleotide cells have no positive "
        f"{column}, so its log is undefined -- {absent} absent from the "
        f"probability grid and {len(flat) - absent} zero or negative. A cell "
        f"absent from the grid means the grid and the assay disagree about the "
        f"site range; a zero means the stratum and the neutral rates disagree "
        f"about which mutations one base change reaches. First few: {sample}."
    )


def founder_frame(
    clade: str, founder_aa: str, table_path: str | Path, archive_path: str | Path
) -> pd.DataFrame:
    """Every single-nucleotide amino-acid mutation to one founder, scored.

    **The whole protein, not the measured part of it.** `fig:probabilities`'
    scatter is about a property of the model on its founder -- how a mutation
    probability sits relative to its fitness effect -- and that question has
    nothing to do with which cells an assay happened to cover. Restricting it
    to measured cells would answer it on a subset chosen by a different
    method. :func:`prob_scores` narrows to the assay because it feeds a
    correlation *against* the assay; this does not, because nothing here is
    correlated against anything.

    What is dropped is only what has no value or was never trained:

    * **Synonymous cells**, which are not mutations of the residue.
    * **Multi-nucleotide cells**, where ``lambda`` is exactly zero, so
      ``log p`` is undefined rather than small. That is the figure's stratum,
      and it is a definition rather than a filter.
    * **Censored and masked mutations**, via `scripts/dms_censor.py`, because
      the model was told to ignore them. The same mask every other population
      in this pipeline carries, so the panel is not the odd one out; it is a
      small share of the grid either way.

    Returns:
        One row per drawn point, carrying the three quantities under the same
        names the predictor join gives them -- ``dasm``, ``dasm_prob`` and
        ``neutral_rate`` -- so the panel reads one vocabulary throughout.
    """
    probabilities = probability_frame(clade, founder_aa, table_path, archive_path)
    grid = dms_dasm.selection_factor_grid(clade, table_path)

    merged = probabilities.merge(
        grid[[*dms_models.JOIN_KEY, dms_dasm.SCORE_COLUMN]],
        on=list(dms_models.JOIN_KEY),
        how="inner",
        validate="one_to_one",
    )

    # `parent_aa` before the filter rather than after it, because
    # `trained_single_nt_cells` reads the synonymous test off it. The two
    # spellings agree: `build_probability_dataframe` sets `is_synonymous` from
    # the same founder this column is taken from, and `_check_grid_founder`
    # has already refused a grid whose parent residues are another clade's.
    merged = merged.assign(
        parent_aa=[founder_aa[site - 1] for site in merged["aa_site"]]
    )
    drawn = dms_models.trained_single_nt_cells(
        merged,
        dms.single_nt_indicator(clade, lib.GENE),
        dms_censor.censored_mutations(clade),
    )

    probability = SCORED_COLUMNS[DASM_PROB]
    positive = drawn[probability] > 0
    if not positive.all():
        raise ValueError(
            f"{int((~positive).sum())} of {len(drawn)} single-nucleotide cells "
            f"have no positive {probability}. Every cell kept here is "
            f"one base from the founder's codon, so the stratum and the "
            f"neutral rates disagree about reachability."
        )

    return drawn.assign(
        dasm=drawn[dms_dasm.SCORE_COLUMN],
        dasm_prob=np.log(drawn[probability]),
        neutral_rate=np.log(drawn[SCORED_COLUMNS[NEUTRAL_RATE]]),
    )[
        [
            "aa_site",
            "parent_aa",
            "mutant_aa",
            "dasm",
            "dasm_prob",
            "neutral_rate",
        ]
    ]


def prob_scores(
    homolog: str, table_path: str | Path, archive_path: str | Path
) -> tuple[dict[str, tuple[pd.DataFrame, dms_models.Manifest]], str]:
    """Both predictors' scores for one homolog, and the founder they read.

    One function for the pair rather than one per predictor: the archive load
    and the partition function are the expensive half and are shared, so a
    second script would run them again for the same founder.

    Returns:
        ``({predictor id: (frame, manifest)}, founder)``. Keyed rather than
        ordered so that a caller binds each frame to its output path by name --
        see :data:`SCORED_COLUMNS`.
    """
    cells, founder_aa = dms_models.homolog_cells(homolog)
    clade = dms_dataset(homolog).clade
    frame = probability_frame(clade, founder_aa, table_path, archive_path)

    reachable = cells[cells["stratum"] == dms.SINGLE_NT]
    merged = reachable.merge(
        frame[[*dms_models.JOIN_KEY, *SCORED_COLUMNS.values()]],
        on=list(dms_models.JOIN_KEY),
        how="left",
        validate="one_to_one",
    )
    for column in SCORED_COLUMNS.values():
        _require_loggable(merged, column)

    # One manifest per predictor, differing only in `model`: both read the same
    # founder through the same archive, which is what the frame string records.
    described = (
        f"clade founder {clade} {lib.GENE}, from mat-pcp's own clade_founders "
        f"FASTA; 1-indexed, so no site mapping. Neutral rates from "
        f"{Path(archive_path).name}"
    )
    digest = dms_models.sequence_sha256(founder_aa)
    return {
        model: (
            merged.assign(model=model, score=np.log(merged[column]))[
                list(dms_models.PREDICTOR_COLUMNS)
            ],
            dms_models.Manifest(
                model=model,
                strata=(dms.SINGLE_NT,),
                frame=described,
                input_sequence_sha256=digest,
            ),
        )
        for model, column in SCORED_COLUMNS.items()
    }, founder_aa


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

    # Bound by predictor id rather than by position: `write_scores` validates a
    # frame against its manifest, and neither of them knows which path it is
    # being written to, so a swap here would be silent.
    outputs = {DASM_PROB: args.output, NEUTRAL_RATE: args.neutral_output}
    scored, sequence = prob_scores(args.homolog, args.table, args.neutral_archive)
    for model, (frame, manifest) in scored.items():
        dms_models.write_scores(frame, manifest, sequence, outputs[model])
        print(f"wrote {outputs[model]}: {len(frame)} cells for {args.homolog}")


if __name__ == "__main__":
    main()
