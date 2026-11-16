"""`scripts/dms_prob.py`: the DASM's probability, and the rate behind it.

The guard that refuses a *log* selection factor where a linear one belongs is
not re-pinned here. `tests/test_dms.py` already does it, and more thoroughly
than a second copy would: it asserts the passing leg as well, including a
legitimate zero factor so the guard has to be ``< 0`` rather than ``<= 0``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import dms
import dms_dasm
import dms_models
import dms_prob
import lib
import numpy as np
import panel_spec
import torch
from build_era_neutral_models import archive_name
from founders import founder_nt_sequence
from test_dms_models import needs_dms

from dasm2.analysis.whichmut_probs import founder_rates, neutral_rate_to_aa
from dasm2.neutral.tabular_rates import TabularRatesNeutralModel
from dasm2.sequence.constants import AA_STR_SORTED

#: The table every check reads. Not rebuilt here: `rule score` wrote it, and a
#: test that rebuilt it would be the one thing this repository must not do.
SCORED_TABLE = lib.ARTIFACT_ROOT / "scored" / "selection_factors.parquet"


def _archive(clade: str) -> Path:
    """One clade's era archive.

    Named through ``archive_name`` rather than by an f-string of its own.
    `workflow/corpus.smk`'s ``NEUTRAL_ZIP`` cannot be imported here -- it is a
    Snakefile -- so without this the test would be a third spelling of the
    path, and the one that would not break when the other two changed.
    """
    era = panel_spec.era_of_clade(clade)
    return lib.ARTIFACT_ROOT / "neutral_models" / f"{archive_name(lib.GENE, era)}.zip"


def _artifacts_readable() -> bool:
    try:
        return SCORED_TABLE.is_file() and all(
            _archive(lib.dms_dataset(homolog).clade).is_file()
            for homolog in lib.dms_homologs()
        )
    except OSError:
        return False


needs_artifacts = pytest.mark.skipif(
    not _artifacts_readable(),
    reason="the scored table or a neutral archive is not readable here",
)


@pytest.fixture(scope="module")
def homologs() -> list[str]:
    return list(lib.dms_homologs())


@needs_dms
@needs_artifacts
@pytest.mark.parametrize("homolog", lib.dms_homologs())
def test_log_p_minus_log_f_minus_log_lambda_is_one_constant(homolog: str):
    """``log p = log f + log lambda - log Z``, across the two written files.

    **The form of this test is the test.** ``aa_mutation_probabilities``
    returns ``selection_factors * rate_to_aa / Z``, so an identity recomputed
    from the tensor that was fed in holds *algebraically* however wrong that
    tensor was -- rolling the pivot by one site leaves the residual at 1e-15
    and passes. So ``log f`` is read back out of the scored table and joined to
    the written score file by ``(aa_site, mutant_aa)``, and ``log lambda``
    comes from the other written file. Re-joined, that same bug gives a spread
    of order 10.

    The tolerance is 1e-6 rather than something tighter because the neutral
    rates are float32 and the scores make a Parquet round trip; the observed
    spread is ~3e-7, where a float64 in-memory computation would give ~4e-15.
    """
    clade = lib.dms_dataset(homolog).clade
    scored, _ = dms_prob.prob_scores(homolog, SCORED_TABLE, _archive(clade))
    (prob, _), (rate, _) = scored

    grid = dms_dasm.selection_factor_grid(clade, SCORED_TABLE)
    merged = (
        prob[["aa_site", "mutant_aa", "score"]]
        .rename(columns={"score": "log_p"})
        .merge(
            rate[["aa_site", "mutant_aa", "score"]].rename(
                columns={"score": "log_lambda"}
            ),
            on=list(dms_models.JOIN_KEY),
            validate="one_to_one",
        )
        .merge(
            grid[["aa_site", "mutant_aa", dms_dasm.SCORE_COLUMN]],
            on=list(dms_models.JOIN_KEY),
            validate="one_to_one",
        )
    )
    assert len(merged) == len(prob)

    residual = merged["log_p"] - merged[dms_dasm.SCORE_COLUMN] - merged["log_lambda"]
    assert residual.max() - residual.min() < 1e-6

    # A real `-log Z`, not zero: a column written as `log f + log lambda` with
    # no normalizer would satisfy the spread assertion above and nothing else.
    assert abs(residual.mean()) > 1.0


@needs_dms
@needs_artifacts
@pytest.mark.parametrize("homolog", lib.dms_homologs())
def test_the_neutral_rate_is_zero_exactly_where_one_base_cannot_reach(homolog: str):
    """The rate's zero set is the complement of single-nucleotide accessibility.

    Computed two independent ways over the whole ``1273 x 20`` grid: from the
    neutral model's own rates, and from `scripts/dms.py`'s codon-level
    indicator. The complement is not only the multi-nucleotide mutations -- it
    also holds the synonymous diagonal at Met and Trp, whose single codon no
    synonymous one-base change reaches.

    ``single_nt_indicator`` is ``[20, L]`` and ``neutral_rate_to_aa`` is
    ``[L, 20]``, so one of them is transposed. Neither name says so.
    """
    clade = lib.dms_dataset(homolog).clade
    model = TabularRatesNeutralModel.from_pretrained(str(_archive(clade)))
    rates = founder_rates(model, founder_nt_sequence(clade, lib.GENE))
    reachable_by_rate = neutral_rate_to_aa(rates).numpy() > 0

    indicator = dms.single_nt_indicator(clade, lib.GENE)
    reachable_by_codon = indicator.loc[list(AA_STR_SORTED)].to_numpy().T

    assert reachable_by_rate.shape == reachable_by_codon.shape
    assert int((reachable_by_rate != reachable_by_codon).sum()) == 0

    # Not vacuous: both must find a substantial reachable set, or an all-False
    # pair would agree and say nothing.
    assert reachable_by_rate.sum() > 1000


@needs_dms
@needs_artifacts
@pytest.mark.parametrize("homolog", lib.dms_homologs())
def test_every_scored_cell_is_single_nucleotide_and_none_is_missing(homolog: str):
    """The scored set *is* the declared stratum -- no more, and no fewer.

    The counts are derived from `melt_dms`'s own stratum labels rather than
    pinned, so this moves with the assays' filtering instead of going stale
    beside it. Both predictors are checked: they are written from one merge,
    but by two `write_scores` calls.
    """
    clade = lib.dms_dataset(homolog).clade
    cells, _ = dms_models.homolog_cells(homolog)
    expected = cells[cells["stratum"] == dms.SINGLE_NT]
    expected_keys = set(zip(expected["aa_site"], expected["mutant_aa"]))
    assert expected_keys, f"{homolog} has no single-nucleotide cells to score"

    scored, _ = dms_prob.prob_scores(homolog, SCORED_TABLE, _archive(clade))
    for frame, manifest in scored:
        assert manifest.strata == (dms.SINGLE_NT,)
        assert set(zip(frame["aa_site"], frame["mutant_aa"])) == expected_keys
        assert np.isfinite(frame["score"]).all()


@needs_dms
@needs_artifacts
def test_the_pivot_refuses_a_grid_that_is_not_one_row_per_cell():
    """The pivot is where a site or residue offset would enter unseen.

    The identity test above cannot catch one -- see its docstring -- so the
    ordering is checked where it is decided. A short grid is the likely way to
    get there: a caller that filtered the table before pivoting it.
    """
    clade = lib.dms_dataset(lib.dms_homologs()[0]).clade
    grid = dms_dasm.selection_factor_grid(clade, SCORED_TABLE)
    founder = dms_models.homolog_cells(lib.dms_homologs()[0])[1]

    with pytest.raises(ValueError, match="rows for a founder of"):
        dms_prob.linear_selection_factors(grid.iloc[:-1], founder, torch.float32)


def test_each_predictor_is_scored_from_its_own_column():
    """Which column each predictor's ``log`` is taken of.

    ``NEUTRAL_RATE`` and ``LINEAR_RATE_COLUMN`` happen to spell the same string
    today, and they are different things: the first is a predictor id whose
    score is ``log lambda``, the second a column of ``build_probability_dataframe``
    holding ``lambda`` itself. This pins the *pairing* rather than that
    coincidence, so renaming the predictor id stays a one-line change instead
    of breaking a test that had no business depending on it.
    """
    assert dms_prob.SCORED_QUANTITIES == (
        (dms_prob.DASM_PROB, dms_prob.PROBABILITY_COLUMN),
        (dms_prob.NEUTRAL_RATE, dms_prob.LINEAR_RATE_COLUMN),
    )
    assert dms_prob.PROBABILITY_COLUMN != dms_prob.LINEAR_RATE_COLUMN
