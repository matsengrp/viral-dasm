"""`scripts/dms_prob.py`: the DASM's probability, and the rate behind it.

The guard that refuses a *log* selection factor where a linear one belongs is
not re-pinned here. `tests/test_dms.py` already does it, and more thoroughly
than a second copy would: it asserts the passing leg as well, including a
legitimate zero factor so the guard has to be ``< 0`` rather than ``<= 0``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

#: Turns a skip into a failure on the host that builds the figures, as every
#: other data-gated module in this suite does.
REQUIRE_DATA = os.environ.get("VIRAL_DASM_REQUIRE_DATA") == "1"

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
import pandas as pd
import panel_spec
import torch
from build_era_neutral_models import archive_name
from founders import founder_nt_sequence
from test_dms_models import needs_dms

from dasm2.analysis.whichmut_probs import founder_rates, neutral_rate_to_aa
from dasm2.neutral.tabular_rates import TabularRatesNeutralModel
from dasm2.sequence.constants import AA_STR_SORTED

#: The table every data check reads, spelled as `tests/test_dms.py` spells it.
#: Not rebuilt here: `rule score` wrote it, and a test that rebuilt it would be
#: the one thing this repository must not do.
SCORED_TABLE = panel_spec.SCORED_DIR / "selection_factors.parquet"

#: ``build_neutral``'s output directory. `workflow/corpus.smk` owns the name and
#: a Snakefile cannot be imported, so this is a second spelling -- the same one
#: `tests/test_era_map.py` already asserts against.
NEUTRAL_DIR = lib.ARTIFACT_ROOT / "neutral_models"


def _archive(clade: str) -> Path:
    """One clade's era archive.

    Named through ``archive_name`` rather than by an f-string of its own.
    `workflow/corpus.smk`'s ``NEUTRAL_ZIP`` cannot be imported here -- it is a
    Snakefile -- so without this the test would be a third spelling of the
    path, and the one that would not break when the other two changed.
    """
    return NEUTRAL_DIR / f"{archive_name(lib.GENE, panel_spec.era_of_clade(clade))}.zip"


def _absent_artifacts() -> list[str]:
    """Which of this module's inputs are missing, by path."""
    try:
        paths = [SCORED_TABLE, *(_archive(c) for c in _CLADES)]
        return [str(path) for path in paths if not path.is_file()]
    except OSError as exc:  # an unreadable mount, not a missing file
        return [f"{lib.ARTIFACT_ROOT} ({exc})"]


_CLADES = [lib.dms_dataset(homolog).clade for homolog in lib.dms_homologs()]
_ABSENT = _absent_artifacts()

needs_artifacts = pytest.mark.skipif(
    bool(_ABSENT), reason=f"not readable here: {', '.join(_ABSENT)}"
)

if REQUIRE_DATA and _ABSENT:
    # `tests/test_dms.py`'s rule, for this module's reason: every check that
    # exercises the probability -- the identity, the reachability set, the
    # stratum -- is gated on these, so on the host that builds the figure a
    # moved archive would leave the suite green with none of them run. Raised
    # at collection so the reason is the path rather than a pile of skips.
    raise RuntimeError(
        "VIRAL_DASM_REQUIRE_DATA=1 but these are not readable: " + ", ".join(_ABSENT)
    )


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
    prob, _ = scored[dms_prob.DASM_PROB]
    rate, _ = scored[dms_prob.NEUTRAL_RATE]

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

    # Not vacuous: an all-False pair would agree and say nothing. The floor is
    # the genetic code's rather than a round number. Every sense codon's nine
    # single-base neighbours encode at least four distinct non-parent, non-stop
    # residues -- CGA is the worst, at exactly four -- so two per site is a
    # bound the code guarantees with room to spare. The observed count is
    # ~8,810 of 25,460 cells, so this catches a collapse without pinning one.
    assert reachable_by_rate.sum() >= 2 * reachable_by_rate.shape[0]


@needs_dms
@needs_artifacts
@pytest.mark.parametrize("homolog", lib.dms_homologs())
def test_every_scored_cell_is_reachable_by_one_base(homolog: str):
    """Every scored cell is one base away, and no measured one is left out.

    **Checked against the codon indicator, not against the stratum column.**
    `prob_scores` filters on ``stratum == single_nt``, so comparing its output
    back to that same filter would hold by construction and prove only that
    the merge kept its rows. `single_nt_indicator` derives reachability from
    the founder's codons independently, which is what makes this a check.

    Both predictors are checked: they come from one merge, but go out through
    two `write_scores` calls.
    """
    clade = lib.dms_dataset(homolog).clade
    cells, _ = dms_models.homolog_cells(homolog)
    indicator = dms.single_nt_indicator(clade, lib.GENE)

    measured = set(zip(cells["aa_site"], cells["mutant_aa"]))
    independently_reachable = {
        (site, aa) for site, aa in measured if bool(indicator.loc[aa, site - 1])
    }
    assert independently_reachable, f"{homolog} has no cell one base away"

    scored, _ = dms_prob.prob_scores(homolog, SCORED_TABLE, _archive(clade))
    for model, (frame, manifest) in scored.items():
        assert model in dms_prob.SCORED_COLUMNS
        assert manifest.strata == (dms.SINGLE_NT,)
        assert set(zip(frame["aa_site"], frame["mutant_aa"])) == (
            independently_reachable
        )
        assert np.isfinite(frame["score"]).all()
        assert (frame["model"] == model).all()


#: A founder short enough to write a whole grid for by hand.
_TOY_FOUNDER = "MKV"


def _toy_grid(founder: str = _TOY_FOUNDER) -> pd.DataFrame:
    """A complete, well-formed `L x 20` grid in the scored table's columns."""
    return pd.DataFrame(
        [
            {
                "aa_site": site,
                "mutant_aa": aa,
                "parent_aa": founder[site - 1],
                dms_dasm.SCORE_COLUMN: 0.0,
            }
            for site in range(1, len(founder) + 1)
            for aa in AA_STR_SORTED
        ]
    )


def test_the_pivot_accepts_a_complete_grid():
    """The guard below has to pass as well as fire, or it proves nothing."""
    out = dms_prob.linear_selection_factors(_toy_grid(), _TOY_FOUNDER, torch.float32)
    assert out.shape == (len(_TOY_FOUNDER), len(AA_STR_SORTED))
    # exp(0) == 1 everywhere, which is also the gauge the real grid is in.
    assert torch.allclose(out, torch.ones_like(out))


@pytest.mark.parametrize(
    "mangle, message",
    [
        pytest.param(lambda g: g.iloc[:-1], "rows for a founder of", id="short-grid"),
        pytest.param(
            lambda g: pd.concat([g, g.iloc[:1]]),
            "rows for a founder of",
            id="long-grid",
        ),
        pytest.param(
            lambda g: g.assign(aa_site=g["aa_site"] - 1),
            "absent after reindexing",
            id="zero-indexed-sites",
        ),
        pytest.param(
            lambda g: g.assign(mutant_aa=g["mutant_aa"].replace({"A": "X"})),
            "absent after reindexing",
            id="residue-outside-alphabet",
        ),
    ],
)
def test_the_pivot_refuses_a_grid_that_is_not_this_founder_s(mangle, message):
    """Each way the grid can be indexed against something else.

    The identity test cannot catch any of these -- see its docstring -- so the
    ordering is checked where it is decided. Synthetic grids rather than a
    slice of the real one, so these run off-cluster too: a site offset is the
    bug this guard exists for, and it should not be gated on a cluster mount.
    """
    with pytest.raises(ValueError, match=message):
        dms_prob.linear_selection_factors(
            mangle(_toy_grid()), _TOY_FOUNDER, torch.float32
        )


def test_a_scored_column_must_be_positive_everywhere():
    """A zero rate inside the declared stratum raises rather than dropping.

    That cell is one the predictor says it can speak on, so a silent drop
    would leave the scored count short of the stratum without saying so.
    """
    merged = pd.DataFrame(
        {
            "aa_site": [1, 2],
            "mutant_aa": ["A", "C"],
            dms_prob.SCORED_COLUMNS[dms_prob.NEUTRAL_RATE]: [1e-3, 0.0],
        }
    )
    with pytest.raises(ValueError, match="no positive"):
        dms_prob._require_loggable(
            merged, dms_prob.SCORED_COLUMNS[dms_prob.NEUTRAL_RATE]
        )
