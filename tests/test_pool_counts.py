"""What the counts half of `fig:within_clade_shifts` has to get right.

Everything here runs on hand-built frames: no corpus, no checkpoint, no GPU.
The expensive parts of the producer are two streaming passes, and neither is
what these tests are about -- they are about the arithmetic the passes feed,
and about the two contracts that join the counts side to the model side.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import panel_spec

from dasm2.analysis.within_clade_pools import (
    PoolSpec,
    permutation_shift_test,
    pool_average_log_sf,
    pooled_log_sf,
    substitution_key,
)


def branch_frame(founder: tuple[int, int], derived: tuple[int, int]) -> pd.DataFrame:
    """One row per branch, as `tally_pcps` returns them.

    Each pool is given as ``(focal events, synonymous events)``, spread one
    event per branch so the shuffle has branches to move.
    """
    rows = []
    for pool, (focal, synonymous) in (("f", founder), ("d", derived)):
        rows += [(pool, 1, 0)] * focal + [(pool, 0, 1)] * synonymous
    return pd.DataFrame(rows, columns=["pool", "focal", "synonymous"])


# --------------------------------------------------------------------------
# The pool average: the DASM quantity the counts effect is compared against
# --------------------------------------------------------------------------


def test_the_pool_average_is_weighted_by_synonymous_events():
    """Not an unweighted mean over haplotypes.

    A pool is dominated by a few heavy haplotypes, so weighting by haplotype
    rather than by evolutionary time would let a rare parent with an extreme
    score move the pool's estimate as far as the parent carrying most of its
    branches.
    """
    founder = "MKV"
    heavy, light = "MKA", "MRV"
    average = pool_average_log_sf(
        {"aaa": 99, "bbb": 1},
        {"aaa": heavy, "bbb": light},
        {
            substitution_key(heavy, founder): 1.0,
            substitution_key(light, founder): 5.0,
        },
        founder_aa=founder,
    )
    unweighted = pooled_log_sf([1.0, 1.0], [1.0, 5.0])
    assert average.log_sf == pytest.approx(pooled_log_sf([99.0, 1.0], [1.0, 5.0]))
    assert average.log_sf < unweighted
    assert average.n_haplotypes == 2
    assert average.synonymous_weight == pytest.approx(100.0)


def test_the_pool_average_is_arithmetic_in_the_linear_domain():
    """A log-space mean would be the geometric mean, which is a different number.

    The counts side divides a sum of events by a sum of expected counts, so
    matching it forces the arithmetic mean of the selection factors. By
    Jensen the geometric mean sits below it, and on a spread this wide the
    gap is large enough that a log-space average cannot pass by accident.
    """
    weights, values = [1.0, 1.0], [0.0, 4.0]
    arithmetic = pooled_log_sf(weights, values)
    geometric = float(np.average(values, weights=weights))
    assert arithmetic == pytest.approx(float(np.log((np.exp(0.0) + np.exp(4.0)) / 2)))
    assert arithmetic > geometric


def test_an_unscored_haplotype_raises_rather_than_shrinking_the_average():
    """The reason the scorer is given the clade's whole parent set.

    Averaging over the intersection would drop that haplotype's evolutionary
    time out of the pool with nothing saying so, which reads as a smaller
    pool rather than as a missing input.
    """
    founder = "MKV"
    with pytest.raises(ValueError, match="no DASM score"):
        pool_average_log_sf(
            {"aaa": 10, "bbb": 5},
            {"aaa": "MKA", "bbb": "MRV"},
            {substitution_key("MKA", founder): 1.0},
            founder_aa=founder,
        )


def test_substitution_key_round_trips_the_founder_itself():
    """The empty key is why `clade_haplotypes` labels the founder haplotype.

    An empty ``background`` is the founder table's convention for "this row
    is a founder", and `score_haplotypes.read_backgrounds` refuses it, so the
    two tables cannot be allowed to collide on it.
    """
    assert substitution_key("MKV", "MKV") == ""
    assert substitution_key("MKA", "MKV") == "V3A"


# --------------------------------------------------------------------------
# The randomization test
# --------------------------------------------------------------------------


def test_the_test_is_two_sided():
    """A contrast and its mirror image get the same p-value.

    This is the sharp form. A one-sided test taking the tail on whichever
    side the observed statistic fell returns the same number for both as
    well -- but it returns about *half* this one, so the companion assertion
    below is what separates them.
    """
    up = permutation_shift_test(
        branch_frame(founder=(2, 100), derived=(30, 100)),
        pool0="f",
        pool1="d",
        n_permutations=2000,
        seed=0,
    )
    down = permutation_shift_test(
        branch_frame(founder=(30, 100), derived=(2, 100)),
        pool0="f",
        pool1="d",
        n_permutations=2000,
        seed=0,
    )
    assert up.observed == pytest.approx(-down.observed)
    assert up.p_value == pytest.approx(down.p_value)


def test_two_sided_counts_both_tails():
    """The assertion a one-sided implementation fails.

    Under a null that is near symmetric, counting ``abs(null) >= abs(obs)``
    reaches about twice as many draws as counting one side, so the two-sided
    p-value is about twice the one-sided one. Asserted as a band rather than
    a ratio, because the null is discrete.
    """
    frame = branch_frame(founder=(6, 100), derived=(18, 100))
    result = permutation_shift_test(
        frame, pool0="f", pool1="d", n_permutations=4000, seed=0
    )
    null = result.null[np.isfinite(result.null)]
    one_sided = int((null >= result.observed).sum())
    two_sided = int((np.abs(null) >= abs(result.observed)).sum())
    assert result.n_extreme == two_sided
    assert two_sided > one_sided
    assert result.p_value == pytest.approx((1 + two_sided) / (1 + 4000))


def test_alike_pools_are_not_called():
    """Two pools drawn from the same rates give a large p-value."""
    result = permutation_shift_test(
        branch_frame(founder=(20, 100), derived=(20, 100)),
        pool0="f",
        pool1="d",
        n_permutations=2000,
        seed=0,
    )
    assert result.p_value > 0.2


def test_the_p_value_floor_is_one_over_draws_plus_one():
    """Never zero, so a run that reached the floor reads as an inequality."""
    result = permutation_shift_test(
        branch_frame(founder=(0, 200), derived=(60, 200)),
        pool0="f",
        pool1="d",
        n_permutations=500,
        seed=0,
    )
    assert result.p_value == pytest.approx(1 / 501)
    assert result.n_extreme == 0


# --------------------------------------------------------------------------
# Pool membership
# --------------------------------------------------------------------------


def test_a_partial_match_joins_neither_pool():
    """A parent carrying part of a background belongs to neither side.

    This is what makes the contrast a split rather than a comparison of two
    overlapping mixtures: the branches in between are counted out, and
    `branch_fraction_table` reports how many they were.
    """
    # Codons: ATG=M, AAA=K, GTT=V, GCT=A, CGT=R.
    founder_side = PoolSpec(
        name="f", at_risk={1: "M"}, marker={2: "K", 3: "V"}, hold={}
    )
    derived_side = PoolSpec(
        name="d", at_risk={1: "M"}, marker={2: "R", 3: "A"}, hold={}
    )
    both = (founder_side, derived_side)
    founder_parent = "ATG" + "AAA" + "GTT"
    derived_parent = "ATG" + "CGT" + "GCT"
    half_way = "ATG" + "CGT" + "GTT"
    assert [spec.name for spec in both if spec.matches(founder_parent)] == ["f"]
    assert [spec.name for spec in both if spec.matches(derived_parent)] == ["d"]
    assert [spec.name for spec in both if spec.matches(half_way)] == []


def test_an_unreadable_codon_matches_nothing():
    """A stop or an ambiguous codon is excluded rather than string-compared."""
    spec = PoolSpec(name="p", at_risk={1: "M"}, marker={}, hold={})
    assert spec.matches("ATG")
    assert not spec.matches("TAA")
    assert not spec.matches("NNN")


def test_a_synonymous_ambiguity_is_resolved_like_the_rest_of_the_repository():
    """`GGN` is glycine whatever the fourth base, and BioPython says so.

    The vendored module dropped its own exact-64-codon table for this: the
    source's own docstring warns that a second table disagrees here, and a
    figure whose background fractions and pool membership used different
    conventions would lose exactly these parents.
    """
    assert PoolSpec(name="p", at_risk={1: "G"}, marker={}, hold={}).matches("GGN")


# --------------------------------------------------------------------------
# The candidate list and its parsing
# --------------------------------------------------------------------------


def test_the_mutation_parser_is_one_based():
    assert panel_spec.parse_mutation("D574V") == ("D", 574, "V")
    assert panel_spec.parse_mutation("M177V") == ("M", 177, "V")
    with pytest.raises(ValueError, match="not a mutation"):
        panel_spec.parse_mutation("574V")


def test_candidate_cells_covers_every_candidate_of_every_clade():
    """Including the ones kept off the figure.

    The designed null is measured through the same pipeline as the rest --
    that is what makes it a null rather than a row nobody ran -- so it has to
    be in the cells the scorer is asked for.
    """
    cells = panel_spec.candidate_cells()
    expected: dict[str, set[tuple[int, str]]] = {}
    for clade, mutation in [
        *panel_spec.figure_candidates(),
        *((c, m) for c, m, _ in panel_spec.off_figure_candidates()),
    ]:
        _, site, mutant = panel_spec.parse_mutation(mutation)
        expected.setdefault(clade, set()).add((site, mutant))
    assert {c: set(v) for c, v in cells.items()} == expected
    assert set(cells) == set(panel_spec.CANDIDATE_CLADES)


def test_no_candidate_sits_on_a_site_that_defines_its_background():
    """A candidate at a split site has no contrast to be drawn in.

    The at-risk requirement would demand the founder residue there and the
    derived pool would demand the substituted one, so one side of the
    contrast would be empty by construction rather than by observation.
    """
    import csv

    with panel_spec.BACKGROUND_CSV.open(newline="") as handle:
        split = {
            (row["clade"], int(row["aa_site"]))
            for row in csv.DictReader(handle)
            if row["role"] in panel_spec.BACKGROUND_SPLIT_ROLES
        }
    for clade, mutation in panel_spec.figure_candidates():
        _, site, _ = panel_spec.parse_mutation(mutation)
        assert (clade, site) not in split, f"{clade}/{mutation} is on a split site"
