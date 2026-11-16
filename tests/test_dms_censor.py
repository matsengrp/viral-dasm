"""The training mask, checked against mat-pcp rather than against itself.

``scripts/dms_censor.py`` rebuilds at the amino-acid level a mask that
`fig:dms` gets for free by dropping rows with no counts estimate. The premise
that makes that legitimate is `scripts/panels.py`'s: mat-pcp's ``exclude`` flag
and this pipeline's censoring "are two applications of one set".
``tests/test_panels.py::test_training_censors_exactly_what_mat_pcp_excludes``
establishes it at the **nucleotide** level, where no codon table is consulted.

**This file establishes that it survives the lift to amino acids**, which is a
different claim and the one `fig:dms_strata` rests on. A mutation is censored
when every single-nucleotide route to it is censored, so the set should equal
the single-nucleotide-reachable mutations that mat-pcp's own per-clade
``aamut_fitness`` table does not carry.

It is a re-derivation and not a pin. A count pinned here would go green on a
rebuilt mat-pcp run while meaning something else, and the failure this is for
is silent in every other test: both sides still produce full-looking panels.

The bug it was written for, and which it fails loudly on: ``AA_TO_SENSE_CODONS``
is keyed in 61-sense-codon space while ``CODONS`` is the 64-codon list, and the
two coincide only below the first stop codon. Reading one with the other's
indices mis-identifies the routes to C, F, W, Y and to the ``T**`` realizations
of L and S -- which leaves some mutations uncensored, censors others through a
route that does not exist, and at one codon offers a stop as a destination.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import lib

import dasm2  # noqa: F401

pd = pytest.importorskip("pandas")

import dms
import dms_censor
import founders
import panels

from dasm2.sequence.constants import AA_STR_SORTED

#: Clades to check. The three the DMS homologs are matched to, plus 24A, which
#: `tests/test_panels.py` also uses and which carries an unobservable founder
#: codon. 21K carries none, so it is the clade where the two sets must agree
#: exactly with nothing added -- the control on the unobservable layer.
CLADES = ("21K", "23A", "24A", "24E")


def _require_mat_pcp():
    root = Path(lib.CONFIG["mat_pcp"]["root"])
    if not root.exists():
        if os.environ.get("VIRAL_DASM_REQUIRE_DATA") == "1":
            pytest.fail(f"{root} is missing")
        pytest.skip(f"{root} is not mounted")
    return root


def _mat_pcp_excluded(clade: str) -> set[tuple[int, str]]:
    """The single-nt-reachable mutations mat-pcp's count table does not carry.

    Synonymous routes are left out of both sides: this module's consumer joins
    to measured substitutions, and ``single_nt_indicator`` answers for the
    founder's own residue as well, so including them would compare two
    conventions rather than two masks.
    """
    indicator = dms.single_nt_indicator(clade)
    founder = founders.founder_aa_sequence(clade, lib.GENE)
    reachable = {
        (position + 1, aa)
        for position in indicator.columns
        for aa in AA_STR_SORTED
        if indicator.loc[aa, position] and aa != founder[position]
    }
    table = pd.read_csv(
        lib.mat_pcp_path("aamut_fitness_dir") / f"{clade}{lib.FITNESS_SUFFIX}"
    )
    table = table[table["gene"] == lib.GENE]
    kept = {
        (int(site), mutant)
        for site, mutant in zip(table["aa_site"], table["mutant_aa"])
        if mutant in AA_STR_SORTED
    }
    return reachable - kept


def _censored_substitutions(clade: str) -> set[tuple[int, str]]:
    """This module's answer, with the founder's own residue dropped."""
    founder = founders.founder_aa_sequence(clade, lib.GENE)
    return {
        (site, aa)
        for site, aa in dms_censor.censored_mutations(clade)
        if aa != founder[site - 1]
    }


@pytest.mark.parametrize("clade", CLADES)
def test_every_mutation_mat_pcp_excluded_is_censored(clade):
    """Nothing mat-pcp excluded is left live, which is the direction that matters.

    A mutation training censored but this module missed would be scored against
    the assay although the model was taught to ignore it -- the disagreement
    the figure would then report is one the pipeline created.
    """
    _require_mat_pcp()
    missed = _mat_pcp_excluded(clade) - _censored_substitutions(clade)
    assert not missed, (
        f"{clade}: {len(missed)} mutations mat-pcp excluded are not censored "
        f"here, so they would be scored although training never saw them. "
        f"First few: {sorted(missed)[:5]}"
    )


@pytest.mark.parametrize("clade", CLADES)
def test_nothing_extra_is_censored_outside_the_unobservable_codons(clade):
    """The other direction, and the one place the two sets legitimately differ.

    ``panels.unobservable_founder_codons`` is applied on purpose and only ever
    adds: its rows are ones mat-pcp *kept*, with ``exclude`` correctly ``False``
    because what is masked is another nucleotide of the codon. So every
    difference in this direction must sit at one of those codons, and anywhere
    else is a route this module invented.
    """
    _require_mat_pcp()
    unobservable = panels.unobservable_founder_codons(clade)
    extra = {
        (site, aa)
        for site, aa in _censored_substitutions(clade) - _mat_pcp_excluded(clade)
        if site not in unobservable
    }
    assert not extra, (
        f"{clade}: {len(extra)} mutations are censored here that mat-pcp kept, "
        f"at codons that are not unobservable. First few: {sorted(extra)[:5]}"
    )


def test_the_clade_with_no_unobservable_codon_matches_exactly():
    """21K pins the two directions together with the added layer switched off.

    The parametrized tests above each allow the unobservable layer its
    exception. 21K has no unobservable codon at all, so on it the mask is
    exactly mat-pcp's exclusion set and the equality can be asserted outright
    -- which is what makes the exception above a statement about that layer
    rather than a hole either test could hide in.
    """
    _require_mat_pcp()
    assert not panels.unobservable_founder_codons("21K"), (
        "21K has gained an unobservable founder codon, so it can no longer be "
        "the control here; pick another clade with none, or drop this test."
    )
    assert _censored_substitutions("21K") == _mat_pcp_excluded("21K")


@pytest.mark.parametrize("clade", CLADES)
def test_no_multi_nucleotide_mutation_is_censored_by_the_route_rule(clade):
    """The guard that keeps the multi-nucleotide stratum from vanishing.

    A multi-nucleotide mutation has no single-nucleotide route, so "every route
    is censored" is vacuously true of it. Reading the rule without the guard
    would censor the entire stratum the figure exists to draw. Only the
    unobservable-codon layer, which is about the founder background rather than
    the destination, may reach these.
    """
    _require_mat_pcp()
    indicator = dms.single_nt_indicator(clade)
    founder = founders.founder_aa_sequence(clade, lib.GENE)
    unobservable = panels.unobservable_founder_codons(clade)
    multi_nt_censored = {
        (site, aa)
        for site, aa in _censored_substitutions(clade)
        if site not in unobservable and not indicator.loc[aa, site - 1]
    }
    assert not multi_nt_censored, (
        f"{clade}: {len(multi_nt_censored)} multi-nucleotide mutations are "
        f"censored by the route rule, which has no routes to judge them by. "
        f"First few: {sorted(multi_nt_censored)[:5]}"
    )
    assert founder, "founder sequence is empty"
