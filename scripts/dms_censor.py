"""Which amino-acid mutations the DASM was never trained to predict.

`fig:dms` inherits this for free and this figure cannot, which is the whole
reason this module exists. There, every frame goes through ``panels.base_frames``
and its ``dropna`` on the counts metric, and `scripts/panels.py` records why that
is also the training mask: mat-pcp's ``exclude`` flag and this pipeline's
censoring "are two applications of one set", with
``tests/test_panels.py::test_training_censors_exactly_what_mat_pcp_excludes``
re-deriving the equality from the files. Dropping the rows mat-pcp could not
estimate therefore drops exactly the routes training censored.

**But that drop is also what restricts those frames to the single-nucleotide
stratum**, since mat-pcp estimates only what the tree could observe, and the
figure this module serves exists to draw the multi-nucleotide half beside it. So
the mask is rebuilt here from its own sources rather than inherited from the
counts table.

**The unit mismatch, and the rule.** Censoring is specified per nucleotide: a
``CensorSpec`` carries whole-site masks, which censor every change away from a
nucleotide, and directional ``[nt_index, from, to]`` rows, which censor one
transition and leave its reverse live. The figure's unit is an amino-acid
mutation, which a parent codon may reach by more than one single-nucleotide
route. **A mutation is censored when every route to it is censored**, matching
mat-pcp's own granularity -- `scripts/panels.py` puts it as excluding "a mutation
one nucleotide route at a time", with a mutation excluded entirely being the one
that has no row at all. ``transforms.py:927`` names the same escape from the
other side: a reversion-censored codon does not censor its residue when "another
single-nucleotide child of the same parent codon encodes the same" one.

**Multi-nucleotide mutations are not censored here, and that is not an
oversight.** A training candidate is one of the parent codon's nine
single-nucleotide neighbours, so a multi-nucleotide substitution was never a
candidate to censor. ``scripts/dms.py``'s ``MULTI_NT`` says what the model is
doing there instead -- "the model is trained on none of them, and mat-pcp
estimates none of them, so both are making what `main.tex` calls an 'educated
guess'" -- and `main.tex:202` says the selection factors cover those mutations
"including those not reachable by a single-nucleotide change from the parent
codon". Reading the rule literally would censor that whole stratum by vacuous
truth, since it has no routes at all, so the rule carries the guard that a
mutation is censored only when it has at least one route and all of them are
censored. The site-level mask still reaches these mutations; it is applied
earlier, in ``dms.load_dms``.

**A fourth layer, which is about the background rather than the destination.**
Where a clade's founder codon carries a nucleotide masked in the tree, mat-pcp
can attribute no observation to that background at all, so nothing at that codon
was learnable from it. ``panels.unobservable_founder_codons`` is the predicate
`fig:dms` already drops on, and it is applied here for the same reason rather
than rebuilt. It is site-level, so it reaches both strata.

**Three sources, because the mask has three layers and only one is applied
upstream.** ``dms.unreadable_sites`` drops globally masked and per-clade censored
*sites* before ``from_multidms`` parses, and says in its own docstring that it is
"not the figure's exclusion, which adds the reversion layer". The two it leaves
are rebuilt here: the per-mutation censor table, and the founder-to-reference
reversions that live in no file because a training run derives them
(``masking.load_reference_base_idxs``, consumed at ``train_command.py:1855``).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import GENE, assert_dasm2_is_this_repo, mat_pcp_path

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

from founders import founder_nt_sequence
from panels import unobservable_founder_codons

from dasm2.codon_table import CODON_AA_INDICATOR_MATRIX
from dasm2.sequence.constants import AA_STR_SORTED, BASES, CODONS
from dasm2.sequence.conversion import iter_codons
from dasm2.sequence.masking import (
    load_censor_table,
    load_reference_base_idxs,
    parse_masked_nt_sites_file,
)

#: Base letter to its index in :data:`~dasm2.sequence.constants.BASES`, the
#: encoding a ``CensorSpec``'s ``directional`` rows and
#: ``load_reference_base_idxs`` both use.
_BASE_INDEX = {base: index for index, base in enumerate(BASES)}

#: Sense codon to the residue it encodes. The three stops are **absent** rather
#: than mapped to a sentinel, so a route to a stop is skipped by the lookup
#: failing rather than by a test someone can forget: the DASM gives no estimate
#: for a stop (`main.tex`: "excluding those to a stop"), so a stop is not a
#: destination this module reasons about.
#:
#: Built from :data:`CODON_AA_INDICATOR_MATRIX`, whose stop rows are all zero,
#: in 64-codon space throughout. **This module reads codons one way only.**
#: ``AA_TO_SENSE_CODONS`` answers the inverse question but is keyed in
#: 61-sense-codon space, and the two spaces coincide only below the first stop
#: (``TAA``, 64-index 48) and drift by one, two and three above it -- so mixing
#: them silently mis-identifies routes to C, F, W, Y and to the ``T**``
#: realizations of L and S. ``tests/test_dms_censor.py`` pins the amino-acid
#: equality that catches exactly that.
_CODON_TO_AA: dict[str, str] = {
    codon: AA_STR_SORTED[int(row.argmax())]
    for codon, row in zip(CODONS, CODON_AA_INDICATOR_MATRIX)
    if int(row.sum()) > 0
}


@lru_cache(maxsize=None)
def _censor_sources(clade: str, gene: str) -> tuple:
    """The three mask layers for one clade, read once.

    Cached because every homolog of a clade asks for the same three files, and
    the reference FASTA and censor table are read from the mat-pcp mount.
    """
    table = load_censor_table(
        mat_pcp_path("censored_mutations_file"),
        mat_pcp_path("censored_clades_file"),
    )
    spec = table[clade]
    founder_nt = founder_nt_sequence(clade, gene)
    reference = load_reference_base_idxs(
        mat_pcp_path("reference_sequence_file"), num_codons=len(founder_nt) // 3
    )

    # The global nucleotide mask joins the per-clade site masks. Every site it
    # covers has already left in `dms.load_dms`, so this is belt and braces
    # rather than a second filter -- but a route blocked by the global mask is
    # blocked whether or not something upstream already dropped its site, and
    # saying so here keeps this function's answer true on its own terms.
    blocked_sites = {int(index) for index in spec.site_nt_indices}
    blocked_sites.update(
        int(index)
        for index in parse_masked_nt_sites_file(mat_pcp_path("masked_nt_sites_file"))
    )
    directional = {
        (int(row[0]), int(row[1]), int(row[2])) for row in spec.directional
    }
    return founder_nt, blocked_sites, directional, [int(x) for x in reference]


def _route_is_censored(
    nt_index: int,
    from_base: int,
    to_base: int,
    blocked_sites: set[int],
    directional: set[tuple[int, int, int]],
    reference: list[int],
) -> bool:
    """Whether one single-nucleotide route is closed to training.

    The three layers in the order they cost least to check. The reversion test
    is last and is the only one that reads the reference: a change *toward* the
    reference base is censored where the founder does not already carry it,
    while the change away from it stays live -- ``transforms.py``'s
    "nucleotide- and direction-granular, never whole-site".
    """
    if nt_index in blocked_sites:
        return True
    if (nt_index, from_base, to_base) in directional:
        return True
    return to_base == reference[nt_index] and from_base != reference[nt_index]


@lru_cache(maxsize=None)
def censored_mutations(clade: str, gene: str = GENE) -> frozenset[tuple[int, str]]:
    """``(aa_site, mutant_aa)`` pairs the DASM was never trained to predict.

    ``aa_site`` is 1-indexed, to match the column of the same name in the scored
    table and in :func:`dms.melt_dms`' output. ``mutant_aa`` is a letter of
    :data:`~dasm2.sequence.constants.AA_STR_SORTED`; stops are not scored and do
    not appear.

    Only mutations with at least one single-nucleotide route are eligible, so
    the multi-nucleotide stratum comes back empty by construction -- see this
    module's docstring for why that is the intended reading rather than a hole.

    Returns:
        The censored pairs. Empty for a clade whose spec censors nothing and
        whose founder matches the reference everywhere, which is legitimate.
    """
    founder_nt, blocked_sites, directional, reference = _censor_sources(clade, gene)

    # Every mutation at a codon whose founder background the tree cannot show.
    # Site-level rather than per route, and deliberately so: the rule is about
    # the *background* rather than the destination, so it reaches the
    # multi-nucleotide stratum too and the single-route guard below does not
    # apply to it. `panels.unobservable_founder_codons` is imported rather than
    # rebuilt -- it is the same predicate `fig:dms` drops on, and two spellings
    # of it would eventually disagree. Note that these rows are *not* in
    # mat-pcp's exclusions: that function's docstring records that ``exclude``
    # is correctly ``False`` for the routes, since what is masked is another
    # nucleotide of the codon, so this layer only ever adds.
    unobservable = unobservable_founder_codons(clade)

    censored: set[tuple[int, str]] = {
        (site, mutant_aa) for site in unobservable for mutant_aa in AA_STR_SORTED
    }
    for position, codon in enumerate(iter_codons(founder_nt)):
        if codon not in CODONS:
            # `dms.single_nt_indicator` refuses the same codons for the same
            # reason: a partially ambiguous codon still translates, so it would
            # survive the wildtype check and be answered here with a guess.
            raise ValueError(
                f"{gene}/{clade}: codon {position + 1} is {codon!r}, which is "
                f"not one of the 64. Censoring is read per nucleotide, so an "
                f"ambiguous base would silently leave every route at this "
                f"codon live rather than raise."
            )
        codon_start = position * 3

        # The codon's nine single-nucleotide neighbours, grouped by the residue
        # each encodes. Enumerated forwards rather than looked up per target
        # residue: a route *is* one substitution in this codon, so building it
        # that way needs no inverse table and so cannot mix two codon index
        # spaces. A residue absent from the grouping has no single-nucleotide
        # route, which is what makes it multi-nucleotide.
        routes: dict[str, list[tuple[int, int, int]]] = {}
        for offset in range(3):
            from_letter = codon[offset]
            for to_letter in BASES:
                if to_letter == from_letter:
                    continue
                neighbour = codon[:offset] + to_letter + codon[offset + 1 :]
                mutant_aa = _CODON_TO_AA.get(neighbour)
                if mutant_aa is None:  # a stop; not a destination we score
                    continue
                routes.setdefault(mutant_aa, []).append(
                    (
                        codon_start + offset,
                        _BASE_INDEX[from_letter],
                        _BASE_INDEX[to_letter],
                    )
                )

        # Only residues with at least one route are eligible, so the
        # multi-nucleotide stratum is untouched here: it has no routes at all,
        # and `all()` over nothing would otherwise censor every one of them.
        for mutant_aa, aa_routes in routes.items():
            if all(
                _route_is_censored(
                    nt_index,
                    from_base,
                    to_base,
                    blocked_sites,
                    directional,
                    reference,
                )
                for nt_index, from_base, to_base in aa_routes
            ):
                censored.add((position + 1, mutant_aa))
    return frozenset(censored)
