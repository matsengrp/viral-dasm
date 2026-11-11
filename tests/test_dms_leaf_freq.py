"""What a wrong leaf reconstruction would report, and what catches it.

A reconstruction off by one codon, or started from the wrong sequence, or
applied in the wrong order still produces a full 20 x 1,273 table of plausible
counts and a plausible positive correlation with the DMS. Nothing downstream
notices. The checks here are what stand between that and a published number:
every mutation token's own reference base has to agree with the base it is
applied over, and the modal residue has to come back as the clade's own founder
everywhere the reconstruction can see.

The synthetic paths run against this run's real founders, so they need mat-pcp;
the table checks need a built tally. Both skip when their inputs are not
readable here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import lib

import dasm2  # noqa: F401

pd = pytest.importorskip("pandas")

import dms
import dms_leaf_freq
import dms_models
import panel_spec
from founders import founder_aa_sequence

from dasm2.neutral.scov2_rates import gene_nt_span
from dasm2.sequence.constants import AA_STR_SORTED, STOP_CODONS

#: The clade panel C is drawn on, and the one every real-data check here uses.
#: Its homolog is `panel_spec.SCATTER_HOMOLOG`'s, so a repoint moves both.
CLADE = "24E"

#: How many of that clade's leaves the reference-base check reads. The rule
#: checks all 124,359 -- ``main`` refuses to write a table with any mismatch --
#: so this is the fast copy of that check, sized to keep the suite under a
#: minute. A map that is off by one fails at the first leaf, not the last.
LEAVES_CHECKED = 3_000

#: Cells of KP.3.1.1's assay whose mutant residue clade 24E's leaves carry at
#: all. Pinned because it is the number the figure's own caveat rests on: most
#: of this predictor's rank information is the observed/unobserved split, and a
#: tally that silently did nothing would make this zero.
EXPECTED_MUTANT_OBSERVED = 2218

#: Sites where the modal residue is not the clade founder's. Masked sites carry
#: no mutation on any path, so the reconstruction keeps 19A's residue there;
#: codon 142 is the one site in 24E where 19A and the founder differ at a masked
#: nucleotide. Reported rather than asserted away: a disagreement anywhere else
#: means the leaves were tallied against the wrong frame.
EXPECTED_FOUNDER_DISAGREEMENTS = {142}

_MAT_PCP_INPUTS = (
    lib.mat_pcp_path("clade_founders_dir"),
    lib.mat_pcp_path("sample_paths_dir"),
    lib.mat_pcp_path("masked_nt_sites_file"),
    lib.mat_pcp_path("censored_mutations_file"),
    lib.mat_pcp_path("censored_clades_file"),
)


def _mat_pcp_readable() -> bool:
    try:
        return all(path.exists() for path in _MAT_PCP_INPUTS)
    except OSError:
        return False


needs_mat_pcp = pytest.mark.skipif(
    not _mat_pcp_readable(), reason="this run's mat-pcp outputs are not readable here"
)

_COUNTS = panel_spec.dms_leaf_count_path(CLADE)


def _counts_readable() -> bool:
    try:
        return _COUNTS.is_file()
    except OSError:
        return False


needs_counts = pytest.mark.skipif(
    not _counts_readable(), reason=f"{_COUNTS} has not been built here"
)


def _path_line(*mutations: str) -> str:
    """A synthetic leaf row: a sample name, a tab, then one node per mutation."""
    tokens = " ".join(
        f"node_{index}:{mutation}" for index, mutation in enumerate(mutations, start=2)
    )
    return f"synthetic|EPI_ISL_0|2024-01-01\t{tokens}\n"


def test_an_unparsable_mutation_token_raises():
    """A token read as nothing is a dropped mutation, which is a wrong leaf."""
    assert dms_leaf_freq.parse_mutation("C14408T") == ("C", 14408, "T")
    for bad in ("CT", "node_2", "C_T", ""):
        with pytest.raises(ValueError):
            dms_leaf_freq.parse_mutation(bad)


@needs_mat_pcp
def test_every_mutation_agrees_with_the_base_it_is_applied_over():
    """The one check that the index map, the start state and the order are right.

    A map off by one, a reconstruction started from a clade founder rather than
    19A, or tokens applied in reverse each disagree with a token's own reference
    base at the first branch that carries one.
    """
    tally = dms_leaf_freq.LeafTally(CLADE)
    with dms_leaf_freq.sample_paths_path(CLADE).open() as handle:
        for index, line in enumerate(handle):
            if index >= LEAVES_CHECKED:
                break
            tally.add_leaf(line)
    assert tally.n_leaves == LEAVES_CHECKED
    assert tally.reference_mismatches == 0, tally.mismatch_examples
    # And the check is live rather than vacuous: a token whose reference base is
    # deliberately wrong is counted.
    start = gene_nt_span(lib.GENE)[0]
    actual = tally.root_nt[0]
    wrong = next(base for base in "ACGT" if base != actual)
    tally.add_leaf(_path_line(f"{wrong}{start}A"))
    assert tally.reference_mismatches == 1


@needs_mat_pcp
def test_a_reversion_leaves_the_leaf_at_the_reference_residue():
    """The result a `leaves_sharing_mutations` tally gets wrong.

    A leaf's residue is set by the *last* mutation on its path, so a site
    mutated and then reverted contributes no deviation. A tally that weighted
    each node's mutation by its descendant count would record both.
    """
    tally = dms_leaf_freq.LeafTally(CLADE)
    start = gene_nt_span(lib.GENE)[0]
    # A site in the second codon, so the first codon stays untouched and the
    # deviation count below cannot be hidden by a boundary effect.
    site = start + 3
    original = tally.root_nt[3]
    mutant = next(base for base in "ACGT" if base != original)
    tally.add_leaf(_path_line(f"{original}{site}{mutant}", f"{mutant}{site}{original}"))
    assert tally.reference_mismatches == 0
    assert int(tally.deviations.sum()) == 0
    assert int(tally.unreadable.sum()) == 0

    # And a leaf that mutates without reverting does deviate, so the assertion
    # above is the reversion and not an inert path.
    tally.add_leaf(_path_line(f"{original}{site}{mutant}"))
    assert int(tally.deviations.sum()) + int(tally.unreadable.sum()) == 1


@needs_mat_pcp
def test_a_codon_outside_the_alphabet_is_dropped_from_its_site():
    """A stop must not be tallied as a residue, nor as the reference."""
    tally = dms_leaf_freq.LeafTally(CLADE)
    start = gene_nt_span(lib.GENE)[0]

    # Find a codon one substitution away from a stop, from the sequence itself
    # rather than from a remembered position.
    found = None
    for codon_site in range(tally.n_sites):
        codon = tally.root_nt[3 * codon_site : 3 * codon_site + 3]
        for offset in range(3):
            for base in "ACGT":
                if base == codon[offset]:
                    continue
                candidate = codon[:offset] + base + codon[offset + 1 :]
                if candidate in STOP_CODONS:
                    found = (codon_site, offset, codon[offset], base)
                    break
            if found:
                break
        if found:
            break
    assert found is not None, "no codon in this founder is one change from a stop"

    codon_site, offset, reference, mutant = found
    site = start + 3 * codon_site + offset
    tally.add_leaf(_path_line(f"{reference}{site}{mutant}"))
    assert tally.reference_mismatches == 0
    assert int(tally.deviations.sum()) == 0
    assert tally.unreadable[codon_site] == 1

    # The site's column then sums to one fewer than the leaf count, which is
    # what "dropped, not counted" means in the written table.
    table = tally.frame()
    residues = table.loc[table[dms_leaf_freq.SITE_COLUMN] == codon_site + 1]
    assert int(residues[list(AA_STR_SORTED)].to_numpy().sum()) == tally.n_leaves - 1


@needs_mat_pcp
def test_the_genome_to_codon_map_agrees_with_mat_pcps_own_table():
    """`clade_founder_nts.csv` as an oracle for a map the code derives.

    The code indexes with ``gene_nt_span``; mat-pcp publishes the same mapping
    as data. They are checked against each other rather than one being read, so
    neither can drift into the other's place.
    """
    oracle = (
        lib.mat_pcp_path("clade_founders_dir").parent.parent / "clade_founder_nts.csv"
    )
    if not oracle.is_file():
        pytest.skip(f"{oracle} is not readable here")
    table = pd.read_csv(oracle)
    rows = table[(table["clade"] == CLADE) & (table["gene"] == lib.GENE)]
    assert not rows.empty
    start = gene_nt_span(lib.GENE)[0]
    for row in rows.itertuples():
        offset = int(row.site) - start
        assert int(row.codon_site) == offset // 3 + 1
        assert int(row.codon_position) == offset % 3 + 1


@needs_counts
def test_the_modal_residue_is_the_clade_founder_away_from_masked_sites():
    """A reconstruction off by a codon still tallies; this is what notices."""
    counts = dms_leaf_freq.read_counts(_COUNTS)
    founder = founder_aa_sequence(CLADE, lib.GENE)
    assert len(counts) == len(founder)
    modal = counts.to_numpy().argmax(axis=1)
    disagree = {
        site + 1
        for site in range(len(founder))
        if AA_STR_SORTED[modal[site]] != founder[site]
    }
    assert disagree == EXPECTED_FOUNDER_DISAGREEMENTS
    assert disagree <= dms.globally_masked_sites()


@needs_counts
def test_the_tally_carries_standing_variation():
    """A table that reproduced the founder everywhere did nothing at all."""
    counts = dms_leaf_freq.read_counts(_COUNTS)
    founder = founder_aa_sequence(CLADE, lib.GENE)
    values = counts.to_numpy()
    non_founder = 0
    for site, residue in enumerate(founder):
        column = AA_STR_SORTED.index(residue)
        non_founder += int((values[site] > 0).sum()) - int(values[site, column] > 0)
    assert non_founder > 0
    # Not every cell either: a table where every residue were observed at every
    # site would mean the deviations were not per leaf.
    assert int((values > 0).sum()) < values.size


@needs_counts
def test_the_written_table_carries_no_sample_name():
    """Its input's rows name GISAID samples; nothing it writes may."""
    text = _COUNTS.read_text()
    header = text.splitlines()[0].split(",")
    assert header == [dms_leaf_freq.SITE_COLUMN, *AA_STR_SORTED]
    assert "EPI_ISL" not in text


@needs_counts
def test_no_masked_site_reaches_a_scored_cell():
    """The reconstruction's known blind spot must not enter a correlation."""
    frame, _, _ = dms_leaf_freq.leaf_logfreq_scores(panel_spec.SCATTER_HOMOLOG, _COUNTS)
    assert not set(frame["aa_site"]) & dms.unreadable_sites(CLADE)
    assert frame["score"].notna().all()


@needs_counts
def test_most_scored_cells_have_an_unobserved_mutant():
    """The caveat the figure's readout rests on, as a number rather than a claim."""
    import dms_summary

    cells = dms_models.homolog_cells(panel_spec.SCATTER_HOMOLOG)
    observed = dms_summary.mutant_observed(cells, _COUNTS)
    assert int(observed.sum()) == EXPECTED_MUTANT_OBSERVED
    assert observed.sum() < len(cells) / 2
