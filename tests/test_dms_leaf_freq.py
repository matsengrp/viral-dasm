"""What a wrong leaf reconstruction would report, and what catches it.

A reconstruction off by one codon, started from the wrong sequence, or applied in
the wrong order still produces a full 20 x 1,273 table of plausible counts and a
plausible positive correlation with the DMS. Nothing downstream notices. So the
checks here are of three kinds: every mutation token's own reference base has to
agree with the base it is applied over, every site's residues have to sum to the
leaves that were read, and the score has to be the log ratio it claims to be --
a swapped numerator, a changed pseudocount or an ``aa_site`` off by one all
return finite numbers that every structural check accepts.

The tally the table checks read is **built here**, not read from the rule's
output: reading the artifact would test whichever version of the code last wrote
it, and would skip wherever it has not been built. One test does read the
artifact, to say it is current.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import lib

import dasm2  # noqa: F401

pd = pytest.importorskip("pandas")

import dms
import dms_leaf_freq
import dms_leaf_scores
import panel_spec
from founders import founder_aa_sequence

from dasm2.neutral.scov2_rates import gene_nt_span
from dasm2.sequence.constants import AA_STR_SORTED, STOP_CODONS
from dasm2.sequence.conversion import translate_codon

#: The clade KP.3.1.1 is matched to, and the one the real-data checks here use. Its
#: homolog is `panel_spec.SCATTER_HOMOLOG`'s, so a repoint moves both.
CLADE = "24E"

#: How many of that clade's leaves the reference-base check reads. `build_tally`
#: checks all 124,359 and refuses the table on any mismatch, so this is the fast
#: copy of that check. A shifted index map fails at the first leaf, not the last:
#: replayed against an offset of plus or minus one, these 3,000 leaves give over
#: 140,000 mismatches.
LEAVES_CHECKED = 3_000

#: Sites where the modal residue is not the clade founder's, per clade.
#:
#: Two different causes, and the distinction is the point. In 23A and 24E it is
#: codon 142, whose nucleotide 21987 is `vcf_masked_sites`: no path carries a
#: mutation there, so the reconstruction keeps 19A's residue at every leaf. In
#: 21K it is site 346, which is neither masked nor censored -- BA.1.1's R346K is
#: carried by 54.6% of the clade's leaves, so the clade's own modal residue
#: genuinely is not its founder's.
#:
#: **So "the consensus reproduces the founder outside the mask" is not a general
#: invariant.** A test that assumed it would fail on 21K for a reason that is a
#: correct reconstruction rather than a wrong one.
EXPECTED_FOUNDER_DISAGREEMENTS = {"21K": {346}, "23A": {142}, "24E": {142}}

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

#: The rule's own output, read by one test that checks it is current.
_COUNTS = panel_spec.dms_leaf_count_path(CLADE)


def _counts_readable() -> bool:
    try:
        return _COUNTS.is_file()
    except OSError:
        return False


needs_counts = pytest.mark.skipif(
    not _counts_readable(), reason=f"{_COUNTS} has not been built here"
)


@pytest.fixture(scope="session")
def tally():
    """Clade 24E's tally, built here rather than read from the rule's output.

    About 20 s for all 124,359 leaves, once for the session.
    """
    if not _mat_pcp_readable():
        pytest.skip("this run's mat-pcp outputs are not readable here")
    return dms_leaf_freq.build_tally(CLADE)


@pytest.fixture(scope="session")
def tally_counts(tally, tmp_path_factory):
    """That tally written out and read back, as a predictor sees it."""
    path = tmp_path_factory.mktemp("leaf_counts") / f"{CLADE}.csv"
    tally.frame().to_csv(path, index=False)
    return path


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
    """The one check that the index map, the start state and the order are right."""
    leaf_tally = dms_leaf_freq.LeafTally(CLADE)
    with dms_leaf_freq.sample_paths_path(CLADE).open() as handle:
        for index, line in enumerate(handle):
            if index >= LEAVES_CHECKED:
                break
            leaf_tally.add_leaf(line)
    assert leaf_tally.n_leaves == LEAVES_CHECKED
    assert leaf_tally.reference_mismatches == 0, leaf_tally.mismatch_examples

    # And the counter is live rather than inert: a deliberately wrong reference
    # base is counted.
    start = gene_nt_span(lib.GENE)[0]
    actual = leaf_tally.root_nt[0]
    wrong = next(base for base in "ACGT" if base != actual)
    leaf_tally.add_leaf(_path_line(f"{wrong}{start}A"))
    assert leaf_tally.reference_mismatches == 1


@needs_mat_pcp
def test_a_disagreeing_reference_base_refuses_the_whole_tally(tmp_path, monkeypatch):
    """`build_tally` must not return a table it knows is unreliable."""
    start = gene_nt_span(lib.GENE)[0]
    probe = dms_leaf_freq.LeafTally(CLADE)
    wrong = next(base for base in "ACGT" if base != probe.root_nt[0])
    paths = tmp_path / f"{CLADE}{lib.SAMPLE_PATHS_SUFFIX}"
    paths.write_text(_path_line(f"{wrong}{start}A"))
    monkeypatch.setattr(dms_leaf_freq, "sample_paths_path", lambda clade: paths)
    with pytest.raises(ValueError, match="disagree with the"):
        dms_leaf_freq.build_tally(CLADE)


@needs_mat_pcp
def test_an_empty_paths_file_is_refused(tmp_path, monkeypatch):
    """An all-zero tally scores every substitution at log(1/1) and raises nothing."""
    paths = tmp_path / f"{CLADE}{lib.SAMPLE_PATHS_SUFFIX}"
    paths.write_text("")
    monkeypatch.setattr(dms_leaf_freq, "sample_paths_path", lambda clade: paths)
    with pytest.raises(ValueError, match="no leaves"):
        dms_leaf_freq.build_tally(CLADE)


@needs_mat_pcp
def test_a_reversion_leaves_the_leaf_at_the_reference_residue():
    """The result a `leaves_sharing_mutations` tally gets wrong.

    A leaf's residue is set by the *last* mutation on its path, so a site mutated
    and then reverted contributes no deviation. An implementation that ignored
    order, took first-wins, or weighted each node by its descendant count would
    record one.
    """
    leaf_tally = dms_leaf_freq.LeafTally(CLADE)
    start = gene_nt_span(lib.GENE)[0]
    # A site in the second codon, so the first stays untouched and the deviation
    # count cannot be hidden by a boundary effect.
    site = start + 3
    original = leaf_tally.root_nt[3]
    mutant = next(base for base in "ACGT" if base != original)
    leaf_tally.add_leaf(
        _path_line(f"{original}{site}{mutant}", f"{mutant}{site}{original}")
    )
    assert leaf_tally.reference_mismatches == 0
    assert int(leaf_tally.deviations.sum()) == 0
    assert int(leaf_tally.unreadable.sum()) == 0

    # And a leaf that mutates without reverting does deviate, so the assertion
    # above is the reversion and not an inert path.
    leaf_tally.add_leaf(_path_line(f"{original}{site}{mutant}"))
    assert int(leaf_tally.deviations.sum()) + int(leaf_tally.unreadable.sum()) == 1


@needs_mat_pcp
def test_a_codon_outside_the_alphabet_is_dropped_from_its_site():
    """A stop must not be tallied as a residue, nor as the reference."""
    leaf_tally = dms_leaf_freq.LeafTally(CLADE)
    start = gene_nt_span(lib.GENE)[0]

    # Found from the sequence itself rather than from a remembered position.
    found = None
    for codon_site in range(leaf_tally.n_sites):
        codon = leaf_tally.root_nt[3 * codon_site : 3 * codon_site + 3]
        for offset in range(3):
            for base in "ACGT":
                candidate = codon[:offset] + base + codon[offset + 1 :]
                if base != codon[offset] and candidate in STOP_CODONS:
                    found = (codon_site, offset, codon[offset], base)
                    break
            if found:
                break
        if found:
            break
    assert found is not None, "no codon in this founder is one change from a stop"

    codon_site, offset, reference, mutant = found
    site = start + 3 * codon_site + offset
    leaf_tally.add_leaf(_path_line(f"{reference}{site}{mutant}"))
    assert leaf_tally.reference_mismatches == 0
    assert int(leaf_tally.deviations.sum()) == 0
    assert leaf_tally.unreadable[codon_site] == 1

    # The site's column then sums to one fewer than the leaf count, which is what
    # "dropped, not counted" means in the written table.
    table = leaf_tally.frame()
    row = table.loc[table[dms_leaf_freq.SITE_COLUMN] == codon_site + 1]
    assert int(row[list(AA_STR_SORTED)].to_numpy().sum()) == leaf_tally.n_leaves - 1


@needs_mat_pcp
def test_the_genome_to_codon_map_agrees_with_mat_pcps_own_table():
    """`clade_founder_nts.csv` as an oracle for a map the code derives.

    The code indexes with ``gene_nt_span``; mat-pcp publishes the same mapping as
    data. Checked against each other rather than one being read, so neither can
    drift into the other's place.
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


def test_every_site_tallies_exactly_the_leaves_it_saw(tally):
    """The identity the deviation optimisation stands on.

    Only the codons a leaf's path touches are counted, and the reference's own
    count is the leaf total minus what deviated and what was unreadable. If that
    arithmetic is wrong anywhere, some site's residues do not sum to the leaves.
    """
    table = tally.frame()
    totals = table[list(AA_STR_SORTED)].to_numpy().sum(axis=1)
    assert (totals + tally.unreadable == tally.n_leaves).all()
    assert int(tally.unreadable.sum()) > 0, (
        "no leaf's codon read outside the alphabet, so the `unreadable` term in "
        "that identity is untested here"
    )


def test_the_modal_residue_is_the_clade_founder_where_it_can_be(tally_counts):
    """A reconstruction off by a codon still tallies; this is what notices."""
    counts = dms_leaf_freq.read_counts(tally_counts)
    founder = founder_aa_sequence(CLADE, lib.GENE)
    assert len(counts) == len(founder)
    modal = counts.to_numpy().argmax(axis=1)
    disagree = {
        site + 1
        for site in range(len(founder))
        if AA_STR_SORTED[modal[site]] != founder[site]
    }
    assert disagree == EXPECTED_FOUNDER_DISAGREEMENTS[CLADE], (
        f"{CLADE}'s founder disagreements moved to {sorted(disagree)}. Check "
        f"`masked_nt_sites_file` and the clade founder FASTAs before changing "
        f"this pin."
    )


def test_the_tally_carries_standing_variation(tally_counts):
    """A table that reproduced the founder everywhere did nothing at all."""
    counts = dms_leaf_freq.read_counts(tally_counts)
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


def test_a_reordered_tally_is_refused(tmp_path, tally):
    """The per-cell lookup indexes by position, so the rows must be 1..L in order."""
    shuffled = tmp_path / "shuffled.csv"
    tally.frame().iloc[::-1].to_csv(shuffled, index=False)
    with pytest.raises(ValueError, match="in order"):
        dms_leaf_freq.read_counts(shuffled)


def test_no_masked_site_reaches_a_scored_cell(tally_counts):
    """The reconstruction's known blind spot must not enter a correlation."""
    frame, _, _ = dms_leaf_scores.leaf_logfreq_scores(
        panel_spec.SCATTER_HOMOLOG, tally_counts
    )
    assert not set(frame["aa_site"]) & dms.unreadable_sites(CLADE)
    assert frame["score"].notna().all()


def test_the_score_is_the_log_ratio_it_claims_to_be(tally_counts):
    """The assertion nothing structural makes: the score's own value.

    A swapped numerator and denominator flips its sign; a pseudocount of 0 or 5
    changes every unobserved cell; indexing the tally by ``aa_site`` rather than
    ``aa_site - 1`` returns the next site's counts. All three satisfy every row
    count, key and hash check in this suite and reach the table as a correlation of
    the wrong sign or magnitude.
    """
    counts = dms_leaf_freq.read_counts(tally_counts)
    frame, _, _ = dms_leaf_scores.leaf_logfreq_scores(
        panel_spec.SCATTER_HOMOLOG, tally_counts
    )
    pseudo = dms_leaf_scores.LEAF_PSEUDOCOUNT
    for row in frame.head(500).itertuples():
        mutant = counts.loc[row.aa_site, row.mutant_aa]
        parent = counts.loc[row.aa_site, row.parent_aa]
        assert row.score == pytest.approx(
            float(np.log((mutant + pseudo) / (parent + pseudo)))
        )

    # And the sign is the way round the panel reads it: at a site carrying one
    # observed substitution and one unobserved, the observed one scores higher.
    melted = counts.reset_index().melt(
        id_vars=dms_leaf_freq.SITE_COLUMN,
        var_name="mutant_aa",
        value_name="leaf_count",
    )
    merged = frame.merge(
        melted,
        left_on=["aa_site", "mutant_aa"],
        right_on=[dms_leaf_freq.SITE_COLUMN, "mutant_aa"],
        how="left",
        validate="one_to_one",
    )
    seen = merged[merged["leaf_count"] > 0]
    unseen = merged[merged["leaf_count"] == 0]
    assert len(seen) and len(unseen)
    assert seen["score"].median() > unseen["score"].median()


@needs_counts
def test_the_rules_output_is_what_this_code_produces(tally):
    """The committed artifact and a fresh build must agree.

    Every other table check here runs on a fresh build, so this is the one that
    says the file the pipeline left on disk is current with the code reading it.
    """
    assert dms_leaf_freq.read_counts(_COUNTS).equals(
        tally.frame().set_index(dms_leaf_freq.SITE_COLUMN)
    )


@needs_counts
def test_the_written_table_carries_no_sample_name():
    """Its input's rows name GISAID samples; nothing it writes may.

    Against the rule's own output, since what is checked is the file this
    pipeline leaves on disk.
    """
    text = _COUNTS.read_text()
    header = text.splitlines()[0].split(",")
    assert header == [dms_leaf_freq.SITE_COLUMN, *AA_STR_SORTED]
    # The sample pattern, not just the accession prefix: every line of
    # `mat/clades/{clade}_samples.txt` is `name|EPI_ISL_n|date`.
    assert not re.search(r"[^,\n]*\|EPI_ISL_\d+\|\d{4}-\d{2}-\d{2}", text)
    # And every field past the header is an integer, so no name of any form fits.
    assert re.fullmatch(r"[\d,\n]+", text[text.index("\n") + 1 :])


@needs_mat_pcp
def test_the_optimised_tally_equals_a_full_count():
    """The optimisation itself, against a tally that counts every site.

    `LeafTally` counts only the codons a leaf's path touches and fills the rest
    in at the end. Asserting that its own rows sum to the leaf count is algebra,
    since it is built that way -- what is worth checking is that the shortcut
    agrees with reconstructing and translating each leaf whole.
    """
    sample = 150
    optimised = dms_leaf_freq.LeafTally(CLADE)
    lines = []
    with dms_leaf_freq.sample_paths_path(CLADE).open() as handle:
        for index, line in enumerate(handle):
            if index >= sample:
                break
            lines.append(line)
            optimised.add_leaf(line)

    # The same leaves, counted the slow way: apply the path, translate all 1,273
    # codons, tally every one.
    start = gene_nt_span(lib.GENE)[0]
    root = optimised.root_nt
    brute = np.zeros((optimised.n_sites, len(AA_STR_SORTED)), dtype=np.int64)
    for line in lines:
        applied = {}
        for _reference, site, mutant in dms_leaf_freq.iter_leaf_mutations(line):
            offset = site - start
            if 0 <= offset < len(root):
                applied[offset] = mutant
        sequence = [applied.get(i, base) for i, base in enumerate(root)]
        for codon_site in range(optimised.n_sites):
            residue = translate_codon("".join(sequence[3 * codon_site :][:3]))
            if residue in AA_STR_SORTED:
                brute[codon_site, AA_STR_SORTED.index(residue)] += 1

    fast = optimised.frame()[list(AA_STR_SORTED)].to_numpy()
    assert (fast == brute).all(), (
        f"the deviation shortcut disagrees with a full count at "
        f"{int((fast != brute).sum())} cells"
    )
    # Not a vacuous agreement: these leaves do deviate from 19A somewhere.
    assert int(optimised.deviations.sum()) > 0


@needs_mat_pcp
def test_a_reordered_residue_column_is_refused(tmp_path, tally):
    """The lookup is positional in both axes, not only in the row index."""
    table = tally.frame()
    reordered = tmp_path / "reordered.csv"
    swapped = list(AA_STR_SORTED)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    table[[dms_leaf_freq.SITE_COLUMN, *swapped]].to_csv(reordered, index=False)
    # `read_counts` projects back to `AA_STR_SORTED`, so the frame it returns is
    # the same whichever order the file carried.
    assert dms_leaf_freq.read_counts(reordered).equals(
        table.set_index(dms_leaf_freq.SITE_COLUMN)
    )
