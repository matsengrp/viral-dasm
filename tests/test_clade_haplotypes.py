"""The contract between the haplotype enumeration and the scorer.

``scripts/score_haplotypes.py`` refuses a sequence carrying anything but the
twenty residues, and refuses an empty ``background`` label. Both refusals are
deliberate -- an ambiguity code reaches the model as a silently wrong input,
and an empty background is the founder table's convention for "this row is a
founder". So whatever ``clade_haplotypes`` emits has to satisfy them, and that
is what these tests pin. The failure they prevent is a scoring run that dies
thousands of haplotypes in, after the enumeration reported success.

No corpus and no checkpoint: the clade founder is real, because the labels are
substitutions against it, and the branches are four hand-built parents.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

pd = pytest.importorskip("pandas")

import clade_haplotypes
import lib
import panels
import score_haplotypes

#: A clade of the figure, so the founder these parents are built from is the
#: one the producer will actually use.
CLADE = "22D"

#: Sites the synthetic parents differ at. Well inside the gene, and distinct,
#: so one parent's substitution cannot be read onto another's.
SUBSTITUTED_SITE = 300
SILENT_SITE = 310
STOPPED_SITE = 320


def founder_nt() -> str:
    """The clade's founder nucleotides, or skip if mat-pcp is not readable."""
    try:
        return panels._founder_nt_sequence(CLADE)
    except (FileNotFoundError, KeyError) as error:  # pragma: no cover
        pytest.skip(f"mat-pcp founders are not readable here: {error}")


def with_codon(sequence: str, site: int, codon: str) -> str:
    """One codon replaced, 1-based."""
    return sequence[: 3 * (site - 1)] + codon + sequence[3 * site :]


@pytest.fixture(scope="module")
def enumerated(tmp_path_factory) -> pd.DataFrame:
    """Four parents through the real producer.

    One is the founder itself, two translate to the same haplotype by
    different codons, and one carries a stop.
    """
    founder = founder_nt()
    # `GCT` and `GCC` are both alanine, so these two nucleotide parents are
    # one amino-acid haplotype and must be merged.
    alanine_one = with_codon(founder, SUBSTITUTED_SITE, "GCT")
    alanine_two = with_codon(alanine_one, SILENT_SITE, founder_codon_synonym(founder))
    stopped = with_codon(founder, STOPPED_SITE, "TAA")
    if len({founder, alanine_one, alanine_two, stopped}) != 4:
        pytest.skip(f"{CLADE}'s founder already carries one of the test codons")

    corpus = tmp_path_factory.mktemp("corpus") / "pcps.csv"
    pd.DataFrame(
        {
            "clade": [CLADE] * 5,
            "parent": [founder, founder, alanine_one, alanine_two, stopped],
        }
    ).to_csv(corpus, index=False)
    return clade_haplotypes.clade_haplotypes([CLADE], corpus, chunk_rows=2)


def founder_codon_synonym(founder: str) -> str:
    """A different codon for whatever the founder carries at ``SILENT_SITE``.

    Picked rather than hard-coded, so the fixture does not assume a residue.
    A synonymous change is what makes the two parents one haplotype.
    """
    from dasm2.sequence.conversion import translate_codon

    current = founder[3 * (SILENT_SITE - 1) : 3 * SILENT_SITE]
    residue = translate_codon(current)
    for first in "ACGT":
        for second in "ACGT":
            for third in "ACGT":
                codon = first + second + third
                if codon != current and translate_codon(codon) == residue:
                    return codon
    pytest.skip(f"no synonymous alternative to {current}")


def test_the_founder_identical_haplotype_is_labelled_not_left_empty(enumerated):
    """An empty label is what `read_backgrounds` refuses.

    It is also usually the heaviest haplotype of the founder-side pool, so
    dropping it instead is not an option.
    """
    labels = set(enumerated["background"])
    assert clade_haplotypes.FOUNDER_LABEL in labels
    assert "" not in labels
    founder_row = enumerated[
        enumerated["background"] == clade_haplotypes.FOUNDER_LABEL
    ].iloc[0]
    assert founder_row["n_pcps"] == 2


def test_a_parent_with_an_unreadable_codon_is_dropped(enumerated):
    """Rather than reaching the model as an `X`.

    Three haplotypes out of four parents: the stopped one is gone, and the
    two alanine parents are one haplotype.
    """
    assert len(enumerated) == 2
    assert not any("X" in sequence for sequence in enumerated["aa_sequence"])


def test_two_codons_for_one_residue_are_one_haplotype(enumerated):
    """The reason the pass translates distinct parents rather than branches.

    Scoring them separately would be two identical forward passes, and would
    split one pool's evolutionary time across two keys that the counts side
    joins as one.
    """
    substituted = enumerated[
        enumerated["background"] != clade_haplotypes.FOUNDER_LABEL
    ].iloc[0]
    assert substituted["n_nt_parents"] == 2
    assert substituted["n_pcps"] == 2


def test_the_label_is_the_substitution_against_the_founder(enumerated):
    """And it is the key the counts side's weights are built on."""
    substituted = enumerated[
        enumerated["background"] != clade_haplotypes.FOUNDER_LABEL
    ].iloc[0]
    founder_aa = panels.founder_aa_sequences([CLADE], lib.GENE)[CLADE]
    expected = f"{founder_aa[SUBSTITUTED_SITE - 1]}{SUBSTITUTED_SITE}A"
    assert substituted["background"] == expected


def test_what_is_emitted_is_what_the_scorer_accepts(enumerated, tmp_path):
    """The whole point: the two modules agree on the table's shape.

    `read_backgrounds` is the producer's own validation, so running it over
    this output is the contract test rather than a restatement of it.
    """
    path = tmp_path / "haplotypes.parquet"
    enumerated.to_parquet(path, index=False)
    sites = len(panels.founder_aa_sequences([CLADE], lib.GENE)[CLADE])
    accepted = score_haplotypes.read_backgrounds(path, sites=sites)
    assert len(accepted) == len(enumerated)
    assert set(accepted["clade"]) == {CLADE}
