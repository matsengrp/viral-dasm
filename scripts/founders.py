"""Clade founder protein sequences, read from mat-pcp's own FASTAs.

Its own module because two rules need it and they should not invalidate each
other. ``rule score`` runs a GPU pass per run, and the panel rules are edited
repeatedly while a figure is being drawn; if the panels imported the scoring
script, or the scoring script the panel module, every edit to one would mark
the other's outputs out of date. A founder is the one thing both genuinely
share, so it is the one thing they import from each other's side of the
pipeline.

Founder sequences are publicly available, so what this module returns may be
committed. The counts and fitness effects it is joined to on the way to a
figure may be as well; the PCP corpora may not.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import assert_dasm2_is_this_repo, mat_pcp_path

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

from dasm2.io import read_fasta_first_sequence
from dasm2.neutral.scov2_rates import GENE_NT_LENGTH
from dasm2.sequence.conversion import translate_sequence


def founder_aa_sequence(clade: str, gene: str) -> str:
    """One clade's founder protein sequence, length-checked.

    Read from mat-pcp's own ``clade_founders/{gene}/{clade}.fa``, *not* derived
    from the fitness table. Taking the table's distinct ``aa_site`` values
    instead drops sites that happen to carry no fitness rows, and the resulting
    sequence is frame-shifted from the first gap onward with nothing raising.

    Raises:
        ValueError: If the translation's length disagrees with
            ``GENE_NT_LENGTH``, which is how a founder for the wrong gene, or a
            truncated FASTA, fails here rather than at the join.
    """
    path = mat_pcp_path("clade_founders_dir") / f"{clade}.fa"
    aa = translate_sequence(read_fasta_first_sequence(path))
    if len(aa) * 3 != GENE_NT_LENGTH[gene]:
        raise ValueError(
            f"{gene}/{clade}: founder translates to {len(aa)} codons but "
            f"GENE_NT_LENGTH says {GENE_NT_LENGTH[gene] // 3}"
        )
    return aa


def founder_nt_sequence(clade: str, gene: str) -> str:
    """One clade's founder coding sequence, trimmed to its sense codons.

    `fig:dms` splits every mutation on whether one nucleotide change to the
    founder's *actual* codon reaches it, so that panel needs the nucleotides
    rather than the translation. The FASTA carries the stop codon and the
    indicator is over sense codons only, so the return is trimmed to three
    times the translated length.

    Here rather than in ``scripts/dms.py`` so there stays one reader of
    ``clade_founders_dir``: a second one could drift from this module's
    length check and index the wrong codon at every site past the first
    disagreement, with nothing raising.

    Raises:
        ValueError: Through :func:`founder_aa_sequence`, if the translation's
            length disagrees with ``GENE_NT_LENGTH``.
    """
    path = mat_pcp_path("clade_founders_dir") / f"{clade}.fa"
    nt = read_fasta_first_sequence(path)
    return nt[: 3 * len(founder_aa_sequence(clade, gene))]


def founder_aa_sequences(clades, gene: str) -> dict[str, str]:
    """``{clade: founder}`` for several clades, in the order given."""
    return {clade: founder_aa_sequence(clade, gene) for clade in clades}
