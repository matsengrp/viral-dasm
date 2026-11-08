"""Sequence constants and basic definitions."""

import itertools

# Base constants
BASES = ("A", "C", "G", "T")
NUM_BASES = len(BASES)
"""Number of nucleotide bases (4: A, C, G, T)."""
NT_STR_SORTED = "".join(BASES)
BASES_AND_N_TO_INDEX = {base: idx for idx, base in enumerate(NT_STR_SORTED + "N")}
"""Mapping from nucleotide bases (A,C,G,T,N) to their index positions."""

# Separator constants for sequence segmentation
NT_SEPARATOR = "^^^"
"""Nucleotide separator token used to split multi-segment sequences."""

AA_SEPARATOR = "^"
"""Amino acid separator token used to split multi-segment sequences."""

TOKEN_CODON_MAPPING = {
    NT_SEPARATOR: AA_SEPARATOR,
}
"""Mapping from nucleotide special token codons to amino acid special tokens.

This ensures reading frame preservation through 3:1 mappings where each
3-character nucleotide token maps to a single amino acid character token.
"""

# Append any new special tokens to the end of this string:
RESERVED_TOKENS = AA_SEPARATOR
"""Reserved tokens used for sequence separators and special purposes."""

# Amino acid constants (netam-compatible)
# Standard 20 amino acids (indices 0-19, alphabetically sorted)
AA_STR_SORTED = "ACDEFGHIKLMNPQRSTVWY"

AA_STR_KOENIG = "YWFLIVAMKRHDESTNQPGC"
"""Biochemically-grouped AA ordering: aromatic (YWF), hydrophobic-aliphatic
(LIVAM), positively charged (KRH), negatively charged (DE), polar uncharged
(STNQ), special (PGC). Useful for heatmap and lattice axis / panel layouts
where within-group block structure should be visually obvious. Same 20-AA
alphabet as AA_STR_SORTED, just permuted. Adopted from
``dnsmex.koenig_helper.KoenigDataset.KOENIG_AA_ORDER`` in the dasm-experiments
repo (the Koenig DMS analysis pipeline)."""

AA_HADDOX_GROUPS = ("KRHEDQNST", "AVILM", "FYW", "GPC")
"""The four biochemical groups behind :data:`AA_STR_HADDOX`, in row order:
charged/polar (positive -> negative -> uncharged polar, ``KRHEDQNST``);
hydrophobic aliphatics by increasing side-chain size (``AVILM``); hydrophobic
aromatics (``FYW``); and the G/P/C "odd balls". Concatenates to
:data:`AA_STR_HADDOX`; the cumulative group sizes give the horizontal
boundary lines drawn between groups on DMS heatmaps."""

AA_STR_HADDOX = "".join(AA_HADDOX_GROUPS)
"""Biochemically-grouped AA ordering, alternative to AA_STR_KOENIG: charged
and polar residues first, ordered from positive charge through negative
charge to uncharged polar (KRH-ED-QNST); then hydrophobic aliphatics ordered
roughly by increasing side-chain size (AVILM); then bulky hydrophobic
aromatics (FYW); then the three "odd ball" residues that don't fit a
hydrophobicity/charge axis -- glycine (no side chain), proline (backbone
constrained), and cysteine (forms disulfides) -- grouped last (GPC). Same
20-AA alphabet as AA_STR_SORTED, just permuted (see :data:`AA_HADDOX_GROUPS`
for the group structure). Proposed by Hugh Haddox as a heatmap/panel row
order alongside AA_STR_KOENIG."""

NUM_AMINO_ACIDS = len(AA_STR_SORTED)
"""Number of standard amino acids (20)."""

# Fixed special token indices (stable across future extensions)
AA_AMBIG_IDX = NUM_AMINO_ACIDS
"""Index value (20) used for ambiguous/unknown amino acids beyond the standard 20."""

# Complete token vocabulary: 20 AAs + X (ambiguous) + special tokens
TOKEN_STR_SORTED = AA_STR_SORTED + "X" + RESERVED_TOKENS  # "ACDEFGHIKLMNPQRSTVWYX^"
"""Complete token vocabulary: 20 AA + X (ambiguous) + ^ (separator) = 22 tokens.

Index organization ensures stability:
- Indices 0-19: Standard amino acids (never change)
- Index 20: Ambiguous token 'X' (never changes)
- Index 21+: Special tokens (extensible, new tokens added at end)
"""

AA_SEPARATOR_IDX = TOKEN_STR_SORTED.index(AA_SEPARATOR)
"""Index value (21) used for separator token '^' in amino acid sequences."""

# Used for padding amino acid sequences to the same length. Differentiated by
# name in case we add a padding token other than AA_AMBIG_IDX later.
AA_PADDING_TOKEN = AA_AMBIG_IDX

# Codon constants
CODONS = ["".join(codon_list) for codon_list in itertools.product(BASES, repeat=3)]
STOP_CODONS = ["TAA", "TAG", "TGA"]
AMBIGUOUS_CODON_IDX = len(CODONS)
"""Index value (64) used for ambiguous/unknown codons beyond the standard 64."""

# Create reverse mappings
AA_STR_TO_IDX = {aa: idx for idx, aa in enumerate(AA_STR_SORTED)}
"""Reverse mapping from amino acid characters to their index positions (0-19)."""
TOKEN_TO_IDX = {token: idx for idx, token in enumerate(TOKEN_STR_SORTED)}
"""Reverse mapping from ``TOKEN_STR_SORTED`` tokens to their index positions (0-21)."""
CODON_TO_IDX = {codon: idx for idx, codon in enumerate(CODONS)}
"""Reverse mapping from codon strings to their index positions (0-63)."""
