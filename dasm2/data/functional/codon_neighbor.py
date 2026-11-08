"""Codon neighbor analysis utilities.

Provides functions for analyzing codon table bias in model predictions.
The key question: do models assign substantially lower scores to amino acid
mutations that require multiple nucleotide changes vs single-nucleotide changes?

Core utilities:
- ``make_codon_neighbor_indicator``: Build a (20, L) boolean DataFrame from a
  nucleotide sequence, marking which amino acids are reachable by single-nt
  mutation at each position.
- ``split_by_neighbor_status``: Split a melted DMS DataFrame into single-nt
  and multi-nt groups.
- ``backtranslate_human_optimal``: Deterministic backtranslation using the
  most frequent human codon for each amino acid.
"""

from typing import Dict, List, Optional

import pandas as pd

from dasm2.codon_table import CODON_NEIGHBOR_MATRIX
from dasm2.sequence.constants import AA_STR_SORTED, CODONS
from dasm2.sequence.conversion import iter_codons, translate_sequences

# Human codon usage frequencies (per thousand) from KAZUSA database.
# Source: https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=9606&aa=1
HUMAN_CODON_USAGE: Dict[str, float] = {
    "TTT": 17.6,
    "TTC": 20.3,  # Phe
    "TTA": 7.7,
    "TTG": 12.9,  # Leu
    "CTT": 13.2,
    "CTC": 19.6,  # Leu
    "CTA": 7.2,
    "CTG": 39.6,  # Leu
    "ATT": 16.0,
    "ATC": 20.8,  # Ile
    "ATA": 7.5,  # Ile
    "ATG": 22.0,  # Met
    "GTT": 11.0,
    "GTC": 14.5,  # Val
    "GTA": 7.1,
    "GTG": 28.1,  # Val
    "TCT": 15.2,
    "TCC": 17.7,  # Ser
    "TCA": 12.2,
    "TCG": 4.4,  # Ser
    "CCT": 17.5,
    "CCC": 19.8,  # Pro
    "CCA": 16.9,
    "CCG": 6.9,  # Pro
    "ACT": 13.1,
    "ACC": 18.9,  # Thr
    "ACA": 15.1,
    "ACG": 6.1,  # Thr
    "GCT": 18.4,
    "GCC": 27.7,  # Ala
    "GCA": 15.8,
    "GCG": 7.4,  # Ala
    "TAT": 12.2,
    "TAC": 15.3,  # Tyr
    "TAA": 1.0,
    "TAG": 0.8,  # Stop
    "CAT": 10.9,
    "CAC": 15.1,  # His
    "CAA": 12.3,
    "CAG": 34.2,  # Gln
    "AAT": 17.0,
    "AAC": 19.1,  # Asn
    "AAA": 24.4,
    "AAG": 31.9,  # Lys
    "GAT": 21.8,
    "GAC": 25.1,  # Asp
    "GAA": 29.0,
    "GAG": 39.6,  # Glu
    "TGT": 10.6,
    "TGC": 12.6,  # Cys
    "TGA": 1.6,  # Stop
    "TGG": 13.2,  # Trp
    "CGT": 4.5,
    "CGC": 10.4,  # Arg
    "CGA": 6.2,
    "CGG": 11.4,  # Arg
    "AGT": 12.1,
    "AGC": 19.5,  # Ser
    "AGA": 12.2,
    "AGG": 12.0,  # Arg
    "GGT": 10.8,
    "GGC": 22.2,  # Gly
    "GGA": 16.5,
    "GGG": 16.5,  # Gly
}


def _build_preferred_codons() -> Dict[str, str]:
    """Build mapping from amino acid to its most frequent human codon.

    Returns:
        Dict mapping single-letter AA code to its preferred codon string.
    """
    from Bio.Data import CodonTable

    table = CodonTable.unambiguous_dna_by_id[1]
    forward_table = table.forward_table
    preferred: Dict[str, str] = {}
    for aa in set(forward_table.values()):
        possible_codons = [
            codon for codon, amino in forward_table.items() if amino == aa
        ]
        preferred[aa] = max(possible_codons, key=lambda c: HUMAN_CODON_USAGE[c])
    return preferred


PREFERRED_CODONS: Dict[str, str] = _build_preferred_codons()
"""Mapping from amino acid single-letter code to most frequent human codon."""


def backtranslate_human_optimal(aa_sequence: str) -> str:
    """Backtranslate AA sequence to NT using most frequent human codons.

    Uses the most frequent codon for each amino acid based on human codon
    usage tables (KAZUSA database, species 9606). This is deterministic
    (no randomness).

    Args:
        aa_sequence: Amino acid sequence (single-letter codes, uppercase).

    Returns:
        Nucleotide sequence (3x length of input).

    Raises:
        KeyError: If an invalid amino acid code is encountered.
    """
    result = "".join(PREFERRED_CODONS[aa] for aa in aa_sequence.upper())
    # Validate round-trip
    round_trip = translate_sequences([result])[0]
    if round_trip != aa_sequence.upper():
        raise ValueError(
            f"Backtranslation round-trip failed: "
            f"expected {aa_sequence.upper()}, got {round_trip}"
        )
    return result


def make_codon_neighbor_indicator(
    nt_sequence: str,
    aa_sequence: Optional[str] = None,
) -> pd.DataFrame:
    """Create a boolean DataFrame indicating single-nucleotide-accessible AAs.

    For each codon position in the nucleotide sequence, determines which of
    the 20 amino acids can be reached via a single nucleotide substitution.

    Args:
        nt_sequence: Nucleotide sequence (must be 3x length of AA sequence).
        aa_sequence: Optional AA sequence for validation. If provided,
            verifies that the nucleotide sequence translates to it.

    Returns:
        DataFrame of shape (20, L) where True = single nucleotide mutation,
        False = requires multiple nucleotide changes.
        Index is AA_STR_SORTED, columns are positions 0..L-1.

    Raises:
        ValueError: If nt_sequence length is not a multiple of 3, or if
            aa_sequence doesn't match the translation.
    """
    if len(nt_sequence) % 3 != 0:
        raise ValueError(
            f"Nucleotide sequence length ({len(nt_sequence)}) must be a multiple of 3"
        )

    codons = list(iter_codons(nt_sequence))

    # Validate AA sequence if provided
    if aa_sequence is not None:
        translated = translate_sequences([nt_sequence])[0]
        if translated != aa_sequence:
            raise ValueError(
                f"Nucleotide sequence translates to '{translated}', "
                f"but expected '{aa_sequence}'"
            )

    # Build the indicator matrix
    indicator_data: Dict[int, List[bool]] = {}
    for pos, codon in enumerate(codons):
        if codon not in CODONS:
            # Ambiguous codon - no neighbors
            indicator_data[pos] = [False] * 20
        else:
            codon_idx = CODONS.index(codon)
            indicator_data[pos] = [
                bool(CODON_NEIGHBOR_MATRIX[codon_idx, aa_idx].item())
                for aa_idx in range(20)
            ]

    return pd.DataFrame(
        indicator_data,
        index=list(AA_STR_SORTED),
    )


def split_by_neighbor_status(
    melted_df: pd.DataFrame,
    neighbor_indicator: pd.DataFrame,
    aa_col: str = "aa",
    position_col: str = "position",
) -> pd.DataFrame:
    """Add neighbor status column to melted DMS data.

    Joins melted DMS data (from ``melt_dms_matrix``) with a codon neighbor
    indicator to classify each (aa, position) entry as "single" (reachable
    by single nucleotide mutation) or "multiple" (requires 2+ changes).

    Args:
        melted_df: Long-format DMS data with aa and position columns
            (e.g., from ``melt_dms_matrix``).
        neighbor_indicator: Boolean DataFrame of shape (20, L) from
            ``make_codon_neighbor_indicator``.
        aa_col: Column name for amino acid in melted_df.
        position_col: Column name for position in melted_df.

    Returns:
        Input DataFrame with added ``is_codon_neighbor`` boolean column.
    """
    # Melt the neighbor indicator to long format for merging
    neighbor_long = neighbor_indicator.reset_index().rename(columns={"index": aa_col})
    neighbor_melted = neighbor_long.melt(
        id_vars=[aa_col],
        var_name=position_col,
        value_name="is_codon_neighbor",
    )
    # Ensure position column types match
    neighbor_melted[position_col] = neighbor_melted[position_col].astype(int)

    result = melted_df.copy()
    result[position_col] = result[position_col].astype(int)

    result = result.merge(
        neighbor_melted,
        on=[aa_col, position_col],
        how="left",
    )

    return result
