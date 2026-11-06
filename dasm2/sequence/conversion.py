"""Sequence conversion operations (AA ↔ NT, codon operations)."""

from functools import cache
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from Bio.Seq import Seq

from dasm2.sequence.constants import (
    AA_AMBIG_IDX,
    AA_SEPARATOR_IDX,
    AA_STR_TO_IDX,
    AMBIGUOUS_CODON_IDX,
    BASES_AND_N_TO_INDEX,
    CODON_TO_IDX,
    STOP_CODONS,
    TOKEN_CODON_MAPPING,
    TOKEN_TO_IDX,
)

# Efficient codon ↔ nucleotide conversion using lookup tables (from netam)
# CODON_TO_NT_TENSOR maps (nt1_idx, nt2_idx, nt3_idx) → codon_idx
CODON_TO_NT_TENSOR = torch.arange(64).reshape(4, 4, 4)

# NT_TO_CODON_TENSOR maps codon_idx → [nt1_idx, nt2_idx, nt3_idx]
NT_TO_CODON_TENSOR = torch.zeros((64, 3), dtype=torch.long)
for i in range(4):
    for j in range(4):
        for k in range(4):
            codon_idx = CODON_TO_NT_TENSOR[i, j, k]
            NT_TO_CODON_TENSOR[codon_idx, 0] = i
            NT_TO_CODON_TENSOR[codon_idx, 1] = j
            NT_TO_CODON_TENSOR[codon_idx, 2] = k


# ---------------------------------------------------------------------------
# Module-level lookup tables for vectorized operations
# ---------------------------------------------------------------------------

# ASCII-indexed nucleotide lookup: maps ord(char) -> nt index (0-4), -1 for unknown
_NT_CHAR_TO_IDX = np.full(128, -1, dtype=np.int8)
for _char, _idx in BASES_AND_N_TO_INDEX.items():
    _NT_CHAR_TO_IDX[ord(_char)] = _idx


def count_codon_nt_differences(
    parent_idxs: torch.Tensor, child_idxs: torch.Tensor
) -> torch.Tensor:
    """Count nucleotide differences between parent and child codon indices.

    Args:
        parent_idxs: Codon indices, any shape. Values in [0, 63] (or 64 for ambiguous).
        child_idxs: Codon indices, same shape as parent_idxs.

    Returns:
        Integer tensor (same shape) with values 0-3 counting nucleotide differences.
        Ambiguous codons (index 64) are clamped to 63 for lookup; callers should
        mask these positions separately.
    """
    parent_clamped = torch.clamp(parent_idxs, max=63)
    child_clamped = torch.clamp(child_idxs, max=63)
    parent_nts = NT_TO_CODON_TENSOR[parent_clamped]  # (*shape, 3)
    child_nts = NT_TO_CODON_TENSOR[child_clamped]  # (*shape, 3)
    return (parent_nts != child_nts).sum(dim=-1)


def idx_of_codon_allowing_ambiguous(codon: str) -> int:
    """Return the index of a codon, using AMBIGUOUS_CODON_IDX for unknown codons.

    Args:
        codon: Three-letter codon string

    Returns:
        Codon index (0-63) or AMBIGUOUS_CODON_IDX (64) if not found
    """
    if codon in CODON_TO_IDX:
        return CODON_TO_IDX[codon]
    else:
        return AMBIGUOUS_CODON_IDX


@cache
def translate_codon(nt_codon: str) -> str:
    """Get amino acid translation for (possibly ambiguous) codon

    The passed nt_codon will be translated as an X if it is of length not
    equal to 3, or a stop codon. Special token codons (e.g., "^^^" → "^")
    are handled via TOKEN_CODON_MAPPING. Otherwise, it will be translated
    by BioPython. Note that a codon containing N will not necessarily be
    translated to X.
    """
    # Handle special token codons (e.g., "^^^" → "^")
    if nt_codon in TOKEN_CODON_MAPPING:
        return TOKEN_CODON_MAPPING[nt_codon]

    # Translate codon using Biopython
    if len(nt_codon) != 3 or nt_codon in STOP_CODONS:
        aa = "X"  # Ambiguous for stop codons or invalid length
    else:
        aa = str(Seq(nt_codon).translate())
    if aa == "*":
        aa = "X"

    return aa


def translate_sequence(nt_sequence: str) -> str:
    """Translate a nucleotide sequence to amino acids.

    Args:
        nt_sequence: Nucleotide sequence (length must be multiple of 3)

    Returns:
        Amino acid sequence string

    Raises:
        ValueError: If sequence length is not divisible by 3
    """
    if len(nt_sequence) % 3 != 0:
        raise ValueError(f"Sequence length {len(nt_sequence)} is not divisible by 3")

    return "".join(
        translate_codon(nt_sequence[i : i + 3]) for i in range(0, len(nt_sequence), 3)
    )


def translate_sequence_gapped(nt_sequence: str) -> str:
    """Translate a gapped nucleotide sequence, preserving gap codons as ``-``.

    Codons containing any ``-`` are emitted as ``-``; ungapped codons are
    translated via :func:`translate_codon` (so ambiguous/stop/special
    tokens remain consistent with the rest of dasm2).

    Args:
        nt_sequence: Nucleotide sequence with gaps preserved (length must
            be a multiple of 3).

    Returns:
        Amino acid sequence with ``-`` at gap codons.

    Raises:
        ValueError: If sequence length is not divisible by 3.
    """
    if len(nt_sequence) % 3 != 0:
        raise ValueError(f"Sequence length {len(nt_sequence)} is not divisible by 3")

    return "".join(
        "-" if "-" in (codon := nt_sequence[i : i + 3]) else translate_codon(codon)
        for i in range(0, len(nt_sequence), 3)
    )


def translate_sequences(nt_sequences: List[str]) -> List[str]:
    """Translate multiple nucleotide sequences to amino acids.

    Args:
        nt_sequences: List of nucleotide sequence strings

    Returns:
        List of amino acid sequence strings
    """
    return [translate_sequence(seq) for seq in nt_sequences]


def iter_codons(nt_seq: str) -> Iterator[str]:
    """Iterate over codons in a nucleotide sequence.

    Yields consecutive 3-nucleotide substrings from the sequence.
    """
    for i in range(0, len(nt_seq), 3):
        yield nt_seq[i : i + 3]


def contains_stop_codon(nt_seq: str) -> bool:
    """Check if nucleotide sequence contains stop codons.

    Returns:
        True if any codon in the sequence matches a stop codon (TAA, TAG, TGA)
    """
    return any(codon in STOP_CODONS for codon in iter_codons(nt_seq))


def unflatten_codon_idxs(codon_idx_tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor of codon indices to tensor of nucleotide indices.

    Uses efficient lookup table from netam for O(1) conversion instead of
    string manipulation loops.

    Args:
        codon_idx_tensor: Tensor of shape [seq_len] containing codon indices (0-63)

    Returns:
        Tensor of shape [seq_len, 3] containing nucleotide indices (0-3 for A,C,G,T)

    Example:
        >>> codon_idxs = torch.tensor([0, 21, 63])  # AAA, CAT, TTT
        >>> nt_idxs = unflatten_codon_idxs(codon_idxs)
        >>> nt_idxs.shape
        torch.Size([3, 3])
        >>> nt_idxs[0]  # AAA = [0, 0, 0]
        tensor([0, 0, 0])
    """
    return NT_TO_CODON_TENSOR[codon_idx_tensor]


def flatten_codon_idxs(nt_codon_tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor of nucleotide indices to codon indices.

    Uses efficient lookup table from netam for O(1) conversion instead of
    string manipulation loops.

    Args:
        nt_codon_tensor: Tensor of shape [seq_len, 3] or [..., seq_len, 3]
                        containing nucleotide indices (0-3 for A,C,G,T)

    Returns:
        Tensor of shape [seq_len] or [..., seq_len] with codon indices (0-63)

    Example:
        >>> nt_idxs = torch.tensor([[0, 0, 0], [1, 0, 3], [3, 3, 3]])  # AAA, CAT, TTT
        >>> codon_idxs = flatten_codon_idxs(nt_idxs)
        >>> codon_idxs
        tensor([0, 21, 63])
    """
    original_shape = nt_codon_tensor.shape
    flat_codon_tensor = nt_codon_tensor.view(-1, 3)
    return CODON_TO_NT_TENSOR[
        flat_codon_tensor[:, 0], flat_codon_tensor[:, 1], flat_codon_tensor[:, 2]
    ].reshape(*original_shape[:-1])


def codon_idx_tensor_of_str_ambig(nt_str: str) -> torch.Tensor:
    """Convert nucleotide string to codon index tensor, allowing ambiguous codons.

    Uses vectorized numpy operations for O(1)-per-codon conversion instead
    of a Python loop with dict lookups.

    Args:
        nt_str: Nucleotide sequence (length must be multiple of 3)

    Returns:
        Long tensor of codon indices (0-63 for standard, AMBIGUOUS_CODON_IDX for ambiguous)

    Raises:
        ValueError: If sequence length not divisible by 3
    """
    if len(nt_str) % 3 != 0:
        raise ValueError("Nucleotide string length must be divisible by 3")

    arr = np.frombuffer(nt_str.encode("ascii"), dtype=np.uint8)
    nt_idxs = _NT_CHAR_TO_IDX[arr]  # vectorized lookup
    codons = nt_idxs.reshape(-1, 3)
    # N maps to index 4, any non-ACGTN maps to -1; both are "ambiguous"
    # Use bitwise OR of "not in 0..3" checks for all three positions
    not_acgt = (
        ((codons[:, 0] < 0) | (codons[:, 0] > 3))
        | ((codons[:, 1] < 0) | (codons[:, 1] > 3))
        | ((codons[:, 2] < 0) | (codons[:, 2] > 3))
    )
    codon_indices = (
        codons[:, 0].astype(np.int64) * 16
        + codons[:, 1].astype(np.int64) * 4
        + codons[:, 2].astype(np.int64)
    )
    codon_indices[not_acgt] = AMBIGUOUS_CODON_IDX
    return torch.from_numpy(codon_indices.copy())


def aa_idx_tensor_of_str_ambig(aa_str: str) -> torch.Tensor:
    """Convert an amino acid string to token indices, allowing ``X`` and ``^``.

    Indices are positions in ``TOKEN_STR_SORTED``. Ported from netam's function
    of the same name, including its ``torch.int`` dtype.

    Raises:
        ValueError: If ``aa_str`` contains a character not in ``TOKEN_STR_SORTED``.
    """
    try:
        return torch.tensor([TOKEN_TO_IDX[aa] for aa in aa_str], dtype=torch.int)
    except KeyError as e:
        raise ValueError(f"Found an invalid amino acid in the string: {aa_str}") from e


def aa_index_of_codon(codon: str) -> int:
    """Get amino acid index for a codon.

    Returns:
        Amino acid index (0-19) or AA_AMBIG_IDX (20) for invalid codons
    """
    aa = translate_codon(codon)
    return AA_STR_TO_IDX.get(aa, AA_AMBIG_IDX)


def three_letter_aa_to_one_letter(three_letter_code: str) -> str:
    """Convert 3-letter amino acid code to 1-letter code.

    This function converts PDB-style 3-letter amino acid codes (e.g., "ALA", "GLY")
    to standard 1-letter codes (e.g., "A", "G").

    Args:
        three_letter_code: 3-letter amino acid code (case insensitive)

    Returns:
        1-letter amino acid code, or "X" for unknown codes

    Example:
        >>> three_letter_aa_to_one_letter("ALA")
        'A'
        >>> three_letter_aa_to_one_letter("gly")
        'G'
        >>> three_letter_aa_to_one_letter("XXX")
        'X'
    """
    mapping = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    return mapping.get(three_letter_code.upper(), "X")


# ---------------------------------------------------------------------------
# Fused translate + tokenize + codon index
# ---------------------------------------------------------------------------


@cache
def _get_fused_lookup_tables() -> Tuple[Dict[str, str], Dict[str, int], Dict[str, int]]:
    """Build lookup tables for fused translate+tokenize+codon_index.

    Covers all 64 standard codons, the "^^^" separator, and all N-containing
    codons ({A,C,G,T,N}^3). Built on first call, cached forever.

    Uses the existing ``translate_codon`` function (which delegates to BioPython
    for N-containing codons) to ensure bit-for-bit identical output.

    Note:
        Returned dicts are read-only; do not mutate. They are shared across all
        callers via ``@functools.cache``.

    Returns:
        Tuple of (codon_to_aa_char, codon_to_aa_token, codon_to_codon_idx).
    """
    codon_to_aa_char: Dict[str, str] = {}
    codon_to_aa_token: Dict[str, int] = {}
    codon_to_codon_idx: Dict[str, int] = {}

    # Standard 64 codons
    for codon, codon_idx in CODON_TO_IDX.items():
        aa_char = translate_codon(codon)
        codon_to_aa_char[codon] = aa_char
        codon_to_aa_token[codon] = TOKEN_TO_IDX.get(aa_char.upper(), AA_AMBIG_IDX)
        codon_to_codon_idx[codon] = codon_idx

    # Separator: "^^^" → "^" (AA), token 21, codon_idx 64
    codon_to_aa_char["^^^"] = "^"
    codon_to_aa_token["^^^"] = AA_SEPARATOR_IDX
    codon_to_codon_idx["^^^"] = AMBIGUOUS_CODON_IDX

    # N-containing codons: all {A,C,G,T,N}^3 not already covered
    # BioPython handles these with IUPAC ambiguity rules, e.g.:
    #   "ACN" → "T" (all ACx codons encode Thr)
    #   "NNN" → "X" (fully ambiguous)
    for b1 in "ACGTN":
        for b2 in "ACGTN":
            for b3 in "ACGTN":
                codon = b1 + b2 + b3
                if codon not in codon_to_aa_char:
                    aa_char = translate_codon(codon)
                    codon_to_aa_char[codon] = aa_char
                    codon_to_aa_token[codon] = TOKEN_TO_IDX.get(
                        aa_char.upper(), AA_AMBIG_IDX
                    )
                    codon_to_codon_idx[codon] = AMBIGUOUS_CODON_IDX

    return codon_to_aa_char, codon_to_aa_token, codon_to_codon_idx


def fused_translate_tokenize_codon_index(
    nt_seq: str,
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """Translate NT sequence and produce AA tokens + codon indices in one pass.

    Bit-for-bit identical to running these three operations separately::

        aa_str = translate_sequence(nt_seq)
        aa_tokens = tokenize_single_aa_sequence(aa_str)
        codon_idxs = codon_idx_tensor_of_str_ambig(nt_seq)

    But faster: single loop over codons, one dict lookup each for AA token and
    codon index, two numpy arrays instead of separate tensor allocations.

    Args:
        nt_seq: Nucleotide sequence (length must be divisible by 3).
            May contain "^^^" separators and N-containing codons.

    Returns:
        Tuple of (aa_string, aa_token_tensor, codon_idx_tensor):
            - aa_string: Translated AA string (e.g., "MK^VL")
            - aa_token_tensor: LongTensor [num_codons] of AA token indices
            - codon_idx_tensor: LongTensor [num_codons] of codon indices (0-63,
              or 64 for ambiguous/separator)

    Raises:
        ValueError: If sequence length is not divisible by 3.
    """
    n = len(nt_seq)
    if n % 3 != 0:
        raise ValueError(f"Sequence length {n} is not divisible by 3")

    codon_to_aa_char, codon_to_aa_token, codon_to_codon_idx = _get_fused_lookup_tables()

    num_codons = n // 3
    aa_tokens = np.empty(num_codons, dtype=np.int64)
    codon_idxs = np.empty(num_codons, dtype=np.int64)
    aa_chars: List[str] = []

    for i in range(num_codons):
        codon = nt_seq[i * 3 : i * 3 + 3]
        try:
            aa_chars.append(codon_to_aa_char[codon])
            aa_tokens[i] = codon_to_aa_token[codon]
            codon_idxs[i] = codon_to_codon_idx[codon]
        except KeyError:
            # Fallback for codons outside {A,C,G,T,N}^3 + "^^^".
            aa_char = translate_codon(codon)
            aa_chars.append(aa_char)
            aa_tokens[i] = AA_AMBIG_IDX
            codon_idxs[i] = AMBIGUOUS_CODON_IDX

    return (
        "".join(aa_chars),
        torch.from_numpy(aa_tokens),
        torch.from_numpy(codon_idxs),
    )


# Byte → nt-window index: A/C/G/T → 0..3; everything else (N, segment
# separators, out-of-range padding) → 4. Used by build_codon_nt_windows.
_NT_BYTE_TO_WINDOW_INDEX = np.full(256, 4, dtype=np.int8)
_NT_BYTE_TO_WINDOW_INDEX[ord("A")] = 0
_NT_BYTE_TO_WINDOW_INDEX[ord("C")] = 1
_NT_BYTE_TO_WINDOW_INDEX[ord("G")] = 2
_NT_BYTE_TO_WINDOW_INDEX[ord("T")] = 3


def build_codon_nt_windows(
    nt_sequence: str, n_codons: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build per-codon-site 15-nt windows + boundary mask for the 13-mer Q.

    For each codon site ``ℓ`` in ``range(n_codons)``, extracts the 15-nt window
    spanning nt positions ``[3ℓ - 6, 3ℓ + 8]`` — the codon's three nt plus 6 nt
    of flank on each side. The window is wide enough to hold the three 13-mers
    centered on each of the codon's nt positions, the receptive field the
    source-aware Thrifty 13-mer lookup needs (see
    ``dasm2.molevol.kmer_q``).

    Single shared implementation behind both
    :class:`dasm2.data.transforms.ComputeParentNtWindows` (dataloader path) and
    the export pipeline in ``dasm2.analysis.rate_extraction`` so the two cannot
    drift.

    Args:
        nt_sequence: Nucleotide string. A/C/G/T map to 0..3; everything else
            (N, segment separators, out-of-range padding) maps to 4 and flags
            the containing codon site in ``boundary_mask``.
        n_codons: Number of codon sites to emit windows for. Single in-frame
            segments pass ``len(nt_sequence) // 3``; the dataloader transform
            passes the precomputed codon count, which can differ from
            ``len(nt_sequence) // 3`` when ``nt_sequence`` contains separators.

    Returns:
        ``(window, boundary_mask)`` where ``window`` is
        ``LongTensor[n_codons, 15]`` (nt indices 0..3, or 4 for
        N/separator/padding) and ``boundary_mask`` is ``BoolTensor[n_codons]``,
        True at sites whose window contains any 4.
    """
    seq_bytes = np.frombuffer(nt_sequence.encode("ascii"), dtype=np.uint8)
    nt_idx_np = _NT_BYTE_TO_WINDOW_INDEX[seq_bytes]
    # Pre-pad 6 nt of boundary marker (4) on each side so codon ℓ reads
    # padded[3ℓ : 3ℓ + 15].
    nt_with_pad = np.full(nt_idx_np.shape[0] + 14, 4, dtype=np.int8)
    nt_with_pad[6 : 6 + nt_idx_np.shape[0]] = nt_idx_np
    starts = np.arange(n_codons) * 3
    idxs = starts[:, None] + np.arange(15)[None, :]
    window = torch.from_numpy(nt_with_pad[idxs].astype(np.int64))
    boundary_mask = (window == 4).any(dim=-1)
    return window, boundary_mask
