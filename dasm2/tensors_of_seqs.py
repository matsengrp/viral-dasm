"""Tensor representations of biological sequences."""

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from dasm2.sequence.constants import (
    AMBIGUOUS_CODON_IDX,
    BASES_AND_N_TO_INDEX,
)
from dasm2.sequence.conversion import codon_idx_tensor_of_str_ambig
from dasm2.tokenizers import AminoAcidTokenizer

# Module-level singleton: AminoAcidTokenizer is stateless after __init__,
# so a single instance is safe for reuse (each DataLoader worker gets its own
# copy of the module).
_AA_TOKENIZER = AminoAcidTokenizer()


def generic_subs_indicator_tensor_of(
    ambiguous_symbol: str, parent_sequence: str, child_sequence: str
) -> torch.Tensor:
    """Create substitution indicator tensor for generic sequences.

    Args:
        ambiguous_symbol: Character to treat as ambiguous (e.g., 'N', 'X')
        parent_sequence: Parent sequence string
        child_sequence: Child sequence string

    Returns:
        Float tensor where 1.0 indicates substitution, 0.0 indicates no change or ambiguous

    Raises:
        ValueError: If sequences have different lengths
    """
    if len(parent_sequence) != len(child_sequence):
        raise ValueError("Parent and child must have same length")

    indicators = []
    for parent_char, child_char in zip(parent_sequence, child_sequence):
        if parent_char == ambiguous_symbol or child_char == ambiguous_symbol:
            indicators.append(0)  # No substitution for ambiguous
        elif parent_char != child_char:
            indicators.append(1)  # Substitution occurred
        else:
            indicators.append(0)  # No substitution

    return torch.tensor(indicators, dtype=torch.float)


def nt_subs_indicator_tensor_of(parent: str, child: str) -> torch.Tensor:
    """Create nucleotide substitution indicator tensor, ignoring 'N' positions."""
    return generic_subs_indicator_tensor_of("N", parent, child)


def nonsyn_mut_indicator_tensor_of(parent: str, child: str) -> torch.Tensor:
    """Create nonsynonymous mutation indicator tensor from AA sequences.

    Compares amino acid sequences to identify positions where the amino acid
    changed. Positions with 'X' (ambiguous) in either sequence are marked False.

    Args:
        parent: Parent amino acid sequence
        child: Child amino acid sequence

    Returns:
        Float tensor where 1.0 indicates AA change, 0.0 indicates same AA or ambiguous
    """
    return generic_subs_indicator_tensor_of("X", parent, child)


def generic_mask_tensor_of(
    ambiguous_symbol: str, sequence_str: str, length: Optional[int] = None
) -> torch.Tensor:
    """Create mask tensor where non-ambiguous positions are True.

    Args:
        ambiguous_symbol: Character to mask (e.g., 'N', 'X')
        sequence_str: Input sequence string
        length: Optional length to truncate/pad to

    Returns:
        Boolean tensor where True indicates valid (non-ambiguous) positions
    """
    if length is None:
        length = len(sequence_str)

    mask = torch.zeros(length, dtype=torch.bool)
    for i, char in enumerate(sequence_str[:length]):
        if char != ambiguous_symbol:
            mask[i] = True

    return mask


def nt_mask_tensor_of(seq_str: str, length: Optional[int] = None) -> torch.Tensor:
    """Create nucleotide mask tensor, masking 'N' positions."""
    return generic_mask_tensor_of("N", seq_str, length)


def aa_mask_tensor_of(seq_str: str, length: Optional[int] = None) -> torch.Tensor:
    """Create amino acid mask tensor, masking 'X' positions."""
    return generic_mask_tensor_of("X", seq_str, length)


def _consider_codon(codon: str) -> bool:
    """Check if codon should be considered for masking.

    Returns:
        True if codon is valid length (3) and contains no ambiguous nucleotides ('N')
    """
    # Consider codon if it's valid length and doesn't contain ambiguous nucleotides
    return len(codon) == 3 and not any(nt in "N" for nt in codon)


def codon_mask_tensor_of(
    nt_parent: str, *other_nt_seqs: str, aa_length: Optional[int] = None
) -> torch.Tensor:
    """Create codon-level mask based on nucleotide sequences.

    Masks positions where any sequence has invalid/ambiguous codons.

    Args:
        nt_parent: Parent nucleotide sequence
        *other_nt_seqs: Additional nucleotide sequences to check
        aa_length: Optional amino acid length (inferred from parent if None)

    Returns:
        Boolean tensor where True indicates valid codon positions across all sequences
    """
    # Determine AA length
    if aa_length is None:
        aa_length = len(nt_parent) // 3

    mask = torch.ones(aa_length, dtype=torch.bool)

    # Check parent sequence
    for i in range(aa_length):
        codon_start = i * 3
        if codon_start + 3 <= len(nt_parent):
            parent_codon = nt_parent[codon_start : codon_start + 3]
            if not _consider_codon(parent_codon):
                mask[i] = False
        else:
            mask[i] = False

    # Check other sequences
    for other_seq in other_nt_seqs:
        for i in range(aa_length):
            codon_start = i * 3
            if codon_start + 3 <= len(other_seq):
                other_codon = other_seq[codon_start : codon_start + 3]
                if not _consider_codon(other_codon):
                    mask[i] = False
            else:
                mask[i] = False

    return mask


def nt_idx_array_of_str(nt_str: str) -> np.ndarray:
    """Convert nucleotide string to numpy index array."""
    indices = []
    for nt in nt_str:
        if nt in BASES_AND_N_TO_INDEX:
            indices.append(BASES_AND_N_TO_INDEX[nt])
        else:
            indices.append(BASES_AND_N_TO_INDEX["N"])  # Default to N for unknown

    return np.array(indices, dtype=int)


def nt_idx_tensor_of_str(nt_str: str) -> torch.Tensor:
    """Convert nucleotide string to tensor."""
    return torch.from_numpy(nt_idx_array_of_str(nt_str))


def tokenize_single_aa_sequence(aa_seq: str) -> torch.Tensor:
    """Tokenize a single amino acid sequence to a variable-length tensor.

    Unlike tokenize_aa_sequences which pads to a fixed length, this returns
    a tensor of exactly len(aa_seq) with no padding. Used by per-sample
    transforms in TransformDataset.

    Args:
        aa_seq: Amino acid sequence string (may include '^' separators).

    Returns:
        LongTensor of shape [seq_len] with token indices (dtype=torch.long).
    """
    # tokenize() returns [1, seq_len] — squeeze to [seq_len]
    return _AA_TOKENIZER.tokenize([aa_seq]).squeeze(0)


def tokenize_aa_sequences(aa_sequences: List[str], max_aa_seq_len: int) -> torch.Tensor:
    """Convert amino acid sequences to tokenized and padded tensor.

    Args:
        aa_sequences: List of amino acid sequence strings
        max_aa_seq_len: Target maximum sequence length for padding/truncation

    Returns:
        Tokenized tensor of shape (num_sequences, max_aa_seq_len)
    """
    aa_batch = _AA_TOKENIZER.tokenize(aa_sequences)

    # Pad to target max length if needed
    if aa_batch.size(1) < max_aa_seq_len:
        pad_size = max_aa_seq_len - aa_batch.size(1)
        aa_batch = F.pad(
            aa_batch,
            (0, pad_size),
            value=_AA_TOKENIZER.token_to_idx["X"],
        )
    elif aa_batch.size(1) > max_aa_seq_len:
        aa_batch = aa_batch[:, :max_aa_seq_len]

    return aa_batch


def create_codon_tensors(nt_sequences: List[str], max_aa_seq_len: int) -> torch.Tensor:
    """Create codon index tensors from nucleotide sequences.

    Args:
        nt_sequences: List of nucleotide sequence strings
        max_aa_seq_len: Maximum amino acid sequence length for tensor allocation

    Returns:
        Codon index tensor of shape (num_sequences, max_aa_seq_len).
        Padding positions use AMBIGUOUS_CODON_IDX (64) to ensure they are
        properly masked out by codon-based masking logic.
    """
    # Use AMBIGUOUS_CODON_IDX for padding so that padding positions are
    # correctly excluded by masks that check (codon != AMBIGUOUS_CODON_IDX)
    codon_batch = torch.full(
        (len(nt_sequences), max_aa_seq_len),
        AMBIGUOUS_CODON_IDX,
        dtype=torch.long,
    )

    for i, nt_seq in enumerate(nt_sequences):
        codon_idxs = codon_idx_tensor_of_str_ambig(nt_seq)
        codon_batch[i, : len(codon_idxs)] = codon_idxs

    return codon_batch
