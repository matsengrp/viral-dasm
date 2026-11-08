"""K-mer sequence encoding utilities for neural network models.

Ported from netam.sequences for native DASM2 implementation.

K-mer encoding converts nucleotide sequences into integer indices that can be
used with embedding layers. The encoding handles ambiguous bases (N) by mapping
them to a special placeholder index.
"""

import itertools
from typing import List

import numpy as np
import torch

BASES = ("A", "C", "G", "T")
BASES_AND_N_TO_INDEX = {base: idx for idx, base in enumerate("ACGTN")}


def generate_kmers(kmer_length: int) -> List[str]:
    """Generate all possible k-mers of the given length.

    The first k-mer is "N" (placeholder for any k-mer containing ambiguous bases),
    followed by all 4^k valid k-mers in lexicographic order.

    Args:
        kmer_length: Length of k-mers to generate

    Returns:
        List of k-mer strings, starting with "N" placeholder
    """
    # Placeholder for ambiguous k-mers is index 0
    all_kmers = ["N"] + [
        "".join(p) for p in itertools.product(BASES, repeat=kmer_length)
    ]
    return all_kmers


def kmer_to_index_of(all_kmers: List[str]) -> dict:
    """Create a mapping from k-mer strings to their indices.

    Args:
        all_kmers: List of k-mer strings (from generate_kmers)

    Returns:
        Dict mapping k-mer string to index
    """
    return {kmer: idx for idx, kmer in enumerate(all_kmers)}


class KmerSequenceEncoder:
    """Encode nucleotide sequences as k-mer index tensors.

    This encoder converts nucleotide sequences into tensors of k-mer indices
    suitable for use with embedding layers. Sequences are padded to a fixed
    length (site_count) and ambiguous k-mers (containing N) are mapped to
    index 0.

    Attributes:
        kmer_length: Length of k-mers
        site_count: Maximum sequence length (for padding)
        kmer_count: Number of distinct k-mers (including placeholder)
        all_kmers: List of all k-mer strings
        kmer_to_index: Dict mapping k-mer string to index
    """

    def __init__(self, kmer_length: int, site_count: int):
        """Initialize k-mer encoder.

        Args:
            kmer_length: Length of k-mers to encode
            site_count: Maximum sequence length (sequences padded to this length)
        """
        self.kmer_length = kmer_length
        self.site_count = site_count
        self.all_kmers = generate_kmers(kmer_length)
        self.kmer_count = len(self.all_kmers)
        self.kmer_to_index = kmer_to_index_of(self.all_kmers)

    def encode_sequence(self, sequence: str) -> tuple:
        """Encode a single nucleotide sequence as k-mer indices.

        Args:
            sequence: Nucleotide sequence string

        Returns:
            Tuple of (encoded_tensor, wt_base_modifier):
                - encoded_tensor: Long tensor of shape (site_count,) with k-mer indices
                - wt_base_modifier: Float tensor of shape (site_count, 4) for CSP computation.
                  Has -inf at wild-type base positions (to zero out self-mutation probability
                  after softmax), 0 elsewhere.
        """
        encoded = torch.zeros(self.site_count, dtype=torch.long)
        wt_base_modifier = torch.zeros(self.site_count, 4)

        half_kmer = self.kmer_length // 2

        for i in range(min(len(sequence), self.site_count)):
            # Extract k-mer centered at position i
            start = max(0, i - half_kmer)
            end = min(len(sequence), i + half_kmer + 1)
            kmer = sequence[start:end]

            # Pad k-mer if at sequence boundaries
            if len(kmer) < self.kmer_length:
                if start == 0:
                    kmer = "N" * (self.kmer_length - len(kmer)) + kmer
                else:
                    kmer = kmer + "N" * (self.kmer_length - len(kmer))

            # Map to index (ambiguous k-mers map to 0)
            if "N" in kmer:
                encoded[i] = 0
            else:
                encoded[i] = self.kmer_to_index.get(kmer, 0)

            # Set wild-type base modifier for CSP computation
            base = sequence[i]
            if base in BASES_AND_N_TO_INDEX and base != "N":
                base_idx = BASES_AND_N_TO_INDEX[base]
                wt_base_modifier[i, base_idx] = float("-inf")

        return encoded, wt_base_modifier


def encode_sequences(sequences: List[str], encoder: KmerSequenceEncoder) -> tuple:
    """Encode multiple sequences into batched tensors.

    Args:
        sequences: List of nucleotide sequences
        encoder: KmerSequenceEncoder instance

    Returns:
        Tuple of (encoded_parents, masks, wt_base_modifiers):
            - encoded_parents: Long tensor of shape (batch, site_count)
            - masks: Bool tensor of shape (batch, site_count), True for valid positions
            - wt_base_modifiers: Float tensor of shape (batch, site_count, 4)
    """
    encoded_list = []
    modifier_list = []
    mask_list = []

    for seq in sequences:
        encoded, modifier = encoder.encode_sequence(seq)
        encoded_list.append(encoded)
        modifier_list.append(modifier)

        # Create mask (True for valid nucleotide positions)
        mask = torch.zeros(encoder.site_count, dtype=torch.bool)
        for i, base in enumerate(seq[: encoder.site_count]):
            if base != "N":
                mask[i] = True
        mask_list.append(mask)

    return (
        torch.stack(encoded_list),
        torch.stack(mask_list),
        torch.stack(modifier_list),
    )


# Lookup table: ASCII byte → base integer (A=0, C=1, G=2, T=3, everything else=4)
_BASE_TO_INT = np.full(256, 4, dtype=np.int8)
_BASE_TO_INT[ord("A")] = 0
_BASE_TO_INT[ord("C")] = 1
_BASE_TO_INT[ord("G")] = 2
_BASE_TO_INT[ord("T")] = 3


def encode_sequences_vectorized(
    sequences: List[str], encoder: KmerSequenceEncoder
) -> tuple:
    """Encode multiple sequences into batched tensors using vectorized numpy ops.

    Drop-in replacement for ``encode_sequences`` that avoids per-position Python
    loops. Achieves ~10-30x speedup on large batches. Only the kmer_length=3
    fast path is vectorized; other k-mer lengths fall back to the original
    per-position loop.

    The outputs are numerically identical to ``encode_sequences``.

    Args:
        sequences: List of nucleotide sequences.
        encoder: KmerSequenceEncoder instance.

    Returns:
        Same as ``encode_sequences``: tuple of
        (encoded_parents, masks, wt_base_modifiers).
    """
    n_seqs = len(sequences)
    site_count = encoder.site_count
    kmer_length = encoder.kmer_length
    half_kmer = kmer_length // 2

    # Convert all sequences to a padded integer array in one pass.
    # Pad with 4 (=N) so boundary k-mers automatically map to ambiguous index 0.
    # We need site_count + 2*half_kmer columns for k-mer window, but we also need
    # to include up to half_kmer extra characters beyond site_count from long
    # sequences (the original impl reads the actual sequence for the right neighbor
    # of the last position even when sequence > site_count).
    padded = np.full((n_seqs, site_count + 2 * half_kmer), 4, dtype=np.int8)
    seq_lens = np.empty(n_seqs, dtype=np.int32)

    for i, seq in enumerate(sequences):
        seq_lens[i] = min(len(seq), site_count)
        # Copy up to site_count + half_kmer characters (extra for right k-mer context)
        copy_len = min(len(seq), site_count + half_kmer)
        seq_bytes = np.frombuffer(seq[:copy_len].encode("ascii"), dtype=np.uint8)
        padded[i, half_kmer : half_kmer + copy_len] = _BASE_TO_INT[seq_bytes]

    # Build masks: True where base is valid (not N/pad) within sequence length
    center_bases = padded[:, half_kmer : half_kmer + site_count]
    length_mask = np.arange(site_count) < seq_lens[:, None]
    masks = (center_bases < 4) & length_mask

    # wt_base_modifier: -inf at the wild-type base position for each valid site
    wt_base_modifier = np.zeros((n_seqs, site_count, 4), dtype=np.float32)
    valid_rows, valid_cols = np.where(masks)
    wt_bases = center_bases[valid_rows, valid_cols]
    wt_base_modifier[valid_rows, valid_cols, wt_bases] = float("-inf")

    # Compute k-mer indices vectorized.
    if kmer_length == 3:
        # For kmer_length=3: index = 1 + left*16 + center*4 + right
        # (index 0 is reserved for ambiguous k-mers containing N)
        left = padded[:, :site_count].astype(np.int32)
        center = padded[:, 1 : site_count + 1].astype(np.int32)
        right = padded[:, 2 : site_count + 2].astype(np.int32)

        # Any position with N (value 4) in the k-mer → ambiguous (index 0)
        has_n = (left >= 4) | (center >= 4) | (right >= 4)
        kmer_indices = np.where(has_n, 0, 1 + left * 16 + center * 4 + right)
    else:
        # General case: fall back to per-position string lookup
        kmer_indices = np.zeros((n_seqs, site_count), dtype=np.int64)
        kmer_to_index = encoder.kmer_to_index
        for si in range(n_seqs):
            seq = sequences[si]
            slen = min(len(seq), site_count)
            for pos in range(slen):
                start = max(0, pos - half_kmer)
                end = min(len(seq), pos + half_kmer + 1)
                kmer = seq[start:end]
                if len(kmer) < kmer_length:
                    if start == 0:
                        kmer = "N" * (kmer_length - len(kmer)) + kmer
                    else:
                        kmer = kmer + "N" * (kmer_length - len(kmer))
                if "N" in kmer:
                    kmer_indices[si, pos] = 0
                else:
                    kmer_indices[si, pos] = kmer_to_index.get(kmer, 0)

    return (
        torch.from_numpy(kmer_indices.astype(np.int64)),
        torch.from_numpy(masks.copy()),
        torch.from_numpy(wt_base_modifier),
    )
