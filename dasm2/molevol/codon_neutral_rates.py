"""Codon-level neutral rate computation for Whichmut."""

from typing import List, Optional, Tuple

import torch

from dasm2.codon_table import CODON_SINGLE_MUTATIONS
from dasm2.molevol.codon_conversion import codon_idx_to_nt_idxs
from dasm2.sequence.constants import (
    AMBIGUOUS_CODON_IDX,
    BASES,
    CODON_TO_IDX,
    NT_SEPARATOR,
)
from dasm2.utils.tensor import stack_heterogeneous

NUM_CODON_INDICES = 64  # Only standard codons (fail-fast design)
_NUM_MUTATIONS_PER_CODON = 9  # 3 positions × (4 bases - 1 identity) = 9
BASE_TO_IDX = {base: idx for idx, base in enumerate(BASES)}

# Sentinel value for separator codons in pre-computed index lists.
# Using AMBIGUOUS_CODON_IDX (64) ensures any accidental tensor indexing into
# the (64, 9) mutation tables raises IndexError rather than silently returning
# wrong data (PyTorch's -1 would wrap to the last row).
_SEPARATOR_CODON_IDX = AMBIGUOUS_CODON_IDX


def _build_mutation_index_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-compute vectorized index tensors from CODON_SINGLE_MUTATIONS.

    Converts the dict-of-tuples structure into three fixed-size tensors that
    enable vectorized gather+scatter in the inner loop of
    compute_codon_mutation_rates(), eliminating 9 scalar tensor operations per
    codon in favor of 2 vectorized operations.

    Each parent codon has exactly 9 single-nucleotide mutations (3 positions ×
    3 alternative bases), so all tensors have shape (64, 9).

    Returns:
        Tuple of:
        - child_indices: (64, 9) tensor of child codon indices for each parent
        - nt_positions: (64, 9) tensor of nucleotide position within codon (0/1/2)
        - base_indices: (64, 9) tensor of target base index (0-3 via BASE_TO_IDX)
    """
    child_indices = torch.zeros(
        NUM_CODON_INDICES, _NUM_MUTATIONS_PER_CODON, dtype=torch.long
    )
    nt_positions = torch.zeros(
        NUM_CODON_INDICES, _NUM_MUTATIONS_PER_CODON, dtype=torch.long
    )
    base_indices = torch.zeros(
        NUM_CODON_INDICES, _NUM_MUTATIONS_PER_CODON, dtype=torch.long
    )

    for parent_idx, mutations in CODON_SINGLE_MUTATIONS.items():
        for m, (child_idx, nt_pos, new_base) in enumerate(mutations):
            child_indices[parent_idx, m] = child_idx
            nt_positions[parent_idx, m] = nt_pos
            base_indices[parent_idx, m] = BASE_TO_IDX[new_base]

    return child_indices, nt_positions, base_indices


# Pre-computed mutation index tensors for vectorized inner-loop operations.
# Computed once at module load from CODON_SINGLE_MUTATIONS.
_MUTATION_CHILD_INDICES, _MUTATION_NT_POSITIONS, _MUTATION_BASE_INDICES = (
    _build_mutation_index_tensors()
)


def combine_nt_rates_and_csps(
    nt_rates: torch.Tensor,
    nt_csps: torch.Tensor,
) -> torch.Tensor:
    """Combine factored nucleotide rates into per-base mutation rates.

    Takes factored representation (total rate × conditional probs) and produces
    direct per-base mutation rates. This is the first step in converting from
    DASM's nucleotide-level representation to Whichmut's codon-level rates.

    Args:
        nt_rates: Total mutation rate per position, shape (seq_length,)
        nt_csps: Conditional substitution probabilities, shape (seq_length, 4)

    Returns:
        Per-base mutation rates, shape (seq_length, 4), where entry [i, j] is
        the instantaneous rate of mutating position i TO base j
    """
    return nt_rates.unsqueeze(-1) * nt_csps


def _sequences_to_codon_indices(nt_parents: List[str]) -> List[torch.Tensor]:
    """Convert nucleotide sequences to codon index tensors.

    Pre-computing codon indices eliminates repeated string slicing and
    dict lookups in inner loops.

    This is a private helper for compute_codon_mutation_rates_for_dataset().
    Callers should not need to use this directly.

    Related: ``dasm2.analysis.rate_extraction.codon_sequence_to_indices``
    handles single-sequence conversion without separator support but with
    explicit stop-codon validation.

    Separator codons (e.g., "^^^") are encoded as _SEPARATOR_CODON_IDX (64,
    intentionally out-of-range) so that compute_codon_mutation_rates can skip
    them without string operations.

    Args:
        nt_parents: List of nucleotide sequences (no ambiguous bases).
                   Each sequence length must be divisible by 3.
                   May contain separator tokens ("^^^") for multi-segment data.

    Returns:
        List of LongTensors, one per sequence. Each tensor contains
        indices into CODONS (0-63) for each codon position, or
        _SEPARATOR_CODON_IDX (64) for separator codons.

    Raises:
        ValueError: If any sequence contains ambiguous bases (N's)

    Example:
        >>> seqs = ["ATGAAA", "CCCGGG"]
        >>> indices = _sequences_to_codon_indices(seqs)
        >>> len(indices)
        2
        >>> indices[0].shape  # ATG AAA = 2 codons
        torch.Size([2])
    """
    codon_indices_batch = []

    for nt_parent in nt_parents:
        num_codons = len(nt_parent) // 3
        codon_indices = []

        for codon_pos in range(num_codons):
            parent_codon = nt_parent[codon_pos * 3 : codon_pos * 3 + 3]

            if parent_codon == NT_SEPARATOR:
                codon_indices.append(_SEPARATOR_CODON_IDX)
                continue

            if "N" in parent_codon:
                raise ValueError(
                    f"Ambiguous codon '{parent_codon}' in sequence. "
                    f"Sequences with N's should be filtered before calling this function."
                )

            codon_indices.append(CODON_TO_IDX[parent_codon])

        codon_indices_batch.append(torch.tensor(codon_indices, dtype=torch.long))

    return codon_indices_batch


def compute_codon_mutation_rates(
    num_codons: int,
    nt_rates_per_base: torch.Tensor,
    codon_idxs: torch.Tensor,
) -> torch.Tensor:
    """Compute codon-level mutation rates from nucleotide-level rates.

    For each codon position, compute the rate of mutating to each reachable
    child codon via single nucleotide substitution. This aggregates nucleotide
    rates into the codon-level representation needed for Whichmut loss.

    Separator codons (encoded as _SEPARATOR_CODON_IDX) are skipped, leaving
    their rates as zero.

    Fully vectorized: masks valid codons, gathers mutation indices from
    pre-computed tables, and scatters rates in bulk — no Python loop over
    codon positions.

    Args:
        num_codons: Number of codon positions in the sequence.
        nt_rates_per_base: Mutation rate TO each base at each nucleotide
                         position, shape (num_codons * 3, 4).
        codon_idxs: Parent codon index tensor, shape (num_codons,).
                   Values 0-63 for standard codons, _SEPARATOR_CODON_IDX (64)
                   for separators.

    Returns:
        Mutation rate to each child codon, shape (num_codons, 64), where
        entry [j, k] is the rate of codon j mutating to child codon k.
        Separator positions have all-zero rates.
    """
    if codon_idxs.shape[0] != num_codons:
        raise ValueError(
            f"codon_idxs length ({codon_idxs.shape[0]}) doesn't match "
            f"num_codons ({num_codons})"
        )

    codon_rates = torch.zeros(num_codons, NUM_CODON_INDICES)

    valid_mask = codon_idxs != _SEPARATOR_CODON_IDX
    if not valid_mask.any():
        return codon_rates

    valid_positions = torch.where(valid_mask)[0]
    valid_codon_idx = codon_idxs[valid_positions]

    child_idxs = _MUTATION_CHILD_INDICES[valid_codon_idx]  # [num_valid, 9]
    nt_positions = _MUTATION_NT_POSITIONS[valid_codon_idx]  # [num_valid, 9]
    base_idxs = _MUTATION_BASE_INDICES[valid_codon_idx]  # [num_valid, 9]

    seq_positions = valid_positions.unsqueeze(1) * 3 + nt_positions  # [num_valid, 9]
    mutation_rates = nt_rates_per_base[seq_positions, base_idxs]  # [num_valid, 9]

    row_idx = valid_positions.unsqueeze(1).expand_as(child_idxs)  # [num_valid, 9]
    codon_rates[row_idx, child_idxs] = mutation_rates

    return codon_rates


def parent_base_at_offset(
    codon_idxs: torch.Tensor, nt_offsets: torch.Tensor
) -> torch.Tensor:
    """Base index carried at a within-codon offset, read out of the codon index.

    A thin selector over :func:`~dasm2.molevol.codon_conversion.codon_idx_to_nt_idxs`,
    which already decodes a codon index into its three base indices. Spelled as a
    wrapper rather than re-deriving the base-4 arithmetic here: the tree already
    carries two decoders (that one and ``sequence.conversion.unflatten_codon_idxs``),
    and a third would be a third place to get the base ordering wrong.

    Preferred over slicing ``raw_parent`` because it works in every dataset mode:
    the precomputed path has already stripped the raw sequences by the time
    ``codon_neutral_rates`` exists, and reading the codon index guarantees the
    answer agrees with the tables that index is used against.

    Args:
        codon_idxs: Parent codon indices, values 0-63. Any value >= 64
            (separator or ambiguous) yields a meaningless result and must be
            filtered by the caller.
        nt_offsets: Offset within the codon, values 0/1/2. Same shape as
            ``codon_idxs``.

    Returns:
        Base indices into :data:`BASES`, same shape as the inputs.

    Example:
        >>> from dasm2.sequence.constants import CODON_TO_IDX
        >>> idx = torch.tensor([CODON_TO_IDX["ATG"]])
        >>> parent_base_at_offset(idx, torch.tensor([1]))  # 'T' -> 3
        tensor([3])
    """
    all_three = codon_idx_to_nt_idxs(codon_idxs)
    return all_three.gather(-1, nt_offsets.unsqueeze(-1).long()).squeeze(-1)


def censored_codon_entries(
    codon_parents_idxs: torch.Tensor,
    codon_positions: torch.Tensor,
    nt_offsets: torch.Tensor,
    to_base_idxs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Locate the ``codon_neutral_rates`` entries a censor entry removes.

    Zeroing a neutral rate already removes that candidate from the whichmut
    partition function, which masks on ``neutral_rates_gathered > 0``. This
    resolves *which* entries of the ``(L, 64)`` destination-indexed rate tensor a
    censored nucleotide corresponds to, reusing the ``(64, 9)`` candidate tables
    the rate computation itself is built on (#1079).

    Args:
        codon_parents_idxs: Parent codon indices, shape ``(L,)``. Entries >= 64
            must already be filtered out of ``codon_positions``.
        codon_positions: Codon index of each censor entry, shape ``(n,)``.
        nt_offsets: Within-codon offset (0/1/2) of each entry, shape ``(n,)``.
        to_base_idxs: Destination base index of each entry, shape ``(n,)``, for
            **directional** censoring -- one entry removed per censor row, the
            reverse direction untouched. ``None`` censors the whole nucleotide:
            all three candidates differing at ``nt_offsets``, leaving the codon's
            other six neighbours live.

    Returns:
        ``(rows, child_codon_idxs)``, parallel flat tensors naming the entries to
        zero. Whole-site censoring yields ``3n`` pairs, directional yields ``n``.

    Example:
        >>> from dasm2.sequence.constants import CODON_TO_IDX, CODONS
        >>> parents = torch.tensor([CODON_TO_IDX["ATG"], CODON_TO_IDX["AAA"]])
        >>> rows, children = censored_codon_entries(
        ...     parents, torch.tensor([1]), torch.tensor([0])
        ... )
        >>> sorted(CODONS[c] for c in children.tolist())  # AAA, offset 0
        ['CAA', 'GAA', 'TAA']
    """
    parents = codon_parents_idxs[codon_positions]
    nt_positions = _MUTATION_NT_POSITIONS[parents]  # (n, 9)
    child_idxs = _MUTATION_CHILD_INDICES[parents]  # (n, 9)

    hit = nt_positions == nt_offsets.unsqueeze(1)
    if to_base_idxs is not None:
        hit = hit & (_MUTATION_BASE_INDICES[parents] == to_base_idxs.unsqueeze(1))

    rows = codon_positions.unsqueeze(1).expand_as(child_idxs)
    return rows[hit], child_idxs[hit]


def compute_codon_mutation_rates_for_dataset(
    nt_parents: List[str],
    nt_rates_batch: torch.Tensor,
    nt_csps_batch: torch.Tensor,
) -> torch.Tensor:
    """Compute codon-level mutation rates for a batch of sequences.

    Public API for Whichmut rate computation. Converts from DASM's factored
    nucleotide-level representation to Whichmut's direct codon-level rates.

    Handles variable-length sequences (including multi-segment sequences with
    separators) via automatic padding.

    Optimized via vectorized gather+scatter in the inner loop: each codon's
    9 mutations are processed in 2 tensor operations instead of 18 scalar
    read/writes, using pre-computed index tensors. Measured ~3× speedup
    (1967ms → 633ms for 500 seq × 300 nt). Maintains exact numerical
    compatibility with netam.

    Args:
        nt_parents: List of parent nucleotide sequences. May contain separator
                   tokens ("^^^") for multi-segment data.
        nt_rates_batch: Batched mutation rates, shape (batch_size, max_nt_len)
        nt_csps_batch: Batched CSPs, shape (batch_size, max_nt_len, 4)

    Returns:
        Codon mutation rates, shape (num_sequences, max_codons, 64), where
        entry [i, j, k] is the rate of sequence i, codon position j mutating
        TO child codon k. Padded with zeros for variable-length sequences.

    Raises:
        ValueError: If nt_parents is empty or any sequence contains ambiguous bases (N's)
    """
    if not nt_parents:
        raise ValueError(
            "nt_parents cannot be empty. "
            "Dataset construction should ensure non-empty batches."
        )

    # Batch-combine rates × CSPs for all sequences at once: (batch, max_nt_len, 4)
    nt_rates_per_base_batch = nt_rates_batch.unsqueeze(-1) * nt_csps_batch

    # Pre-compute codon indices to avoid string ops in inner loop
    codon_indices_batch = _sequences_to_codon_indices(nt_parents)

    codon_rates_list = []
    for i, (nt_parent, codon_indices) in enumerate(
        zip(nt_parents, codon_indices_batch)
    ):
        actual_len = len(nt_parent)
        nt_rates_per_base = nt_rates_per_base_batch[i, :actual_len, :]
        num_codons = actual_len // 3
        codon_rates = compute_codon_mutation_rates(
            num_codons, nt_rates_per_base, codon_indices
        )
        codon_rates_list.append(codon_rates)

    return stack_heterogeneous(codon_rates_list)
