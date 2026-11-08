"""Codon table operations and genetic code utilities for DASM2.

This module provides constants and functions for working with the genetic code:

64-codon space (all codons including stop codons):
    - CODON_AA_INDICATOR_MATRIX: Shape (64, 20) mapping all codons to amino acids
    - STOP_CODON_INDICATOR: Shape (64,) indicating stop codon positions
    - STOP_CODON_ZAPPER: For zeroing stop codon probabilities in log-space

61-sense-codon space (excluding stop codons):
    - NUM_SENSE_CODONS: Number of sense codons (61)
    - SENSE_CODON_LIST: List of 61 codon indices in 64-space
    - CODON64_TO_SENSE61: Dict mapping 64-space → 61-space (-1 for stops)
    - SENSE_CODON_TO_AA: Shape (61,) tensor mapping sense codon to AA index
    - AA_TO_SENSE_CODONS: Dict mapping AA index → list of sense codon indices
    - SENSE_CODON_AA_INDICATOR: Shape (20, 61) indicator for vectorized marginalization

Sense-codon difference indices (for vectorized rate matrix construction):
    Single-nucleotide differences:
    - SENSE_CODON_DIFF_I/J: Source/target sense codon indices
    - SENSE_CODON_DIFF_POS: Shape (N,) codon position (0/1/2) that differs
    - SENSE_CODON_NT_PAIRS: Shape (N, 2) with [nt_from, nt_to] for each pair
    Multi-nucleotide differences (for multi-hit rate entries):
    - SENSE_CODON_2DIFF_I/J: Source/target indices for 2-nt-diff pairs
    - SENSE_CODON_2DIFF_POS: Shape (M, 2) differing codon positions
    - SENSE_CODON_2DIFF_TARGET_NTS: Shape (M, 2) target nucleotide indices
    - SENSE_CODON_3DIFF_I/J: Source/target indices for 3-nt-diff pairs
    - SENSE_CODON_3DIFF_POS: Shape (K, 3) differing codon positions
    - SENSE_CODON_3DIFF_TARGET_NTS: Shape (K, 3) target nucleotide indices
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from Bio.Data import CodonTable

from dasm2.constants import BIG_VALUE
from dasm2.sequence.constants import (
    AA_STR_SORTED,
    AMBIGUOUS_CODON_IDX,
    BASES,
    CODON_TO_IDX,
    CODONS,
    STOP_CODONS,
)
from dasm2.sequence.conversion import (
    idx_of_codon_allowing_ambiguous,
    iter_codons,
    translate_sequences,
)

# =============================================================================
# Sense codon mappings (61 codons excluding stop codons)
# =============================================================================

# Number of sense codons (excluding stop codons)
NUM_SENSE_CODONS = 64 - len(STOP_CODONS)  # 61

# Number of amino acids
NUM_AA = 20


def _build_sense_codon_indices() -> Tuple[List[int], Dict[int, int]]:
    """Build mapping between 64-codon indices and 61-sense-codon indices.

    Returns:
        Tuple of:
        - sense_codon_list: List of 61 codon indices (in original 0-63 space)
        - codon64_to_sense61: Dict mapping original index to sense index (-1 for stops)
    """
    stop_indices = {CODON_TO_IDX[codon] for codon in STOP_CODONS}
    sense_codon_list = [i for i in range(64) if i not in stop_indices]
    codon64_to_sense61 = {
        codon_idx: sense_idx for sense_idx, codon_idx in enumerate(sense_codon_list)
    }
    # Map stop codons to -1
    for stop_idx in stop_indices:
        codon64_to_sense61[stop_idx] = -1
    return sense_codon_list, codon64_to_sense61


# Pre-computed mappings (constant, computed once at module load)
SENSE_CODON_LIST, CODON64_TO_SENSE61 = _build_sense_codon_indices()

# Stop codon indices in 64-space (TAA=48, TAG=50, TGA=56). Derived from the
# canonical CODON_TO_IDX mapping so the source of truth is one place.
STOP_CODON_64_INDICES: Tuple[int, ...] = tuple(
    sorted(CODON_TO_IDX[codon] for codon in STOP_CODONS)
)


def _build_sense_codon_aa_mapping() -> Tuple[
    torch.Tensor, Dict[int, List[int]], torch.Tensor
]:
    """Build codon-to-amino-acid mapping for sense codons.

    Returns:
        Tuple of:
        - codon_to_aa: Tensor of shape (61,) mapping sense codon index to AA index
        - aa_to_codons: Dict mapping AA index (0-19) to list of sense codon indices
        - aa_indicator: Tensor of shape (20, 61) where aa_indicator[a, c] = 1 if
          codon c encodes amino acid a, 0 otherwise (for vectorized marginalization)
    """
    standard_table = CodonTable.unambiguous_dna_by_id[1]

    codon_to_aa = torch.zeros(NUM_SENSE_CODONS, dtype=torch.long)
    aa_to_codons: Dict[int, List[int]] = {i: [] for i in range(NUM_AA)}
    aa_indicator = torch.zeros((NUM_AA, NUM_SENSE_CODONS), dtype=torch.float32)

    for sense_idx, codon64_idx in enumerate(SENSE_CODON_LIST):
        codon = CODONS[codon64_idx]
        aa = standard_table.forward_table[codon]
        aa_idx = AA_STR_SORTED.index(aa)
        codon_to_aa[sense_idx] = aa_idx
        aa_to_codons[aa_idx].append(sense_idx)
        aa_indicator[aa_idx, sense_idx] = 1.0

    return codon_to_aa, aa_to_codons, aa_indicator


# Pre-computed codon-AA mappings for sense codons
# SENSE_CODON_TO_AA: Shape (61,) tensor mapping sense codon index to AA index
# AA_TO_SENSE_CODONS: Dict mapping AA index (0-19) to list of sense codon indices
# SENSE_CODON_AA_INDICATOR: Shape (20, 61) indicator matrix for vectorized marginalization
SENSE_CODON_TO_AA, AA_TO_SENSE_CODONS, SENSE_CODON_AA_INDICATOR = (
    _build_sense_codon_aa_mapping()
)


def _build_sense_codon_nt_indices() -> torch.Tensor:
    """Build tensor mapping sense codons to their nucleotide indices.

    Returns:
        Tensor of shape (61, 3) where row c contains [nt1, nt2, nt3] indices
        for the three positions of sense codon c.
    """
    nt_indices = torch.zeros((NUM_SENSE_CODONS, 3), dtype=torch.long)
    for sense_idx, codon64_idx in enumerate(SENSE_CODON_LIST):
        codon = CODONS[codon64_idx]
        for pos in range(3):
            nt_indices[sense_idx, pos] = BASES.index(codon[pos])
    return nt_indices


# Pre-computed nucleotide indices for each sense codon: shape (61, 3)
SENSE_CODON_NT_INDICES = _build_sense_codon_nt_indices()


def _build_sense_codon_diff_indices() -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Build precomputed indices for single-nucleotide sense codon differences.

    For efficient rate matrix construction, precompute which sense codon pairs
    differ by exactly one nucleotide, and the nucleotide indices involved.

    Returns:
        Tuple of:
        - codon_i: Tensor of source codon indices (61-space)
        - codon_j: Tensor of target codon indices (61-space)
        - nt_pairs: Tensor of shape (N, 2) with [nt_from, nt_to] indices
        - diff_pos: Tensor of shape (N,) with codon position {0,1,2} that differs
        - target_nt: Tensor of shape (N,) with target nucleotide index
    """
    codon_i_list = []
    codon_j_list = []
    nt_pairs_list = []
    diff_pos_list = []
    target_nt_list = []

    for sense_i, codon64_i in enumerate(SENSE_CODON_LIST):
        codon_i_str = CODONS[codon64_i]
        for sense_j, codon64_j in enumerate(SENSE_CODON_LIST):
            if sense_i == sense_j:
                continue
            codon_j_str = CODONS[codon64_j]

            # Find positions that differ
            diff_positions = [k for k in range(3) if codon_i_str[k] != codon_j_str[k]]

            if len(diff_positions) == 1:
                k = diff_positions[0]
                nt_i = BASES.index(codon_i_str[k])
                nt_j = BASES.index(codon_j_str[k])
                codon_i_list.append(sense_i)
                codon_j_list.append(sense_j)
                nt_pairs_list.append([nt_i, nt_j])
                diff_pos_list.append(k)
                target_nt_list.append(nt_j)

    return (
        torch.tensor(codon_i_list, dtype=torch.long),
        torch.tensor(codon_j_list, dtype=torch.long),
        torch.tensor(nt_pairs_list, dtype=torch.long),
        torch.tensor(diff_pos_list, dtype=torch.long),
        torch.tensor(target_nt_list, dtype=torch.long),
    )


def _build_sense_codon_multidiff_indices() -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build precomputed indices for multi-nucleotide sense codon differences.

    For Q-matrix construction, precompute which sense codon pairs differ by
    exactly 2 or 3 nucleotides, and the positions/nucleotides involved.

    Returns:
        Tuple of:
        - codon_2diff_i: Source codon indices for 2-diff pairs (61-space)
        - codon_2diff_j: Target codon indices for 2-diff pairs (61-space)
        - codon_2diff_pos: Shape (M, 2) positions that differ
        - codon_2diff_target_nts: Shape (M, 2) target nucleotides at those positions
        - codon_3diff_i: Source codon indices for 3-diff pairs (61-space)
        - codon_3diff_j: Target codon indices for 3-diff pairs (61-space)
        - codon_3diff_pos: Shape (K, 3) positions that differ
        - codon_3diff_target_nts: Shape (K, 3) target nucleotides at those positions
    """
    # 2-diff lists
    i2_list: List[int] = []
    j2_list: List[int] = []
    pos2_list: List[List[int]] = []
    nt2_list: List[List[int]] = []

    # 3-diff lists
    i3_list: List[int] = []
    j3_list: List[int] = []
    pos3_list: List[List[int]] = []
    nt3_list: List[List[int]] = []

    for sense_i, codon64_i in enumerate(SENSE_CODON_LIST):
        codon_i_str = CODONS[codon64_i]
        for sense_j, codon64_j in enumerate(SENSE_CODON_LIST):
            if sense_i == sense_j:
                continue
            codon_j_str = CODONS[codon64_j]

            diff_positions = [k for k in range(3) if codon_i_str[k] != codon_j_str[k]]

            if len(diff_positions) == 2:
                i2_list.append(sense_i)
                j2_list.append(sense_j)
                pos2_list.append(diff_positions)
                nt2_list.append([BASES.index(codon_j_str[k]) for k in diff_positions])
            elif len(diff_positions) == 3:
                i3_list.append(sense_i)
                j3_list.append(sense_j)
                pos3_list.append(diff_positions)
                nt3_list.append([BASES.index(codon_j_str[k]) for k in diff_positions])

    return (
        torch.tensor(i2_list, dtype=torch.long),
        torch.tensor(j2_list, dtype=torch.long),
        torch.tensor(pos2_list, dtype=torch.long),
        torch.tensor(nt2_list, dtype=torch.long),
        torch.tensor(i3_list, dtype=torch.long),
        torch.tensor(j3_list, dtype=torch.long),
        torch.tensor(pos3_list, dtype=torch.long),
        torch.tensor(nt3_list, dtype=torch.long),
    )


# Pre-computed single-nucleotide sense codon differences for vectorized rate matrix ops
# SENSE_CODON_DIFF_I: source codon indices (61-space)
# SENSE_CODON_DIFF_J: target codon indices (61-space)
# SENSE_CODON_NT_PAIRS: shape (N, 2) with [nt_from, nt_to] for each pair
# SENSE_CODON_DIFF_POS: shape (N,) codon position {0,1,2} that differs
# SENSE_CODON_DIFF_TARGET_NT: shape (N,) target nucleotide index
(
    SENSE_CODON_DIFF_I,
    SENSE_CODON_DIFF_J,
    SENSE_CODON_NT_PAIRS,
    SENSE_CODON_DIFF_POS,
    SENSE_CODON_DIFF_TARGET_NT,
) = _build_sense_codon_diff_indices()

# Pre-computed multi-nucleotide sense codon differences for Q-matrix construction
# 2-diff: codon pairs differing at exactly 2 positions
# 3-diff: codon pairs differing at all 3 positions
(
    SENSE_CODON_2DIFF_I,
    SENSE_CODON_2DIFF_J,
    SENSE_CODON_2DIFF_POS,
    SENSE_CODON_2DIFF_TARGET_NTS,
    SENSE_CODON_3DIFF_I,
    SENSE_CODON_3DIFF_J,
    SENSE_CODON_3DIFF_POS,
    SENSE_CODON_3DIFF_TARGET_NTS,
) = _build_sense_codon_multidiff_indices()


def single_mutant_aa_indices(codon: str) -> List[int]:
    """Given a codon, return the amino acid indices for all single-mutant neighbors.

    Args:
        codon: A three-letter codon (e.g., "ATG")

    Returns:
        Sorted list of amino acid indices (0-19) for all single nucleotide mutations
    """
    standard_table = CodonTable.unambiguous_dna_by_id[1]  # Standard codon table
    bases = ["A", "C", "G", "T"]

    mutant_aa_indices = set()  # Use a set to avoid duplicates

    # Generate all single-mutant neighbors
    for pos in range(3):  # Codons have 3 positions
        for base in bases:
            if base != codon[pos]:  # Mutate only if it's a different base
                mutant_codon = codon[:pos] + base + codon[pos + 1 :]

                # Check if the mutant codon translates to a valid amino acid
                if mutant_codon in standard_table.forward_table:
                    mutant_aa = standard_table.forward_table[mutant_codon]
                    mutant_aa_indices.add(AA_STR_SORTED.index(mutant_aa))

    return sorted(mutant_aa_indices)


def generate_codon_aa_indicator_matrix() -> np.ndarray:
    """Generate a matrix that maps codons (rows) to amino acids (columns).

    Returns:
        Binary matrix of shape (64, 20) where entry (i,j) is 1 if codon i encodes amino acid j
    """

    matrix = np.zeros((len(CODONS), len(AA_STR_SORTED)))

    for i, codon in enumerate(CODONS):
        try:
            aa = translate_sequences([codon])[0]
        except ValueError:  # Handle STOP codon
            pass
        else:
            # Skip stop codons (represented as '*')
            if aa == "*" or codon in STOP_CODONS:
                pass  # Don't add to matrix (leave as zeros)
            elif aa in AA_STR_SORTED:
                aa_idx = AA_STR_SORTED.index(aa)
                matrix[i, aa_idx] = 1
            # Skip unknown amino acids

    return matrix


CODON_AA_INDICATOR_MATRIX = torch.tensor(
    generate_codon_aa_indicator_matrix(), dtype=torch.float32
)


def build_stop_codon_indicator_tensor() -> torch.Tensor:
    """Return a tensor indicating the stop codons.

    Returns:
        Binary tensor of length 64 where 1.0 indicates stop codon positions
    """
    stop_codon_indicator = torch.zeros(len(CODONS))
    for stop_codon in STOP_CODONS:
        stop_codon_indicator[CODONS.index(stop_codon)] = 1.0
    return stop_codon_indicator


STOP_CODON_INDICATOR = build_stop_codon_indicator_tensor()

# Bool view of STOP_CODON_INDICATOR for cheap "is this codon a stop?" gathers
# on the molevol hot path (avoids per-call torch.tensor(STOP_CODON_64_INDICES)).
STOP_CODON_INDICATOR_BOOL = STOP_CODON_INDICATOR.bool()

SENSE_CODON_MASK = ~STOP_CODON_INDICATOR_BOOL
"""``(64,)`` bool mask, True at the 61 sense codons (``CODON_TO_IDX`` order)."""


def raise_if_stop_codon_parent(parent_codon_idxs: torch.Tensor, caller: str) -> None:
    """Fail loud if any element of ``parent_codon_idxs`` is a stop codon.

    Stop-codon parents would silently wrap to sense-row 60 via PyTorch's
    negative-index semantics on ``CODON64_TO_SENSE61``. ``ComputeMasks``
    filters them upstream; this guard catches contract violations at the
    molevol-layer entry. ``raise`` (not ``assert``) so it survives ``-O``.
    """
    if STOP_CODON_INDICATOR_BOOL.to(parent_codon_idxs.device)[
        parent_codon_idxs.clamp(max=63)
    ].any():
        raise RuntimeError(
            f"Stop-codon parent passed to {caller}; "
            "ComputeMasks should have filtered these"
        )


STOP_CODON_ZAPPER = STOP_CODON_INDICATOR * -BIG_VALUE

# Tensor lookup of CODON64_TO_SENSE61 for the molevol hot path. Stops map to
# -1; the molevol layer raises before this tensor is read with a stop parent.
CODON64_TO_SENSE61_TENSOR = torch.tensor(
    [CODON64_TO_SENSE61[i] for i in range(64)], dtype=torch.long
)

# Tensor view of SENSE_CODON_LIST for the molevol hot path (avoids per-call
# torch.tensor(SENSE_CODON_LIST) inside batched matrix_exp drivers).
SENSE_CODON_LIST_TENSOR = torch.tensor(SENSE_CODON_LIST, dtype=torch.long)

# We build a table that will allow us to look up the amino acid index
# from the codon indices. Argmax gets the aa index.
AA_IDX_FROM_CODON = CODON_AA_INDICATOR_MATRIX.argmax(dim=1).view(4, 4, 4)

#: Amino-acid index for a codon that is a stop, in the STOP-aware lookups below.
#: One past the last real amino acid, so ``aa_idx < STOP_AA_IDX`` is the test for
#: "this codon translates to something a selection factor exists for".
STOP_AA_IDX = len(AA_STR_SORTED)


def _build_codon_to_aa_with_stop() -> np.ndarray:
    """``(64,)`` codon index -> AA index, :data:`STOP_AA_IDX` for the three stops.

    :data:`AA_IDX_FROM_CODON` cannot answer this question. It is an ``argmax``
    over :data:`CODON_AA_INDICATOR_MATRIX`, whose stop-codon rows are all zero,
    so every stop comes back as index 0 -- alanine -- with nothing raising. Code
    that translates arbitrary codons rather than pre-filtered sense codons needs
    a lookup that says "stop" instead of quietly saying "A".

    **The mapping is not re-derived here.** It is
    :func:`get_codon_to_aa_lookup_tensor`'s, moved to a host int8 array, because
    numpy callers indexing by a codon index built from base arithmetic should
    not be reading a second, independently translated table. What this adds is
    the dtype, the host residency, and the guard below.

    The guard pins the one assumption the caller-side base arithmetic makes about
    :data:`CODONS`: that its ordering is ``16 * b0 + 4 * b1 + b2`` over
    :data:`BASES`. A different ordering would relabel every amino acid silently.
    The torch lookup is built from a dict keyed by codon index and so is immune
    to that, which is exactly why the numpy view of it needs the check.
    """
    expected = [f"{a}{b}{c}" for a in BASES for b in BASES for c in BASES]
    if list(CODONS) != expected:
        raise ValueError(
            "CODONS is not in 16*b0 + 4*b1 + b2 order over BASES, so a codon "
            "index built from base arithmetic would not address it. Got "
            f"{list(CODONS)[:4]}..., expected {expected[:4]}..."
        )
    lookup = get_codon_to_aa_lookup_tensor(torch.device("cpu"))
    if int(lookup.max()) != STOP_AA_IDX:
        raise ValueError(
            f"get_codon_to_aa_lookup_tensor's stop sentinel is {int(lookup.max())}, "
            f"not STOP_AA_IDX ({STOP_AA_IDX}). The two have to agree or a stop "
            "reads as a real residue on one side of the codebase."
        )
    return lookup.numpy().astype(np.int8)


def aa_idxs_of_codon_idxs(codon_idx_tensor: torch.Tensor) -> torch.Tensor:
    """Translate an unflattened codon index tensor to amino acid indices.

    Args:
        codon_idx_tensor: Tensor of shape (L, 3) with nucleotide indices per position

    Returns:
        Tensor of shape (L,) with amino acid indices
    """
    # Get the amino acid index for each parent codon.
    return AA_IDX_FROM_CODON[
        (
            codon_idx_tensor[:, 0],
            codon_idx_tensor[:, 1],
            codon_idx_tensor[:, 2],
        )
    ]


def generate_codon_neighbor_matrix() -> torch.Tensor:
    """Generate codon neighbor matrix for efficient single-mutation lookups.

    Returns:
        torch.Tensor: A (65, 20) boolean matrix where entry (i, j) is True if
                     codon i can mutate to amino acid j via single nucleotide substitution.
                     Row 64 (AMBIGUOUS_CODON_IDX) will be all False.
    """
    # Include space for ambiguous codon at index 64
    matrix = np.zeros((AMBIGUOUS_CODON_IDX + 1, len(AA_STR_SORTED)), dtype=bool)

    # Only process the 64 standard codons, not the ambiguous codon
    for i, codon in enumerate(CODONS):
        mutant_aa_indices = single_mutant_aa_indices(codon)
        matrix[i, mutant_aa_indices] = True

    # Row 64 (AMBIGUOUS_CODON_IDX) remains all False

    return torch.tensor(matrix, dtype=torch.bool)


def generate_codon_single_mutation_map() -> dict:
    """Generate mapping of codon-to-codon single mutations.

    Returns:
        Dict[int, List[Tuple[int, int, str]]]: Maps parent codon index to list of
        (child_codon_idx, nt_position, new_base) for all single mutations.
        Only includes valid codons (0-63), not AMBIGUOUS_CODON_IDX (64).
    """
    mutation_map = {}

    # Only process the 64 valid codons, not the ambiguous codon at index 64
    for parent_idx, parent_codon in enumerate(CODONS):
        mutations = []
        for nt_pos in range(3):
            for new_base in ["A", "C", "G", "T"]:
                if new_base != parent_codon[nt_pos]:
                    child_codon = (
                        parent_codon[:nt_pos] + new_base + parent_codon[nt_pos + 1 :]
                    )
                    child_idx = CODONS.index(child_codon)
                    mutations.append((child_idx, nt_pos, new_base))
        mutation_map[parent_idx] = mutations

    return mutation_map


# Global tensors/mappings for efficient lookups
CODON_NEIGHBOR_MATRIX = generate_codon_neighbor_matrix()  # (65, 20)
CODON_SINGLE_MUTATIONS = generate_codon_single_mutation_map()


def _build_functional_codon_single_mutations() -> dict:
    """Build filtered codon mutations excluding transitions to/from stop codons.

    Returns:
        Dict[int, List[Tuple[int, int, str]]]: Maps parent codon index to list of
        (child_codon_idx, nt_position, new_base) for single mutations.
        Excludes both mutations TO stop codons and mutations FROM stop codons.
    """
    stop_codon_idxs = [CODONS.index(codon) for codon in STOP_CODONS]
    return {
        parent_idx: [
            (child_idx, nt_pos, new_base)
            for child_idx, nt_pos, new_base in mutations
            if child_idx not in stop_codon_idxs  # Exclude transitions TO stop codons
        ]
        for parent_idx, mutations in CODON_SINGLE_MUTATIONS.items()
        if parent_idx not in stop_codon_idxs  # Exclude mutations FROM stop codons
    }


FUNCTIONAL_CODON_SINGLE_MUTATIONS = _build_functional_codon_single_mutations()


def _build_aa_idx_from_codon_idx() -> dict:
    """Build flat dictionary mapping codon index to amino acid index.

    Returns:
        Dict[int, int]: Maps codon index (0-63) to amino acid index (0-19).
        Only includes codons that encode amino acids (not stop codons).
    """
    aa_idx_dict = {}
    for codon_idx in range(64):
        if CODON_AA_INDICATOR_MATRIX[codon_idx].sum() > 0:
            aa_idx = CODON_AA_INDICATOR_MATRIX[codon_idx].argmax().item()
            aa_idx_dict[codon_idx] = aa_idx
    return aa_idx_dict


AA_IDX_FROM_CODON_IDX = _build_aa_idx_from_codon_idx()


def _build_codon_min_nt_changes() -> torch.Tensor:
    """Build the minimum-nucleotide-change table from parent codon to each AA.

    For every parent codon (0-63) and every amino acid (0-19), this is the
    minimum Hamming distance (number of differing nucleotide positions) from the
    parent codon to the nearest codon encoding that amino acid::

        table[parent_codon_idx, aa_idx] =
            min(hamming(parent_codon, c) for c in codons where AA(c) == aa)

    Values:
        - 0 for the amino acid the parent codon already encodes (the parent codon
          itself is at Hamming distance 0).
        - 1, 2, or 3 for substitutions, where 3 means every codon of that amino
          acid differs from the parent codon at all three positions
          (e.g. from ``AAA``: → Trp ``TGG``, → Cys ``TGT``/``TGC``).

    Stop codons are never targets — ``AA_IDX_FROM_CODON_IDX`` excludes them — so
    the table only ever measures distance to sense codons. Unlike
    ``CODON_NEIGHBOR_MATRIX``, which encodes only single-nucleotide reachability,
    this table captures the 2- and 3-nt-away distances needed to label the
    multi-hit paths that still carry nonzero neutral probability at
    O(t^2)/O(t^3).

    Returns:
        Tensor of shape (64, 20), dtype long, of minimum Hamming distances.
    """
    table = torch.zeros((64, 20), dtype=torch.long)
    for parent_idx, parent_codon in enumerate(CODONS):
        for aa_idx in range(len(AA_STR_SORTED)):
            table[parent_idx, aa_idx] = min(
                sum(p != c for p, c in zip(parent_codon, CODONS[codon_idx]))
                for codon_idx in range(64)
                if AA_IDX_FROM_CODON_IDX.get(codon_idx) == aa_idx
            )
    return table


CODON_MIN_NT_CHANGES = _build_codon_min_nt_changes()  # (64, 20)


def get_codon_to_aa_lookup_tensor(device: torch.device) -> torch.Tensor:
    """Build tensor for fast codon index → amino acid index lookup.

    Args:
        device: Target device for tensor

    Returns:
        Tensor of shape (64,) mapping codon indices to AA indices (0-19).
        Stop codons map to 20 (invalid AA index).
    """
    codon_to_aa = torch.full((64,), 20, dtype=torch.long, device=device)
    for codon_idx, aa_idx in AA_IDX_FROM_CODON_IDX.items():
        codon_to_aa[codon_idx] = aa_idx
    return codon_to_aa


#: ``(64,)`` int8 codon index -> AA index, with stops at :data:`STOP_AA_IDX`.
CODON_TO_AA_WITH_STOP = _build_codon_to_aa_with_stop()


def _build_codon_class_masks() -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the ``(64, 64)`` ``[parent, destination]`` endpoint-class masks.

    See :data:`SYNONYMOUS_CODON_MASK` and :data:`NONSYNONYMOUS_CODON_MASK`.
    """
    codon_to_aa = get_codon_to_aa_lookup_tensor(torch.device("cpu"))  # stops -> 20
    same_aa = codon_to_aa.unsqueeze(1) == codon_to_aa.unsqueeze(0)
    sense_destination = SENSE_CODON_MASK.unsqueeze(0)
    not_self = ~torch.eye(64, dtype=torch.bool)
    return same_aa & not_self & sense_destination, ~same_aa & sense_destination


SYNONYMOUS_CODON_MASK, NONSYNONYMOUS_CODON_MASK = _build_codon_class_masks()
"""``(64, 64)`` bool masks indexed ``[parent_codon, destination_codon]``.

``SYNONYMOUS_CODON_MASK[p, d]``: ``d`` is a sense codon, ``d != p``, and
``d`` encodes the same amino acid as ``p``. A stop parent's row is all False.

``NONSYNONYMOUS_CODON_MASK[p, d]``: ``d`` is a sense codon encoding a
different amino acid from ``p``. A stop parent's row is True at every sense
codon (a stop encodes no amino acid, so every sense destination differs).

Stop destinations are False in both, and the diagonal is False in both, so
together with the parent's own entry the two masks partition the sense
columns of each sense parent's row. These are the ``syn`` / ``nonsyn``
classes of
:func:`dasm2.molevol.hit_class_conditional.compute_expected_codon_class_probs`.
"""


def get_codon_to_aa_lookup_with_sentinel(device: torch.device) -> torch.Tensor:
    """Codon→AA lookup extended with one sentinel row at ``AMBIGUOUS_CODON_IDX``.

    Validation-metrics accumulators index the lookup by raw batch codon
    indices, which include ``AMBIGUOUS_CODON_IDX`` (= 64) for padded
    positions. The sentinel row maps that index to AA-index 20 — outside
    the 0-19 valid range — so a stray padded position never matches a
    real AA in downstream comparisons but the gather doesn't go
    out-of-bounds. Shared between :class:`CodonPredictionMetrics`
    (``dasm2.training.codon_metrics``) and :class:`SelectionFactorMetrics`
    (``dasm2.training.sf_metrics``).
    """
    from dasm2.sequence.constants import AMBIGUOUS_CODON_IDX

    base = get_codon_to_aa_lookup_tensor(device)
    sentinel = torch.tensor([20], dtype=base.dtype, device=device)
    out = torch.cat([base, sentinel])
    assert out.shape[0] == AMBIGUOUS_CODON_IDX + 1
    return out


def _build_parent_to_children_lookup() -> tuple[torch.Tensor, torch.Tensor]:
    """Build lookup tensors for parent→children codon mutations.

    Returns:
        Tuple of:
        - parent_to_children: (64, max_children) tensor, -1 for padding
        - n_children_per_parent: (64,) tensor with mutation counts
    """
    max_children = max(
        len(mutations) for mutations in FUNCTIONAL_CODON_SINGLE_MUTATIONS.values()
    )
    parent_to_children = torch.full((64, max_children), -1, dtype=torch.long)
    n_children_per_parent = torch.zeros(64, dtype=torch.long)

    for parent_idx, mutations in FUNCTIONAL_CODON_SINGLE_MUTATIONS.items():
        for i, (child_idx, _, _) in enumerate(mutations):
            parent_to_children[parent_idx, i] = child_idx
        n_children_per_parent[parent_idx] = len(mutations)

    return parent_to_children, n_children_per_parent


PARENT_TO_CHILDREN_LOOKUP, N_CHILDREN_PER_PARENT = _build_parent_to_children_lookup()


def encode_codon_mutations(
    nt_parents: pd.Series, nt_children: pd.Series
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert parent/child nucleotide sequences to codon indices and mutation
    indicators.

    Args:
        nt_parents: Parent nucleotide sequences
        nt_children: Child nucleotide sequences

    Returns:
        Tuple of:
        - codon_parents_batch: (N, L_codon) tensor of parent codon indices
        - codon_children_batch: (N, L_codon) tensor of child codon indices
        - codon_mutation_indicators: (N, L_codon) boolean tensor indicating mutation positions

    Example:
        >>> parents = pd.Series(['ATGAAACCC'])
        >>> children = pd.Series(['ATGAAACCG'])  # CCC->CCG mutation
        >>> p_idx, c_idx, mut = encode_codon_mutations(parents, children)
        >>> mut[0]  # tensor([False, False, True])
    """
    _validate_codon_mutation_inputs(nt_parents, nt_children)

    # Convert sequences to lists for processing
    parent_seqs = nt_parents.tolist()
    child_seqs = nt_children.tolist()
    seq_len = len(parent_seqs[0])

    codon_len = seq_len // 3
    n_sequences = len(parent_seqs)

    # Extract all codons at once for vectorized processing
    all_parent_codons = []
    all_child_codons = []

    for parent_seq, child_seq in zip(parent_seqs, child_seqs):
        parent_codons = list(iter_codons(parent_seq))
        child_codons = list(iter_codons(child_seq))
        all_parent_codons.append(parent_codons)
        all_child_codons.append(child_codons)

    # Vectorized codon index lookup
    parent_codon_indices = torch.zeros((n_sequences, codon_len), dtype=torch.long)
    child_codon_indices = torch.zeros((n_sequences, codon_len), dtype=torch.long)
    mutation_indicators = torch.zeros((n_sequences, codon_len), dtype=torch.bool)

    # Process in batches for better cache locality
    for seq_idx in range(n_sequences):
        parent_codons = all_parent_codons[seq_idx]
        child_codons = all_child_codons[seq_idx]

        # Vectorized index lookup using list comprehension (faster than nested loops)
        parent_indices = [
            idx_of_codon_allowing_ambiguous(codon) for codon in parent_codons
        ]
        child_indices = [
            idx_of_codon_allowing_ambiguous(codon) for codon in child_codons
        ]
        mutations = [
            p_codon != c_codon for p_codon, c_codon in zip(parent_codons, child_codons)
        ]

        # Assign to tensors
        parent_codon_indices[seq_idx] = torch.tensor(parent_indices, dtype=torch.long)
        child_codon_indices[seq_idx] = torch.tensor(child_indices, dtype=torch.long)
        mutation_indicators[seq_idx] = torch.tensor(mutations, dtype=torch.bool)

    return parent_codon_indices, child_codon_indices, mutation_indicators


def _validate_codon_mutation_inputs(
    nt_parents: pd.Series, nt_children: pd.Series
) -> None:
    """Validate inputs for codon mutation encoding with comprehensive checks."""
    # Check for empty inputs
    if nt_parents.empty:
        raise ValueError("Parent sequences cannot be empty")
    if nt_children.empty:
        raise ValueError("Child sequences cannot be empty")

    # Check sequence counts match
    if len(nt_parents) != len(nt_children):
        raise ValueError(
            f"Parent count ({len(nt_parents)}) != child count ({len(nt_children)})"
        )

    # Check all sequences have same length
    parent_seqs = nt_parents.tolist()
    child_seqs = nt_children.tolist()

    if not parent_seqs:  # Additional safety check
        raise ValueError("No parent sequences provided")

    expected_len = len(parent_seqs[0])
    if expected_len == 0:
        raise ValueError("Sequences cannot be empty strings")

    # Validate all parent sequences have same length
    for i, seq in enumerate(parent_seqs):
        if len(seq) != expected_len:
            raise ValueError(
                f"Parent sequence {i} length ({len(seq)}) != expected ({expected_len})"
            )

    # Validate all child sequences have same length
    for i, seq in enumerate(child_seqs):
        if len(seq) != expected_len:
            raise ValueError(
                f"Child sequence {i} length ({len(seq)}) != expected ({expected_len})"
            )

    # Check sequences are multiples of 3 (codon requirement)
    if expected_len % 3 != 0:
        raise ValueError(
            f"Sequence length ({expected_len}) must be a multiple of 3 for codon encoding"
        )

    # Validate nucleotide content (basic check for valid DNA/RNA characters)
    valid_nucleotides = set("ACGTUN")  # Include U for RNA and N for ambiguous
    for i, seq in enumerate(parent_seqs):
        invalid_chars = set(seq.upper()) - valid_nucleotides
        if invalid_chars:
            raise ValueError(
                f"Parent sequence {i} contains invalid nucleotides: {invalid_chars}"
            )

    for i, seq in enumerate(child_seqs):
        invalid_chars = set(seq.upper()) - valid_nucleotides
        if invalid_chars:
            raise ValueError(
                f"Child sequence {i} contains invalid nucleotides: {invalid_chars}"
            )
