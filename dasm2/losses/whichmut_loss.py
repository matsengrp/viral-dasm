"""Whichmut loss computation functions.

Ported from netam with DASM2 naming conventions and fail-fast design.

"""

import torch

from dasm2.codon_table import (
    N_CHILDREN_PER_PARENT,
    PARENT_TO_CHILDREN_LOOKUP,
    get_codon_to_aa_lookup_tensor,
)
from dasm2.constants import SMALL_PROBABILITY
from dasm2.losses.loss_utils import zero_loss_with_grad
from dasm2.sequence.constants import AMBIGUOUS_CODON_IDX


def compute_whichmut_loss_batch(
    selection_factors: torch.Tensor,
    neutral_rates: torch.Tensor,
    codon_parents_idxs: torch.Tensor,
    codon_children_idxs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute whichmut loss for a batch of sequences.

    Computes loss using the formula:
    Loss = -sum_k log(p_{j_k, c_k->c'_k}) over all observed mutations k
    where p_{j,c->c'} = λ_{j,c->c'} * f_{j,a->a'} / partition_function_n

    The partition function sums over ALL structurally valid positions (not just
    mutated ones), while loss terms are summed only over mutated positions.
    The ``mask`` parameter should already be narrowed to mutation positions
    (via ``WhichmutDataset.get_loss_mask()``), and the partition function
    computes its own mask from ``codon_parents_idxs``.

    Args:
        selection_factors: Shape (N, L_aa, 20) in log space - model output selection factors
        neutral_rates: Shape (N, L_codon, 64) - codon mutation rates (precomputed λ rates per child codon)
        codon_parents_idxs: Shape (N, L_codon) - parent codon indices
        codon_children_idxs: Shape (N, L_codon) - child codon indices
        mask: Shape (N, L_codon) - **mutation-narrowed** loss mask from dataset.
            Must be True only at positions with actual codon mutations in valid
            (non-ambiguous, non-padding) positions. The dataset's ``get_loss_mask()``
            produces this narrowed mask. Passing a non-narrowed (broad) mask will
            silently compute incorrect loss values — the exact bug this narrowing
            was introduced to prevent. The partition function uses its own broader
            mask derived from ``codon_parents_idxs``.

    Returns:
        Scalar sum-reduced loss for the batch (negative log likelihood summed
        over all observed mutations). The trainer normalizes by total positions.

    Preconditions:
        - All tensors must be on the same device
        - L_aa and L_codon must be equal (1:1 codon-to-AA mapping assumed)
        - Codon indices must be in range [0, 63]
        - selection_factors must be in LOG space (not linear)
        - mask must be narrowed to mutation positions (not the broad structural mask)

    Edge Cases:
        - Returns 0.0 if no mutations are in the mask
        - Mutations to stop codons are filtered out and don't contribute to loss
        - Adds epsilon (1e-10) to probabilities for numerical stability
    """
    N, L_codon = codon_parents_idxs.shape
    _, L_aa, _ = selection_factors.shape
    device = selection_factors.device

    # Convert selection factors from log space to linear space
    # Selection model outputs log(f_{j,a->a'})
    linear_selection_factors = torch.exp(selection_factors)  # (N, L_aa, 20)

    # Partition function mask: all structurally valid positions (not just mutations).
    # All invalid positions (padding, separators, ambiguous codons) have
    # codon_parents_idxs == AMBIGUOUS_CODON_IDX, so this single check suffices.
    partition_mask = codon_parents_idxs != AMBIGUOUS_CODON_IDX

    # 1. Compute partition functions (normalization constants) for each sequence
    partition_functions = compute_per_sequence_partition_functions(
        linear_selection_factors, neutral_rates, codon_parents_idxs, partition_mask
    )  # (N,)

    # 2. Vectorized computation of log probabilities for all mutations
    # mask is already narrowed to mutation positions by the dataset's get_loss_mask()

    # Get indices of all mutations using nonzero()
    mutation_indices = mask.nonzero(
        as_tuple=False
    )  # (num_mutations, 2) -> [seq_idx, codon_pos]

    if mutation_indices.numel() == 0:
        return zero_loss_with_grad(device)

    # Extract sequence and position indices for all mutations
    seq_indices = mutation_indices[:, 0]  # (num_mutations,)
    pos_indices = mutation_indices[:, 1]  # (num_mutations,)

    # Gather child codon indices for all mutations
    child_codon_indices = codon_children_idxs[
        seq_indices, pos_indices
    ]  # (num_mutations,)

    # Get neutral rates for all mutations
    # Dense format: look up rate TO the child codon
    # For efficiency in the vectorized case, we just look up the rate
    # (it will be 0 if not reachable)
    neutral_rates_mut = neutral_rates[
        seq_indices, pos_indices, child_codon_indices
    ]  # (num_mutations,)

    # Convert child codon indices to AA indices using shared lookup
    codon_to_aa = get_codon_to_aa_lookup_tensor(device)
    child_aa_indices = codon_to_aa[child_codon_indices]  # (num_mutations,)

    # Filter out mutations to stop codons (AA index >= 20)
    valid_aa_mask = child_aa_indices < 20

    if not valid_aa_mask.any():
        return zero_loss_with_grad(device)

    # Apply filter to keep only valid AA mutations
    seq_indices = seq_indices[valid_aa_mask]
    pos_indices = pos_indices[valid_aa_mask]
    neutral_rates_mut = neutral_rates_mut[valid_aa_mask]
    child_aa_indices = child_aa_indices[valid_aa_mask]

    # Ensure AA positions are within bounds
    aa_pos_mask = pos_indices < L_aa
    if not aa_pos_mask.any():
        return zero_loss_with_grad(device)

    # Apply AA position filter
    seq_indices = seq_indices[aa_pos_mask]
    pos_indices = pos_indices[aa_pos_mask]
    neutral_rates_mut = neutral_rates_mut[aa_pos_mask]
    child_aa_indices = child_aa_indices[aa_pos_mask]

    # Gather selection factors for all valid mutations
    selection_factors_mut = linear_selection_factors[
        seq_indices, pos_indices, child_aa_indices
    ]  # (num_valid_mutations,)

    # Gather partition functions for all mutations
    partition_function_values = partition_functions[seq_indices].clamp(
        min=SMALL_PROBABILITY
    )  # (num_valid_mutations,)

    # Compute probabilities: p = (λ * f) / partition_function
    probs = (neutral_rates_mut * selection_factors_mut) / partition_function_values

    # Compute log probabilities with numerical stability
    log_probs = torch.log(probs + SMALL_PROBABILITY)

    # Return negative log likelihood
    total_log_likelihood = log_probs.sum()
    return -total_log_likelihood


def predict_whichmut_child_codons(
    selection_factors: torch.Tensor,
    neutral_rates: torch.Tensor,
    codon_parents_idxs: torch.Tensor,
) -> torch.Tensor:
    """Top-1 reachable child codon per position under the whichmut model.

    For each position the whichmut model scores a single-NT child ``c'`` of
    parent ``c`` by ``λ_{c->c'} · exp(SF_{a->a'})`` — the same per-mutation
    factor that :func:`compute_whichmut_loss_batch` sums into its likelihood.
    This returns the ``argmax`` of that score over the parent's reachable
    children, i.e. "given a mutation here, which child codon does the model
    most expect". It is the prediction the conditional codon-accuracy metric
    (``whichmut_codon`` provider, issue #738) compares against the observed
    child. The score is normalization-free, so the per-sequence partition
    function is irrelevant to the argmax.

    The candidate set mirrors the loss partition exactly: ``PARENT_TO_CHILDREN``
    already excludes stop codons (both directions), and positions are further
    gated on a representable child, a nonzero neutral rate, and the parent's
    own child count. Positions with no reachable child (padding / ambiguous
    codons) return ``AMBIGUOUS_CODON_IDX``; callers gate on a valid mask, so
    that sentinel is never read.

    Args:
        selection_factors: ``(N, L_aa, 20)`` log-space selection factors.
        neutral_rates: ``(N, L_codon, 64)`` neutral rate to each child codon.
        codon_parents_idxs: ``(N, L_codon)`` parent codon indices.

    Returns:
        ``(N, L_codon)`` long tensor of predicted child codon indices.
    """
    N, L_codon = codon_parents_idxs.shape
    device = selection_factors.device

    linear_selection_factors = torch.exp(selection_factors)  # (N, L_aa, 20)

    parent_to_children = PARENT_TO_CHILDREN_LOOKUP.to(device)
    n_children_per_parent = N_CHILDREN_PER_PARENT.to(device)
    max_children = parent_to_children.shape[1]

    codon_parents_clamped = torch.clamp(codon_parents_idxs, 0, 63)
    child_codon_indices = parent_to_children[
        codon_parents_clamped
    ]  # (N, L_codon, max_children)
    child_codon_indices_clamped = torch.clamp(child_codon_indices, 0, 63)

    batch_indices = (
        torch.arange(N, device=device).view(N, 1, 1).expand(N, L_codon, max_children)
    )
    pos_indices = (
        torch.arange(L_codon, device=device)
        .view(1, L_codon, 1)
        .expand(N, L_codon, max_children)
    )

    neutral_rates_gathered = neutral_rates[
        batch_indices, pos_indices, child_codon_indices_clamped
    ]

    codon_to_aa = get_codon_to_aa_lookup_tensor(device)
    child_aa_indices = codon_to_aa[child_codon_indices_clamped]
    child_aa_indices_clamped = torch.clamp(child_aa_indices, 0, 19)

    selection_factors_gathered = linear_selection_factors[
        batch_indices, pos_indices, child_aa_indices_clamped
    ]

    scores = neutral_rates_gathered * selection_factors_gathered

    # Candidate mask, identical to the partition function's: a representable
    # child, a non-stop AA, within the parent's child count, and a nonzero rate.
    valid_child_mask = child_codon_indices >= 0
    valid_aa_mask = child_aa_indices < 20
    n_children_per_position = n_children_per_parent[codon_parents_clamped]
    mutation_range = torch.arange(max_children, device=device).view(1, 1, max_children)
    within_range_mask = mutation_range < n_children_per_position.unsqueeze(-1)
    nonzero_rate_mask = neutral_rates_gathered > 0
    candidate_mask = (
        valid_child_mask & valid_aa_mask & within_range_mask & nonzero_rate_mask
    )

    # Candidate scores are strictly positive (rate > 0, exp(SF) > 0), so a
    # negative sentinel sorts every non-candidate below every candidate.
    masked_scores = torch.where(candidate_mask, scores, scores.new_full((), -1.0))
    best_child_pos = masked_scores.argmax(dim=-1)  # (N, L_codon)

    predicted = torch.gather(
        child_codon_indices, 2, best_child_pos.unsqueeze(-1)
    ).squeeze(-1)
    has_candidate = candidate_mask.any(dim=-1)
    return torch.where(
        has_candidate, predicted, predicted.new_full((), AMBIGUOUS_CODON_IDX)
    ).long()


def compute_per_sequence_partition_functions(
    selection_factors: torch.Tensor,
    neutral_rates: torch.Tensor,
    codon_parents_idxs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute partition function (normalization constant) for each sequence.

    Computes partition_function_n = sum over all positions j and possible mutations c->c':
                                    λ_{j,c->c'} * f_{j,a->a'}

    This is the vectorized implementation that achieves high performance
    by avoiding Python loops and using batched tensor operations.

    Args:
        selection_factors: Shape (N, L_aa, 20) - linear space selection factors (not log!)
        neutral_rates: Shape (N, L_codon, 64) - neutral rates TO each child codon
        codon_parents_idxs: Shape (N, L_codon) - parent codon indices
        mask: Shape (N, L_codon) - valid non-ambiguous positions

    Returns:
        Partition functions of shape (N,) - one per sequence
    """
    N, L_codon, _ = neutral_rates.shape
    device = selection_factors.device

    # Move mask to the correct device
    mask = mask.to(device)

    # Get pre-computed lookup tables on correct device
    parent_to_children = PARENT_TO_CHILDREN_LOOKUP.to(device)
    n_children_per_parent = N_CHILDREN_PER_PARENT.to(device)
    max_children = parent_to_children.shape[1]

    # Clamp codon indices to valid range [0, 63] to avoid index errors
    # Ambiguous codons (index 64) will be clamped to 63 but masked out
    codon_parents_idxs_clamped = torch.clamp(codon_parents_idxs, 0, 63)

    # Get child codon indices for all positions: (N, L_codon, max_children)
    # Use advanced indexing to map parent codons to their possible children
    child_codon_indices = parent_to_children[
        codon_parents_idxs_clamped
    ]  # (N, L_codon, max_children)

    # Create batch and position index tensors for advanced indexing
    batch_indices = (
        torch.arange(N, device=device).view(N, 1, 1).expand(N, L_codon, max_children)
    )
    pos_indices = (
        torch.arange(L_codon, device=device)
        .view(1, L_codon, 1)
        .expand(N, L_codon, max_children)
    )

    # Clamp child indices to valid range [0, 63] for safe indexing
    # Invalid indices (-1) will be clamped to 0, but masked out later
    child_codon_indices_clamped = torch.clamp(child_codon_indices, 0, 63)

    # Gather neutral rates for all possible mutations: (N, L_codon, max_children)
    neutral_rates_gathered = neutral_rates[
        batch_indices, pos_indices, child_codon_indices_clamped
    ]

    # Convert child codon indices to AA indices using shared lookup
    codon_to_aa = get_codon_to_aa_lookup_tensor(device)
    child_aa_indices = codon_to_aa[
        child_codon_indices_clamped
    ]  # (N, L_codon, max_children)

    # Clamp AA indices for safe indexing into selection_factors
    child_aa_indices_clamped = torch.clamp(child_aa_indices, 0, 19)

    # Gather selection factors: (N, L_codon, max_children)
    # selection_factors is (N, L_aa, 20), we need to gather the appropriate factors
    # Assuming 1:1 mapping between codon_pos and aa_pos
    selection_factors_gathered = selection_factors[
        batch_indices, pos_indices, child_aa_indices_clamped
    ]

    # Compute products: neutral_rates * selection_factors
    products = (
        neutral_rates_gathered * selection_factors_gathered
    )  # (N, L_codon, max_children)

    # Create masks for valid mutations
    # 1. Mask out padding (where child_codon_indices == -1)
    valid_child_mask = child_codon_indices >= 0

    # 2. Mask out stop codons (where child_aa_indices >= 20)
    valid_aa_mask = child_aa_indices < 20

    # 3. Mask out positions beyond n_children_per_parent for each parent
    # Get n_children for each position's parent codon (use clamped indices)
    n_children_per_position = n_children_per_parent[
        codon_parents_idxs_clamped
    ]  # (N, L_codon)
    mutation_range = torch.arange(max_children, device=device).view(1, 1, max_children)
    within_range_mask = mutation_range < n_children_per_position.unsqueeze(-1)

    # 4. Mask out zero rates
    nonzero_rate_mask = neutral_rates_gathered > 0

    # 5. Mask out ambiguous/invalid positions (expand mask to mutation dimension)
    position_mask = mask.unsqueeze(-1).expand(
        N, L_codon, max_children
    )  # (N, L_codon, max_children)

    # Combine all masks
    combined_mask = (
        valid_child_mask
        & valid_aa_mask
        & within_range_mask
        & nonzero_rate_mask
        & position_mask
    )

    # Apply mask and sum over positions and mutations
    masked_products = products * combined_mask.float()
    partition_functions = masked_products.sum(
        dim=(1, 2)
    )  # Sum over (L_codon, max_children) -> (N,)

    return partition_functions
