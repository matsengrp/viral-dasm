"""AAO Whichmut loss computation."""

import torch
from torch import Tensor

from dasm2.constants import SMALL_PROBABILITY
from dasm2.losses.loss_utils import zero_loss_with_grad


def compute_aao_whichmut_loss_batch(
    selection_factors: Tensor,
    aao_neutral_rates: Tensor,
    aa_children_idxs: Tensor,
    mask: Tensor,
) -> Tensor:
    """Compute AAO Whichmut loss.

    For each observed AA substitution at position j from parent a to child a':
        p_{j, a→a'} = λ_{j, a→a'} × f_{j, a→a'} / Z_j
        Z_j = Σ_{a''} λ_{j, a→a''} × f_{j, a→a''}

    Loss = -Σ log(p) over all observed AA changes

    The per-position partition function Z_j sums over all 20 AAs at each
    position and does not depend on a site mask, so narrowing ``mask`` to
    mutation-only positions does not affect the partition function.

    Args:
        selection_factors: Log selection factors log(f_{j,a→a'}), shape (N, L, 20).
            Diagonal (a→a) should be zapped to 0 (so f_{a→a} = 1 in linear space).
        aao_neutral_rates: Marginalized neutral rates λ_{j, a→a'}, shape (N, L, 20).
            Entry [i, j, a'] is total rate of position j mutating to AA a'.
            The parent AA identity is already encoded in these rates via marginalization.
        aa_children_idxs: Child AA indices, shape (N, L). Values in [0, 19].
        mask: **Mutation-narrowed** loss mask from dataset, shape (N, L).
            Must be True only at positions with actual AA changes in valid
            positions. The dataset's ``get_loss_mask()`` produces this narrowed
            mask. Passing a non-narrowed (broad) mask will silently compute
            incorrect loss values. The per-position partition function Z_j
            sums over all 20 AAs and does not depend on a site mask.

    Returns:
        Scalar sum-reduced loss (negative log likelihood summed over all
        observed AA changes). The trainer normalizes by total positions.

    Note:
        Unlike compute_whichmut_loss_batch, this function does not require aa_parents_idxs
        because the parent AA information is already encoded in the marginalized rates.
        The diagonal term λ_{a→a} × f_{a→a} = λ_{a→a} × 1 in Z provides the gauge.
    """
    device = selection_factors.device

    # Convert log selection factors to linear space (diagonal zapped to 0 → exp(0) = 1)
    linear_selection_factors = torch.exp(selection_factors)  # (N, L, 20)

    # Z[i,j] = Σ_{a'} λ[i,j,a'] × f[i,j,a'] (includes synonymous rate × gauge factor)
    partition_functions = (aao_neutral_rates * linear_selection_factors).sum(
        dim=-1
    )  # (N, L)

    # mask is already narrowed to AA mutation positions by the dataset's get_loss_mask()
    mutation_indices = mask.nonzero(as_tuple=False)  # (num_mutations, 2)

    if mutation_indices.numel() == 0:
        return zero_loss_with_grad(device)

    seq_indices = mutation_indices[:, 0]
    pos_indices = mutation_indices[:, 1]
    child_aa = aa_children_idxs[seq_indices, pos_indices]

    # Filter out invalid AA indices (>= 20: ambiguous or stop codons)
    valid_aa_mask = child_aa < 20
    seq_indices = seq_indices[valid_aa_mask]
    pos_indices = pos_indices[valid_aa_mask]
    child_aa = child_aa[valid_aa_mask]

    if not valid_aa_mask.any():
        return zero_loss_with_grad(device)

    # Gather values for observed mutations
    neutral_rates_mut = aao_neutral_rates[seq_indices, pos_indices, child_aa]
    selection_factors_mut = linear_selection_factors[seq_indices, pos_indices, child_aa]
    Z_mut = partition_functions[seq_indices, pos_indices].clamp(min=SMALL_PROBABILITY)

    # p = (λ × f) / Z
    probs = (neutral_rates_mut * selection_factors_mut) / Z_mut

    # Compute log probabilities with numerical stability (matches original whichmut_loss.py)
    log_probs = torch.log(probs + SMALL_PROBABILITY)

    return -log_probs.sum()
