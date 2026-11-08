"""Tensor manipulation utilities specific to molecular evolution calculations.

Pure tensor manipulation functions for molecular evolution that operate on tensors
without involving model training or complex business logic. These are the foundation
utilities used by higher-level molecular evolution functions.
"""

import torch
from torch import Tensor

import dasm2.sequence.conversion as sequences

# Constants for numerical stability
LOG_HALF = -0.6931471805599453  # log(0.5)
NEG_INF = float("-inf")

# Maximum log(sum) for gradient clamping in log1mexp.
# The gradient of log1mexp(x) is exp(x)/(1-exp(x)), which explodes as x→0.
# Clamping to -1e-4 ensures gradient ≤ exp(-1e-4)/(1-exp(-1e-4)) ≈ 10000.
# Empirically: -0.01 distorted loss, -1e-4 balances stability and accuracy.
LOG_SUM_CLAMP_MAX = -1e-4


def normalize_rate_matrix(Q: Tensor, equilibrium: Tensor) -> Tensor:
    """Normalize a rate matrix so its mean substitution rate equals 1.

    The scaling factor is -sum_i(Q[i, i] * equilibrium[i]), the expected
    number of substitutions per unit time under equilibrium. After
    normalization, branch lengths (t in P(t) = exp(Qt)) represent expected
    substitutions per site -- the standard convention used by PAML, Pyvolve,
    and most phylogenetics software.

    Args:
        Q: Un-normalized rate matrix (n x n), rows summing to 0.
        equilibrium: Equilibrium frequencies (n,), summing to 1.

    Returns:
        Normalized rate matrix where -sum_i(Q[i, i] * equilibrium[i]) == 1
        (unchanged if the scaling factor is non-positive, e.g. an all-zero Q).
    """
    scaling_factor = -(Q.diag() * equilibrium).sum()
    if scaling_factor > 0:
        return Q / scaling_factor
    return Q


def flatten_codons(array: Tensor) -> Tensor:
    """Reshape a tensor from (..., 4, 4, 4) to (..., 64)."""
    shape = array.shape
    assert shape[-3:] == (4, 4, 4), "Last three dimensions must be (4, 4, 4)"
    return array.reshape(shape[:-3] + (64,))


def unflatten_codons(array: Tensor) -> Tensor:
    """Reshape a tensor from (..., 64) to (..., 4, 4, 4)."""
    shape = array.shape
    assert shape[-1] == 64, "Last dimension must be 64"
    return array.reshape(shape[:-1] + (4, 4, 4))


def reshape_for_codons(array: Tensor) -> Tensor:
    """Reshape a tensor to add a codon dimension by taking groups of 3 sites.

    Args:
    array (torch.Tensor): Original tensor.

    Returns:
    torch.Tensor: Reshaped tensor with an added codon dimension.
    """
    site_count = array.shape[0]
    assert site_count % 3 == 0, "Site count must be a multiple of 3"
    codon_count = site_count // 3
    return array.reshape(codon_count, 3, *array.shape[1:])


def set_parent_codon_prob(codon_probs: Tensor, parent_codon_idxs: Tensor) -> Tensor:
    """Adjust the parent codon probability so that codon probs sum to one at each site.

    Args:
        codon_probs: The codon probabilities in linear space.
            Shape: [B, L, 64]
        parent_codon_idxs (torch.Tensor): The indices of the parent codons.
            Shape: [B, L]
    Returns:
        torch.Tensor: The adjusted codon probabilities.
            Shape: [B, L, 64]
    """
    valid_mask = parent_codon_idxs != sequences.AMBIGUOUS_CODON_IDX  # Shape: [B, L]

    # Zero out the parent indices in codon_probs, while keeping the computation
    # graph intact.
    preds_zeroer = torch.ones_like(codon_probs)
    preds_zeroer[valid_mask, parent_codon_idxs[valid_mask]] = 0.0
    codon_probs = codon_probs * preds_zeroer

    # Calculate the non-parent sum after zeroing out the parent indices.
    non_parent_sum = codon_probs[valid_mask, :].sum(dim=-1)

    # Add these parent values back in, again keeping the computation graph intact.
    preds_parent = torch.zeros_like(codon_probs)
    preds_parent[valid_mask, parent_codon_idxs[valid_mask]] = 1.0 - non_parent_sum
    preds = codon_probs + preds_parent
    # Set ambiguous codons/separators to zero instead of NaN
    # The mask will ensure these positions don't contribute to loss
    # Using zero prevents NaN propagation in gradients
    preds[~valid_mask, :] = 0.0
    return preds


def log1mexp(x: Tensor) -> Tensor:
    """Compute log(1 - exp(x)) in a numerically stable way.

    This is used for computing log(1 - p) when we have log(p).

    For x < log(0.5) ≈ -0.693: use log1p(-exp(x)) which is accurate when exp(x) is small
    For x >= log(0.5): use log(-expm1(x)) which is accurate when exp(x) is close to 1

    Args:
        x: Input tensor with x <= 0 (since we need exp(x) <= 1)

    Returns:
        log(1 - exp(x)) computed stably

    Reference:
        https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    return torch.where(
        x > LOG_HALF,
        torch.log(-torch.expm1(x)),  # More accurate when exp(x) close to 1
        torch.log1p(-torch.exp(x)),  # More accurate when exp(x) close to 0
    )


def mask_ambiguous_codon_rows(
    log_codon_probs: Tensor, parent_codon_idxs: Tensor, *, inplace: bool = False
) -> Tensor:
    """Set every codon at an ambiguous-parent position to ``-inf``.

    A position whose parent codon index is
    ``sequences.AMBIGUOUS_CODON_IDX`` (64) carries no usable neutral row,
    so the whole row is driven to ``-inf`` (probability exactly zero).
    Downstream consumers depend on this specific encoding:
    ``DASMLoss.forward``'s NaN assertion reads such rows only through a
    ``loss_mask`` that excludes them, and
    :func:`dasm2.molevol.simulation.nan_fill_invalid_codon_probs` detects
    an all-zero probability row and replaces it rather than handing it to
    ``Multinomial``.

    Shared by both :func:`compute_log_adjusted_codon_probs` normalizations
    (``parent_absorbing`` via :func:`set_parent_codon_log_prob`,
    ``sense_conditional`` directly): ``log_softmax`` over a garbage row
    returns finite junk rather than ``-inf``, so the conditional kernel
    needs this step just as much as the native one.

    Args:
        log_codon_probs: Log codon probabilities. Shape: ``[B, L, 64]`` or
            ``[L, 64]``. Not modified unless ``inplace``.
        parent_codon_idxs: Parent codon indices. Shape: ``[B, L]`` or ``[L]``.
        inplace: Fill ``log_codon_probs`` itself rather than a copy. Only
            for a tensor the caller has just built and nothing has saved
            for backward, as in :func:`set_parent_codon_log_prob`. The
            conditional path passes ``log_softmax``'s output, which autograd
            saves, and an in-place fill there breaks backward.

    Returns:
        ``log_codon_probs`` with ambiguous-parent rows set to ``-inf``: a
        copy, unless ``inplace`` is set or there are no such rows, in which
        case ``log_codon_probs`` itself.
    """
    valid = parent_codon_idxs != sequences.AMBIGUOUS_CODON_IDX
    if valid.all():
        return log_codon_probs
    if inplace:
        return log_codon_probs.masked_fill_(~valid.unsqueeze(-1), NEG_INF)
    return log_codon_probs.masked_fill(~valid.unsqueeze(-1), NEG_INF)


def set_parent_codon_log_prob(
    log_codon_probs: Tensor, parent_codon_idxs: Tensor, min_log_prob: float = -80.0
) -> Tensor:
    """Set parent codon log probability so probabilities sum to one - LOG SPACE VERSION.

    This function works entirely in log space to avoid gradient explosion issues
    that occur when computing log(small_probability).

    The parent codon probability is: p_parent = 1 - sum(p_other)
    In log space: log(p_parent) = log(1 - exp(logsumexp(log_p_other)))
                                = log1mexp(logsumexp(log_p_other))

    When sum(p_other) >= 1 (logsumexp >= 0), this would give log(<= 0) = -inf
    or NaN. To avoid that and bound the gradient, logsumexp(log_p_other) is
    first clamped to at most ``LOG_SUM_CLAMP_MAX`` (-1e-4), so log(p_parent)
    never falls below log1mexp(-1e-4) ~= -9.21. That is the floor in practice:
    with the default ``min_log_prob`` of -80 the second clamp never binds, and
    it only matters if a caller passes a value above -9.21. Where the floor
    binds, the row sums to slightly more than one. Issue #1267 (PR #1320)
    found it binding at 119 of 147M Compara v10 test sites.

    Uses scatter_ with a [..., 1] index to set exactly one element per row,
    avoiding allocation of [..., 64] boolean masks.

    Args:
        log_codon_probs: Log codon probabilities (unnormalized for parent).
            Shape: [B, L, 64] or [L, 64]
        parent_codon_idxs: The indices of the parent codons.
            Shape: [B, L] or [L]
        min_log_prob: Lower clamp on the parent log probability, applied
            after the ``LOG_SUM_CLAMP_MAX`` clamp. Default -80, which never
            binds, because that clamp already floors the parent at ~-9.21.

    Returns:
        Log codon probabilities with parent set so exp(probs).sum() = 1
            Shape: same as input
    """
    # Clamp index to valid range [0, 63]; ambiguous positions (idx=64) are
    # overwritten to -inf below, so the clamped index value doesn't matter.
    idx = torch.clamp(parent_codon_idxs, max=63).unsqueeze(-1)  # [..., 1]

    # Set parent to -inf to compute logsumexp of non-parent codons only.
    # Under no_grad (BL optimization), reuse one tensor. Under grad
    # (training), use a separate tensor to avoid autograd version conflicts.
    if torch.is_grad_enabled():
        masked = log_codon_probs.clone()
        masked.scatter_(-1, idx, NEG_INF)
    else:
        masked = log_codon_probs.scatter(-1, idx, NEG_INF)

    # Compute logsumexp of non-parent codons
    log_sum_other = torch.logsumexp(masked, dim=-1)

    # Compute log(1 - sum_other) = log1mexp(log_sum_other)
    # Clamp to LOG_SUM_CLAMP_MAX to bound gradients (see constant definition).
    log_sum_other = torch.clamp(log_sum_other, max=LOG_SUM_CLAMP_MAX)
    log_parent_prob = torch.clamp(log1mexp(log_sum_other), min=min_log_prob)

    # Build result: clone original and set parent to computed value.
    # Under no_grad, reuse `masked` (parent position is stale but overwritten).
    if torch.is_grad_enabled():
        result = log_codon_probs.clone()
    else:
        result = masked
    result.scatter_(-1, idx, log_parent_prob.unsqueeze(-1))

    # ``result`` was built above and nothing has saved it, so fill in place
    # rather than allocate another [..., 64] tensor.
    return mask_ambiguous_codon_rows(result, parent_codon_idxs, inplace=True)
