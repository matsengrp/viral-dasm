"""Shared utilities for loss computation."""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from dasm2.molevol.q_matrix import GLOBAL_KMER_Q_VARIANTS, QVariant
from dasm2.utils.validation import validate_batch_keys

# Batch keys the Q-family (QDASM, Hybrid) codon kernel reads under every
# ``q_variant``. Global k-mer variants also need context windows and segment scaling.
Q_FAMILY_PREDICT_BASE_KEYS = (
    "codon_parents_idxs",
    "codon_rates",
    "codon_csps",
    "branch_length",
)
Q_FAMILY_GLOBAL_KMER_KEYS = (
    "codon_parent_nt_window",
    "codon_parent_boundary_mask",
    "codon_boundary_features",
    "codon_rate_adjustments",
)

# Keys ``forward`` needs on top of the ``log_codon_probs`` set, shared by
# every evolutionary loss.
EVOLUTIONARY_LOSS_ONLY_KEYS = ("codon_children_idxs", "loss_mask")

# ``validate_batch_keys``' dataset hint for the evolutionary losses. Both
# assemblers produce the same keys from the same transform objects; see
# ``dasm2/data/evolutionary_dataset.py::_evolutionary_transforms``.
EVOLUTIONARY_DATASET_HINT = (
    "create_evolutionary_dataset (or, for simulation, _create_sampling_batch)"
)


def q_family_predict_required_keys(q_variant: QVariant) -> Tuple[str, ...]:
    """Batch keys the Q-family ``log_codon_probs`` reads under ``q_variant``.

    ``PREDICT_REQUIRED_KEYS`` is a *static* declaration in the
    :class:`~dasm2.losses.protocols.CodonSamplingLoss` contract, but which
    tensors the Q-matrix kernel reads is ``q_variant``-dependent — so
    ``QDASMLoss`` and ``HybridLoss`` resolve the union once at construction
    rather than at every call. Keeping the global-k-mer variant set here
    mirrors ``QDASMLossConfig.needs_parent_nt_windows``
    (``dasm2/config.py``): the variant→tensor mapping is the loss family's
    business, not its callers'.
    """
    if q_variant in GLOBAL_KMER_Q_VARIANTS:
        return Q_FAMILY_PREDICT_BASE_KEYS + Q_FAMILY_GLOBAL_KMER_KEYS
    return Q_FAMILY_PREDICT_BASE_KEYS


def q_family_global_kmer_inputs(
    batch: Dict[str, Tensor], q_variant: QVariant, context: str
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Read the four global-kmer batch tensors, or return None for other variants.

    Losses and branch-length optimization use the same strict contract so stale
    datasets report every missing field together with the dataset construction
    hint. Non-global variants use their own already-scaled neutral inputs.
    """
    if q_variant not in GLOBAL_KMER_Q_VARIANTS:
        return None, None, None, None
    validate_batch_keys(
        batch, list(Q_FAMILY_GLOBAL_KMER_KEYS), context, EVOLUTIONARY_DATASET_HINT
    )
    return (
        batch["codon_parent_nt_window"],
        batch["codon_parent_boundary_mask"],
        batch["codon_boundary_features"],
        batch["codon_rate_adjustments"],
    )


def focal_nll_loss(log_probs: Tensor, targets: Tensor, gamma: float) -> Tensor:
    """Compute focal loss from log probabilities, summed over all elements.

    Focal loss (Lin et al., 2017) down-weights well-classified examples to focus
    learning on hard cases. Given log probabilities and target indices:

        FL(p_t) = -(1 - p_t)^γ · log(p_t)

    When γ=0, this is equivalent to F.nll_loss with reduction="sum". As γ
    increases, easy examples (high p_t) contribute less to the loss.

    Always returns a sum (not mean) so the trainer can normalize by the total
    loss_mask count across the gradient accumulation window. Callers are
    responsible for masking inputs to valid positions before calling this function.

    Numerical stability: p_t is materialized via exp() and clamped
    (``pt.clamp(max=1.0 - 1e-7)``) before computing ``log1p(-pt)``.
    This is safe because the focal weight ``(1-p_t)^gamma`` only matters
    for easy examples (high p_t) where exp() is accurate; for hard
    examples (low p_t) the focal weight approaches 1.0 regardless.

    Args:
        log_probs: Log probabilities [N, C] (already log-softmaxed)
        targets: Target class indices [N]
        gamma: Focusing parameter. 0.0 = standard NLL, 2.0 = typical focal.

    Returns:
        Scalar focal loss (sum over all elements)
    """
    # Gather log p_t for the correct class: [N]
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Materialize p_t and compute log(1 - p_t) via log1p with clamping.
    pt = log_pt.exp()
    # Clamp to avoid log(0) when pt ≈ 1.0 (perfect prediction)
    log_one_minus_pt = torch.log1p(-pt.clamp(max=1.0 - 1e-7))

    # Focal weight in log space: gamma * log(1 - p_t), then exponentiate
    focal_weight = (gamma * log_one_minus_pt).exp()

    # Loss per element: -(1 - p_t)^gamma * log(p_t), then sum
    loss: Tensor = (-focal_weight * log_pt).sum()
    return loss


def zero_loss_with_grad(device: torch.device) -> Tensor:
    """Return zero loss with gradient tracking enabled.

    Used when a loss function has no valid mutations to compute loss over
    (e.g., all sequences are synonymous). Returns a zero tensor with
    requires_grad=True to maintain gradient flow through the computation graph.

    Args:
        device: Device for the tensor

    Returns:
        Scalar zero tensor with requires_grad=True
    """
    return torch.tensor(0.0, device=device, requires_grad=True)
