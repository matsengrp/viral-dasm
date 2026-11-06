"""Contrastive Ab-Ag binding losses (issue #362, CALM-inspired).

Three variants share the same ``[B, B]`` scaled-cosine logit matrix:

* ``infonce_row_loss``: row-wise multi-positive InfoNCE (CALM's direction).
* ``siglip_loss``: per-pair sigmoid cross-entropy with a learnable bias
  (Zhai et al. 2023).

After #621 these are dispatched by :class:`ContrastiveBindingLoss`
(the first-class ``nn.Module`` Loss) which the head's matching
``Task.loss`` invokes; pre-#621 they were composed via the deleted
``ContrastiveBindingLossMixin``.
"""

import math
from typing import Dict, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Local variant discriminator for the three contrastive-loss families.
# Distinct from the package-level ``dasm2.losses.types.LossType`` enum
# (which names the loss family); this string ``Literal`` names the
# variant *within* the contrastive family — used by the loss config's
# ``contrastive_loss_type`` field.
ContrastiveLossVariant = Literal["infonce_symmetric", "infonce_ab_to_ag", "siglip"]


def row_normalize_positives(positive_pair_matrix: Tensor) -> Tensor:
    """Row-normalize a 0/1 positive matrix so each row sums to 1.

    Rows with zero positives (should not happen with a well-formed batch, but
    guard anyway) are left zeroed, contributing no loss signal.
    """
    counts = positive_pair_matrix.sum(dim=1, keepdim=True).clamp_min(1.0)
    return positive_pair_matrix.float() / counts


def infonce_row_loss(logits: Tensor, positive_pair_matrix: Tensor) -> Tensor:
    """Row-wise multi-positive InfoNCE: -(1/B) Σ_i Σ_j Y_ij log P_ij.

    ``Y`` is the row-normalized form of ``positive_pair_matrix`` so each row
    contributes equally regardless of how many positives it has. ``logits``
    should already be scaled by ``1/τ``.
    """
    Y = row_normalize_positives(positive_pair_matrix)
    log_probs = F.log_softmax(logits, dim=1)
    per_row = -(Y * log_probs).sum(dim=1)  # [B]
    return per_row.mean()


def siglip_loss(
    logits: Tensor, positive_pair_matrix: Tensor, logit_bias: Tensor
) -> Tensor:
    """Per-pair sigmoid cross-entropy (SigLIP, Zhai et al. 2023).

    Treats every (i, j) pair as an independent binary classification: is this
    a true binding pair? No row-wise softmax, so each pair contributes
    regardless of batch composition. ``logit_bias`` is a learnable scalar
    (initialized to 0) that lets the model calibrate the overall
    positive/negative balance.
    """
    targets = positive_pair_matrix.float()
    return F.binary_cross_entropy_with_logits(
        logits + logit_bias, targets, reduction="mean"
    )


def compute_contrastive_binding_loss(
    logits: Tensor,
    positive_pair_matrix: Tensor,
    loss_type: ContrastiveLossVariant,
    logit_bias: Tensor,
) -> Tensor:
    """Dispatch over contrastive loss variants.

    Returns the *mean* per-pair loss. The head multiplies by batch size to
    convert to a sum-over-pairs before returning, which the trainer then
    divides by the effective batch size to recover the per-pair mean.
    """
    if loss_type == "infonce_symmetric":
        return 0.5 * (
            infonce_row_loss(logits, positive_pair_matrix)
            + infonce_row_loss(logits.T, positive_pair_matrix.T)
        )
    if loss_type == "infonce_ab_to_ag":
        return infonce_row_loss(logits, positive_pair_matrix)
    if loss_type == "siglip":
        return siglip_loss(logits, positive_pair_matrix, logit_bias)
    raise ValueError(
        f"Unknown loss_type: {loss_type!r}. "
        f"Expected 'infonce_symmetric', 'infonce_ab_to_ag', or 'siglip'."
    )


def random_baseline_per_pair_loss(
    loss_type: ContrastiveLossVariant, batch_size: int
) -> float:
    """Theoretical per-pair loss for unit-norm random projections at init.

    For the InfoNCE variants, random normalized logits give approximately
    uniform row softmax, so each valid positive contributes ``-log(1/B)``.
    For SigLIP with ``logit_bias = 0``, each pair's sigmoid is ~0.5 and the
    per-pair BCE is ``-log(0.5) = ln(2)``.

    Empirically the observed initial loss is within a small multiple of the
    standard error of this baseline (depending on τ and projection init).
    """
    if loss_type in ("infonce_symmetric", "infonce_ab_to_ag"):
        return math.log(batch_size)
    if loss_type == "siglip":
        return math.log(2.0)
    raise ValueError(f"Unknown loss_type: {loss_type!r}")


class ContrastiveBindingLoss(nn.Module):
    """Symmetric multi-positive contrastive loss over ``[B, B]`` similarity logits.

    Owns the SigLIP-only learnable ``logit_bias``. The same attribute name
    is kept on the InfoNCE variants as a registered buffer (zero-initialized
    and never updated) so the state-dict *key* spelling is identical across
    variants. Note that the *kind* differs: an InfoNCE checkpoint stores
    ``logit_bias`` as a buffer entry, while a SigLIP checkpoint stores it
    as a parameter — switching ``loss_type`` post-train therefore requires
    manual state-dict surgery (drop or convert the key) before
    ``load_state_dict``. Pre-#621 ``logit_bias`` lived on
    ``ContrastiveBindingHead``.

    Returns ``mean × batch_size`` (sum-over-pairs) and implements
    :class:`~dasm2.losses.protocols.SelfNormalizingLoss` with
    ``n_loss_labels == batch_size``. The trainer therefore divides by the
    total pair count across the gradient-accumulation window — matching
    CLIP/CALM conventions — even though the batch carries the antibody
    padding ``mask`` (whose sum is a residue count, not a pair count, and
    is what the trainer's default would otherwise divide by).

    Scale depends on batch size. A random model scores ``ln(B)`` per pair
    (InfoNCE) for a batch of ``B`` pairs, so the logged value and the
    gradient magnitude are not comparable across batch sizes: train
    (e.g. 32) and validation (e.g. 128) sit at different chance levels,
    and a contrastive task with a small batch contributes a smaller loss
    than one with a large batch. When co-training with other heads, pick
    ``loss_weight`` with this in mind and compare against ``ln(B)``, not
    against zero.

    Learnable-parameter split: ``ContrastiveBindingHead.log_tau`` (the
    learnable temperature on the projection geometry) lives on the head;
    ``logit_bias`` is variant-specific (a per-loss-family calibration
    parameter for SigLIP only) and lives here. Future scalar params that
    parametrize the *projection* should land on the head; those that
    parametrize the *loss-family-specific* objective should land here.
    """

    def __init__(self, loss_type: ContrastiveLossVariant = "infonce_symmetric") -> None:
        super().__init__()
        if loss_type not in ("infonce_symmetric", "infonce_ab_to_ag", "siglip"):
            raise ValueError(
                f"loss_type must be one of 'infonce_symmetric', "
                f"'infonce_ab_to_ag', 'siglip'; got {loss_type!r}"
            )
        self.loss_type: ContrastiveLossVariant = loss_type
        if loss_type == "siglip":
            self.logit_bias = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("logit_bias", torch.zeros(()))

    def n_loss_labels(self, batch: Dict[str, Tensor]) -> int:
        """One label per pair — the trainer's mask-based default would count
        antibody residues (``batch['mask'].sum()``) instead."""
        if "positive_pair_matrix" not in batch:
            raise ValueError(
                "ContrastiveBindingLoss missing required batch key: "
                "'positive_pair_matrix'."
            )
        return int(batch["positive_pair_matrix"].shape[0])

    def forward(
        self,
        predictions: Tensor,
        batch: Dict[str, Tensor],
    ) -> Tensor:
        if "positive_pair_matrix" not in batch:
            raise ValueError(
                "ContrastiveBindingLoss missing required batch key: "
                "'positive_pair_matrix'."
            )
        Y = batch["positive_pair_matrix"]
        if Y.shape != predictions.shape:
            raise ValueError(
                f"positive_pair_matrix shape {tuple(Y.shape)} must match "
                f"logits shape {tuple(predictions.shape)}."
            )
        if predictions.shape[0] < 2:
            raise ValueError(
                f"Contrastive binding loss requires batch_size >= 2, got "
                f"{predictions.shape[0]}. Use drop_last=True in the DataLoader "
                f"or pick a batch size that divides the dataset."
            )
        mean_loss = compute_contrastive_binding_loss(
            logits=predictions,
            positive_pair_matrix=Y,
            loss_type=self.loss_type,
            logit_bias=self.logit_bias,
        )
        return mean_loss * predictions.shape[0]


__all__ = [
    "ContrastiveLossVariant",
    "row_normalize_positives",
    "infonce_row_loss",
    "siglip_loss",
    "compute_contrastive_binding_loss",
    "random_baseline_per_pair_loss",
    "ContrastiveBindingLoss",
]
