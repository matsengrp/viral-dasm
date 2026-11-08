"""Binding classification loss (issue #801).

Sum-reduced BCE-with-logits against an explicit binary label, for the
cross-attention ``BindingClassificationHead``. Nearly identical to
``PooledMultiLabelLoss`` (``dasm2/losses/pooled.py``), plus a configurable
``pos_weight`` (a plain config float — the loss module does not reach into
the dataset to compute label balance; that is a training-script concern).
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dasm2.data.delta_affinity import BINARY_TARGET_KEY


def _require(batch: Dict[str, Tensor], key: str, context: str) -> None:
    if key not in batch:
        raise ValueError(
            f"{context} missing required batch key: {key!r}. "
            f"Got keys: {list(batch.keys())}."
        )


class BindingClassificationLoss(nn.Module):
    """Sum-reduced BCE-with-logits against an explicit binary label.

    Implements :class:`~dasm2.losses.protocols.SelfNormalizingLoss`: this
    loss sums one BCE term per *pair* (``predictions.shape[0]``), but
    ``BindingClassificationHead``'s batch contract mandates a per-residue
    ``loss_mask``/``mask`` for entity-A masking. Without overriding
    ``n_loss_labels``, the trainer's default ``_get_n_loss_positions``
    would find that mask in the (shared) batch dict and normalize by its
    per-residue count instead of the per-pair count — the same trap
    ``ContrastiveBindingLoss`` avoids by asserting ``loss_mask`` is absent
    from its batch. This loss can't make that assumption (the mask is
    mandatory here), so it self-reports the correct label unit instead.
    """

    def __init__(self, pos_weight: Optional[float] = None) -> None:
        super().__init__()
        if pos_weight is not None and pos_weight <= 0.0:
            raise ValueError(f"pos_weight must be > 0, got {pos_weight}")
        self.pos_weight = pos_weight
        self.pos_weight_tensor: Optional[Tensor]
        if pos_weight is not None:
            self.register_buffer("pos_weight_tensor", torch.tensor(pos_weight))
        else:
            self.pos_weight_tensor = None

    def forward(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        _require(batch, BINARY_TARGET_KEY, "BindingClassificationLoss")
        return F.binary_cross_entropy_with_logits(
            predictions,
            batch[BINARY_TARGET_KEY].to(predictions.dtype),
            pos_weight=self.pos_weight_tensor,
            reduction="sum",
        )

    def n_loss_labels(self, batch: Dict[str, Tensor]) -> int:
        """One label per pair — see class docstring for why this overrides
        the trainer's mask-based default."""
        _require(batch, BINARY_TARGET_KEY, "BindingClassificationLoss")
        return int(batch[BINARY_TARGET_KEY].shape[0])


__all__ = ["BindingClassificationLoss"]
