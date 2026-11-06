"""Pooled prediction losses — Huber / BCE / cross-entropy variants.

Each ``Loss`` class wraps a stock ``torch.nn.functional`` call. Sum reduction
matches the head-owned compute_loss the pre-#621 ``PooledPredictionHead`` and
its three concrete subclasses (regression, multi-label, classification) used
to call.
"""

from __future__ import annotations

from typing import Dict

import torch.nn.functional as F
from torch import Tensor, nn


def _require(batch: Dict[str, Tensor], key: str, context: str) -> None:
    if key not in batch:
        raise ValueError(
            f"{context} missing required batch key: {key!r}. "
            f"Got keys: {list(batch.keys())}."
        )


class PooledRegressionLoss(nn.Module):
    """Sum-reduced Huber loss against ``delta_affinity`` targets.

    Batch contract: requires ``batch["delta_affinity"]``. Unlike
    ``PooledRegressionHead.forward``, this loss does NOT derive a fallback
    mask from ``aa_parents_idxs``; the head is what shapes the predictions
    before the loss receives them. See ``dasm2/heads/pooled_prediction.py``
    for the head-side mask logic.
    """

    def __init__(self, huber_delta: float = 1.0) -> None:
        super().__init__()
        if huber_delta <= 0.0:
            raise ValueError(f"huber_delta must be > 0, got {huber_delta}")
        self.huber_delta = huber_delta

    def forward(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        _require(batch, "delta_affinity", "PooledRegressionLoss")
        return F.huber_loss(
            predictions,
            batch["delta_affinity"],
            delta=self.huber_delta,
            reduction="sum",
        )


class PooledMultiLabelLoss(nn.Module):
    """Sum-reduced binary-cross-entropy-with-logits against ``binary_target``."""

    def forward(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        _require(batch, "binary_target", "PooledMultiLabelLoss")
        return F.binary_cross_entropy_with_logits(
            predictions,
            batch["binary_target"].to(predictions.dtype),
            reduction="sum",
        )


class PooledClassificationLoss(nn.Module):
    """Sum-reduced cross-entropy against ``class_label`` indices."""

    def forward(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        _require(batch, "class_label", "PooledClassificationLoss")
        return F.cross_entropy(predictions, batch["class_label"], reduction="sum")
