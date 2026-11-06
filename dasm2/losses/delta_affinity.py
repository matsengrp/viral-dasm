"""Delta-affinity Huber loss — first-class ``nn.Module`` Loss class.

Mirrors the pre-#621 ``DeltaAffinityLossMixin`` (``dasm2/heads/mixins.py``).
"""

from __future__ import annotations

from typing import Dict

import torch.nn.functional as F
from torch import Tensor, nn


class DeltaAffinityLoss(nn.Module):
    """Sum-reduced Huber loss between predicted scalar and ``delta_affinity``."""

    def __init__(self, huber_delta: float = 1.0) -> None:
        super().__init__()
        if huber_delta <= 0.0:
            raise ValueError(f"huber_delta must be > 0, got {huber_delta}")
        self.huber_delta = huber_delta

    def forward(
        self,
        predictions: Tensor,
        batch: Dict[str, Tensor],
    ) -> Tensor:
        if "delta_affinity" not in batch:
            raise ValueError(
                "DeltaAffinityLoss missing required batch key: 'delta_affinity'. "
                f"Got keys: {list(batch.keys())}."
            )
        return F.huber_loss(
            predictions,
            batch["delta_affinity"],
            delta=self.huber_delta,
            reduction="sum",
        )
