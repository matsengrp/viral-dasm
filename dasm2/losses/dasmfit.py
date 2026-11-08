"""DASMfit residual Poisson loss — first-class ``nn.Module`` Loss class.

Absorbs the loss math from ``DASMFitResidualHead.compute_loss``. The head is
forward-only after #621.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn


class DASMFitResidualLoss(nn.Module):
    """Negated Poisson log-likelihood Σ [M·exp(λ) − D·λ] with λ = Σ_i α_i·r_i.

    Reads ``D`` (observed counts, "data") and ``M`` (expected counts under the
    base per-site Q_j, "model") from the batch, and α from either the batch
    (written by ``DASMFitAlphaOptimizer.pre_cycle``) or defaults to zeros (the
    α=0 base model).
    """

    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self.k = k

    def forward(
        self,
        predictions: Tensor,
        batch: Dict[str, Tensor],
    ) -> Tensor:
        for key in ("D", "M"):
            if key not in batch:
                raise ValueError(
                    f"DASMFitResidualLoss missing required batch key: {key!r}. "
                    f"Got keys: {list(batch.keys())}."
                )
        d_counts = batch["D"]
        m_counts = batch["M"]

        if "alpha" in batch:
            alpha = batch["alpha"]
        else:
            batch_size = predictions.shape[0]
            alpha = torch.zeros(
                batch_size, self.k, device=predictions.device, dtype=predictions.dtype
            )

        alpha_expanded = alpha[:, None, :, None]
        lam = (alpha_expanded * predictions).sum(dim=2)
        loss: Tensor = (m_counts * torch.exp(lam) - d_counts * lam).sum()
        return loss
