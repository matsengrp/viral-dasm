"""DMS functional-score loss with learnable per-loss scale parameter.

Pre-#621 the ``scale`` ``nn.Parameter`` lived on ``DMSFunctionalScoreHead``.
After #621 it lives here so the head is forward-only and the trainer can
register loss parameters in ``model.task_losses`` for optimizer pickup.
The parameter's optimizer-visible value is unchanged; only its state-dict
key path changes (``prediction_heads.dms_functional_score_head.scale``
becomes ``task_losses.dms_functional_score_head.scale``).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DMSFunctionalScoreLoss(nn.Module):
    """Huber loss between scaled predicted SF differences and DMS measurements.

    Relies on the head's diagonal zapping (``log f(a→a) = 0``) so the
    mutant selection factor directly represents the WT-relative effect.
    The learnable ``scale`` accounts for per-dataset normalization
    conventions; ``huber_delta`` controls the Huber crossover.
    """

    def __init__(self, scale_init: float = 1.0, huber_delta: float = 1.0) -> None:
        super().__init__()
        if huber_delta <= 0.0:
            raise ValueError(f"huber_delta must be > 0, got {huber_delta}")
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.huber_delta = huber_delta

    def forward(
        self,
        predictions: Tensor,
        batch: Dict[str, Tensor],
    ) -> Tensor:
        for key in ("dms_position", "dms_mut_aa_idx", "dms_func_score"):
            if key not in batch:
                raise ValueError(
                    f"DMSFunctionalScoreLoss missing required batch key: {key!r}. "
                    f"Got keys: {list(batch.keys())}."
                )
        positions = batch["dms_position"]
        mut_aa_idxs = batch["dms_mut_aa_idx"]
        func_scores = batch["dms_func_score"]

        batch_size = predictions.size(0)
        batch_indices = torch.arange(batch_size, device=predictions.device)
        log_sel_mut = predictions[batch_indices, positions, mut_aa_idxs]

        predicted_effects = self.scale * log_sel_mut

        return F.huber_loss(
            predicted_effects, func_scores, delta=self.huber_delta, reduction="sum"
        )
