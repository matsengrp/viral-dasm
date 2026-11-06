"""Masked-language-model loss (issue #884, splitting off #851's backbone slice).

Cross-entropy over collator-masked positions only, via
``ignore_index=MLM_IGNORE_INDEX``. ``batch["mlm_labels"]`` carries the true
residue id at every masked position (chosen by ``MaskedTransformCollator``)
and the ignore-index sentinel everywhere else, so ``ignore_index`` does the
position-selection that a separate boolean mask would otherwise need.
"""

from typing import Dict

import torch.nn.functional as F
from torch import Tensor, nn

from dasm2.sequence.mlm_constants import MLM_IGNORE_INDEX


class MLMLoss(nn.Module):
    """Sum-reduced cross-entropy over masked positions.

    Implements the ``SelfNormalizingLoss`` protocol
    (``dasm2/losses/protocols.py``) via ``n_loss_labels``, so the trainer
    divides this sum-loss by exactly the count of masked positions —
    giving a clean mean-cross-entropy-per-masked-position regardless of how
    sparsely or densely a given batch happens to be masked.
    """

    def forward(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        if "mlm_labels" not in batch:
            raise ValueError("MLMLoss missing required batch key: 'mlm_labels'.")
        vocab_size = predictions.shape[-1]
        return F.cross_entropy(
            predictions.reshape(-1, vocab_size),
            batch["mlm_labels"].reshape(-1),
            ignore_index=MLM_IGNORE_INDEX,
            reduction="sum",
        )

    def n_loss_labels(self, batch: Dict[str, Tensor]) -> int:
        return int((batch["mlm_labels"] != MLM_IGNORE_INDEX).sum().item())


__all__ = ["MLMLoss"]
