"""AAH (Amino Acid Hybrid) head that takes both AA and NT data.

AAH uses both:
- Codon-level (nucleotide) data via Whichmut loss to anchor the gauge
- Amino-acid-only data via AAO loss for larger-scale training

The gauge anchoring from NT data allows meaningful AAO training.

After #621 Task #4 the head only owns the forward pass — the two loss
objects (``WhichmutLoss`` and ``AAOWhichmutLoss``) are attached to two
separate ``Task`` entries that share this head. Batch-type dispatch is
gone: the trainer routes each batch to the matching ``Task`` (and thus
the matching ``Loss``) directly.
"""

from typing import Any, Dict

import torch.nn as nn
from torch import Tensor

from dasm2.heads.base import SelectionFactorOutputHead
from dasm2.utils.tensor import zap_predictions_along_diagonal


class AAHWhichmutHead(SelectionFactorOutputHead):
    """Amino Acid Hybrid head — produces L×20 selection factors via MLP.

    Two-layer MLP (Linear → ReLU → Linear) with diagonal zapping. AAH
    expansion now happens at task-construction time (one ``Task`` per
    loss type, both pointing at the same head); the per-batch dispatch
    that used to live in ``compute_loss`` is gone.
    """

    def __init__(self, input_dim: int, output_dim: int = 20):
        """Initialize AAHWhichmutHead.

        Args:
            input_dim: Input embedding dimension from encoder
            output_dim: Number of amino acids (default 20)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prediction_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Any]) -> Tensor:
        """Forward pass producing selection factors.

        Args:
            sequence_embeddings: Encoder output [batch_size, seq_len, d_model]
            batch: Batch dict containing aa_parents_idxs for wildtype masking

        Returns:
            Log selection factors [batch_size, seq_len, 20] with wildtype zeroed

        Raises:
            KeyError: If batch missing aa_parents_idxs
        """
        logits = self.prediction_layer(sequence_embeddings)

        # Fail fast if required batch key missing
        if "aa_parents_idxs" not in batch:
            raise KeyError(
                "Batch missing 'aa_parents_idxs' required for wildtype masking. "
                f"Available keys: {list(batch.keys())}"
            )

        # Zero out wildtype (diagonal) positions: f(a->a) = 1 means log(f) = 0
        # Uses utility that handles padding tokens (aa_parents_idxs >= 20)
        return zap_predictions_along_diagonal(
            logits, batch["aa_parents_idxs"], fill=0.0
        )

    def to_config(self) -> "AAHWhichmutHeadConfig":
        """Reconstruct config from head architecture."""
        return AAHWhichmutHeadConfig(input_dim=self.input_dim)

    # from_config() inherited from PredictionHead base class


# Set config type after class definition to avoid circular import
from dasm2.config import AAHWhichmutHeadConfig  # noqa: E402

AAHWhichmutHead.head_config_type = AAHWhichmutHeadConfig
