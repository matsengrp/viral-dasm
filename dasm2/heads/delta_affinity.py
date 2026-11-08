"""Delta-affinity prediction head for binding affinity changes.

Predicts scalar delta-affinity (change in binding affinity relative to
wildtype) from per-residue sequence embeddings. Unlike evolutionary heads
that output L x 20 per-residue predictions, this head pools over sequence
positions to produce a single scalar per sequence.

Architecture:
    MaskedMeanPooling -> Linear(d_model, 1) -> squeeze

Loss:
    Huber loss on ``delta_affinity`` targets, owned by
    ``DeltaAffinityLoss`` (``dasm2/losses/delta_affinity.py``).
"""

from typing import Dict

from torch import Tensor, nn

from dasm2.blocks.pooling import MaskedMeanPooling

from .base import PredictionHead


class DeltaAffinityHead(PredictionHead):
    """Delta-affinity prediction head.

    Deprecated: superseded by
    ``PooledRegressionHead(pooling_type="mean", num_targets=1)``, which is
    bitwise-equivalent under weight copy (verified by the phase-gate test in
    ``tests/heads/test_pooled_prediction.py``). Kept for checkpoint-config
    compatibility; prefer ``PooledRegressionHead`` for new configs.

    Predicts scalar delta-affinity from per-residue embeddings using
    masked mean pooling followed by a linear projection.

    Architecture:
        1. MaskedMeanPooling: [B, L, D] -> [B, D] (averages over valid positions)
        2. Linear: [B, D] -> [B, 1]
        3. Squeeze: [B, 1] -> [B]

    Loss:
        Huber loss owned by ``DeltaAffinityLoss``.
    """

    def __init__(self, input_dim: int) -> None:
        """Initialize delta-affinity prediction head.

        Args:
            input_dim: Input embedding dimension from sequence encoder.
        """
        super().__init__()
        self.input_dim = input_dim

        self.pooling = MaskedMeanPooling()
        self.prediction_layer = nn.Linear(input_dim, 1)

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Predict delta-affinity from per-residue embeddings.

        Args:
            sequence_embeddings: Per-residue embeddings [batch_size, seq_len, input_dim]
            batch: Batch data containing:
                - loss_mask: Boolean mask [batch_size, seq_len] where True = valid position
                  (excludes padding, ambiguous, and separator tokens)

        Returns:
            Predicted delta-affinity values [batch_size]

        Raises:
            ValueError: If 'loss_mask' not in batch
        """
        if "loss_mask" not in batch:
            raise ValueError(
                "DeltaAffinityHead requires 'loss_mask' in batch: "
                "a BoolTensor of shape [batch_size, seq_len] where True marks "
                "valid (non-padding, non-separator) positions."
            )

        mask = batch["loss_mask"]
        pooled = self.pooling(sequence_embeddings, mask)  # [B, D]
        result: Tensor = self.prediction_layer(pooled).squeeze(-1)  # [B]
        return result

    def to_config(self) -> "DeltaAffinityHeadConfig":
        """Reconstruct DeltaAffinityHeadConfig from head architecture."""
        return DeltaAffinityHeadConfig(
            input_dim=self.input_dim,
        )

    # from_config() inherited from PredictionHead base class


# Set config type after class definition to avoid circular import
from dasm2.config import DeltaAffinityHeadConfig  # noqa: E402

DeltaAffinityHead.head_config_type = DeltaAffinityHeadConfig
