"""Structural guidance head for HERMES alignment.

After #621 Task #4 the head only owns the (L2-normalize) forward pass.
The site-wise cosine-similarity loss lives on
``StructuralGuidanceLoss``.
"""

from typing import Dict

import torch.nn.functional as F
from torch import Tensor

from dasm2.heads.base import PredictionHead
from dasm2.losses.structural_guidance import HERMES_EMBED_DIM


class StructuralGuidanceHead(PredictionHead):
    """Head for aligning sequence embeddings with HERMES structural embeddings.

    L2-normalizes encoder outputs (no learned parameters). The
    structural-guidance loss lives on ``StructuralGuidanceLoss``.

    **CRITICAL REQUIREMENT**: Applies L2-normalization to encoder outputs before
    returning them. This ensures unit-norm embeddings as required by structural
    guidance loss for correct cosine similarity computation.
    """

    def __init__(self, input_dim: int) -> None:
        """Initialize structural guidance head.

        Args:
            input_dim: Input embedding dimension from encoder (must match HERMES_EMBED_DIM)

        Raises:
            ValueError: If input_dim doesn't match HERMES embedding dimension
        """
        super().__init__()

        # Validate dimension matches HERMES (fail-fast)
        if input_dim != HERMES_EMBED_DIM:
            raise ValueError(
                f"StructuralGuidanceHead requires input_dim={HERMES_EMBED_DIM} to match HERMES embeddings. "
                f"Got input_dim={input_dim}. Adjust encoder d_model to {HERMES_EMBED_DIM}."
            )

        self.input_dim = input_dim

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Return L2-normalized encoder outputs.

        Args:
            sequence_embeddings: Sequence embeddings from encoder (B x L x D)
            batch: Full batch data dict (unused for pass-through)

        Returns:
            L2-normalized embeddings (B x L x D)

        Note:
            Applies L2-normalization to ensure embeddings have unit norm as required
            by structural guidance loss. Without normalization, cosine similarity
            computation would be incorrect.
        """
        return F.normalize(sequence_embeddings, dim=-1)

    def to_config(self) -> "StructuralGuidanceHeadConfig":
        """Reconstruct StructuralGuidanceHeadConfig from head architecture."""
        return StructuralGuidanceHeadConfig(input_dim=self.input_dim)

    # from_config() inherited from PredictionHead base class


# Set config type after class definition to avoid circular import
from dasm2.config import StructuralGuidanceHeadConfig  # noqa: E402

StructuralGuidanceHead.head_config_type = StructuralGuidanceHeadConfig
