"""Masked-language-model head (issue #884, splitting off #851's backbone slice).

Per-position amino-acid reconstruction head for a standalone MLM training
objective: a single ``nn.Linear(input_dim, num_amino_acids)`` projection
producing ``[B, L, 20]`` logits from raw encoder embeddings, following the
same "own its own projection" pattern as ``ContrastiveBindingHead``.
"""

from typing import Dict

from torch import Tensor, nn

from dasm2.sequence.constants import NUM_AMINO_ACIDS

from .base import PredictionHead


class MLMHead(PredictionHead):
    """Per-position amino-acid reconstruction logits from raw sequence embeddings.

    Forward contract:
        ``sequence_embeddings``: ``[B, L, D]`` per-position encoder output.
        ``batch``: unused — this head needs no additional batch fields (the
        masking that produces ``mlm_labels`` happens in
        ``MaskedTransformCollator``, consumed only by ``MLMLoss``).

    Returns ``[B, L, num_amino_acids]`` reconstruction logits.

    Unlike DASM-family heads, ``MLMHead`` subclasses ``PredictionHead``
    directly rather than ``SelectionFactorOutputHead``: despite the
    ``[B, L, 20]`` shape coincidence, MLM logits are reconstruction logits,
    not log-selection-factors, so there is no gauge anchoring (no
    ``zap_predictions_along_diagonal``-style convention — there is no parent
    residue in this objective) and ``OUTPUT_CAPABILITIES`` stays the default
    empty ``frozenset()`` (no ``val_sf_*`` metrics fire for MLM tasks).
    """

    requires_masked_marginal_scoring = True

    def __init__(self, input_dim: int, num_amino_acids: int = NUM_AMINO_ACIDS) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_amino_acids = num_amino_acids
        self.projection = nn.Linear(input_dim, num_amino_acids)

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        result: Tensor = self.projection(sequence_embeddings)
        return result

    def to_config(self) -> "MLMHeadConfig":
        return MLMHeadConfig(
            input_dim=self.input_dim,
            num_amino_acids=self.num_amino_acids,
        )


# Set config type after class definition to avoid circular import
from dasm2.config import MLMHeadConfig  # noqa: E402

MLMHead.head_config_type = MLMHeadConfig
