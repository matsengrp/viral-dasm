"""Hybrid prediction head implementation.

Hybrid model that uses exact neutral CTMC (matrix exponential with unit
selection factors) for transition probabilities, then applies learned
selection factors post-hoc.

After #621 Task #4 the head only owns the forward pass (linear → wiggle
→ diagonal zap producing L×20 log selection factors). All Q-matrix
parameters and the hybrid loss live on ``HybridLoss``
(``dasm2/losses/hybrid.py``).
"""

from typing import Dict, Optional

from torch import Tensor, nn

from dasm2.codon_table import NUM_AA

from ._selection_factor_forward import selection_factor_forward
from .base import SelectionFactorOutputHead


class HybridEvolHead(SelectionFactorOutputHead):
    """Hybrid evolutionary prediction head for amino acid selection factors.

    Architecture: linear layer → optional wiggle activation → diagonal
    zapping. Returns L×20 log selection factors. The hybrid loss (exact
    neutral CTMC + post-hoc selection) lives on ``HybridLoss``.
    """

    def __init__(
        self,
        input_dim: int,
        wiggle_beta: Optional[float] = 0.3,
    ) -> None:
        """Initialize Hybrid prediction head.

        Args:
            input_dim: Input embedding dimension from sequence encoder.
            wiggle_beta: Beta parameter for the wiggle activation
                (netam-compatible default: 0.3). Set to ``None`` to skip
                wiggle and pass linear output through directly.
        """
        super().__init__()
        self.input_dim = input_dim
        self.wiggle_beta = wiggle_beta
        self.prediction_layer = nn.Linear(input_dim, NUM_AA)

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Predict amino acid selection factors from sequence embeddings.

        Args:
            sequence_embeddings: [batch_size, seq_len, input_dim]
            batch: Full batch data dict containing aa_parents_idxs for diagonal zapping

        Returns:
            Amino acid selection factors [batch_size, seq_len, 20]
        """
        return selection_factor_forward(
            self.prediction_layer, sequence_embeddings, batch, self.wiggle_beta
        )

    def to_config(self) -> "HybridEvolHeadConfig":
        """Reconstruct HybridEvolHeadConfig from head architecture."""
        return HybridEvolHeadConfig(
            input_dim=self.input_dim,
            wiggle_beta=self.wiggle_beta,
        )

    # from_config() inherited from PredictionHead base class


# Set config type after class definition to avoid circular import
from dasm2.config import HybridEvolHeadConfig  # noqa: E402

HybridEvolHead.head_config_type = HybridEvolHeadConfig
