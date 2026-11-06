"""QDASM prediction head implementation.

Q-matrix variant of DASM that uses exact codon transition probabilities
via matrix exponential exp(tQ) rather than the Poisson approximation.

After #621 Task #4 the head only owns the forward pass (linear → wiggle
→ diagonal zap producing L×20 log selection factors). All Q-matrix
parameters and the matrix-exp loss live on ``QDASMLoss``
(``dasm2/losses/qdasm.py``).
"""

from typing import Dict, Optional

from torch import Tensor, nn

from dasm2.codon_table import NUM_AA

from ._selection_factor_forward import selection_factor_forward
from .base import SelectionFactorOutputHead


class QDASMEvolHead(SelectionFactorOutputHead):
    """QDASM evolutionary prediction head for amino acid selection factors.

    Architecture: linear layer → optional wiggle activation → diagonal
    zapping. Returns L×20 log selection factors. Q-matrix construction
    and the matrix-exponential codon-level loss live on ``QDASMLoss``.
    """

    def __init__(
        self,
        input_dim: int,
        wiggle_beta: Optional[float] = 0.3,
        selection_log_floor: Optional[float] = -6.0,
    ) -> None:
        """Initialize QDASM prediction head.

        Args:
            input_dim: Input embedding dimension from sequence encoder.
            wiggle_beta: Beta parameter for the wiggle activation
                (netam-compatible default: 0.3). Set to ``None`` to skip
                wiggle and pass linear output through directly.
            selection_log_floor: Smooth lower bound on the log selection
                factors (default -6.0 ⇒ f ≥ e^-6 ≈ 2.5e-3). ``wiggle`` is
                unbounded below, and QDASM's ratio-form Q uses ``1/f_source``,
                so an un-floored source factor near zero makes ``||tQ||`` blow
                up (silently corrupting exp(tQ)). The floor caps it while
                keeping gradients alive (softplus). Set to ``None`` to disable
                (reproduces the unbounded pre-fix behavior, for ablation only).
        """
        super().__init__()
        self.input_dim = input_dim
        self.wiggle_beta = wiggle_beta
        self.selection_log_floor = selection_log_floor
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
            self.prediction_layer,
            sequence_embeddings,
            batch,
            self.wiggle_beta,
            log_floor=self.selection_log_floor,
        )

    def to_config(self) -> "QDASMEvolHeadConfig":
        """Reconstruct QDASMEvolHeadConfig from head architecture."""
        return QDASMEvolHeadConfig(
            input_dim=self.input_dim,
            wiggle_beta=self.wiggle_beta,
            selection_log_floor=self.selection_log_floor,
        )

    # from_config() inherited from PredictionHead base class


# Set config type after class definition to avoid circular import
from dasm2.config import QDASMEvolHeadConfig  # noqa: E402

QDASMEvolHead.head_config_type = QDASMEvolHeadConfig
