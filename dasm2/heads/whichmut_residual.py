"""Whichmut residual prediction head implementation.

Produces output = zap(frozen_structural_logits + delta), where delta is learned
by the encoder pathway. When use_residual=True, the prediction_layer (which
produces delta) is zero-initialized so the model reproduces the structural
prior at initialization.
"""

from typing import Dict

from torch import Tensor, nn

from dasm2.utils.tensor import zap_predictions_along_diagonal

from .base import SelectionFactorOutputHead


class WhichmutResidualHead(SelectionFactorOutputHead):
    """Residual prediction head: output = zap(frozen_logits + delta).

    At initialization with use_residual=True, delta=0 and the output equals
    the zapped frozen structural logits (SC-002). Training learns corrections
    to the structural prior through the delta pathway.

    Architecture:
        - Forward: prediction_layer(embeddings) → delta
        - If use_residual: output = batch[logits_key] + delta
        - Else: output = delta
        - Apply zap_predictions_along_diagonal to total output
        - Loss: ``WhichmutLoss`` (or the dedicated ``WhichmutResidualLoss``
          dataset-dispatch alias) — owned by the Task, not the head.

    Args:
        input_dim: Input embedding dimension from encoder.
        logits_key: Batch dict key for frozen structural logits.
        use_residual: Whether to add frozen logits as residual connection.
    """

    def __init__(
        self,
        input_dim: int,
        logits_key: str = "structural_logits",
        use_residual: bool = True,
        num_amino_acids: int = 20,
    ) -> None:
        """Initialize WhichmutResidualHead.

        Args:
            input_dim: Input embedding dimension from sequence encoder.
            logits_key: Batch dict key for frozen structural logits.
            use_residual: If True, zero-init prediction_layer so delta=0 at init.
            num_amino_acids: Number of amino acids (default 20).
        """
        super().__init__()
        self.input_dim = input_dim
        self.logits_key = logits_key
        self.use_residual = use_residual
        self.num_amino_acids = num_amino_acids

        self.prediction_layer = nn.Linear(input_dim, num_amino_acids)

        # Zero-init so delta=0 at initialization (output = frozen logits)
        if use_residual:
            nn.init.zeros_(self.prediction_layer.weight)
            nn.init.zeros_(self.prediction_layer.bias)

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Predict amino acid selection factors with optional residual connection.

        Args:
            sequence_embeddings: Embeddings [batch_size, seq_len, input_dim]
            batch: Batch data containing:
                - aa_parents_idxs for diagonal zapping
                - self.logits_key for frozen structural logits (if use_residual)

        Returns:
            Log selection factors [batch_size, seq_len, 20] with diagonal zapped.
        """
        delta = self.prediction_layer(sequence_embeddings)

        if self.use_residual:
            output = batch[self.logits_key] + delta
        else:
            output = delta

        # Zap diagonal to set wildtype AA selection factor to 1 (log=0)
        aa_parents_idxs = batch["aa_parents_idxs"]
        output = zap_predictions_along_diagonal(output, aa_parents_idxs, fill=0.0)

        return output

    def to_config(self) -> "WhichmutResidualHeadConfig":
        """Reconstruct WhichmutResidualHeadConfig from head architecture.

        Returns:
            WhichmutResidualHeadConfig matching current head state.
        """
        return WhichmutResidualHeadConfig(
            input_dim=self.input_dim,
            logits_key=self.logits_key,
            use_residual=self.use_residual,
        )


# Set config type after class definition to avoid circular import
from dasm2.config import WhichmutResidualHeadConfig  # noqa: E402

WhichmutResidualHead.head_config_type = WhichmutResidualHeadConfig
