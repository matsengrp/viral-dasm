"""DMS functional score prediction head.

After #621 Task #4 the head only owns the forward pass. The Huber loss
plus its learnable ``scale`` parameter live on ``DMSFunctionalScoreLoss``
(``dasm2/losses/dms.py``).
"""

from typing import Dict

from torch import Tensor, nn

from dasm2.utils.tensor import zap_predictions_along_diagonal

from .base import PredictionHead


class DMSFunctionalScoreHead(PredictionHead):
    """DMS functional score prediction head.

    Predicts L×20 amino acid selection factors with diagonal zapping
    (WT selection factors set to 0 in log space, matching DASM /
    Whichmut convention). Loss is computed by ``DMSFunctionalScoreLoss``
    against DMS experimental measurements; the learnable scale
    parameter lives on the loss object.

    Architecture:
        - Forward: linear layer → diagonal zapping (no wiggle)
    """

    def __init__(self, input_dim: int, num_amino_acids: int = 20) -> None:
        """Initialize DMS functional score prediction head.

        Args:
            input_dim: Input embedding dimension from sequence encoder
            num_amino_acids: Number of amino acids (default 20)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_amino_acids = num_amino_acids

        self.prediction_layer = nn.Linear(input_dim, num_amino_acids)

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Predict amino acid selection factors in log space with diagonal zapping.

        Args:
            sequence_embeddings: Embeddings [batch_size, seq_len, input_dim]
            batch: Batch data containing aa_parents_idxs for diagonal zapping

        Returns:
            Log selection factors [batch_size, seq_len, 20] with diagonal zapped

        Raises:
            ValueError: If aa_parents_idxs not in batch (required for diagonal zapping)

        Note:
            Returns log(f_{j,a->a'}) with wildtype (diagonal) set to 0 (factor=1).
            This matches netam's models.py:703-710 pattern.
            Diagonal zapping ensures synonymous codons aren't affected by selection.
        """
        predictions = self.prediction_layer(sequence_embeddings)

        # Diagonal zapping is mandatory for DMS loss semantics - fail fast if missing
        if "aa_parents_idxs" not in batch:
            raise ValueError(
                "DMSFunctionalScoreHead requires 'aa_parents_idxs' in batch for "
                "diagonal zapping. Ensure DMSDataset is used."
            )

        # Zap diagonal to set wildtype AA selection factor to 1 (log=0)
        # This matches netam behavior for output_dim >= 20
        aa_parents_idxs = batch["aa_parents_idxs"]
        predictions = zap_predictions_along_diagonal(
            predictions, aa_parents_idxs, fill=0.0
        )

        return predictions

    def to_config(self) -> "DMSFunctionalScoreHeadConfig":
        """Reconstruct DMSFunctionalScoreHeadConfig from head architecture."""
        return DMSFunctionalScoreHeadConfig(
            input_dim=self.input_dim,
        )

    # from_config() inherited from PredictionHead base class


# Set config type after class definition to avoid circular import
from dasm2.config import DMSFunctionalScoreHeadConfig  # noqa: E402

DMSFunctionalScoreHead.head_config_type = DMSFunctionalScoreHeadConfig
