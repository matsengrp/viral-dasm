"""Per-site child prediction head implementation.

Based on netam ParentIndependentBinarySelectionModel.
"""

from typing import ClassVar, Dict, Optional

import torch
from torch import Tensor, nn

from dasm2.sequence.constants import TOKEN_STR_SORTED
from dasm2.sequence.conversion import aa_idx_tensor_of_str_ambig
from dasm2.utils.tensor import zap_predictions_along_diagonal

from .base import PredictionHead


class PerSiteChildHead(PredictionHead):
    """Position-specific selection factors independent of parent sequence.

    Learns a fixed ``[max_aa_seq_len, 20]`` tensor of log selection factors.
    Unlike transformer-based heads, completely ignores sequence embeddings.

    Loss is provided by ``WhichmutLoss``.

    Diagonal-zap mode is selected by ``wildtype_sequence``:
    - ``None`` (default): zap **per-parent** using ``batch["aa_parents_idxs"]``
      (each sequence's own parent AA is its neutral reference), identical to
      ``PerSiteParentChildHead``. This is the architectural baseline anchor.
    - a string: zap on that single fixed reference for every sequence
      (legacy / comparison mode; used by the fitness experiments).

    Matches netam's ``ParentIndependentBinarySelectionModel`` (fixed mode).
    """

    requires_max_aa_seq_len: ClassVar[bool] = True

    # Type annotation for buffer (set by register_buffer; None in per-parent mode)
    wildtype_aa_idxs: Optional[Tensor]

    def __init__(
        self,
        input_dim: int,
        max_aa_seq_len: int,
        wildtype_sequence: Optional[str] = None,
    ) -> None:
        """Initialize per-site child head.

        Args:
            input_dim: Embedding dimension (required by interface, ignored).
            max_aa_seq_len: Maximum amino acid sequence length. Auto-populated
                from ``ModelConfig.max_aa_seq_len`` by the config layer.
            wildtype_sequence: Optional fixed wildtype AA string. When omitted,
                the head zaps per-parent. When given, it zaps on this fixed
                reference for every sequence.

        Raises:
            ValueError: If a wildtype_sequence contains invalid amino acid
                characters.
        """
        super().__init__()

        self.max_aa_seq_len = max_aa_seq_len
        self.input_dim = input_dim  # Store for interface compatibility

        if wildtype_sequence is None:
            # Per-parent mode: no fixed reference buffer.
            self.register_buffer("wildtype_aa_idxs", None)
        else:
            # Fixed mode: validate and store the reference indices once.
            # aa_idx_tensor_of_str_ambig raises ValueError on invalid characters.
            wildtype_aa_idxs = aa_idx_tensor_of_str_ambig(wildtype_sequence)
            self.register_buffer("wildtype_aa_idxs", wildtype_aa_idxs)

        # Learnable position-specific factors [max_aa_seq_len, 20].
        # Matches netam parameter name exactly.
        self.log_selection_factors = nn.Parameter(torch.zeros(max_aa_seq_len, 20))

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Return position-specific log selection factors.

        Ignores sequence_embeddings. Applies diagonal zapping at the neutral
        reference: per-parent (``batch["aa_parents_idxs"]``) when no fixed
        wildtype was given, else the fixed ``wildtype_aa_idxs``.

        Args:
            sequence_embeddings: IGNORED — parent-independent model.
            batch: Must contain 'mask'; per-parent mode also uses
                'aa_parents_idxs'.

        Returns:
            Log selection factors [batch_size, seq_len, 20] with the diagonal
            zapped at the neutral reference.
        """
        batch_size, seq_len = batch["mask"].shape

        # Slice position factors to sequence length, expand to batch.
        factors = self.log_selection_factors[:seq_len]  # [seq_len, 20]
        result = factors.unsqueeze(0).expand(batch_size, -1, -1).clone()

        if self.wildtype_aa_idxs is None:
            # Per-parent: zap each row at its own parent AA (matches
            # PerSiteParentChildHead.forward); the helper's internal
            # `< 20` guard skips ambiguous positions.
            zap_idxs = batch["aa_parents_idxs"]
        else:
            # Fixed: same reference for every row in the batch.
            zap_idxs = (
                self.wildtype_aa_idxs[:seq_len].unsqueeze(0).expand(batch_size, -1)
            )
        result = zap_predictions_along_diagonal(result, zap_idxs, fill=0.0)

        # Apply masking (positions with mask=False get multiplied by 0).
        result = result * batch["mask"].unsqueeze(-1)

        return result

    def to_config(self) -> "PerSiteChildHeadConfig":
        """Reconstruct PerSiteChildHeadConfig from head state.

        Round-trips the zap mode: per-parent heads serialize with
        ``wildtype_sequence=None``; fixed heads serialize the AA string.
        """
        if self.wildtype_aa_idxs is None:
            wildtype_str: Optional[str] = None
        else:
            wildtype_str = "".join(
                TOKEN_STR_SORTED[idx] for idx in self.wildtype_aa_idxs.cpu().tolist()
            )

        return PerSiteChildHeadConfig(
            input_dim=self.input_dim,
            max_aa_seq_len=self.max_aa_seq_len,
            wildtype_sequence=wildtype_str,
        )

    # from_config() inherited from PredictionHead base class


# Set config type after class definition to avoid circular import
from dasm2.config import PerSiteChildHeadConfig  # noqa: E402

PerSiteChildHead.head_config_type = PerSiteChildHeadConfig
