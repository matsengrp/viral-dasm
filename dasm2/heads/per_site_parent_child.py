"""Per-site parent/child prediction head implementation.

Learns position-specific, parent-conditioned selection factors without
any encoder/transformer dependence.
"""

from typing import Dict

import torch
from torch import Tensor, nn

from dasm2.sequence.constants import NUM_AMINO_ACIDS
from dasm2.utils.tensor import zap_predictions_along_diagonal

from .base import PerSiteEncoderFreeHead


class PerSiteParentChildHead(PerSiteEncoderFreeHead):
    """Position-specific selection factors conditioned on parent amino acid.

    Learns a [max_aa_seq_len, 20, 20] tensor of log selection factors indexed as
    log_sf[position, parent_aa, child_aa]. More expressive than
    PerSiteChildHead (which uses a fixed wildtype and fits L x 20) because
    this conditions on the actual parent amino acid at each site.

    Loss is provided by ``WhichmutLoss``; pre-#621 this head inherited
    from ``WhichmutLossMixin`` and the post-#621 bridge defaults match.
    """

    def __init__(
        self,
        input_dim: int,
        max_aa_seq_len: int,
    ) -> None:
        """Initialize per-site parent/child head.

        Args:
            input_dim: Embedding dimension (required by interface, but ignored).
            max_aa_seq_len: Maximum amino acid sequence length.
        """
        super().__init__(input_dim=input_dim, max_aa_seq_len=max_aa_seq_len)

        # Learnable position-specific factors [max_aa_seq_len, 20, 20]
        # Indexed as log_sf[position, parent_aa, child_aa]
        # Initialized to zero (neutral selection)
        self.log_selection_factors = nn.Parameter(
            torch.zeros(max_aa_seq_len, NUM_AMINO_ACIDS, NUM_AMINO_ACIDS)
        )

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Return parent-conditioned log selection factors.

        Ignores sequence_embeddings. Indexes into the parent dimension using
        batch["aa_parents_idxs"] to produce [B, L, 20] output.

        Args:
            sequence_embeddings: IGNORED
            batch: Must contain 'aa_parents_idxs' and 'mask'

        Returns:
            Log selection factors [batch_size, seq_len, 20] with diagonal zapped
        """
        _batch_size, seq_len = batch["mask"].shape
        aa_parents = batch["aa_parents_idxs"]  # [B, L]

        # Slice to sequence length: [seq_len, 20, 20]
        factors = self.log_selection_factors[:seq_len]

        # Clamp ambiguous-AA indices (X at idx 20, ^ at idx 21) into the
        # valid [0, 19] range before gathering so the advanced index never
        # IndexErrors on aligned PCPs that contain gap codons (gap codons
        # translate to AA_AMBIG_IDX). batch["mask"] zeros those positions
        # downstream so the clamp choice is irrelevant. The
        # zap_predictions_along_diagonal call below has its own
        # `aa_parents_idxs < 20` guard, so we still pass the unclamped
        # tensor there. Pattern matches SingleMatrixHead.
        aa_parents_safe = aa_parents.clamp(0, NUM_AMINO_ACIDS - 1)

        # Index into parent dimension: for each (batch, pos), select the
        # parent_aa slice to get [B, L, 20].
        # Broadcasting: arange [L] broadcasts against aa_parents [B, L].
        # Advanced indexing returns a fresh tensor, so no .clone() needed
        # before the in-place zap_predictions_along_diagonal below.
        pos_indices = torch.arange(seq_len, device=factors.device)
        result = factors[pos_indices, aa_parents_safe]  # [B, L, 20]

        # Diagonal zapping using actual (unclamped) parent indices so the
        # internal `< 20` guard skips ambiguous positions.
        result = zap_predictions_along_diagonal(result, aa_parents, fill=0.0)

        # Apply masking
        result = result * batch["mask"].unsqueeze(-1)

        return result

    # to_config() inherited from PerSiteEncoderFreeHead


# Set config type after class definition to avoid circular import
from dasm2.config import PerSiteParentChildHeadConfig  # noqa: E402

PerSiteParentChildHead.head_config_type = PerSiteParentChildHeadConfig
