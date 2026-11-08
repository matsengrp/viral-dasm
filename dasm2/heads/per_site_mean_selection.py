"""Per-site mean selection prediction head implementation.

Learns a single scalar log selection factor per site, broadcast identically
to all 20 child amino acids. The simplest possible site-level selection model.
"""

from typing import Dict

import torch
from torch import Tensor, nn

from dasm2.sequence.constants import NUM_AMINO_ACIDS
from dasm2.utils.tensor import zap_predictions_along_diagonal

from .base import PerSiteEncoderFreeHead


class PerSiteMeanSelectionHead(PerSiteEncoderFreeHead):
    """Position-specific mean selection factor, child-AA-agnostic.

    Learns a [max_aa_seq_len] vector of log selection factors — one scalar per
    site — and broadcasts each to all 20 child amino acids.  This is the
    simplest site-level selection model: every nonsynonymous mutation at a given
    position shares the same selection effect.

    The parent→parent diagonal is explicitly zapped to 0 (neutral) using the
    batch's actual parent AAs, matching the convention in
    ``PerSiteParentChildHead``.

    Loss is provided by ``WhichmutLoss``.
    """

    def __init__(
        self,
        input_dim: int,
        max_aa_seq_len: int,
    ) -> None:
        """Initialize per-site mean selection head.

        Args:
            input_dim: Embedding dimension (required by interface, but ignored).
            max_aa_seq_len: Maximum amino acid sequence length.
        """
        super().__init__(input_dim=input_dim, max_aa_seq_len=max_aa_seq_len)

        # Learnable log selection factor per site [max_aa_seq_len]
        # Initialized to zero (neutral selection)
        self.log_site_selection = nn.Parameter(torch.zeros(max_aa_seq_len))

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Return site-level log selection factors broadcast to all child AAs.

        Ignores sequence_embeddings.

        Args:
            sequence_embeddings: IGNORED
            batch: Must contain 'aa_parents_idxs' and 'mask'

        Returns:
            Log selection factors [batch_size, seq_len, 20] with diagonal = 0
        """
        batch_size, seq_len = batch["mask"].shape
        aa_parents = batch["aa_parents_idxs"]  # [B, L]

        # Slice to sequence length: [L]
        site_sf = self.log_site_selection[:seq_len]

        # Broadcast to [B, L, 20]
        result = (
            site_sf.unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, seq_len, NUM_AMINO_ACIDS)
        )
        # expand returns a view; zap_predictions_along_diagonal needs a
        # contiguous tensor it can write to, so clone.
        result = result.clone()

        # Zap parent→parent to 0 (neutral in log space)
        result = zap_predictions_along_diagonal(result, aa_parents, fill=0.0)

        # Apply masking
        result = result * batch["mask"].unsqueeze(-1)

        return result

    # to_config() inherited from PerSiteEncoderFreeHead


# Set config type after class definition to avoid circular import
from dasm2.config import PerSiteMeanSelectionHeadConfig  # noqa: E402

PerSiteMeanSelectionHead.head_config_type = PerSiteMeanSelectionHeadConfig
