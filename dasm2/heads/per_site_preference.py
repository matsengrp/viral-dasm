"""Per-site preference prediction head implementation.

Fits a per-site preference (fitness) vector and derives selection factors
as ratios, enforcing cycle consistency of the selection field.
"""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dasm2.sequence.constants import NUM_AMINO_ACIDS

from .base import PerSiteEncoderFreeHead


class PerSitePreferenceHead(PerSiteEncoderFreeHead):
    """Position-specific preference model deriving selection factors as ratios.

    Learns a [max_aa_seq_len, 20] preference vector and computes selection factors
    as log_sf[pos, parent -> child] = log_pi[pos, child] - log_pi[pos, parent].

    This is the population-genetics "fitness landscape" parameterization where
    substitution rates are proportional to destination/source fitness ratios.
    Strictly less expressive than PerSiteParentChildHead (rank-1 in parent x
    child interaction) but enforces cycle consistency of the selection field
    (not detailed balance, which would additionally require the neutral SHM
    mutation process to be reversible — it plausibly is not).

    Gauge fixed via softmax: preferences are normalized into a proper
    distribution at each site.

    Loss is provided by ``WhichmutLoss``; pre-#621 this head inherited
    from ``WhichmutLossMixin``.
    """

    def __init__(
        self,
        input_dim: int,
        max_aa_seq_len: int,
    ) -> None:
        """Initialize per-site preference head.

        Args:
            input_dim: Embedding dimension (required by interface, but ignored).
            max_aa_seq_len: Maximum amino acid sequence length.
        """
        super().__init__(input_dim=input_dim, max_aa_seq_len=max_aa_seq_len)

        # Learnable raw preferences [max_aa_seq_len, 20]
        # Initialized to zero (uniform preferences before softmax)
        self.raw_preferences = nn.Parameter(
            torch.zeros(max_aa_seq_len, NUM_AMINO_ACIDS)
        )

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Return preference-derived log selection factors.

        Ignores sequence_embeddings. Computes log_pi via log_softmax, then
        derives selection factors as log_pi[child] - log_pi[parent].

        Diagonal zapping is automatic: log_pi[a] - log_pi[a] = 0.

        Args:
            sequence_embeddings: IGNORED
            batch: Must contain 'aa_parents_idxs' and 'mask'

        Returns:
            Log selection factors [batch_size, seq_len, 20] with diagonal = 0
        """
        batch_size, seq_len = batch["mask"].shape
        aa_parents = batch["aa_parents_idxs"]  # [B, L]

        # Clamp ambiguous-AA indices (X at idx 20, ^ at idx 21) into the
        # valid [0, 19] range so the gather doesn't IndexError on aligned
        # PCPs that contain gap codons (gap codons translate to
        # AA_AMBIG_IDX). batch["mask"] zeros those positions downstream
        # so the clamp choice is irrelevant. Pattern matches
        # SingleMatrixHead / PerSiteParentChildHead.
        aa_parents_safe = aa_parents.clamp(0, NUM_AMINO_ACIDS - 1)

        # Compute log preferences via log_softmax for gauge fixing
        log_pi = F.log_softmax(self.raw_preferences[:seq_len], dim=-1)  # [L, 20]

        # log_pi[pos, child] for all children: expand to [B, L, 20]
        log_pi_child = log_pi.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, 20]

        # log_pi[pos, parent] for the actual parent at each position: [B, L]
        # Then expand to [B, L, 1] for broadcasting
        log_pi_parent = log_pi_child.gather(
            dim=2, index=aa_parents_safe.unsqueeze(-1)
        )  # [B, L, 1]

        # Selection factors: log_pi[child] - log_pi[parent]
        # Diagonal (parent == child) automatically gives 0
        result = log_pi_child - log_pi_parent  # [B, L, 20]

        # Apply masking
        result = result * batch["mask"].unsqueeze(-1)

        return result

    # to_config() inherited from PerSiteEncoderFreeHead


# Set config type after class definition to avoid circular import
from dasm2.config import PerSitePreferenceHeadConfig  # noqa: E402

PerSitePreferenceHead.head_config_type = PerSitePreferenceHeadConfig
