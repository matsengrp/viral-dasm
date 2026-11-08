"""Single-value (one scalar) prediction head — adapted from netam SingleValueBinarySelectionModel."""

from typing import Dict

import torch
from torch import Tensor, nn

from dasm2.sequence.constants import NUM_AMINO_ACIDS

from .base import PositionAgnosticHead


class SingleValueHead(PositionAgnosticHead):
    """One learnable scalar log selection factor, broadcast to every site and AA.

    Adapts netam's ``SingleValueBinarySelectionModel`` to dasm2's
    mask-zeroed output convention (the netam reference does not zero
    padded positions; per-site encoder-free heads in dasm2 do — see
    ``PerSiteParentChildHead`` / ``PerSitePreferenceHead``). On
    all-ones masks the two are bitwise identical (verified in
    ``tests/heads/test_single_value_netam_compat.py``).

    Strictly less expressive than every other per-site head — serves as
    a floor baseline for OE / val-loss comparisons against
    ``SelectionFactorHead`` + ``DASMLoss``-trained models.

    Pairs with ``DASMLoss`` so the baseline trains on the same loss
    function as the standard SelectionFactorHead + DASM pairing — making
    val-loss numbers directly comparable. Because the DASM loss formulation sets the parent-codon
    probability to ``1 - sum(other codon probs)`` (the "small change"
    parameterization, not a normalized softmax), a constant log selection
    factor still has nonzero gradient and the scalar will learn an overall
    mutation-rate-like quantity.

    No diagonal zapping is applied: the broadcast scalar is identical for
    every (parent, child) amino-acid pair, so a parent-conditioning ratio
    would collapse to zero regardless. This matches the netam reference.

    Routes through the evolutionary (branch-length) data path; the v4
    YAML schema (Task #6) records the matching loss as
    ``LossType.DASM`` for this head.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__(input_dim=input_dim)
        self.log_selection_factor = nn.Parameter(torch.tensor(0.0))

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Return the single scalar broadcast to ``[batch_size, seq_len, 20]``.

        Ignores ``sequence_embeddings``. Applies ``batch["mask"]`` so padded
        positions are zero in the returned tensor, matching
        ``PerSiteParentChildHead`` and ``PerSitePreferenceHead`` and
        keeping pad values out of position-agnostic consumers.

        Args:
            sequence_embeddings: IGNORED.
            batch: Must contain ``mask`` with shape ``[B, L]``.

        Returns:
            Log selection factors with shape ``[B, L, 20]``.
        """
        batch_size, seq_len = batch["mask"].shape
        result = self.log_selection_factor.expand(batch_size, seq_len, NUM_AMINO_ACIDS)
        return result * batch["mask"].unsqueeze(-1)

    # to_config() inherited from PositionAgnosticHead


# Set config type after class definition to avoid circular import
from dasm2.config import SingleValueHeadConfig  # noqa: E402

SingleValueHead.head_config_type = SingleValueHeadConfig
