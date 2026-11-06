"""M0 head — single learnable scalar with parent-zap so ``exp(s)`` reads as M0 ω."""

from typing import TYPE_CHECKING, Dict

import torch
from torch import Tensor, nn

from dasm2.sequence.constants import NUM_AMINO_ACIDS
from dasm2.utils.tensor import zap_predictions_along_diagonal

from .base import PositionAgnosticHead

if TYPE_CHECKING:
    from dasm2.config import M0HeadConfig


class M0Head(PositionAgnosticHead):
    """Single learnable scalar with parent-zap so ``exp(s)`` reads as M0 ω.

    Structurally a one-parameter scalar like :class:`SingleValueHead`,
    but with ``zap_predictions_along_diagonal`` applied in the forward.
    The zap sets the parent-AA column to ``0.0`` in log space, leaving
    every non-parent AA at the scalar ``s``. Under the DASMLoss
    "small-change" parameterization this gauges the broadcast scalar so
    its MLE recovers the M0 ω: ``exp(s) ≈ (Σ observed non-syn) / (Σ
    expected non-syn at f≡1)`` — empirical dN/dS — directly from
    training, with no ``dasmfit-export`` + phyz ``--fit-omega``
    round-trip.

    Differs from :class:`SingleValueHead` by exactly the one-line zap.
    The ``expand → clone → zap → mask`` shape mirrors
    :class:`SingleMatrixHead`'s forward; the ``.clone()`` is
    load-bearing because ``zap_predictions_along_diagonal`` writes
    in-place and an expanded view cannot be written through. Autograd
    still flows: every non-parent cell traces back to
    ``log_selection_factor``; zapped cells receive no gradient by
    construction (same property as ``SingleMatrixHead``'s diagonal).

    Routes through the evolutionary (branch-length) data path; the v4
    YAML schema records the matching loss as ``LossType.DASM``.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__(input_dim=input_dim)
        self.log_selection_factor = nn.Parameter(torch.tensor(0.0))

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Return the parent-zapped scalar broadcast to ``[B, L, 20]``.

        Ignores ``sequence_embeddings``. Materializes the broadcast with
        ``.clone()`` so ``zap_predictions_along_diagonal`` can write the
        parent-AA column in place, then zeros padded positions via
        ``batch["mask"]``.

        Positions with ``aa_parents_idxs >= 20`` (the X / ``^`` tokens
        whose indices live outside the 20-AA range) are left un-zapped by
        the guard inside ``zap_predictions_along_diagonal``; the gauge
        claim ``exp(s) ≈ ω`` therefore relies on ``batch["mask"]`` being
        ``False`` at those positions. ``BranchLengthDatasetBundle``
        already maintains this invariant; callers that bypass the bundle
        must preserve it.

        Args:
            sequence_embeddings: IGNORED.
            batch: Must contain ``aa_parents_idxs`` of shape ``[B, L]``
                and ``mask`` of shape ``[B, L]``.

        Returns:
            Log selection factors with shape ``[B, L, 20]``. Every
            non-parent AA cell equals ``log_selection_factor``; every
            parent-AA cell is ``0.0``; every padded position is ``0.0``.
        """
        batch_size, seq_len = batch["mask"].shape
        result = self.log_selection_factor.expand(
            batch_size, seq_len, NUM_AMINO_ACIDS
        ).clone()
        result = zap_predictions_along_diagonal(
            result, batch["aa_parents_idxs"], fill=0.0
        )
        return result * batch["mask"].unsqueeze(-1)

    # to_config() inherited from PositionAgnosticHead


# Set config type after class definition to avoid circular import
from dasm2.config import M0HeadConfig  # noqa: E402

M0Head.head_config_type = M0HeadConfig
