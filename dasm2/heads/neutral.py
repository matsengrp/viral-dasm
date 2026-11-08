"""Neutral head — every log selection factor is 0, so the model is f ≡ 1."""

from typing import Dict

import torch
from torch import Tensor

from dasm2.sequence.constants import NUM_AMINO_ACIDS

from .base import PositionAgnosticHead


class NeutralHead(PositionAgnosticHead):
    """Parameter-free head emitting 0 in log space: the f ≡ 1 model.

    Paired with ``DASMLoss``, a model built from this head is the neutral
    model alone, scored under whichever endpoint kernel
    ``DASMLossConfig.normalization`` names: ``parent_absorbing`` gives the
    stop-absorbed neutral row, ``sense_conditional`` gives
    ``u_y / (1 - s)`` (issue #1238). That makes it the zero-parameter floor
    for likelihood and O/E comparisons against trained DASM checkpoints,
    configured like any other model instead of built by overwriting a
    trained checkpoint's forward.

    Differs from :class:`SingleValueHead` only in having no parameter: the
    single-value scalar fixed at 0. There is nothing to train, so a model
    built from this head is not trained; it is materialized with
    :func:`dasm2.serialization.checkpoints.save_parameter_free_checkpoint`
    and then loaded, scored and refit (branch lengths) like any
    checkpoint. Pair it with ``IdentityEncoder`` so the model carries no
    parameters at all.

    Routes through the evolutionary (branch-length) data path; the
    matching loss is ``LossType.DASM``.
    """

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Return zeros of shape ``[batch_size, seq_len, 20]``.

        Args:
            sequence_embeddings: IGNORED.
            batch: Must contain ``mask`` with shape ``[B, L]``.

        Returns:
            Log selection factors, all ``0.0``, with shape ``[B, L, 20]``.
        """
        mask = batch["mask"]
        return torch.zeros(
            *mask.shape, NUM_AMINO_ACIDS, device=mask.device, dtype=torch.float32
        )

    # to_config() inherited from PositionAgnosticHead


# Set config type after class definition to avoid circular import
from dasm2.config import NeutralHeadConfig  # noqa: E402

NeutralHead.head_config_type = NeutralHeadConfig
