"""Shared forward pass for AA-selection-factor heads (DASM/QDASM/Hybrid).

The three head implementations all run ``Linear → optional wiggle → diagonal
zap`` on the input embeddings. This helper centralizes that body so the
concrete heads only have to wire it in.
"""

from typing import Dict, Optional

import torch.nn.functional as F
from torch import Tensor, nn

from dasm2.models.core.sf_projection import wiggle
from dasm2.utils.tensor import zap_predictions_along_diagonal


def selection_factor_forward(
    prediction_layer: nn.Linear,
    sequence_embeddings: Tensor,
    batch: Dict[str, Tensor],
    wiggle_beta: Optional[float],
    log_floor: Optional[float] = None,
) -> Tensor:
    """Linear → optional wiggle → optional lower-bound floor → diagonal zap.

    Args:
        prediction_layer: Linear projection to AA selection factors.
        sequence_embeddings: ``[batch_size, seq_len, input_dim]``.
        batch: Full batch dict; if ``aa_parents_idxs`` is present the parent
            AA's selection factor is zapped to 0 (matches DASM's
            "no self-mutation" convention).
        wiggle_beta: Beta parameter for the wiggle activation; ``None``
            disables wiggle and passes the linear output through.
        log_floor: Optional smooth lower bound on the log selection factors,
            applied as ``log_floor + softplus(x - log_floor)`` between wiggle
            and the diagonal zap. ``None`` disables it. This generic param is
            wired from the head/config field ``selection_log_floor`` (renamed at
            the call site because the helper is shared by all selection-factor
            heads). Used by QDASM: its ratio-form Q (``f_target / f_source``) is
            numerically unsafe when ``wiggle`` (unbounded below) drives a source
            factor toward zero, because ``1/f_source`` then explodes. The
            softplus floor caps that while keeping the gradient nonzero
            everywhere (``σ(x - log_floor)``, no dead units). The zap
            (parent → 0) runs *after* the floor, so the wildtype gauge
            ``f_parent = 1`` is preserved exactly: the zap target 0 lies above
            any valid floor (``log_floor < 0``).

    Returns:
        AA selection factors ``[batch_size, seq_len, 20]``.
    """
    output: Tensor = prediction_layer(sequence_embeddings)

    if wiggle_beta is not None:
        output = wiggle(output, wiggle_beta)

    if log_floor is not None:
        output = log_floor + F.softplus(output - log_floor)

    if "aa_parents_idxs" in batch:
        output = zap_predictions_along_diagonal(
            output, batch["aa_parents_idxs"], fill=0.0
        )

    return output
