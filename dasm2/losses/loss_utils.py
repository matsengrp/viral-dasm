"""Shared utilities for loss computation.

Reduced to the one helper the whichmut loss uses. The Q-family key contracts
and ``focal_nll_loss`` went with the QDASM, Hybrid and DASM losses (#4).
"""

import torch
from torch import Tensor


def zero_loss_with_grad(device: torch.device) -> Tensor:
    """Return zero loss with gradient tracking enabled.

    Used when a loss function has no valid mutations to compute loss over
    (e.g., all sequences are synonymous). Returns a zero tensor with
    requires_grad=True to maintain gradient flow through the computation graph.

    Args:
        device: Device for the tensor

    Returns:
        Scalar zero tensor with requires_grad=True
    """
    return torch.tensor(0.0, device=device, requires_grad=True)
