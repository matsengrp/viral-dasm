"""Positional encoding for neural network components."""

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models.

    Implements the sinusoidal positional encoding from "Attention Is All You Need"
    with optional dropout for regularization.

    Operates seq-first: input shape is ``[seq_len, batch, d_model]``.
    """

    pe: torch.Tensor  # Type annotation for registered buffer

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """Initialize positional encoding with sinusoidal patterns.

        Args:
            d_model: Model dimension (must be even for sin/cos pairs)
            dropout: Dropout rate applied after adding positional encoding
            max_len: Maximum sequence length to precompute encodings for
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe: torch.Tensor = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape [seq_len, batch_size, d_model]

        Returns:
            Tensor with positional encoding added and dropout applied
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)  # type: ignore[no-any-return]

    def add_at_positions(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Add positional encoding gathered at explicit per-token positions.

        Unlike :meth:`forward` (which uses the absolute prefix ``pe[:seq_len]``
        and operates seq-first), this gathers the encoding for each token from
        arbitrary position indices — used for per-segment positional encoding,
        where positions reset at segment boundaries. Operates batch-first.

        Args:
            x: Input tensor of shape ``[batch_size, seq_len, d_model]``.
            positions: Long tensor of shape ``[batch_size, seq_len]`` with a
                position index (into the precomputed buffer) for each token.

        Returns:
            Tensor with positional encoding added and dropout applied, same
            shape as ``x``.
        """
        pe = self.pe.squeeze(1)  # [max_len, d_model]
        return self.dropout(x + pe[positions])  # type: ignore[no-any-return]
