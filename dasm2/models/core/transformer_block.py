"""Modular transformer block with composable sub-components.

Replaces PyTorch's monolithic nn.TransformerEncoderLayer with a composable
TransformerBlock where each sub-component (normalization, activation, FFN,
attention) is independently configurable.

Attention and feed-forward are separately configurable blocks.
"""

from typing import Callable, Literal, Optional

import torch.nn as nn
from torch import Tensor

from dasm2.models.core.attention import FlashVarlenBatch, flash_varlen_batch

# ---------------------------------------------------------------------------
# SequenceMixer protocol
# ---------------------------------------------------------------------------

# We cannot use typing.Protocol for SequenceMixer because nn.Module subclasses
# don't structurally match Protocol types in mypy (nn.Module uses complex
# metaclass machinery). Instead we document the contract and rely on duck typing,
# which is how nn.TransformerEncoder works with nn.MultiheadAttention.
#
# The contract: any module returned by an AttentionFactory must implement:
#
#   forward(query, key, value,
#           attn_mask=None, key_padding_mask=None,
#           need_weights=True, average_attn_weights=True) -> (output, Optional[weights])
#
# This is encoder-only — is_causal is intentionally omitted.
#
# A mixer whose class sets ``accepts_varlen = True`` (the flash mixers) also
# takes ``varlen=``, the batch's FlashVarlenBatch. TransformerStack builds it
# once per forward and TransformerBlock passes it only to such mixers, so the
# stack makes one host sync for the padding metadata instead of several per
# layer.

AttentionFactory = Callable[[], nn.Module]
"""Factory that creates a sequence-mixing module per layer.

The factory is a closure capturing d_model, nhead, dropout from the encoder's
config. Each call creates an independent module for one layer. Must be a named
method or function (not a lambda) so the factory type is identifiable for
debugging and logging.
"""


def make_softmax_attention(d_model: int, nhead: int, dropout: float) -> nn.Module:
    """Create a standard softmax multi-head attention module for one layer.

    Shared by the encoders' softmax ``AttentionFactory`` methods
    (``TransformerEncoder``, ``ESM2Decoder``).
    """
    return nn.MultiheadAttention(
        embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
    )


def _validate_attention_factory(attention_factory: AttentionFactory) -> None:
    """Validate that the attention factory is a named function, not a lambda.

    Args:
        attention_factory: Factory callable to validate.

    Raises:
        ValueError: If factory is a lambda.
    """
    if getattr(attention_factory, "__name__", None) == "<lambda>":
        raise ValueError(
            "AttentionFactory must be a named function or method, not a lambda. "
            "Use a named method (e.g., self._make_softmax_attention) so the factory "
            "type is identifiable for debugging and logging."
        )


# ---------------------------------------------------------------------------
# Feed-forward networks
# ---------------------------------------------------------------------------


# Scale factor so SwiGLU's 3-matrix FFN has roughly the same parameter count
# as StandardFFN's 2-matrix FFN at the same dim_feedforward.
_SWIGLU_DIM_SCALE = 2 / 3

# Round internal FFN dimensions to this multiple for GPU SIMD efficiency.
_HARDWARE_ALIGNMENT = 8


def _round_to_multiple(value: int, multiple: int) -> int:
    """Round value to the nearest multiple for hardware efficiency."""
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


class StandardFFN(nn.Module):
    """Standard feed-forward network: Linear → Activation → Dropout → Linear.

    Args:
        d_model: Input and output dimension.
        dim_feedforward: Hidden layer dimension.
        activation: Activation module instance.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        activation: nn.Module,
        dropout: float,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Apply feed-forward transformation.

        Args:
            x: Input tensor of shape [..., d_model].

        Returns:
            Output tensor of shape [..., d_model].
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))  # type: ignore[no-any-return]


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (Shazeer 2020).

    Gated architecture: Linear_down(SiLU(Linear_gate(x)) * Linear_up(x)).

    No internal dropout — dropout is applied at the residual connection level
    by ``TransformerBlock.post_ffn_drop``.

    The internal dimension is scaled to ``2/3 * dim_feedforward`` (rounded to
    nearest multiple of 8 for hardware efficiency) so that the total FFN
    parameter count is approximately equal to StandardFFN at the same
    ``dim_feedforward``.

    Args:
        d_model: Input and output dimension.
        dim_feedforward: Nominal hidden dimension (scaled internally to 2/3).
    """

    def __init__(self, d_model: int, dim_feedforward: int) -> None:
        super().__init__()
        dim_inner = _round_to_multiple(
            int(dim_feedforward * _SWIGLU_DIM_SCALE), _HARDWARE_ALIGNMENT
        )
        self.linear_gate = nn.Linear(d_model, dim_inner)
        self.linear_up = nn.Linear(d_model, dim_inner)
        self.linear_down = nn.Linear(dim_inner, d_model)
        self.gate_activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU transformation.

        Args:
            x: Input tensor of shape [..., d_model].

        Returns:
            Output tensor of shape [..., d_model].
        """
        gate = self.gate_activation(self.linear_gate(x))
        up = self.linear_up(x)
        return self.linear_down(gate * up)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

_NORM_REGISTRY = {
    "layer_norm": nn.LayerNorm,
    "rms_norm": nn.RMSNorm,
}

_ACTIVATION_REGISTRY = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def create_norm(
    norm_type: Literal["layer_norm", "rms_norm"], d_model: int
) -> nn.Module:
    """Create a normalization layer.

    Args:
        norm_type: Type of normalization.
        d_model: Model dimension.

    Returns:
        Normalization module instance.
    """
    try:
        cls = _NORM_REGISTRY[norm_type]
    except KeyError:
        raise ValueError(
            f"Unknown norm_type {norm_type!r}. Supported: {sorted(_NORM_REGISTRY)}"
        ) from None
    return cls(d_model)  # type: ignore[no-any-return]


def create_ffn(
    activation: Literal["relu", "gelu", "silu", "swiglu"],
    d_model: int,
    dim_feedforward: int,
    dropout: float,
) -> nn.Module:
    """Create a feed-forward network based on the activation type.

    When activation is ``"swiglu"``, returns a ``SwiGLUFFN`` (gated architecture
    with fixed SiLU gate). Otherwise returns a ``StandardFFN`` with the
    corresponding activation function.

    Args:
        activation: Activation / FFN type selector.
        d_model: Input and output dimension.
        dim_feedforward: Hidden layer dimension (scaled internally for SwiGLU).
        dropout: Dropout probability (unused for SwiGLU — residual dropout
            is handled by ``TransformerBlock``).

    Returns:
        Feed-forward module instance.
    """
    if activation == "swiglu":
        return SwiGLUFFN(d_model, dim_feedforward)

    try:
        act_cls = _ACTIVATION_REGISTRY[activation]
    except KeyError:
        raise ValueError(
            f"Unknown activation {activation!r}. "
            f"Supported: {sorted(_ACTIVATION_REGISTRY)} or 'swiglu'"
        ) from None
    return StandardFFN(d_model, dim_feedforward, act_cls(), dropout)


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Composable transformer encoder layer.

    A single transformer encoder layer composed of swappable sub-components:
    sequence mixer (attention), feed-forward network, normalization, and dropout.

    Supports both Pre-LN (norm_first=True) and Post-LN (norm_first=False)
    architectures.

    Args:
        d_model: Model dimension.
        self_attention: Sequence-mixing module (e.g., nn.MultiheadAttention).
        feed_forward: Feed-forward module (e.g., StandardFFN or SwiGLUFFN).
        norm_type: Type of normalization layer.
        norm_first: If True, use Pre-LN (modern). If False, use Post-LN (original).
        dropout: Dropout probability for residual connections.
    """

    def __init__(
        self,
        d_model: int,
        self_attention: nn.Module,
        feed_forward: nn.Module,
        norm_type: Literal["layer_norm", "rms_norm"] = "layer_norm",
        norm_first: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.accepts_varlen = getattr(self_attention, "accepts_varlen", False)
        self.feed_forward = feed_forward
        self.norm_first = norm_first

        # Residual dropout — same role in both Pre-LN and Post-LN
        self.post_attn_drop = nn.Dropout(dropout)
        self.post_ffn_drop = nn.Dropout(dropout)

        if norm_first:
            # Pre-LN: x + drop(attn(norm(x))), x + drop(ffn(norm(x)))
            self.pre_attn_norm = create_norm(norm_type, d_model)
            self.pre_ffn_norm = create_norm(norm_type, d_model)
        else:
            # Post-LN: norm(x + drop(attn(x))), norm(x + drop(ffn(x)))
            self.post_attn_norm = create_norm(norm_type, d_model)
            self.post_ffn_norm = create_norm(norm_type, d_model)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        varlen: Optional[FlashVarlenBatch] = None,
    ) -> Tensor:
        """Forward pass through the transformer block.

        Args:
            src: Input tensor [batch, seq_len, d_model].
            src_mask: Attention mask [seq_len, seq_len] or [batch*nhead, seq_len, seq_len].
            src_key_padding_mask: Padding mask [batch, seq_len].
            varlen: ``src_key_padding_mask``'s varlen metadata, passed to the
                mixer if it accepts it (``accepts_varlen``), else ignored.

        Returns:
            Output tensor [batch, seq_len, d_model].
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.pre_attn_norm(x), src_mask, src_key_padding_mask, varlen
            )
            x = x + self._ff_block(self.pre_ffn_norm(x))
        else:
            x = self.post_attn_norm(
                x + self._sa_block(x, src_mask, src_key_padding_mask, varlen)
            )
            x = self.post_ffn_norm(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        varlen: Optional[FlashVarlenBatch],
    ) -> Tensor:
        """Self-attention block with residual dropout."""
        varlen_kwargs = {"varlen": varlen} if self.accepts_varlen else {}
        out = self.self_attention(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            **varlen_kwargs,
        )[0]
        return self.post_attn_drop(out)  # type: ignore[no-any-return]

    def _ff_block(self, x: Tensor) -> Tensor:
        """Feed-forward block with residual dropout."""
        return self.post_ffn_drop(self.feed_forward(x))  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# TransformerStack
# ---------------------------------------------------------------------------


class TransformerStack(nn.Module):
    """Stack of TransformerBlock layers replacing nn.TransformerEncoder.

    Args:
        d_model: Model dimension.
        num_layers: Number of transformer blocks.
        attention_factory: Callable that creates a sequence-mixing module per layer.
        activation: Activation / FFN type selector.
        dim_feedforward: Hidden layer dimension for FFN.
        norm_type: Type of normalization layer.
        norm_first: If True, use Pre-LN. If False, use Post-LN.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        attention_factory: AttentionFactory,
        activation: Literal["relu", "gelu", "silu", "swiglu"] = "relu",
        dim_feedforward: int = 2048,
        norm_type: Literal["layer_norm", "rms_norm"] = "layer_norm",
        norm_first: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_attention_factory(attention_factory)
        self.norm_first = norm_first

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    self_attention=attention_factory(),
                    feed_forward=create_ffn(
                        activation, d_model, dim_feedforward, dropout
                    ),
                    norm_type=norm_type,
                    norm_first=norm_first,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.builds_varlen = any(layer.accepts_varlen for layer in self.layers)

        # Pre-LN architectures need a final norm because each block normalizes
        # *before* its sublayers, leaving the last residual sum un-normalized.
        self.final_norm: Optional[nn.Module] = (
            create_norm(norm_type, d_model) if norm_first else None
        )

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through all transformer blocks.

        Args:
            src: Input tensor [batch, seq_len, d_model].
            src_mask: Attention mask [seq_len, seq_len] or
                [batch*nhead, seq_len, seq_len].
            src_key_padding_mask: Padding mask [batch, seq_len]. When the
                mixers accept ``varlen``, the stack converts it to a
                FlashVarlenBatch once here and every layer reuses it.

        Returns:
            Output tensor [batch, seq_len, d_model].
        """
        output = src
        varlen = None
        if self.builds_varlen:
            bsz, seq_len = src.shape[:2]
            varlen = flash_varlen_batch(src_key_padding_mask, bsz, seq_len, src.device)
        for layer in self.layers:
            output = layer(output, src_mask, src_key_padding_mask, varlen)
        if self.final_norm is not None:
            output = self.final_norm(output)
        return output
