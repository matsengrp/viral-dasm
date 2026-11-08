"""Attention modules for the TransformerBlock sequence mixer slot.

All attention variants live here so there is a single place to find,
compare, and compose them.  ``TransformerEncoder`` selects among these
via ``_select_attention_factory`` based on ``EncoderConfig`` fields.

Variants
--------
- **Softmax** (vanilla): ``nn.MultiheadAttention`` — used directly, no wrapper.
- **Entmax** (sparse): ``EntmaxMultiheadAttention`` — replaces softmax with
  alpha-entmax for sparse attention patterns.
- **Rotary** (relative position): ``RotaryMultiheadAttention`` — applies
  Rotary Position Embeddings (RoPE) to Q/K at every layer. Handles entmax
  internally via its ``entmax_alpha`` parameter.
- **Flash** (hardware-fused softmax): ``FlashMultiheadAttention`` — uses
  standalone flash-attn v2 via ``flash_attn_varlen_func`` for IO-aware
  attention on fp16/bf16 inputs. Softmax path only.
- **Flash + Rotary** (hardware-fused softmax with relative position):
  ``FlashRotaryMultiheadAttention`` — applies RoPE to Q/K (as in
  ``RotaryMultiheadAttention``) then dispatches to the flash-attn kernel
  (as in ``FlashMultiheadAttention``). Softmax path only.

Shared helpers
--------------
- ``apply_attention_normalization``: softmax / entmax15 / sparsemax dispatch
  used by both ``EntmaxMultiheadAttention`` and ``RotaryMultiheadAttention``.
- ``RotaryEmbedding``: precomputed sin/cos frequency tables for RoPE.
- ``apply_rotary_embeddings``: pure function that rotates Q/K tensors.
- ``flash_varlen_batch``: converts ``key_padding_mask`` to the cumulative
  sequence lengths and pack indices for the varlen flash-attn API, once per
  forward (``FlashVarlenBatch``).
- ``_pack_padded`` / ``_unpack_padded``: remove and restore padding tokens.
"""

import warnings
from typing import ClassVar, Literal, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15, sparsemax
from torch import Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter

# ---------------------------------------------------------------------------
# Shared normalization helper
# ---------------------------------------------------------------------------


def apply_attention_normalization(
    attn_weights: torch.Tensor,
    alpha: Optional[Literal[1.5, 2.0]],  # type: ignore[valid-type]
) -> torch.Tensor:
    """Apply attention normalization based on alpha value.

    Shared by ``EntmaxMultiheadAttention`` and ``RotaryMultiheadAttention``.

    Args:
        attn_weights: Attention scores of shape ``[..., seq_len, seq_len]``.
        alpha: ``None`` = softmax (default), ``1.5`` = entmax15,
            ``2.0`` = sparsemax.

    Returns:
        Normalized attention weights.
    """
    if alpha is None:
        return F.softmax(attn_weights, dim=-1)  # type: ignore[no-any-return]
    elif alpha == 1.5:
        return entmax15(attn_weights, dim=-1)  # type: ignore[no-any-return]
    elif alpha == 2.0:
        return sparsemax(attn_weights, dim=-1)  # type: ignore[no-any-return]
    else:
        raise ValueError(
            f"alpha must be None, 1.5, or 2.0, got {alpha}. "
            f"Intermediate alpha values require slow bisection-based algorithms."
        )


# ---------------------------------------------------------------------------
# Entmax multi-head attention (sparse)
# ---------------------------------------------------------------------------


class EntmaxMultiheadAttention(nn.Module):
    """MultiheadAttention with configurable alpha-entmax normalization.

    This class implements the PyTorch MultiheadAttention interface but replaces
    the standard softmax attention normalization with alpha-entmax, enabling
    sparse attention patterns for better biological interpretability.

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        alpha: Attention normalization function selector
            - 1.0: Standard softmax (dense)
            - 1.5: Entmax15 (moderately sparse, default)
            - 2.0: Sparsemax (maximally sparse)
        dropout: Dropout probability on attention weights
        bias: Add bias to input/output projection layers
        batch_first: If True, batch dimension is first (default: True)
        add_bias_kv: Add bias to key and value sequences at dim=0
        add_zero_attn: Add new batch of zeros to key and value at dim=1
        kdim: Total number of features for keys (default: embed_dim)
        vdim: Total number of features for values (default: embed_dim)
        device: Device for parameters
        dtype: Data type for parameters
    """

    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        alpha: Literal[1.0, 1.5, 2.0] = 1.5,  # type: ignore[valid-type]
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {"device": device, "dtype": dtype}  # type: ignore[dict-item]
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.alpha = alpha
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(
                torch.empty((embed_dim, embed_dim), **factory_kwargs)  # type: ignore[arg-type]
            )
            self.k_proj_weight = Parameter(
                torch.empty((embed_dim, self.kdim), **factory_kwargs)  # type: ignore[arg-type]
            )
            self.v_proj_weight = Parameter(
                torch.empty((embed_dim, self.vdim), **factory_kwargs)  # type: ignore[arg-type]
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)  # type: ignore[arg-type]
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))  # type: ignore[arg-type]
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            **factory_kwargs,  # type: ignore[arg-type]
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))  # type: ignore[arg-type]
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))  # type: ignore[arg-type]
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters using Xavier initialization."""
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state: dict) -> None:
        """Support loading old MultiheadAttention checkpoints generated by v1.1.0."""
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super().__setstate__(state)

    def merge_masks(
        self,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        query: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[int]]:
        """Determine mask type and combine masks if necessary.

        This method exists for PyTorch 2.8+ compatibility with TransformerEncoderLayer.
        If only one mask is provided, that mask and the corresponding mask type will be
        returned. If both masks are provided, they will be both expanded to shape
        (batch_size, num_heads, seq_len, seq_len), combined with logical OR and mask
        type 2 will be returned.

        Args:
            attn_mask: Attention mask of shape (seq_len, seq_len), mask type 0
            key_padding_mask: Padding mask of shape (batch_size, seq_len), mask type 1
            query: Query embeddings of shape (batch_size, seq_len, embed_dim)

        Returns:
            Tuple of (merged_mask, mask_type):
                - merged_mask: Combined mask or None
                - mask_type: Integer indicator (0, 1, 2, or None)
        """
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(
                    batch_size, self.num_heads, -1, -1
                )
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(
                    batch_size, 1, 1, seq_len
                ).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type

    def _apply_attention_normalization(
        self, attn_weights: torch.Tensor
    ) -> torch.Tensor:
        """Apply alpha-entmax normalization based on alpha value.

        Delegates to the shared ``apply_attention_normalization`` function.

        Args:
            attn_weights: Attention scores of shape [..., seq_len, seq_len]

        Returns:
            Normalized attention weights (probabilities)
        """
        # alpha=1.0 is the legacy softmax sentinel; map to None for the shared helper
        mapped_alpha = None if self.alpha == 1.0 else self.alpha
        return apply_attention_normalization(attn_weights, mapped_alpha)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute attention outputs using query, key, and value embeddings.

        This implements the core multi-head attention mechanism with entmax normalization.
        We do NOT use PyTorch's fastpath optimizations since we need custom normalization.

        Note: Parameter order matches nn.MultiheadAttention (attn_mask before
        key_padding_mask), for SequenceMixer protocol compatibility.

        Args:
            query: Query embeddings [batch, seq_len, embed_dim] if batch_first=True
            key: Key embeddings [batch, seq_len, embed_dim] if batch_first=True
            value: Value embeddings [batch, seq_len, embed_dim] if batch_first=True
            attn_mask: Attention mask [seq_len, seq_len] or [batch*num_heads, seq_len, seq_len]
            key_padding_mask: Mask for padded keys [batch, seq_len]
            need_weights: Return attention weights
            average_attn_weights: Average attention weights across heads
            is_causal: Apply causal mask

        Returns:
            Tuple of (attn_output, attn_output_weights)
        """
        # Handle batch_first dimension ordering
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        # Get dimensions
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        # Compute Q, K, V projections
        if self._qkv_same_embed_dim:
            # Combined projection for efficiency when Q/K/V have same dimension
            q, k, v = self._in_proj_qkv(query, key, value)
        else:
            # Separate projections when dimensions differ
            q = F.linear(
                query,
                self.q_proj_weight,
                self.in_proj_bias[:embed_dim]
                if self.in_proj_bias is not None
                else None,
            )
            k = F.linear(
                key,
                self.k_proj_weight,
                self.in_proj_bias[embed_dim : 2 * embed_dim]
                if self.in_proj_bias is not None
                else None,
            )
            v = F.linear(
                value,
                self.v_proj_weight,
                self.in_proj_bias[2 * embed_dim :]
                if self.in_proj_bias is not None
                else None,
            )

        # Add bias_k and bias_v if specified
        if self.bias_k is not None and self.bias_v is not None:
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
            src_len = k.size(0)

        # Add zero attention if specified
        if self.add_zero_attn:
            k = torch.cat(
                [k, torch.zeros((1, bsz, embed_dim), dtype=k.dtype, device=k.device)],
                dim=0,
            )
            v = torch.cat(
                [v, torch.zeros((1, bsz, embed_dim), dtype=v.dtype, device=v.device)],
                dim=0,
            )
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
            src_len = k.size(0)

        # Reshape for multi-head attention: [tgt_len, bsz, embed_dim] -> [bsz * num_heads, tgt_len, head_dim]
        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = (
            k.contiguous()
            .view(src_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        v = (
            v.contiguous()
            .view(src_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        # Update source sequence length after adding bias/zero attention
        src_len = k.size(1)

        # Compute attention scores: Q * K^T / sqrt(d_k)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        attn_output_weights = attn_output_weights / (self.head_dim**0.5)

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.shape) != [1, tgt_len, src_len]:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {list(attn_mask.shape)}, but should be {[tgt_len, src_len]}."
                    )
                attn_mask = attn_mask.expand(bsz * self.num_heads, -1, -1)
            elif attn_mask.dim() == 3:
                if list(attn_mask.shape) != [bsz * self.num_heads, tgt_len, src_len]:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {list(attn_mask.shape)}, but should be {[bsz * self.num_heads, tgt_len, src_len]}."
                    )
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )

            attn_output_weights = attn_output_weights + attn_mask

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            # Handle both boolean and float masks
            if key_padding_mask.dtype == torch.bool:
                attn_output_weights = attn_output_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float("-inf"),
                )
            else:
                # Float mask: add directly to attention scores
                attn_output_weights = attn_output_weights + key_padding_mask.unsqueeze(
                    1
                ).unsqueeze(2)
            attn_output_weights = attn_output_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        # **THIS IS THE KEY LINE**: Apply entmax normalization instead of softmax
        attn_output_weights = self._apply_attention_normalization(attn_output_weights)

        # Apply dropout
        attn_output_weights = F.dropout(
            attn_output_weights, p=self.dropout, training=self.training
        )

        # Compute attention output: weights * V
        attn_output = torch.bmm(attn_output_weights, v)

        # Reshape back: [bsz * num_heads, tgt_len, head_dim] -> [tgt_len, bsz, embed_dim]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        )

        # Apply output projection
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        # Handle batch_first output
        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        # Prepare attention weights for return
        if need_weights:
            # Reshape: [bsz * num_heads, tgt_len, src_len] -> [bsz, num_heads, tgt_len, src_len]
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            if average_attn_weights:
                attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
            return attn_output, attn_output_weights
        else:
            return attn_output, None

    def _in_proj_qkv(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute Q, K, V projections when using combined in_proj_weight.

        Args:
            query: Query tensor [tgt_len, bsz, embed_dim]
            key: Key tensor [src_len, bsz, embed_dim]
            value: Value tensor [src_len, bsz, embed_dim]

        Returns:
            Tuple of (q, k, v) projections
        """
        w_q, w_k, w_v = self.in_proj_weight.chunk(3)

        if self.in_proj_bias is not None:
            b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        else:
            b_q = b_k = b_v = None  # type: ignore[unreachable]

        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)

        return q, k, v


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021).

    Precomputes sin/cos tables for rotating Q and K vectors.
    Position information is encoded as rotations in 2D subspaces
    of the head dimension, so that the dot product between Q and K
    depends on their relative position.

    The rotation frequencies follow the standard formula::

        theta_i = 1 / (base^(2i/d))  for i in [0, d/2)

    Inherits from ``nn.Module`` for ``register_buffer`` (automatic device
    placement and state_dict inclusion), not for trainable parameters.
    """

    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(
        self, head_dim: int, max_seq_len: int = 5000, base: float = 10000.0
    ) -> None:
        """Initialize rotary embedding with precomputed frequency tables.

        Args:
            head_dim: Dimension per attention head (must be even).
            max_seq_len: Maximum sequence length to precompute tables for.
            base: Base for frequency computation (default 10000.0).
        """
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")

        # Compute frequency bands: theta_i = 1 / (base^(2i/d))
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # Precompute position * frequency for all positions
        positions = torch.arange(max_seq_len).float()
        # [max_seq_len, head_dim/2]
        freqs = torch.outer(positions, inv_freq)
        # [max_seq_len, head_dim] — duplicate for paired sin/cos
        freqs = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(
        self, seq_len: int, dtype: Optional[torch.dtype] = None
    ) -> Tuple[Tensor, Tensor]:
        """Return (cos, sin) tables for the given sequence length.

        Args:
            seq_len: Sequence length to return tables for.
            dtype: Optional dtype to cast tables to (e.g. ``torch.bfloat16``
                for mixed-precision training). If None, returns tables in
                their precomputed dtype (float32 by default).

        Returns:
            Tuple of (cos, sin) tensors of shape ``[seq_len, head_dim]``.

        Raises:
            ValueError: If seq_len exceeds the precomputed table size.
        """
        max_seq_len = self.cos_cached.shape[0]
        if seq_len > max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_seq_len ({max_seq_len}). "
                f"Increase max_seq_len at construction time."
            )
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        if dtype is not None:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
        return cos, sin


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate pairs of dimensions by swapping and negating the first half.

    For input [..., d], splits into [..., d/2] halves and returns
    [-x2, x1] which implements the 2D rotation when combined with
    cos/sin multiplication.
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_embeddings(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to Q and K tensors.

    Args:
        q: Query tensor of shape ``[batch * num_heads, seq_len, head_dim]``.
        k: Key tensor of shape ``[batch * num_heads, seq_len, head_dim]``.
        cos: Cosine table of shape ``[seq_len, head_dim]``.
        sin: Sine table of shape ``[seq_len, head_dim]``.

    Returns:
        Tuple of (q_rotated, k_rotated) with same shapes as inputs.

    Note:
        ``cos`` and ``sin`` are broadcast over leading batch dimensions.
    """
    if q.shape[-2] != cos.shape[0] or q.shape[-1] != cos.shape[-1]:
        raise ValueError(
            f"q shape {list(q.shape)} incompatible with cos shape {list(cos.shape)}: "
            f"expected q[..., seq_len, head_dim] to match cos[seq_len, head_dim]"
        )
    if k.shape[-2] != cos.shape[0] or k.shape[-1] != cos.shape[-1]:
        raise ValueError(
            f"k shape {list(k.shape)} incompatible with cos shape {list(cos.shape)}: "
            f"expected k[..., seq_len, head_dim] to match cos[seq_len, head_dim]"
        )
    q_rotated = q * cos + _rotate_half(q) * sin
    k_rotated = k * cos + _rotate_half(k) * sin
    return q_rotated, k_rotated


class RotaryMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embeddings.

    Applies rotary embeddings to Q and K after projection and before the
    dot product. Follows ``nn.MultiheadAttention``'s call signature for
    drop-in compatibility with the TransformerBlock SequenceMixer contract.

    Uses an ``entmax_alpha`` parameter (consistent with ``EncoderConfig``)
    to select attention normalization: ``None`` = softmax (default),
    ``1.5`` = entmax15, ``2.0`` = sparsemax.

    Always operates batch-first. Uses batch-first internal layout
    ``[batch * num_heads, seq_len, head_dim]`` throughout.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 5000,
        entmax_alpha: Optional[Literal[1.5, 2.0]] = None,  # type: ignore[valid-type]
    ) -> None:
        """Initialize rotary multi-head attention.

        Args:
            embed_dim: Total model dimension.
            num_heads: Number of parallel attention heads.
            dropout: Dropout probability on attention weights.
            max_seq_len: Maximum sequence length for RoPE table precomputation.
            entmax_alpha: Attention normalization. None=softmax (default),
                1.5=entmax15, 2.0=sparsemax.
        """
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0, "
                f"got embed_dim={embed_dim} and num_heads={num_heads}"
            )
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.entmax_alpha = entmax_alpha

        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Q, K, V projections (combined for efficiency)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters using Xavier initialization."""
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        average_attn_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute attention with rotary position embeddings.

        Args:
            query: Query tensor ``[batch, seq_len, embed_dim]``.
            key: Key tensor ``[batch, seq_len, embed_dim]``.
            value: Value tensor ``[batch, seq_len, embed_dim]``.
            attn_mask: Attention mask ``[seq_len, seq_len]`` or
                ``[batch * num_heads, seq_len, seq_len]``.
            key_padding_mask: Padding mask ``[batch, seq_len]`` (bool or float).
            need_weights: Return attention weights.
            average_attn_weights: Average attention weights across heads.

        Returns:
            Tuple of (attn_output, attn_weights). attn_output has shape
            ``[batch, seq_len, embed_dim]``.
        """
        bsz, tgt_len, _ = query.shape
        src_len = key.shape[1]

        if tgt_len != src_len:
            raise ValueError(
                f"RoPE only supports self-attention (tgt_len == src_len), "
                f"got tgt_len={tgt_len}, src_len={src_len}"
            )

        # Project Q, K, V using combined weight matrix
        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)

        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)

        # Reshape to [batch * num_heads, seq_len, head_dim]
        q = (
            q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        )
        k = (
            k.view(bsz, src_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz * self.num_heads, src_len, self.head_dim)
        )
        v = (
            v.view(bsz, src_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz * self.num_heads, src_len, self.head_dim)
        )

        # Apply rotary embeddings to Q and K
        cos, sin = self.rotary_emb(tgt_len, dtype=q.dtype)
        q, k = apply_rotary_embeddings(q, k, cos, sin)

        # Compute attention scores: Q * K^T / sqrt(d_k)
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim**0.5)

        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).expand(bsz * self.num_heads, -1, -1)
            elif attn_mask.dim() == 3:
                if list(attn_mask.shape) != [bsz * self.num_heads, tgt_len, src_len]:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {list(attn_mask.shape)}, "
                        f"but should be {[bsz * self.num_heads, tgt_len, src_len]}."
                    )
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )
            attn_weights = attn_weights + attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if key_padding_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )
            else:
                # Float mask: add directly to attention scores
                attn_weights = attn_weights + key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # Apply attention normalization (softmax or entmax)
        attn_weights = apply_attention_normalization(attn_weights, self.entmax_alpha)

        # Apply dropout
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Compute output: weights * V
        attn_output = torch.bmm(attn_weights, v)

        # Reshape back to [batch, seq_len, embed_dim]
        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(bsz, tgt_len, self.embed_dim)
        )

        # Output projection
        attn_output = self.out_proj(attn_output)

        if need_weights:
            attn_weights_out = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_weights_out = attn_weights_out.sum(dim=1) / self.num_heads
            return attn_output, attn_weights_out
        return attn_output, None


# ---------------------------------------------------------------------------
# Flash Attention (standalone flash-attn v2)
# ---------------------------------------------------------------------------


class FlashVarlenBatch(NamedTuple):
    """Varlen metadata for one padded batch, shared by every flash layer.

    Built once per forward by :func:`flash_varlen_batch` and passed to each
    layer, so the stack pays one host sync per forward instead of several per
    layer.

    Attributes:
        cu_seqlens: int32 ``(B + 1,)`` cumulative sequence lengths.
        max_seqlen: Longest unpadded sequence, as a Python int.
        pack_indices: int64 ``(total_tokens,)`` positions of the valid tokens
            in the flattened ``(B * L)`` batch, or ``None`` when nothing is
            padded.
        seq_len: The padded length ``L`` the batch was built for;
            ``pack_indices`` are positions in a ``(B * L)`` flattening.
    """

    cu_seqlens: Tensor
    max_seqlen: int
    pack_indices: Optional[Tensor]
    seq_len: int


def flash_varlen_batch(
    key_padding_mask: Optional[Tensor],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> FlashVarlenBatch:
    """Build ``cu_seqlens``, ``max_seqlen`` and the pack indices from a padding mask.

    One host sync: a single copy of the min, max and total length. The pack
    indices are integers, so packing and unpacking with them
    (``index_select`` / ``index_copy``) needs no sync in the forward or the
    backward, unlike boolean-mask indexing.

    Args:
        key_padding_mask: ``(B, L)`` where ``True`` = padded. May be ``None``
            for uniform-length batches (all positions valid).
        batch_size: Number of sequences in the batch.
        seq_len: Padded sequence length.
        device: Target device.

    Returns:
        A :class:`FlashVarlenBatch`.

    Raises:
        ValueError: If any sequence has length 0 (all positions masked).
    """
    if key_padding_mask is None:
        # Uniform-length: [0, L, 2L, ..., B*L]
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=device
        )
        return FlashVarlenBatch(cu_seqlens, seq_len, None, seq_len)

    valid = ~key_padding_mask  # (B, L) — True = valid
    lengths = valid.sum(dim=1)  # (B,)
    min_len, max_seqlen, total = torch.stack(
        [lengths.min(), lengths.max(), lengths.sum()]
    ).tolist()
    if min_len == 0:
        raise ValueError(
            "flash_attention received a sequence with all positions masked "
            "(length 0). Every sequence must have at least one valid token."
        )
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = lengths.cumsum(0)
    # nonzero_static takes the size it would otherwise sync to learn.
    pack_indices = torch.nonzero_static(valid.reshape(-1), size=total).squeeze(1)
    return FlashVarlenBatch(cu_seqlens, max_seqlen, pack_indices, seq_len)


def _pack_padded(x: Tensor, pack_indices: Optional[Tensor]) -> Tensor:
    """Remove padding positions, returning a flat ``(total_tokens, ...)`` tensor.

    Args:
        x: Padded tensor of shape ``(B, L, ...)``.
        pack_indices: ``FlashVarlenBatch.pack_indices``, or ``None`` for
            uniform-length batches.

    Returns:
        Flat tensor of shape ``(total_tokens, ...)`` with padding removed.
    """
    bsz, slen = x.shape[:2]
    flat = x.reshape(bsz * slen, *x.shape[2:])
    if pack_indices is None:
        return flat
    return flat.index_select(0, pack_indices)


def _unpack_padded(
    packed: Tensor,
    pack_indices: Optional[Tensor],
    batch_size: int,
    seq_len: int,
) -> Tensor:
    """Scatter packed tokens back into a padded ``(B, L, ...)`` tensor.

    Args:
        packed: ``(total_tokens, ...)`` tensor produced by flash attention.
        pack_indices: ``FlashVarlenBatch.pack_indices``, or ``None`` for
            uniform-length batches.
        batch_size: Batch size.
        seq_len: Padded sequence length.

    Returns:
        Padded tensor of shape ``(B, L, ...)`` with zeros at padded positions.
    """
    feature_shape = packed.shape[1:]
    if pack_indices is None:
        return packed.reshape(batch_size, seq_len, *feature_shape)
    flat = packed.new_zeros(batch_size * seq_len, *feature_shape)
    flat = flat.index_copy(0, pack_indices, packed)
    return flat.reshape(batch_size, seq_len, *feature_shape)


def _cast_for_flash_kernel(
    q: Tensor, k: Tensor, v: Tensor, class_name: str
) -> Tuple[Tensor, Tensor, Tensor]:
    """Cast Q/K/V to fp16/bf16 for the flash kernel, warning if not already cast.

    flash-attn requires fp16/bf16. Under torch.autocast the projections
    should already be bf16, but if autocast is not active (e.g. during
    validation or standalone use) the projections remain float32. Cast to
    bf16 in that case rather than raising — this matches how mixed-precision
    training works in practice.

    Args:
        q: Query tensor, any float dtype.
        k: Key tensor, any float dtype.
        v: Value tensor, any float dtype.
        class_name: Calling class's name, embedded in the warning message.

    Returns:
        ``(q, k, v)`` unchanged if already fp16/bf16, else cast to bf16 (or
        fp16 if bf16 isn't supported on this device).
    """
    if q.dtype in (torch.float16, torch.bfloat16):
        return q, k, v
    target = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    warnings.warn(
        f"{class_name}: float32 inputs will be cast to "
        f"{target} for the flash kernel. Use torch.autocast or pass "
        f"fp16/bf16 inputs to suppress this warning.",
        stacklevel=3,
    )
    if not torch.cuda.is_bf16_supported():
        return q.half(), k.half(), v.half()
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def _flash_attend(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    key_padding_mask: Optional[Tensor],
    varlen: Optional[FlashVarlenBatch],
    bsz: int,
    seq_len: int,
    dropout: float,
    training: bool,
    deterministic: bool,
) -> Tensor:
    """Run ``flash_attn_varlen_func`` over padded ``(B, L, nheads, headdim)`` inputs.

    Shared by ``FlashMultiheadAttention`` and ``FlashRotaryMultiheadAttention``:
    packs to remove padding, calls the flash kernel, and unpacks back to
    padded layout.

    Args:
        q, k, v: ``(B, L, nheads, headdim)`` tensors, fp16/bf16.
        key_padding_mask: ``(B, L)`` bool where ``True`` = padded, or ``None``.
            Read only when ``varlen`` is ``None``.
        varlen: The batch's :class:`FlashVarlenBatch`, built once per forward
            by the caller; ``None`` builds it here from ``key_padding_mask``.
        bsz: Batch size.
        seq_len: Padded sequence length.
        dropout: Dropout probability (applied only when ``training``).
        training: Whether the calling module is in training mode.
        deterministic: Use flash-attn's deterministic (slower) backward.

    Returns:
        ``(B, L, nheads, headdim)`` output tensor, zeros at padded positions.
    """
    from flash_attn import flash_attn_varlen_func

    if varlen is None:
        varlen = flash_varlen_batch(key_padding_mask, bsz, seq_len, q.device)
    elif (varlen.cu_seqlens.shape[0] - 1, varlen.seq_len) != (bsz, seq_len):
        raise ValueError(
            f"varlen was built for {varlen.cu_seqlens.shape[0] - 1} sequences "
            f"padded to {varlen.seq_len}, but this layer received {bsz} padded "
            f"to {seq_len}."
        )

    q_packed = _pack_padded(q, varlen.pack_indices)
    k_packed = _pack_padded(k, varlen.pack_indices)
    v_packed = _pack_padded(v, varlen.pack_indices)

    out_packed = flash_attn_varlen_func(
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens_q=varlen.cu_seqlens,
        cu_seqlens_k=varlen.cu_seqlens,
        max_seqlen_q=varlen.max_seqlen,
        max_seqlen_k=varlen.max_seqlen,
        dropout_p=dropout if training else 0.0,
        causal=False,
        deterministic=deterministic,
    )
    return _unpack_padded(out_packed, varlen.pack_indices, bsz, seq_len)


class FlashRotaryMultiheadAttention(nn.Module):
    """Multi-head attention combining RoPE with the standalone flash-attn v2 kernel.

    Applies rotary position embeddings to Q/K (as in ``RotaryMultiheadAttention``)
    before calling ``flash_attn_varlen_func`` (as in ``FlashMultiheadAttention``).
    RoPE is applied after the bf16 cast required by the flash kernel, trading
    ~1 bit of precision for the ability to use the fused kernel — acceptable at
    the ``atol=1e-2`` equivalence target used elsewhere in this module.

    Always operates batch-first. Only supports the softmax attention path
    (``entmax_alpha`` is not configurable here — entmax's custom normalization
    is incompatible with the fused kernel). Requires fp16 or bf16 inputs.
    """

    accepts_varlen: ClassVar[bool] = True  # takes varlen=; see TransformerStack

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 5000,
        deterministic: bool = False,
    ) -> None:
        try:
            import flash_attn  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "FlashRotaryMultiheadAttention requires the flash-attn package. "
                "Install it with: pip install flash-attn --no-build-isolation\n"
                "If loading a checkpoint trained with flash_attention=True and "
                "pos_encoding='rotary', either install flash-attn or set "
                "flash_attention=False in the config to fall back to "
                "RotaryMultiheadAttention (weights are compatible). If you see "
                "a native build error above, check your CUDA version matches "
                "flash-attn's build."
            ) from e
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0, "
                f"got embed_dim={embed_dim} and num_heads={num_heads}"
            )
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.deterministic = deterministic

        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Q, K, V projections (combined for efficiency)
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters using Xavier initialization."""
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
        varlen: Optional[FlashVarlenBatch] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute flash attention with rotary position embeddings.

        Args:
            query: ``(B, L, E)`` query tensor (must equal key/value: self-attention only).
            key: ``(B, L, E)`` key tensor.
            value: ``(B, L, E)`` value tensor.
            attn_mask: Must be ``None`` — flash-attn varlen does not support
                arbitrary attention masks.
            key_padding_mask: ``(B, L)`` bool where ``True`` = padded.
            need_weights: Must be ``False`` (default). Flash-attn does not
                return attention weights.
            average_attn_weights: Ignored (no weights returned).
            varlen: ``key_padding_mask``'s :class:`FlashVarlenBatch`, built
                once per forward by ``TransformerStack``; ``None`` builds it
                in this layer.

        Returns:
            ``(attn_output, None)`` where ``attn_output`` has shape ``(B, L, E)``.
        """
        if attn_mask is not None:
            raise ValueError(
                "FlashRotaryMultiheadAttention does not support attn_mask. "
                "Use key_padding_mask for variable-length sequences."
            )
        if need_weights:
            raise ValueError(
                "FlashRotaryMultiheadAttention does not support need_weights=True. "
                "flash-attn does not return attention weights."
            )
        bsz, seq_len, _ = query.shape
        if key.shape[1] != seq_len:
            raise ValueError(
                f"RoPE only supports self-attention (tgt_len == src_len), "
                f"got tgt_len={seq_len}, src_len={key.shape[1]}"
            )

        # Project Q, K, V (under torch.autocast, F.linear casts float32 → bf16)
        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)

        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)

        # RoPE is applied after this cast (see class docstring), so sin/cos
        # multiplication happens in bf16.
        q, k, v = _cast_for_flash_kernel(q, k, v, "FlashRotaryMultiheadAttention")

        # Apply rotary embeddings to Q and K in [B*H, L, D] layout
        q = (
            q.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz * self.num_heads, seq_len, self.head_dim)
        )
        k = (
            k.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz * self.num_heads, seq_len, self.head_dim)
        )
        cos, sin = self.rotary_emb(seq_len, dtype=q.dtype)
        q, k = apply_rotary_embeddings(q, k, cos, sin)

        # Reshape to (B, L, nheads, headdim) for flash-attn packing
        q = q.view(bsz, self.num_heads, seq_len, self.head_dim).transpose(1, 2)
        k = k.view(bsz, self.num_heads, seq_len, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim)

        out = _flash_attend(
            q,
            k,
            v,
            key_padding_mask,
            varlen,
            bsz,
            seq_len,
            self.dropout,
            self.training,
            self.deterministic,
        )

        # Reshape to (B, L, E) and apply output projection.
        # Cast back to input dtype so out_proj (float32 weights) works
        # when autocast is not active.
        out = out.reshape(bsz, seq_len, self.embed_dim)
        out = out.to(query.dtype)
        out = self.out_proj(out)

        return out, None


class FlashMultiheadAttention(nn.Module):
    """Multi-head attention using standalone flash-attn v2 ``flash_attn_varlen_func``.

    Bypasses ``nn.MultiheadAttention`` entirely to avoid the
    ``key_padding_mask`` → ``attn_mask`` conversion that blocks flash dispatch.
    Variable-length sequences are handled via ``cu_seqlens`` (cumulative
    sequence lengths) instead of masks.

    Always operates batch-first. Only supports the softmax attention path
    (``entmax_alpha is None``). Requires fp16 or bf16 inputs.
    """

    accepts_varlen: ClassVar[bool] = True  # takes varlen=; see TransformerStack

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        deterministic: bool = False,
    ) -> None:
        try:
            import flash_attn  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "FlashMultiheadAttention requires the flash-attn package. "
                "Install it with: pip install flash-attn --no-build-isolation\n"
                "If loading a checkpoint trained with flash_attention=True, "
                "either install flash-attn or set flash_attention=False in the "
                "config to fall back to nn.MultiheadAttention (weights are "
                "compatible). If you see a native build error above, check "
                "your CUDA version matches flash-attn's build."
            ) from e
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0, "
                f"got embed_dim={embed_dim} and num_heads={num_heads}"
            )
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.deterministic = deterministic

        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Q, K, V projections (combined for efficiency)
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters using Xavier initialization."""
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
        varlen: Optional[FlashVarlenBatch] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute flash attention.

        Args:
            query: ``(B, L, E)`` query tensor.
            key: ``(B, L, E)`` key tensor (must equal query for self-attention).
            value: ``(B, L, E)`` value tensor.
            attn_mask: Must be ``None`` — flash-attn varlen does not support
                arbitrary attention masks.
            key_padding_mask: ``(B, L)`` bool where ``True`` = padded.
            need_weights: Must be ``False`` (default). Flash-attn does not
                return attention weights.
            average_attn_weights: Ignored (no weights returned).
            varlen: ``key_padding_mask``'s :class:`FlashVarlenBatch`, built
                once per forward by ``TransformerStack``; ``None`` builds it
                in this layer.

        Returns:
            ``(attn_output, None)`` where ``attn_output`` has shape ``(B, L, E)``.
        """
        if attn_mask is not None:
            raise ValueError(
                "FlashMultiheadAttention does not support attn_mask. "
                "Use key_padding_mask for variable-length sequences."
            )
        if need_weights:
            raise ValueError(
                "FlashMultiheadAttention does not support need_weights=True. "
                "flash-attn does not return attention weights."
            )
        bsz, seq_len, _ = query.shape

        # Project Q, K, V (under torch.autocast, F.linear casts float32 → bf16)
        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)

        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)

        q, k, v = _cast_for_flash_kernel(q, k, v, "FlashMultiheadAttention")

        # Reshape to (B, L, nheads, headdim)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim)

        out = _flash_attend(
            q,
            k,
            v,
            key_padding_mask,
            varlen,
            bsz,
            seq_len,
            self.dropout,
            self.training,
            self.deterministic,
        )

        # Reshape to (B, L, E) and apply output projection.
        # Cast back to input dtype so out_proj (float32 weights) works
        # when autocast is not active.
        out = out.reshape(bsz, seq_len, self.embed_dim)
        out = out.to(query.dtype)
        out = self.out_proj(out)

        return out, None
