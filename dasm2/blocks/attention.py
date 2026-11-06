"""Cross-attention block for two-entity heads (issue #801).

Generalizes the masked cross-attention hand-rolled in
``experiments/2026-06-07-binding-classification-sabdab/scripts/binding_classifier.py``
into a shared building block, following the same convention as
``dasm2/blocks/pooling.py`` (``MultiHeadScorePooling``): shared architecture
lives in ``dasm2/blocks/`` so more than one head can reuse it.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from one sequence, keys/values another.

    ``forward`` optionally takes ``context_mask`` (``True`` = valid context
    position) so padded key/value positions are excluded from the softmax,
    and ``query_mask`` (``True`` = valid query position) so padded query
    positions never contribute downstream.

    Masking is asymmetric by necessity: ``context_mask`` gates the softmax
    (masking a key), while ``query_mask`` zeroes the *output* rather than the
    score row — masking an entire query's score row to ``-inf`` would make
    its softmax NaN. Every context sequence must have at least one valid
    token: since the sentinel used to mask is a large finite negative number
    (not literal ``-inf``), a fully-masked context row would not raise or
    NaN — it would silently produce a uniform average over padding. ``forward``
    checks for this and raises instead.
    """

    def __init__(
        self, query_dim: int, context_dim: int, attn_dim: int, num_heads: int
    ) -> None:
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError(
                f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        self.q_proj = nn.Linear(query_dim, attn_dim)
        self.k_proj = nn.Linear(context_dim, attn_dim)
        self.v_proj = nn.Linear(context_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, attn_dim)

    def forward(
        self,
        query_seq: Tensor,
        context_seq: Tensor,
        query_mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Cross-attend ``query_seq`` into ``context_seq``.

        Args:
            query_seq: ``[B, Q, query_dim]``.
            context_seq: ``[B, C, context_dim]``.
            query_mask: ``[B, Q]`` bool, True = valid query position.
            context_mask: ``[B, C]`` bool, True = valid context position.

        Returns:
            ``(out, attn)``: ``out`` is ``[B, Q, attn_dim]``; ``attn`` is the
            post-softmax attention weights ``[B, H, Q, C]``.
        """
        B, Q_len, _ = query_seq.shape
        _, C_len, _ = context_seq.shape

        Q = self.q_proj(query_seq)
        K = self.k_proj(context_seq)
        V = self.v_proj(context_seq)

        Q = Q.view(B, Q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, C_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, C_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim**0.5)  # [B, H, Q, C]
        if context_mask is not None:
            valid_counts = context_mask.sum(dim=-1)
            if (valid_counts == 0).any():
                empty = torch.where(valid_counts == 0)[0].tolist()
                raise ValueError(
                    f"CrossAttention context_mask has no valid positions for "
                    f"samples at indices {empty}; every context sequence must "
                    "have at least one valid token."
                )
            pad = ~context_mask[:, None, None, :]  # [B, 1, 1, C]
            scores = scores.masked_fill(pad, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        out = attn @ V  # [B, H, Q, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, Q_len, self.attn_dim)
        out = self.out_proj(out)
        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1).to(out.dtype)  # [B, Q, attn_dim]
        return out, attn
