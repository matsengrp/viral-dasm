"""Pooling modules for aggregating variable-length sequences to fixed-size representations."""

import math
from typing import Literal, Optional

import torch
from torch import Tensor, nn


def _validate_pooling_inputs(embeddings: Tensor, mask: Tensor) -> None:
    """Validate shapes and mask for pooling forward methods.

    Args:
        embeddings: Per-residue embeddings, expected [B, L, D].
        mask: Boolean mask, expected [B, L].

    Raises:
        ValueError: If shapes are incompatible or any sequence has no valid positions.
    """
    if embeddings.dim() != 3:
        raise ValueError(
            f"embeddings must be 3D [B, L, D], got shape {embeddings.shape}"
        )
    if mask.dim() != 2:
        raise ValueError(f"mask must be 2D [B, L], got shape {mask.shape}")
    if embeddings.shape[:2] != mask.shape:
        raise ValueError(
            f"Shape mismatch: embeddings {embeddings.shape[:2]} vs mask {mask.shape}"
        )

    counts = mask.sum(dim=1)  # [B]
    if (counts == 0).any():
        zero_indices = torch.where(counts == 0)[0].tolist()
        raise ValueError(
            f"Sequences at indices {zero_indices} have no valid positions (all masked)."
        )


class MaskedMeanPooling(nn.Module):
    """Mean pooling over sequence positions with mask support.

    Computes the mean of embeddings at valid (unmasked) positions,
    producing a fixed-size representation regardless of sequence length.

    This is the standard approach for pooling variable-length sequences
    (e.g., antibody heavy+light chains) into a single vector for
    scalar prediction tasks like binding affinity.

    Example:
        >>> pool = MaskedMeanPooling()
        >>> embeddings = torch.randn(4, 100, 256)  # [B, L, D]
        >>> mask = torch.ones(4, 100, dtype=torch.bool)
        >>> pooled = pool(embeddings, mask)  # [4, 256]
    """

    def forward(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        """Pool embeddings over valid positions using masked mean.

        Args:
            embeddings: Per-residue embeddings [batch_size, seq_len, d_model]
            mask: Boolean mask where True = valid position [batch_size, seq_len]

        Returns:
            Pooled embeddings [batch_size, d_model]

        Raises:
            ValueError: If shapes are incompatible or any sequence has no valid positions.
        """
        _validate_pooling_inputs(embeddings, mask)

        counts = mask.sum(dim=1)  # [B]

        # Masked mean: sum valid embeddings / count of valid positions
        mask_expanded = mask.unsqueeze(-1).float()  # [B, L, 1]
        summed = (embeddings * mask_expanded).sum(dim=1)  # [B, D]
        pooled = summed / counts.unsqueeze(-1).float()  # [B, D]

        return pooled


class MultiHeadScorePooling(nn.Module):
    """Multi-head score-based pooling for variable-length sequences.

    Generalizes Bahdanau-style attention pooling to K independent heads, each with its
    own score vector and optional projection. This allows different heads to
    weight positions differently and (in projected modes) attend to different
    feature subspaces.

    Architecture for K heads on inputs h_i in R^D:

        hidden = tanh(W h + b)              # [B, L, D_hidden], shared across heads
        scores = hidden @ u.T               # [B, L, K], u is [K, D_hidden]
        alpha  = softmax(scores, dim=L)     # [B, L, K], per-head over positions
        proj   = project(h, mode)           # [B, L, K, d] (identity: d=D)
        o      = einsum("blk,blkd->bkd", alpha, proj)  # [B, K, d]
        output = o.reshape(B, K*d)          # [B, K*d]

    Projection modes:
        - "identity": no projection, head_dim must equal input_dim. With
          num_heads=1, this is exactly Bahdanau-style rank-1 attention pooling.
        - "learned": per-head learned projection P_k in R^(d x D).
        - "random": per-head fixed random projection (JL-style buffer).

    With default config (num_heads=1, projection_mode="identity"), this module
    is equivalent to Bahdanau-style rank-1 attention pooling.

    Example:
        >>> pool = MultiHeadScorePooling(input_dim=256)
        >>> embeddings = torch.randn(4, 100, 256)
        >>> mask = torch.ones(4, 100, dtype=torch.bool)
        >>> pooled, weights = pool(embeddings, mask)  # [4, 256], [4, 100, 1]

        >>> pool8 = MultiHeadScorePooling(input_dim=256, num_heads=8, head_dim=32,
        ...                                projection_mode="learned")
        >>> pooled8, weights8 = pool8(embeddings, mask)  # [4, 256], [4, 100, 8]
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 1,
        head_dim: Optional[int] = None,
        hidden_dim: int = 64,
        projection_mode: Literal["identity", "random", "learned"] = "identity",
    ) -> None:
        """Initialize the multi-head score pooling module.

        Args:
            input_dim: Dimensionality of input embeddings (D).
            num_heads: Number of independent attention heads (K).
            head_dim: Output dimensionality per head (d). Defaults to input_dim.
            hidden_dim: Dimensionality of the shared scoring hidden layer.
            projection_mode: How to project inputs per head. "identity" uses raw
                embeddings (requires head_dim == input_dim), "learned" uses
                trainable per-head projections, "random" uses fixed random
                projections (JL-style).

        Raises:
            ValueError: If any dimension argument is < 1, or if
                projection_mode is "identity" and head_dim != input_dim.
        """
        super().__init__()

        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")

        if head_dim is None:
            head_dim = input_dim
        if head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {head_dim}")

        if projection_mode == "identity" and head_dim != input_dim:
            raise ValueError(
                f"projection_mode='identity' requires head_dim == input_dim, "
                f"got head_dim={head_dim}, input_dim={input_dim}"
            )

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.projection_mode = projection_mode

        # Shared scoring projection (same W, b for all heads)
        self.score_projection = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.score_projection.weight)
        nn.init.zeros_(self.score_projection.bias)

        # Per-head score vectors u: [K, D_hidden]
        self.score_vectors = nn.Parameter(torch.empty(num_heads, hidden_dim))
        bound = 1.0 / math.sqrt(hidden_dim)
        nn.init.uniform_(self.score_vectors, -bound, bound)

        # Per-head projections
        if projection_mode == "learned":
            self.projections = nn.Parameter(torch.empty(num_heads, head_dim, input_dim))
            # Xavier init per head: reshape to 2D so fan_in/fan_out are correct
            nn.init.xavier_uniform_(
                self.projections.view(num_heads * head_dim, input_dim)
            )
        elif projection_mode == "random":
            proj = torch.randn(num_heads, head_dim, input_dim) / math.sqrt(head_dim)
            self.register_buffer("projections", proj)

    @property
    def output_dim(self) -> int:
        """Total output dimensionality (K * d)."""
        return self.num_heads * self.head_dim

    def forward(self, embeddings: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Pool embeddings using multi-head attention-weighted sum.

        Args:
            embeddings: Per-residue embeddings [batch_size, seq_len, input_dim].
            mask: Boolean mask where True = valid position [batch_size, seq_len].

        Returns:
            Tuple of (pooled, attention_weights):
                pooled: Pooled embeddings [batch_size, num_heads * head_dim].
                attention_weights: Per-head attention weights [batch_size,
                    seq_len, num_heads], detached from the computation graph.

        Raises:
            ValueError: If shapes are incompatible or any sequence has no
                valid positions.
        """
        _validate_pooling_inputs(embeddings, mask)

        B, L, D = embeddings.shape
        K = self.num_heads

        # Shared hidden representation: tanh(W h + b)
        hidden = torch.tanh(self.score_projection(embeddings))  # [B, L, D_hidden]

        # K=1 fast path: use 1D matmul and 2D einsum to avoid extra dimensions
        if K == 1:
            scores_2d = torch.matmul(hidden, self.score_vectors[0])  # [B, L]
            scores_2d = scores_2d.masked_fill(~mask, torch.finfo(scores_2d.dtype).min)
            weights_2d = torch.softmax(scores_2d, dim=1)  # [B, L]
            attn_weights = weights_2d.detach().unsqueeze(-1)  # [B, L, 1]

            if self.projection_mode == "identity":
                output = torch.einsum("bl,bld->bd", weights_2d, embeddings)  # [B, D]
            else:
                proj = torch.einsum(
                    "bld,jd->blj", embeddings, self.projections[0]
                )  # [B, L, d]
                output = torch.einsum("bl,bld->bd", weights_2d, proj)  # [B, d]
            return output, attn_weights

        # Multi-head path (K > 1)
        scores = torch.matmul(hidden, self.score_vectors.T)  # [B, L, K]
        scores = scores.masked_fill(~mask.unsqueeze(-1), torch.finfo(scores.dtype).min)
        attention_weights = torch.softmax(scores, dim=1)  # [B, L, K]
        attn_weights = attention_weights.detach()

        if self.projection_mode == "identity":
            proj = embeddings.unsqueeze(2).expand(B, L, K, D)  # [B, L, K, D]
        else:
            proj = torch.einsum(
                "bld,kjd->blkj", embeddings, self.projections
            )  # [B, L, K, d]

        output = torch.einsum("blk,blkd->bkd", attention_weights, proj)  # [B, K, d]
        return output.reshape(B, K * self.head_dim), attn_weights


class SingleTokenPooling(nn.Module):
    """Pool a sequence to a single fixed-size vector by selecting one token.

    Selects the embedding at the first occurrence of a designated token index
    per sequence. This is a zero-parameter pooling strategy that treats a
    specific token as a BERT-style ``[CLS]`` aggregator. For paired antibody
    sequences, the ``^`` chain separator (``AA_SEPARATOR_IDX``) is a natural
    anchor because it sits between heavy and light chains and attends to both
    through self-attention.

    Architecture:

        pooled[b] = embeddings[b, first_position_where(token_ids[b] == target_idx), :]

    Example:
        >>> from dasm2.sequence.constants import AA_SEPARATOR_IDX
        >>> pool = SingleTokenPooling(target_token_idx=AA_SEPARATOR_IDX)
        >>> embeddings = torch.randn(4, 100, 256)
        >>> token_ids = torch.randint(0, 20, (4, 100))
        >>> token_ids[:, 50] = AA_SEPARATOR_IDX  # separator at position 50 in each sequence
        >>> pooled = pool(embeddings, token_ids)  # [4, 256]
    """

    def __init__(self, target_token_idx: int) -> None:
        """Initialize single-token pooling.

        Args:
            target_token_idx: Token index to select. Every sequence in a batch
                must contain at least one occurrence of this token; the first
                occurrence is used.

        Raises:
            ValueError: If target_token_idx is negative.
        """
        super().__init__()
        if target_token_idx < 0:
            raise ValueError(
                f"target_token_idx must be non-negative, got {target_token_idx}"
            )
        self.target_token_idx = target_token_idx

    def forward(self, embeddings: Tensor, token_ids: Tensor) -> Tensor:
        """Pool by selecting the embedding at the target token's first occurrence.

        Args:
            embeddings: Per-residue embeddings [batch_size, seq_len, input_dim].
            token_ids: Token index tensor [batch_size, seq_len]. Must contain
                ``target_token_idx`` at least once per sequence.

        Returns:
            Pooled embeddings [batch_size, input_dim].

        Raises:
            ValueError: If shapes are incompatible or any sequence does not
                contain ``target_token_idx``.
        """
        if embeddings.dim() != 3:
            raise ValueError(
                f"embeddings must be 3D [B, L, D], got shape {embeddings.shape}"
            )
        if token_ids.dim() != 2:
            raise ValueError(
                f"token_ids must be 2D [B, L], got shape {token_ids.shape}"
            )
        if embeddings.shape[:2] != token_ids.shape:
            raise ValueError(
                f"Shape mismatch: embeddings {embeddings.shape[:2]} vs "
                f"token_ids {token_ids.shape}"
            )

        matches = token_ids == self.target_token_idx  # [B, L]
        has_match = matches.any(dim=1)  # [B]
        if not bool(has_match.all()):
            missing = torch.where(~has_match)[0].tolist()
            raise ValueError(
                f"Sequences at indices {missing} do not contain target token "
                f"index {self.target_token_idx}."
            )

        # First-occurrence index per sequence (argmax of a bool tensor returns
        # the first True position).
        positions = matches.int().argmax(dim=1)  # [B]
        batch_idx = torch.arange(embeddings.size(0), device=embeddings.device)
        pooled: Tensor = embeddings[batch_idx, positions, :]
        return pooled
