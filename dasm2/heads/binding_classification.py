"""Cross-attention binding-classification head (issue #801).

Data flow (one binding logit per entity-A/entity-B pair)::

    entity-A tokens [B, L_a, D_a] ─┐
                                    ├─ CrossAttention(Q=A, K/V=B)
    entity-B tokens [B, L_b, D_b] ─┘        → A' [B, L_a, attn_dim]
        → residual skip + LayerNorm
        → Conv1DBlock × n_conv_blocks    (local sequence structure)
        → MultiHeadScorePooling          (attention-pool entity-A → one vector)
        → linear                         (single binding logit)

Entity-A residues (e.g. antibody paratope), used as queries, gather context
from entity-B (e.g. antigen epitope), a small conv stack adds local sequence
structure, and ``MultiHeadScorePooling`` (MHSP) collapses the
variable-length entity-A sequence into one vector via learned per-head
attention weights over positions. The head is agnostic to how either
tensor was produced — it consumes whatever hidden states the batch
provides for entity-A (``sequence_embeddings``, from the shared encoder)
and entity-B (``binding_context_embedding``); nothing here requires either
side to come from a frozen or trainable source. A common setup is entity-A
from the trainable DASM2 encoder and entity-B from a frozen external
encoder (e.g. ESM-2), but that is a property of the training pipeline that
builds the batch, not of this head.

Ported from the architecture prototyped in
``experiments/2026-06-07-binding-classification-sabdab/scripts/binding_classifier.py``
(the ``xattn_*`` variants). ``n_conv_blocks=0`` reproduces the prototype's
"no-conv" variant; ``n_conv_blocks``/``conv_kernel`` replace what the
prototype hard-coded as five separate head classes.

Masking is threaded throughout — the entity-B key-padding mask gates the
cross-attention softmax (``CrossAttention``'s ``context_mask``), and padded
entity-A positions are zeroed after attention and after each conv block, so
padding never leaks into a real position or into the pooled output.
"""

from typing import Dict, Optional, Tuple

from torch import Tensor, nn

from dasm2.blocks.attention import CrossAttention
from dasm2.blocks.pooling import MultiHeadScorePooling
from dasm2.config import ProjectionMode
from dasm2.data.pairwise_binding import (
    BINDING_CONTEXT_EMBEDDING_KEY,
    BINDING_CONTEXT_MASK_KEY,
    entity_a_mask_from_batch,
)
from dasm2.utils.validation import validate_batch_keys

from .base import PredictionHead


class Conv1DBlock(nn.Module):
    """Depth-preserving residual Conv1D over the sequence axis (GELU + LayerNorm).

    Kept private to this head (not promoted to ``dasm2/blocks/``) — no
    second head needs it yet.
    """

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq_len, dim)."""
        residual = x
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, L, D)
        x = self.activation(x)
        x = self.norm(x + residual)
        return x


def _resolve_pool_head_dim(
    pool_head_dim: Optional[int],
    pool_projection_mode: ProjectionMode,
    attn_dim: int,
    pool_heads: int,
) -> int:
    """Resolve an unset ``pool_head_dim``.

    "identity" pooling requires head_dim == input_dim; otherwise default
    head_dim to attn_dim // pool_heads so the pooled width stays attn_dim.
    Raises rather than silently flooring to 1 when pool_heads > attn_dim,
    which would otherwise collapse pooling to a near-useless 1-dim head
    without any signal that the config is degenerate.
    """
    if pool_head_dim is not None:
        return pool_head_dim
    if pool_projection_mode == "identity":
        return attn_dim
    if pool_heads > attn_dim:
        raise ValueError(
            f"Cannot auto-resolve pool_head_dim: pool_heads ({pool_heads}) "
            f"exceeds attn_dim ({attn_dim}), which would floor head_dim to "
            "a degenerate 1. Lower pool_heads, raise attn_dim, or set "
            "pool_head_dim explicitly."
        )
    return attn_dim // pool_heads


class BindingClassificationHead(PredictionHead):
    """Cross-attention MHSP head: entity-A attends into entity-B's context.

    Agnostic to how either side's embeddings were produced — the head has
    no mechanism that inspects or requires ``requires_grad``/frozen state
    on either input. Gradients flow to entity-A's ``sequence_embeddings``
    and entity-B's ``binding_context_embedding`` alike, if either is part
    of the active autograd graph.

    Forward contract:
        ``sequence_embeddings``: entity-A per-residue embeddings from the
            shared encoder, ``[B, L_a, D_a]``.
        ``batch`` must contain:
            - ``binding_context_embedding``: ``[B, L_b, D_b]`` entity-B
              per-residue embeddings.
            - ``binding_context_mask``: ``[B, L_b]`` bool, True at real
              entity-B positions.
            - ``loss_mask`` or ``mask``: ``[B, L_a]`` bool for entity-A
              (whichever ``ComputeMasks`` produced for this batch).

    Returns a single binding logit per pair, ``[B]``.

    ``dropout`` (issue #1096 follow-up, default 0.0 — inert) is applied via
    one shared ``nn.Dropout`` module at three points: after the
    cross-attention residual+norm, after the conv stack, and on the pooled
    vector before the final linear. The 2nd and 3rd applications are
    post-masking, so dropping an already-zeroed padded position is a no-op
    either way. The 1st runs before the entity-A padding mask is applied
    (``LayerNorm``'s bias makes padded rows non-zero at that point), but
    those positions are zeroed immediately after and never reach the loss,
    so this has no effect on the result.
    """

    def __init__(
        self,
        input_dim: int,
        context_input_dim: int = 1280,
        attn_dim: int = 256,
        attn_heads: int = 8,
        conv_kernel: int = 3,
        n_conv_blocks: int = 2,
        pool_heads: int = 16,
        pool_head_dim: Optional[int] = 16,
        pool_hidden_dim: int = 64,
        pool_projection_mode: ProjectionMode = "learned",
        cross_skip: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_conv_blocks > 0 and conv_kernel % 2 == 0:
            raise ValueError(
                f"conv_kernel must be odd (padding=kernel_size // 2 only "
                f"preserves sequence length for odd kernels), got {conv_kernel}"
            )
        self.input_dim = input_dim
        self.context_input_dim = context_input_dim
        self.attn_dim = attn_dim
        self.attn_heads = attn_heads
        self.conv_kernel = conv_kernel
        self.n_conv_blocks = n_conv_blocks
        self.pool_heads = pool_heads
        self.pool_hidden_dim = pool_hidden_dim
        self.pool_projection_mode: ProjectionMode = pool_projection_mode
        self.cross_skip = cross_skip
        self.dropout = nn.Dropout(dropout)

        self.cross_attn = CrossAttention(
            query_dim=input_dim,
            context_dim=context_input_dim,
            attn_dim=attn_dim,
            num_heads=attn_heads,
        )
        # Residual skip around the cross-attention, transformer-style:
        # x = norm(query + attn(query, context)). A linear projection matches
        # widths when input_dim != attn_dim; identity when they already match.
        self.cross_skip_proj = (
            nn.Identity()
            if (input_dim == attn_dim or not cross_skip)
            else nn.Linear(input_dim, attn_dim)
        )
        self.cross_norm = nn.LayerNorm(attn_dim)
        self.conv_blocks = nn.ModuleList(
            Conv1DBlock(attn_dim, kernel_size=conv_kernel) for _ in range(n_conv_blocks)
        )

        self.pool_head_dim = _resolve_pool_head_dim(
            pool_head_dim, pool_projection_mode, attn_dim, pool_heads
        )
        self.pool = MultiHeadScorePooling(
            input_dim=attn_dim,
            num_heads=pool_heads,
            head_dim=self.pool_head_dim,
            hidden_dim=pool_hidden_dim,
            projection_mode=pool_projection_mode,
        )
        self.output_linear = nn.Linear(self.pool.output_dim, 1, bias=False)

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Compute a single binding logit per pair, ``[B]``."""
        logits, _attn = self._forward_impl(sequence_embeddings, batch)
        return logits

    def forward_with_attn(
        self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Same computation as :meth:`forward`, also returning attention
        weights for interpretability analyses (epitope-focus enrichment, MIL
        pooling-weight profiles; issue #1187).

        A separate method rather than a ``return_attn`` flag on ``forward``
        itself: widening ``forward``'s return type conditionally would
        violate ``PredictionHead.forward``'s fixed ``-> Tensor`` contract
        (mypy's ``[override]`` check catches exactly this), and
        ``CoreModel._compute_head_outputs`` always calls
        ``head(sequence_embeddings, batch)`` expecting a bare ``Tensor``
        back. Returns ``(logits, {"cross_attn": [B, H, L_a, L_b],
        "pool_attn": [B, L_a, pool_heads]})`` — the post-softmax
        cross-attention weights and the MHSP pooling weights, both already
        computed internally by ``forward`` and previously discarded.
        """
        return self._forward_impl(sequence_embeddings, batch)

    def _forward_impl(
        self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        validate_batch_keys(
            batch,
            required_keys=[BINDING_CONTEXT_EMBEDDING_KEY, BINDING_CONTEXT_MASK_KEY],
            context="BindingClassificationHead.forward()",
            dataset_hint="the binding-classification DatasetBundle (issue #802)",
        )
        entity_a_mask = entity_a_mask_from_batch(
            batch, context="BindingClassificationHead.forward()"
        )
        context_embedding = batch[BINDING_CONTEXT_EMBEDDING_KEY]
        context_mask = batch[BINDING_CONTEXT_MASK_KEY]

        # Entity-A queries gather entity-B context: padded entity-B keys
        # excluded from the softmax, padded entity-A query outputs zeroed.
        attn_out, cross_attn = self.cross_attn(
            sequence_embeddings,
            context_embedding,
            query_mask=entity_a_mask,
            context_mask=context_mask,
        )
        if self.cross_skip:
            attn_out = attn_out + self.cross_skip_proj(sequence_embeddings)
        x = self.cross_norm(attn_out)
        x = self.dropout(x)

        # Zero padded entity-A positions so the conv never mixes pad into
        # real tokens (Conv1d mixes across the sequence/neighbor axis).
        mask_f = entity_a_mask.unsqueeze(-1).to(x.dtype)  # [B, L_a, 1]
        x = x * mask_f
        for block in self.conv_blocks:
            x = block(x)
            x = x * mask_f  # re-mask after each conv's padding spillover
        x = self.dropout(x)

        pooled, pool_attn = self.pool(x, entity_a_mask)  # [B, pool_out]
        pooled = self.dropout(pooled)
        logits: Tensor = self.output_linear(pooled)  # [B, 1]
        logits = logits.squeeze(-1)  # [B]
        return logits, {"cross_attn": cross_attn, "pool_attn": pool_attn}

    def to_config(self) -> "BindingClassificationHeadConfig":
        return BindingClassificationHeadConfig(
            input_dim=self.input_dim,
            context_input_dim=self.context_input_dim,
            attn_dim=self.attn_dim,
            attn_heads=self.attn_heads,
            conv_kernel=self.conv_kernel,
            n_conv_blocks=self.n_conv_blocks,
            pool_heads=self.pool_heads,
            pool_head_dim=self.pool_head_dim,
            pool_hidden_dim=self.pool_hidden_dim,
            pool_projection_mode=self.pool_projection_mode,
            cross_skip=self.cross_skip,
            dropout=self.dropout.p,
        )


# Set config type after class definition to avoid circular import
from dasm2.config import BindingClassificationHeadConfig  # noqa: E402

BindingClassificationHead.head_config_type = BindingClassificationHeadConfig
