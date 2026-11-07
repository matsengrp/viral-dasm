"""Embedding pipeline strategies for TransformerEncoder.

Each pipeline class orchestrates the embedding forward pass — converting
``token_ids`` into a ``[batch, seq_len, d_model]`` tensor ready for the
transformer stack. Pipeline classes are **strategy objects**, not ``nn.Module``
subclasses: they receive the embedding layers (owned by ``TransformerEncoder``)
as constructor arguments and do not own parameters themselves. This preserves
state dict key stability across the migration.

Variants:

- **StandardEmbeddingPipeline**: Full ``d_model``-wide AA embedding with
  optional sinusoidal PE (or no PE for rotary).

- **ConcatPosEmbeddingPipeline**: Split-subspace embedding where
  ``d_model = d_aa + d_pos``. The AA embedding is ``d_aa``-wide and a learned
  positional embedding table is ``d_pos``-wide; the two are concatenated.
  Optionally adds sinusoidal PE across all ``d_model`` dimensions on top.

Both pipelines handle segment embedding and distance encoding at the
correct points in the pipeline.
Both also support resetting positional encoding at segment boundaries (see
:func:`dasm2.sequence.layout.per_segment_positions_of_layout`), gated by
``EncoderConfig.per_segment_positions``.

Distance encoding is off by default
(``distance_encoding_mode="none"``), in which case both pipelines behave
exactly as they did before it was added. When enabled, ordering differs by
pipeline: in ``StandardEmbeddingPipeline`` distance is applied **before**
sinusoidal PE so that it passes through the same dropout; in
``ConcatPosEmbeddingPipeline`` it is applied after sinusoidal PE (which uses
``dropout=0.0``) but before segment embedding.

Each pipeline documents its own embedding ordering in its class docstring
and ``__call__`` method.
"""

import math
from typing import Callable, Dict, NamedTuple, Optional

import torch
import torch.nn as nn

from dasm2.positional_encoding import PositionalEncoding
from dasm2.sequence.constants import AA_AMBIG_IDX, AA_SEPARATOR_IDX
from dasm2.sequence.layout import per_segment_positions_of_layout

from .segment_embedding import apply_segment_embedding

# Matches the constant in transformer.py — kept as a local reference so the
# pipeline helpers don't need to import from transformer (avoids circular deps).
_LOG_DISTANCE_EPS = 1e-6


def _distance_content_mask(
    batch: Dict[str, torch.Tensor], dtype: torch.dtype
) -> torch.Tensor:
    """Build a [B, L, 1] content mask for the distance broadcast.

    Mask is True for chain content tokens (any amino acid in any segment)
    and False for separator and padding tokens. Distance is added only at
    True positions. Naming follows the existing mask convention in the
    codebase (e.g. ``token_inclusive_mask``, ``loss_mask``) — this is the unconditional, distance-specific variant.

    Behavior across input shapes:

    - Heavy-only input: no separator → mask is True on all heavy tokens,
      False on trailing padding. Distance is added to every heavy
      position.
    - Light-only input: same as heavy-only structurally — distance is
      added to every light position. (Token-based, not layout-based, so
      the "first segment is heavy" assumption that broke the prior
      layout-gate design no longer applies.)
    - Paired ``HEAVY^LIGHT`` input: mask is True on both heavy and light
      content, False on the separator and padding. Distance is added to
      both chains. Per Erick (issue #199, 2026-04-28) distance is
      conceptually a maturation-stage scalar, applied uniformly to chain
      content rather than a chain-specific quantity.
    """
    token_ids = batch["token_ids"]
    is_content = (token_ids != AA_AMBIG_IDX) & (token_ids != AA_SEPARATOR_IDX)
    return is_content.to(dtype).unsqueeze(-1)


class EmbeddingResult(NamedTuple):
    """Return type for embedding pipelines.

    Attrs:
        embeddings: Embedding tensor, ``[B, L, d_model]`` or ``[B, L+1, d_model]``
            when a context token is prepended.
        masks: Boolean mask tensor matching the embedding sequence length.
        context_prepended: True if a context token was prepended (caller must
            strip the first position from the transformer output).
    """

    embeddings: torch.Tensor
    masks: torch.Tensor
    context_prepended: bool


# ---------------------------------------------------------------------------
# Distance encoding helpers
# ---------------------------------------------------------------------------


def _require_distance_in_batch(batch: Dict[str, torch.Tensor], mode: str) -> None:
    """Raise ValueError if 'distance' key is missing from batch."""
    if "distance" not in batch:
        raise ValueError(
            f"distance_encoding_mode='{mode}' but 'distance' key is "
            f"missing from batch. Ensure the dataset provides a "
            f"'distance' field."
        )


def _apply_continuous_distance(
    embeddings: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    distance_proj: nn.Linear,
    noise_std: float,
    is_training: bool,
) -> torch.Tensor:
    """Project log(distance) and add at chain-content positions.

    The broadcast is masked to chain-content tokens (any amino acid),
    skipping separator and padding tokens. See ``_distance_content_mask``
    for the per-input-shape behavior. Issue #199.

    Args:
        embeddings: Current embeddings ``[B, L, d_model]``.
        batch: Must contain ``distance`` and ``token_ids``.
        distance_proj: Linear projection from 1 → d_model.
        noise_std: Gaussian noise std on log(distance) during training.
        is_training: Whether the model is in training mode.

    Returns:
        Embeddings with content-masked distance encoding added.
    """
    distance = batch["distance"]  # [B]
    log_distance = torch.log(distance + _LOG_DISTANCE_EPS)
    if is_training and noise_std > 0:
        log_distance = log_distance + torch.randn_like(log_distance) * noise_std
    distance_emb = distance_proj(log_distance.unsqueeze(-1))  # [B, d_model]
    mask = _distance_content_mask(batch, embeddings.dtype)  # [B, L, 1]
    return embeddings + distance_emb.unsqueeze(1) * mask  # type: ignore[no-any-return]


def _apply_binned_distance(
    embeddings: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    distance_embedding: nn.Embedding,
    bin_boundaries_fn: Callable[[], torch.Tensor],
) -> torch.Tensor:
    """Quantize log(distance), look up embedding, add at chain-content positions.

    Content-masked like ``_apply_continuous_distance`` — see issue #199.

    Args:
        embeddings: Current embeddings ``[B, L, d_model]``.
        batch: Must contain ``distance`` and ``token_ids``.
        distance_embedding: Embedding table for bins.
        bin_boundaries_fn: Callable returning interior bin boundaries
            (deferred access to avoid stale device references).

    Returns:
        Embeddings with content-masked binned distance encoding added.
    """
    distance = batch["distance"]  # [B]
    log_distance = torch.log(distance + _LOG_DISTANCE_EPS)
    bin_indices = torch.bucketize(log_distance, bin_boundaries_fn())
    distance_emb = distance_embedding(bin_indices)  # [B, d_model]
    mask = _distance_content_mask(batch, embeddings.dtype)  # [B, L, 1]
    return embeddings + distance_emb.unsqueeze(1) * mask  # type: ignore[no-any-return]


def _prepend_context_token(
    embeddings: torch.Tensor,
    masks: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    context_embedding: nn.Embedding,
    unknown_embedding: nn.Embedding,
    bin_boundaries_fn: Callable[[], torch.Tensor],
    mask_prob: float,
    is_training: bool,
    *,
    dropout: Optional[nn.Module] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize distance, build context token, prepend to embeddings.

    The caller is responsible for ensuring positional encoding has already
    been applied to ``embeddings`` (so AAs occupy PE positions 0..L-1). The
    context token is added without a PE component and prepended at position 0.

    Args:
        embeddings: Current embeddings ``[B, L, d_model]``, expected to
            already include any positional encoding for AA positions.
        masks: Boolean masks ``[B, L]``.
        batch: Must contain ``distance`` key.
        context_embedding: Embedding table for binned distance tokens.
        unknown_embedding: Single-row embedding for masked tokens.
        bin_boundaries_fn: Callable returning interior bin boundaries.
        mask_prob: Probability of replacing with unknown during training.
        is_training: Whether the model is in training mode.
        dropout: Optional dropout module applied to the context-token
            embedding before prepending, so the context token undergoes
            the same regularization rate as AAs received via PE.

    Returns:
        ``(extended_embeddings, extended_masks)`` with shapes
        ``[B, L+1, d_model]`` and ``[B, L+1]``.
    """
    distance = batch["distance"]  # [B]
    batch_size = distance.size(0)
    device = distance.device

    log_distance = torch.log(distance + _LOG_DISTANCE_EPS)
    bin_indices = torch.bucketize(log_distance, bin_boundaries_fn())
    context_emb = context_embedding(bin_indices)  # [B, d_model]

    # Training-time masking: replace with learned unknown embedding
    if is_training and mask_prob > 0:
        mask = torch.rand(batch_size, device=device) < mask_prob
        unknown_emb = unknown_embedding.weight[0]  # [d_model]
        context_emb = torch.where(
            mask.unsqueeze(-1), unknown_emb.unsqueeze(0), context_emb
        )

    if dropout is not None:
        context_emb = dropout(context_emb)

    # Prepend context token
    extended_embeddings = torch.cat(
        [context_emb.unsqueeze(1), embeddings], dim=1
    )  # [B, L+1, d_model]

    # Extend masks: context token is always visible
    context_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
    extended_masks = torch.cat([context_mask, masks], dim=1)  # [B, L+1]

    return extended_embeddings, extended_masks


def _per_segment_positions_of_batch(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Derive per-segment reset positions from a batch's layout tensor.

    Fail-fast: raises if ``layout`` is missing, mirroring
    :func:`apply_segment_embedding`. The layout tensor is always produced by
    the data pipeline and, at inference, whenever an encoder uses per-segment
    positions (see ``CoreModel._build_inference_batch``).

    Args:
        batch: Input batch; must contain ``layout``.

    Returns:
        Long tensor of shape ``[batch_size, seq_len]`` of per-segment positions.

    Raises:
        ValueError: If ``layout`` is missing from the batch.
    """
    if "layout" not in batch:
        raise ValueError(
            "Encoder is configured with per_segment_positions but batch is "
            "missing 'layout' key. Ensure your dataset provides layout tensors."
        )
    return per_segment_positions_of_layout(batch["layout"])


class StandardEmbeddingPipeline:
    """Standard embedding pipeline: AA embed + segment + distance + PE.

    Ordering depends on distance mode:

    - For ``"none"``, ``"continuous"``, ``"binned"``: AA embed → segment
      → **distance broadcast** → sinusoidal PE
      (includes dropout). Distance is added before PE so it shares the
      AA dropout mask.
    - For ``"context_token"``: AA embed → segment → sinusoidal PE
      (drops AAs at chain positions 0..L-1) → **prepend distance token**
      (run through a dedicated ``embed_dropout`` matched to the PE rate) →
      done. The context token receives no PE component, so AAs keep their
      true positions.

    Args:
        amino_acid_embedding: AA token embedding layer (d_model-wide).
        d_model: Model dimension (for sqrt scaling).
        pos_encoder: Optional sinusoidal positional encoding (None for rotary).
        segment_embedding: Optional segment embedding layer.
        per_segment_positions: If True, gather sinusoidal PE at per-segment
            reset positions (derived from ``layout``) instead of absolute
            positions.
        distance_encoding_mode: One of ``"none"``, ``"continuous"``,
            ``"binned"``, or ``"context_token"``.
        distance_proj: ``nn.Linear(1, d_model)`` for continuous mode.
        distance_noise_std: Gaussian noise std for continuous mode.
        distance_embedding: ``nn.Embedding`` for binned mode.
        distance_bin_boundaries_fn: Callable returning the bin boundaries
            tensor. Deferred access avoids stale device references after
            ``.to(device)`` on the parent module.
        distance_context_embedding: ``nn.Embedding`` for context_token mode.
        distance_unknown_embedding: ``nn.Embedding(1, d_model)`` for masking.
        distance_mask_prob: Masking probability for context_token mode.
        embed_dropout: Optional ``nn.Dropout`` module owned by the encoder,
            applied to the context-token embedding in ``"context_token"``
            mode. Decouples the pipeline from PE internals.
        training_fn: Callable returning the parent module's ``.training`` flag.
    """

    def __init__(
        self,
        amino_acid_embedding: nn.Embedding,
        d_model: int,
        pos_encoder: Optional[PositionalEncoding] = None,
        segment_embedding: Optional[nn.Embedding] = None,
        per_segment_positions: bool = False,
        *,
        distance_encoding_mode: str = "none",
        distance_proj: Optional[nn.Linear] = None,
        distance_noise_std: float = 0.0,
        distance_embedding: Optional[nn.Embedding] = None,
        distance_bin_boundaries_fn: Optional[Callable[[], torch.Tensor]] = None,
        distance_context_embedding: Optional[nn.Embedding] = None,
        distance_unknown_embedding: Optional[nn.Embedding] = None,
        distance_mask_prob: float = 0.0,
        embed_dropout: Optional[nn.Module] = None,
        training_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.amino_acid_embedding = amino_acid_embedding
        self.d_model = d_model
        self.pos_encoder = pos_encoder
        self.segment_embedding = segment_embedding
        self.per_segment_positions = per_segment_positions
        # Distance encoding state (references to modules owned by TransformerEncoder)
        self._distance_mode = distance_encoding_mode
        self._distance_proj = distance_proj
        self._distance_noise_std = distance_noise_std
        self._distance_embedding = distance_embedding
        self._distance_bin_boundaries_fn = distance_bin_boundaries_fn
        self._distance_context_embedding = distance_context_embedding
        self._distance_unknown_embedding = distance_unknown_embedding
        self._distance_mask_prob = distance_mask_prob
        self._embed_dropout = embed_dropout
        self._training_fn = training_fn

    def _is_training(self) -> bool:
        return self._training_fn() if self._training_fn is not None else False

    def __call__(self, batch: Dict[str, torch.Tensor]) -> EmbeddingResult:
        """Build embeddings from token IDs.

        See class docstring for the mode-dependent ordering. PE is applied
        to AA positions only; the distance context token (if any) is
        prepended *after* PE with its own dropout, so AAs keep PE
        positions 0..L-1.

        Args:
            batch: Must contain ``token_ids`` and ``masks``. May contain
                ``layout`` (required if segment_embedding is configured or
                per_segment_positions is set), and ``distance`` (required if
                distance_encoding_mode is not ``"none"``).

        Returns:
            ``EmbeddingResult`` with embeddings, masks, and context flag.
            ``masks`` is ``batch["masks"]`` unchanged and ``context_prepended``
            is False except in ``"context_token"`` mode.

        Raises:
                is missing, per_segment_positions is set but ``layout`` is
                missing, or distance encoding is enabled but ``distance`` is
                missing from batch.
        """
        token_ids = batch["token_ids"]
        masks = batch["masks"]

        embeddings = self.amino_acid_embedding(token_ids) * math.sqrt(self.d_model)
        embeddings = apply_segment_embedding(self.segment_embedding, embeddings, batch)

        mode = self._distance_mode
        context_prepended = False

        # Distance dispatch — order matters; the three blocks below run in
        # sequence for every forward call:
        #   1. Broadcast modes (continuous, binned): add to AAs *before* PE.
        #   2. PE on AAs only (context token, if any, is added later).
        #   3. context_token mode: prepend distance token *after* PE.
        # If you add a new broadcast mode, do it in block (1) so it shares the
        # PE dropout mask; if you add a new prepend mode, do it in block (3).
        if mode == "continuous":
            _require_distance_in_batch(batch, mode)
            assert self._distance_proj is not None
            embeddings = _apply_continuous_distance(
                embeddings,
                batch,
                self._distance_proj,
                self._distance_noise_std,
                self._is_training(),
            )
        elif mode == "binned":
            _require_distance_in_batch(batch, mode)
            assert self._distance_embedding is not None
            assert self._distance_bin_boundaries_fn is not None
            embeddings = _apply_binned_distance(
                embeddings,
                batch,
                self._distance_embedding,
                self._distance_bin_boundaries_fn,
            )

        # Block (2) — positional encoding (includes its own dropout for
        # standard sinusoidal PE). In context_token mode the context token
        # has not been prepended yet, so AAs occupy PE positions 0..L-1 —
        # i.e., their true chain positions.
        if self.pos_encoder is not None:
            if self.per_segment_positions:
                positions = _per_segment_positions_of_batch(batch)
                embeddings = self.pos_encoder.add_at_positions(embeddings, positions)
            else:
                embeddings = self.pos_encoder(embeddings.permute(1, 0, 2)).permute(
                    1, 0, 2
                )

        # Block (3) — context_token: prepend after PE so the context token
        # has no PE component, and route it through the encoder-owned
        # ``embed_dropout`` so it is regularized at the same rate as AAs
        # (independent draw — dropout masks are sampled per forward call).
        if mode == "context_token":
            _require_distance_in_batch(batch, mode)
            assert self._distance_context_embedding is not None
            assert self._distance_unknown_embedding is not None
            assert self._distance_bin_boundaries_fn is not None
            embeddings, masks = _prepend_context_token(
                embeddings,
                masks,
                batch,
                self._distance_context_embedding,
                self._distance_unknown_embedding,
                self._distance_bin_boundaries_fn,
                self._distance_mask_prob,
                self._is_training(),
                dropout=self._embed_dropout,
            )
            context_prepended = True

        return EmbeddingResult(
            embeddings=embeddings, masks=masks, context_prepended=context_prepended
        )


class ConcatPosEmbeddingPipeline:
    """Split-subspace embedding: AA(d_aa) | learned_pos(d_pos) + optional sinusoidal.

    Instead of adding positional encoding to a full ``d_model``-wide AA embedding,
    this pipeline concatenates a learned positional embedding into a dedicated
    subspace::

        embedding = concat(aa_embed[token], pos_embed[position])

    The total dimension is ``d_model = d_aa + d_pos``, where:

    - ``d_aa``: dimensions for the scaled AA embedding
    - ``d_pos``: dimensions for the learned positional embedding

    Optionally, sinusoidal PE can be added across all ``d_model`` dimensions
    on top of the concatenated embedding (controlled by ``add_sinusoidal``).

    Distance encoding is applied after the concat+sinusoidal step, before
    segment embedding. For ``"context_token"`` mode the prepended token is
    routed through the encoder-owned ``embed_dropout`` (matched to the
    encoder ``dropout`` rate) so it is regularized even though ConcatPos's
    own ``sinusoidal_pe`` uses ``dropout=0.0``.

    Args:
        amino_acid_embedding: AA token embedding layer (d_aa-wide, NOT d_model).
        d_aa: AA embedding dimension.
        d_pos: Positional embedding dimension.
        positional_embedding: Optional learned positional embedding table.
        sinusoidal_pe: Optional sinusoidal PE across full d_model.
        segment_embedding: Optional segment embedding layer.
        per_segment_positions: If True, index the learned positional embedding
            (and any sinusoidal overlay) at per-segment reset positions derived
            from ``layout`` instead of absolute positions.
        distance_encoding_mode: Distance encoding mode string.
        distance_proj: Linear projection for continuous mode.
        distance_noise_std: Gaussian noise std for continuous mode.
        distance_embedding: Embedding table for binned mode.
        distance_bin_boundaries_fn: Callable returning bin boundaries tensor.
        distance_context_embedding: Embedding for context_token mode.
        distance_unknown_embedding: Single-row embedding for masking.
        distance_mask_prob: Masking probability for context_token mode.
        embed_dropout: Optional ``nn.Dropout`` module owned by the encoder,
            applied to the context-token embedding in ``"context_token"``
            mode. Decouples the pipeline from PE internals.
        training_fn: Callable returning the parent module's ``.training`` flag.
    """

    def __init__(
        self,
        amino_acid_embedding: nn.Embedding,
        d_aa: int,
        d_pos: int,
        positional_embedding: Optional[nn.Embedding] = None,
        sinusoidal_pe: Optional[PositionalEncoding] = None,
        segment_embedding: Optional[nn.Embedding] = None,
        per_segment_positions: bool = False,
        *,
        distance_encoding_mode: str = "none",
        distance_proj: Optional[nn.Linear] = None,
        distance_noise_std: float = 0.0,
        distance_embedding: Optional[nn.Embedding] = None,
        distance_bin_boundaries_fn: Optional[Callable[[], torch.Tensor]] = None,
        distance_context_embedding: Optional[nn.Embedding] = None,
        distance_unknown_embedding: Optional[nn.Embedding] = None,
        distance_mask_prob: float = 0.0,
        embed_dropout: Optional[nn.Module] = None,
        training_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.amino_acid_embedding = amino_acid_embedding
        self.d_aa = d_aa
        self.d_pos = d_pos
        self.positional_embedding = positional_embedding
        self.sinusoidal_pe = sinusoidal_pe
        self.segment_embedding = segment_embedding
        self.per_segment_positions = per_segment_positions
        # Distance encoding state
        self._distance_mode = distance_encoding_mode
        self._distance_proj = distance_proj
        self._distance_noise_std = distance_noise_std
        self._distance_embedding = distance_embedding
        self._distance_bin_boundaries_fn = distance_bin_boundaries_fn
        self._distance_context_embedding = distance_context_embedding
        self._distance_unknown_embedding = distance_unknown_embedding
        self._distance_mask_prob = distance_mask_prob
        self._embed_dropout = embed_dropout
        self._training_fn = training_fn

    def _is_training(self) -> bool:
        return self._training_fn() if self._training_fn is not None else False

    def __call__(self, batch: Dict[str, torch.Tensor]) -> EmbeddingResult:
        """Build split-subspace embeddings from token IDs.

        Ordering: AA embed(d_aa) → concat with
        pos_embed(d_pos) → sinusoidal PE(d_model) → distance → segment(d_model).

        Args:
            batch: Must contain ``token_ids`` and ``masks``.

        Returns:
            ``EmbeddingResult`` with embeddings, masks, and context flag.
        """
        token_ids = batch["token_ids"]
        masks = batch["masks"]

        # AA subspace: scale by sqrt(d_aa), then optional 3Di in AA subspace
        aa_emb = self.amino_acid_embedding(token_ids) * math.sqrt(self.d_aa)

        # Per-segment reset positions ([batch, seq]) or absolute positions
        # ([seq], broadcast over batch). Computed once and shared by the learned
        # positional embedding and the optional sinusoidal overlay.
        if self.per_segment_positions:
            positions = _per_segment_positions_of_batch(batch)  # [batch, seq]
        else:
            positions = torch.arange(token_ids.size(1), device=token_ids.device)

        if self.positional_embedding is not None:
            # Positional subspace: no sqrt scaling (experimental — see #340)
            pos_emb = self.positional_embedding(positions)  # [..., seq, d_pos]
            if not self.per_segment_positions:
                pos_emb = pos_emb.unsqueeze(0).expand(token_ids.size(0), -1, -1)
            embeddings = torch.cat([aa_emb, pos_emb], dim=-1)  # [batch, seq, d_model]
        else:
            embeddings = aa_emb

        # Optional sinusoidal PE across all d_model dimensions
        if self.sinusoidal_pe is not None:
            if self.per_segment_positions:
                embeddings = self.sinusoidal_pe.add_at_positions(embeddings, positions)
            else:
                embeddings = self.sinusoidal_pe(embeddings.permute(1, 0, 2)).permute(
                    1, 0, 2
                )

        # Distance encoding — applied after sinusoidal PE, before segment
        mode = self._distance_mode
        context_prepended = False

        if mode == "continuous":
            _require_distance_in_batch(batch, mode)
            assert self._distance_proj is not None
            embeddings = _apply_continuous_distance(
                embeddings,
                batch,
                self._distance_proj,
                self._distance_noise_std,
                self._is_training(),
            )
        elif mode == "binned":
            _require_distance_in_batch(batch, mode)
            assert self._distance_embedding is not None
            assert self._distance_bin_boundaries_fn is not None
            embeddings = _apply_binned_distance(
                embeddings,
                batch,
                self._distance_embedding,
                self._distance_bin_boundaries_fn,
            )
        elif mode == "context_token":
            _require_distance_in_batch(batch, mode)
            assert self._distance_context_embedding is not None
            assert self._distance_unknown_embedding is not None
            assert self._distance_bin_boundaries_fn is not None
            embeddings, masks = _prepend_context_token(
                embeddings,
                masks,
                batch,
                self._distance_context_embedding,
                self._distance_unknown_embedding,
                self._distance_bin_boundaries_fn,
                self._distance_mask_prob,
                self._is_training(),
                dropout=self._embed_dropout,
            )
            context_prepended = True

        embeddings = apply_segment_embedding(self.segment_embedding, embeddings, batch)

        return EmbeddingResult(
            embeddings=embeddings, masks=masks, context_prepended=context_prepended
        )
