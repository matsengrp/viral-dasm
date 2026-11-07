"""TransformerCoreModel with netam architecture compatibility."""

import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from dasm2.config import EncoderConfig, HeadConfig
from dasm2.positional_encoding import PositionalEncoding

from .attention import (
    EntmaxMultiheadAttention,
    FlashMultiheadAttention,
    FlashRotaryMultiheadAttention,
    RotaryMultiheadAttention,
)
from .base import CoreModel
from .embeddings import (
    ConcatPosEmbeddingPipeline,
    EmbeddingResult,
    StandardEmbeddingPipeline,
)
from .segment_embedding import create_segment_embedding, get_num_segment_types
from .sf_projection import SelectionFactorProjection
from .transformer_block import (
    AttentionFactory,
    TransformerStack,
    make_softmax_attention,
)

LOG_DISTANCE_EPS = (
    1e-6  # Stability constant for log(distance + eps) in distance encoding
)


class TransformerEncoder(nn.Module):
    """Unified transformer encoder with composable embedding and attention.

    Assembles a model from config-driven components:

    - **Embedding**: ``StandardEmbeddingPipeline`` (default) or
      ``ConcatPosEmbeddingPipeline`` (when ``d_pos > 0``). The pipeline
      orchestrates the forward logic; the ``nn.Module`` layers are owned
      by this class directly (state dict key stability).
    - **Attention**: softmax, entmax, rotary, flash, or flash+rotary via
      ``_select_attention_factory``.
    - **Stack**: ``TransformerStack`` with configurable blocks.
    """

    def __init__(self, config: EncoderConfig):
        """Initialize transformer encoder from configuration.

        Args:
            config: Encoder configuration with transformer hyperparameters
        """
        super().__init__()

        self.d_model = config.d_model  # nhead * d_model_per_head
        self.nhead = config.nhead
        self.dim_feedforward = config.dim_feedforward

        # Exposed for inference batch construction (CoreModel._build_inference_batch
        # builds a layout tensor when this is set, even without segment embedding).
        self.per_segment_positions = config.per_segment_positions

        # Store config for to_config() round-trip (scalars that have no
        # runtime-derived source). Runtime-derived values (num_layers,
        # vocab_size, num_segment_types) are read from modules in to_config().
        self._config = config

        # --- Embedding layers (owned here for state dict stability) ---

        # AA embedding width depends on whether we use concat-pos subspaces
        if config.d_pos > 0:
            self.d_aa = self.d_model - config.d_pos
            self.amino_acid_embedding = nn.Embedding(config.vocab_size, self.d_aa)
            # Learned positional embedding in d_pos-dimensional subspace
            self.positional_embedding: Optional[nn.Embedding] = nn.Embedding(
                config.pe_max_len, config.d_pos
            )
        else:
            self.d_aa = self.d_model
            self.amino_acid_embedding = nn.Embedding(config.vocab_size, self.d_model)
            self.positional_embedding = None
        if config.token_embedding_init == "inverse_sqrt_width":
            # The pipelines multiply by sqrt(d_aa); start the scaled rows at PE's scale.
            with torch.no_grad():
                self.amino_acid_embedding.weight /= math.sqrt(self.d_aa)

        # Sinusoidal PE — two attribute names for state dict compatibility:
        #   pos_encoder:   used by StandardEmbeddingPipeline (existing checkpoint key)
        #   sinusoidal_pe: used by ConcatPosEmbeddingPipeline (concat-pos checkpoint key)
        # These are always mutually exclusive (at most one is non-None).
        if config.d_pos > 0 and config.add_sinusoidal:
            # Concat-pos with sinusoidal overlay
            self.sinusoidal_pe: Optional[PositionalEncoding] = PositionalEncoding(
                self.d_model, dropout=0.0
            )
            self.pos_encoder: Optional[PositionalEncoding] = None
        elif config.pos_encoding == "sinusoidal" and config.d_pos == 0:
            # Standard sinusoidal positional encoding
            self.pos_encoder = PositionalEncoding(
                self.d_model, config.dropout, max_len=config.pe_max_len
            )
            self.sinusoidal_pe = None
        else:
            # Rotary (handles position in attention) or no PE
            self.pos_encoder = None
            self.sinusoidal_pe = None

        # Dropout module for context_token distance encoding. Lives on the
        # encoder (not on a positional-encoding internal) so the pipeline does
        # not have to reach into PE state. Matches the PE dropout rate so the
        # context token is regularized at the same level as AAs.
        # nn.Dropout has no params/buffers, so this does not affect state_dict.
        self._embed_dropout = nn.Dropout(p=config.dropout)

        # Optional segment embedding
        self.segment_embedding = create_segment_embedding(
            config.num_segment_types, self.d_model
        )

        # Distance encoding for maturation stage conditioning.
        # Created before the pipeline so modules can be passed as references.
        self.distance_encoding_mode = config.distance_encoding_mode

        if self.distance_encoding_mode == "continuous":
            # bias=False so distance_proj actually conditions on log(d)
            # rather than learning a position-independent global bias
            # (issue #199, Exp 3).
            self.distance_proj = nn.Linear(1, self.d_model, bias=False)
            # Zero-init so the projection starts as a no-op: a model warm-started
            # from a no-distance checkpoint is then *identical* to its base at
            # epoch 0, and an initial branch-length optimization recovers the base
            # model's optimum. The random Kaiming default would instead inject a
            # large constant embedding shift (~|log d| per dim), biasing that
            # optimization. Gradients still flow (d/dW = dL/dout · log d ≠ 0), so
            # the projection learns normally from zero. See issue #199 frozen-BL.
            nn.init.zeros_(self.distance_proj.weight)
            self.distance_noise_std = config.distance_noise_std
        elif self.distance_encoding_mode == "binned":
            self.num_distance_bins = config.num_distance_bins
            self.distance_embedding = nn.Embedding(
                config.num_distance_bins + 1, self.d_model
            )
            self.register_buffer(
                "distance_bin_boundaries",
                torch.zeros(config.num_distance_bins - 1),
            )
        elif self.distance_encoding_mode == "context_token":
            self.num_distance_bins = config.num_distance_bins
            self.distance_context_embedding = nn.Embedding(
                config.num_distance_bins + 1, self.d_model
            )
            self.distance_unknown_embedding = nn.Embedding(1, self.d_model)
            self.distance_mask_prob = config.distance_mask_prob
            self.register_buffer(
                "distance_bin_boundaries",
                torch.zeros(config.num_distance_bins - 1),
            )

        # --- Embedding pipeline (strategy, no parameters) ---
        # Distance modules are passed as references; the pipeline applies
        # distance encoding before pos_encoder so it shares dropout.
        _distance_kwargs = self._distance_pipeline_kwargs()

        if config.d_pos > 0:
            self._embedding_pipeline: Union[
                StandardEmbeddingPipeline, ConcatPosEmbeddingPipeline
            ] = ConcatPosEmbeddingPipeline(
                amino_acid_embedding=self.amino_acid_embedding,
                d_aa=self.d_aa,
                d_pos=config.d_pos,
                positional_embedding=self.positional_embedding,
                sinusoidal_pe=self.sinusoidal_pe,
                segment_embedding=self.segment_embedding,
                per_segment_positions=config.per_segment_positions,
                **_distance_kwargs,
            )
        else:
            self._embedding_pipeline = StandardEmbeddingPipeline(
                amino_acid_embedding=self.amino_acid_embedding,
                d_model=self.d_model,
                pos_encoder=self.pos_encoder,
                segment_embedding=self.segment_embedding,
                per_segment_positions=config.per_segment_positions,
                **_distance_kwargs,
            )

        # --- Attention and transformer stack ---

        attention_factory = self._select_attention_factory(config)

        self.transformer_encoder = TransformerStack(
            d_model=self.d_model,
            num_layers=config.num_layers,
            attention_factory=attention_factory,
            activation=config.activation,
            dim_feedforward=self.dim_feedforward,
            norm_type=config.norm_type,
            norm_first=config.norm_first,
            dropout=config.dropout,
        )

    def _select_attention_factory(self, config: EncoderConfig) -> "AttentionFactory":
        """Select attention factory based on config flags.

        Priority: flash_attention + rotary > flash_attention > rotary > entmax > softmax.
        Config validation (in ``EncoderConfig._validate_field_consistency``)
        rejects invalid combinations before we get here.
        """
        if config.flash_attention and config.pos_encoding == "rotary":
            return self._make_flash_rotary_attention
        elif config.flash_attention:
            return self._make_flash_attention
        elif config.pos_encoding == "rotary":
            return self._make_rotary_attention
        elif config.entmax_alpha is not None:
            return self._make_entmax_attention
        else:
            return self._make_softmax_attention

    def _make_softmax_attention(self) -> nn.Module:
        """Create a standard softmax multi-head attention module."""
        return make_softmax_attention(self.d_model, self.nhead, self._config.dropout)

    def _make_entmax_attention(self) -> nn.Module:
        """Create an entmax multi-head attention module for sparse attention."""
        if self._config.entmax_alpha is None:
            raise ValueError(
                "_make_entmax_attention called with entmax_alpha=None; "
                "this is a bug in _select_attention_factory"
            )
        return EntmaxMultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self._config.dropout,
            alpha=self._config.entmax_alpha,
            batch_first=True,
        )

    def _make_flash_attention(self) -> nn.Module:
        """Create flash attention module using standalone flash-attn v2."""
        if not self._config.flash_attention:
            raise ValueError(
                "_make_flash_attention called with flash_attention=False; "
                "this is a bug in _select_attention_factory"
            )
        return FlashMultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self._config.dropout,
            deterministic=self._config.flash_deterministic,
        )

    def _make_flash_rotary_attention(self) -> nn.Module:
        """Create flash attention with rotary position embeddings."""
        return FlashRotaryMultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self._config.dropout,
            max_seq_len=self._config.pe_max_len,
            deterministic=self._config.flash_deterministic,
        )

    def _make_rotary_attention(self) -> nn.Module:
        """Create rotary MHA, optionally with entmax normalization."""
        return RotaryMultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self._config.dropout,
            max_seq_len=self._config.pe_max_len,
            entmax_alpha=self._config.entmax_alpha,
        )

    def set_distance_bins(self, boundaries: torch.Tensor) -> None:
        """Set quantile bin boundaries for binned or context_token modes.

        Called by the Trainer after computing quantiles from training data.
        Boundaries are stored as a buffer and persist through checkpoints.

        Args:
            boundaries: 1-D tensor of interior bin boundaries, shape
                ``[num_distance_bins - 1]``.
        """
        if self.distance_encoding_mode not in ("binned", "context_token"):
            raise ValueError(
                f"set_distance_bins called but distance_encoding_mode="
                f"'{self.distance_encoding_mode}'. Only valid for 'binned' "
                f"or 'context_token'."
            )
        assert isinstance(self.distance_bin_boundaries, torch.Tensor)
        self.distance_bin_boundaries.copy_(boundaries)

    def _distance_pipeline_kwargs(self) -> dict:
        """Build keyword arguments for passing distance modules to the pipeline.

        ``distance_bin_boundaries`` is passed as a callable (``lambda``)
        because ``register_buffer`` replaces the attribute on ``.to(device)``,
        which would leave a direct tensor reference stale on the old device.
        """
        mode = self.distance_encoding_mode
        kwargs: dict = {
            "distance_encoding_mode": mode,
            "training_fn": lambda: self.training,
        }
        if mode == "continuous":
            kwargs["distance_proj"] = self.distance_proj
            kwargs["distance_noise_std"] = self.distance_noise_std
        elif mode == "binned":
            kwargs["distance_embedding"] = self.distance_embedding
            kwargs["distance_bin_boundaries_fn"] = lambda: self.distance_bin_boundaries
        elif mode == "context_token":
            kwargs["distance_context_embedding"] = self.distance_context_embedding
            kwargs["distance_unknown_embedding"] = self.distance_unknown_embedding
            kwargs["distance_bin_boundaries_fn"] = lambda: self.distance_bin_boundaries
            kwargs["distance_mask_prob"] = self.distance_mask_prob
            kwargs["embed_dropout"] = self._embed_dropout
        return kwargs

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass: embedding pipeline (with distance encoding) → transformer.

        Distance encoding is applied inside the embedding pipeline. For
        broadcast modes (``"continuous"``, ``"binned"``) distance is added to
        AA embeddings before positional encoding, so it shares the PE
        dropout mask. For ``"context_token"`` mode the pipeline prepends a
        distance token *after* PE (so AAs keep their natural positions) and
        routes it through a dedicated ``embed_dropout`` module that runs at
        the same rate as PE dropout; the token is stripped from the output
        before returning.

        Args:
            batch: Input batch containing ``token_ids``, ``masks``, and
                optionally ``layout`` and ``distance``.

        Returns:
            Transformer representations of shape ``[batch_size, seq_len, d_model]``.
        """
        result: EmbeddingResult = self._embedding_pipeline(batch)
        embeddings, masks, context_prepended = result

        output = self.transformer_encoder(embeddings, src_key_padding_mask=~masks)

        if context_prepended:
            # Strip context token — head expects [B, L, d_model]
            output = output[:, 1:, :]

        return output  # type: ignore[no-any-return]

    def to_config(self) -> EncoderConfig:
        """Reconstruct EncoderConfig from encoder architecture.

        Uses ``self._config`` for pure scalar fields and reads runtime-derived
        values from the actual modules (source of truth after construction).

        Returns:
            EncoderConfig matching current encoder state.
        """
        # Runtime-derived values (these are the source of truth, not _config)
        num_layers = len(self.transformer_encoder.layers)
        vocab_size = self.amino_acid_embedding.num_embeddings
        num_segment_types = get_num_segment_types(self.segment_embedding)
        # Distance encoding config — read from runtime state
        mode = self.distance_encoding_mode
        distance_noise_std = 0.0
        num_distance_bins = 10
        distance_mask_prob = 0.0
        if mode == "continuous":
            distance_noise_std = getattr(self, "distance_noise_std", 0.0)
        elif mode in ("binned", "context_token"):
            num_distance_bins = self.num_distance_bins
            if mode == "context_token":
                distance_mask_prob = getattr(self, "distance_mask_prob", 0.0)

        return EncoderConfig(
            type="TransformerEncoder",
            nhead=self.nhead,
            d_model_per_head=self.d_model // self.nhead,
            num_layers=num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self._config.dropout,
            vocab_size=vocab_size,
            num_segment_types=num_segment_types,
            d_pos=self._config.d_pos,
            add_sinusoidal=self._config.add_sinusoidal,
            activation=self._config.activation,
            norm_type=self._config.norm_type,
            norm_first=self._config.norm_first,
            pos_encoding=self._config.pos_encoding,
            token_embedding_init=self._config.token_embedding_init,
            pe_max_len=self._config.pe_max_len,
            per_segment_positions=self._config.per_segment_positions,
            entmax_alpha=self._config.entmax_alpha,
            flash_attention=self._config.flash_attention,
            flash_deterministic=self._config.flash_deterministic,
            distance_encoding_mode=mode,
            distance_noise_std=distance_noise_std,
            num_distance_bins=num_distance_bins,
            distance_mask_prob=distance_mask_prob,
        )


class TransformerCoreModel(CoreModel):
    """Concrete transformer-based implementation with netam compatibility."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        heads_config: List[HeadConfig],
        sf_projection: Optional[SelectionFactorProjection] = None,
    ):
        """Initialize transformer core model with encoder and pluggable heads.

        Args:
            encoder_config: Configuration for the transformer encoder
            heads_config: List of head configurations
            sf_projection: Shared selection factor projection. Auto-created
                when SF-consuming heads are present and no explicit projection
                is provided.
        """
        # Create encoder and pass configs to base class
        encoder = TransformerEncoder(encoder_config)

        # Auto-create SF projection when SF-consuming heads need it.
        if sf_projection is None:
            from dasm2.models.factory import needs_sf_projection

            if needs_sf_projection(heads_config):
                sf_projection = SelectionFactorProjection(
                    input_dim=encoder_config.d_model,
                )

        super().__init__(
            encoder=encoder,
            prediction_heads=heads_config,
            sf_projection=sf_projection,
        )

