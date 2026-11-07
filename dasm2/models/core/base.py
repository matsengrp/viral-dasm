"""Abstract CoreModel base class for DASM2."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, cast

if TYPE_CHECKING:
    from dasm2.config import ModelConfig

import torch
import torch.nn as nn

from dasm2.config import HeadConfig
from dasm2.heads.base import PredictionHead
from dasm2.heads.types import HeadType
from dasm2.models.core.segment_embedding import get_num_segment_types
from dasm2.models.core.sf_projection import (
    SelectionFactorProjection,
    resident_relative_log_phi,
)
from dasm2.sequence.constants import AA_AMBIG_IDX, AA_SEPARATOR_IDX
from dasm2.sequence.layout import layout_of_aa_idxs
from dasm2.sequence.masking import split_predictions_by_sequences
from dasm2.tokenizers import AA_TOKENIZER
from dasm2.utils.tensor import zap_predictions_along_diagonal

# Median maturation distance in the v1tangCC training data. Used as the default
# inference distance when none is provided — distance=0 lies far below the
# training distribution (log(0+eps) is a large-negative outlier) and is not a
# representative inference point.
DEFAULT_INFERENCE_DISTANCE = 0.062


class CoreModel(nn.Module):
    """Core model with encoder and pluggable heads for multi-task learning."""

    _device_anchor: torch.Tensor

    def __init__(
        self,
        encoder: nn.Module,
        prediction_heads: Optional[List[HeadConfig]] = None,
        sf_projection: Optional[SelectionFactorProjection] = None,
    ) -> None:
        super().__init__()
        # Moves with ``.to(device)`` like any buffer, so ``device`` works for
        # a model with no parameters (``IdentityEncoder`` + ``NeutralHead``).
        # Empty and non-persistent: nothing enters ``state_dict()``.
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.encoder = encoder
        self.sf_projection = sf_projection
        # Pluggable heads registry
        self.prediction_heads = nn.ModuleDict()
        if prediction_heads:
            self._add_heads_from_configs(prediction_heads)
        self._validate_sf_projection()
        # Populated by ``build_task_loss_module_dict`` when tasks are
        # registered (trainer / serialization paths). Stays ``None`` for
        # fresh inference-only models. Declared here so mypy sees the slot
        # and the parameters fold cleanly under ``model.state_dict()``.
        self.task_losses: Optional[nn.ModuleDict] = None
        # Tracks whether *this* generic freeze switch was engaged, as
        # opposed to inspecting ``requires_grad`` directly. The two are not
        # interchangeable: some encoders freeze themselves at construction
        # time independent of this switch (e.g. ``ESM2Encoder`` honors
        # ``ESM2EncoderConfig.freeze_parameters``, which defaults to
        # ``True``), so a bare ``all(not p.requires_grad ...)`` check can't
        # tell "this run asked for head-only fine-tuning" apart from "this
        # encoder always freezes itself" (issue #1076).
        self._encoder_frozen_via_freeze_call = False

    def _validate_sf_projection(self) -> None:
        """Validate sf_projection is present when SF-consuming heads exist.

        Raises:
            ValueError: If any head has consumes_shared_sf=True but
                sf_projection is None.
        """
        for head_key, head in self.prediction_heads.items():
            if head.consumes_shared_sf and self.sf_projection is None:
                raise ValueError(
                    f"Head '{head_key}' has consumes_shared_sf=True but "
                    f"no SelectionFactorProjection was provided to CoreModel. "
                    f"Pass sf_projection to CoreModel.__init__()."
                )

    def _build_encoder_input(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Build encoder input dict from batch fields.

        Constructs the input dict expected by the encoder, using
        token-inclusive masking. Subclasses can override to add
        extra fields (e.g., 3Di token IDs, MPNN features).

        Args:
            batch: Input batch containing aa_parents_idxs and mask.

        Returns:
            Dict with at least token_ids, masks, and optionally layout.
        """
        # The encoder also attends to "^" separators, which batch["mask"] excludes.
        token_inclusive_mask = batch["mask"] | (
            batch["aa_parents_idxs"] == AA_SEPARATOR_IDX
        )
        encoder_input: Dict[str, torch.Tensor] = {
            "token_ids": batch["aa_parents_idxs"],
            "masks": token_inclusive_mask,
        }
        if "layout" in batch:
            encoder_input["layout"] = batch["layout"]
        if "distance" in batch:
            encoder_input["distance"] = batch["distance"]
        return encoder_input

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        head_names: Optional[List[HeadType]] = None,
    ) -> Dict:
        """Forward pass with optional head selection using token-inclusive masking.

        Args:
            batch: Input batch with required fields:
                - aa_parents_idxs: Parent amino acid indices [batch_size, seq_len]
                - mask: Valid position indicators [batch_size, seq_len]
                Additional data depends on head requirements (e.g., targets for loss computation)
            head_names: List of HeadType enums to compute, defaults to all active heads

        Returns:
            Dict mapping HeadType enums to their outputs, plus "sequence_embeddings" key

        Note:
            Uses token-inclusive masking strategy which includes special tokens
            (like separators) in the mask to ensure correct position handling.
            This combines the original mask with separator/special token positions.

            For ESM2 models, use ESM2CoreModel which handles ESM2-specific inputs.
        """
        outputs = {}

        # Validate required fields
        if "aa_parents_idxs" not in batch:
            raise ValueError("Batch must contain 'aa_parents_idxs' field")
        if "mask" not in batch:
            raise ValueError("Batch must contain 'mask' field")

        encoder_input = self._build_encoder_input(batch)
        sequence_embeddings = self.encoder(encoder_input)

        outputs["sequence_embeddings"] = sequence_embeddings

        self._compute_head_outputs(outputs, sequence_embeddings, batch, head_names)

        return outputs

    def _compute_head_outputs(
        self,
        outputs: Dict,
        sequence_embeddings: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        head_names: Optional[List[HeadType]] = None,
    ) -> None:
        """Compute predictions for requested heads and store in outputs dict.

        When SF-consuming heads are active, computes the shared selection
        factor projection once and passes it to those heads (identity forward).
        Other heads receive sequence_embeddings directly.

        Args:
            outputs: Dict to populate with head outputs (mutated in place).
            sequence_embeddings: Encoder output [batch, seq_len, d_model].
            batch: Full input batch (heads may read additional fields).
            head_names: Heads to compute; defaults to all active heads.
        """
        active_heads = head_names or [
            HeadType(key) for key in self.prediction_heads.keys()
        ]

        # Resolve each requested head_type to its actual ``prediction_heads``
        # dict key. Each ``HeadType`` value is the dict key directly.
        active_pairs = [
            (ht, ht.value if ht.value in self.prediction_heads else None)
            for ht in active_heads
        ]

        # Compute shared SF once if any active head consumes it
        sf_zapped = None
        if self.sf_projection is not None:
            needs_sf = any(
                self.prediction_heads[key].consumes_shared_sf
                for _, key in active_pairs
                if key is not None
            )
            if needs_sf:
                projected = self.sf_projection(sequence_embeddings)
                if self.sf_projection.link == "phi":
                    # The projection is the site fitness F; log f =
                    # log phi(F - F_parent), already 0 in the parent column
                    # before the zap.
                    outputs["site_fitness"] = projected
                    log_f = resident_relative_log_phi(
                        projected.float(), batch["aa_parents_idxs"]
                    )
                else:
                    log_f = projected
                # ``outputs[HeadType.SELECTION_FACTOR_HEAD]`` would carry the
                # same zapped tensor (the head is identity), so the canonical
                # zapped-SF lookup is the per-head entry. Only the pre-zap
                # log f needs its own string key.
                outputs["raw_selection_factors"] = log_f
                sf_zapped = zap_predictions_along_diagonal(
                    log_f.clone(), batch["aa_parents_idxs"], fill=0.0
                )

        for head_type, head_key in active_pairs:
            if head_key is not None:
                head = self.prediction_heads[head_key]
                if head.consumes_shared_sf:
                    # Shared-SF heads receive parent-gauged ``sf_zapped``
                    # and return it via an identity forward, so the
                    # output has ``log_f[parent] = 0``.
                    outputs[head_type] = head(sf_zapped, batch)
                else:
                    # Non-shared-SF heads receive raw embeddings. Almost
                    # every per-site head (``QDASMEvolHead``,
                    # ``HybridEvolHead``, ``SingleMatrixHead``,
                    # ``PerSite*Head``, ...) zap-gauges inside its own
                    # forward via ``selection_factor_forward``.
                    # ``SingleValueHead`` is the only un-gauged exception;
                    # see the note in
                    # ``dasm2.analysis.rate_extraction.extract_selection_factors_single``.
                    outputs[head_type] = head(sequence_embeddings, batch)

    def add_prediction_head(self, head_type: HeadType, head: PredictionHead) -> None:
        """Add a new prediction head to the model.

        Args:
            head_type: HeadType enum identifying the head
            head: Initialized prediction head instance
        """
        self.prediction_heads[head_type.value] = head
        self._validate_sf_projection()

    def add_prediction_heads(self, heads: Dict[HeadType, PredictionHead]) -> None:
        """Add multiple prediction heads to the model.

        Args:
            heads: Dictionary mapping HeadType enums to prediction head instances
        """
        for head_type, head in heads.items():
            self.prediction_heads[head_type.value] = head
        self._validate_sf_projection()

    def _add_heads_from_configs(self, heads: List[HeadConfig]) -> None:
        """Add heads from config objects.

        Args:
            heads: List of HeadConfig instances

        Raises:
            ValueError: If list contains duplicate head types
        """
        # Check for duplicates and convert to dict
        heads_dict: Dict[HeadType, HeadConfig] = {}
        for head_config in heads:
            if head_config.head_type in heads_dict:
                raise ValueError(
                    f"Duplicate head type '{head_config.head_type.value}' found. "
                    "Each head type can only appear once."
                )
            heads_dict[head_config.head_type] = head_config

        # Function-local: dasm2.heads.factory imports dasm2.models (#1294).
        from dasm2.heads.factory import create_head

        # Construct heads from configs
        constructed_heads = {
            head_type: create_head(head_config)
            for head_type, head_config in heads_dict.items()
        }
        self.add_prediction_heads(constructed_heads)

    def require_head(self, head_type: HeadType) -> PredictionHead:
        """Get prediction head by type or raise error.

        Args:
            head_type: HeadType enum identifying the head (e.g., HeadType.SELECTION_FACTOR_HEAD)

        Returns:
            The prediction head instance

        Raises:
            ValueError: If head not found, with list of available heads

        Example:
            >>> from dasm2.heads.types import HeadType
            >>> sf_head = model.require_head(HeadType.SELECTION_FACTOR_HEAD)
            >>> predictions = sf_head(embeddings, batch)
        """
        head_key = head_type.value
        if head_key not in self.prediction_heads:
            available = list(self.prediction_heads.keys())
            raise ValueError(
                f"No head of type '{head_type.value}'. Available heads: {available}"
            )
        return cast(PredictionHead, self.prediction_heads[head_key])

    def has_head(self, head_type: HeadType) -> bool:
        """Check if model has a specific prediction head.

        Args:
            head_type: HeadType enum identifying the head

        Returns:
            True if head exists, False otherwise
        """
        return head_type.value in self.prediction_heads

    def get_or_create_head(
        self,
        head_type: HeadType,
        head_config: HeadConfig,
    ) -> PredictionHead:
        """Get existing head or create new one if it doesn't exist.

        This enables head sharing across multiple tasks (e.g., for gauge
        coupling between Whichmut and AAO losses).

        Args:
            head_type: Type of head to get/create
            head_config: Configuration for creating new head. **Ignored if
                head already exists** - this is intentional for head sharing.

        Returns:
            Head instance (either existing or newly created)

        Note:
            When the head already exists, head_config is silently ignored.
            This enables head sharing: the first task to request a head type
            creates it with its config, and subsequent tasks reuse that
            same head instance regardless of their config. This is the
            desired behavior for gauge coupling, where multiple datasets
            train the same MLP.

            If you need to verify configs match, add validation at the
            trainer level before calling this method.
        """
        if head_type in self.prediction_heads:
            # Head exists - return existing instance (config ignored)
            return cast(PredictionHead, self.prediction_heads[head_type.value])

        from dasm2.heads.factory import create_head  # function-local: see #1294

        head = create_head(head_config)
        self.prediction_heads[head_type.value] = head
        return head

    def prepare_training_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Adjust a training batch before the trainer counts its loss labels.

        The trainer calls this on every training batch before it counts the
        batch's ``loss_mask`` for loss normalization, and before the forward.
        A model that chooses its loss sites itself (``SiteMaskedCoreModel``)
        draws them here, so the count and the loss see the same sites. The
        default does nothing.

        An override must be idempotent: the trainer calls it before the
        count, and a model's own ``forward`` may call it again on the same
        batch, as may a NaN-gradient replay of the window.
        """

    def freeze_encoder(self) -> None:
        """Freeze encoder for head-only training."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self._encoder_frozen_via_freeze_call = True

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder for multi-task training."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self._encoder_frozen_via_freeze_call = False

    @property
    def encoder_frozen_via_freeze_call(self) -> bool:
        """Whether ``freeze_encoder()`` is the reason the encoder is frozen.

        Unlike checking ``requires_grad`` directly, this can't be confused
        by an encoder that freezes its own parameters for reasons unrelated
        to this switch (see the note in ``__init__``). Consulted by
        ``Trainer.export_metadata`` to decide ``TrainingConfig.freeze_encoder``
        (issue #1076).
        """
        return self._encoder_frozen_via_freeze_call

    def freeze_head(self, head_type: HeadType) -> None:
        """Freeze a specific prediction head.

        Args:
            head_type: HeadType enum identifying the head to freeze
        """
        head_key = head_type.value
        if head_key in self.prediction_heads:
            for param in self.prediction_heads[head_key].parameters():
                param.requires_grad = False

    def unfreeze_head(self, head_type: HeadType) -> None:
        """Unfreeze a specific prediction head.

        Args:
            head_type: HeadType enum identifying the head to unfreeze
        """
        head_key = head_type.value
        if head_key in self.prediction_heads:
            for param in self.prediction_heads[head_key].parameters():
                param.requires_grad = True

    def to_config(self) -> "ModelConfig":
        """Reconstruct ModelConfig from current model architecture.

        Returns:
            ModelConfig representing the encoder and prediction heads

        Note:
            This enables stateless serialization - configs are reconstructed
            from runtime objects rather than stored redundantly.
        """
        from dasm2.config import ModelConfig, SelectionFactorProjectionConfig

        # Reconstruct encoder config
        encoder_config = self.encoder.to_config()  # type: ignore[operator]

        # Reconstruct head configs as list
        head_configs: List[HeadConfig] = []
        for head_key, head in self.prediction_heads.items():
            head_type = HeadType(head_key)  # Convert string back to enum
            config = head.to_config()  # type: ignore[operator]
            # Ensure head_type is set in the config
            config.head_type = head_type
            head_configs.append(config)

        # Reconstruct SF projection config if present
        sf_projection_config = None
        if self.sf_projection is not None:
            sf_projection_config = SelectionFactorProjectionConfig(
                input_dim=self.sf_projection.input_dim,
                num_amino_acids=self.sf_projection.num_amino_acids,
                wiggle_beta=self.sf_projection.wiggle_beta,
                link=self.sf_projection.link,
            )

        return ModelConfig(
            encoder=encoder_config,
            heads=head_configs,
            sf_projection=sf_projection_config,
        )

    @property
    def device(self) -> torch.device:
        """The device the model lives on, including a parameter-free model."""
        return self._device_anchor.device

    def _build_inference_batch(
        self,
        aa_sequence: str,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Build a batch dict for single-sequence inference.

        Tokenizes the input, constructs the mask, and adds layout tensors
        when the encoder uses segment embeddings.

        Args:
            aa_sequence: Unified sequence string ("SEQ1^SEQ2" or single).

        Returns:
            (batch, tokens_1d) where ``batch`` is ready for ``forward()``
            and ``tokens_1d`` is the [seq_len] token tensor used for
            post-processing (splitting by segments).
        """
        tokens = AA_TOKENIZER.tokenize([aa_sequence]).to(self.device)  # [1, seq_len]
        tokens_1d = tokens.squeeze(0)  # [seq_len]

        batch: Dict[str, torch.Tensor] = {
            "aa_parents_idxs": tokens,
            "mask": tokens != AA_AMBIG_IDX,
        }

        # Construct layout tensor when the encoder uses segment embedding OR
        # per-segment positional encoding — either feature needs layout.
        # getattr is intentional: TransformerEncoder has segment_embedding (possibly
        # None), but MPNNLightAttentionEncoder does not have the attribute at all.
        segment_embedding = getattr(self.encoder, "segment_embedding", None)
        num_segment_types = get_num_segment_types(segment_embedding)
        per_segment_positions = getattr(self.encoder, "per_segment_positions", False)
        if num_segment_types is not None or per_segment_positions:
            num_segments = int((tokens == AA_SEPARATOR_IDX).sum().item()) + 1
            # Segment-count validation only applies when a segment embedding
            # exists (its table size is fixed to a specific segment count).
            if num_segment_types is not None:
                expected_num_segment_types = num_segments + 2  # +separator +padding
                if expected_num_segment_types != num_segment_types:
                    raise ValueError(
                        f"Input has {num_segments} segment(s) "
                        f"(num_segment_types={expected_num_segment_types}) but model "
                        f"is configured with num_segment_types={num_segment_types}. "
                        f"Check that your input matches the model's segment "
                        f"configuration."
                    )
            batch["layout"] = layout_of_aa_idxs(tokens, num_segments).to(self.device)

        return batch, tokens_1d

    def _run_inference(
        self,
        batch: Dict[str, torch.Tensor],
        tokens_1d: torch.Tensor,
        head_type: HeadType,
        raw: bool = False,
    ) -> List[torch.Tensor]:
        """Run forward pass, exponentiate, and split by segments.

        Args:
            batch: Batch dict ready for ``forward()``.
            tokens_1d: [seq_len] token tensor for splitting results by segment.
            head_type: HeadType enum for the head to compute.
            raw: If True, return un-zapped selection factors from
                ``outputs["raw_selection_factors"]`` instead of the
                gauge-zapped head output.

        Returns:
            List of CPU selection factor tensors, one per sequence segment.
        """
        with torch.no_grad():
            outputs = self.forward(batch, head_names=[head_type])
            if raw:
                head_output = outputs["raw_selection_factors"]
            else:
                head_output = outputs[head_type]
            result = torch.exp(head_output).squeeze(0)
            filtered_and_split_result = split_predictions_by_sequences(
                result, tokens_1d
            )
        return [tensor.cpu() for tensor in filtered_and_split_result]

    def selection_factors_of_aa_str(
        self,
        aa_sequence: str,
        head_type: HeadType = HeadType.SELECTION_FACTOR_HEAD,
        distance: Optional[float] = None,
        raw: bool = False,
    ) -> List[torch.Tensor]:
        """DASM2 equivalent to netam's selection_factors_of_aa_str method.

        Routes through ``self.forward()`` so that encoder masking is identical
        to training (token-inclusive mask construction, layout, etc.).

        Args:
            aa_sequence: Unified sequence string ("SEQ1^SEQ2" or single sequence)
            head_type: HeadType enum for head to use (default: HeadType.SELECTION_FACTOR_HEAD)
            distance: Maturation distance (sum of IQ-TREE branch lengths from
                naive ancestor to parent). Required when the encoder has
                distance_encoding_mode != "none"; ``None`` raises ``ValueError``
                in that case. There is no safe silent default — see issue #199.
            raw: If True, return un-zapped selection factors (before gauge
                enforcement). Default False returns zapped SF where the parent
                amino acid entry is 1.0 (log SF = 0).

        Returns:
            List of selection factor tensors, one per sequence segment.
            For single sequences: [tensor_for_sequence]
            For multi-sequence inputs like "HEAVY^LIGHT": [heavy_tensor, light_tensor]
            Each tensor has shape [seq_len, 20] with separator positions excluded.

        Raises:
            ValueError: If model doesn't have the specified head.
        """
        self.require_head(head_type)
        batch, tokens_1d = self._build_inference_batch(aa_sequence)

        self._add_inference_distance(batch, distance)
        return self._run_inference(batch, tokens_1d, head_type, raw=raw)

    def _add_inference_distance(
        self, batch: Dict[str, torch.Tensor], distance: Optional[float]
    ) -> None:
        """Add ``distance`` to a one-row inference batch if the encoder uses it.

        Raises:
            ValueError: If the encoder is distance-aware and ``distance`` is
                None. There is no safe default — see issue #199.
        """
        encoder_distance_mode = getattr(self.encoder, "distance_encoding_mode", "none")
        if encoder_distance_mode == "none":
            return
        if distance is None:
            raise ValueError(
                "Encoder is configured with distance_encoding_mode="
                f"{encoder_distance_mode!r}; selection_factors_of_aa_str() "
                "requires an explicit `distance=` argument. There is no "
                "safe default — see issue #199."
            )
        batch["distance"] = torch.tensor(
            [distance], dtype=torch.float32, device=self.device
        )
