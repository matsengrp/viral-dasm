"""Configuration models with Pydantic validation for DASM2."""

import warnings
from importlib.resources import files
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import yaml

from dasm2.data.local import get_user_config

PoolingType = Literal["mean", "attention", "single_token"]
ProjectionMode = Literal["identity", "random", "learned"]

_DEFAULT_EXPERIMENT_ARTIFACT_ROOT = (
    "/fh/fast/matsen_e/shared/bcr-mut-sel/dasm2/experiments"
)


def get_experiment_artifact_root() -> str:
    """Get experiment artifact root from user config or default.

    Reads ``experiment_artifact_root`` from the cached user config
    (:func:`dasm2.data.local.get_user_config`), so it agrees with
    :func:`dasm2.data.local.get_artifacts_dir` within a process.
    Falls back to the default shared filesystem path if not configured. See
    :func:`dasm2.data.local.get_config_path` for where the config file lives.

    The config is read once per process: a long-lived process (a notebook,
    Snakemake) that edits it mid-run sees the old value until
    :func:`dasm2.data.local.reset_config_cache` is called.

    Returns:
        Path to the experiment artifact root directory.

    Example:
        >>> # With config.yaml containing:
        >>> # experiment_artifact_root: /tmp/dasm2-experiments
        >>> get_experiment_artifact_root()
        '/tmp/dasm2-experiments'

        >>> # Without config file, returns default:
        >>> get_experiment_artifact_root()
        '/fh/fast/matsen_e/shared/bcr-mut-sel/dasm2/experiments'
    """
    return (
        get_user_config().experiment_artifact_root or _DEFAULT_EXPERIMENT_ARTIFACT_ROOT
    )


def foreign_experiment_path(experiment_name: str, *parts: str) -> Path:
    """Build a path under a foreign experiment's artifact directory.

    E.g. foreign_experiment_path("2026-07-21-dasm-baseline-5fold", "results",
    "results_table.jsonl") -> <artifact_root>/2026-07-21-dasm-baseline-5fold/results/results_table.jsonl.

    Args:
        experiment_name: The other experiment's directory name under the
            artifact root (matches its own EXPERIMENT_NAME constant).
        *parts: Additional path components appended under that directory.

    Returns:
        The resolved absolute Path. Does not check existence -- callers
        that need existence-or-skip semantics should use
        resolve_reused_results_table / resolve_reused_checkpoint in
        dasm2.analysis.experiment_reuse instead of checking this path
        directly.
    """
    return (
        Path(get_experiment_artifact_root()).expanduser()
        / experiment_name
        / Path(*parts)
    )


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from dasm2.esm2_constants import get_esm2_hidden_dim
from dasm2.heads.types import HeadType
from dasm2.losses.q_scale_defaults import (
    DEFAULT_DOUBLE_HIT_SCALE,
    DEFAULT_SINGLE_HIT_SCALE,
    DEFAULT_TRIPLE_HIT_SCALE,
    S5F_HH_POOLED_OOF_DOUBLE_HIT_SCALE,
    S5F_HH_POOLED_OOF_SINGLE_HIT_SCALE,
    S5F_HH_POOLED_OOF_TRIPLE_HIT_SCALE,
)
from dasm2.losses.types import LossType, WeightMode
from dasm2.model_catalog import resolve_model
from dasm2.molevol.mutsel import DASMNormalization
from dasm2.molevol.q_matrix import GLOBAL_KMER_Q_VARIANTS
from dasm2.neutral.types import NeutralModelType
from dasm2.sequence.constants import NUM_AMINO_ACIDS
from dasm2.sequence.masking import CensorMode

if TYPE_CHECKING:
    from dasm2.neutral.base import AbstractNeutralModel, RatesCSPNeutralModel


_NUM_SEGMENT_TYPES_DESCRIPTION = (
    "Number of segment types for segment embedding. None disables segment "
    "embedding (backward compatible). For antibodies use 4 "
    "(separator=0, heavy=1, light=2, padding=3). For single-segment use 3 "
    "(separator=0, sequence=1, padding=2)."
)


class EncoderConfig(BaseModel):
    """Unified transformer encoder configuration.

    Configures the ``TransformerEncoder`` stack, including positional encoding
    strategy (``pos_encoding``: sinusoidal or rotary), maximum precomputed
    sequence length (``pe_max_len``), optional sparse attention
    (``entmax_alpha``), and optional standalone flash attention
    (``flash_attention``).

    When ``entmax_alpha`` is None (default) and ``flash_attention`` is False,
    standard ``nn.MultiheadAttention`` with softmax is used. When
    ``flash_attention`` is True, ``FlashMultiheadAttention`` (or, when
    ``pos_encoding="rotary"``, ``FlashRotaryMultiheadAttention``) bypasses
    ``nn.MultiheadAttention`` and calls ``flash_attn_varlen_func`` directly
    (requires flash-attn package, fp16/bf16; incompatible with entmax).
    When ``entmax_alpha`` is 1.5 or 2.0, ``EntmaxMultiheadAttention`` or
    ``RotaryMultiheadAttention`` (depending on ``pos_encoding``) with the
    corresponding alpha is used.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["TransformerEncoder"] = Field(
        default="TransformerEncoder", description="Encoder type"
    )
    nhead: int = Field(gt=0, description="Number of attention heads")
    d_model_per_head: int = Field(gt=0, description="Model dimension per head")
    num_layers: int = Field(gt=0, description="Number of transformer layers")
    dim_feedforward: int = Field(gt=0, description="Feedforward dimension")
    dropout: float = Field(ge=0.0, le=1.0, default=0.1, description="Dropout rate")
    vocab_size: int = Field(
        gt=0,
        default=25,
        description="Vocabulary size (20 amino acids + special tokens)",
    )
    num_segment_types: Optional[int] = Field(
        default=None,
        ge=3,
        description=_NUM_SEGMENT_TYPES_DESCRIPTION,
    )
    d_pos: int = Field(
        default=0,
        ge=0,
        description="Learned positional embedding dimensions for concat-pos mode. "
        "When d_pos > 0, AA embedding is narrowed to d_model - d_pos and a "
        "learned positional embedding of width d_pos is concatenated.",
    )
    add_sinusoidal: bool = Field(
        default=False,
        description="Add sinusoidal positional encoding across all d_model dimensions "
        "on top of concatenated embeddings. Only meaningful when d_pos > 0.",
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_use_distance_encoding(cls, data: Any) -> Any:
        """Backward compat: map old use_distance_encoding bool to distance_encoding_mode.

        Also deprecates distance_dropout: distance embeddings now pass through
        pos_encoder dropout instead of a separate dropout layer.
        """
        if isinstance(data, dict) and "use_distance_encoding" in data:
            if "distance_encoding_mode" in data:
                raise ValueError(
                    "Cannot specify both use_distance_encoding (deprecated) "
                    "and distance_encoding_mode. Use distance_encoding_mode only."
                )
            if data["use_distance_encoding"]:
                data["distance_encoding_mode"] = "continuous"
            del data["use_distance_encoding"]
        if isinstance(data, dict) and data.get("distance_dropout", 0.0) > 0:
            warnings.warn(
                "distance_dropout is deprecated and ignored. Distance embeddings "
                "now pass through pos_encoder dropout instead of a separate "
                "dropout layer. Remove distance_dropout from your config.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["distance_dropout"] = 0.0
        return data

    @model_validator(mode="after")
    def _validate_field_consistency(self) -> "EncoderConfig":
        """Validate field consistency across embedding and encoding options."""
        if self.d_pos > 0 and self.d_pos >= self.d_model:
            raise ValueError(
                f"d_pos ({self.d_pos}) must be less than d_model ({self.d_model}). "
                f"No room for AA embedding (d_aa would be {self.d_model - self.d_pos})."
            )
        if self.add_sinusoidal and self.d_pos == 0:
            raise ValueError(
                "add_sinusoidal=True requires d_pos > 0 (concat-pos embedding). "
                "Set d_pos > 0 or remove add_sinusoidal."
            )
        if self.d_pos > 0 and self.pos_encoding == "rotary":
            raise ValueError(
                "d_pos > 0 (concat-pos embedding) is incompatible with "
                "pos_encoding='rotary'. Concat-pos handles its own positional "
                "encoding via the learned positional subspace."
            )
        if (
            self.flash_attention
            and self.pos_encoding == "rotary"
            and self.entmax_alpha is not None
        ):
            # Checked ahead of the general flash+entmax case below so this
            # rotary-specific message actually fires for this combination
            # rather than the general one.
            raise ValueError(
                "flash_attention + pos_encoding='rotary' is incompatible with "
                f"entmax (entmax_alpha={self.entmax_alpha}). Flash-attn uses a "
                "fused softmax kernel; sparse attention requires the manual path."
            )
        if self.flash_attention and self.entmax_alpha is not None:
            raise ValueError(
                "flash_attention is incompatible with entmax "
                f"(entmax_alpha={self.entmax_alpha}). Flash-attn uses a fused "
                "softmax kernel; sparse attention requires the manual path."
            )
        if self.per_segment_positions and self.pos_encoding == "rotary":
            raise ValueError(
                "per_segment_positions is incompatible with pos_encoding='rotary'. "
                "Rotary encodes position inside attention; per-segment reset is "
                "only implemented for sinusoidal PE and the concat-pos learned "
                "positional embedding."
            )
        if (
            self.pos_encoding == "rotary"
            and self.distance_encoding_mode == "context_token"
        ):
            # Rotary applies position info inside attention, so the prepended
            # context token would occupy rotary position 0 and shift AAs to
            # positions 1..L — defeating the "no PE on context token" design.
            raise ValueError(
                "distance_encoding_mode='context_token' is incompatible with "
                "pos_encoding='rotary'. Rotary applies position info inside "
                "attention, which would assign the context token rotary "
                "position 0 and shift AAs by one. Use pos_encoding='sinusoidal' "
                "with context_token mode."
            )
        # Distance encoding field consistency
        mode = self.distance_encoding_mode
        if mode != "continuous" and self.distance_noise_std > 0:
            raise ValueError(
                f"distance_noise_std > 0 is only valid with "
                f"distance_encoding_mode='continuous', got '{mode}'."
            )
        if mode != "context_token" and self.distance_mask_prob > 0:
            raise ValueError(
                f"distance_mask_prob > 0 is only valid with "
                f"distance_encoding_mode='context_token', got '{mode}'."
            )
        return self

    # Transformer block sub-component configuration
    activation: Literal["relu", "gelu", "silu", "swiglu"] = Field(
        default="relu",
        description="Activation / FFN type: relu/gelu/silu use StandardFFN, swiglu uses SwiGLUFFN",
    )
    norm_type: Literal["layer_norm", "rms_norm"] = Field(
        default="layer_norm",
        description="Normalization layer type",
    )
    norm_first: bool = Field(
        default=False,
        description="Pre-LN (True) or Post-LN (False, backward compatible default)",
    )

    # Positional encoding
    pos_encoding: Literal["sinusoidal", "rotary"] = Field(
        default="sinusoidal",
        description="Positional encoding strategy. 'sinusoidal' adds absolute position "
        "before the stack; 'rotary' applies relative position via Q/K rotation at every layer.",
    )
    token_embedding_init: Literal["standard_normal", "inverse_sqrt_width"] = Field(
        default="standard_normal",
        description="Initialization of the amino-acid token embedding, which the "
        "embedding pipeline multiplies by sqrt(width) (width = d_model, or d_model - "
        "d_pos with concat-pos). 'standard_normal' keeps nn.Embedding's N(0, 1), so a "
        "scaled row starts with norm about width, far above a sinusoidal PE row's "
        "sqrt(d_model / 2). 'inverse_sqrt_width' divides the initial weights by "
        "sqrt(width), so a scaled row starts with norm about sqrt(width), at PE's scale.",
    )
    pe_max_len: int = Field(
        default=5000,
        gt=0,
        description="Maximum sequence length for positional encoding precomputation. "
        "Used by both sinusoidal (PE buffer size) and rotary (frequency table size).",
    )
    per_segment_positions: bool = Field(
        default=False,
        description="Reset positional encoding at each segment boundary so a "
        "segment's (chain's) positions restart at 0 instead of accumulating "
        "across the concatenated sequence. Applies to sinusoidal PE and to the "
        "concat-pos learned positional embedding (d_pos > 0). Independent of "
        "segment embedding; incompatible with pos_encoding='rotary'.",
    )

    # Attention sparsity (None = softmax, 1.5 = entmax15, 2.0 = sparsemax)
    entmax_alpha: Optional[Literal[1.5, 2.0]] = Field(  # type: ignore[valid-type]
        default=None,
        description="Entmax alpha for sparse attention. None=softmax (default), "
        "1.5=entmax15, 2.0=sparsemax.",
    )

    # Flash attention (standalone flash-attn v2)
    flash_attention: bool = Field(
        default=False,
        description="Use standalone flash-attn v2 via FlashMultiheadAttention "
        "(or FlashRotaryMultiheadAttention when pos_encoding='rotary'). "
        "Requires flash-attn package and fp16/bf16 inputs. Incompatible with entmax.",
    )
    flash_deterministic: bool = Field(
        default=False,
        description="Use deterministic flash-attn backward pass. Slower but "
        "bitwise reproducible. Only applies when flash_attention=True.",
    )

    # Distance encoding for maturation stage conditioning
    distance_encoding_mode: Literal["none", "continuous", "binned", "context_token"] = (
        Field(
            default="none",
            description=(
                "How to encode maturation distance (IQ-TREE branch length from "
                "naive ancestor). 'none': no distance encoding. 'continuous': "
                "project log(distance + ε) to d_model and add to all positions. "
                "'binned': quantile-bin log distance into num_distance_bins and "
                "add learned embedding to all positions. 'context_token': "
                "quantile-bin and prepend as an extra context token."
            ),
        )
    )
    distance_noise_std: float = Field(
        default=0.0,
        ge=0.0,
        description="Gaussian noise std added to log(distance) during training "
        "(only for 'continuous' mode).",
    )
    distance_dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="DEPRECATED — ignored. Distance embeddings now pass through "
        "pos_encoder dropout. Kept for backward compatibility with old configs.",
    )
    num_distance_bins: int = Field(
        default=10,
        gt=0,
        description="Number of quantile bins for 'binned' and 'context_token' modes.",
    )
    distance_mask_prob: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of masking depth context token during training "
        "(only for 'context_token' mode).",
    )

    # Optional 3Di structural embedding
    num_tdi_tokens: Optional[int] = Field(
        default=None,
        ge=20,
        description="3Di vocabulary size for structural embedding. "
        "None=no structural embedding, 22=standard (20 states + mask + padding).",
    )

    @property
    def d_model(self) -> int:
        """Total model dimension calculated as nhead × d_model_per_head."""
        return self.nhead * self.d_model_per_head

    @property
    def supports_tdi(self) -> bool:
        """Whether this encoder is configured with 3Di structural tokens."""
        return self.num_tdi_tokens is not None

    @property
    def uses_distance(self) -> bool:
        """Whether this encoder requires distance in the batch."""
        return self.distance_encoding_mode != "none"


class ESM2DecoderConfig(BaseModel):
    """Trainable transformer layers between ESM2's embeddings and the heads.

    ``ESM2Encoder`` projects ESM2's output down to ``d_model`` with a
    Linear layer, then runs ``num_layers`` ``TransformerStack`` blocks with
    softmax attention over the residue positions (issue #1509). It adds no
    positional encoding: ESM2's embeddings already carry position from its
    rotary attention. The layers use ``TransformerStack``'s defaults of
    ReLU FFN and LayerNorm.

    Attributes:
        d_model: Width of the down-projection and of the transformer layers;
            the encoder's output dimension when the block is set.
        num_layers: Number of transformer layers.
        nhead: Attention heads per layer; must divide ``d_model``.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout probability in attention, FFN and residuals.
        norm_first: Pre-LN (True, with a final norm) or Post-LN (False).
            Defaults to Pre-LN, unlike ``EncoderConfig.norm_first``, whose
            Post-LN default keeps netam compatibility.
    """

    d_model: int = Field(gt=0, description="Projection and layer width")
    num_layers: int = Field(gt=0, description="Number of transformer layers")
    nhead: int = Field(gt=0, description="Attention heads per layer")
    dim_feedforward: int = Field(gt=0, description="FFN hidden dimension")
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0, description="Dropout")
    norm_first: bool = Field(default=True, description="Pre-LN if True")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_nhead_divides_d_model(self) -> "ESM2DecoderConfig":
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"decoder d_model={self.d_model} is not divisible by nhead={self.nhead}"
            )
        return self


class ESM2EncoderConfig(BaseModel):
    """ESM2 pretrained encoder configuration.

    Uses HuggingFace ESM2 models as frozen feature extractors or
    fine-tunable backbones for DASM2.

    Attributes:
        freeze_parameters: Configuration intent for parameter freezing.
            Note: The encoder's `parameters_frozen` property reflects actual
            runtime state, which may differ if parameters are later unfrozen.
        num_frozen_layers: Number of early transformer layers to freeze (0-indexed).
            If set, overrides freeze_parameters. Layers 0 to num_frozen_layers-1 are
            frozen, remaining layers are trainable. For ESM2-t30-150M with 30 layers,
            setting num_frozen_layers=25 freezes layers 0-24 and trains layers 25-29.
        decoder: Optional trainable transformer layers on top of ESM2's
            embeddings (``ESM2DecoderConfig``). ``freeze_parameters`` and
            ``num_frozen_layers`` apply to ESM2 only; ``--freeze_encoder``
            freezes the decoder too.
    """

    type: Literal["ESM2Encoder"] = Field(
        default="ESM2Encoder", description="Encoder type"
    )
    model_name: str = Field(
        default="facebook/esm2_t30_150M_UR50D",
        description="HuggingFace ESM2 model identifier",
    )
    freeze_parameters: bool = Field(
        default=True,
        description="Whether to freeze ESM2 parameters (True for feature extraction)",
    )
    num_frozen_layers: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of early layers to freeze. If set, overrides freeze_parameters.",
    )
    decoder: Optional[ESM2DecoderConfig] = Field(
        default=None,
        description="Optional trainable transformer layers after ESM2",
    )

    @property
    def d_model(self) -> int:
        """Output embedding dimension: the decoder's width if set, else ESM2's.

        Returns:
            Dimension of the embeddings the heads consume
        """
        if self.decoder is not None:
            return self.decoder.d_model
        return self.esm2_hidden_dim

    @property
    def esm2_hidden_dim(self) -> int:
        """ESM2's own output dimension, whether or not a decoder follows it."""
        return get_esm2_hidden_dim(self.model_name)

    @property
    def layers_label(self) -> str:
        """Summary-table layer label: "ESM2", or "ESM2+5" with a 5-layer decoder."""
        if self.decoder is None:
            return "ESM2"
        return f"ESM2+{self.decoder.num_layers}"

    @property
    def num_layers(self) -> None:
        """ESM2 layers are internal to the pretrained model.

        Returns None to indicate layers are not configurable.
        This allows uniform handling with EncoderConfig in metrics code.
        """
        return None

    @property
    def supports_tdi(self) -> bool:
        """Whether this encoder supports 3Di structural tokens (always False)."""
        return False

    model_config = ConfigDict(extra="forbid")


class MPNNLightAttentionEncoderConfig(BaseModel):
    """Configuration for MPNNLightAttentionEncoder (ThermoMPNN-compatible).

    Matches ThermoMPNN's light-attention/MLP architecture for processing
    precomputed ProteinMPNN decoder features with frozen sequence embeddings.

    Attributes:
        d_output: Encoder output dimension (default 32, matching ThermoMPNN).
        hidden_dims: MLP hidden layer dimensions (default [64]).
        kernel_size: Conv1d kernel size for light attention (default 9).
        dropout: Dropout rate for light attention feature path (default 0.25).
        seq_embedding_path: Path to ProteinMPNN W_s matrix .npy file [21, 128].
        thermompnn_weights_path: Optional path to ThermoMPNN checkpoint for warm-start.
    """

    type: Literal["MPNNLightAttentionEncoder"] = Field(
        default="MPNNLightAttentionEncoder", description="Encoder type"
    )
    d_output: int = Field(default=32, gt=0, description="Encoder output dimension")
    hidden_dims: List[int] = Field(
        default=[64], description="MLP hidden layer dimensions"
    )
    kernel_size: int = Field(
        default=9, gt=0, description="Conv1d kernel size for light attention"
    )
    dropout: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Dropout rate for light attention feature path",
    )
    seq_embedding_path: str = Field(
        description="Path to ProteinMPNN seq_embedding.npy [21, 128]"
    )
    thermompnn_weights_path: Optional[str] = Field(
        default=None,
        description="Optional path to ThermoMPNN checkpoint for warm-start",
    )

    @property
    def d_model(self) -> int:
        """Output embedding dimension seen by prediction heads."""
        return self.d_output

    @property
    def supports_tdi(self) -> bool:
        """Whether this encoder supports 3Di structural tokens (always False)."""
        return False

    model_config = ConfigDict(extra="forbid")


class IdentityEncoderConfig(BaseModel):
    """No-op encoder configuration for encoder-free heads.

    Pairs with ``dasm2.models.core.identity.IdentityEncoder`` (a thin
    nn.Module that returns ``torch.zeros([B, L, d_model])`` on every
    forward and contributes zero learnable parameters).

    The five context-independent baseline heads in dasm2/heads/ —
    ``PerSiteParentChildHead``, ``PerSiteChildHead``,
    ``PerSitePreferenceHead``, ``PerSiteMeanSelectionHead``, and
    ``SingleMatrixHead`` (plus the floor-of-the-floor ``SingleValueHead``) —
    all ignore the encoder output in their ``forward()``. Configs that
    pair them with this encoder make the "no transformer" architectural
    choice explicit (and skip the ~77K dead transformer parameters that
    historical baseline configs accidentally trained).

    Attributes:
        d_model: Nominal embedding dimension. Defaults to 1 because the
            output tensor is never read by encoder-free heads, but the
            field is kept so a non-default value can be set if a future
            head ever wants to use the IdentityEncoder for shape
            compatibility with a SF projection.

    Note: ``ModelConfig.validate_heads`` auto-sets ``head_config.input_dim
    = encoder.d_model`` for every head in the model, so setting
    ``d_model`` here also sets ``input_dim`` on all paired heads. The
    encoder-free baseline heads ignore that field by design, but a future
    head that accidentally consumes ``sequence_embeddings`` while paired
    with this encoder would receive ``input_dim=1`` and likely fail with
    a shape mismatch rather than a clear config error.
    """

    type: Literal["IdentityEncoder"] = Field(
        default="IdentityEncoder", description="Encoder type"
    )
    d_model: int = Field(
        default=1,
        gt=0,
        description="Nominal embedding dimension. Default 1; the output "
        "tensor is never read by encoder-free heads. Setting this also "
        "controls each paired head's auto-set input_dim — see class "
        "docstring for the coupling.",
    )

    @property
    def supports_tdi(self) -> bool:
        """Whether this encoder supports 3Di structural tokens (always False)."""
        return False

    model_config = ConfigDict(extra="forbid")


class BaseHeadConfig(BaseModel):
    """Base head configuration with common fields.

    All heads share these fields. Specific head types extend this with
    additional required fields. Uses discriminated union pattern for
    fail-fast validation at deserialization time.

    Per #621 Task #4 the canonical home for ``loss_weight`` is the
    ``Task`` object, not the head config; the field has been retired
    from this base class as part of the v3-compat shim retirement.
    """

    head_type: HeadType = Field(description="Head type discriminator")
    input_dim: Optional[int] = Field(
        default=None,
        gt=0,
        description="Input dimension (auto-set to encoder d_model if not specified)",
    )

    model_config = ConfigDict(extra="forbid")


class SelectionFactorHeadConfig(BaseHeadConfig):
    """Configuration for ``SelectionFactorHead`` (post-#621 Task #5).

    The unified config for the head that consumes the shared selection
    factor tensor and identity-forwards it. The matching loss class —
    ``DASMLoss`` / ``WhichmutLoss`` / ``AAOWhichmutLoss`` — is chosen
    independently via ``LossConfig`` and attached to ``Task.loss``.
    """

    head_type: Literal[HeadType.SELECTION_FACTOR_HEAD] = HeadType.SELECTION_FACTOR_HEAD


class SiteMaskedSelectionFactorHeadConfig(BaseHeadConfig):
    """Configuration for ``SiteMaskedSelectionFactorHead`` (issue #1550).

    Selects ``SiteMaskedCoreModel``: each site's fitness vector ``F_j`` is
    read from an encoder pass with site ``j`` masked, and the model returns
    ``log f_j(y) = log phi(F_j(y) - F_j(x))`` with Halpern-Bruno ``phi``,
    plus ``S[x, y]`` when ``pair_table`` is set. The encoder needs
    ``vocab_size >= 23`` for the mask token, and the shared
    ``sf_projection`` must not set ``wiggle_beta``.
    """

    head_type: Literal[HeadType.SITE_MASKED_SELECTION_FACTOR_HEAD] = (
        HeadType.SITE_MASKED_SELECTION_FACTOR_HEAD
    )
    mask_rate: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
        description="Fraction of maskable sites masked per row in each "
        "training forward. Always with the mask token (no 80/10/10). Keep it "
        "near 1/eval_num_partitions so training and evaluation see the same "
        "fraction of context.",
    )
    eval_num_partitions: int = Field(
        default=7,
        ge=1,
        description="Number of encoder passes per evaluation batch. Each "
        "maskable site is masked in exactly one pass.",
    )
    partition_seed: int = Field(
        default=0,
        description="Seed of the evaluation partition. With the row's tokens "
        "it fixes which pass masks each site.",
    )
    pair_table: bool = Field(
        default=False,
        description="Add a learnable 20x20 table S, indexed by (parent, "
        "child) amino acid and zapped on its diagonal as SingleMatrixHead's "
        "is, to the masked-context term at scored sites: log f_j(y | x) = "
        "S[x, y] + log phi(F_j(y) - F_j(x)) (issue #1670).",
    )


class QDASMEvolHeadConfig(BaseHeadConfig):
    """Configuration for QDASMEvolHead with optional wiggle activation.

    Q-matrix variant of DASM that uses exact codon transition probabilities
    via matrix exponential exp(tQ) rather than the Poisson approximation.

    Per #621 Task #4 the canonical home for the Q-matrix construction
    parameters (``use_cubic``, ``single/double/triple_hit_scale``,
    ``adaptive``, ``tq_threshold``, ``adaptive_mode``, ``quadratic_ste``,
    ``q_variant``, ``thrifty_13mer_table_path``) is ``QDASMLossConfig``.
    The head-side parameters that remain here both act on the forward pass:
    ``wiggle_beta`` (activation) and ``selection_log_floor`` (smooth lower
    bound on the log selection factors).
    """

    head_type: Literal[HeadType.QDASM_EVOL_HEAD] = HeadType.QDASM_EVOL_HEAD
    wiggle_beta: Optional[float] = Field(
        default=0.3,
        description="Beta parameter for wiggle activation (netam-compatible default: 0.3). "
        "Set to null/None to disable wiggle and use linear pass-through.",
    )
    selection_log_floor: Optional[float] = Field(
        default=-6.0,
        description="Smooth lower bound on log selection factors (default -6.0 => "
        "f >= e^-6). wiggle is unbounded below and QDASM's ratio-form Q uses "
        "1/f_source, so an un-floored source factor near zero makes ||tQ|| blow up "
        "and silently corrupts exp(tQ). The softplus floor caps it while keeping "
        "gradients alive. Set to null/None to disable (unbounded; ablation only).",
    )

    @model_validator(mode="after")
    def validate_wiggle_beta(self) -> "QDASMEvolHeadConfig":
        """Validate wiggle_beta is positive when specified."""
        if self.wiggle_beta is not None and self.wiggle_beta <= 0.0:
            raise ValueError(
                f"wiggle_beta must be positive when specified, got {self.wiggle_beta}"
            )
        return self

    @model_validator(mode="after")
    def validate_selection_log_floor(self) -> "QDASMEvolHeadConfig":
        """Validate selection_log_floor is a negative log bound when specified."""
        if self.selection_log_floor is not None and self.selection_log_floor >= 0.0:
            raise ValueError(
                "selection_log_floor must be strictly negative when specified: it is "
                "a lower bound on the log selection factors, and the diagonal zap sets "
                "the parent log-factor to 0, so the floor must sit below 0 for the "
                f"wildtype gauge (f[parent] = 1) to survive. Got {self.selection_log_floor}"
            )
        return self


class HybridEvolHeadConfig(BaseHeadConfig):
    """Configuration for HybridEvolHead with optional wiggle activation.

    Hybrid model that uses exact neutral CTMC (matrix exponential with unit
    selection factors) for transition probabilities, then applies learned
    selection factors post-hoc.

    Per #621 Task #4 Q-matrix construction parameters live on
    ``HybridLossConfig``. Only the head-side ``wiggle_beta`` activation
    parameter remains here.
    """

    head_type: Literal[HeadType.HYBRID_EVOL_HEAD] = HeadType.HYBRID_EVOL_HEAD
    wiggle_beta: Optional[float] = Field(
        default=0.3,
        description="Beta parameter for wiggle activation (netam-compatible default: 0.3). "
        "Set to null/None to disable wiggle and use linear pass-through.",
    )

    @model_validator(mode="after")
    def validate_wiggle_beta(self) -> "HybridEvolHeadConfig":
        """Validate wiggle_beta is positive when specified."""
        if self.wiggle_beta is not None and self.wiggle_beta <= 0.0:
            raise ValueError(
                f"wiggle_beta must be positive when specified, got {self.wiggle_beta}"
            )
        return self


class AAHWhichmutHeadConfig(BaseHeadConfig):
    """Configuration for AAHWhichmutHead (Amino Acid Hybrid).

    AAH takes both nucleotide (codon) and amino-acid-only data:
    - NT data via WhichmutDataset anchors the gauge through synonymous mutations
    - AA data via AAOWhichmutDataset uses the anchored gauge for training

    Single-head architecture that receives batches from two different datasets
    and computes the appropriate loss based on batch contents.

    Uses a 2-layer MLP with ReLU activation for predictions with diagonal zapping.
    """

    head_type: Literal[HeadType.AAH_WHICHMUT_HEAD] = HeadType.AAH_WHICHMUT_HEAD


class PerSiteChildHeadConfig(BaseHeadConfig):
    """Configuration for PerSiteChildHead.

    Position-specific selection factors independent of parent sequence
    (a learned ``[max_aa_seq_len, 20]`` table shared across all sequences).

    The presence of ``wildtype_sequence`` selects the diagonal-zap mode:

    - **omitted / ``None`` (default):** zap **per-parent** — each
      sequence's own parent AA (``batch["aa_parents_idxs"]``) is its
      neutral reference, exactly like ``PerSiteParentChildHead``. This is
      the correct anchor for the architectural baseline comparison
      (transformer vs context-independent).
    - **an AA string:** zap on that single fixed reference for every
      sequence (legacy / side-by-side comparison mode; used by the
      fitness experiments for the Bloom/Neher comparison).

    ``max_aa_seq_len`` is auto-populated from ``ModelConfig.max_aa_seq_len``
    via the ``requires_max_aa_seq_len`` auto-wiring; it does not need to be
    set on the head config directly.
    """

    head_type: Literal[HeadType.PER_SITE_CHILD_HEAD] = HeadType.PER_SITE_CHILD_HEAD
    max_aa_seq_len: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum amino acid sequence length "
        "(auto-populated from ModelConfig.max_aa_seq_len).",
    )
    wildtype_sequence: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Fixed wildtype AA reference for diagonal zapping. "
        "Omit for per-parent zapping (the default).",
    )


class StructuralGuidanceHeadConfig(BaseHeadConfig):
    """Configuration for StructuralGuidanceHead.

    Minimal config - only input_dim (inherited from BaseHeadConfig).
    HERMES dimension constraint (256) is enforced by loss function, not config.
    """

    head_type: Literal[HeadType.STRUCTURAL_GUIDANCE_HEAD] = (
        HeadType.STRUCTURAL_GUIDANCE_HEAD
    )
    # No additional fields needed - input_dim inherited from base


class DMSFunctionalScoreHeadConfig(BaseHeadConfig):
    """Configuration for DMSFunctionalScoreHead.

    Predicts L×20 amino acid selection factors with loss computed against
    DMS experimental measurements. Uses diagonal zapping (no wiggle).
    """

    head_type: Literal[HeadType.DMS_FUNCTIONAL_SCORE_HEAD] = (
        HeadType.DMS_FUNCTIONAL_SCORE_HEAD
    )
    # No additional fields needed - input_dim inherited from base


class DeltaAffinityHeadConfig(BaseHeadConfig):
    """Configuration for DeltaAffinityHead.

    Predicts scalar delta-affinity (change in binding affinity relative to
    wildtype) from pooled sequence embeddings. Uses masked mean pooling
    followed by a linear projection to a single scalar output.

    Loss is Huber; ``DeltaAffinityLossConfig.huber_delta`` controls the
    transition point (default 1.0).
    """

    head_type: Literal[HeadType.DELTA_AFFINITY_HEAD] = HeadType.DELTA_AFFINITY_HEAD
    # No additional fields needed - input_dim inherited from base


class ContrastiveBindingHeadConfig(BaseHeadConfig):
    """Configuration for ContrastiveBindingHead (issue #362).

    Dual-encoder contrastive head that pulls the DASM2 encoder toward a joint
    Ab/Ag embedding space. Antibody side uses the trainable DASM2 encoder
    output; antigen side uses a pre-computed frozen ESM-2 per-residue
    embedding passed in via the batch.

    The head adds three trainable tensors on top of the shared backbone:
    an Ab-side linear projection, an Ag-side linear projection, and a
    scalar ``log_tau`` (inverse log-temperature). The contrastive loss
    variant (and the ``logit_bias`` Parameter for siglip) lives on
    ``ContrastiveBindingLossConfig.contrastive_loss_type``.
    """

    head_type: Literal[HeadType.CONTRASTIVE_BINDING_HEAD] = (
        HeadType.CONTRASTIVE_BINDING_HEAD
    )
    projection_dim: int = Field(
        default=256,
        gt=0,
        description="Output dim of the shared joint Ab/Ag embedding space.",
    )
    antigen_input_dim: int = Field(
        default=1280,
        gt=0,
        description="Per-residue antigen embedding dim (1280 for esm2_t33_650M_UR50D).",
    )
    temperature_init: float = Field(
        default=0.2,
        gt=0.0,
        description="Initial softmax temperature τ; stored as log_tau for "
        "positivity. Default 0.2 matches CALM (inv_tau=5, Lee et al. 2026, "
        "doi:10.64898/2026.02.25.707916). CLIP uses τ=0.07 (Radford et al. "
        "2021) — plausible alternative if the batch ends up much larger than "
        "CALM's ~128.",
    )
    # Ab-side pooling configuration. The Ag side is pre-pooled in
    # ``LoadAntigenEmbedding`` at data-loading time (the ESM-2 encoder is
    # frozen), so no Ag-side pooling module lives in the head today.
    ab_pooling_type: PoolingType = Field(
        default="mean",
        description="Pooling strategy for the Ab-side per-residue encoder "
        "output. 'mean' uses MaskedMeanPooling on ab_pool_mask; 'attention' "
        "uses MultiHeadScorePooling; 'single_token' selects the embedding at "
        "the first occurrence of ab_target_token_idx (CLS-style, defaults to "
        "the ^ chain separator).",
    )
    ab_pooling_hidden_dim: int = Field(
        default=64,
        gt=0,
        description="MHSP scoring network hidden dim (ab_pooling_type='attention').",
    )
    ab_num_heads: int = Field(
        default=1,
        gt=0,
        description="Number of MHSP attention heads (ab_pooling_type='attention').",
    )
    ab_head_dim: Optional[int] = Field(
        default=None,
        gt=0,
        description="MHSP per-head output dim; defaults to input_dim "
        "(ab_pooling_type='attention').",
    )
    ab_projection_mode: ProjectionMode = Field(
        default="identity",
        description="MHSP projection mode (ab_pooling_type='attention').",
    )
    ab_target_token_idx: Optional[int] = Field(
        default=None,
        ge=0,
        description="Token index for single-token Ab pooling; defaults to "
        "AA_SEPARATOR_IDX when ab_pooling_type='single_token'.",
    )


class BindingClassificationHeadConfig(BaseHeadConfig):
    """Configuration for ``BindingClassificationHead`` (issue #801).

    Cross-attention binding-classification head: entity-A (e.g.
    antibody) per-residue embeddings from the shared encoder cross-attend
    into entity-B (e.g. antigen) per-residue embeddings carried in the
    batch, pass through a small residual Conv1D stack, get aggregated by
    ``MultiHeadScorePooling`` (MHSP), and produce one binding logit. The
    head is agnostic to how either side's embeddings were produced
    (trainable or frozen upstream) — it just consumes two hidden-state
    tensors from the batch.
    """

    head_type: Literal[HeadType.BINDING_CLASSIFICATION_HEAD] = (
        HeadType.BINDING_CLASSIFICATION_HEAD
    )
    context_input_dim: int = Field(
        default=1280,
        gt=0,
        description="Entity-B per-residue embedding dim (default 1280 "
        "matches esm2_t33_650M_UR50D, a common source for this field, but "
        "the head does not require entity-B embeddings to come from a "
        "frozen encoder).",
    )
    attn_dim: int = Field(default=256, gt=0, description="Cross-attention hidden dim.")
    attn_heads: int = Field(
        default=8, gt=0, description="Number of cross-attention heads."
    )
    conv_kernel: int = Field(
        default=3, gt=0, description="Kernel size of each residual Conv1D block."
    )
    n_conv_blocks: int = Field(
        default=2,
        ge=0,
        description="Number of residual Conv1D blocks after cross-attention. "
        "0 reproduces the prototype's no-conv variant.",
    )
    pool_heads: int = Field(
        default=16, gt=0, description="Number of MultiHeadScorePooling heads."
    )
    pool_head_dim: Optional[int] = Field(
        default=16,
        gt=0,
        description="MultiHeadScorePooling per-head output dim. Set to null "
        "to auto-resolve: attn_dim when pool_projection_mode='identity', "
        "otherwise attn_dim // pool_heads.",
    )
    pool_hidden_dim: int = Field(
        default=64,
        gt=0,
        description="MultiHeadScorePooling scoring network hidden dim.",
    )
    pool_projection_mode: ProjectionMode = Field(
        default="learned", description="MultiHeadScorePooling projection mode."
    )
    cross_skip: bool = Field(
        default=True,
        description="Add a residual skip from the entity-A input around the "
        "cross-attention before the LayerNorm (transformer-style).",
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout applied after the cross-attention block, after the "
        "conv stack, and on the pooled vector before the final linear (issue "
        "#1096 follow-up). 0.0 reproduces the head's original no-dropout behavior.",
    )


class DASMFitResidualHeadConfig(BaseHeadConfig):
    """Configuration for DASMFitResidualHead.

    2-layer MLP mapping frozen DASM embeddings to k×20 log-selection modifiers
    r_i(j, a) for alternating optimization in DASMfit.

    Attributes:
        k: Number of residual components (directions in selection-factor space).
        hidden_dim: Hidden dimension of the 2-layer MLP.
    """

    head_type: Literal[HeadType.DASM_FIT_RESIDUAL_HEAD] = (
        HeadType.DASM_FIT_RESIDUAL_HEAD
    )
    k: int = Field(
        default=3,
        gt=0,
        description="Number of residual components",
    )
    hidden_dim: int = Field(
        default=128,
        gt=0,
        description="Hidden dimension of the 2-layer MLP",
    )


class PerSiteParentChildHeadConfig(BaseHeadConfig):
    """Configuration for PerSiteParentChildHead.

    Position-specific selection factors conditioned on parent amino acid.
    Learns a [max_aa_seq_len, 20, 20] tensor indexed by (position, parent_aa, child_aa).
    """

    head_type: Literal[HeadType.PER_SITE_PARENT_CHILD_HEAD] = (
        HeadType.PER_SITE_PARENT_CHILD_HEAD
    )
    max_aa_seq_len: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum amino acid sequence length (auto-set from ModelConfig)",
    )


class PerSitePreferenceHeadConfig(BaseHeadConfig):
    """Configuration for PerSitePreferenceHead.

    Position-specific preference model deriving selection factors as ratios.
    Learns a [max_aa_seq_len, 20] preference vector; selection factors are
    log_pi[child] - log_pi[parent], enforcing detailed balance.
    """

    head_type: Literal[HeadType.PER_SITE_PREFERENCE_HEAD] = (
        HeadType.PER_SITE_PREFERENCE_HEAD
    )
    max_aa_seq_len: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum amino acid sequence length (auto-set from ModelConfig)",
    )


class PerSiteMeanSelectionHeadConfig(BaseHeadConfig):
    """Configuration for PerSiteMeanSelectionHead.

    One scalar log selection factor per site, broadcast to all 20 child amino
    acids. The simplest site-level selection model (L parameters total).
    """

    head_type: Literal[HeadType.PER_SITE_MEAN_SELECTION_HEAD] = (
        HeadType.PER_SITE_MEAN_SELECTION_HEAD
    )
    max_aa_seq_len: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum amino acid sequence length (auto-set from ModelConfig)",
    )


class SingleValueHeadConfig(BaseHeadConfig):
    """Configuration for SingleValueHead.

    One-parameter baseline: a single learnable scalar log selection factor
    broadcast to every site and amino acid. Direct port of netam
    ``SingleValueBinarySelectionModel``.
    """

    head_type: Literal[HeadType.SINGLE_VALUE_HEAD] = HeadType.SINGLE_VALUE_HEAD


class M0HeadConfig(BaseHeadConfig):
    """Configuration for M0Head.

    One-parameter parent-zapped baseline: a single learnable scalar log
    selection factor broadcast to every site, with the parent-AA column
    zapped to ``0.0`` so ``exp(scalar)`` recovers the M0 ω directly at
    MLE under the DASMLoss "small-change" parameterization. See
    :class:`dasm2.heads.m0.M0Head` for the gauge derivation.
    """

    head_type: Literal[HeadType.M0_HEAD] = HeadType.M0_HEAD


class NeutralHeadConfig(BaseHeadConfig):
    """Configuration for NeutralHead.

    Parameter-free f ≡ 1 model: every log selection factor is 0, so a
    ``DASMLoss`` scores the neutral model alone under its configured
    ``normalization``. See :class:`dasm2.heads.neutral.NeutralHead`.
    """

    head_type: Literal[HeadType.NEUTRAL_HEAD] = HeadType.NEUTRAL_HEAD


class SingleMatrixHeadConfig(BaseHeadConfig):
    """Configuration for SingleMatrixHead.

    Context-free baseline: a learnable 20x20 log selection factor table
    indexed by parent amino acid, with the parent->parent diagonal zapped
    to 0 so synonymous codon substitutions stay neutral (380 effective
    free parameters). Generalizes ``SingleValueHead`` from one scalar to
    a per-(parent, child) cell while still ignoring per-position and
    per-context structure.
    """

    head_type: Literal[HeadType.SINGLE_MATRIX_HEAD] = HeadType.SINGLE_MATRIX_HEAD


class PooledRegressionHeadConfig(BaseHeadConfig):
    """Configuration for PooledRegressionHead.

    Pooled sequence-level head for real-valued regression targets. Pools
    per-residue embeddings to a fixed-size vector via one of three strategies
    (``mean``, ``attention``, ``single_token``), projects to ``num_targets``
    outputs, and trains with Huber loss on ``batch['delta_affinity']``.

    With ``pooling_type="mean"`` and ``num_targets=1`` (defaults), this head
    is architecturally and numerically equivalent to :class:`DeltaAffinityHead`
    (verified via a weight-copy phase-gate test).

    Attributes:
        pooling_type: Pooling strategy.
            - ``"mean"`` uses :class:`MaskedMeanPooling` (zero-parameter).
            - ``"attention"`` uses :class:`MultiHeadScorePooling` (configurable
              via ``num_heads``, ``head_dim``, ``projection_mode``,
              ``pooling_hidden_dim``).
            - ``"single_token"`` uses :class:`SingleTokenPooling` to select the
              embedding at the first occurrence of ``target_token_idx``.
        pooling_hidden_dim: Hidden dimension of the MHSP scoring network
            (only used when ``pooling_type="attention"``).
        num_heads: Number of attention heads (only used when
            ``pooling_type="attention"``).
        head_dim: Per-head output dimension. Defaults to ``input_dim`` (only
            used when ``pooling_type="attention"``).
        projection_mode: MHSP projection mode (only used when
            ``pooling_type="attention"``).
        target_token_idx: Token index to select (only used when
            ``pooling_type="single_token"``). Defaults to the ``^`` chain
            separator index.
        num_targets: Number of output targets. Output shape is ``[B]`` when
            ``num_targets=1`` and ``[B, num_targets]`` otherwise.
    """

    head_type: Literal[HeadType.POOLED_REGRESSION_HEAD] = (
        HeadType.POOLED_REGRESSION_HEAD
    )
    pooling_type: PoolingType = Field(
        default="mean",
        description="Pooling strategy for aggregating per-residue embeddings.",
    )
    pooling_hidden_dim: int = Field(
        default=64,
        gt=0,
        description="Hidden dim of MHSP scoring network (attention mode).",
    )
    num_heads: int = Field(
        default=1,
        gt=0,
        description="Number of MHSP attention heads (attention mode).",
    )
    head_dim: Optional[int] = Field(
        default=None,
        gt=0,
        description="MHSP per-head output dim; defaults to input_dim.",
    )
    projection_mode: ProjectionMode = Field(
        default="identity",
        description="MHSP projection mode (attention mode).",
    )
    target_token_idx: Optional[int] = Field(
        default=None,
        ge=0,
        description="Token index for single-token pooling; defaults to "
        "AA_SEPARATOR_IDX when pooling_type='single_token'.",
    )
    num_targets: int = Field(
        default=1,
        gt=0,
        description="Number of output targets.",
    )


class PooledMultiLabelHeadConfig(BaseHeadConfig):
    """Configuration for PooledMultiLabelHead.

    Pooled sequence-level head for binary / multi-label classification.
    Trains with BCE-with-logits loss on ``batch['binary_target']``. Covers
    binary classification at ``num_targets=1`` and multi-label at
    ``num_targets > 1``. See :class:`PooledRegressionHeadConfig` for the
    pooling-related attribute docs (identical shape).
    """

    head_type: Literal[HeadType.POOLED_MULTI_LABEL_HEAD] = (
        HeadType.POOLED_MULTI_LABEL_HEAD
    )
    pooling_type: PoolingType = Field(
        default="mean",
        description="Pooling strategy for aggregating per-residue embeddings.",
    )
    pooling_hidden_dim: int = Field(
        default=64,
        gt=0,
        description="Hidden dim of MHSP scoring network (attention mode).",
    )
    num_heads: int = Field(
        default=1,
        gt=0,
        description="Number of MHSP attention heads (attention mode).",
    )
    head_dim: Optional[int] = Field(
        default=None,
        gt=0,
        description="MHSP per-head output dim; defaults to input_dim.",
    )
    projection_mode: ProjectionMode = Field(
        default="identity",
        description="MHSP projection mode (attention mode).",
    )
    target_token_idx: Optional[int] = Field(
        default=None,
        ge=0,
        description="Token index for single-token pooling; defaults to "
        "AA_SEPARATOR_IDX when pooling_type='single_token'.",
    )
    num_targets: int = Field(
        default=1,
        gt=0,
        description="Number of output targets.",
    )


class PooledClassificationHeadConfig(BaseHeadConfig):
    """Configuration for PooledClassificationHead.

    Pooled sequence-level head for mutually-exclusive multi-class
    classification. Trains with cross-entropy on ``batch['class_label']``
    (LongTensor of class indices in ``[0, num_targets)``). Requires
    ``num_targets >= 2``; use :class:`PooledMultiLabelHeadConfig` for binary
    classification with a single output. See
    :class:`PooledRegressionHeadConfig` for the pooling-related attribute docs
    (identical shape except ``num_targets`` is constrained to ``>= 2``).
    """

    head_type: Literal[HeadType.POOLED_CLASSIFICATION_HEAD] = (
        HeadType.POOLED_CLASSIFICATION_HEAD
    )
    pooling_type: PoolingType = Field(
        default="mean",
        description="Pooling strategy for aggregating per-residue embeddings.",
    )
    pooling_hidden_dim: int = Field(
        default=64,
        gt=0,
        description="Hidden dim of MHSP scoring network (attention mode).",
    )
    num_heads: int = Field(
        default=1,
        gt=0,
        description="Number of MHSP attention heads (attention mode).",
    )
    head_dim: Optional[int] = Field(
        default=None,
        gt=0,
        description="MHSP per-head output dim; defaults to input_dim.",
    )
    projection_mode: ProjectionMode = Field(
        default="identity",
        description="MHSP projection mode (attention mode).",
    )
    target_token_idx: Optional[int] = Field(
        default=None,
        ge=0,
        description="Token index for single-token pooling; defaults to "
        "AA_SEPARATOR_IDX when pooling_type='single_token'.",
    )
    num_targets: int = Field(
        default=2,
        ge=2,
        description="Number of output classes (must be >= 2).",
    )


class WhichmutResidualHeadConfig(BaseHeadConfig):
    """Configuration for WhichmutResidualHead.

    Residual prediction head where output = zap(frozen_logits + delta).
    Delta is zero-initialized when use_residual=True so the model reproduces
    the structural prior at initialization.

    Attributes:
        logits_key: Batch dict key for frozen structural logits (default: "structural_logits").
        use_residual: If True, output = frozen_logits + delta (zero-init).
            If False, output = delta only (learned from scratch).
    """

    head_type: Literal[HeadType.WHICHMUT_RESIDUAL_HEAD] = (
        HeadType.WHICHMUT_RESIDUAL_HEAD
    )
    logits_key: str = Field(
        default="structural_logits",
        description="Batch dict key for frozen structural logits",
    )
    use_residual: bool = Field(
        default=True,
        description="Whether to add frozen logits as residual connection",
    )


class MLMHeadConfig(BaseHeadConfig):
    """Configuration for ``MLMHead`` (issue #884 / #851).

    Per-position amino-acid reconstruction head for the masked-language-model
    objective: a single ``nn.Linear(input_dim, num_amino_acids)`` projection
    with no gauge anchoring (there is no parent residue in this objective, so
    it does not subclass ``SelectionFactorOutputHead``).
    """

    head_type: Literal[HeadType.MLM_HEAD] = HeadType.MLM_HEAD
    num_amino_acids: int = Field(
        default=NUM_AMINO_ACIDS,
        gt=0,
        description="Number of standard amino acids the head reconstructs. "
        "Distinct from the encoder's vocab_size (25 for MLM configs), which "
        "also covers MASK/PAD/CLS/X/^ input tokens the head never predicts.",
    )


# Discriminated union for fail-fast validation at deserialization
HeadConfig = Annotated[
    Union[
        SelectionFactorHeadConfig,
        QDASMEvolHeadConfig,
        HybridEvolHeadConfig,
        AAHWhichmutHeadConfig,
        PerSiteChildHeadConfig,
        StructuralGuidanceHeadConfig,
        DMSFunctionalScoreHeadConfig,
        WhichmutResidualHeadConfig,
        DeltaAffinityHeadConfig,
        DASMFitResidualHeadConfig,
        PerSiteMeanSelectionHeadConfig,
        PerSiteParentChildHeadConfig,
        PerSitePreferenceHeadConfig,
        SingleValueHeadConfig,
        SingleMatrixHeadConfig,
        M0HeadConfig,
        NeutralHeadConfig,
        PooledRegressionHeadConfig,
        PooledMultiLabelHeadConfig,
        PooledClassificationHeadConfig,
        ContrastiveBindingHeadConfig,
        BindingClassificationHeadConfig,
        MLMHeadConfig,
        SiteMaskedSelectionFactorHeadConfig,
    ],
    Field(discriminator="head_type"),
]


# ──────────────────────────────────────────────────────────────────────────────
# LossConfig discriminated union (issue #621).
# Each variant maps 1:1 to a `Loss(nn.Module)` class via `create_loss` in
# `dasm2.losses.factory`. `loss_type` is the discriminator. Defaults are
# unambiguous per loss family — the migration script populates these from
# pre-#621 head_config values one-to-one.
# ──────────────────────────────────────────────────────────────────────────────


class BaseLossConfig(BaseModel):
    """Base loss configuration with discriminator."""

    loss_type: LossType = Field(description="Loss type discriminator")
    model_config = ConfigDict(extra="forbid")

    @property
    def needs_parent_nt_windows(self) -> bool:
        """Whether this loss requires the 15-nt parent-window batch fields.

        Default: False. Overridden by :class:`QDASMLossConfig` and
        :class:`HybridLossConfig` when their ``q_variant`` is a global k-mer
        lookup — ``"global_13mer"`` for either, or ``"global_5mer"``, which
        only :class:`QDASMLossConfig` offers. Those variants build Q by
        looking the neutral model up at each site's nt context, which is what
        the per-sample 15-nt window allocation in the dataset supplies; no
        other configuration needs it. Driving the decision through this
        property keeps the variant magic-strings and the ``hasattr`` on
        ``q_variant`` out of ``dasm2/train_command.py``.
        """
        return False


class DASMLossConfig(BaseLossConfig):
    """Configuration for ``DASMLoss``.

    The cross-entropy can be re-weighted by a binary
    ``{conserved, mutated}`` class budget (issue #678). The user-facing
    knob is a per-class budget summing to 1; per-position weights are
    derived as ``budget / class_freq`` so the loss-mask-count denominator
    (``trainer._compute_window_loss``) still recovers a true weighted
    mean and the effective LR is stable across runs. Class frequencies
    are computed **per batch** from the (post-syn-mask) loss mask — no
    pre-train scan and no extra config field is needed.
    """

    loss_type: Literal[LossType.DASM] = LossType.DASM
    focal_gamma: float = Field(
        default=0.0,
        ge=0.0,
        description="Focal loss focusing parameter (Lin et al., 2017). "
        "0.0 (default) is standard NLL.",
    )
    weight_mode: WeightMode = Field(
        default="uniform",
        description="How the binary {conserved, mutated} class budget is "
        "determined (issue #678). 'uniform' (default) is the current "
        "unweighted CE — no-op. 'static' uses ``mut_budget`` directly with "
        "``w_cons = 1 - mut_budget``. 'inverse_freq' uses "
        "``w_b ∝ p_b^(1-alpha)`` normalized to sum to 1, with ``p_b`` "
        "estimated from the current batch's loss mask.",
    )
    mut_budget: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Fraction of the loss budget allocated to mutated "
        "positions (child != parent) when ``weight_mode='static'``. "
        "``w_cons = 1 - mut_budget``. 0.5 is balanced.",
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Tempered inverse-frequency exponent when "
        "``weight_mode='inverse_freq'``. alpha=0 reduces to uniform "
        "(budget ∝ class size); alpha=1 is pure inverse-freq / balanced "
        "(budget = 0.5 per class); alpha=0.5 (default) is sqrt-inverse.",
    )
    syn_loss_mask: bool = Field(
        default=False,
        description="If True, exclude synonymous-mutation positions (same "
        "AA as parent, different codon) from the loss mask. Syn positions "
        "are gradient-inert by construction (parent-AA SF is zapped to 0) "
        "so this is primarily a logging / null-test control; see issue "
        "#678 §2.",
    )
    normalization: DASMNormalization = Field(
        default="parent_absorbing",
        description="Which DASM endpoint kernel normalizes the codon row "
        "(issue #1238). 'parent_absorbing' (default) leaves non-parent "
        "sense weights raw and absorbs all stop mass into the exact parent "
        "codon via ``1 - sum(others)``, so rows normalize only "
        "conditionally and stop columns sit at exp(-80). "
        "'sense_conditional' is one log_softmax over the sense-restricted "
        "weight row, so every sense destination (the parent included) is a "
        "normalized share, rows sum to 1 unconditionally, and stop "
        "probability is exactly 0. The default is preserved because the "
        "native kernel is the #1205 scientific baseline.",
    )

    @model_validator(mode="after")
    def _focal_and_weighting_are_orthogonal_for_now(self) -> "DASMLossConfig":
        """Combining focal_gamma > 0 with non-uniform class weighting is not
        yet supported: focal and class-weight are independent re-weightings of
        the same per-position NLL, but the orthogonal composition path has
        not been validated against baselines (issue #678). Until it is, we
        reject the combination at config-parse time rather than silently
        running one of the two.
        """
        if self.focal_gamma > 0.0 and self.weight_mode != "uniform":
            raise ValueError(
                "DASMLossConfig: combining ``focal_gamma > 0`` with "
                f"``weight_mode={self.weight_mode!r}`` is not supported yet "
                "(issue #678). Use exactly one of focal-difficulty weighting "
                "or class-membership weighting."
            )
        return self


class QDASMLossConfig(BaseLossConfig):
    """Configuration for ``QDASMLoss`` (Q-matrix CTMC family)."""

    loss_type: Literal[LossType.QDASM] = LossType.QDASM
    use_cubic: bool = Field(default=False)
    # ``None`` preserves whether the user stated the calibration. TrainingConfig
    # requires explicit scales when the rate source is not the calibrated
    # Thrifty model, then resolves safe omissions to the canonical defaults.
    single_hit_scale: Optional[float] = Field(default=None, gt=0.0)
    double_hit_scale: Optional[float] = Field(default=None, ge=0.0)
    triple_hit_scale: Optional[float] = Field(default=None, ge=0.0)
    adaptive: bool = Field(default=False)
    # ||tQ||_inf cutoff: _row_expmv at/below, ss_quintic above (issue #793 WI 1).
    # Default 1.0 — just under the measured row-vs-full crossover; the le=10.0
    # bound also keeps _row_expmv's sub-step count (ceil(||tQ||/0.35)) small.
    tq_threshold: float = Field(default=1.0, gt=0.0, le=10.0)
    long_path: Literal["matrix_exp", "ss_quintic"] = Field(default="ss_quintic")
    adaptive_mode: Literal["per_chunk", "per_sequence"] = Field(default="per_sequence")
    quadratic_ste: bool = Field(default=False)
    # Uniformization training surrogate (issue #838): "uniformization" takes
    # precedence over adaptive/use_cubic/quadratic_ste and dispatches to the
    # nonnegative-by-construction kernel on unscaled Q with per-matrix λ.
    matexp_method: Literal["adaptive", "uniformization"] = Field(default="adaptive")
    # Uniformization truncation order. Only used when matexp_method="uniformization".
    # Default 4 documents the "low-order" regime this issue evaluates
    # (N ∈ {2,3,4}); inert under the default matexp_method="adaptive".
    n_terms: int = Field(default=4, gt=0)
    # Uniformization parent-row renormalization convention. Only used when
    # matexp_method="uniformization"; see uniformization._normalize_parent_row.
    normalization: Literal["parent_absorbing", "proportional"] = Field(
        default="parent_absorbing"
    )
    q_variant: Literal[
        "parent_row_degenerate", "global_13mer", "global_5mer", "shm1_codon"
    ] = Field(default="parent_row_degenerate")
    thrifty_13mer_table_path: Optional[str] = Field(default=None)
    # Configured neutral-model identity expected in the 13-mer table's JSON
    # provenance sidecar. ``TrainingConfig`` fills this from ``neutral_model``;
    # the table loader compares it at loss construction (issue #959).
    thrifty_13mer_model_name: Optional[str] = Field(default=None)
    # Catalog name of the S5F model supplying the [4^5, 5] table for
    # ``q_variant="global_5mer"``. Leave unset: ``TrainingConfig`` fills it in
    # from ``neutral_model.model_name`` so the Q and the neutral model cannot
    # disagree (issue #959). Setting it to a different model is an error, not a
    # silent override.
    five_mer_model_name: Optional[str] = Field(default=None)
    # Path to the SHM-1 non-reversible 61×61 codon generator (phyz
    # nonrev_codon .dat) for ``q_variant="shm1_codon"``. ``None`` → the
    # matrix packaged at ``dasm2/neutral/data/shm1_codon_q.dat``.
    neutral_codon_q_path: Optional[str] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _migrate_bl_threshold(cls, data: Any) -> Any:
        """Map the legacy ``bl_threshold`` field onto ``tq_threshold``.

        Pre-fix YAMLs and checkpoints carried ``bl_threshold`` (BL-only
        routing); adaptive routing is now ``||tQ||_∞``-based. The numeric
        value carries over unchanged, so in-flight checkpoint resume keeps
        working (a legacy ``bl_threshold=0.1`` stays 0.1 — still correct under
        the parent-row router, just less of its speedup than the new 1.0
        default). See experiments/2026-05-06-matexp-negativity-audit/FINDINGS.md.
        """
        if (
            isinstance(data, dict)
            and "bl_threshold" in data
            and "tq_threshold" not in data
        ):
            data["tq_threshold"] = data.pop("bl_threshold")
        return data

    @model_validator(mode="after")
    def _warn_slow_full_p_matexp(self) -> "QDASMLossConfig":
        """Warn when the config silently lands on the slow exact full-P path.

        With ``adaptive=False``, ``use_cubic=False``, ``quadratic_ste=False``
        and ``matexp_method="adaptive"`` (all defaults), QDASM runs the exact
        ``torch.linalg.matrix_exp`` with Fréchet-derivative backward on the
        *full* ``[B·L, 61, 61]`` transition matrix — correct, but ~4.9× slower
        and ~1.5× the peak memory of the parent-row path, which never
        materializes the full P and matches it to ~1e-6 (issue #793 WI 1). This
        is almost never intended: it means the config just didn't opt into the
        fast path. Left as a warning rather than a flipped default because
        ``adaptive=True`` is mutually exclusive with ``use_cubic`` /
        ``quadratic_ste`` / ``matexp_method="uniformization"`` — flipping the
        default would hard-error any config that deliberately picks one of
        those. Set ``adaptive: true`` to silence this and get the fast path.
        """
        if (
            not self.adaptive
            and not self.use_cubic
            and not self.quadratic_ste
            and self.matexp_method == "adaptive"
        ):
            warnings.warn(
                "QDASMLossConfig is on the slow exact full-P matrix_exp path "
                "(adaptive=False, use_cubic=False, quadratic_ste=False, "
                "matexp_method='adaptive'): ~4.9x slower and ~1.5x the peak "
                "memory of the parent-row path, which matches it to ~1e-6. "
                "Set `adaptive: true` for the fast path (issue #793 WI 1), or "
                "pick use_cubic / quadratic_ste / matexp_method='uniformization' "
                "deliberately to silence this.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @property
    def needs_parent_nt_windows(self) -> bool:
        return self.q_variant in GLOBAL_KMER_Q_VARIANTS

    @property
    def needs_neutral_codon_q(self) -> bool:
        return self.q_variant == "shm1_codon"


class HybridLossConfig(BaseLossConfig):
    """Configuration for ``HybridLoss`` (CTMC neutral + post-hoc selection)."""

    loss_type: Literal[LossType.HYBRID] = LossType.HYBRID
    use_cubic: bool = Field(default=False)
    # ``None`` preserves whether the user stated the calibration. TrainingConfig
    # requires explicit scales when the rate source is not the calibrated
    # Thrifty model, then resolves safe omissions to the canonical defaults.
    single_hit_scale: Optional[float] = Field(default=None, gt=0.0)
    double_hit_scale: Optional[float] = Field(default=None, ge=0.0)
    triple_hit_scale: Optional[float] = Field(default=None, ge=0.0)
    q_variant: Literal["parent_row_degenerate", "global_13mer"] = Field(
        default="parent_row_degenerate"
    )
    thrifty_13mer_table_path: Optional[str] = Field(default=None)
    thrifty_13mer_model_name: Optional[str] = Field(default=None)

    @property
    def needs_parent_nt_windows(self) -> bool:
        return self.q_variant == "global_13mer"


class WhichmutLossConfig(BaseLossConfig):
    """Configuration for ``WhichmutLoss``. No tunables today."""

    loss_type: Literal[LossType.WHICHMUT] = LossType.WHICHMUT


class AAOWhichmutLossConfig(BaseLossConfig):
    """Configuration for ``AAOWhichmutLoss``. No tunables today."""

    loss_type: Literal[LossType.AAO_WHICHMUT] = LossType.AAO_WHICHMUT


class DMSFunctionalScoreLossConfig(BaseLossConfig):
    """Configuration for ``DMSFunctionalScoreLoss`` (Huber + learnable scale)."""

    loss_type: Literal[LossType.DMS_FUNCTIONAL_SCORE] = LossType.DMS_FUNCTIONAL_SCORE
    scale_init: float = Field(
        default=1.0, description="Initial value for the learnable scale parameter."
    )
    huber_delta: float = Field(default=1.0, gt=0.0)


class DeltaAffinityLossConfig(BaseLossConfig):
    """Configuration for ``DeltaAffinityLoss`` (Huber)."""

    loss_type: Literal[LossType.DELTA_AFFINITY] = LossType.DELTA_AFFINITY
    huber_delta: float = Field(default=1.0, gt=0.0)


class DMSMatrixCalibrationLossConfig(BaseLossConfig):
    """Configuration for ``DMSMatrixCalibrationLoss`` (MSE + Pearson + ListMLE)."""

    loss_type: Literal[LossType.DMS_MATRIX_CALIBRATION] = (
        LossType.DMS_MATRIX_CALIBRATION
    )
    w_mse: float = Field(default=1.0, ge=0.0, description="Weight on the MSE term.")
    w_pearson: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight on the masked Pearson-correlation term.",
    )
    w_listmle: float = Field(
        default=0.0, ge=0.0, description="Weight on the masked ListMLE ranking term."
    )
    scale_init: float = Field(
        default=1.0, description="Initial value for the learnable scale parameter."
    )
    bias_init: float = Field(
        default=0.0, description="Initial value for the learnable bias parameter."
    )
    mask_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of measured cells randomly dropped from the loss "
        "per training batch (DMSMatrixCollator.mask_rate). 0.0 is a no-op. "
        "Training split only; validation never drops cells.",
    )


class StructuralGuidanceLossConfig(BaseLossConfig):
    """Configuration for ``StructuralGuidanceLoss``."""

    loss_type: Literal[LossType.STRUCTURAL_GUIDANCE] = LossType.STRUCTURAL_GUIDANCE


class PooledRegressionLossConfig(BaseLossConfig):
    """Configuration for ``PooledRegressionLoss`` (sum-reduced Huber)."""

    loss_type: Literal[LossType.POOLED_REGRESSION] = LossType.POOLED_REGRESSION
    huber_delta: float = Field(default=1.0, gt=0.0)


class PooledMultiLabelLossConfig(BaseLossConfig):
    """Configuration for ``PooledMultiLabelLoss`` (sum-reduced BCE-with-logits)."""

    loss_type: Literal[LossType.POOLED_MULTI_LABEL] = LossType.POOLED_MULTI_LABEL


class PooledClassificationLossConfig(BaseLossConfig):
    """Configuration for ``PooledClassificationLoss`` (sum-reduced cross-entropy)."""

    loss_type: Literal[LossType.POOLED_CLASSIFICATION] = LossType.POOLED_CLASSIFICATION


class DASMFitResidualLossConfig(BaseLossConfig):
    """Configuration for ``DASMFitResidualLoss`` (Poisson residual)."""

    loss_type: Literal[LossType.DASM_FIT_RESIDUAL] = LossType.DASM_FIT_RESIDUAL
    k: int = Field(default=3, gt=0, description="Number of residual components.")


#: Which in-batch pairs the contrastive collator marks as positives.
PositiveRule = Literal["antigen_key", "sequence"]


class ContrastiveBindingLossConfig(BaseLossConfig):
    """Configuration for ``ContrastiveBindingLoss`` (CALM / SigLIP variants)."""

    loss_type: Literal[LossType.CONTRASTIVE_BINDING] = LossType.CONTRASTIVE_BINDING
    contrastive_loss_type: Literal[
        "infonce_symmetric", "infonce_ab_to_ag", "siglip"
    ] = Field(default="infonce_symmetric")
    positive_rule: PositiveRule = Field(
        default="antigen_key",
        description="Which in-batch pairs are positives. The loss only sees "
        "the resulting matrix: ``dasm2.data.contrastive_binding`` builds it "
        "(``ContrastiveBindingCollator``), so the rule lives beside the loss "
        "it feeds. 'antigen_key' (default): rows sharing an antigen_key. "
        "'sequence': rows with an identical antigen sequence OR an identical "
        "antibody (heavy + light) sequence; needs a sequence_antigen column.",
    )


class BindingClassificationLossConfig(BaseLossConfig):
    """Configuration for ``BindingClassificationLoss`` (sum-reduced BCE)."""

    loss_type: Literal[LossType.BINDING_CLASSIFICATION] = (
        LossType.BINDING_CLASSIFICATION
    )
    pos_weight: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Positive-class weight for BCE-with-logits. Typically "
        "n_neg / n_pos, computed by the training script from the dataset's "
        "label balance — this loss does not compute it itself.",
    )
    mask_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of the binding task's own real (non-padding, "
        "non-separator) input positions randomly corrupted per training "
        "batch — entity-A residues replaced with the ambiguous-residue "
        "token, entity-B residues excluded from cross-attention (issue "
        "#1221, dasm2.data.binding_corruption.BindingCorruptionConfig). "
        "0.0 (default) is a no-op. Applied to the training split only, "
        "never validation — the loss config is what train_command.py's "
        "per-head dispatch already has in hand when it builds the binding "
        "data loader, matching MLMLossConfig.mask_rate's precedent for "
        "threading a masking tunable through YAML without a new "
        "TrainingConfig field.",
    )


class WhichmutResidualLossConfig(BaseLossConfig):
    """Configuration for ``WhichmutResidualLoss`` (delegates to WhichmutLoss math)."""

    loss_type: Literal[LossType.WHICHMUT_RESIDUAL] = LossType.WHICHMUT_RESIDUAL


class MLMLossConfig(BaseLossConfig):
    """Configuration for ``MLMLoss`` (issue #884 / #851).

    ``MLMLoss`` itself has no tunables — cross-entropy over masked positions
    is parameter-free. These three fields instead carry the BERT masking
    recipe (``dasm2.data.masking.MaskingConfig``) through YAML: the loss
    config is what ``train_command.py``'s per-head dispatch already has in
    hand at the point it builds the MLM data loader, so it's the natural
    place to thread masking tunables from config to
    ``MaskedTransformCollator`` without inventing a new top-level
    ``TrainingConfig`` field with no other precedent.
    """

    loss_type: Literal[LossType.MLM] = LossType.MLM
    mask_rate: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Fraction of real (non-special) positions selected for "
        "masking per sequence.",
    )
    mask_token_prob: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Of the selected positions, fraction replaced with MASK.",
    )
    random_token_prob: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Of the selected positions, fraction replaced with a "
        "uniformly random canonical amino acid. The remainder (implied "
        "1 - mask_token_prob - random_token_prob) are left unchanged but "
        "still scored.",
    )

    @model_validator(mode="after")
    def _mask_and_random_prob_do_not_exceed_one(self) -> "MLMLossConfig":
        combined = self.mask_token_prob + self.random_token_prob
        if combined > 1.0:
            raise ValueError(
                f"MLMLossConfig: mask_token_prob + random_token_prob must be "
                f"<= 1.0, got {combined} (mask_token_prob={self.mask_token_prob}, "
                f"random_token_prob={self.random_token_prob})."
            )
        return self


LossConfig = Annotated[
    Union[
        DASMLossConfig,
        QDASMLossConfig,
        HybridLossConfig,
        WhichmutLossConfig,
        AAOWhichmutLossConfig,
        DMSFunctionalScoreLossConfig,
        DeltaAffinityLossConfig,
        DMSMatrixCalibrationLossConfig,
        StructuralGuidanceLossConfig,
        PooledRegressionLossConfig,
        PooledMultiLabelLossConfig,
        PooledClassificationLossConfig,
        DASMFitResidualLossConfig,
        ContrastiveBindingLossConfig,
        BindingClassificationLossConfig,
        WhichmutResidualLossConfig,
        MLMLossConfig,
    ],
    Field(discriminator="loss_type"),
]


class TaskConfig(BaseModel):
    """v4 declarative task entry: pairs a head with a loss + weight.

    Introduced by issue #621 Task #6. The v4 ``TrainingConfig.tasks`` block
    holds one ``TaskConfig`` per (head, loss) pair the trainer should
    construct. ``head_type`` points at one of the entries in
    ``model.heads`` (deduplicated — AAH yields two ``TaskConfig`` entries
    sharing the same ``head_type`` pointer, one per loss).

    Pre-v4 YAMLs must be rewritten by ``scripts/migrate_yaml_to_v4.py``
    before they can be loaded.
    """

    head_type: HeadType = Field(
        description="HeadType pointer; must match one of model.heads[*].head_type."
    )
    loss_config: LossConfig = Field(
        description="Loss configuration (discriminated union, ``loss_type`` field)."
    )
    loss_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Multiplicative weight applied to this task's loss. 0.0 is "
        "valid (not just >0): OverfitGuardHook (issue #1096) legitimately zeroes "
        "a live Task's loss_weight at runtime, and Trainer.export_metadata() "
        "round-trips that live value back into a TaskConfig, so the schema must "
        "tolerate it or checkpoint/cache saves crash after the guard fires.",
    )
    batch_size: Optional[int] = Field(
        default=None,
        gt=0,
        description="Per-device training batch size for this task's loader only. "
        "None (default) falls back to TrainingConfig.batch_size. Lower this when a "
        "co-trained task's dataset is much smaller than the others', so its loader "
        "does not replay its rows many more times per step budget than the rest. "
        "Does not affect validation batch size.",
    )
    eval_hooks: List["HookConfig"] = Field(
        default_factory=list,
        description="Per-task eval hooks (e.g. OverfitGuardHook), instantiated via "
        "create_hook() and appended to this task's Task.eval_hooks alongside any "
        "hooks train_command.py attaches procedurally (e.g. checkpointing on the "
        "last task).",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _single_value_head_has_no_gradient_under_sense_conditional(
        self,
    ) -> "TaskConfig":
        """Reject ``SingleValueHead`` + ``DASMLoss(normalization='sense_conditional')``.

        The conditional kernel normalizes over the sense-restricted weight
        row, which makes it exactly invariant to adding a constant to all 20
        log selection factors. ``SingleValueHead`` emits nothing *but* such a
        constant (one scalar broadcast to every site and AA, with no diagonal
        zap by design), so its single parameter lies entirely in the kernel's
        flat direction and its gradient is float32 noise — measured at
        -5.4e-07 under ``sense_conditional`` against -5.0 under
        ``parent_absorbing`` (issue #1238 §2.1). The head's own docstring
        says the scalar trains *because* of the parent-absorb
        parameterization; that reasoning does not survive this kernel.

        Adding a diagonal zap would not rescue it — a zapped constant is
        identically zero, leaving the head equally dead and less obviously
        so. Heads that do zap (``M0Head``, ``SingleMatrixHead``) keep
        nonzero gradients and are unaffected by this rejection.
        """
        if (
            self.head_type is HeadType.SINGLE_VALUE_HEAD
            and isinstance(self.loss_config, DASMLossConfig)
            and self.loss_config.normalization == "sense_conditional"
        ):
            raise ValueError(
                "TaskConfig: head_type=single_value_head cannot be paired with "
                "DASMLossConfig(normalization='sense_conditional') (issue #1238). "
                "The sense-conditional kernel is invariant to a global shift of "
                "all 20 log selection factors, and SingleValueHead emits nothing "
                "but a global shift, so the head's one parameter has no gradient "
                "under this kernel. Use normalization='parent_absorbing' for this "
                "floor baseline, or pick a head with at least one identifiable "
                "direction (e.g. m0_head or single_matrix_head, both of which "
                "zap the parent-AA diagonal)."
            )
        return self


class HookConfig(BaseModel):
    """Base configuration for training hooks with flexible parameters.

    All fields except 'type' are passed as constructor parameters to the hook.
    This allows hooks to define their own parameters without requiring config subclasses.
    """

    type: str = Field(description="Hook type identifier")

    # Allow extra fields - they become hook constructor parameters
    model_config = ConfigDict(extra="allow")


class OptimizerConfig(BaseModel):
    """Optimizer configuration with validation."""

    optimizer_name: str = Field(default="RMSprop", description="Optimizer type")
    learning_rate: float = Field(
        gt=0.0, le=1.0, default=1e-3, description="Learning rate"
    )
    weight_decay: float = Field(ge=0.0, default=1e-6, description="Weight decay")
    betas: Optional[Tuple[float, float]] = Field(
        default=None, description="Adam/AdamW beta parameters (not used for RMSprop)"
    )

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Validate learning rate is within reasonable bounds for training stability."""
        if not 1e-6 <= v <= 1e-1:
            raise ValueError(f"Learning rate {v} outside reasonable range [1e-6, 1e-1]")
        return v

    @field_validator("betas", mode="before")
    @classmethod
    def validate_betas(cls, v: Any) -> Optional[Tuple[float, float]]:
        """Convert list to tuple for YAML compatibility.

        PyYAML's safe_load returns lists, so convert back to tuple on load.
        """
        if v is None:
            return None
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError("betas must contain exactly 2 values")
            return (float(v[0]), float(v[1]))
        if isinstance(v, tuple):
            return v
        raise ValueError(f"betas must be a list or tuple, got {type(v).__name__}")

    @field_serializer("betas")
    def serialize_betas(
        self, betas: Optional[Tuple[float, float]]
    ) -> Optional[List[float]]:
        """Serialize betas tuple as list for YAML compatibility.

        PyYAML's safe_load can't handle !!python/tuple tags, so convert to list.
        """
        if betas is None:
            return None
        return list(betas)


class ReduceOnPlateauSchedulerConfig(BaseModel):
    """ReduceLROnPlateau scheduler configuration.

    Steps on each evaluation's val loss. ``eval_patience`` counts
    evaluations (not epochs): at ``eval_every_steps=500`` an
    ``eval_patience=10`` waits 5000 steps before reducing. The field is
    deliberately named ``eval_patience`` rather than ``patience`` so a
    pre-#578 YAML carrying ``patience: N`` (epoch semantics) fails with
    ``extra="forbid"`` instead of silently reinterpreting the value
    under eval semantics.
    """

    scheduler_type: Literal["reduce_on_plateau"] = "reduce_on_plateau"
    factor: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Multiplicative factor applied to the LR on plateau.",
    )
    eval_patience: int = Field(
        default=10,
        gt=0,
        description="Number of evaluations with no improvement before reducing LR.",
    )

    model_config = ConfigDict(extra="forbid")


class WarmupPolynomialSchedulerConfig(BaseModel):
    """Linear warmup followed by polynomial decay (per-step stepping).

    Requires ``total_training_steps`` to be computed at training time.
    """

    scheduler_type: Literal["warmup_polynomial"] = "warmup_polynomial"
    warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Number of linear warmup steps before polynomial decay begins",
    )
    polynomial_power: float = Field(
        default=2.0,
        gt=0.0,
        description="Polynomial decay power",
    )
    warmup_start_factor: float = Field(
        default=1e-8,
        gt=0.0,
        lt=1.0,
        description="Initial LR multiplier at the start of warmup. "
        "LR ramps linearly from lr * warmup_start_factor to lr over warmup_steps.",
    )

    model_config = ConfigDict(extra="forbid")


class CosineWarmRestartsSchedulerConfig(BaseModel):
    """Cosine annealing with warm restarts (per-step stepping).

    The LR follows a half cosine from the configured LR down to ``eta_min``
    over ``t_0_steps`` optimizer steps, then jumps back to the configured LR
    and anneals again; each period is ``t_mult`` times longer than the last
    (``torch.optim.lr_scheduler.CosineAnnealingWarmRestarts``). Unlike
    ``reduce_on_plateau`` it does not look at the validation loss.
    """

    scheduler_type: Literal["cosine_warm_restarts"] = "cosine_warm_restarts"
    t_0_steps: int = Field(
        gt=0,
        description="Length of the first annealing period, in optimizer steps.",
    )
    t_mult: int = Field(
        default=1,
        ge=1,
        description="Each period is this many times longer than the previous one.",
    )
    eta_min: float = Field(
        default=0.0,
        ge=0.0,
        description="Learning rate at the bottom of each period. Keep it below "
        "the optimizer's learning_rate, or the LR rises instead of decaying.",
    )

    model_config = ConfigDict(extra="forbid")


# Discriminated union for fail-fast validation at deserialization.
# Invalid combinations (e.g., warmup_steps with reduce_on_plateau) are
# rejected at parse time rather than silently ignored.
SchedulerConfig = Annotated[
    Union[
        ReduceOnPlateauSchedulerConfig,
        WarmupPolynomialSchedulerConfig,
        CosineWarmRestartsSchedulerConfig,
    ],
    Field(discriminator="scheduler_type"),
]


class BaseNeutralModelConfig(BaseModel):
    """Base neutral model configuration with common fields.

    All neutral models share the type discriminator. Specific model types extend
    this with additional required fields. Uses discriminated union pattern for
    fail-fast validation at deserialization time.
    """

    type: NeutralModelType = Field(description="Neutral model type discriminator")

    model_config = ConfigDict(extra="forbid")


class MultihitWrappableNeutralModelConfig(BaseNeutralModelConfig):
    """Base for neutral model configs that support ``MultihitWrapper`` composition.

    Only rates/CSPs-decomposition models (ThriftySHM, TabularRates, S5F,
    PrecomputedRates) can be wrapped for multi-hit correction —
    ``GTRCodonNeutralModel`` handles
    multi-hit exactly via matrix exponentiation and deliberately does not
    inherit from this class (see ``MultihitWrapper``'s docstring). Sharing
    this base (rather than redeclaring ``multihit_model_name`` on each
    config) keeps `_maybe_wrap_multihit`'s signature and
    ``MultihitWrapper.to_config()``'s type check in sync with which configs
    actually support wrapping — a config added here is automatically both
    wrappable and recognized by ``to_config()``.
    """

    multihit_model_name: Optional[str] = Field(
        default=None,
        description="Name of a pretrained multihit-correction model (e.g., "
        "'ThriftyHumV0.2-59-hc-tangshm') applied via MultihitWrapper.",
    )
    multihit_log_factors: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="The multihit model's actual current log hit-class "
        "factors (hit classes 1-3), as of save time. Set by "
        "MultihitWrapper.to_config() from the live HitClassModel — not just "
        "the pretrained registry value named by multihit_model_name. Takes "
        "precedence over multihit_model_name at reload time so a checkpoint "
        "refit by issue #501's MultihitOptimizer hook reloads with the "
        "fitted factors instead of reverting to the pretrained triple "
        "(trap 5); multihit_model_name is kept for provenance only.",
    )


class ThriftySHMNeutralModelConfig(MultihitWrappableNeutralModelConfig):
    """Configuration for ThriftySHMModel resolved from the dasm2 model catalog.

    ``model_name`` references a catalog entry (e.g., ``ThriftyHumV0.2-59``) in
    ``model_catalog.yaml``, resolved to a zip archive via ``resolve_model()``.
    Supports optional MultihitWrapper for codon clustering correction.
    """

    type: Literal[NeutralModelType.THRIFTY_SHM] = NeutralModelType.THRIFTY_SHM
    model_name: str = Field(
        default="ThriftyHumV0.2-59",
        description="Catalog model name (resolved via model_catalog.yaml)",
    )

    boundary_table_path: Optional[str] = Field(
        default=None,
        description="Custom observed-context boundary table; defaults to the shared common-v1 table. Required only for global-kmer Q data.",
    )


class TabularRatesNeutralModelConfig(MultihitWrappableNeutralModelConfig):
    """Configuration for TabularRatesNeutralModel with tabular rate data.

    Supports two modes:

    1. **Portable name**: provide ``model_name`` referencing a catalog entry
       (e.g., ``TabularRates-SCoV2-v1``). Resolved via ``model_catalog.yaml``.
    2. **Explicit path**: provide ``model_path`` to a zip archive created by
       ``TabularRatesNeutralModel.save_pretrained()``.

    Exactly one of ``model_name`` or ``model_path`` must be specified.

    Examples:
        >>> # By catalog name (portable)
        >>> config = TabularRatesNeutralModelConfig(model_name="TabularRates-SCoV2-v1")
        >>> # By explicit path
        >>> config = TabularRatesNeutralModelConfig(
        ...     model_path="/path/to/tabular_rates.zip"
        ... )
    """

    type: Literal[NeutralModelType.TABULAR_RATES] = NeutralModelType.TABULAR_RATES
    model_name: Optional[str] = Field(
        default=None,
        description="Catalog model name (resolved via model_catalog.yaml)",
    )
    model_path: Optional[Path] = Field(
        default=None,
        description="Path to zip archive from save_pretrained()",
    )

    @model_validator(mode="after")
    def validate_model_name_or_path(self) -> "TabularRatesNeutralModelConfig":
        """Validate that exactly one of model_name or model_path is specified."""
        num_set = sum([self.model_name is not None, self.model_path is not None])
        if num_set != 1:
            raise ValueError(
                "Exactly one of 'model_name' or 'model_path' must be specified, "
                f"got {num_set} of them"
            )
        return self


class S5FNeutralModelConfig(MultihitWrappableNeutralModelConfig):
    """Configuration for ``S5FNeutralModel`` (five-mer SHM neutral model).

    S5F is a pure five-mer lookup table (netam's ``RSFivemerModel``), usable as
    a drop-in neutral model in place of Thrifty. Mirrors ``TabularRates``:

    1. **Portable name**: provide ``model_name`` referencing a catalog entry
       (e.g., ``S5F-HH-classic-v1``). Resolved via ``model_catalog.yaml``.
    2. **Explicit path**: provide ``model_path`` to a zip archive created by
       ``S5FNeutralModel.save_pretrained()``.

    Exactly one of ``model_name`` or ``model_path`` must be specified.
    Inheriting ``MultihitWrappableNeutralModelConfig`` grants multihit
    correction (``multihit_model_name``) for free.
    """

    type: Literal[NeutralModelType.S5F] = NeutralModelType.S5F
    model_name: Optional[str] = Field(
        default=None,
        description="Catalog model name (resolved via model_catalog.yaml)",
    )
    model_path: Optional[Path] = Field(
        default=None,
        description="Path to zip archive from save_pretrained()",
    )

    @model_validator(mode="after")
    def validate_model_name_or_path(self) -> "S5FNeutralModelConfig":
        """Validate that exactly one of model_name or model_path is specified."""
        num_set = sum([self.model_name is not None, self.model_path is not None])
        if num_set != 1:
            raise ValueError(
                "Exactly one of 'model_name' or 'model_path' must be specified, "
                f"got {num_set} of them"
            )
        return self


class PrecomputedRatesNeutralModelConfig(MultihitWrappableNeutralModelConfig):
    """Configuration for ``PrecomputedRatesNeutralModel`` (multihit carrier).

    Expresses "rates come from the data, but do apply this multihit
    correction" — the configuration Compara-style Parquet corpora (per-family
    GTR rates baked into ``nt_rates``/``nt_csps``, ``neutral_model: null``)
    previously had no way to write, since ``multihit_model_name`` exists only
    inside a neutral-model block. Inheriting
    ``MultihitWrappableNeutralModelConfig`` is the whole point: it is what
    grants the multihit fields and makes ``MultihitWrapper.to_config()``
    round-trip the live factors into checkpoint metadata.

    Parameter-free otherwise — the model computes no rates and raises if asked
    to (see ``dasm2/neutral/precomputed_rates.py``), so it is only valid on
    rate-precomputed (Parquet) data.

    Examples:
        >>> config = PrecomputedRatesNeutralModelConfig(
        ...     multihit_model_name="ThriftyHumV0.2-59-hc-tangshm"
        ... )
    """

    type: Literal[NeutralModelType.PRECOMPUTED_RATES] = (
        NeutralModelType.PRECOMPUTED_RATES
    )


class GTRCodonNeutralModelConfig(BaseNeutralModelConfig):
    """Configuration for GTRCodonNeutralModel with GTR parameters.

    Supports three modes:

    1. **Portable name**: provide ``model_name`` referencing a catalog entry
       (e.g., ``K80-thrifty-pcps-v1``). Resolved via ``model_catalog.yaml``.
    2. **Load from archive**: provide ``model_path`` to a zip archive created by
       ``GTRCodonNeutralModel.save_pretrained()``.
    3. **Inline specification**: provide ``variant`` and any required parameters
       to construct the model directly.

    Exactly one of ``model_name``, ``model_path``, or ``variant`` must be specified.

    Inline variants and their required parameters:

    - ``jc69``: No additional parameters (equal rates, equal frequencies).
    - ``k80``: Requires ``kappa`` (transition/transversion ratio).
    - ``hky``: Requires ``kappa`` and ``frequencies`` (length 4).
    - ``gtr``: Requires ``exchangeabilities`` (length 6) and ``frequencies`` (length 4).

    Examples:
        >>> # By catalog name (portable)
        >>> config = GTRCodonNeutralModelConfig(model_name="K80-thrifty-pcps-v1")
        >>> # From archive
        >>> config = GTRCodonNeutralModelConfig(model_path="/path/to/gtr.zip")
        >>> # Inline JC69
        >>> config = GTRCodonNeutralModelConfig(variant="jc69")
        >>> # Inline K80 with kappa
        >>> config = GTRCodonNeutralModelConfig(variant="k80", kappa=2.0)
    """

    type: Literal[NeutralModelType.GTR_CODON] = NeutralModelType.GTR_CODON

    # Option 1: Portable catalog name
    model_name: Optional[str] = Field(
        default=None,
        description="Catalog model name (resolved via model_catalog.yaml)",
    )

    # Option 2: Load from archive (existing)
    model_path: Optional[Path] = Field(
        default=None,
        description="Path to zip archive from save_pretrained()",
    )

    # Option 3: Inline specification
    variant: Optional[Literal["jc69", "k80", "hky", "gtr"]] = Field(
        default=None,
        description="GTR variant shorthand: jc69, k80, hky, or gtr",
    )
    kappa: Optional[float] = Field(
        default=None,
        description="Transition/transversion ratio for k80/hky variants",
    )
    frequencies: Optional[List[float]] = Field(
        default=None,
        description="Nucleotide frequencies [A, C, G, T] for hky/gtr (length 4)",
    )
    exchangeabilities: Optional[List[float]] = Field(
        default=None,
        description="GTR exchangeability parameters (length 6) for full gtr variant",
    )

    @model_validator(mode="after")
    def validate_model_name_path_or_variant(self) -> "GTRCodonNeutralModelConfig":
        """Validate that exactly one of model_name, model_path, or variant is specified."""
        has_name = self.model_name is not None
        has_path = self.model_path is not None
        has_variant = self.variant is not None
        num_set = sum([has_name, has_path, has_variant])
        if num_set != 1:
            raise ValueError(
                "Exactly one of 'model_name', 'model_path', or 'variant' must be "
                f"specified, got {num_set} of them"
            )
        if has_variant:
            self._validate_variant_params()
        return self

    def _validate_variant_params(self) -> None:
        """Validate parameters for inline variant specification."""
        variant = self.variant
        if variant == "jc69":
            if any(
                p is not None
                for p in [self.kappa, self.frequencies, self.exchangeabilities]
            ):
                raise ValueError("jc69 variant takes no additional parameters")
        elif variant == "k80":
            if self.kappa is None:
                raise ValueError("k80 variant requires 'kappa' parameter")
            if self.frequencies is not None or self.exchangeabilities is not None:
                raise ValueError("k80 variant only accepts 'kappa' parameter")
        elif variant == "hky":
            if self.kappa is None or self.frequencies is None:
                raise ValueError(
                    "hky variant requires 'kappa' and 'frequencies' parameters"
                )
            if len(self.frequencies) != 4:
                raise ValueError(
                    f"frequencies must have length 4, got {len(self.frequencies)}"
                )
            if self.exchangeabilities is not None:
                raise ValueError("hky variant does not accept 'exchangeabilities'")
        elif variant == "gtr":
            if self.exchangeabilities is None or self.frequencies is None:
                raise ValueError(
                    "gtr variant requires 'exchangeabilities' and 'frequencies'"
                )
            if len(self.frequencies) != 4:
                raise ValueError(
                    f"frequencies must have length 4, got {len(self.frequencies)}"
                )
            if len(self.exchangeabilities) != 6:
                raise ValueError(
                    f"exchangeabilities must have length 6, "
                    f"got {len(self.exchangeabilities)}"
                )


class PerFamilyGTRNeutralModelConfig(BaseNeutralModelConfig):
    """Configuration for ``PerFamilyGTRNeutralModel`` (issue #1299).

    One GTR model per family, chosen by each row's ``sample_id``. This is how
    a rate-free Compara parquet gets its per-family GTR+F rates at load. The
    model reads ``sample_id`` from each row, so it works only on Parquet
    data, whose loader carries that column into each sample.

    Exactly one of:

    1. **Portable name**: ``model_name``, a catalog entry resolved via
       ``model_catalog.yaml``.
    2. **Archive**: ``model_path``, a zip written by
       ``PerFamilyGTRNeutralModel.save_pretrained()``
       (``scripts/build_per_family_gtr_archive.py`` builds one from a GTR
       results TSV).

    There is deliberately no TSV option, so every checkpoint records an
    archive or a catalog name. Not multihit-wrappable, like
    ``GTRCodonNeutralModelConfig``.

    Examples:
        >>> config = PerFamilyGTRNeutralModelConfig(
        ...     model_path="/path/to/per-family-gtr-v1.zip"
        ... )
    """

    type: Literal[NeutralModelType.PER_FAMILY_GTR] = NeutralModelType.PER_FAMILY_GTR
    model_name: Optional[str] = Field(
        default=None,
        description="Catalog model name (resolved via model_catalog.yaml)",
    )
    model_path: Optional[Path] = Field(
        default=None,
        description="Path to zip archive from save_pretrained()",
    )
    tandem_delta: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Fixed tandem rate δ of each adjacent codon-position pair (1–2, "
            "2–3), in units of √(r_p r_q) (issue #1683; dasm2.molevol.tandem). "
            "0 turns the term off."
        ),
    )
    tandem_delta13: float = Field(
        default=0.0,
        ge=0.0,
        description="Fixed tandem rate δ_13 of the outer pair (1–3).",
    )

    @model_validator(mode="after")
    def validate_model_name_or_path(self) -> "PerFamilyGTRNeutralModelConfig":
        """Validate that exactly one of model_name or model_path is specified."""
        num_set = sum([self.model_name is not None, self.model_path is not None])
        if num_set != 1:
            raise ValueError(
                "Exactly one of 'model_name' or 'model_path' must be specified, "
                f"got {num_set} of them"
            )
        return self


class ShmCodonNeutralModelConfig(BaseNeutralModelConfig):
    """Configuration for ``ShmCodonNeutralModel`` (SHM-1 codon CTMC).

    SHM-1 is Erick's context-free, non-reversible 61×61 empirical codon rate
    matrix estimated from BCR out-of-frame parent–child pairs (the phyz
    ``EmpCodon``/``SHM1`` preset). Being a native CTMC generator, it
    co-calibrates with QDASM's ``exp(tQ)`` Q construction without the Poisson
    correction Thrifty needs, and is context-free (no 13-mer lookup).

    ``dat_path`` points at a phyz ``nonrev_codon`` codon-Q .dat file; ``None``
    uses the matrix packaged at ``dasm2/neutral/data/shm1_codon_q.dat``.
    """

    type: Literal[NeutralModelType.SHM_CODON] = NeutralModelType.SHM_CODON
    dat_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to a phyz nonrev_codon codon-Q .dat file. "
            "None → the packaged SHM-1 matrix."
        ),
    )


# Discriminated union for fail-fast validation at deserialization
NeutralModelConfig = Annotated[
    Union[
        ThriftySHMNeutralModelConfig,
        TabularRatesNeutralModelConfig,
        S5FNeutralModelConfig,
        PrecomputedRatesNeutralModelConfig,
        GTRCodonNeutralModelConfig,
        PerFamilyGTRNeutralModelConfig,
        ShmCodonNeutralModelConfig,
    ],
    Field(discriminator="type"),
]


class SelectionFactorProjectionConfig(BaseModel):
    """Configuration for the shared selection factor projection.

    Projects encoder embeddings to L x 20 log selection factors. Used when
    SF-consuming heads (DASM, Whichmut, etc.) share a single projection.
    """

    input_dim: Optional[int] = Field(
        default=None,
        gt=0,
        description="Input dimension (auto-set to encoder d_model if not specified)",
    )
    num_amino_acids: int = Field(
        default=20,
        gt=0,
        description="Number of amino acids to predict",
    )
    wiggle_beta: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Beta for wiggle activation. Applies identically to all loss paths. "
        "Set to 0.3 for netam-compatible behavior, or leave None to disable.",
    )
    link: Optional[Literal["phi"]] = Field(
        default=None,
        description="Link from the projection's output, the site fitness F, to "
        "log f. 'phi' gives log f(y) = log phi(F_y - F_parent), phi(S) = S / (1 - exp(-S)), the "
        "Halpern-Bruno fixation factor (issue #1583). None keeps the wiggle "
        "path. Mutually exclusive with wiggle_beta.",
    )

    @model_validator(mode="after")
    def validate_link_excludes_wiggle(self) -> "SelectionFactorProjectionConfig":
        """Refuse ``link`` together with ``wiggle_beta``: both set the link."""
        if self.link is not None and self.wiggle_beta is not None:
            raise ValueError(
                f"sf_projection.link={self.link!r} and wiggle_beta="
                f"{self.wiggle_beta} both set the link from F to log f; "
                "set at most one."
            )
        return self

    model_config = ConfigDict(extra="forbid")


class ModelConfig(BaseModel):
    """Model architecture configuration (encoder + heads + optional SF projection)."""

    encoder: Union[
        EncoderConfig,
        ESM2EncoderConfig,
        MPNNLightAttentionEncoderConfig,
        IdentityEncoderConfig,
    ]
    heads: List[HeadConfig]
    sf_projection: Optional[SelectionFactorProjectionConfig] = Field(
        default=None,
        description="Shared selection factor projection config. Required when "
        "any head has consumes_shared_sf=True (DASM, Whichmut, AAO Whichmut).",
    )
    max_aa_seq_len: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum amino acid sequence length. Required when using "
        "PerSiteParentChildHead or PerSitePreferenceHead.",
    )

    @model_validator(mode="after")
    def validate_heads(self) -> "ModelConfig":
        """Validate heads for input dimension compatibility.

        Raises:
            ValueError: If input dimensions don't match encoder
        """
        encoder_dim = self.encoder.d_model

        # Validate/auto-set input dimensions for heads that need it
        for head_config in self.heads:
            if head_config.input_dim is None:
                head_config.input_dim = encoder_dim
            elif head_config.input_dim != encoder_dim:
                raise ValueError(
                    f"Head '{head_config.head_type.value}' input_dim ({head_config.input_dim}) != "
                    f"encoder output dimension ({encoder_dim})"
                )

        # StructuralGuidanceHead is an identity forward into a HERMES-aligned
        # cosine loss; the encoder's d_model must match HERMES_EMBED_DIM (256).
        # Without this fail-fast a config with the wrong d_model would build
        # cleanly and only crash inside StructuralGuidanceLoss on the first step.
        from dasm2.losses.structural_guidance import HERMES_EMBED_DIM

        if (
            any(hc.head_type == HeadType.STRUCTURAL_GUIDANCE_HEAD for hc in self.heads)
            and encoder_dim != HERMES_EMBED_DIM
        ):
            raise ValueError(
                f"StructuralGuidanceHead requires encoder d_model={HERMES_EMBED_DIM} "
                f"(HERMES embedding dimension); got encoder d_model={encoder_dim}."
            )

        # Auto-set max_aa_seq_len for heads that declare they need it via
        # the requires_max_aa_seq_len ClassVar.  Looking it up on the head
        # class (via HEAD_REGISTRY) instead of hasattr on the config avoids
        # silently auto-populating unrelated future configs that happen to
        # gain a max_aa_seq_len field.  The ClassVar gate narrows the union
        # type for us, but mypy can't see that — so use getattr/setattr.
        from dasm2.heads.factory import HEAD_REGISTRY

        for head_config in self.heads:
            head_cls = HEAD_REGISTRY.get(head_config.head_type)
            if head_cls is None or not head_cls.requires_max_aa_seq_len:
                continue
            if getattr(head_config, "max_aa_seq_len", None) is None:
                if self.max_aa_seq_len is None:
                    raise ValueError(
                        f"Head '{head_config.head_type.value}' requires max_aa_seq_len "
                        f"but it is not set on ModelConfig"
                    )
                setattr(head_config, "max_aa_seq_len", self.max_aa_seq_len)

        self._validate_site_masked_heads()

        # Auto-set SF projection input_dim
        if self.sf_projection is not None:
            if self.sf_projection.input_dim is None:
                self.sf_projection.input_dim = encoder_dim
            elif self.sf_projection.input_dim != encoder_dim:
                raise ValueError(
                    f"sf_projection input_dim ({self.sf_projection.input_dim}) != "
                    f"encoder output dimension ({encoder_dim})"
                )

        return self

    def _validate_site_masked_heads(self) -> None:
        """Check the heads, encoder and projection a site-masked DASM needs.

        The masked head must be the model's only head (#1550), it feeds the
        mask token, index ``MLM_MASK_IDX = 22``, to a plain
        TransformerEncoder, and its F has no defined wiggle.
        """
        # Function-local, as in validate_heads: dasm2.heads imports this module.
        from dasm2.heads.factory import HEAD_REGISTRY
        from dasm2.sequence.mlm_constants import MLM_MASK_IDX

        if not any(
            (head_cls := HEAD_REGISTRY.get(hc.head_type)) is not None
            and head_cls.requires_site_masked_core_model
            for hc in self.heads
        ):
            return
        if len(self.heads) != 1:
            raise ValueError(
                "SiteMaskedSelectionFactorHead must be the model's only head: "
                "the model applies log phi to F itself, so no other head can "
                "share its masked passes. Got "
                f"{[hc.head_type.value for hc in self.heads]}."
            )
        if not isinstance(self.encoder, EncoderConfig):
            raise ValueError(
                "SiteMaskedSelectionFactorHead requires a TransformerEncoder "
                f"(EncoderConfig), got {type(self.encoder).__name__}."
            )
        if self.encoder.vocab_size <= MLM_MASK_IDX:
            raise ValueError(
                "SiteMaskedSelectionFactorHead needs the mask token at index "
                f"{MLM_MASK_IDX}, so encoder.vocab_size must be at least "
                f"{MLM_MASK_IDX + 1}; got {self.encoder.vocab_size}."
            )
        if (
            self.sf_projection is not None
            and self.sf_projection.wiggle_beta is not None
        ):
            raise ValueError(
                "SiteMaskedSelectionFactorHead has no wiggle: wiggle acts on "
                "log f, and the masked model's projection outputs F. Leave "
                "sf_projection.wiggle_beta unset."
            )
        if self.sf_projection is not None and self.sf_projection.link is not None:
            raise ValueError(
                "SiteMaskedSelectionFactorHead applies log phi itself, and its "
                "forward never reads sf_projection.link. Leave it unset."
            )

    model_config = ConfigDict(extra="forbid")


class CvSplitConfig(BaseModel):
    """On-the-fly cross-validation fold selection.

    When set on ``TrainingConfig``, the loader assigns each row to a
    fold and returns the rows with ``fold == fold_index`` as validation
    and the rest as training. The original ``in_train`` column (if any)
    and the registry's ``holdout_samples`` fallback are both ignored.

    Two splitting strategies, selected by ``group_col``:

    - ``group_col=None`` (default): row-positional draw. Each row is
      assigned a fold by ``np.random.default_rng(seed).integers(num_folds)``.
      Independent rows; clades / families / sample_ids will be scattered
      across folds. Suitable for viral datasets where row-level CV is
      meaningful.

    - ``group_col="<column>"``: ``sklearn.model_selection.GroupKFold`` on
      the named column. Every row sharing a group ends up in the same
      fold, so train and val are disjoint at the group level.
      ``seed`` is ignored in this mode (sklearn's ``GroupKFold`` is
      deterministic: it sorts unique group labels lexicographically and
      distributes them across folds by descending group size, so the
      assignment depends on the set of group labels and their row
      counts, not on the row order in the file). Required when
      group-level leakage would corrupt the metric (e.g. BCR data
      grouped by ``sample_id``).

    Folds are fully determined by ``(group_col, num_folds, fold_index,
    seed, N)`` (row-level mode) or ``(group_col, num_folds, fold_index,
    {(group_label, group_size)})`` (group mode), so the same dataset +
    spec yields bit-identical splits across runs.

    One documented exception: the pairwise-binding-classification loader
    (``dasm2.data.pairwise_binding.load_pairwise_binding_train_val``)
    ignores this class's ``group_col``/``seed`` entirely when ``cv_split``
    is set, using only ``num_folds``/``fold_index`` (plus
    ``binding_split_strategy`` below). It always splits on its own fixed
    grouping column (``antigen_cluster_40``, matching that dataset's own
    build script) — via ``StratifiedGroupKFold``
    (``binding_split_strategy="cluster"``, disjoint groups across folds) or
    ``StratifiedKFold`` on ``(antigen_cluster_40, label)`` jointly
    (``binding_split_strategy="random"``, a group can appear in both train
    and val) — so two datasets can be trained jointly under one shared
    ``cv_split`` even though their natural grouping columns mean different
    things (e.g. the binding dataset's antigen clusters vs. an evolutionary
    dataset's ``sample_id``).

    A second documented exception:
    ``dasm2.data.dms_matrix_dataset.DMSMatrixDataset.split_dataset`` also
    ignores this class's ``group_col``, always splitting on its own fixed
    ``cluster_id`` column (via ``group_kfold_in_train``, applied twice: once
    for a fixed held-out test set, once for ``cv_split``'s own
    ``num_folds``/``fold_index`` train/val split within the remaining pool).

    A third, opt-in exception: ``val_num_folds`` (read only by
    ``dasm2.data.contrastive_binding.load_contrastive_binding_train_val``)
    turns ``fold_index`` into a held-out TEST fold and carves the validation
    set from the remaining groups with an inner ``GroupKFold``; see the
    field's description.
    """

    num_folds: int = Field(
        gt=1,
        description="Total number of CV folds. Must be > 1.",
    )
    fold_index: int = Field(
        ge=0,
        description="Index of the held-out fold (0..num_folds-1).",
    )
    seed: int = Field(
        default=42,
        description="RNG seed for row-level fold assignment. Ignored "
        "when group_col is set.",
    )
    group_col: Optional[str] = Field(
        default=None,
        description="Optional column name. When set, GroupKFold is used "
        "so every row sharing this column value lands in the same fold "
        "(no group-level leakage between train and val). When None, "
        "fold assignment is row-level via the seeded RNG.",
    )
    binding_split_strategy: Literal["cluster", "random"] = Field(
        default="cluster",
        description="Read only by dasm2.data.pairwise_binding.load_pairwise_binding_train_val. "
        "'cluster' (default) holds antigen families out across folds "
        "(StratifiedGroupKFold). 'random' stratifies on (antigen_cluster_40, label) "
        "instead (StratifiedKFold), matching the SAbDab grid's own 'random' condition — "
        "a family can then appear in both train and val.",
    )
    val_num_folds: Optional[int] = Field(
        default=None,
        gt=1,
        description="Read only by dasm2.data.contrastive_binding.load_contrastive_binding_train_val. "
        "None (default): fold_index is the validation fold, as for every other "
        "loader. Set: fold_index is instead a TEST fold that training never "
        "loads, and validation is inner fold 0 of a GroupKFold with this many "
        "folds over the remaining groups, so early stopping and checkpoint "
        "selection never see the test fold. Requires group_col. GroupKFold "
        "puts the largest remaining group in inner fold 0, so cap very large "
        "groups upstream.",
    )

    @model_validator(mode="after")
    def _validate_fold_index(self) -> "CvSplitConfig":
        if self.fold_index >= self.num_folds:
            raise ValueError(
                f"fold_index ({self.fold_index}) must be < num_folds ({self.num_folds})"
            )
        if self.val_num_folds is not None and self.group_col is None:
            raise ValueError(
                "val_num_folds requires group_col: the inner validation fold is "
                "a GroupKFold over the groups left after the test fold is removed."
            )
        return self

    model_config = ConfigDict(extra="forbid")


class _QScaleConfigurationError(ValueError):
    """Validation error carrying the exact omitted Q-scale field names."""

    def __init__(self, message: str, missing: list[str]) -> None:
        self.missing = tuple(missing)
        super().__init__(message)


def _resolve_q_scales(
    loss_config: Union["QDASMLossConfig", "HybridLossConfig"],
    neutral_model_type: Optional[NeutralModelType],
) -> None:
    """Validate calibration provenance, then resolve omitted Thrifty defaults.

    The shipped defaults were calibrated for Thrifty. A named non-Thrifty rate
    model, or a task with no neutral-model block (rates supplied directly by
    the dataset), must state all three scales because this validator cannot
    infer a corpus/model calibration. ``PrecomputedRates`` is only a multihit
    carrier and supplies no rates itself, so it also requires an explicit
    corpus calibration rather than pretending to have a model-specific one.
    """
    missing: list[str] = []
    if loss_config.single_hit_scale is None:
        missing.append("single_hit_scale")
    if loss_config.double_hit_scale is None:
        missing.append("double_hit_scale")
    if loss_config.triple_hit_scale is None:
        missing.append("triple_hit_scale")

    requires_explicit = neutral_model_type != NeutralModelType.THRIFTY_SHM
    if missing and requires_explicit:
        if neutral_model_type is None:
            source = "a Q-family task with no neutral_model (dataset-precomputed rates)"
            hint = ""
        elif neutral_model_type == NeutralModelType.PRECOMPUTED_RATES:
            source = "neutral_model.type=PrecomputedRates (dataset-precomputed rates)"
            hint = ""
        else:
            source = f"neutral_model.type={neutral_model_type.value}"
            hint = (
                f" The S5F-HH triple fitted in #959 is single_hit_scale: "
                f"{S5F_HH_POOLED_OOF_SINGLE_HIT_SCALE}, double_hit_scale: "
                f"{S5F_HH_POOLED_OOF_DOUBLE_HIT_SCALE}, triple_hit_scale: "
                f"{S5F_HH_POOLED_OOF_TRIPLE_HIT_SCALE} — see "
                f"dasm2/configs/qdasm_s5f_4m.yaml."
                if neutral_model_type == NeutralModelType.S5F
                else ""
            )
        raise _QScaleConfigurationError(
            f"{source} requires {', '.join(missing)} to be set explicitly. "
            "The defaults are Thrifty-calibrated (#589), so an unnamed "
            f"calibration would be ambiguous.{hint} Re-fit them for any other "
            "(neutral model, corpus) pair.",
            missing,
        )

    if loss_config.single_hit_scale is None:
        loss_config.single_hit_scale = DEFAULT_SINGLE_HIT_SCALE
    if loss_config.double_hit_scale is None:
        loss_config.double_hit_scale = DEFAULT_DOUBLE_HIT_SCALE
    if loss_config.triple_hit_scale is None:
        loss_config.triple_hit_scale = DEFAULT_TRIPLE_HIT_SCALE


class TrainingConfig(BaseModel):
    """Top-level training configuration owning all components."""

    # Model architecture
    model: ModelConfig

    # Training components
    neutral_model: Optional[NeutralModelConfig] = None
    codon_pref_table: Optional[Path] = Field(
        default=None,
        description="Path to a CSC (codon stability coefficient) optimality TSV. "
        "When set, the within-AA log-optimality is applied to neutral codon "
        "probabilities (issue #747). The strength β_CSC is fixed at 1 unless "
        "``fit_codon_strengths`` is set. Incompatible with "
        "DASM2_NEUTRAL_MODE=matrix_exp. Reaches the training loss on every "
        "loader/mode (CSV, pickle, Parquet; precomputed or streaming) as of "
        "issue #1039.",
    )
    codon_pref_tai_table: Optional[Path] = Field(
        default=None,
        description="Path to a tAI (tRNA adaptation index) optimality TSV "
        "(second codon-optimality axis, issue #747). Requires "
        "``codon_pref_table`` (CSC) to also be set; the two axes are combined "
        "as Σ_k β_k·ξ̃ᵏ. Same matrix-exp incompatibility as ``codon_pref_table``, "
        "and reaches the training loss on every loader/mode the same way "
        "(issue #1039).",
    )
    fit_codon_strengths: bool = Field(
        default=False,
        description="When set (and a codon-optimality table is configured), fit "
        "the per-axis strengths β_k jointly with the model via the "
        "CodonStrengthOptimizer between-cycle hook (issue #747). When False, the "
        "strengths stay fixed at 1 (the β=1 smoke-test behavior). Works on every "
        "loader/mode: codon_pref_log is shared-memory dataset state on "
        "streaming/Parquet bundles (DESIGN.md D6), so the between-cycle refit "
        "reaches them the same way BranchLengthOptimizer already reaches branch "
        "lengths (issue #1039).",
    )
    fit_multihit_factors: bool = Field(
        default=False,
        description="When set (and a multihit-correction model is configured via "
        "neutral_model.multihit_model_name), refit the 3 log hit-class factors "
        "jointly with the model via the MultihitOptimizer between-cycle hook "
        "(issue #501), replacing the pretrained antibody-fit values with a fit "
        "conditioned on the current dataset and selection factors. When False, "
        "the factors stay fixed at whatever the neutral model was loaded with. "
        "Works on every loader/mode: log_hit_class_factors is a shared-memory "
        "parameter on the bundle's multihit model (DESIGN.md D6), so the "
        "between-cycle refit reaches streaming and Parquet-backed bundles the "
        "same way BranchLengthOptimizer already reaches branch lengths "
        "(issue #1041). On a corpus with no neutral model of its own, configure "
        "the correction through the PrecomputedRates carrier "
        "(PrecomputedRatesNeutralModelConfig).",
    )
    context_free_neutral_model: Optional[NeutralModelConfig] = Field(
        default=None,
        description="Optional context-free neutral model for AA-only datasets. "
        "Must expose neutral_aa_rates. "
        "Only needed when AAH head uses AA-only data (parent_aa/child_aa columns).",
    )
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=ReduceOnPlateauSchedulerConfig)
    head_optimizer: Optional[OptimizerConfig] = Field(
        default=None,
        description="Optional separate optimizer scoped to one head's own "
        "parameters (issue #1220), e.g. slowing BindingClassificationHead's "
        "own convergence in joint training without touching the shared "
        "encoder or any other head. Requires head_optimizer_head_type. Only "
        "that head's parameters move under this optimizer's own "
        "learning_rate/weight_decay; every other parameter (including the "
        "shared encoder) stays on the single top-level optimizer above, "
        "unchanged. None (default) reproduces today's single-optimizer "
        "behavior exactly. When `scheduler` below is "
        "ReduceOnPlateauSchedulerConfig, this optimizer also gets its own "
        "ReduceLROnPlateau (same factor/eval_patience as `scheduler`, "
        "applied independently — the two optimizers' LRs never mix), "
        "stepped on head_optimizer_head_type's own raw validation loss "
        "rather than the combined loss `scheduler` watches. Under a "
        "per-step scheduler (WarmupPolynomialSchedulerConfig, "
        "CosineWarmRestartsSchedulerConfig) there is no per-eval plateau "
        "signal to mirror, so this optimizer's LR stays flat within a "
        "cycle. Either way, its LR is also reset once at every cycle "
        "boundary the same way the main optimizer's is (a geometric-mean "
        "blend under ReduceOnPlateau, a restart at the configured LR under "
        "a per-step scheduler), seeded from this config's own learning_rate "
        "rather than the main optimizer's.",
    )
    head_optimizer_head_type: Optional[HeadType] = Field(
        default=None,
        description="Which head's parameters head_optimizer owns. Required "
        "iff head_optimizer is set; the named head must exist on the model.",
    )

    # Training hyperparameters — step-based budget (issue #578).
    # ``epochs`` was removed: legacy configs that still set it fail to parse
    # because of ``extra="forbid"`` below, forcing the author to rewrite the
    # budget deliberately (see also ReduceOnPlateauSchedulerConfig for the
    # new patience semantic).
    max_train_steps: int = Field(
        gt=0,
        description="Maximum optimizer steps per ``_train_for_steps`` "
        "invocation. In joint multi-task mode this is per cycle; in "
        "sequential single-task mode this is per task per cycle "
        "(total per cycle = num_tasks * max_train_steps).",
    )
    eval_every_steps: int = Field(
        gt=0,
        description="Run validation every N optimizer steps. "
        "Checkpoint writes and best-model updates happen only at "
        "evaluation boundaries.",
    )
    max_eval_steps: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum micro-batches per evaluation. Each task is "
        "evaluated over its own validation loader for at most this many "
        "batches, or its whole loader if that is shorter. None exhausts every "
        "task's validation loader.",
    )
    batch_size: int = Field(gt=0, description="Training batch size")
    val_batch_size: int = Field(
        default=32,
        gt=0,
        description="Per-device batch size for validation. Validation has no "
        "backward pass and no gradient accumulation, so it typically admits a "
        "larger batch than training on the same device (2-8x is common).",
    )
    gradient_clip_val: Optional[float] = Field(
        None, gt=0.0, description="Gradient clipping value"
    )
    early_stopping_patience: Optional[int] = Field(
        None,
        gt=0,
        description="Stop if validation loss doesn't improve for this many "
        "evaluations. What counts as 'improvement' is controlled by "
        "early_stopping_min_delta.",
    )
    early_stopping_min_delta: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum validation loss improvement required to reset "
        "the patience counter. Only effective when "
        "early_stopping_patience is set. An improvement counts when "
        "val_loss < best_val_loss - min_delta (strict less-than). "
        "Default 0.0 preserves current behavior.",
    )
    monitor_task: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Joint training: the task whose own validation loss drives "
        "early stopping, best-model tracking (and so the best checkpoint) and "
        "the reduce-on-plateau scheduler, given as that task's log key (the key "
        "of its val_<key> entry in the eval log, e.g. 'SelectionFactorHead'). "
        "The task's loss_weight is divided out, so changing it does not rescale "
        "early_stopping_min_delta. The task must have a positive loss_weight and "
        "no OverfitGuardHook on it (the hook would zero the weight mid-run), and "
        "the run must evaluate all tasks together (joint training). The eval log "
        "gains a monitored_val_loss entry; best_val_loss and is_best then refer "
        "to it. None (default) monitors the sum of every task's weighted "
        "validation loss, the eval log's val_loss.",
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        gt=0,
        description="Number of micro-batches to accumulate gradients over before "
        "performing an optimizer step. Effective batch size = "
        "batch_size * gradient_accumulation_steps (single-GPU training). "
        "When > 1, training logs report one entry per optimizer step "
        "(not per micro-batch), so batch counts reflect optimizer steps.",
    )

    # Performance features
    float32_matmul_precision: Literal["highest", "high", "medium"] = Field(
        default="high",
        description="Precision for float32 matrix multiplications via "
        "torch.set_float32_matmul_precision(). 'high' uses TF32 on Ampere+ GPUs "
        "(10-bit mantissa, negligible accuracy impact, noticeable speedup). "
        "'medium' uses BF16 accumulation for further speed. 'highest' is full FP32. "
        "This is complementary to use_mixed_precision (BF16 autocast).",
    )
    use_mixed_precision: bool = Field(
        default=False,
        description="Enable BF16 mixed precision for forward pass (requires CUDA with BF16 support)",
    )
    num_dataloader_workers: int = Field(
        default=0,
        ge=0,
        description="Number of worker processes for DataLoader. 0 (default) uses the "
        "main process. > 0 requires the unified pipeline (streaming or precomputed) "
        "which provides worker_init_fn. On Linux, workers fork (copy-on-write). "
        "On macOS (spawn), each worker re-initializes — correct but uses more memory.",
    )

    max_tokens_per_batch: Optional[int] = Field(
        default=None,
        gt=0,
        description="If set, form micro-batches under this cap on total PADDED "
        "tokens (max sequence length in the batch x number of rows) instead of a "
        "fixed batch_size, so a long-sequence cluster holds fewer rows and no "
        "batch's padded forward blows up (issue #724 load-imbalance fix). "
        "batch_size is reused as the per-batch row cap. None (default) keeps "
        "fixed-size batching. Only applies to datasets exposing per-sample "
        "lengths (evolutionary heads); other heads fall back to fixed batching.",
    )
    dataloader_prefetch_factor: int = Field(
        default=4,
        ge=1,
        description="Per-worker DataLoader prefetch depth (only used when "
        "num_dataloader_workers > 0). Lower it (e.g. 2) on memory-constrained "
        "nodes: each prefetched batch is held in host RAM, and with long "
        "sequences padded to maxlen a deep queue is a large, bursty footprint "
        "that can OOM a worker (issue #724).",
    )
    parquet_cache_budget_gb: Optional[float] = Field(
        default=None,
        gt=0,
        description="RAM budget in GB of decoded row groups (arrow nbytes, "
        "measured at insert) for each copy of a Parquet backend's row-group "
        "cache; the main process and every DataLoader worker hold their own. "
        "None (default) keeps the pre-#1272 bound: as many row groups as fit "
        "2 GiB of parquet total_byte_size (~24 on the v10 corpus, ~14 GB "
        "decoded), capped at the working set. When set, the budget is not "
        "clamped, but startup warns if the main process and train workers at "
        "this budget, plus the val workers' sweep caches, would exceed 25% of "
        "MemTotal (issue #1272; "
        "issue #724 for the OOM a cache bound prevents). The "
        "DASM2_PARQUET_CACHE_BUDGET_GB env var overrides this field.",
    )
    dataloader_pin_memory: bool = Field(
        default=True,
        description="Whether to pin DataLoader host memory. Pinned memory speeds "
        "CPU->GPU transfer but is page-locked and non-swappable, so under host-RAM "
        "pressure it cannot be paged out and can push a constrained node into the "
        "OOM-killer (issue #724). Set False to disable on such nodes. Only takes "
        "effect on CUDA devices.",
    )

    bl_optimization_num_workers: Optional[int] = Field(
        default=None,
        gt=0,
        description="Worker processes for post-cycle branch-length optimization. "
        "None (default) auto-picks min(cpu_count // 2, 10). Lower it (e.g. 6) to "
        "bound peak host RAM: each worker holds its own optimization chunk, and on "
        "a large per-cycle corpus the aggregate footprint can OOM-kill a worker at "
        "the cycle boundary (issue #724).",
    )
    bl_optimization_max_chunk_size: int = Field(
        default=1000,
        gt=0,
        description="Upper bound on sequences per branch-length optimization chunk. "
        "Each in-flight chunk is held in host RAM (at most num_workers + 1 alive at "
        "once), so this caps the BL-optimization working set. 1000 (default) matches "
        "historical behavior; lower it (e.g. 250) on a large per-cycle corpus that "
        "OOM-kills a worker at the cycle boundary (issue #724). The effective chunk "
        "size may be smaller to keep at least num_workers chunks for parallelism.",
    )
    freeze_branch_lengths: bool = Field(
        default=False,
        description="If True, skip all branch-length re-estimation for "
        "evolutionary (DASM-family) losses: the BranchLengthOptimizer's "
        "pre_cycle cold-start init/cache/model-optimization fallback and its "
        "post_cycle unconditional re-optimization both become no-ops, so the "
        "branch lengths already present in the input data (e.g. known "
        "true/generating values from a simulation) are used unchanged "
        "throughout training. The input's branch_length column is what gets "
        "frozen on both the Parquet and the DataFrame/CSV path; a corpus "
        "without that column freezes at DEFAULT_BRANCH_LENGTH=0.1 (issue "
        "#1168). Without this flag, even a cycles=1 run overwrites the input "
        "branch lengths at both cycle boundaries (issue #821: needed to "
        "remove the BL-estimation confound when comparing training on "
        "reconstructed vs. true ancestors at matched branch lengths).",
    )
    freeze_branch_lengths_after_first_cycle: bool = Field(
        default=False,
        description="If True, branch lengths are set once by the cycle-0 "
        "pre_cycle pass and then held fixed: the BranchLengthOptimizer's "
        "post_cycle re-optimization becomes a no-op at every cycle boundary, "
        "while pre_cycle still runs. What cycle 0 does depends on the active "
        "segment handler: with a calibrated bl_init_strategy (e.g. "
        "AntibodySegmentHandler's ProportionalStrategy) nothing is optimized "
        "at all — the mutation-frequency init runs and that estimate is what "
        "gets frozen; without one, cycle 0 falls through to the cache and "
        "then to model-based optimization, and the optimized values are what "
        "get frozen. The weaker sibling of freeze_branch_lengths, which "
        "additionally freezes cycle 0. Used by the issue #199 / #713 "
        "frozen-BL retrain to stop the model from co-adapting branch lengths "
        "to its own suppressed selection across cycles while still training "
        "against a sensible BL estimate. Mutually exclusive with "
        "freeze_branch_lengths.",
    )
    init_branch_lengths_from_input: bool = Field(
        default=False,
        description="If True, cycle 0 trains at the input's branch_length "
        "column instead of the BranchLengthOptimizer's cycle-0 cold start, "
        "and post_cycle re-optimizes after every cycle as usual (see "
        "BranchLengthOptimizer's init_from_input). A corpus without the "
        "column raises at cycle 0. Mutually exclusive with "
        "freeze_branch_lengths and freeze_branch_lengths_after_first_cycle.",
    )
    save_training_state: bool = Field(
        default=False,
        description="If True, checkpoints written by CheckpointSaveHook "
        "include full resumable training state (optimizer, scheduler, RNG, "
        "step counters) in addition to model weights, so `dasm2 train "
        "--resume <checkpoint>` can continue the optimizer/LR schedule "
        "exactly where it left off instead of restarting it from the "
        "config's base learning rate (issue #516). Default False preserves "
        "the historical weights-only checkpoint behavior.",
    )
    freeze_encoder: bool = Field(
        default=False,
        description="If True, the encoder's parameters have requires_grad "
        "set to False before the optimizer is built, so only head "
        "parameters are trained (head-only fine-tuning). Recorded here "
        "(rather than only as a CLI flag) so `dasm2 train --resume "
        "<checkpoint>` can rebuild the model with the same frozen encoder "
        "the interrupted run had: the freeze must happen before "
        "create_optimizer builds its param groups from "
        "requires_grad-filtered parameters, and before any DDP wrap, since "
        "the reducer registers every parameter that requires grad at "
        "construction time (issue #1076).",
    )

    # Dataset pipeline mode
    dataset_mode: Literal["precomputed", "streaming"] = Field(
        default="precomputed",
        description="Dataset pipeline mode. 'precomputed' tokenizes and transforms "
        "all samples at init (fast training, more memory). 'streaming' runs "
        "transforms on-the-fly per __getitem__ (lower memory, slower — hidden "
        "by model compute on large models). DMS and HERMES heads always use "
        "precomputed regardless of this setting.",
    )

    # 3Di structural tokens
    tdi_tokens_name: Optional[str] = Field(
        default=None,
        description="Catalog name for 3Di tokens directory (resolved via model_catalog.yaml). "
        "Mutually exclusive with tdi_tokens_dir.",
    )
    tdi_tokens_dir: Optional[Path] = Field(
        default=None,
        description="Directory with {structure_id}_3di.txt files for 3Di structural tokens. "
        "When set, datasets are wrapped with TdiWhichmutDataset.",
    )
    tdi_pad_shorter_tokens: bool = Field(
        default=False,
        description="Right-pad 3Di tokens shorter than max_aa_seq_len with the padding index. "
        "Required for multi-structure datasets with varying protein lengths "
        "(e.g., flu HA subtypes: H3=566 AA, H5=568 AA).",
    )

    # UShER censoring (#1079). Two paths rather than one plus a derived sibling:
    # the roster is what makes a clade censoring nothing distinguishable from a
    # mistyped clade name, so a silently-absent sidecar would reintroduce exactly
    # the failure it exists to prevent.
    censored_mutations_file: Optional[Path] = Field(
        default=None,
        description="mat-pcp's pcps/{gene}/censored_mutations.csv for this gene. "
        "When set, UShER-masked mutations are removed from both the whichmut "
        "partition function and the loss_mask. Requires a 'clade' column in the "
        "corpus and censored_clades_file alongside.",
    )
    censored_clades_file: Optional[Path] = Field(
        default=None,
        description="The censored_mutations.clades.txt roster emitted beside "
        "censored_mutations.csv. Required with it: a clade with zero censored "
        "mutations contributes no rows, so without the roster it is "
        "indistinguishable from a typo'd clade name.",
    )
    # The other mask file (#1083). Named for `{gene}_codon_sites_to_mask.txt` and
    # for the manifest's `sites_to_mask` / `resolve_sites_to_mask()`, so the config
    # key, the manifest field and the filename stay one vocabulary.
    codon_sites_to_mask_file: Optional[Path] = Field(
        default=None,
        description="mat-pcp's {gene}_codon_sites_to_mask.txt. When set, every "
        "candidate at these codon sites is censored in every clade, on top of the "
        "per-clade censored_mutations_file. These sites carry observations mat-pcp "
        "flags as artifacts, and the upstream drop filter removed most of them "
        "while leaving all of their candidates in the partition function. "
        "Requires the censored_mutations_file/censored_clades_file pair.",
    )
    # The same global mask at nucleotide granularity (mat-pcp#153), and the one to
    # prefer. Named for `pcps/{gene}/masked_nt_sites.csv` and for the manifest's
    # `masked_nt_sites`, keeping the config key, the manifest field and the
    # filename one vocabulary, as the codon field above does.
    masked_nt_sites_file: Optional[Path] = Field(
        default=None,
        description="mat-pcp's pcps/{gene}/masked_nt_sites.csv. The nucleotide-"
        "granular form of codon_sites_to_mask_file, expressing the same three "
        "clade-independent reasons without widening each flagged nucleotide to "
        "its whole codon (2.85x fewer masked nucleotides across the six v12 "
        "genes; 18 against 51 on S). Mutually exclusive with "
        "codon_sites_to_mask_file. Requires the censored_mutations_file/"
        "censored_clades_file pair.",
    )
    # The third source of censored candidates (#1099). Unlike the two above this
    # one stands alone -- see `_validate_censoring` for why that is the whole
    # point rather than an oversight.
    reference_sequence_file: Optional[Path] = Field(
        default=None,
        description="Gene-local reference FASTA, from the aggregate manifest's "
        "founder_sequence (v11: clade_founders/{gene}/19A.fa, Wuhan-Hu-1). When "
        "set, at every nucleotide where the parent differs from the reference "
        "the single candidate restoring the reference base is censored. "
        "Clade-independent, so it needs neither the per-clade table nor a "
        "'clade' column in the corpus.",
    )
    censor_mode: CensorMode = Field(
        default="both",
        description="Which half of censoring to apply. 'both' is the only "
        "coherent objective and the only value to use in production. The two "
        "single-half modes exist for the ablation demonstrating the halves do "
        "different things (#1079); each on its own still trains and produces a "
        "plausible curve, which is precisely why they must be named explicitly. "
        "Shared by every censoring source, including reference_sequence_file, "
        "where the halves matter more because that rule is "
        "numerator-dominated (#1099).",
    )

    # Extensible hook configuration
    pre_hooks: List[HookConfig] = Field(
        default_factory=list, description="Pre-cycle training hooks with parameters"
    )
    post_hooks: List[HookConfig] = Field(
        default_factory=list, description="Post-cycle training hooks with parameters"
    )
    combine_gradient_hooks: List[HookConfig] = Field(
        default_factory=list,
        max_length=1,
        description="Top-level (not per-task) hooks controlling how multiple "
        "tasks' gradients on shared parameters are combined before the "
        "optimizer step — e.g. PCGradHook (issue #1144). Unlike pre_hooks/"
        "post_hooks above (populated only by Trainer.export_metadata for "
        "checkpoint round-tripping), this field is read at load time: "
        "train_command.py resolves it via create_hook() once and passes the "
        "result into create_trainer(), not per-task. Empty by default, which "
        "reproduces today's plain-sum gradient combination exactly — at "
        "most one entry is supported (Trainer raises otherwise), since two "
        "combination strategies cannot both apply to the same gradient.",
    )

    # On-the-fly cross-validation fold selection.
    cv_split: Optional[CvSplitConfig] = Field(
        default=None,
        description="When set, the loader builds the train/val split from "
        "fold position (row-level RNG draw, or GroupKFold over a column) "
        "instead of reading the in_train column or the registry holdouts. "
        "Lets one data spec feed K independent fold-index runs without "
        "regenerating data.",
    )

    # v4 declarative task block (issue #621 Task #6). One entry per
    # (head, loss) pair. AAH expands to two entries sharing the same
    # ``head_type`` pointer. Pre-v4 YAMLs must be migrated with
    # ``scripts/migrate_yaml_to_v4.py``.
    tasks: List[TaskConfig] = Field(
        default_factory=list,
        description=(
            "v4 declarative tasks block (issue #621). Each entry pairs a head "
            "(by ``head_type`` discriminator) with a ``LossConfig`` and a "
            "``loss_weight``. AAH expands to two entries (whichmut + "
            "aao_whichmut) sharing the same ``head_type`` pointer."
        ),
    )

    @property
    def has_tdi_encoder(self) -> bool:
        """Whether the encoder is configured with 3Di structural tokens."""
        return self.model.encoder.supports_tdi

    @model_validator(mode="after")
    def validate_tdi_consistency(self) -> "TrainingConfig":
        """Validate TDI token configuration consistency.

        Checks:
        1. At most one of tdi_tokens_name or tdi_tokens_dir may be set.
        2. tdi_tokens_dir is not set on a non-TDI encoder.

        The reverse check (num_tdi_tokens without tdi_tokens_dir) is
        intentionally omitted so configs can be loaded as templates before
        tdi_tokens_dir is resolved at training time.

        Raises:
            ValueError: If both tdi_tokens_name and tdi_tokens_dir are set,
                or if tdi_tokens_dir is set with a non-TDI encoder.
        """
        has_tdi_name = self.tdi_tokens_name is not None
        has_tdi_dir = self.tdi_tokens_dir is not None
        if has_tdi_name and has_tdi_dir:
            raise ValueError(
                "At most one of 'tdi_tokens_name' or 'tdi_tokens_dir' may be set, "
                "got both"
            )
        if has_tdi_dir and not self.has_tdi_encoder:
            raise ValueError(
                "tdi_tokens_dir is set but encoder has no num_tdi_tokens configured. "
                "Set num_tdi_tokens on the encoder or remove tdi_tokens_dir."
            )
        return self

    @model_validator(mode="after")
    def _validate_censoring(self) -> "TrainingConfig":
        """Check the censoring inputs are complete and reach a whichmut loss.

        Both files or neither: with only the table, the loader cannot tell an
        empty clade from a typo; with only the roster, there is nothing to
        censor. And censoring is defined only where ``codon_neutral_rates``
        exists, so pointing it at a non-whichmut task would configure a file that
        is silently never read -- the failure mode this whole change is about.

        The global mask -- ``masked_nt_sites_file`` (mat-pcp#153) or the older
        ``codon_sites_to_mask_file`` (#1083), never both -- rides on top of that
        pair rather than standing alone; see the comments at their checks for why
        global-only censoring is deliberately unrepresentable and why the two
        granularities exclude each other.

        ``reference_sequence_file`` (#1099) is the opposite case and **must** be
        permitted to stand alone. Its rule is clade-independent by construction --
        "at every nucleotide where the parent differs from the reference" needs no
        clade -- so requiring the per-clade pair the way the global mask does
        would make a reversion-only arm silently depend on an UShER table it never
        reads. It still has to reach a whichmut loss, for the same reason the
        others do.

        Raises:
            ValueError: If exactly one of the two per-clade files is set, if a
                global mask file is set without them, if both global mask files
                are set at once, or if any censoring source is set on a config
                with no whichmut-family task.
        """
        has_table = self.censored_mutations_file is not None
        has_roster = self.censored_clades_file is not None
        has_reference = self.reference_sequence_file is not None
        if has_table != has_roster:
            present, absent = (
                ("censored_mutations_file", "censored_clades_file")
                if has_table
                else ("censored_clades_file", "censored_mutations_file")
            )
            raise ValueError(
                f"{present} is set but {absent} is not. Both are required: the "
                f"roster is what distinguishes a clade that censors nothing from "
                f"a mistyped clade name."
            )
        # The global mask needs no clade information of its own, so in principle
        # the roster alone would do. Requiring the whole pair keeps one loading
        # path -- `with_global_codon_sites` merges into a loaded table -- at the
        # cost of making global-only censoring unrepresentable. Deliberate: no arm
        # of #1083's pilot wants it, and a second construction path for a case
        # nobody needs is the speculative generality CONSTITUTION IX warns about.
        if self.codon_sites_to_mask_file is not None and not has_table:
            raise ValueError(
                f"codon_sites_to_mask_file is set "
                f"({self.codon_sites_to_mask_file}) but "
                f"censored_mutations_file and censored_clades_file are not. The "
                f"global codon mask is applied on top of the per-clade table, not "
                f"instead of it: censoring is keyed by clade, so the roster is "
                f"what defines which clades exist. Global-only censoring is "
                f"deliberately not supported."
            )
        if self.masked_nt_sites_file is not None and not has_table:
            raise ValueError(
                f"masked_nt_sites_file is set ({self.masked_nt_sites_file}) but "
                f"censored_mutations_file and censored_clades_file are not. The "
                f"global nucleotide mask is applied on top of the per-clade "
                f"table, not instead of it: censoring is keyed by clade, so the "
                f"roster is what defines which clades exist. Global-only "
                f"censoring is deliberately not supported."
            )
        # Mutually exclusive rather than additive, because the codon set contains
        # the nucleotide set by construction -- both files express mat-pcp's same
        # three clade-independent reasons, one widened to whole codons. Applying
        # both would let the wider mask win silently: the run would read as
        # nucleotide-granular in its config and be codon-granular in its loss,
        # with no shape change, no warning and a plausible loss curve.
        if (
            self.codon_sites_to_mask_file is not None
            and self.masked_nt_sites_file is not None
        ):
            raise ValueError(
                f"codon_sites_to_mask_file ({self.codon_sites_to_mask_file}) and "
                f"masked_nt_sites_file ({self.masked_nt_sites_file}) are both "
                f"set. They are two granularities of one mask, not two masks: "
                f"the codon file widens each flagged nucleotide to its whole "
                f"codon, so it strictly contains the nucleotide file and would "
                f"silently win. Set exactly one -- masked_nt_sites_file unless "
                f"you are reproducing a pre-mat-pcp#153 run."
            )
        if (has_table or has_reference) and self.tasks:
            sources = [
                name
                for name, present in (
                    ("censored_mutations_file", has_table),
                    ("reference_sequence_file", has_reference),
                )
                if present
            ]
            # WHICHMUT_RESIDUAL belongs here even though it is absent from
            # `WHICHMUT_LOSS_TYPES`: `losses.factory._build_whichmut_residual`
            # returns a plain `WhichmutLoss`, so it consumes the censored tensors
            # identically. Omitting it would reject a config this change does
            # support (the MPNN branch of `train_model` forwards the table).
            censorable = {
                LossType.WHICHMUT,
                LossType.AAO_WHICHMUT,
                LossType.WHICHMUT_RESIDUAL,
            }
            if not any(task.loss_config.loss_type in censorable for task in self.tasks):
                configured = sorted({str(t.loss_config.loss_type) for t in self.tasks})
                raise ValueError(
                    f"{' and '.join(sources)} "
                    f"{'is' if len(sources) == 1 else 'are'} set but no task "
                    f"uses a whichmut loss ({configured}). Censoring acts on "
                    f"codon_neutral_rates "
                    f"and loss_mask, which only whichmut-family losses consume, "
                    f"so the file would be silently ignored."
                )
            # AAO is whichmut-family and censorable in principle -- its chain
            # builds codon_neutral_rates before MarginalizeCodonsToAAO -- but
            # #1079 scoped it out to keep this change one variable, and
            # `_load_whichmut_head_data` raises NotImplementedError on it. Say so
            # here rather than at data-loading time: both are loud, but a config
            # that cannot run should not survive config validation.
            if any(
                task.loss_config.loss_type == LossType.AAO_WHICHMUT
                for task in self.tasks
            ):
                raise ValueError(
                    f"Censoring is not yet implemented for the AAO whichmut path "
                    f"(#1079 scoped it out deliberately, and #1099 kept it out "
                    f"for the same one-variable reason). Remove "
                    f"{' and '.join(sources)} or use the standard whichmut head."
                )
        return self

    @model_validator(mode="after")
    def _validate_head_optimizer_pairing(self) -> "TrainingConfig":
        """head_optimizer and head_optimizer_head_type must be set together."""
        if (self.head_optimizer is None) != (self.head_optimizer_head_type is None):
            raise ValueError(
                "head_optimizer and head_optimizer_head_type must both be "
                "set or both be unset "
                f"(head_optimizer={self.head_optimizer!r}, "
                f"head_optimizer_head_type={self.head_optimizer_head_type!r})."
            )
        return self

    @model_validator(mode="after")
    def _validate_step_budget(self) -> "TrainingConfig":
        """Check step-budget knobs are mutually consistent.

        - ``eval_every_steps`` must be >= ``gradient_accumulation_steps``,
          otherwise an evaluation could fire mid-accumulation-window.
        - ``max_train_steps`` must be >= ``eval_every_steps``, otherwise no
          evaluation would ever fire (the terminal-eval guarantee in
          ``_train_for_steps`` covers this, but rejecting it up front
          avoids surprising "no metrics emitted" runs).
        """
        if self.eval_every_steps < self.gradient_accumulation_steps:
            raise ValueError(
                f"eval_every_steps ({self.eval_every_steps}) must be >= "
                f"gradient_accumulation_steps ({self.gradient_accumulation_steps}) "
                "so evaluations only fire at optimizer-step boundaries."
            )
        if self.max_train_steps < self.eval_every_steps:
            raise ValueError(
                f"max_train_steps ({self.max_train_steps}) must be >= "
                f"eval_every_steps ({self.eval_every_steps}); a cycle "
                "must contain at least one evaluation."
            )
        return self

    @model_validator(mode="after")
    def _validate_q_table_matches_neutral_model(self) -> "TrainingConfig":
        """Bind the Q-matrix k-mer table to the configured neutral model.

        A global k-mer Q variant builds every interior Q row from its lookup
        table, so the table — not ``neutral_model`` — is what actually drives
        the CTMC. Pairing a table with a different neutral model therefore
        trains a model whose mutation process is not the one the config names,
        and nothing downstream notices: under ``global_13mer`` a
        Thrifty-table/S5F-model pairing leaves the neutral model influencing
        only the ~4% of sites that hit the boundary splice (issue #959).

        For ``global_13mer`` this binds the configured model name into the loss;
        loss construction then compares it with the source model recorded in
        the table builder's JSON sidecar. For ``global_5mer`` the table is
        *derived* from the neutral model, so this fills ``five_mer_model_name``
        directly. Explicit conflicts are rejected rather than silently
        preferring one. Constitution IV: fail before training starts.
        """
        for task in self.tasks or []:
            loss_config = task.loss_config
            if not isinstance(loss_config, (QDASMLossConfig, HybridLossConfig)):
                continue
            if loss_config.q_variant not in GLOBAL_KMER_Q_VARIANTS:
                continue
            if self.neutral_model is None:
                raise ValueError(
                    f"q_variant='{loss_config.q_variant}' requires a "
                    "neutral_model so the Q-table source can be bound and "
                    "validated."
                )

            if loss_config.q_variant == "global_13mer":
                configured = getattr(self.neutral_model, "model_name", None)
                if not isinstance(configured, str) or not configured:
                    raise ValueError(
                        "q_variant='global_13mer' requires "
                        "neutral_model.model_name so it can be compared with "
                        "the source model recorded in the table metadata."
                    )
                declared = loss_config.thrifty_13mer_model_name
                if declared is None:
                    loss_config.thrifty_13mer_model_name = configured
                elif declared != configured:
                    raise ValueError(
                        f"thrifty_13mer_model_name='{declared}' does not match "
                        f"neutral_model.model_name='{configured}'. Leave "
                        "thrifty_13mer_model_name unset to inherit it; loss "
                        "construction will then verify the table sidecar."
                    )
                continue

            # Hybrid does not offer global_5mer, so this branch is QDASM-only.
            if not isinstance(loss_config, QDASMLossConfig):
                raise TypeError(
                    "q_variant='global_5mer' requires QDASMLossConfig; "
                    f"got {type(loss_config).__name__}"
                )
            if self.neutral_model.type != NeutralModelType.S5F:
                raise ValueError(
                    f"q_variant='global_5mer' requires neutral_model.type=S5F "
                    f"(the five-mer table is read off the S5F lookup), got "
                    f"'{self.neutral_model.type.value}'. Use "
                    "q_variant='global_13mer' for Thrifty."
                )
            configured = self.neutral_model.model_name
            if configured is None:
                raise ValueError(
                    "q_variant='global_5mer' requires neutral_model.model_name "
                    "because QDASMLoss reloads the same S5F catalog model to "
                    "derive its table."
                )
            declared = loss_config.five_mer_model_name
            if declared is None:
                loss_config.five_mer_model_name = configured
            elif declared != configured:
                raise ValueError(
                    f"five_mer_model_name='{declared}' does not match "
                    f"neutral_model.model_name='{configured}'. The five-mer "
                    "table drives every interior row of Q, so a mismatch would "
                    "train a Q from one neutral model while the config claims "
                    "another. Leave five_mer_model_name unset to inherit it."
                )
        return self

    @model_validator(mode="after")
    def _validate_q_scales_match_neutral_model(self) -> "TrainingConfig":
        """Refuse a Q-family task that inherits Thrifty's (γ, δ, ψ) under a
        non-Thrifty neutral model.

        Separate from ``_validate_q_table_matches_neutral_model`` because it is
        a different binding: that one ties the *table* to the neutral model,
        this one ties the *calibration* to it. Both are silent mismatches that
        train a mutation process the config does not name; see
        :func:`_resolve_q_scales`. Runs second so a table/model
        identity error — the more fundamental misconfiguration — surfaces
        first. Constitution IV: fail before training starts.
        """
        neutral_model_type = (
            self.neutral_model.type if self.neutral_model is not None else None
        )
        for task in self.tasks or []:
            loss_config = task.loss_config
            if isinstance(loss_config, (QDASMLossConfig, HybridLossConfig)):
                _resolve_q_scales(loss_config, neutral_model_type)
        return self

    model_config = ConfigDict(extra="forbid")


def _get_configs_dir() -> Path:
    """Get path to configs directory from package data.

    Uses importlib.resources to access configs bundled with the package.
    This works correctly for both development (editable install) and
    production (pip install) environments.

    Note: Deviates from loris pattern (repository root configs) to enable
    proper package distribution. Configs are versioned with the package
    as package data in dasm2/configs/.
    """
    return Path(str(files("dasm2").joinpath("configs")))


def _resolve_relative_model_paths(config: "TrainingConfig", config_dir: Path) -> None:
    """Resolve relative model_path fields in neutral model configs.

    When a relative ``model_path`` exists relative to the YAML file's
    directory, it is resolved to an absolute path.  This makes experiment
    configs work regardless of CWD.  If the path doesn't exist relative to
    the config directory, it is left as-is (preserving the existing
    CWD-relative behaviour for built-in package configs).

    Skips resolution when ``model_name`` is set (catalog-based resolution
    handles paths internally).

    Args:
        config: Parsed training config (modified in place).
        config_dir: Directory containing the YAML file.
    """
    catalog_types = (
        TabularRatesNeutralModelConfig,
        GTRCodonNeutralModelConfig,
        S5FNeutralModelConfig,
    )
    for neutral_cfg in (config.neutral_model, config.context_free_neutral_model):
        if neutral_cfg is None:
            continue
        # Skip catalog-based configs where model_name handles resolution
        if (
            isinstance(neutral_cfg, catalog_types)
            and neutral_cfg.model_name is not None
        ):
            continue
        model_path = getattr(neutral_cfg, "model_path", None)
        if model_path is not None and not model_path.is_absolute():
            candidate = (config_dir / model_path).resolve()
            if candidate.exists():
                object.__setattr__(neutral_cfg, "model_path", candidate)

    # Resolve relative codon-optimality table paths the same way (issue #747),
    # so experiment configs that point at the checked-in TSVs work from any CWD.
    for field in ("codon_pref_table", "codon_pref_tai_table"):
        table_path = getattr(config, field, None)
        if table_path is not None and not table_path.is_absolute():
            candidate = (config_dir / table_path).resolve()
            if candidate.exists():
                # Plain setattr: TrainingConfig is mutable. (So are the neutral
                # sub-configs; the object.__setattr__ above is not needed.)
                setattr(config, field, candidate)


def load_config_from_yaml(yaml_path: Path) -> TrainingConfig:
    """Load and validate configuration from YAML file.

    Relative ``model_path`` values in neutral model configs are resolved
    against the YAML file's parent directory so configs work regardless of CWD.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Validated TrainingConfig instance (top-level config)

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid or config validation fails
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        raise ValueError(f"Empty config file: {yaml_path}")

    try:
        config = TrainingConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Invalid config in {yaml_path}: {e}") from e

    _resolve_relative_model_paths(config, yaml_path.resolve().parent)
    return config


def load_named_config(config_name: str) -> TrainingConfig:
    """Load a named configuration from the configs/ directory.

    Args:
        config_name: Name of config (e.g., '77k', '8m', 'multitask_77k')

    Returns:
        Validated TrainingConfig instance (top-level config)

    Raises:
        FileNotFoundError: If named config doesn't exist
        ValueError: If config validation fails
    """
    configs_dir = _get_configs_dir()

    # Support both with and without .yaml extension
    config_path = configs_dir / f"{config_name}.yaml"
    if not config_path.exists():
        config_path = configs_dir / f"{config_name}.yml"

    if not config_path.exists():
        available = get_available_config_names()
        raise FileNotFoundError(
            f"Config '{config_name}' not found. Available: {', '.join(available)}"
        )

    return load_config_from_yaml(config_path)


def get_available_config_names() -> List[str]:
    """Get available named configuration names.

    Returns:
        List of config names (without .yaml extension), sorted alphabetically
    """
    configs_dir = _get_configs_dir()

    if not configs_dir.exists():
        return []

    config_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
    return sorted([f.stem for f in config_files])


def get_config(config_name: str) -> TrainingConfig:
    """Get config by name or path (convenience wrapper).

    Args:
        config_name: Named config (e.g., '8m') or path to YAML file

    Returns:
        Validated TrainingConfig instance (top-level config)

    Raises:
        FileNotFoundError: If config not found
        ValueError: If config validation fails
    """
    path = Path(config_name)
    if path.exists() and path.suffix in [".yaml", ".yml"]:
        return load_config_from_yaml(path)
    return load_named_config(config_name)


def list_configs() -> None:
    """List all available named model configurations with architecture details.

    Displays config names with encoder architecture and prediction heads.
    Follows loris show-model-configs pattern.
    """
    configs_dir = _get_configs_dir()

    if not configs_dir.exists():
        print("No configs directory found")
        print(f"Expected location: {configs_dir}")
        return

    config_files = sorted(configs_dir.glob("*.yaml"))
    if not config_files:
        config_files = sorted(configs_dir.glob("*.yml"))

    if not config_files:
        print(f"No configuration files found in {configs_dir}")
        return

    print("DASM2 Model Configurations")
    print("=" * 80)
    print(
        f"{'Name':<20} {'d_model':<8} {'Layers':<8} {'Heads':<8} {'FFN':<10} {'Prediction Heads'}"
    )
    print("-" * 80)

    for config_file in config_files:
        config_name = config_file.stem

        try:
            # Load and validate config
            training_config = load_config_from_yaml(config_file)

            # Extract encoder info from model
            enc = training_config.model.encoder
            d_model = enc.d_model

            # Handle different encoder types
            if isinstance(enc, ESM2EncoderConfig):
                layers = enc.layers_label
                nhead = "-" if enc.decoder is None else str(enc.decoder.nhead)
                ffn = "-" if enc.decoder is None else str(enc.decoder.dim_feedforward)
            elif isinstance(enc, MPNNLightAttentionEncoderConfig):
                layers = "MPNN"
                nhead = "-"
                ffn = str(enc.hidden_dims)
            elif isinstance(enc, IdentityEncoderConfig):
                layers = "none"
                nhead = "-"
                ffn = "-"
            else:
                layers = str(enc.num_layers)
                nhead = str(enc.nhead)
                ffn = str(enc.dim_feedforward)

            # Get prediction heads
            head_names = [h.head_type.value for h in training_config.model.heads]
            head_summary = ", ".join(head_names)

            print(
                f"{config_name:<20} {d_model:<8} {layers:<8} {nhead:<8} {ffn:<10} {head_summary}"
            )

        except Exception as e:
            print(f"{config_name:<20} ERROR: {str(e)[:50]}")

    print("-" * 80)
    print(f"Found {len(config_files)} configuration(s)")
    print("\nUse config names with training command: dasm2 train --config-name <name>")


# =============================================================================
# Neutral Model Factory (Registry Pattern)
# =============================================================================


def _maybe_wrap_multihit(
    base_model: "RatesCSPNeutralModel",
    config: "MultihitWrappableNeutralModelConfig",
) -> "AbstractNeutralModel":
    """Wrap ``base_model`` in a ``MultihitWrapper`` iff the config names one.

    Shared by every rates/CSPs-decomposition neutral model's factory
    (ThriftySHM, TabularRates, S5F, PrecomputedRates) —
    ``GTRCodonNeutralModel`` is deliberately
    excluded since it handles multi-hit exactly via matrix exponentiation
    and cannot be composed with ``MultihitWrapper`` (see the wrapper's
    docstring).

    ``multihit_log_factors`` (issue #501 trap 5), when set, takes precedence
    over a registry lookup by ``multihit_model_name``: it carries the
    multihit model's actual factors as of save time, which may have been
    refit away from whatever ``multihit_model_name`` originally named.
    """
    from dasm2.neutral.multihit import MultihitWrapper

    if config.multihit_log_factors is not None:
        from dasm2.neutral.hit_class import HitClassModel

        multihit_model = HitClassModel.from_weights(config.multihit_log_factors)
        return MultihitWrapper(base_model, multihit_model, config.multihit_model_name)

    if not config.multihit_model_name:
        return base_model

    return MultihitWrapper.from_pretrained(base_model, config.multihit_model_name)


def _create_thrifty_shm_model(
    config: ThriftySHMNeutralModelConfig,
) -> "AbstractNeutralModel":
    """Factory function for ThriftySHM models."""
    from dasm2.neutral.thrifty import ThriftySHMModel

    model_path = resolve_model(config.model_name)
    base_model = ThriftySHMModel.from_catalog(
        config.model_name, model_path, boundary_table_path=config.boundary_table_path
    )
    return _maybe_wrap_multihit(base_model, config)


def _create_tabular_rates_model(
    config: TabularRatesNeutralModelConfig,
) -> "AbstractNeutralModel":
    """Factory function for tabular rates models."""
    from dasm2.neutral.tabular_rates import TabularRatesNeutralModel

    if config.model_name is not None:
        model_path = resolve_model(config.model_name)
        base_model = TabularRatesNeutralModel.from_catalog(
            config.model_name, model_path
        )
    elif config.model_path is not None:
        base_model = TabularRatesNeutralModel.from_pretrained(config.model_path)
    else:
        raise ValueError(
            "TabularRatesNeutralModelConfig has neither model_name nor model_path; "
            "this should have been caught by the validator."
        )

    return _maybe_wrap_multihit(base_model, config)


def _create_s5f_model(
    config: S5FNeutralModelConfig,
) -> "AbstractNeutralModel":
    """Factory function for S5F five-mer models."""
    from dasm2.neutral.s5f import S5FNeutralModel

    if config.model_name is not None:
        model_path = resolve_model(config.model_name)
        base_model = S5FNeutralModel.from_catalog(config.model_name, model_path)
    elif config.model_path is not None:
        base_model = S5FNeutralModel.from_pretrained(config.model_path)
    else:
        raise ValueError(
            "S5FNeutralModelConfig has neither model_name nor model_path; "
            "this should have been caught by the validator."
        )

    return _maybe_wrap_multihit(base_model, config)


def _create_precomputed_rates_model(
    config: PrecomputedRatesNeutralModelConfig,
) -> "AbstractNeutralModel":
    """Factory for the multihit-carrier neutral model.

    The wrap is the entire point of this model, so an unwrapped carrier (no
    ``multihit_model_name``, no ``multihit_log_factors``) would be an
    elaborate way to write ``neutral_model: null`` — one that also turns the
    clean ``RuntimeError`` from
    ``get_branch_length_optimization_inputs``'s no-neutral-model guard into a
    ``NotImplementedError`` from deeper in. Rejected here rather than
    tolerated.

    Checked in the factory rather than as a ``model_validator`` (the usual
    place for this project's fail-fast config checks) because
    ``PrecomputedRatesNeutralModel.to_config()`` legitimately emits a bare
    config, which ``MultihitWrapper.to_config()`` then fills in — a validator
    would reject that intermediate and break the checkpoint round-trip.
    """
    from dasm2.neutral.precomputed_rates import PrecomputedRatesNeutralModel

    if config.multihit_model_name is None and config.multihit_log_factors is None:
        raise ValueError(
            "PrecomputedRates neutral model requires a multihit correction: "
            "set multihit_model_name (or multihit_log_factors). It carries "
            "nothing else — without one, omit neutral_model entirely."
        )

    return _maybe_wrap_multihit(PrecomputedRatesNeutralModel(), config)


def _create_gtr_codon_model(
    config: GTRCodonNeutralModelConfig,
) -> "AbstractNeutralModel":
    """Factory function for GTR codon models.

    Supports three modes:
    - model_name: Resolve from catalog, then load from pretrained archive
    - model_path: Load from pretrained archive
    - variant: Create from inline specification (jc69, k80, hky, gtr)
    """
    import torch

    from dasm2.neutral.gtr_codon import GTRCodonNeutralModel

    if config.model_name is not None:
        model_path = resolve_model(config.model_name)
        return GTRCodonNeutralModel.from_catalog(config.model_name, model_path)

    if config.model_path is not None:
        return GTRCodonNeutralModel.from_pretrained(config.model_path)

    variant = config.variant
    if variant == "jc69":
        return GTRCodonNeutralModel.from_jc69()
    elif variant == "k80":
        assert config.kappa is not None  # Guaranteed by Pydantic validation
        return GTRCodonNeutralModel.from_k80(config.kappa)
    elif variant == "hky":
        assert config.kappa is not None  # Guaranteed by Pydantic validation
        assert config.frequencies is not None  # Guaranteed by Pydantic validation
        return GTRCodonNeutralModel.from_hky(
            config.kappa, torch.tensor(config.frequencies)
        )
    elif variant == "gtr":
        assert config.exchangeabilities is not None  # Guaranteed by Pydantic validation
        assert config.frequencies is not None  # Guaranteed by Pydantic validation
        return GTRCodonNeutralModel(
            torch.tensor(config.exchangeabilities),
            torch.tensor(config.frequencies),
        )
    else:
        raise ValueError(f"Unknown GTR variant: {variant}")


def _create_per_family_gtr_model(
    config: PerFamilyGTRNeutralModelConfig,
) -> "AbstractNeutralModel":
    """Factory function for the per-family GTR model (catalog name or archive)."""
    from dasm2.molevol.tandem import TandemTerm
    from dasm2.neutral.per_family_gtr import PerFamilyGTRNeutralModel

    if config.model_name is not None:
        model = PerFamilyGTRNeutralModel.from_catalog(
            config.model_name, resolve_model(config.model_name)
        )
    else:
        assert config.model_path is not None  # Guaranteed by Pydantic validation
        model = PerFamilyGTRNeutralModel.from_pretrained(config.model_path)
    model.set_tandem(TandemTerm(config.tandem_delta, config.tandem_delta13))
    return model


def _create_shm_codon_model(
    config: ShmCodonNeutralModelConfig,
) -> "AbstractNeutralModel":
    """Factory function for the SHM-1 codon CTMC neutral model.

    ``dat_path=None`` loads the SHM-1 matrix packaged with dasm2.
    """
    from dasm2.neutral.shm_codon import ShmCodonNeutralModel

    return ShmCodonNeutralModel.from_dat(config.dat_path)


# Registry mapping NeutralModelType to factory functions
# Adding new model types requires only: (1) a factory function, (2) registry entry
_NEUTRAL_MODEL_FACTORY_REGISTRY: Dict[
    NeutralModelType, Callable[[NeutralModelConfig], "AbstractNeutralModel"]
] = {
    NeutralModelType.THRIFTY_SHM: _create_thrifty_shm_model,  # type: ignore[dict-item]
    NeutralModelType.TABULAR_RATES: _create_tabular_rates_model,  # type: ignore[dict-item]
    NeutralModelType.S5F: _create_s5f_model,  # type: ignore[dict-item]
    NeutralModelType.PRECOMPUTED_RATES: _create_precomputed_rates_model,  # type: ignore[dict-item]
    NeutralModelType.GTR_CODON: _create_gtr_codon_model,  # type: ignore[dict-item]
    NeutralModelType.PER_FAMILY_GTR: _create_per_family_gtr_model,  # type: ignore[dict-item]
    NeutralModelType.SHM_CODON: _create_shm_codon_model,  # type: ignore[dict-item]
}


def create_neutral_model_from_config(
    config: NeutralModelConfig,
) -> "AbstractNeutralModel":
    """Factory function to instantiate neutral model from config.

    Uses registry pattern - adding new model types requires only:
    1. Creating a factory function (_create_xxx_model)
    2. Adding an entry to _NEUTRAL_MODEL_FACTORY_REGISTRY

    Args:
        config: NeutralModelConfig specifying model type and parameters

    Returns:
        Instantiated neutral model

    Raises:
        ValueError: If config.type is not in the registry

    Example:
        >>> # ThriftySHM model (default)
        >>> config = ThriftySHMNeutralModelConfig(model_name="ThriftyHumV0.2-59")
        >>> model = create_neutral_model_from_config(config)
        >>>
        >>> # With multihit correction
        >>> config = ThriftySHMNeutralModelConfig(
        ...     model_name="ThriftyHumV0.2-59",
        ...     multihit_model_name="ThriftyHumV0.2-59-hc-tangshm"
        ... )
        >>> model = create_neutral_model_from_config(config)
        >>>
        >>> # Tabular rates model
        >>> config = TabularRatesNeutralModelConfig(
        ...     model_path="data/neutral_models/my_rates.zip"
        ... )
        >>> model = create_neutral_model_from_config(config)
    """
    if config.type not in _NEUTRAL_MODEL_FACTORY_REGISTRY:
        available_types = [t.value for t in _NEUTRAL_MODEL_FACTORY_REGISTRY.keys()]
        raise ValueError(
            f"Unknown neutral model type: '{config.type}'. "
            f"Available types: {available_types}"
        )

    factory = _NEUTRAL_MODEL_FACTORY_REGISTRY[config.type]
    return factory(config)
