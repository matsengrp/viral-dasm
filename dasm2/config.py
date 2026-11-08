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

from dasm2.heads.types import HeadType
from dasm2.losses.types import LossType, WeightMode
from dasm2.sequence.constants import NUM_AMINO_ACIDS
from dasm2.sequence.masking import CensorMode



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

    @property
    def d_model(self) -> int:
        """Total model dimension calculated as nhead × d_model_per_head."""
        return self.nhead * self.d_model_per_head

    @property
    def uses_distance(self) -> bool:
        """Whether this encoder requires distance in the batch."""
        return self.distance_encoding_mode != "none"










class BaseHeadConfig(BaseModel):
    """Base head configuration with common fields.

    All heads share these fields. Specific head types extend this with
    additional required fields. Uses discriminated union pattern for
    fail-fast validation at deserialization time.

    The canonical home for ``loss_weight`` is the ``Task`` object, not the
    head config; the field is not carried on this base class.
    """

    head_type: HeadType = Field(description="Head type discriminator")
    input_dim: Optional[int] = Field(
        default=None,
        gt=0,
        description="Input dimension (auto-set to encoder d_model if not specified)",
    )

    model_config = ConfigDict(extra="forbid")


class SelectionFactorHeadConfig(BaseHeadConfig):
    """Configuration for ``SelectionFactorHead``.

    The unified config for the head that consumes the shared selection
    factor tensor and identity-forwards it. The matching loss class —
    ``DASMLoss`` / ``WhichmutLoss`` / ``AAOWhichmutLoss`` — is chosen
    independently via ``LossConfig`` and attached to ``Task.loss``.
    """

    head_type: Literal[HeadType.SELECTION_FACTOR_HEAD] = HeadType.SELECTION_FACTOR_HEAD












    # No additional fields needed - input_dim inherited from base


    # No additional fields needed - input_dim inherited from base


    # No additional fields needed - input_dim inherited from base








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
























# Discriminated union for fail-fast validation at deserialization
HeadConfig = Annotated[
    Union[
        SelectionFactorHeadConfig,
        PerSiteParentChildHeadConfig,
    ],
    Field(discriminator="head_type"),
]


# ──────────────────────────────────────────────────────────────────────────────
# LossConfig discriminated union.
# Each variant maps 1:1 to a `Loss(nn.Module)` class via `create_loss` in
# `dasm2.losses.factory`. `loss_type` is the discriminator. Defaults are
# unambiguous per loss family.
# ──────────────────────────────────────────────────────────────────────────────


class BaseLossConfig(BaseModel):
    """Base loss configuration with discriminator."""

    loss_type: LossType = Field(description="Loss type discriminator")
    model_config = ConfigDict(extra="forbid")









class WhichmutLossConfig(BaseLossConfig):
    """Configuration for ``WhichmutLoss``. No tunables today."""

    loss_type: Literal[LossType.WHICHMUT] = LossType.WHICHMUT




















#: Which in-batch pairs the contrastive collator marks as positives.
PositiveRule = Literal["antigen_key", "sequence"]










#: One loss survives the fork, so this is an alias rather than a discriminated
#: union. ``loss_type`` is still a literal field on the model, so configs that
#: name it keep validating and one that names another loss fails there.
LossConfig = WhichmutLossConfig


class TaskConfig(BaseModel):
    """v4 declarative task entry: pairs a head with a loss + weight.

    The ``TrainingConfig.tasks`` block holds one ``TaskConfig`` per
    (head, loss) pair the trainer should construct. ``head_type`` points at
    one of the entries in ``model.heads`` (deduplicated -- a head with two
    losses yields two ``TaskConfig`` entries sharing one ``head_type``
    pointer).
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
        "valid (not just >0): OverfitGuardHook legitimately zeroes "
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
    an older YAML carrying ``patience: N`` (epoch semantics) fails with
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




















# Discriminated union for fail-fast validation at deserialization
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
        "Halpern-Bruno fixation factor. None keeps the wiggle "
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

    encoder: EncoderConfig
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

        The masked head must be the model's only head, it feeds the
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






class TrainingConfig(BaseModel):
    """Top-level training configuration owning all components."""

    # Model architecture
    model: ModelConfig

    # Training components
    #: Always ``None`` in this fork. viral-DASM takes neutral rates from the
    #: mat-pcp censor tables through ``data/neutral_rates_source.py`` and never
    #: fits one, so every neutral-model config class and its builder went with
    #: the ``neutral/`` package (#4). The field stays, typed to admit nothing
    #: but ``None``, because the no-neutral-model branches downstream are the
    #: path this pipeline takes; a config that names one now fails validation
    #: here rather than at the first batch.
    neutral_model: None = None
    codon_pref_table: Optional[Path] = Field(
        default=None,
        description="Path to a CSC (codon stability coefficient) optimality TSV. "
        "When set, the within-AA log-optimality is applied to neutral codon "
        "probabilities. The strength β_CSC is fixed at 1 unless "
        "``fit_codon_strengths`` is set. Incompatible with "
        "DASM2_NEUTRAL_MODE=matrix_exp. Reaches the training loss on every "
        "loader/mode (CSV, pickle, Parquet; precomputed or streaming).",
    )
    codon_pref_tai_table: Optional[Path] = Field(
        default=None,
        description="Path to a tAI (tRNA adaptation index) optimality TSV "
        "(second codon-optimality axis). Requires "
        "``codon_pref_table`` (CSC) to also be set; the two axes are combined "
        "as Σ_k β_k·ξ̃ᵏ. Same matrix-exp incompatibility as ``codon_pref_table``, "
        "and reaches the training loss on every loader/mode the same way.",
    )
    fit_codon_strengths: bool = Field(
        default=False,
        description="When set (and a codon-optimality table is configured), fit "
        "the per-axis strengths β_k jointly with the model via the "
        "CodonStrengthOptimizer between-cycle hook. When False, the "
        "strengths stay fixed at 1 (the β=1 smoke-test behavior). Works on every "
        "loader/mode: codon_pref_log is shared-memory dataset state on "
        "streaming/Parquet bundles (DESIGN.md D6), so the between-cycle refit "
        "reaches them the same way BranchLengthOptimizer already reaches branch "
        "lengths.",
    )
    fit_multihit_factors: bool = Field(
        default=False,
        description="When set (and a multihit-correction model is configured via "
        "neutral_model.multihit_model_name), refit the 3 log hit-class factors "
        "jointly with the model via the MultihitOptimizer between-cycle hook "
        "replacing the model's pretrained values with a fit "
        "conditioned on the current dataset and selection factors. When False, "
        "the factors stay fixed at whatever the neutral model was loaded with. "
        "Works on every loader/mode: log_hit_class_factors is a shared-memory "
        "parameter on the bundle's multihit model (DESIGN.md D6), so the "
        "between-cycle refit reaches streaming and Parquet-backed bundles the "
        "same way BranchLengthOptimizer already reaches branch lengths. "
        "On a corpus with no neutral model of its own, configure "
        "the correction through the PrecomputedRates carrier "
        "(PrecomputedRatesNeutralModelConfig).",
    )
    context_free_neutral_model: None = Field(
        default=None,
        description="Always None in this fork; see neutral_model.",
    )
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=ReduceOnPlateauSchedulerConfig)
    head_optimizer: Optional[OptimizerConfig] = Field(
        default=None,
        description="Optional separate optimizer scoped to one head's own "
        "parameters, e.g. slowing one head's "
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

    # Training hyperparameters — step-based budget.
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
        "batch's padded forward blows up. "
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
        "that can OOM a worker.",
    )
    parquet_cache_budget_gb: Optional[float] = Field(
        default=None,
        gt=0,
        description="RAM budget in GB of decoded row groups (arrow nbytes, "
        "measured at insert) for each copy of a Parquet backend's row-group "
        "cache; the main process and every DataLoader worker hold their own. "
        "None (default) keeps the older bound: as many row groups as fit "
        "2 GiB of parquet total_byte_size, capped at the working set. When "
        "set, the budget is not clamped, but startup warns if the main "
        "process and train workers at this budget, plus the val workers' "
        "sweep caches, would exceed 25% of MemTotal. The "
        "DASM2_PARQUET_CACHE_BUDGET_GB env var overrides this field.",
    )
    dataloader_pin_memory: bool = Field(
        default=True,
        description="Whether to pin DataLoader host memory. Pinned memory speeds "
        "CPU->GPU transfer but is page-locked and non-swappable, so under host-RAM "
        "pressure it cannot be paged out and can push a constrained node into the "
        "OOM-killer. Set False to disable on such nodes. Only takes "
        "effect on CUDA devices.",
    )

    bl_optimization_num_workers: Optional[int] = Field(
        default=None,
        gt=0,
        description="Worker processes for post-cycle branch-length optimization. "
        "None (default) auto-picks min(cpu_count // 2, 10). Lower it (e.g. 6) to "
        "bound peak host RAM: each worker holds its own optimization chunk, and on "
        "a large per-cycle corpus the aggregate footprint can OOM-kill a worker at "
        "the cycle boundary.",
    )
    bl_optimization_max_chunk_size: int = Field(
        default=1000,
        gt=0,
        description="Upper bound on sequences per branch-length optimization chunk. "
        "Each in-flight chunk is held in host RAM (at most num_workers + 1 alive at "
        "once), so this caps the BL-optimization working set. 1000 (default) matches "
        "historical behavior; lower it (e.g. 250) on a large per-cycle corpus that "
        "OOM-kills a worker at the cycle boundary. The effective chunk "
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
        "without that column freezes at DEFAULT_BRANCH_LENGTH=0.1. Without "
        "this flag, even a cycles=1 run overwrites the input branch lengths "
        "at both cycle boundaries, which confounds any comparison that needs "
        "branch lengths held fixed.",
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
        "additionally freezes cycle 0. It stops the model from co-adapting "
        "branch lengths to its own suppressed selection across cycles while "
        "still training against a sensible BL estimate. Mutually exclusive "
        "with "
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
        "config's base learning rate. Default False preserves "
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
        "construction time.",
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

    # UShER censoring. Two paths rather than one plus a derived sibling:
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
    # The other mask file. Named for `{gene}_codon_sites_to_mask.txt` and
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
    # The same global mask at nucleotide granularity, and the one to
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
    # The third source of censored candidates. Unlike the two above this
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
        "different things; each on its own still trains and produces a "
        "plausible curve, which is precisely why they must be named explicitly. "
        "Shared by every censoring source, including reference_sequence_file, "
        "where the halves matter more because that rule is "
        "numerator-dominated.",
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
        "optimizer step — e.g. PCGradHook. Unlike pre_hooks/"
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

    # Declarative task block: one entry per (head, loss) pair. A head with
    # two losses expands to two entries sharing one ``head_type`` pointer.
    tasks: List[TaskConfig] = Field(
        default_factory=list,
        description=(
            "Declarative tasks block. Each entry pairs a head "
            "(by ``head_type`` discriminator) with a ``LossConfig`` and a "
            "``loss_weight``. AAH expands to two entries (whichmut + "
            "aao_whichmut) sharing the same ``head_type`` pointer."
        ),
    )

    @model_validator(mode="after")
    def _validate_censoring(self) -> "TrainingConfig":
        """Check the censoring inputs are complete and reach a whichmut loss.

        Both files or neither: with only the table, the loader cannot tell an
        empty clade from a typo; with only the roster, there is nothing to
        censor. And censoring is defined only where ``codon_neutral_rates``
        exists, so pointing it at a non-whichmut task would configure a file that
        is silently never read -- the failure mode this whole change is about.

        The global mask -- ``masked_nt_sites_file`` or the older
        ``codon_sites_to_mask_file``, never both -- rides on top of that
        pair rather than standing alone; see the comments at their checks for why
        global-only censoring is deliberately unrepresentable and why the two
        granularities exclude each other.

        ``reference_sequence_file`` is the opposite case and **must** be
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
        # case needs it, and a second construction path for a case nobody
        # needs is speculative generality.
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
                f"you are reproducing a run that predates it."
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
            # it was scoped out to keep that change one variable, and
            # `_load_whichmut_head_data` raises NotImplementedError on it. Say so
            # here rather than at data-loading time: both are loud, but a config
            # that cannot run should not survive config validation.
            if any(
                task.loss_config.loss_type == LossType.AAO_WHICHMUT
                for task in self.tasks
            ):
                raise ValueError(
                    f"Censoring is not yet implemented for the AAO whichmut path "
                    f"(it was scoped out deliberately, to keep that change "
                    f"one variable). Remove "
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


















