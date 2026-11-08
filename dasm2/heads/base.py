"""Base classes for prediction heads.

Heads own only the forward pass; loss math lives on
``Loss`` modules in ``dasm2/losses/`` (and is registered on the trainer
via the per-task ``Task.loss``). The ``compute_loss`` ABC is gone;
``SharedSelectionFactorHead`` is now a thin identity-forward
``PredictionHead`` for SF-consumer heads.
"""

import inspect
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Dict, FrozenSet, Type

from torch import Tensor, nn

if TYPE_CHECKING:
    from dasm2.config import BaseHeadConfig


class Capability(Enum):
    """An output space a prediction head can produce.

    Validation metric groups are gated on the head's declared
    capabilities rather than on the concrete loss class: a metric is a
    property of the *tensor the head emits*, not of the loss that happens
    to consume it. See ``dasm2/training/metric_providers.py`` and the
    "Eval-metric gating" section of ``DESIGN.md``.

    Only output spaces that are genuinely a function of the head belong
    here. Codon-distribution eligibility is deliberately *not* a
    capability: it is loss-intrinsic (``AAOWhichmutLoss`` is AA-level and
    cannot produce a codon distribution even though it shares a head with
    codon-capable losses), so the codon provider gates on the loss
    supplying ``predict_codons`` instead.
    """

    SELECTION_FACTOR = "selection_factor"


class PredictionHead(nn.Module, ABC):
    """Abstract base class for prediction heads.

    Subclasses must define:
    - head_config_type: ClassVar pointing to their specific config class
      (set after class definition to avoid circular imports)
    - All abstract methods (forward, to_config)

    Heads that consume the shared selection factor projection (instead of
    raw sequence embeddings) should subclass ``SharedSelectionFactorHead``
    instead, which sets ``consumes_shared_sf = True`` and provides an
    identity forward.
    """

    # Subclasses must set this after class definition
    head_config_type: ClassVar[Type["BaseHeadConfig"]]

    # When True, CoreModel passes shared SF (not sequence embeddings)
    # to this head's forward(). Set by SharedSelectionFactorHead; leave
    # False for heads that operate on raw encoder output.
    consumes_shared_sf: ClassVar[bool] = False

    # Output spaces this head produces. Drives which
    # capability-gated validation metric groups are emitted for tasks
    # using this head; see ``dasm2/training/metric_providers.py``. Empty
    # by default — heads opt in by listing the capabilities their forward
    # output supports.
    OUTPUT_CAPABILITIES: ClassVar[FrozenSet[Capability]] = frozenset()

    # When True, the head's config exposes a ``max_aa_seq_len`` field that
    # ModelConfig.validate_heads auto-populates from ModelConfig.max_aa_seq_len.
    # Set on heads that allocate a [max_aa_seq_len, ...] parameter tensor at
    # construction time (e.g. per-site encoder-free heads).  Typed and greppable
    # replacement for a hasattr check on the config object.
    requires_max_aa_seq_len: ClassVar[bool] = False

    # When True, models/factory.py builds an MLMCoreModel (masked-marginal
    # scoring adapter) instead of a plain CoreModel, regardless of encoder
    # type — a single unmasked forward pass is semantically wrong for a head
    # whose logits are raw reconstruction targets, not gauge-anchored
    # selection factors. Set by MLMHead; mirrors
    # consumes_shared_sf's pattern of driving model-construction dispatch
    # off a ClassVar rather than a data-loader head-type set.
    requires_masked_marginal_scoring: ClassVar[bool] = False

    # When True, models/factory.py builds a SiteMaskedCoreModel, which
    # predicts each site's fitness vector from a pass with that site masked
    #. Set by SiteMaskedSelectionFactorHead.
    requires_site_masked_core_model: ClassVar[bool] = False

    def __init__(self) -> None:
        """Initialize prediction head."""
        super().__init__()

    @abstractmethod
    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Make predictions from sequence embeddings.

        Args:
            sequence_embeddings: Embeddings [batch_size, seq_len, embed_dim]
            batch: Full batch data dict for heads that need additional context

        Returns:
            Predictions tensor with head-specific shape
        """
        pass

    @abstractmethod
    def to_config(self) -> "BaseHeadConfig":
        """Reconstruct head architecture configuration from current state.

        Returns a config object with architecture-only parameters (input_dim,
        activation settings, etc.).

        Returns:
            HeadConfig subclass matching this head's type and architecture.
        """
        pass

    @classmethod
    def from_config(cls, config: "BaseHeadConfig") -> "PredictionHead":
        """Create head instance from configuration.

        Generic implementation that works for all head types. Validates config type
        matches the head's expected config type, then unpacks all fields except
        ``head_type`` as constructor arguments.

        Args:
            config: Head configuration matching cls.head_config_type

        Returns:
            Initialized head instance

        Raises:
            TypeError: If config type doesn't match cls.head_config_type
            ValueError: If input_dim is not set (should be auto-set by ModelConfig)

        Example:
            >>> config = SelectionFactorHeadConfig(input_dim=256)
            >>> head = SelectionFactorHead.from_config(config)
        """
        # Validate config type matches expected type
        if not isinstance(config, cls.head_config_type):
            raise TypeError(
                f"{cls.__name__} expects {cls.head_config_type.__name__}, "
                f"got {type(config).__name__}"
            )

        # Validate input_dim is set (fail-fast before construction).
        # SharedSelectionFactorHead subclasses don't need input_dim — they
        # consume shared SF.
        if config.input_dim is None and not cls.consumes_shared_sf:
            raise ValueError(
                f"input_dim must be set before creating {cls.__name__} "
                f"(should be auto-set by ModelConfig.validate_heads)"
            )

        # SharedSelectionFactorHead subclasses take no constructor args — they
        # consume shared SF
        if cls.consumes_shared_sf:
            return cls()

        # Extract config fields and filter to those the constructor actually
        # takes. Some head configs may carry fields that are not constructor
        # args (e.g. fields handled out-of-band by ModelConfig validators);
        # we filter rather than crash.
        config_dict = config.model_dump()
        try:
            sig = inspect.signature(cls.__init__)
            accepted = {
                name
                for name, param in sig.parameters.items()
                if name != "self"
                and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
            }
        except (TypeError, ValueError):
            accepted = set(config_dict.keys())
        constructor_kwargs = {
            key: value
            for key, value in config_dict.items()
            if key != "head_type" and key in accepted
        }

        # Unpack to constructor
        return cls(**constructor_kwargs)


class SelectionFactorOutputHead(PredictionHead):
    """Mixin for encoder-direct heads whose forward output is the L×20 log-SF tensor.

    Distinct from ``SharedSelectionFactorHead``: those heads consume the
    shared SF projection and forward it unchanged, whereas these own their
    own projection but still emit the same selection-factor output space.
    Subclassing here declares ``Capability.SELECTION_FACTOR`` once instead
    of repeating the ``OUTPUT_CAPABILITIES`` literal on every such head, and
    makes the set of SF-producing heads discoverable through the class
    hierarchy. Subclasses still implement their own ``forward``/``to_config``.
    """

    OUTPUT_CAPABILITIES: ClassVar[FrozenSet[Capability]] = frozenset(
        {Capability.SELECTION_FACTOR}
    )


class PerSiteEncoderFreeHead(PredictionHead):
    """Base for per-site heads that ignore encoder output.

    Stores ``input_dim`` and ``max_aa_seq_len`` and provides a generic
    ``to_config()`` that delegates to ``head_config_type``.  Subclasses
    only need to implement ``forward()`` and register their parameter
    tensors in ``__init__``. The matching loss class is attached via
    ``Task.loss``.
    """

    requires_max_aa_seq_len: ClassVar[bool] = True

    def __init__(self, input_dim: int, max_aa_seq_len: int) -> None:
        """Initialize per-site encoder-free head.

        Args:
            input_dim: Embedding dimension (required by interface, but ignored).
            max_aa_seq_len: Maximum amino acid sequence length.
        """
        super().__init__()
        self.max_aa_seq_len = max_aa_seq_len
        self.input_dim = input_dim

    def to_config(self) -> "BaseHeadConfig":
        """Reconstruct config from head state."""
        return self.head_config_type(  # type: ignore[call-arg]
            input_dim=self.input_dim,
            max_aa_seq_len=self.max_aa_seq_len,
        )


class PositionAgnosticHead(PredictionHead):
    """Base for heads whose parameter shape has no ``max_aa_seq_len`` axis.

    Used by ``SingleValueHead`` (one scalar) and ``SingleMatrixHead``
    (a ``[20, 20]`` table). Distinct from the other encoder-free
    per-site heads (``PerSiteParentChildHead``,
    ``PerSitePreferenceHead``, ``PerSiteChildHead``), which are
    position-indexed (``[max_aa_seq_len, ...]`` parameters) but likewise
    ignore the encoder. The name "context-free" was avoided because that
    phrase is reserved in this codebase for AA-level neutral models
    (``context_free_neutral_model``).

    Stores ``input_dim`` (kept for interface symmetry; the heads
    themselves ignore embeddings) and provides a generic ``to_config()``
    that delegates to ``head_config_type``. Concrete subclasses should
    also inherit a loss mixin (e.g. ``DASMLossMixin``) to satisfy the
    ``compute_loss`` contract.
    """

    def __init__(self, input_dim: int) -> None:
        """Initialize position-agnostic head.

        Args:
            input_dim: Embedding dimension (required by interface, but ignored).
        """
        super().__init__()
        self.input_dim = input_dim

    def to_config(self) -> "BaseHeadConfig":
        """Reconstruct config from head state."""
        return self.head_config_type(  # type: ignore[call-arg]
            input_dim=self.input_dim,
        )


class SharedSelectionFactorHead(PredictionHead):
    """Head that consumes shared selection factors from CoreModel.

    SF-consuming heads (DASM, Whichmut, AAO Whichmut) don't own their
    own ``nn.Linear(D, 20)`` projection. Instead, CoreModel computes shared
    selection factors via ``SelectionFactorProjection`` and passes them to
    this head's ``forward()``. The forward is an identity — the matching
    loss class (attached via ``Task.loss``) does all the work.

    Subclasses only need to implement ``to_config()``.
    """

    consumes_shared_sf: ClassVar[bool] = True
    OUTPUT_CAPABILITIES: ClassVar[FrozenSet[Capability]] = frozenset(
        {Capability.SELECTION_FACTOR}
    )

    def forward(self, sf_or_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Identity — returns input unchanged.

        Satisfies the @abstractmethod contract from PredictionHead.forward().
        CoreModel passes shared selection factors (not embeddings) to this
        argument when consumes_shared_sf is True.
        """
        return sf_or_embeddings
