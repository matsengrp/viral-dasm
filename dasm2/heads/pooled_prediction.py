"""Pooled sequence-level prediction heads: base + task-specific subclasses.

Predicts scalar or multi-target outputs from per-residue sequence embeddings by
pooling to a single fixed-size vector, then applying a linear projection. Each
task (regression, multi-label, multi-class classification) is a separate head
class because the v3 head_type discriminator picks one config per shape; the
loss math now lives on the matching ``PooledRegressionLoss`` /
``PooledMultiLabelLoss`` / ``PooledClassificationLoss`` after #621 Task #4.

Pooling strategies (shared across all three heads):
    - ``"mean"``: equal-weight average over valid positions (MaskedMeanPooling).
    - ``"attention"``: multi-head score pooling with optional projections
      (MultiHeadScorePooling, the PMA block from Lee et al. 2019).
    - ``"single_token"``: select the embedding at the first occurrence of a
      designated token index (CLS-style; defaults to the ``^`` chain separator).
"""

from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional

import torch
from torch import Tensor, nn

from dasm2.blocks.pooling import (
    MaskedMeanPooling,
    MultiHeadScorePooling,
    SingleTokenPooling,
)
from dasm2.config import PoolingType, ProjectionMode
from dasm2.data.delta_affinity import (
    BINARY_TARGET_KEY,
    CLASS_LABEL_KEY,
    DELTA_AFFINITY_KEY,
)
from dasm2.sequence.constants import AA_AMBIG_IDX, AA_SEPARATOR_IDX

from .base import PredictionHead

if TYPE_CHECKING:
    from dasm2.config import BaseHeadConfig


class PooledPredictionHead(PredictionHead):
    """Abstract base for pooled sequence-level prediction heads.

    Owns all architectural concerns: pooling module construction, mask
    derivation, forward pass, and config round-trip. Subclasses declare
    the target batch key / dtype they consume via ``target_key`` /
    ``target_dtype`` ClassVars (used by the training pipeline to populate
    the batch correctly — callers should not hardcode batch keys per
    head type). Loss math lives on the matching ``Pooled*Loss`` after
    #621 Task #4.

    Batch contract:
        - All pooling modes require ``aa_parents_idxs: LongTensor[B, L]`` in
          ``batch`` (used to compute valid-position masks for mean/attention
          pooling and to locate the target token for single-token pooling).
          Mean and attention modes additionally accept a precomputed
          ``loss_mask: BoolTensor[B, L]``; if absent, a mask is derived from
          ``aa_parents_idxs`` by keeping only standard amino-acid tokens
          (indices in ``[0, AA_AMBIG_IDX)``) and explicitly excluding the
          separator token.

    Output shape:
        - ``num_targets == 1``: ``[B]`` (squeezed).
        - ``num_targets > 1``: ``[B, num_targets]``.
    """

    # Subclasses declare the batch key + dtype for their target column.
    # Set as ClassVars so the data loader / training orchestrator can query
    # them without hardcoding the mapping elsewhere.
    target_key: ClassVar[str]
    target_dtype: ClassVar[torch.dtype]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Verify concrete subclasses declare their target spec.

        Fail-fast at class-definition time so missing ClassVars surface
        during import rather than at first training step. Abstract
        intermediate subclasses are allowed to skip the check.
        ``head_config_type`` is set via a deferred-import block at the
        bottom of this module and is checked there (see module end).
        """
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return  # still abstract — don't enforce
        for attr in ("target_key", "target_dtype"):
            if attr not in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} must declare ClassVar '{attr}' "
                    f"(see PooledPredictionHead docstring)."
                )

    def __init__(
        self,
        input_dim: int,
        num_targets: int = 1,
        pooling_type: PoolingType = "mean",
        pooling_hidden_dim: int = 64,
        num_heads: int = 1,
        head_dim: Optional[int] = None,
        projection_mode: ProjectionMode = "identity",
        target_token_idx: Optional[int] = None,
    ) -> None:
        """Initialize pooled prediction head.

        Args:
            input_dim: Input embedding dimension from sequence encoder.
            num_targets: Output dimensionality (targets per sequence).
            pooling_type: One of ``"mean"``, ``"attention"``, ``"single_token"``.
            pooling_hidden_dim: Hidden dim of MHSP scoring network (attention).
            num_heads: Number of MHSP attention heads (attention).
            head_dim: Per-head output dim (attention); defaults to input_dim.
            projection_mode: MHSP projection mode (attention).
            target_token_idx: Token index for single-token pooling; defaults to
                ``AA_SEPARATOR_IDX`` when ``pooling_type="single_token"``.

        Raises:
            ValueError: On invalid ``pooling_type`` or non-positive ``num_targets``.
        """
        super().__init__()

        if num_targets < 1:
            raise ValueError(f"num_targets must be >= 1, got {num_targets}")

        self.input_dim = input_dim
        self.pooling_type = pooling_type
        self.pooling_hidden_dim = pooling_hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.projection_mode = projection_mode
        self.num_targets = num_targets

        self.pooling, pooled_dim, resolved_target = self._build_pooling(
            pooling_type=pooling_type,
            input_dim=input_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            pooling_hidden_dim=pooling_hidden_dim,
            projection_mode=projection_mode,
            target_token_idx=target_token_idx,
        )
        self.target_token_idx: Optional[int] = resolved_target

        self.prediction_layer = nn.Linear(pooled_dim, num_targets)

    def _build_pooling(
        self,
        *,
        pooling_type: PoolingType,
        input_dim: int,
        num_heads: int,
        head_dim: Optional[int],
        pooling_hidden_dim: int,
        projection_mode: ProjectionMode,
        target_token_idx: Optional[int],
    ) -> tuple[nn.Module, int, Optional[int]]:
        """Construct the pooling module, pooled output dim, and resolved target token."""
        if pooling_type == "mean":
            return MaskedMeanPooling(), input_dim, None
        if pooling_type == "attention":
            pooling = MultiHeadScorePooling(
                input_dim=input_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_dim=pooling_hidden_dim,
                projection_mode=projection_mode,
            )
            return pooling, pooling.output_dim, None
        if pooling_type == "single_token":
            resolved = (
                target_token_idx if target_token_idx is not None else AA_SEPARATOR_IDX
            )
            return SingleTokenPooling(target_token_idx=resolved), input_dim, resolved
        raise ValueError(
            "pooling_type must be one of 'mean', 'attention', 'single_token'; "
            f"got {pooling_type!r}"
        )

    def _resolve_mask(self, batch: Dict[str, Tensor]) -> Tensor:
        """Return a boolean mask of valid (non-padding, non-separator) positions.

        Called only for ``pooling_type="mean"`` and ``pooling_type="attention"``;
        ``single_token`` pooling reads ``aa_parents_idxs`` directly.

        Uses ``batch['loss_mask']`` when present. Otherwise derives a mask from
        ``batch['aa_parents_idxs']`` by keeping only standard amino-acid tokens
        (indices in ``[0, AA_AMBIG_IDX)``) and explicitly excluding the
        ``^`` separator. The explicit separator check guards against future
        token-table reorderings where a reserved token might sit below
        ``AA_AMBIG_IDX``.
        """
        if "loss_mask" in batch:
            mask = batch["loss_mask"]
            if mask.dtype != torch.bool:
                mask = mask.bool()
            return mask

        if "aa_parents_idxs" not in batch:
            raise ValueError(
                f"{type(self).__name__} requires 'loss_mask' or 'aa_parents_idxs' "
                "in batch. Got keys: " + str(list(batch.keys()))
            )

        token_ids = batch["aa_parents_idxs"]
        return (token_ids < AA_AMBIG_IDX) & (token_ids != AA_SEPARATOR_IDX)

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Predict sequence-level targets from per-residue embeddings.

        Args:
            sequence_embeddings: Per-residue embeddings [B, L, input_dim].
            batch: Batch data. Must contain ``aa_parents_idxs`` for
                single-token pooling and either ``loss_mask`` or
                ``aa_parents_idxs`` for mean/attention pooling.

        Returns:
            Predictions with shape ``[B]`` when ``num_targets == 1`` and
            ``[B, num_targets]`` otherwise.
        """
        pooled: Tensor
        if self.pooling_type == "mean":
            pooled = self.pooling(sequence_embeddings, self._resolve_mask(batch))
        elif self.pooling_type == "attention":
            # MHSP returns (pooled, attn_weights); callers that want weights
            # should interact with self.pooling directly.
            pooled, _ = self.pooling(sequence_embeddings, self._resolve_mask(batch))
        else:  # "single_token" — validated in _build_pooling
            if "aa_parents_idxs" not in batch:
                raise ValueError(
                    f"{type(self).__name__} with pooling_type='single_token' "
                    "requires 'aa_parents_idxs' in batch."
                )
            pooled = self.pooling(sequence_embeddings, batch["aa_parents_idxs"])
        logits: Tensor = self.prediction_layer(pooled)  # [B, num_targets]
        if self.num_targets == 1:
            return logits.squeeze(-1)  # [B]
        return logits

    def to_config(self) -> "BaseHeadConfig":
        """Reconstruct config from current head state.

        Follows the :class:`PerSiteEncoderFreeHead.to_config` pattern: delegates
        to ``self.head_config_type`` so each subclass emits its own
        discriminated config. Uses the *resolved* ``target_token_idx`` so
        round-trips of ``pooling_type="single_token"`` with a defaulted
        target record the resolved separator index.
        """
        return self.head_config_type(  # type: ignore[call-arg]
            input_dim=self.input_dim,
            pooling_type=self.pooling_type,
            pooling_hidden_dim=self.pooling_hidden_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            projection_mode=self.projection_mode,
            target_token_idx=self.target_token_idx,
            num_targets=self.num_targets,
        )


class PooledRegressionHead(PooledPredictionHead):
    """Pooled sequence-level head for real-valued regression targets.

    Loss: ``PooledRegressionLoss`` (Huber on ``batch['delta_affinity']``).

    With ``pooling_type="mean"`` and ``num_targets=1`` (defaults), this head
    is architecturally and numerically equivalent to :class:`DeltaAffinityHead`
    *under matching batch contracts* — verified by a weight-copy phase-gate
    test in ``tests/heads/test_pooled_prediction.py``.

    **Batch-contract caveat**: ``DeltaAffinityHead`` hard-requires
    ``batch["loss_mask"]`` (raises if missing). ``PooledRegressionHead``
    falls back to deriving a mask from ``aa_parents_idxs`` via
    ``_resolve_mask`` when ``loss_mask`` is absent. Datasets that pass
    only ``aa_parents_idxs`` will run through ``PooledRegressionHead`` and
    crash inside ``DeltaAffinityHead``, even though the weights and
    pooling are identical. Either supply ``loss_mask`` in the batch, or
    accept that the two heads are interchangeable only when the data
    pipeline produces both keys.
    """

    target_key: ClassVar[str] = DELTA_AFFINITY_KEY
    target_dtype: ClassVar[torch.dtype] = torch.float32


class PooledMultiLabelHead(PooledPredictionHead):
    """Pooled sequence-level head for binary / multi-label classification.

    Loss: ``PooledMultiLabelLoss`` (BCE-with-logits on
    ``batch['binary_target']``). Covers binary classification at
    ``num_targets=1`` and multi-label at ``num_targets > 1``.
    """

    target_key: ClassVar[str] = BINARY_TARGET_KEY
    target_dtype: ClassVar[torch.dtype] = torch.float32


class PooledClassificationHead(PooledPredictionHead):
    """Pooled sequence-level head for mutually-exclusive multi-class classification.

    Loss: ``PooledClassificationLoss`` (cross-entropy on
    ``batch['class_label']``).

    Requires ``num_targets >= 2``: cross-entropy over a single class is
    degenerate (the softmax of a scalar is always 1). Use
    :class:`PooledMultiLabelHead` for binary classification with a single
    output.
    """

    target_key: ClassVar[str] = CLASS_LABEL_KEY
    target_dtype: ClassVar[torch.dtype] = torch.long

    def __init__(
        self,
        input_dim: int,
        num_targets: int = 2,
        pooling_type: PoolingType = "mean",
        pooling_hidden_dim: int = 64,
        num_heads: int = 1,
        head_dim: Optional[int] = None,
        projection_mode: ProjectionMode = "identity",
        target_token_idx: Optional[int] = None,
    ) -> None:
        """Initialize classification head. See :class:`PooledPredictionHead`."""
        super().__init__(
            input_dim=input_dim,
            num_targets=num_targets,
            pooling_type=pooling_type,
            pooling_hidden_dim=pooling_hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            projection_mode=projection_mode,
            target_token_idx=target_token_idx,
        )
        if self.num_targets < 2:
            raise ValueError(
                "PooledClassificationHead requires num_targets >= 2 (got "
                f"num_targets={self.num_targets}). Use PooledRegressionHead for "
                "scalar regression or PooledMultiLabelHead for binary "
                "classification with a single output."
            )


# Set config types after class definitions to avoid circular import
from dasm2.config import (  # noqa: E402
    PooledClassificationHeadConfig,
    PooledMultiLabelHeadConfig,
    PooledRegressionHeadConfig,
)

PooledRegressionHead.head_config_type = PooledRegressionHeadConfig
PooledMultiLabelHead.head_config_type = PooledMultiLabelHeadConfig
PooledClassificationHead.head_config_type = PooledClassificationHeadConfig

# Fail-fast: if a future subclass is added above and this block forgets to
# assign its head_config_type, surface the omission at import time.
for _cls in (PooledRegressionHead, PooledMultiLabelHead, PooledClassificationHead):
    assert "head_config_type" in _cls.__dict__, (
        f"{_cls.__name__} missing head_config_type — add it to the "
        f"assignment block above."
    )
del _cls
