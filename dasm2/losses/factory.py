"""Factory for constructing ``Loss(nn.Module)`` instances from ``LossConfig``.

Each ``LossConfig`` discriminated-union variant maps to one concrete Loss
class. ``create_loss(loss_config)`` returns a fresh instance — losses are
never shared across tasks; the trainer constructor asserts uniqueness by
``id()`` to keep the parameter graph tractable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

from torch import nn

from dasm2.losses.binding_classification import BindingClassificationLoss
from dasm2.losses.contrastive_binding import ContrastiveBindingLoss
from dasm2.losses.dasm import DASMLoss
from dasm2.losses.dasmfit import DASMFitResidualLoss
from dasm2.losses.delta_affinity import DeltaAffinityLoss
from dasm2.losses.dms import DMSFunctionalScoreLoss
from dasm2.losses.dms_matrix import DMSMatrixCalibrationLoss
from dasm2.losses.hybrid import HybridLoss
from dasm2.losses.mlm import MLMLoss
from dasm2.losses.pooled import (
    PooledClassificationLoss,
    PooledMultiLabelLoss,
    PooledRegressionLoss,
)
from dasm2.losses.q_scale_defaults import (
    DEFAULT_DOUBLE_HIT_SCALE,
    DEFAULT_SINGLE_HIT_SCALE,
    DEFAULT_TRIPLE_HIT_SCALE,
)
from dasm2.losses.qdasm import QDASMLoss
from dasm2.losses.structural_guidance import StructuralGuidanceLoss
from dasm2.losses.types import LossType
from dasm2.losses.whichmut_losses import AAOWhichmutLoss, WhichmutLoss

if TYPE_CHECKING:
    from dasm2.config import LossConfig


def _build_dasm(cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return DASMLoss(
        focal_gamma=cfg.focal_gamma,
        weight_mode=cfg.weight_mode,
        mut_budget=cfg.mut_budget,
        alpha=cfg.alpha,
        syn_loss_mask=cfg.syn_loss_mask,
        normalization=cfg.normalization,
    )


def _build_qdasm(cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return QDASMLoss(
        use_cubic=cfg.use_cubic,
        single_hit_scale=(
            cfg.single_hit_scale
            if cfg.single_hit_scale is not None
            else DEFAULT_SINGLE_HIT_SCALE
        ),
        double_hit_scale=(
            cfg.double_hit_scale
            if cfg.double_hit_scale is not None
            else DEFAULT_DOUBLE_HIT_SCALE
        ),
        triple_hit_scale=(
            cfg.triple_hit_scale
            if cfg.triple_hit_scale is not None
            else DEFAULT_TRIPLE_HIT_SCALE
        ),
        adaptive=cfg.adaptive,
        tq_threshold=cfg.tq_threshold,
        long_path=cfg.long_path,
        adaptive_mode=cfg.adaptive_mode,
        quadratic_ste=cfg.quadratic_ste,
        matexp_method=cfg.matexp_method,
        n_terms=cfg.n_terms,
        normalization=cfg.normalization,
        q_variant=cfg.q_variant,
        thrifty_13mer_table_path=cfg.thrifty_13mer_table_path,
        thrifty_13mer_model_name=cfg.thrifty_13mer_model_name,
        five_mer_model_name=cfg.five_mer_model_name,
        neutral_codon_q_path=cfg.neutral_codon_q_path,
    )


def _build_hybrid(cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return HybridLoss(
        use_cubic=cfg.use_cubic,
        single_hit_scale=(
            cfg.single_hit_scale
            if cfg.single_hit_scale is not None
            else DEFAULT_SINGLE_HIT_SCALE
        ),
        double_hit_scale=(
            cfg.double_hit_scale
            if cfg.double_hit_scale is not None
            else DEFAULT_DOUBLE_HIT_SCALE
        ),
        triple_hit_scale=(
            cfg.triple_hit_scale
            if cfg.triple_hit_scale is not None
            else DEFAULT_TRIPLE_HIT_SCALE
        ),
        q_variant=cfg.q_variant,
        thrifty_13mer_table_path=cfg.thrifty_13mer_table_path,
        thrifty_13mer_model_name=cfg.thrifty_13mer_model_name,
    )


def _build_whichmut(_cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return WhichmutLoss()


def _build_aao_whichmut(_cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return AAOWhichmutLoss()


def _build_dms(cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return DMSFunctionalScoreLoss(
        scale_init=cfg.scale_init, huber_delta=cfg.huber_delta
    )


def _build_delta_affinity(cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return DeltaAffinityLoss(huber_delta=cfg.huber_delta)


def _build_dms_matrix_calibration(cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return DMSMatrixCalibrationLoss(
        w_mse=cfg.w_mse,
        w_pearson=cfg.w_pearson,
        w_listmle=cfg.w_listmle,
        scale_init=cfg.scale_init,
        bias_init=cfg.bias_init,
    )


def _build_structural_guidance(_cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return StructuralGuidanceLoss()


def _build_pooled_regression(cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return PooledRegressionLoss(huber_delta=cfg.huber_delta)


def _build_pooled_multi_label(_cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return PooledMultiLabelLoss()


def _build_pooled_classification(_cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return PooledClassificationLoss()


def _build_dasm_fit_residual(cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return DASMFitResidualLoss(k=cfg.k)


def _build_contrastive_binding(cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    # `cfg.contrastive_loss_type` (variant) is distinct from `cfg.loss_type`
    # (the discriminator naming the family).
    return ContrastiveBindingLoss(loss_type=cfg.contrastive_loss_type)


def _build_binding_classification(cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return BindingClassificationLoss(pos_weight=cfg.pos_weight)


def _build_mlm(_cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return MLMLoss()


def _build_whichmut_residual(_cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    """WhichmutResidualHead reuses Whichmut math; same loss class is appropriate.

    NB: ``LossType.WHICHMUT`` and ``LossType.WHICHMUT_RESIDUAL`` both
    produce a plain ``WhichmutLoss`` instance, so ``type(task.loss)``
    cannot distinguish the two. This is intentional — the residual variant
    is a different *head* architecture sharing the same loss math.
    Anything that needs to fork on the variant must branch on
    ``task.head_type`` (or ``task.loss_config.loss_type``), not on the
    loss-instance class.
    """
    return WhichmutLoss()


LOSS_REGISTRY: Dict[LossType, Callable[..., nn.Module]] = {
    LossType.DASM: _build_dasm,
    LossType.QDASM: _build_qdasm,
    LossType.HYBRID: _build_hybrid,
    LossType.WHICHMUT: _build_whichmut,
    LossType.AAO_WHICHMUT: _build_aao_whichmut,
    LossType.DMS_FUNCTIONAL_SCORE: _build_dms,
    LossType.DELTA_AFFINITY: _build_delta_affinity,
    LossType.DMS_MATRIX_CALIBRATION: _build_dms_matrix_calibration,
    LossType.STRUCTURAL_GUIDANCE: _build_structural_guidance,
    LossType.POOLED_REGRESSION: _build_pooled_regression,
    LossType.POOLED_MULTI_LABEL: _build_pooled_multi_label,
    LossType.POOLED_CLASSIFICATION: _build_pooled_classification,
    LossType.DASM_FIT_RESIDUAL: _build_dasm_fit_residual,
    LossType.CONTRASTIVE_BINDING: _build_contrastive_binding,
    LossType.WHICHMUT_RESIDUAL: _build_whichmut_residual,
    LossType.BINDING_CLASSIFICATION: _build_binding_classification,
    LossType.MLM: _build_mlm,
}


def create_loss(loss_config: "LossConfig") -> nn.Module:
    """Construct a fresh Loss module for a single Task.

    Losses are not shared across tasks. Each ``Task`` gets its own instance
    so loss-owned parameters (DMS scale, contrastive logit_bias) live
    cleanly in the parameter graph without ``nn.ModuleDict`` aliasing.
    """
    loss_type = loss_config.loss_type
    if loss_type not in LOSS_REGISTRY:
        available = sorted(lt.value for lt in LOSS_REGISTRY)
        raise ValueError(
            f"Unknown loss type: {loss_type.value!r}. Available: {available}"
        )
    return LOSS_REGISTRY[loss_type](loss_config)


def register_loss(loss_type: LossType, builder: Callable[..., nn.Module]) -> None:
    """Register a Loss class with the factory (extension point)."""
    LOSS_REGISTRY[loss_type] = builder
