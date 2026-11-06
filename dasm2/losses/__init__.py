"""Loss functions for DASM2 training objectives.

After issue #621, every training objective is a first-class
``nn.Module`` Loss class registered in :data:`LOSS_REGISTRY`. The Task
layer holds a fresh per-task instance constructed via :func:`create_loss`.

The pre-#621 mixin classes in ``dasm2/heads/mixins.py`` are deleted; their
math kernels (pure functions in this package) are unchanged.
"""

from dasm2.losses.binding_classification import BindingClassificationLoss
from dasm2.losses.capability import (
    AmbiguousTaskLossError,
    CodonProbsFn,
    MissingTaskLossError,
    force_exact_evaluation,
    is_simulable_head,
    lookup_task_loss_for_head,
    resolve_codon_probs_fn,
    resolve_sampling_loss,
    resolve_simulable_head,
)
from dasm2.losses.contrastive_binding import (
    ContrastiveBindingLoss,
    ContrastiveLossVariant,
    compute_contrastive_binding_loss,
)
from dasm2.losses.dasm import DASMLoss
from dasm2.losses.dasmfit import DASMFitResidualLoss
from dasm2.losses.delta_affinity import DeltaAffinityLoss
from dasm2.losses.dms import DMSFunctionalScoreLoss
from dasm2.losses.dms_matrix import (
    DMSMatrixCalibrationLoss,
    ListMLELoss,
    PearsonCorrelationLoss,
)
from dasm2.losses.factory import LOSS_REGISTRY, create_loss, register_loss
from dasm2.losses.hybrid import HybridLoss
from dasm2.losses.pooled import (
    PooledClassificationLoss,
    PooledMultiLabelLoss,
    PooledRegressionLoss,
)
from dasm2.losses.protocols import CodonSamplingLoss, SelfNormalizingLoss
from dasm2.losses.qdasm import QDASMLoss
from dasm2.losses.structural_guidance import (
    HERMES_EMBED_DIM,
    NORMALIZATION_TOLERANCE,
    StructuralGuidanceLoss,
    compute_structural_guidance_loss,
)
from dasm2.losses.types import (
    BINDING_CLASSIFICATION_LOSS_TYPES,
    EVOLUTIONARY_LOSS_TYPES,
    QDASM_FAMILY_LOSS_TYPES,
    WHICHMUT_LOSS_TYPES,
    LossType,
)
from dasm2.losses.whichmut_losses import AAOWhichmutLoss, WhichmutLoss

__all__ = [
    # Module classes
    "AAOWhichmutLoss",
    "BindingClassificationLoss",
    "ContrastiveBindingLoss",
    "DASMLoss",
    "DASMFitResidualLoss",
    "DeltaAffinityLoss",
    "DMSFunctionalScoreLoss",
    "DMSMatrixCalibrationLoss",
    "HybridLoss",
    "ListMLELoss",
    "PearsonCorrelationLoss",
    "PooledClassificationLoss",
    "PooledMultiLabelLoss",
    "PooledRegressionLoss",
    "QDASMLoss",
    "StructuralGuidanceLoss",
    "WhichmutLoss",
    # Protocols
    "CodonSamplingLoss",
    "SelfNormalizingLoss",
    # Capability resolution (issue #1092)
    "AmbiguousTaskLossError",
    "CodonProbsFn",
    "MissingTaskLossError",
    "force_exact_evaluation",
    "is_simulable_head",
    "lookup_task_loss_for_head",
    "resolve_codon_probs_fn",
    "resolve_sampling_loss",
    "resolve_simulable_head",
    # Types and factory
    "LossType",
    "BINDING_CLASSIFICATION_LOSS_TYPES",
    "EVOLUTIONARY_LOSS_TYPES",
    "QDASM_FAMILY_LOSS_TYPES",
    "WHICHMUT_LOSS_TYPES",
    "LOSS_REGISTRY",
    "create_loss",
    "register_loss",
    # Legacy re-exports
    "ContrastiveLossVariant",
    "HERMES_EMBED_DIM",
    "NORMALIZATION_TOLERANCE",
    "compute_contrastive_binding_loss",
    "compute_structural_guidance_loss",
]
