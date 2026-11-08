"""Loss type enumeration and dataset-dispatch frozensets.

Because a loss is owned by a Task rather than a head, dataset dispatch
for the L×20 evolutionary head family keys off ``LossType`` rather than
``HeadType``. The architecture-determined head sets (DMS, structural,
pooled, contrastive, delta-affinity) remain head-keyed in
``dasm2/data/registry.py`` because there is one loss per head shape.
"""

from enum import Enum
from typing import Literal

# Single source of truth for the DASMLoss class-weighting mode literal —
# referenced by both ``DASMLossConfig`` (pydantic field) and
# ``DASMLoss.__init__`` (constructor arg).
WeightMode = Literal["uniform", "static", "inverse_freq"]


class LossType(str, Enum):
    """Discriminator for ``LossConfig`` and key for state-dict task-loss naming.

    Each value is the canonical YAML/identifier name. The corresponding
    ``Loss(nn.Module)`` class lives under ``dasm2/losses/`` and is registered
    in ``LOSS_REGISTRY`` (``dasm2/losses/factory.py``).
    """

    DASM = "dasm"
    QDASM = "qdasm"
    HYBRID = "hybrid"
    WHICHMUT = "whichmut"
    AAO_WHICHMUT = "aao_whichmut"
    DMS_FUNCTIONAL_SCORE = "dms_functional_score"
    DELTA_AFFINITY = "delta_affinity"
    DMS_MATRIX_CALIBRATION = "dms_matrix_calibration"
    STRUCTURAL_GUIDANCE = "structural_guidance"
    POOLED_REGRESSION = "pooled_regression"
    POOLED_MULTI_LABEL = "pooled_multi_label"
    POOLED_CLASSIFICATION = "pooled_classification"
    DASM_FIT_RESIDUAL = "dasm_fit_residual"
    CONTRASTIVE_BINDING = "contrastive_binding"
    WHICHMUT_RESIDUAL = "whichmut_residual"
    BINDING_CLASSIFICATION = "binding_classification"
    MLM = "mlm"


# Loss types whose batches come from `WhichmutDataset` / `AAOWhichmutDataset`.
# Replaces a former `uses_whichmut_dataset` ClassVar on heads.
WHICHMUT_LOSS_TYPES: frozenset[LossType] = frozenset(
    {
        LossType.WHICHMUT,
        LossType.AAO_WHICHMUT,
    }
)


# Loss types whose batches come from `EvolutionaryDataset` (with branch
# lengths). The `BranchLengthOptimizer` hook attaches to tasks with these.
#
# Residual losses (``DASM_FIT_RESIDUAL``, ``WHICHMUT_RESIDUAL``) are
# *not* members: their datasets are whichmut-shaped (no branch lengths),
# so they neither use ``EvolutionaryDataset`` nor need BL optimization.
EVOLUTIONARY_LOSS_TYPES: frozenset[LossType] = frozenset(
    {
        LossType.DASM,
        LossType.QDASM,
        LossType.HYBRID,
    }
)


# Loss types in the QDASM family (codon-CTMC). They share the QDASM-specific
# branch-length optimization path (`optimize_branch_lengths_for_dataset_qdasm`).
QDASM_FAMILY_LOSS_TYPES: frozenset[LossType] = frozenset(
    {
        LossType.QDASM,
        LossType.HYBRID,
    }
)


# Loss types whose batches come from the pairwise-binding CSV loader
# (`dasm2/data/pairwise_binding.py`). Lets `dasm2 train`'s
# on-the-fly `cv_split` gate accept `BindingClassificationHead` alongside
# the evolutionary/whichmut families.
BINDING_CLASSIFICATION_LOSS_TYPES: frozenset[LossType] = frozenset(
    {
        LossType.BINDING_CLASSIFICATION,
    }
)
