"""Head factory using registry pattern."""

from typing import TYPE_CHECKING, Dict, Type, Union

if TYPE_CHECKING:
    from dasm2.config import (
        AAHWhichmutHeadConfig,
        BindingClassificationHeadConfig,
        ContrastiveBindingHeadConfig,
        DASMFitResidualHeadConfig,
        DeltaAffinityHeadConfig,
        DMSFunctionalScoreHeadConfig,
        HybridEvolHeadConfig,
        M0HeadConfig,
        MLMHeadConfig,
        NeutralHeadConfig,
        PerSiteChildHeadConfig,
        PerSiteMeanSelectionHeadConfig,
        PerSiteParentChildHeadConfig,
        PerSitePreferenceHeadConfig,
        PooledClassificationHeadConfig,
        PooledMultiLabelHeadConfig,
        PooledRegressionHeadConfig,
        QDASMEvolHeadConfig,
        SelectionFactorHeadConfig,
        SingleMatrixHeadConfig,
        SingleValueHeadConfig,
        SiteMaskedSelectionFactorHeadConfig,
        StructuralGuidanceHeadConfig,
        WhichmutResidualHeadConfig,
    )

from .aah_whichmut import AAHWhichmutHead
from .base import PredictionHead
from .binding_classification import BindingClassificationHead
from .contrastive_binding import ContrastiveBindingHead
from .dasmfit_residual import DASMFitResidualHead
from .delta_affinity import DeltaAffinityHead
from .dms import DMSFunctionalScoreHead
from .hybrid import HybridEvolHead
from .m0 import M0Head
from .mlm import MLMHead
from .neutral import NeutralHead
from .per_site_child import PerSiteChildHead
from .per_site_mean_selection import PerSiteMeanSelectionHead
from .per_site_parent_child import PerSiteParentChildHead
from .per_site_preference import PerSitePreferenceHead
from .pooled_prediction import (
    PooledClassificationHead,
    PooledMultiLabelHead,
    PooledRegressionHead,
)
from .qdasm import QDASMEvolHead
from .selection_factor import SelectionFactorHead
from .single_matrix import SingleMatrixHead
from .single_value import SingleValueHead
from .site_masked import SiteMaskedSelectionFactorHead
from .structural_guidance import StructuralGuidanceHead
from .types import HeadType
from .whichmut_residual import WhichmutResidualHead

# Type alias for HeadConfig discriminated union
HeadConfig = Union[
    "AAHWhichmutHeadConfig",
    "BindingClassificationHeadConfig",
    "ContrastiveBindingHeadConfig",
    "DASMFitResidualHeadConfig",
    "DeltaAffinityHeadConfig",
    "HybridEvolHeadConfig",
    "M0HeadConfig",
    "MLMHeadConfig",
    "NeutralHeadConfig",
    "QDASMEvolHeadConfig",
    "SelectionFactorHeadConfig",
    "PerSiteMeanSelectionHeadConfig",
    "PerSiteParentChildHeadConfig",
    "PerSitePreferenceHeadConfig",
    "SingleMatrixHeadConfig",
    "SiteMaskedSelectionFactorHeadConfig",
    "SingleValueHeadConfig",
    "PerSiteChildHeadConfig",
    "PooledRegressionHeadConfig",
    "PooledMultiLabelHeadConfig",
    "PooledClassificationHeadConfig",
    "StructuralGuidanceHeadConfig",
    "DMSFunctionalScoreHeadConfig",
    "WhichmutResidualHeadConfig",
]

# Registry pattern - Open/Closed principle compliant
# Maps HeadType enum values to head classes.
HEAD_REGISTRY: Dict[HeadType, Type[PredictionHead]] = {
    HeadType.SELECTION_FACTOR_HEAD: SelectionFactorHead,
    HeadType.AAH_WHICHMUT_HEAD: AAHWhichmutHead,
    HeadType.PER_SITE_CHILD_HEAD: PerSiteChildHead,
    HeadType.STRUCTURAL_GUIDANCE_HEAD: StructuralGuidanceHead,
    HeadType.DMS_FUNCTIONAL_SCORE_HEAD: DMSFunctionalScoreHead,
    HeadType.WHICHMUT_RESIDUAL_HEAD: WhichmutResidualHead,
    HeadType.DELTA_AFFINITY_HEAD: DeltaAffinityHead,
    HeadType.DASM_FIT_RESIDUAL_HEAD: DASMFitResidualHead,
    HeadType.PER_SITE_MEAN_SELECTION_HEAD: PerSiteMeanSelectionHead,
    HeadType.PER_SITE_PARENT_CHILD_HEAD: PerSiteParentChildHead,
    HeadType.PER_SITE_PREFERENCE_HEAD: PerSitePreferenceHead,
    HeadType.QDASM_EVOL_HEAD: QDASMEvolHead,
    HeadType.HYBRID_EVOL_HEAD: HybridEvolHead,
    HeadType.SINGLE_MATRIX_HEAD: SingleMatrixHead,
    HeadType.SINGLE_VALUE_HEAD: SingleValueHead,
    HeadType.M0_HEAD: M0Head,
    HeadType.NEUTRAL_HEAD: NeutralHead,
    HeadType.POOLED_REGRESSION_HEAD: PooledRegressionHead,
    HeadType.POOLED_MULTI_LABEL_HEAD: PooledMultiLabelHead,
    HeadType.POOLED_CLASSIFICATION_HEAD: PooledClassificationHead,
    HeadType.CONTRASTIVE_BINDING_HEAD: ContrastiveBindingHead,
    HeadType.BINDING_CLASSIFICATION_HEAD: BindingClassificationHead,
    HeadType.MLM_HEAD: MLMHead,
    HeadType.SITE_MASKED_SELECTION_FACTOR_HEAD: SiteMaskedSelectionFactorHead,
}


def create_head(head_config: HeadConfig) -> PredictionHead:
    """Factory function to create prediction heads from configuration.

    Uses registry pattern with class-based construction to eliminate conditionals.
    Each head class implements from_config() to handle its own construction logic.
    Adding new head types only requires: (1) implementing from_config() and
    (2) registering in HEAD_REGISTRY.

    Args:
        head_config: Head configuration with type and parameters

    Returns:
        Initialized prediction head

    Raises:
        ValueError: If head type is unknown

    Example:
        >>> config = SelectionFactorHeadConfig(input_dim=256)
        >>> head = create_head(config)
    """
    head_type = head_config.head_type

    if head_type not in HEAD_REGISTRY:
        available_heads = [ht.value for ht in HEAD_REGISTRY.keys()]
        raise ValueError(
            f"Unknown head type: '{head_type.value}'. Available heads: {available_heads}"
        )

    head_class = HEAD_REGISTRY[head_type]

    # Delegate to head class's from_config() - no conditionals needed
    # Each head knows how to construct itself from its specific config type
    return head_class.from_config(head_config)  # type: ignore[arg-type]


def register_head(head_type: HeadType, head_class: Type[PredictionHead]) -> None:
    """Register a new head type with the factory.

    Args:
        head_type: HeadType enum value for this head
        head_class: PredictionHead class to register

    Example:
        >>> class MyCustomHead(PredictionHead):
        ...     pass
        >>> # First add to HeadType enum, then register:
        >>> register_head(HeadType.MY_CUSTOM_HEAD, MyCustomHead)
    """
    HEAD_REGISTRY[head_type] = head_class
