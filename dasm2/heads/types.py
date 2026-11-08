"""Head type enumeration for prediction heads.

The dispatch frozensets that once lived here have moved:

- Loss-keyed sets (``WHICHMUT_LOSS_TYPES``, ``EVOLUTIONARY_LOSS_TYPES``,
  ``QDASM_FAMILY_LOSS_TYPES``) live in ``dasm2/losses/types.py``.
- The architecture-determined head sets (``STRUCTURAL_HEAD_TYPES``,
  ``MPNN_WHICHMUT_HEAD_TYPES``, ``DMS_HEAD_TYPES``,
  ``DELTA_AFFINITY_HEAD_TYPES``, ``CONTRASTIVE_BINDING_HEAD_TYPES``,
  ``BINDING_CLASSIFICATION_HEAD_TYPES``) live in ``dasm2/data/registry.py``.

The head-type-keyed dispatch sets ``EVOLUTIONARY_HEAD_TYPES`` and
``QDASM_FAMILY_HEAD_TYPES`` are gone. ``DASM_LOSS_HEAD_TYPES`` is
retained here as the single
source for the BL-optimizer and OE-plot dispatch paths that still need a
head-type filter — see its docstring below.
"""

from enum import Enum


class HeadType(str, Enum):
    """Enumeration of available prediction head types.

    This enum serves as both the type identifier and the configuration key
    for prediction heads. The enum values are the actual class names.

    Examples:
        >>> # In YAML config
        >>> heads:
        >>>   - head_type: SelectionFactorHead
        >>>
        >>> # In Python code
        >>> head = model.require_head(HeadType.SELECTION_FACTOR_HEAD)
        >>> predictions = head(embeddings, batch)
    """

    SELECTION_FACTOR_HEAD = "SelectionFactorHead"
    AAH_WHICHMUT_HEAD = "AAHWhichmutHead"
    PER_SITE_CHILD_HEAD = "PerSiteChildHead"
    STRUCTURAL_GUIDANCE_HEAD = "StructuralGuidanceHead"
    DMS_FUNCTIONAL_SCORE_HEAD = "DMSFunctionalScoreHead"
    WHICHMUT_RESIDUAL_HEAD = "WhichmutResidualHead"
    DELTA_AFFINITY_HEAD = "DeltaAffinityHead"
    DASM_FIT_RESIDUAL_HEAD = "DASMFitResidualHead"
    PER_SITE_PARENT_CHILD_HEAD = "PerSiteParentChildHead"
    PER_SITE_PREFERENCE_HEAD = "PerSitePreferenceHead"
    PER_SITE_MEAN_SELECTION_HEAD = "PerSiteMeanSelectionHead"
    QDASM_EVOL_HEAD = "QDASMEvolHead"
    HYBRID_EVOL_HEAD = "HybridEvolHead"
    SINGLE_VALUE_HEAD = "SingleValueHead"
    SINGLE_MATRIX_HEAD = "SingleMatrixHead"
    M0_HEAD = "M0Head"
    NEUTRAL_HEAD = "NeutralHead"
    POOLED_REGRESSION_HEAD = "PooledRegressionHead"
    POOLED_MULTI_LABEL_HEAD = "PooledMultiLabelHead"
    POOLED_CLASSIFICATION_HEAD = "PooledClassificationHead"
    CONTRASTIVE_BINDING_HEAD = "ContrastiveBindingHead"
    BINDING_CLASSIFICATION_HEAD = "BindingClassificationHead"
    MLM_HEAD = "MLMHead"
    SITE_MASKED_SELECTION_FACTOR_HEAD = "SiteMaskedSelectionFactorHead"

    @property
    def display_name(self) -> str:
        """Short human-readable name for this head type.

        Examples:
            >>> HeadType.SELECTION_FACTOR_HEAD.display_name
            'Selection Factor'
            >>> HeadType.QDASM_EVOL_HEAD.display_name
            'QDASM'
        """
        return _DISPLAY_NAMES[self]

    @classmethod
    def from_display_name(cls, name: str) -> "HeadType":
        """Look up a HeadType by its display name.

        Args:
            name: Display name (e.g. "DASM", "Whichmut").

        Returns:
            Corresponding HeadType member.

        Raises:
            KeyError: If no head type has the given display name.
        """
        try:
            return _DISPLAY_NAME_TO_HEAD_TYPE[name]
        except KeyError:
            available = sorted(_DISPLAY_NAME_TO_HEAD_TYPE.keys())
            raise KeyError(
                f"Unknown display name: '{name}'. Available: {available}"
            ) from None


# Maps each HeadType member to a short human-readable display name.
_DISPLAY_NAMES: dict[HeadType, str] = {
    HeadType.SELECTION_FACTOR_HEAD: "Selection Factor",
    HeadType.AAH_WHICHMUT_HEAD: "AAH Whichmut",
    HeadType.PER_SITE_CHILD_HEAD: "Per-Site Child",
    HeadType.STRUCTURAL_GUIDANCE_HEAD: "Structural Guidance",
    HeadType.DMS_FUNCTIONAL_SCORE_HEAD: "DMS Functional Score",
    HeadType.WHICHMUT_RESIDUAL_HEAD: "Whichmut Residual",
    HeadType.DELTA_AFFINITY_HEAD: "Delta Affinity",
    HeadType.DASM_FIT_RESIDUAL_HEAD: "DASMFit Residual",
    HeadType.PER_SITE_PARENT_CHILD_HEAD: "Per-Site Parent/Child",
    HeadType.PER_SITE_PREFERENCE_HEAD: "Per-Site Preference",
    HeadType.PER_SITE_MEAN_SELECTION_HEAD: "Per-Site Mean Selection",
    HeadType.QDASM_EVOL_HEAD: "QDASM",
    HeadType.HYBRID_EVOL_HEAD: "Hybrid",
    HeadType.SINGLE_VALUE_HEAD: "Single Value",
    HeadType.SINGLE_MATRIX_HEAD: "Single Matrix",
    HeadType.M0_HEAD: "M0",
    HeadType.NEUTRAL_HEAD: "Neutral",
    HeadType.POOLED_REGRESSION_HEAD: "Pooled Regression",
    HeadType.POOLED_MULTI_LABEL_HEAD: "Pooled Multilabel",
    HeadType.POOLED_CLASSIFICATION_HEAD: "Pooled Classification",
    HeadType.CONTRASTIVE_BINDING_HEAD: "Contrastive Binding",
    HeadType.BINDING_CLASSIFICATION_HEAD: "Binding Classification",
    HeadType.MLM_HEAD: "MLM",
    HeadType.SITE_MASKED_SELECTION_FACTOR_HEAD: "Site-Masked DASM",
}

_DISPLAY_NAME_TO_HEAD_TYPE: dict[str, HeadType] = {
    v: k for k, v in _DISPLAY_NAMES.items()
}


# Heads whose forward produces L×20 log selection factors suitable for the
# DASMLoss objective. The BL-optimizer hook and the OE-plot pipeline still
# dispatch in head space (they unpack ``model.prediction_heads`` directly),
# so this head-keyed filter survives alongside the loss-keyed
# ``EVOLUTIONARY_LOSS_TYPES`` set in ``dasm2/losses/types.py`` — they are
# *not* synonyms: the loss-keyed set names objectives ({DASM, QDASM,
# HYBRID}), this one names heads whose output shape is DASM-compatible.
DASM_LOSS_HEAD_TYPES: frozenset[HeadType] = frozenset(
    {
        HeadType.SELECTION_FACTOR_HEAD,
        HeadType.SITE_MASKED_SELECTION_FACTOR_HEAD,
        HeadType.SINGLE_VALUE_HEAD,
        HeadType.SINGLE_MATRIX_HEAD,
        HeadType.M0_HEAD,
        HeadType.NEUTRAL_HEAD,
    }
)
