"""Head factory using registry pattern."""

from typing import TYPE_CHECKING, Dict, Type, Union

if TYPE_CHECKING:
    from dasm2.config import (
        PerSiteParentChildHeadConfig,
        SelectionFactorHeadConfig,
    )

from .base import PredictionHead
from .per_site_parent_child import PerSiteParentChildHead
from .selection_factor import SelectionFactorHead
from .types import HeadType

#: Two heads. ``SelectionFactorHead`` is what dasm-85k and dasm-293k train;
#: ``PerSiteParentChildHead`` is ci-484k's. Every other head went with its
#: config class and loss (#4).
HeadConfig = Union[
    "SelectionFactorHeadConfig",
    "PerSiteParentChildHeadConfig",
]

# Registry pattern - Open/Closed principle compliant
# Maps HeadType enum values to head classes.
HEAD_REGISTRY: Dict[HeadType, Type[PredictionHead]] = {
    HeadType.SELECTION_FACTOR_HEAD: SelectionFactorHead,
    HeadType.PER_SITE_PARENT_CHILD_HEAD: PerSiteParentChildHead,
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
