"""Selection factor prediction head (#621 Task #5).

The unified head that consumes the shared ``L×20`` selection-factor tensor
from ``CoreModel``'s ``SelectionFactorProjection`` and returns it
unchanged. The matching loss class — ``DASMLoss``, ``WhichmutLoss``, or
``AAOWhichmutLoss`` — is attached via ``Task.loss`` and does all the
work.

This single head replaces the pre-#621 trio of identity-forward
``DASMEvolHead`` / ``WhichmutHead`` / ``AAOWhichmutHead`` heads. The
three legacy ``HeadConfig`` classes and ``HeadType`` enum members are
retained as YAML-compat shims (see ``dasm2/config.py`` and
``dasm2/heads/types.py``); ``HEAD_REGISTRY`` routes all three plus the
new ``SELECTION_FACTOR_HEAD`` to ``SelectionFactorHead``.
"""

from .base import SharedSelectionFactorHead


class SelectionFactorHead(SharedSelectionFactorHead):
    """Identity head over the shared selection factor tensor.

    Has no learnable parameters of its own; the upstream
    ``SelectionFactorProjection`` owns the ``Linear(D, 20)`` projection
    and the matching ``Loss`` (``DASMLoss`` / ``WhichmutLoss`` /
    ``AAOWhichmutLoss``) does the loss math. ``forward`` is the identity
    inherited from ``SharedSelectionFactorHead``: the shared SF tensor
    passes through unchanged so downstream loss code can treat the model
    output as ``predictions`` consistently with other heads.
    """

    def __init__(self) -> None:
        """Initialize selection factor head (no parameters)."""
        super().__init__()

    def to_config(self) -> "SelectionFactorHeadConfig":
        """Reconstruct ``SelectionFactorHeadConfig`` from current state."""
        return SelectionFactorHeadConfig()


# Set config type after class definition to avoid circular import
from dasm2.config import SelectionFactorHeadConfig  # noqa: E402

SelectionFactorHead.head_config_type = SelectionFactorHeadConfig
