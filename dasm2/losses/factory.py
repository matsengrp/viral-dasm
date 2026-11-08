"""Factory for constructing ``Loss(nn.Module)`` instances from ``LossConfig``.

Each ``LossConfig`` discriminated-union variant maps to one concrete Loss
class. ``create_loss(loss_config)`` returns a fresh instance — losses are
never shared across tasks; the trainer constructor asserts uniqueness by
``id()`` to keep the parameter graph tractable.

The registry holds the one loss this repository trains. Every other
``LossType`` the enum names — DASM, QDASM, hybrid, the DMS and binding
families, the pooled variants, AAO whichmut — went with its module, so asking
for one raises from ``create_loss`` with the available list rather than
building a loss no config here can reach (#4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

from torch import nn

from dasm2.losses.types import LossType
from dasm2.losses.whichmut_losses import WhichmutLoss

if TYPE_CHECKING:
    from dasm2.config import LossConfig


def _build_whichmut(_cfg) -> nn.Module:  # type: ignore[no-untyped-def]
    return WhichmutLoss()


LOSS_REGISTRY: Dict[LossType, Callable[..., nn.Module]] = {
    LossType.WHICHMUT: _build_whichmut,
}


def create_loss(loss_config: "LossConfig") -> nn.Module:
    """Construct a fresh Loss module for a single Task.

    Losses are not shared across tasks. Each ``Task`` gets its own instance
    so loss-owned parameters live cleanly in the parameter graph without
    ``nn.ModuleDict`` aliasing.
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
