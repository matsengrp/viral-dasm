"""Loss functions for DASM2 training objectives.

Every training objective is a first-class ``nn.Module`` Loss class registered
in :data:`LOSS_REGISTRY`. The Task layer holds a fresh per-task instance
constructed via :func:`create_loss`.

This fork keeps the whichmut loss, which is what viral-DASM trains. The other
objectives were removed with their modules (#4); :data:`LossType` still names
them, because it also types the config union, but nothing here builds one.
"""

from dasm2.losses.factory import LOSS_REGISTRY, create_loss, register_loss
from dasm2.losses.protocols import CodonSamplingLoss, SelfNormalizingLoss
from dasm2.losses.types import (
    EVOLUTIONARY_LOSS_TYPES,
    WHICHMUT_LOSS_TYPES,
    LossType,
)
from dasm2.losses.whichmut_losses import WhichmutLoss

__all__ = [
    "WhichmutLoss",
    "CodonSamplingLoss",
    "SelfNormalizingLoss",
    "LossType",
    "EVOLUTIONARY_LOSS_TYPES",
    "WHICHMUT_LOSS_TYPES",
    "LOSS_REGISTRY",
    "create_loss",
    "register_loss",
]
