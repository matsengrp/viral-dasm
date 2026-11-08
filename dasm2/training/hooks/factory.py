"""Hook factory using registry pattern with flexible parameter passing."""

from typing import Dict, Type

from dasm2.config import HookConfig
from dasm2.training.hooks.alpha_optimizer import DASMFitAlphaOptimizer
from dasm2.training.hooks.base import GradientCombineHook, TrainingHook
from dasm2.training.hooks.branch_lengths import BranchLengthOptimizer
from dasm2.training.hooks.checkpoints import CheckpointSaveHook
from dasm2.training.hooks.multihit_optimizer import MultihitOptimizer
from dasm2.training.hooks.overfit_guard import OverfitGuardHook
from dasm2.training.hooks.pcgrad import PCGradHook

# Registry pattern - extensible without modifying this file
HOOK_REGISTRY: Dict[str, Type[TrainingHook]] = {
    "BranchLengthOptimizer": BranchLengthOptimizer,
    "CheckpointSaveHook": CheckpointSaveHook,
    "DASMFitAlphaOptimizer": DASMFitAlphaOptimizer,
    "MultihitOptimizer": MultihitOptimizer,
    "OverfitGuardHook": OverfitGuardHook,
}

# Separate registry (issue #1144): GradientCombineHook is its own ABC
# hierarchy, not a TrainingHook subclass (it has no pre_cycle/post_cycle/
# on_eval — see dasm2/training/hooks/base.py), so it gets its own typed
# registry + factory function rather than sharing HOOK_REGISTRY's
# Type[TrainingHook] annotation.
GRADIENT_COMBINE_HOOK_REGISTRY: Dict[str, Type[GradientCombineHook]] = {
    "PCGradHook": PCGradHook,
}


def create_hook(hook_config: HookConfig) -> TrainingHook:
    """Create hook from configuration using registry pattern.

    All fields in hook_config except 'type' are passed as constructor parameters.
    Hooks receive model and datasets through pre_cycle/post_cycle method parameters,
    not through constructor.

    Args:
        hook_config: Hook configuration with type and optional parameters

    Returns:
        Initialized hook instance

    Raises:
        ValueError: If hook type is unknown
        TypeError: If hook constructor receives invalid parameters
    """
    hook_type = hook_config.type

    if hook_type not in HOOK_REGISTRY:
        available = list(HOOK_REGISTRY.keys())
        raise ValueError(
            f"Unknown hook type: '{hook_type}'. Available hooks: {available}"
        )

    hook_class = HOOK_REGISTRY[hook_type]

    # Extract all fields except 'type' as constructor params
    hook_params = hook_config.model_dump(exclude={"type"})

    # Pass only configuration parameters - no model/neutral_model/datasets
    return hook_class(**hook_params)


def create_gradient_combine_hook(hook_config: HookConfig) -> GradientCombineHook:
    """Create a ``GradientCombineHook`` from configuration (issue #1144).

    Mirrors :func:`create_hook` exactly, against the separate
    ``GRADIENT_COMBINE_HOOK_REGISTRY`` — use this for
    ``TrainingConfig.combine_gradient_hooks`` entries, not ``create_hook``
    (a ``combine_gradient_hooks`` entry naming a ``TrainingHook`` type, or
    vice versa, is an unknown-type error here rather than a silent
    isinstance mismatch discovered later).

    Args:
        hook_config: Hook configuration with type and optional parameters

    Returns:
        Initialized hook instance

    Raises:
        ValueError: If hook type is unknown
        TypeError: If hook constructor receives invalid parameters
    """
    hook_type = hook_config.type

    if hook_type not in GRADIENT_COMBINE_HOOK_REGISTRY:
        available = list(GRADIENT_COMBINE_HOOK_REGISTRY.keys())
        raise ValueError(
            f"Unknown gradient-combine hook type: '{hook_type}'. Available: {available}"
        )

    hook_class = GRADIENT_COMBINE_HOOK_REGISTRY[hook_type]
    hook_params = hook_config.model_dump(exclude={"type"})
    return hook_class(**hook_params)


def register_hook(hook_name: str, hook_class: Type[TrainingHook]) -> None:
    """Register a new hook type with the factory.

    Args:
        hook_name: String identifier for the hook type
        hook_class: TrainingHook class to register
    """
    HOOK_REGISTRY[hook_name] = hook_class


def register_gradient_combine_hook(
    hook_name: str, hook_class: Type[GradientCombineHook]
) -> None:
    """Register a new ``GradientCombineHook`` type with the factory.

    Args:
        hook_name: String identifier for the hook type
        hook_class: GradientCombineHook class to register
    """
    GRADIENT_COMBINE_HOOK_REGISTRY[hook_name] = hook_class
