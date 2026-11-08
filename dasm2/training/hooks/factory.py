"""Hook factory using registry pattern with flexible parameter passing.

One hook survives the fork. ``BranchLengthOptimizer``,
``DASMFitAlphaOptimizer``, ``MultihitOptimizer``, ``OverfitGuardHook`` and
``PCGradHook`` went with the losses and heads that drove them (#4), so the
gradient-combine registry is empty and its factory is gone with it.
"""

from typing import Dict, Type

from dasm2.config import HookConfig
from dasm2.training.hooks.base import TrainingHook
from dasm2.training.hooks.checkpoints import CheckpointSaveHook

HOOK_REGISTRY: Dict[str, Type[TrainingHook]] = {
    "CheckpointSaveHook": CheckpointSaveHook,
}


def create_hook(hook_config: HookConfig) -> TrainingHook:
    """Create hook from configuration using registry pattern.

    Every field except ``type`` is passed to the hook's constructor.
    """
    hook_type = hook_config.type
    if hook_type not in HOOK_REGISTRY:
        available = list(HOOK_REGISTRY.keys())
        raise ValueError(
            f"Unknown hook type: '{hook_type}'. Available hooks: {available}"
        )

    hook_class = HOOK_REGISTRY[hook_type]
    hook_params = hook_config.model_dump(exclude={"type"})
    return hook_class(**hook_params)  # type: ignore[call-arg]
