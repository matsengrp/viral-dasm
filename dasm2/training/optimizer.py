"""Optimizer creation utilities."""

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from dasm2.config import OptimizerConfig

if TYPE_CHECKING:
    from dasm2.training.compute_backend import ComputeBackend


def create_optimizer(
    model: torch.nn.Module,
    optimizer_config: OptimizerConfig,
    exclude_module: Optional[torch.nn.Module] = None,
) -> torch.optim.Optimizer:
    """Create optimizer from configuration with parameter group separation.

    Separates parameters into two groups for weight decay following the netam
    pattern (netam/framework.py:599-616). This improves training stability by
    avoiding regularization on bias terms and normalization parameters, which
    can interfere with their ability to shift activations appropriately.

    Parameter groups:
    - Decay params (dim >= 2): Apply weight_decay (weight matrices)
    - No-decay params (dim < 2): weight_decay = 0.0 (biases, layernorm)

    Args:
        model: Model to optimize (should be the unwrapped model, not DDP-wrapped)
        optimizer_config: Optimizer configuration with type and parameters
        exclude_module: Optional submodule whose parameters are left out of
            this optimizer entirely (a separate, head-scoped optimizer owns
            them instead). Matched by parameter identity
            (``id(p)``), not by name, so this is safe even if the excluded
            submodule's parameter names happen to collide with another
            submodule's. None (default) reproduces today's behavior exactly
            — every parameter in ``model`` goes into this optimizer.

    Returns:
        Initialized optimizer

    Raises:
        ValueError: If optimizer type is unsupported
    """
    optimizer_name = optimizer_config.optimizer_name

    if not hasattr(torch.optim, optimizer_name):
        raise ValueError(
            f"Unknown optimizer: '{optimizer_name}'. Check torch.optim for available optimizers."
        )

    optimizer_class = getattr(torch.optim, optimizer_name)

    excluded_param_ids = (
        {id(p) for p in exclude_module.parameters()}
        if exclude_module is not None
        else set()
    )

    # Separate parameters by dimension for weight decay
    param_dict = {
        pn: p
        for pn, p in model.named_parameters()
        if p.requires_grad and id(p) not in excluded_param_ids
    }
    decay_params = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": optimizer_config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    # Build optimizer kwargs
    optimizer_kwargs: Dict[str, Any] = {"lr": optimizer_config.learning_rate}

    # Add betas for Adam/AdamW optimizers
    if optimizer_name in ["Adam", "AdamW"] and optimizer_config.betas is not None:
        optimizer_kwargs["betas"] = optimizer_config.betas

    optimizer: torch.optim.Optimizer = optimizer_class(optim_groups, **optimizer_kwargs)
    return optimizer


def reconstruct_optimizer_config(
    backend: "ComputeBackend", optimizer: torch.optim.Optimizer, learning_rate: float
) -> OptimizerConfig:
    """Reconstruct an ``OptimizerConfig`` from a live optimizer + its stored initial LR.

    Shared by ``Trainer``'s main-optimizer and head_optimizer
    config reconstruction — both need identical unwrap/extract logic,
    differing only in which (possibly backend-wrapped) optimizer and which
    stored initial-LR value they read.

    Args:
        backend: Strips any backend-specific optimizer wrapper (e.g.
            accelerate's ``AcceleratedOptimizer``) so ``optimizer_name``
            matches a class ``create_optimizer`` recognizes.
        optimizer: The (possibly backend-wrapped) optimizer to read
            weight_decay/betas from.
        learning_rate: The *initial* LR to embed (not the current,
            possibly-decayed value from ``optimizer`` itself).

    Returns:
        OptimizerConfig matching that optimizer's original configuration.
    """
    inner_optimizer = backend.unwrap_optimizer(optimizer)
    optimizer_name = type(inner_optimizer).__name__
    # Extract weight_decay from first param group (decay params with dim >= 2).
    # With parameter group separation, param_groups[0] has the configured weight_decay
    # while param_groups[1] has weight_decay=0.0 for biases/layernorm.
    param_group = inner_optimizer.param_groups[0]
    weight_decay = param_group["weight_decay"]

    # Extract betas for Adam/AdamW
    betas = None
    if optimizer_name in ["Adam", "AdamW"]:
        betas = param_group["betas"]

    return OptimizerConfig(
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
    )
