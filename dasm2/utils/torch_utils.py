"""PyTorch training and optimization utilities for DASM2.

This module contains functions related to model training, optimization,
and other PyTorch framework concerns. Pure tensor operations are in tensor.py."""

import random
from typing import Any, Iterable

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python's random module, NumPy, and PyTorch (CPU and CUDA).
    This ensures reproducible model initialization, data shuffling, and dropout.

    Note: This does NOT enable deterministic algorithms. Cross-GPU exact
    reproducibility requires torch.use_deterministic_algorithms(True), which
    has performance penalties and isn't needed for most training runs --
    see `set_deterministic_mode` below for the opt-in that enables it when
    it is needed.

    Args:
        seed: Integer seed value for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pin_single_thread(*_: Any) -> None:
    """Pin this process to one intra-op thread.

    Used as a worker initializer (``Pool(initializer=...)``, DataLoader
    ``worker_init_fn``) so parallel workers don't each try to use every
    core. Takes and ignores any arguments, such as DataLoader's
    ``worker_id``. Module-level so spawn workers can unpickle it.
    """
    torch.set_num_threads(1)


def set_deterministic_mode(num_threads: int = 1) -> None:
    """Pin intra-op thread count and enable deterministic algorithms.

    `set_seed` alone leaves run-to-run recovery-r drift possible even with
    fixed seeds (issue #816). The `torch.set_num_threads` call here is
    redundant inside training, which pins one thread itself
    (`TRAINING_NUM_THREADS` in `dasm2.train_command`); it's kept so this
    function is a complete, self-contained determinism switch for other
    callers. The load-bearing addition is
    `torch.use_deterministic_algorithms`, which nothing else in the
    codebase enables. Not called by `set_seed`/default training since it
    can slow training and is only needed when exact bitwise
    reproducibility across reruns matters more than speed. `warn_only=True`
    because a few ops lack a deterministic CPU kernel; failing hard on
    those would be worse than the (rare) residual non-determinism they
    contribute.

    Args:
        num_threads: Intra-op thread count to pin PyTorch to.
    """
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


def parameter_count_of_model(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def optimizer_of_name(
    optimizer_name: str, model_parameters: Iterable[torch.Tensor], **kwargs: Any
) -> torch.optim.Optimizer:
    """Create PyTorch optimizer from string name and model parameters.

    Factory function that instantiates optimizers by name with special handling
    for "SGDMomentum" which becomes SGD with momentum=0.9.

    Args:
        optimizer_name: Name of optimizer class in torch.optim (e.g., "Adam", "SGD")
        model_parameters: Iterable of model parameters to optimize
        **kwargs: Additional arguments passed to optimizer constructor

    Returns:
        Configured PyTorch optimizer instance

    Raises:
        ValueError: If optimizer_name is not found in torch.optim
    """
    if optimizer_name == "SGDMomentum":
        optimizer_name = "SGD"
        kwargs["momentum"] = 0.9
    try:
        optimizer_class = getattr(optim, optimizer_name)
        return optimizer_class(model_parameters, **kwargs)  # type: ignore[no-any-return]
    except AttributeError:
        raise ValueError(
            f"Optimizer '{optimizer_name}' is not recognized in torch.optim"
        )


def create_optimized_dataloader(
    dataset: Any,
    batch_size: int,
    shuffle: bool = True,
    collate_fn: Any = None,
    num_workers: int = 2,
) -> DataLoader:
    """Create optimized DataLoader for GPU training performance.

    Configures DataLoader with performance optimizations including persistent
    workers and pinned memory for efficient GPU data transfer.

    Args:
        dataset: PyTorch dataset to load from
        batch_size: Number of samples per batch
        shuffle: Randomize sample order between epochs
        collate_fn: Function to combine samples into batches
        num_workers: Number of subprocesses for data loading

    Returns:
        Optimized DataLoader with persistent_workers=True and automatic
        pin_memory based on CUDA availability
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )


def normalize_embeddings(
    embeddings: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """L2-normalize embeddings along the last dimension with numerical stability.

    Computes unit-length embeddings by dividing by L2 norm. Adds small epsilon
    to prevent division by zero for near-zero embeddings.

    Args:
        embeddings: Input tensor of shape [..., embed_dim]
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        L2-normalized embeddings of same shape as input, where each vector
        along the last dimension has unit length

    Example:
        >>> emb = torch.randn(4, 10, 256)
        >>> normalized = normalize_embeddings(emb)
        >>> torch.norm(normalized[0, 0], p=2).item()  # Should be ~1.0
        1.0
    """
    norms: torch.Tensor = torch.norm(embeddings, p=2, dim=-1, keepdim=True)
    result: torch.Tensor = embeddings / (norms + epsilon)
    return result
