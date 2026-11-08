"""Batch operation utilities for DASM2."""

from typing import Any, Dict, Union

import torch


def add_batch_dimension_to_sample(sample_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add batch dimension to dataset sample for loss computation.

    This is a utility for converting single dataset samples to batch format
    that models expect. Used when we manually convert dataset[0] samples to
    batched format for loss computation during optimization.

    Args:
        sample_dict: Dictionary from dataset[0] with tensor and scalar values

    Returns:
        Dictionary with batch dimensions added to tensors
    """
    batch_dict = {}
    for key, value in sample_dict.items():
        if torch.is_tensor(value):
            # Add batch dimension if tensor doesn't have it
            batch_dict[key] = value.unsqueeze(0) if value.dim() > 0 else value
        else:
            # Non-tensor values (scalars, etc.)
            batch_dict[key] = value

    return batch_dict


def move_batch_to_device(
    batch: Dict[str, Any], device: Union[str, torch.device]
) -> Dict[str, Any]:
    """Move all tensor values in batch to the training device.

    Uses ``non_blocking=True`` for CUDA targets so that pinned-memory
    transfers can overlap with computation.

    Args:
        batch: Dictionary containing tensors and other values
        device: Target device (e.g. "cuda", "cpu")

    Returns:
        Dictionary with all tensors moved to device
    """
    non_blocking = torch.device(device).type == "cuda"
    device_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            device_batch[key] = value.to(device, non_blocking=non_blocking)
        else:
            device_batch[key] = value
    return device_batch
