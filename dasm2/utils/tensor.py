"""Pure tensor manipulation utilities for DASM2.

This module contains functions that operate purely on tensors without involving
model training, optimization, or other PyTorch framework concerns.
"""

import math
from typing import Any, List

import numpy as np
import torch
from torch import Tensor

from dasm2.constants import BIG_VALUE, SMALL_PROBABILITY


def zap_predictions_along_diagonal(
    predictions: torch.Tensor, aa_parents_idxs: torch.Tensor, fill: float = -BIG_VALUE
) -> torch.Tensor:
    """Set the diagonal (no amino acid change) of predictions tensor to fill value.

    Used to prevent models from predicting the parent amino acid at each position,
    forcing predictions to focus on actual mutations. Only applies to valid amino
    acid positions (0 <= aa_parents_idxs < 20); positions holding any other index
    are left unchanged.

    Args:
        predictions: Tensor of shape (batch_size, seq_len, 20) with amino acid predictions
        aa_parents_idxs: Tensor of shape (batch_size, seq_len) with parent amino acid indices
        fill: Value to use for diagonal positions (default: -BIG for log probabilities)

    Returns:
        Modified predictions tensor with diagonal positions set to fill value
    """

    device = predictions.device
    batch_size, L, _ = predictions.shape
    batch_indices = torch.arange(batch_size, device=device)[:, None].expand(-1, L)
    sequence_indices = torch.arange(L, device=device)[None, :].expand(batch_size, -1)

    # Create a mask for valid positions (standard amino acids, indices 0-19).
    # The lower bound matters: a ``< 20``-only mask admits negatives, which then
    # wrap on the advanced-indexing write below (-1 lands on slot 19), zapping a
    # different amino acid.
    #
    # Skipped rather than rejected, matching how 20 (padding) and 21 (separator)
    # are already handled: the contract is "zap the standard amino acids, leave
    # everything else alone", so a negative is out of domain, not a violated
    # assumption. Raising would also need a ``min()`` read, i.e. a host-device
    # sync (+27-30 us per call on an L40S) on a function that runs on every
    # head's forward pass; this elementwise bound is free. Nothing upstream
    # produces negatives today (``AA_PADDING_TOKEN`` is 20), so this guards a
    # latent wrap-around rather than fixing an observed one.
    valid_mask = (aa_parents_idxs >= 0) & (aa_parents_idxs < 20)

    # Only update the predictions for valid positions
    predictions[
        batch_indices[valid_mask],
        sequence_indices[valid_mask],
        aa_parents_idxs[valid_mask],
    ] = fill

    return predictions


def clamp_probability(x: Tensor) -> Tensor:
    """Clamp probability values to avoid numerical issues."""
    return torch.clamp(x, min=SMALL_PROBABILITY, max=(1.0 - SMALL_PROBABILITY))


def clamp_probability_above_only(x: Tensor) -> Tensor:
    """Clamp probability values from above only."""
    return torch.clamp(x, max=(1.0 - SMALL_PROBABILITY))


def clamp_log_probability(x: Tensor) -> Tensor:
    """Clamp log probability values to avoid numerical issues."""
    return torch.clamp(x, max=np.log(1.0 - SMALL_PROBABILITY))


def csps_to_4d(codon_csps: Tensor) -> Tensor:
    """Reshape collator-flat ``[..., L, 12]`` CSPs to ``[..., L, 3, 4]``.

    The collator emits CSPs flattened to 12 entries per codon to keep the
    on-disk layout simple; QDASM/Hybrid Q-matrix construction wants the
    ``[3 codon-positions, 4 nucleotides]`` shape. Already-4D inputs (some
    tests pass ``[..., L, 3, 4]`` directly) are returned unchanged.
    """
    if codon_csps.shape[-2:] == (3, 4):
        return codon_csps
    if codon_csps.shape[-1] != 12:
        raise ValueError(
            f"csps_to_4d expects last dim = 12 or trailing (3, 4), "
            f"got shape {tuple(codon_csps.shape)}"
        )
    return codon_csps.reshape(*codon_csps.shape[:-1], 3, 4)


def stack_heterogeneous(
    tensors: List[torch.Tensor], pad_value: float = 0.0
) -> torch.Tensor:
    """Stack tensors of different lengths by padding to maximum length.

    Takes a list of 1D or 2D tensors with variable lengths in the first dimension
    and stacks them into a single tensor by padding shorter tensors to match the
    longest tensor's first dimension.

    Args:
        tensors: List of 1D or 2D tensors with variable first dimension lengths
        pad_value: Value used for padding shorter tensors

    Returns:
        Stacked tensor with shape (num_tensors, max_length, *other_dims)

    Raises:
        ValueError: If tensors are not 1D or 2D, or if list is empty
    """
    if tensors is None or len(tensors) == 0:
        return torch.Tensor()  # Return an empty tensor if no tensors are provided

    dim = tensors[0].dim()
    if dim not in [1, 2]:
        raise ValueError("This function only supports 1D or 2D tensors.")

    max_length = max(tensor.size(0) for tensor in tensors)

    if dim == 1:
        # If 1D, simply pad the end of the tensor.
        padded_tensors = [
            torch.nn.functional.pad(
                tensor, (0, max_length - tensor.size(0)), value=pad_value
            )
            for tensor in tensors
        ]
    else:
        # If 2D, pad the end of the first dimension (rows); the argument to pad
        # is a tuple of (padding_left, padding_right, padding_top,
        # padding_bottom)
        padded_tensors = [
            torch.nn.functional.pad(
                tensor, (0, 0, 0, max_length - tensor.size(0)), value=pad_value
            )
            for tensor in tensors
        ]

    return torch.stack(padded_tensors)


def cat_heterogeneous(
    tensors: List[torch.Tensor],
    cat_dim: int = 0,
    pad_dim: int = 1,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Concatenate tensors that differ in a non-concat dimension by padding.

    Common when accumulating batched predictions from a DataLoader: each batch
    is padded to its own max sequence length, so ``torch.cat`` along the batch
    dimension fails when sequence lengths differ across batches.

    Args:
        tensors: List of tensors to concatenate. All dimensions except
            ``pad_dim`` (after padding) must already match.
        cat_dim: Dimension along which to concatenate (default 0, batch).
        pad_dim: Dimension to pad to the global maximum (default 1, seq_len).
        pad_value: Fill value for padding positions.

    Returns:
        Concatenated tensor with ``pad_dim`` equal to the global max.
    """
    if not tensors:
        raise ValueError("Cannot concatenate an empty list of tensors")

    max_pad = max(t.shape[pad_dim] for t in tensors)
    padded = []
    for t in tensors:
        deficit = max_pad - t.shape[pad_dim]
        if deficit > 0:
            # Build F.pad arg: pairs from last dim backwards to pad_dim
            ndim = t.dim()
            pad_arg = [0] * (2 * (ndim - pad_dim))
            # The pair for pad_dim is at index [-2, -1] of pad_arg
            pad_arg[-1] = deficit
            t = torch.nn.functional.pad(t, pad_arg, value=pad_value)
        padded.append(t)

    return torch.cat(padded, dim=cat_dim)


def rank_stride_pad_len(n_total: int, world_size: int) -> int:
    """Equal per-rank length that covers ``n_total`` under a rank-stride layout.

    Returns ``ceil(n_total / world_size)`` — the length every rank pads its
    local tensor to before an ``all_gather`` collective that requires equal
    shapes across ranks (e.g., ``accelerate.utils.gather``).
    """
    return math.ceil(n_total / world_size)


def rank_stride_local_n(n_total: int, world_size: int, rank: int) -> int:
    """Count of elements rank ``rank`` owns under a rank-stride layout.

    Equivalent to ``torch.arange(n_total)[rank::world_size].numel()`` but
    closed-form. Used by ``reassemble_rank_stride`` to slice each rank's
    contribution out of the gathered buffer, and by callers (e.g. the
    sharded BL re-opt) that build the stride index list.
    """
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} out of range for world_size {world_size}")
    return (n_total - rank + world_size - 1) // world_size


def pad_for_gather(local: torch.Tensor, pad_len: int) -> torch.Tensor:
    """Right-pad a rank's local tensor to ``pad_len`` along dim 0 with zeros.

    Companion to ``reassemble_rank_stride``. The pad value is irrelevant —
    reassembly indexes by per-rank original counts and never reads padded
    slots. Zero is chosen so a buggy read returns an in-dtype sentinel
    rather than NaN.
    """
    if local.ndim != 1:
        raise ValueError(
            f"pad_for_gather expects a 1-d tensor, got shape {tuple(local.shape)}"
        )
    if local.shape[0] > pad_len:
        raise ValueError(f"local length {local.shape[0]} exceeds pad_len {pad_len}")
    padded = torch.zeros(pad_len, dtype=local.dtype)
    padded[: local.shape[0]] = local
    return padded


def reassemble_rank_stride(
    gathered: torch.Tensor, n_total: int, world_size: int
) -> torch.Tensor:
    """Invert a rank-stride shard from a padded all-gather.

    Given the concatenated output of ``all_gather`` over rank-padded slices
    (each padded to ``pad_len = ceil(n_total / world_size)``), rebuild the
    original-order length-``n_total`` tensor whose rank-``r`` rows are
    ``[r, r + world_size, r + 2*world_size, ...]``.

    Args:
        gathered: Concatenated all-gather output, shape ``[world_size * pad_len]``.
        n_total: Original total row count before sharding.
        world_size: Number of ranks the gather covered.

    Returns:
        Length-``n_total`` tensor in original sample order. ``dtype`` matches
        ``gathered``.
    """
    pad_len = rank_stride_pad_len(n_total, world_size)
    expected_len = world_size * pad_len
    if gathered.shape[0] != expected_len:
        raise ValueError(
            f"gathered length {gathered.shape[0]} != world_size * pad_len "
            f"= {world_size} * {pad_len} = {expected_len}"
        )
    full = torch.empty(n_total, dtype=gathered.dtype)
    for r in range(world_size):
        r_local_n = rank_stride_local_n(n_total, world_size, r)
        if r_local_n <= 0:
            continue
        full[r::world_size] = gathered[r * pad_len : r * pad_len + r_local_n]
    return full


def tensors_to_numpy(value: Any) -> Any:
    """A tensor, or a list of tensors, as numpy; anything else unchanged.

    For crossing a process boundary by value: a pickled tensor goes through
    PyTorch's sharing strategy (one file descriptor per storage under
    ``file_descriptor``), a numpy array is copied into the pickle.
    """
    if isinstance(value, torch.Tensor):
        return value.numpy()
    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
        return [t.numpy() for t in value]
    return value


def numpy_to_tensors(value: Any) -> Any:
    """Undo :func:`tensors_to_numpy`, sharing memory with the arrays."""
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
        return [torch.from_numpy(v) for v in value]
    return value
