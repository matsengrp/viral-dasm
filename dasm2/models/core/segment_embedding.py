"""Shared segment embedding helpers for transformer encoders."""

from typing import Dict, Optional

import torch
import torch.nn as nn


def create_segment_embedding(
    num_segment_types: Optional[int], d_model: int
) -> Optional[nn.Embedding]:
    """Create an optional segment embedding layer.

    Args:
        num_segment_types: Number of segment types (including separator and padding).
            If None, no embedding is created.
        d_model: Embedding dimension.

    Returns:
        Embedding layer or None if num_segment_types is None.
    """
    if num_segment_types is None:
        return None
    return nn.Embedding(
        num_segment_types,
        d_model,
        padding_idx=num_segment_types - 1,
    )


def apply_segment_embedding(
    segment_embedding: Optional[nn.Embedding],
    embeddings: torch.Tensor,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Add segment embeddings to token embeddings if configured.

    Args:
        segment_embedding: Optional segment embedding layer.
        embeddings: Token embeddings of shape [batch_size, seq_len, d_model].
        batch: Input batch; must contain 'layout' key when segment_embedding is not None.

    Returns:
        Embeddings with segment information added.

    Raises:
        ValueError: If segment_embedding is configured but 'layout' is missing from batch.
    """
    if segment_embedding is None:
        return embeddings
    if "layout" not in batch:
        raise ValueError(
            "Encoder is configured with segment_embedding but batch is missing "
            "'layout' key. Ensure your dataset provides layout tensors."
        )
    return embeddings + segment_embedding(batch["layout"])  # type: ignore[no-any-return]


def get_num_segment_types(
    segment_embedding: Optional[nn.Embedding],
) -> Optional[int]:
    """Extract num_segment_types from an optional segment embedding.

    Args:
        segment_embedding: Optional segment embedding layer.

    Returns:
        Number of segment types or None if no embedding exists.
    """
    if segment_embedding is None:
        return None
    return segment_embedding.num_embeddings
