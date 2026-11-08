"""Structural guidance loss for sequence-structure alignment.

This module implements a structural guidance objective that uses HERMES structural
embeddings to guide DASM sequence embeddings toward structurally-aware representations.

Unlike contrastive learning approaches (CLIP), this loss only maximizes similarity
between paired (sequence_i, structure_i) examples. It does not perform cross-batch
contrastive learning because positional correspondence across different antibody
sequences is biologically meaningless without explicit alignment.
- DECISION: GitHub issue #72 comment on 2025-11-12 (structural guidance vs contrastive)

"""

from typing import Dict

import torch
from torch import Tensor, nn

# Tolerance for L2-normalization validation
NORMALIZATION_TOLERANCE = 1e-6

# HERMES structural embedding dimension
HERMES_EMBED_DIM = 256


def compute_structural_guidance_loss(
    sequence_embeddings: Tensor,
    structural_embeddings: Tensor,
    mask: Tensor,
) -> Tensor:
    """Maximize site-wise similarity between paired sequence-structure examples.

    This loss function uses structural embeddings as a guide to nudge sequence
    embeddings toward structurally-aware representations. It computes site-wise
    cosine similarities for each paired (sequence_i, structure_i) example and
    maximizes their aggregated similarity.

    **Important**: This is NOT contrastive learning. It does not compare sequences
    across the batch. Each sequence_i is only compared to its paired structure_i
    at corresponding positions.

    Args:
        sequence_embeddings: [batch, seq_len, embed_dim] - L2-normalized sequence embeddings
        structural_embeddings: [batch, seq_len, embed_dim] - L2-normalized structural embeddings
                                (zero vectors at separator positions)
        mask: [batch, seq_len] - valid position indicators (True for valid, False for padding/separators)

    Returns:
        loss: Scalar loss value (negative sum of per-position cosine similarities
              across all valid positions). Magnitude scales with n_valid_positions;
              the trainer normalizes by ``loss_mask.sum()`` for GA-correct averaging.

    Raises:
        ValueError: If shapes don't match, embedding dimension is not 256, or embeddings
                   are not L2-normalized at valid positions

    Example:
        >>> seq_emb = torch.randn(4, 100, 256)
        >>> seq_emb = F.normalize(seq_emb, dim=-1)
        >>> struct_emb = torch.randn(4, 100, 256)
        >>> struct_emb = F.normalize(struct_emb, dim=-1)
        >>> mask = torch.ones(4, 100, dtype=torch.bool)
        >>> loss = compute_structural_guidance_loss(seq_emb, struct_emb, mask)
        >>> loss.item() / mask.sum().item()  # Per-position loss in [-1, 1]
        -0.456
    """
    batch_size, seq_len, embed_dim = sequence_embeddings.shape

    # Validate shape compatibility
    if sequence_embeddings.shape != structural_embeddings.shape:
        raise ValueError(
            f"Shape mismatch: sequence_embeddings has shape {sequence_embeddings.shape}, "
            f"but structural_embeddings has shape {structural_embeddings.shape}. "
            f"Both must have the same shape [batch, seq_len, embed_dim]."
        )

    # Validate embedding dimension matches HERMES
    if embed_dim != HERMES_EMBED_DIM:
        raise ValueError(
            f"Embedding dimension must be {HERMES_EMBED_DIM} to match HERMES structural embeddings. "
            f"Got sequence_embeddings with dim={embed_dim}. "
            f"Ensure encoder d_model={HERMES_EMBED_DIM} or add a projection layer."
        )

    # Validate normalization requirement (critical for cosine similarity correctness)
    seq_norms = torch.norm(sequence_embeddings, dim=-1)
    struct_norms = torch.norm(structural_embeddings, dim=-1)

    # Check sequence embeddings (should have no NaN)
    if not torch.allclose(
        seq_norms, torch.ones_like(seq_norms), atol=NORMALIZATION_TOLERANCE
    ):
        raise ValueError(
            "sequence_embeddings must be L2-normalized. "
            f"Found norms: min={seq_norms.min():.6f}, max={seq_norms.max():.6f}"
        )

    # Check structural embeddings normalization at valid positions
    # Separator positions (mask=False) contain zero vectors which are excluded from validation
    valid_struct_norms = struct_norms[mask]
    if valid_struct_norms.numel() > 0:
        if not torch.allclose(
            valid_struct_norms,
            torch.ones_like(valid_struct_norms),
            atol=NORMALIZATION_TOLERANCE,
        ):
            raise ValueError(
                "structural_embeddings must be L2-normalized at valid positions. "
                f"Found norms: min={valid_struct_norms.min():.6f}, max={valid_struct_norms.max():.6f}"
            )

    # Compute site-wise cosine similarities for paired examples
    # For normalized vectors, dot product = cosine similarity
    # Separator positions (zero vectors) naturally contribute 0 to similarity
    site_similarities = torch.sum(
        sequence_embeddings * structural_embeddings, dim=-1
    )  # [batch, seq_len]

    # Mask and sum over all valid positions across the batch.
    # Invalid positions are zeroed so .sum() only counts valid ones.
    # The trainer normalizes by total mask.sum() across the accumulation
    # window, ensuring GA correctness with variable-length sequences.
    masked_similarities = torch.where(
        mask, site_similarities, torch.zeros_like(site_similarities)
    )

    return -masked_similarities.sum()


class StructuralGuidanceLoss(nn.Module):
    """Maximize site-wise cosine similarity with HERMES structural embeddings.

    Stateless wrapper around :func:`compute_structural_guidance_loss` so the
    Task layer can hold a loss instance. Required batch keys:
    ``structure_embeddings`` and ``mask``.
    """

    def forward(
        self,
        predictions: Tensor,
        batch: Dict[str, Tensor],
    ) -> Tensor:
        if "structure_embeddings" not in batch:
            raise ValueError(
                "StructuralGuidanceLoss missing required batch key: "
                "'structure_embeddings'."
            )
        mask = batch["mask"]
        if not mask.any():
            raise ValueError(
                "Mask contains no valid positions. Cannot compute structural "
                "guidance loss on empty sequences."
            )
        return compute_structural_guidance_loss(
            sequence_embeddings=predictions,
            structural_embeddings=batch["structure_embeddings"],
            mask=mask,
        )
