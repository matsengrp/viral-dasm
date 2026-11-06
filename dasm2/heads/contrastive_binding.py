"""Contrastive Ab-Ag binding head (issue #362, CALM-inspired).

Pulls the DASM2 encoder toward a joint antibody/antigen embedding space using
a symmetric multi-positive InfoNCE loss over in-batch Ab-Ag pairs. The
antibody side runs through the trainable DASM2 backbone; the antigen side
consumes pre-computed frozen ESM-2 per-residue embeddings carried in the
batch dict.

Loss variants are selected via
``ContrastiveBindingLossConfig.contrastive_loss_type``:

- ``"infonce_symmetric"`` — CALM's symmetric, multi-positive, row-normalized
  cross-entropy (default).
- ``"infonce_ab_to_ag"`` — single-direction row InfoNCE (ablation).
- ``"siglip"`` — per-pair sigmoid cross-entropy with a learnable logit bias
  (Zhai et al., 2023). No row-wise softmax; scales to tiny or huge batches.

Ab-side pooling is configurable via ``ab_pooling_type``: mean over
``ab_pool_mask`` (default, matches CALM), multi-head attention pooling, or
single-token pooling (picks the embedding at the first ``^`` separator, or
any caller-specified token index). The Ag side stays pre-pooled at
data-loading time — see ``LoadAntigenEmbedding``.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dasm2.blocks.pooling import (
    MaskedMeanPooling,
    MultiHeadScorePooling,
    SingleTokenPooling,
)
from dasm2.config import PoolingType, ProjectionMode
from dasm2.sequence.constants import AA_SEPARATOR_IDX
from dasm2.utils.validation import validate_batch_keys

from .base import PredictionHead


class ContrastiveBindingHead(PredictionHead):
    """Contrastive Ab-Ag binding head.

    Forward contract:
        ``sequence_embeddings``: Ab per-residue embeddings [B, L_ab, D_ab].
        ``batch`` must contain:
            - ``antigen_embedding``: [B, D_esm] pre-pooled ESM-2 vectors.
              Pooling happens in ``LoadAntigenEmbedding`` at data-loading
              time, not here — the Ag side is frozen, so there's nothing
              to gain from keeping per-residue tensors around.
            - ``positive_pair_matrix``: [B, B] bool (used by ``compute_loss``;
              the collator always emits it).
        Plus, depending on ``ab_pooling_type``:
            - ``"mean"`` / ``"attention"``: ``ab_pool_mask``: [B, L_ab] bool.
            - ``"single_token"``: ``aa_parents_idxs``: [B, L_ab] long.

    Returns a scaled similarity matrix ``[B, B]`` (logits = cosine / τ).
    Loss math (and the ``logit_bias`` Parameter for siglip) lives on
    ``ContrastiveBindingLoss`` after #621 Task #4.

    Inference API: use ``encode_ab`` and ``encode_ag`` to get the
    L2-normalized joint-space embeddings for an arbitrary number of
    antibodies and antigens (they don't have to share a batch). Score
    with ``z_ab @ z_ag.T``.

    Normalization: ``ContrastiveBindingLoss`` returns ``mean_loss *
    batch_size`` (sum over pairs) rather than a true mean, and reports
    ``n_loss_labels == batch_size`` (``SelfNormalizingLoss``), so the
    trainer divides by the total pair count across the gradient-
    accumulation window. Result: the effective gradient is normalized by
    the effective batch size (``batch_size × accumulation_steps``),
    matching standard CLIP/CALM training regardless of GA setting.

    The loss value scales with batch size: chance is ``ln(B)`` per pair,
    so train and validation batches of different sizes, and tasks with
    different ``batch_size``, are not on the same scale. Account for this
    when choosing ``loss_weight`` in joint training.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 256,
        antigen_input_dim: int = 1280,
        temperature_init: float = 0.2,
        ab_pooling_type: PoolingType = "mean",
        ab_pooling_hidden_dim: int = 64,
        ab_num_heads: int = 1,
        ab_head_dim: Optional[int] = None,
        ab_projection_mode: ProjectionMode = "identity",
        ab_target_token_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        if temperature_init <= 0.0:
            raise ValueError(f"temperature_init must be > 0, got {temperature_init}.")
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.antigen_input_dim = antigen_input_dim
        self.temperature_init = temperature_init

        # Ab-side pooling config — store so ``to_config`` round-trips exactly
        # what the caller asked for. ``ab_target_token_idx`` is stored *after*
        # resolution so reloads record the concrete token index in use.
        self.ab_pooling_type: PoolingType = ab_pooling_type
        self.ab_pooling_hidden_dim = ab_pooling_hidden_dim
        self.ab_num_heads = ab_num_heads
        self.ab_head_dim = ab_head_dim
        self.ab_projection_mode: ProjectionMode = ab_projection_mode

        self.ab_pooling, ab_pooled_dim, resolved_target = self._build_ab_pooling(
            ab_pooling_type=ab_pooling_type,
            input_dim=input_dim,
            ab_num_heads=ab_num_heads,
            ab_head_dim=ab_head_dim,
            ab_pooling_hidden_dim=ab_pooling_hidden_dim,
            ab_projection_mode=ab_projection_mode,
            ab_target_token_idx=ab_target_token_idx,
        )
        self.ab_target_token_idx: Optional[int] = resolved_target

        self.ab_projection = nn.Linear(ab_pooled_dim, projection_dim)
        self.ag_projection = nn.Linear(antigen_input_dim, projection_dim)
        self.log_tau = nn.Parameter(torch.tensor(math.log(temperature_init)))

        # ``logit_bias`` (Parameter for siglip, buffer otherwise) is owned by
        # ``ContrastiveBindingLoss`` after #621 Task #4.

    @staticmethod
    def _build_ab_pooling(
        *,
        ab_pooling_type: PoolingType,
        input_dim: int,
        ab_num_heads: int,
        ab_head_dim: Optional[int],
        ab_pooling_hidden_dim: int,
        ab_projection_mode: ProjectionMode,
        ab_target_token_idx: Optional[int],
    ) -> Tuple[nn.Module, int, Optional[int]]:
        """Construct the Ab pooling module and report its output dim.

        Returns ``(module, pooled_dim, resolved_target_token_idx)``. The
        resolved target token is ``None`` for non-single-token pooling.
        """
        if ab_pooling_type == "mean":
            return MaskedMeanPooling(), input_dim, None
        if ab_pooling_type == "attention":
            pool = MultiHeadScorePooling(
                input_dim=input_dim,
                num_heads=ab_num_heads,
                head_dim=ab_head_dim,
                hidden_dim=ab_pooling_hidden_dim,
                projection_mode=ab_projection_mode,
            )
            return pool, pool.output_dim, None
        if ab_pooling_type == "single_token":
            resolved = (
                ab_target_token_idx
                if ab_target_token_idx is not None
                else AA_SEPARATOR_IDX
            )
            return SingleTokenPooling(target_token_idx=resolved), input_dim, resolved
        raise ValueError(
            "ab_pooling_type must be one of 'mean', 'attention', 'single_token'; "
            f"got {ab_pooling_type!r}"
        )

    def encode_ab(
        self,
        sequence_embeddings: Tensor,
        ab_pool_mask: Optional[Tensor] = None,
        aa_parents_idxs: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode antibodies into the joint L2-normalized space.

        Primary inference-time entry point for the Ab side. Required inputs
        depend on the configured Ab pooling:

            - ``"mean"`` / ``"attention"``: ``ab_pool_mask`` required.
            - ``"single_token"``: ``aa_parents_idxs`` required.

        Args:
            sequence_embeddings: ``[M, L_ab, D_ab]`` from the DASM2 encoder.
            ab_pool_mask: ``[M, L_ab]`` bool — True at real AA positions.
            aa_parents_idxs: ``[M, L_ab]`` long — tokenized AA indices.

        Returns:
            ``[M, projection_dim]`` unit-norm joint-space embeddings.
        """
        if self.ab_pooling_type == "mean":
            if ab_pool_mask is None:
                raise ValueError(
                    "ContrastiveBindingHead.encode_ab with "
                    "ab_pooling_type='mean' requires ab_pool_mask."
                )
            pooled = self.ab_pooling(sequence_embeddings, ab_pool_mask)
        elif self.ab_pooling_type == "attention":
            if ab_pool_mask is None:
                raise ValueError(
                    "ContrastiveBindingHead.encode_ab with "
                    "ab_pooling_type='attention' requires ab_pool_mask."
                )
            # MultiHeadScorePooling returns (pooled, attention_weights);
            # the weights are only useful for interpretability.
            pooled, _ = self.ab_pooling(sequence_embeddings, ab_pool_mask)
        else:  # "single_token" — validated at build time
            if aa_parents_idxs is None:
                raise ValueError(
                    "ContrastiveBindingHead.encode_ab with "
                    "ab_pooling_type='single_token' requires aa_parents_idxs."
                )
            pooled = self.ab_pooling(sequence_embeddings, aa_parents_idxs)
        return F.normalize(self.ab_projection(pooled), dim=-1)

    def encode_ag(self, antigen_embedding: Tensor) -> Tensor:
        """Encode antigens into the joint L2-normalized space.

        Pooling has already happened in ``LoadAntigenEmbedding`` at data
        loading time; this method just projects and normalizes. Primary
        inference-time entry point for the Ag side.

        Args:
            antigen_embedding: ``[N, D_esm]`` pre-pooled ESM-2 vectors.

        Returns:
            ``[N, projection_dim]`` unit-norm joint-space embeddings.
        """
        return F.normalize(self.ag_projection(antigen_embedding), dim=-1)

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Compute logit matrix ``[B, B]`` for training.

        For inference against arbitrary Ab and Ag sets, call
        ``encode_ab`` / ``encode_ag`` separately and dot the results.
        """
        required = ["antigen_embedding"]
        if self.ab_pooling_type == "single_token":
            required.append("aa_parents_idxs")
        else:
            required.append("ab_pool_mask")
        validate_batch_keys(
            batch,
            required_keys=required,
            context="ContrastiveBindingHead.forward()",
            dataset_hint="create_contrastive_binding_dataset",
        )
        z_ab = self.encode_ab(
            sequence_embeddings,
            ab_pool_mask=batch.get("ab_pool_mask"),
            aa_parents_idxs=batch.get("aa_parents_idxs"),
        )
        z_ag = self.encode_ag(batch["antigen_embedding"])
        inv_tau = torch.exp(-self.log_tau)
        return (z_ab @ z_ag.T) * inv_tau  # [B, B]

    def to_config(self) -> "ContrastiveBindingHeadConfig":
        return ContrastiveBindingHeadConfig(
            input_dim=self.input_dim,
            projection_dim=self.projection_dim,
            antigen_input_dim=self.antigen_input_dim,
            temperature_init=self.temperature_init,
            ab_pooling_type=self.ab_pooling_type,
            ab_pooling_hidden_dim=self.ab_pooling_hidden_dim,
            ab_num_heads=self.ab_num_heads,
            ab_head_dim=self.ab_head_dim,
            ab_projection_mode=self.ab_projection_mode,
            ab_target_token_idx=self.ab_target_token_idx,
        )


# Set config type after class definition to avoid circular import
from dasm2.config import ContrastiveBindingHeadConfig  # noqa: E402

ContrastiveBindingHead.head_config_type = ContrastiveBindingHeadConfig
