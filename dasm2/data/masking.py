"""Batch-level BERT-style masking for MLM training (issue #884 / #851).

Masking must run as a collator wrapper, not a per-sample ``Transform``:
precomputed-mode datasets (``dasm2/data/evolutionary_dataset.py::TransformDataset``)
run every per-sample transform once at construction time and cache the
result, so a mask chosen inside a ``Transform`` would be sampled once and
reused identically every epoch forever, silently defeating masking as a
stochastic augmentation. ``MaskedTransformCollator`` instead wraps an inner
``TransformCollator`` and applies masking after every batch draw, the same
"collator wraps collator" shape as ``ContrastiveBindingCollator``
(``dasm2/data/contrastive_binding.py``) and ``PairwiseBindingCollator``
(``dasm2/data/pairwise_binding.py``).

Output batch keys follow the existing ``CoreModel`` contract
(``dasm2/models/core/base.py``): the (corrupted) token ids are written back
under ``"aa_parents_idxs"`` and a real-vs-padding ``"mask"`` is derived from
the MLM ``PAD`` token, so an ``MLMHead``/``MLMLoss`` task can run through the
same ``TransformerCoreModel`` / ``dasm2 train`` pipeline as every other head
with no encoder-side special-casing.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Tuple

import torch

from dasm2.data.collators import TensorScale, TransformCollator, split_tensor_samples
from dasm2.sequence.constants import NUM_AMINO_ACIDS
from dasm2.sequence.mlm_constants import (
    MLM_AMBIG_IDX,
    MLM_CLS_IDX,
    MLM_IGNORE_INDEX,
    MLM_MASK_IDX,
    MLM_PAD_IDX,
    MLM_SEPARATOR_IDX,
)

# Batch dict keys produced by MaskedTransformCollator.
TOKEN_IDS_KEY = "aa_parents_idxs"
MASK_KEY = "mask"
MLM_LABELS_KEY = "mlm_labels"

# Special-token indices that are never selected for masking, and whose true
# identity is never scored: CLS (whole-sequence pooling position), the ``^``
# chain separator, PAD (batch padding), and AMBIG (genuine sequencing
# ambiguity — there is no real residue identity there to reconstruct).
_UNMASKABLE_TOKEN_IDXS = frozenset(
    {MLM_CLS_IDX, MLM_SEPARATOR_IDX, MLM_PAD_IDX, MLM_AMBIG_IDX}
)


def maskable_token_mask(tokens: torch.Tensor) -> torch.Tensor:
    """Boolean mask of positions eligible for MLM masking/scoring.

    True for every position holding one of the 20 real amino-acid tokens;
    False for ``CLS``, ``^``, ``PAD``, and ``AMBIG`` (see
    ``_UNMASKABLE_TOKEN_IDXS`` above). Shared by ``MaskedTransformCollator``
    (train-time corruption selection, below) and ``MLMCoreModel``
    (eval-time masked-marginal scoring, ``dasm2/models/core/mlm_core_model.py``)
    so both agree on exactly which positions the model was trained to
    reconstruct — issue #923.
    """
    maskable = torch.ones_like(tokens, dtype=torch.bool)
    for special_idx in _UNMASKABLE_TOKEN_IDXS:
        maskable &= tokens != special_idx
    return maskable


#: Number of masked copies per forward pass in a masked-marginal scan. Bounds
#: memory for long wild types (the longest ProteinGym assay's wild type has
#: 3,423 residues); does not change the scores, only the throughput/memory
#: tradeoff.
DEFAULT_MASK_BATCH_SIZE = 16


def masked_copy_chunks(
    tokens: torch.Tensor,
    positions: torch.Tensor,
    mask_idx: int,
    chunk_size: int,
) -> Iterator[Tuple[slice, torch.Tensor]]:
    """Iterate over a masked-marginal scan's masked copies of ``tokens``, in chunks.

    Copy ``i`` is ``tokens`` with ``positions[i]`` replaced by ``mask_idx``.
    The copies are yielded ``chunk_size`` at a time, so one forward pass
    holds at most ``chunk_size`` copies rather than all ``len(positions)``.
    Shared by ``MLMCoreModel`` (``dasm2/models/core/mlm_core_model.py``) and
    ``Esm2ZeroShotSelectionModel`` (``dasm2/analysis/esm2_zeroshot.py``).

    Args:
        tokens: ``[seq_len]`` token ids of the unmasked sequence.
        positions: ``[n]`` indices into ``tokens`` to mask, one per copy.
        mask_idx: The tokenizer's mask token id.
        chunk_size: Maximum number of copies per chunk.

    Yields:
        ``(rows, batch)`` per chunk: ``rows`` is the chunk's slice of
        ``positions``, and ``batch`` is its ``[rows.stop - rows.start,
        seq_len]`` copies, row ``j`` masked at ``positions[rows][j]``.

    Callers validate ``chunk_size >= 1`` (``MLMCoreModel`` and
    ``Esm2ZeroShotSelectionModel`` check ``mask_batch_size`` at construction).
    """
    num_copies = int(positions.shape[0])
    for start in range(0, num_copies, chunk_size):
        rows = slice(start, min(start + chunk_size, num_copies))
        chunk_positions = positions[rows]
        batch = tokens.unsqueeze(0).repeat(chunk_positions.shape[0], 1)
        row_idx = torch.arange(chunk_positions.shape[0], device=tokens.device)
        batch[row_idx, chunk_positions] = mask_idx
        yield rows, batch


def masked_marginal_scan(
    tokens: torch.Tensor,
    mask_idx: int,
    separator_idx: int,
    chunk_size: int,
    score_chunk: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> List[torch.Tensor]:
    """One-site masked scan of a reference sequence, split into segments.

    Masks each maskable position of ``tokens`` in its own copy (via
    ``masked_copy_chunks``), scores each chunk of copies with
    ``score_chunk``, exponentiates, and splits the per-position rows on
    ``separator_idx``. Shared by ``MLMCoreModel`` (log-softmax log-odds) and
    ``SiteMaskedCoreModel`` (``F_j(y) - F_j(wt)``); the caller wraps it in
    ``torch.no_grad()``.

    Args:
        tokens: ``[seq_len]`` token ids of the reference sequence.
        mask_idx: The vocabulary's mask token id.
        separator_idx: The chain separator's token id.
        chunk_size: Maximum number of masked copies per ``score_chunk`` call.
        score_chunk: ``(batch_tokens, positions, ref_residues) -> [n, 20]``
            log selection factors for a chunk of ``n`` copies, row ``i``
            masked at ``positions[i]``, whose reference residue is
            ``ref_residues[i]``.

    Returns:
        One ``[segment_len, 20]`` CPU tensor of ``exp`` scores per segment.
        Special positions (``^``, ``X``, and the MLM vocabulary's ``CLS``
        and ``PAD``) get no row.

    Raises:
        ValueError: If ``tokens`` has no maskable position.
    """
    maskable_positions = torch.where(maskable_token_mask(tokens))[0]
    if maskable_positions.shape[0] == 0:
        raise ValueError(
            "No maskable (real amino-acid) positions in sequence tokens "
            f"{tokens.tolist()}"
        )
    ref_residues = tokens[maskable_positions]
    chunk_scores = [
        score_chunk(batch_tokens, maskable_positions[rows], ref_residues[rows]).cpu()
        for rows, batch_tokens in masked_copy_chunks(
            tokens, maskable_positions, mask_idx, chunk_size
        )
    ]
    selection_factors = torch.exp(torch.cat(chunk_scores))

    # Can't reuse split_predictions_by_sequences here: it splits a
    # full-length [seq_len, C] array on separator positions, but
    # `selection_factors` is already reduced to one row per maskable
    # position. Instead, count separators in the full-length tokens and
    # carry that count to each maskable position's row as its segment id.
    is_separator = tokens == separator_idx
    # Number of separators strictly before each position (a separator
    # position's own flag is subtracted back out of its inclusive count).
    exclusive_separator_count = (
        torch.cumsum(is_separator.long(), dim=0) - is_separator.long()
    )
    segment_ids = exclusive_separator_count[maskable_positions].cpu()
    _, segment_counts = torch.unique_consecutive(segment_ids, return_counts=True)
    return list(torch.split(selection_factors, segment_counts.tolist(), dim=0))


@dataclass
class MaskingConfig:
    """BERT-style masking recipe (Devlin et al. 2019 defaults).

    Of the ``mask_rate`` fraction of real (non-special) positions selected
    for masking, ``mask_token_prob`` are replaced with ``MASK``,
    ``random_token_prob`` are replaced with a uniformly random one of the 20
    canonical amino acids, and the remainder (implied
    ``1 - mask_token_prob - random_token_prob``) are left unchanged. All
    three groups are still scored in ``mlm_labels`` regardless of which
    branch they took.
    """

    mask_rate: float = 0.15
    mask_token_prob: float = 0.8
    random_token_prob: float = 0.1

    def __post_init__(self) -> None:
        for name in ("mask_rate", "mask_token_prob", "random_token_prob"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        combined = self.mask_token_prob + self.random_token_prob
        if combined > 1.0:
            raise ValueError(
                f"mask_token_prob + random_token_prob must be <= 1.0, got "
                f"{combined} (mask_token_prob={self.mask_token_prob}, "
                f"random_token_prob={self.random_token_prob})."
            )


def select_and_corrupt(
    tokens: torch.Tensor,
    maskable: torch.Tensor,
    masking_config: "MaskingConfig",
    mask_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select a random subset of ``maskable`` positions and corrupt them.

    Each maskable position is selected independently with probability
    ``masking_config.mask_rate``. Of the selected positions,
    ``mask_token_prob`` get ``mask_idx``, ``random_token_prob`` get a
    uniformly random one of the 20 canonical amino acids, and the rest keep
    their token. Shared by ``MaskedTransformCollator`` (MLM training) and
    ``SiteMaskedCoreModel``'s training forward
    (``dasm2/models/core/site_masked_core_model.py``), which uses
    ``mask_token_prob=1.0`` so a selected site never shows its resident.

    Args:
        tokens: ``[B, L]`` token ids.
        maskable: ``[B, L]`` bool, True where a position may be selected.
        masking_config: The masking recipe.
        mask_idx: The vocabulary's mask token id.

    Returns:
        ``(select_mask, corrupted)``: the ``[B, L]`` bool selection and a
        corrupted copy of ``tokens``.
    """
    select_mask = maskable & (
        torch.rand(tokens.shape, device=tokens.device) < masking_config.mask_rate
    )
    branch_draw = torch.rand(tokens.shape, device=tokens.device)
    mask_token_positions = select_mask & (branch_draw < masking_config.mask_token_prob)
    random_token_positions = (
        select_mask
        & (branch_draw >= masking_config.mask_token_prob)
        & (
            branch_draw
            < masking_config.mask_token_prob + masking_config.random_token_prob
        )
    )
    # The remaining selected positions (branch_draw >= mask_token_prob +
    # random_token_prob) are left unchanged; callers still score them.

    corrupted = tokens.clone()
    corrupted[mask_token_positions] = mask_idx
    num_random = int(random_token_positions.sum().item())
    if num_random:
        corrupted[random_token_positions] = torch.randint(
            0, NUM_AMINO_ACIDS, (num_random,), device=tokens.device
        )
    return select_mask, corrupted


@dataclass
class MaskedTransformCollator:
    """Thin wrapper over ``TransformCollator`` that adds batch-level MLM masking.

    ``TransformCollator`` handles all per-sample tensor padding/stacking
    (declared via ``tensor_specs``, which must include an entry for
    ``TOKEN_IDS_KEY`` ("aa_parents_idxs") — the un-corrupted token ids — at
    ``"codon"`` scale with ``pad_value=MLM_PAD_IDX``). The only batch-level
    concern left for this class is selecting a ``mask_rate`` fraction of real
    positions per sequence, applying the 80/10/10 BERT corruption recipe, and
    recording a single ``mlm_labels`` tensor (true residue at every selected
    position, ``-100`` everywhere else).
    """

    tensor_specs: Dict[str, Tuple[float, TensorScale]] = field(default_factory=dict)
    masking_config: MaskingConfig = field(default_factory=MaskingConfig)

    def __post_init__(self) -> None:
        if TOKEN_IDS_KEY not in self.tensor_specs:
            raise ValueError(
                f"MaskedTransformCollator.tensor_specs must include an entry "
                f"for {TOKEN_IDS_KEY!r} (the un-corrupted token ids), got "
                f"keys: {sorted(self.tensor_specs)}."
            )
        pad_value, scale = self.tensor_specs[TOKEN_IDS_KEY]
        if pad_value != MLM_PAD_IDX or scale != "codon":
            raise ValueError(
                f"MaskedTransformCollator.tensor_specs[{TOKEN_IDS_KEY!r}] must "
                f"be (MLM_PAD_IDX, 'codon') = ({MLM_PAD_IDX}, 'codon'), got "
                f"({pad_value}, {scale!r}) — the mask/label logic below "
                f"assumes tokens are padded with MLM_PAD_IDX at codon scale."
            )
        self._inner = TransformCollator(tensor_specs=self.tensor_specs)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        tensor_samples = split_tensor_samples(
            samples, self.tensor_specs, frozenset(), "MaskedTransformCollator"
        )
        batch = self._inner(tensor_samples)

        tokens = batch[TOKEN_IDS_KEY]
        select_mask, corrupted = select_and_corrupt(
            tokens, maskable_token_mask(tokens), self.masking_config, MLM_MASK_IDX
        )

        labels = torch.full_like(tokens, MLM_IGNORE_INDEX)
        labels[select_mask] = tokens[select_mask]

        batch[TOKEN_IDS_KEY] = corrupted
        batch[MASK_KEY] = corrupted != MLM_PAD_IDX
        batch[MLM_LABELS_KEY] = labels
        return batch


__all__ = [
    "TOKEN_IDS_KEY",
    "MASK_KEY",
    "MLM_LABELS_KEY",
    "MaskingConfig",
    "MaskedTransformCollator",
    "maskable_token_mask",
    "DEFAULT_MASK_BATCH_SIZE",
    "masked_copy_chunks",
    "masked_marginal_scan",
    "select_and_corrupt",
]
