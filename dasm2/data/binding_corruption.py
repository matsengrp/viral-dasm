"""Batch-level input corruption for BindingClassificationHead (issue #1221).

Regularizes the auxiliary binding-classification task by corrupting its own
inputs during training — a different mechanism than shrinking the head
(fewer parameters) or raising its dropout (noise in forward activations),
both already shown not to delay ``OverfitGuardHook``'s freeze point (issue
#1212, PR #1219). Follows the same "collator wraps collator" shape as
``dasm2/data/masking.py``'s ``MaskedTransformCollator`` (which wraps
``TransformCollator`` for the MLM task): corruption must run as a collator
wrapper, not a per-sample ``Transform``, because ``TransformDataset`` in
precomputed mode caches each sample's transform output once at construction
time (see ``dasm2/data/evolutionary_dataset.py``), so a mask chosen inside a
``Transform`` would be sampled once and reused identically every epoch,
silently defeating corruption as a stochastic augmentation.

Entity-A (antibody, discrete tokens ``aa_parents_idxs``): among positions
where the batch's ``loss_mask`` (or ``mask``, whichever ``ComputeMasks``
produced) marks a real residue —
``dasm2.data.pairwise_binding.entity_a_mask_from_batch``, the same
real-vs-padding/separator precedent ``BindingClassificationHead`` itself
reads — **and that do not already hold ``AA_AMBIG_IDX``** (a genuinely
ambiguous residue has no real identity left to hide, mirroring
``dasm2.data.masking.maskable_token_mask``'s identical exclusion of
``MLM_AMBIG_IDX`` for the same reason), a ``mask_rate`` fraction is
replaced with ``AA_AMBIG_IDX`` ("X"). This vocab
(``dasm2/sequence/constants.py``) has no dedicated ``MASK`` token the way
MLM's vocab does; ``AA_AMBIG_IDX`` is the right substitute since it
already means "no real residue identity here", unlike swapping in a
different real amino acid (which would look like genuine, if noisy, data).

Entity-B (antigen, continuous per-residue embedding
``binding_context_embedding``): among positions where
``binding_context_mask`` is currently ``True`` (real antigen residues, not
already excluded as ``<cls>``/``<eos>``/padding), a ``mask_rate`` fraction
is flipped to ``False``, excluding them from ``CrossAttention``'s softmax.
There is no discrete token to substitute for a continuous embedding, so
exclusion from attention — the same gate already used for padding — is the
natural analog of BERT-style masking here; a position already excluded
(``binding_context_mask`` already ``False``) is, by construction, never
selected again — ``context_mask & (...)`` only ever narrows the ``True``
set. The embedding values themselves are never modified. Independent
per-position Bernoulli(``mask_rate``) selection can, for a sample with few
real antigen residues, exclude every real position at once;
``CrossAttention`` (``dasm2/blocks/attention.py``) raises ``ValueError`` on
a context mask with zero valid positions, so ``entity_b_corruption_positions``
always leaves at least one real position unexcluded per sample.

Applied to the training collator only — never to the validation collator
(see ``dasm2.data.pairwise_binding.load_pairwise_binding_train_val``):
``OverfitGuardHook``'s freeze decision, and this experiment's whole
question, both depend on clean, uncorrupted validation loss.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch

from dasm2.data.collators import TensorScale
from dasm2.data.masking import TOKEN_IDS_KEY
from dasm2.data.pairwise_binding import (
    BINDING_CONTEXT_MASK_KEY,
    PairwiseBindingCollator,
    entity_a_mask_from_batch,
)
from dasm2.sequence.constants import AA_AMBIG_IDX


def entity_a_corruption_positions(
    tokens: torch.Tensor, entity_a_mask: torch.Tensor, rate: float
) -> torch.Tensor:
    """Boolean mask of entity-A positions eligible for corruption.

    True for real positions (``entity_a_mask``) that do not already hold
    ``AA_AMBIG_IDX`` (mirrors ``dasm2.data.masking.maskable_token_mask``'s
    identical exclusion of ``MLM_AMBIG_IDX`` — an already-ambiguous residue
    has no real identity left to hide), independently selected at random
    with probability ``rate``. A separate function (not inlined in
    ``__call__``) so the exclusion can be unit-tested directly against a
    controlled ``torch.rand`` draw, since substituting ``AA_AMBIG_IDX`` at
    a position that already holds it produces no visible change in the
    corrupted output either way.
    """
    eligible = entity_a_mask & (tokens != AA_AMBIG_IDX)
    return eligible & (torch.rand(tokens.shape) < rate)


def entity_b_corruption_positions(
    context_mask: torch.Tensor, rate: float
) -> torch.Tensor:
    """Boolean mask of entity-B positions to exclude from cross-attention.

    True for real positions (``context_mask``) independently selected at
    random with probability ``rate``, with one guarantee: at least one real
    position per sample is never selected, even if every real position
    happens to draw below ``rate``. Without this floor, a sample with few
    real antigen residues could have all of them excluded at once, and
    ``CrossAttention`` (``dasm2/blocks/attention.py``) raises ``ValueError``
    on a context mask with zero valid positions -- a training crash from
    corruption alone, not a modeling choice. Which real position survives is
    chosen uniformly at random (a fresh ``torch.rand`` draw, masked to real
    positions, then ``argmax``) rather than always the same positional
    index, so the floor doesn't introduce a per-position bias of its own. A
    separate function (not inlined in ``__call__``) so the floor can be
    unit-tested directly against a controlled ``torch.rand`` draw.
    """
    select_b = context_mask & (torch.rand(context_mask.shape) < rate)
    fully_excluded = (context_mask & ~select_b).sum(dim=-1) == 0
    if fully_excluded.any():
        # -inf at non-real positions so argmax always lands on a real one;
        # among real positions the iid uniform draws make every one equally
        # likely to be the max, i.e. a uniformly random survivor per row.
        survivor_priority = torch.rand(context_mask.shape).masked_fill(
            ~context_mask, float("-inf")
        )
        survivor_idx = survivor_priority.argmax(dim=-1)
        select_b[fully_excluded, survivor_idx[fully_excluded]] = False
    return select_b


@dataclass
class BindingCorruptionConfig:
    """Random input-corruption recipe for the binding-classification task.

    ``mask_rate=0.0`` (default) is a no-op: a ``CorruptedPairwiseBindingCollator``
    built with this config produces the exact same batch as a plain
    ``PairwiseBindingCollator``, so existing binding-classification configs
    are unaffected by this feature's addition.
    """

    mask_rate: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.mask_rate <= 1.0:
            raise ValueError(f"mask_rate must be in [0, 1], got {self.mask_rate}.")


@dataclass
class CorruptedPairwiseBindingCollator:
    """Wraps ``PairwiseBindingCollator``, corrupting both entities' real positions.

    ``tensor_specs`` is forwarded to the inner ``PairwiseBindingCollator``
    unchanged (entity-A padding specs; see that class's own docstring).
    """

    tensor_specs: Dict[str, Tuple[float, TensorScale]] = field(default_factory=dict)
    corruption_config: BindingCorruptionConfig = field(
        default_factory=BindingCorruptionConfig
    )

    def __post_init__(self) -> None:
        if TOKEN_IDS_KEY not in self.tensor_specs:
            raise ValueError(
                f"CorruptedPairwiseBindingCollator.tensor_specs must include "
                f"an entry for {TOKEN_IDS_KEY!r} (the un-corrupted token "
                f"ids), got keys: {sorted(self.tensor_specs)}."
            )
        pad_value, scale = self.tensor_specs[TOKEN_IDS_KEY]
        if pad_value != AA_AMBIG_IDX or scale != "codon":
            raise ValueError(
                f"CorruptedPairwiseBindingCollator.tensor_specs[{TOKEN_IDS_KEY!r}] "
                f"must be (AA_AMBIG_IDX, 'codon') = ({AA_AMBIG_IDX}, 'codon'), "
                f"got ({pad_value}, {scale!r})."
            )
        self._inner = PairwiseBindingCollator(tensor_specs=self.tensor_specs)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self._inner(samples)
        rate = self.corruption_config.mask_rate
        if rate == 0.0:
            return batch

        tokens = batch[TOKEN_IDS_KEY]
        entity_a_mask = entity_a_mask_from_batch(
            batch, context="CorruptedPairwiseBindingCollator.__call__()"
        )
        select_a = entity_a_corruption_positions(tokens, entity_a_mask, rate)
        corrupted = tokens.clone()
        corrupted[select_a] = AA_AMBIG_IDX
        batch[TOKEN_IDS_KEY] = corrupted

        context_mask = batch[BINDING_CONTEXT_MASK_KEY]
        select_b = entity_b_corruption_positions(context_mask, rate)
        batch[BINDING_CONTEXT_MASK_KEY] = context_mask & ~select_b

        return batch


__all__ = [
    "BindingCorruptionConfig",
    "CorruptedPairwiseBindingCollator",
    "entity_a_corruption_positions",
    "entity_b_corruption_positions",
]
