"""Token-budget (length-bucketed) batch sampling.

Fixed-size batching pads every micro-batch to its longest member, so a batch
that happens to cluster many long sequences becomes a very heavy padded
forward/backward. Under DDP the ``DistributedSampler`` can hand one rank such a
cluster while the others get light batches — the load imbalance behind the
Compara v8 cycle-1 rank-SIGSEGV (issue #724): the real crash window held a
64-row batch padded to ~2475 codons on one rank with the others idle at the
gradient all-reduce.

:class:`TokenBudgetBatchSampler` forms variable-size micro-batches under a cap on
total *padded* tokens (``max(len) * num_rows`` — exactly what the collator pads
to, see ``dasm2/data/collators.py``), so a long-sequence cluster holds
proportionally fewer rows and no batch's padded forward blows up. Under
Accelerate's ``BatchSamplerShard`` (round-robin of whole batches across ranks)
uniform-budget batches also give natural per-rank token balance. The variable
micro-batch sizes are absorbed by the trainer's per-task label-count loss
normalization (issue #688), which divides by the window-total label count rather
than a fixed sample count, so the effective batch becomes token/label-defined.

The feature is opt-in via ``TrainingConfig.max_tokens_per_batch``; unset → the
fixed-size path is used unchanged.
"""

from __future__ import annotations

import logging
import random
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler, Subset

logger = logging.getLogger(__name__)

# Mega-batch sizing nominal when no max_batch_size is given (rows per chunk =
# mega_batch_factor * this). Only affects the locality of length-grouping.
_DEFAULT_NOMINAL_ROWS = 64


class TokenBudgetBatchSampler(Sampler[List[int]]):
    """Yield batches of dataset indices bounded by a padded-token budget.

    Each yielded batch ``b`` satisfies ``max(lengths[i] for i in b) * len(b) <=
    max_tokens_per_batch`` (and ``len(b) <= max_batch_size`` when set), except a
    lone sample whose own length already exceeds the budget, which is emitted as
    a singleton (the minimum possible cost for that sample).

    The batch list is rebuilt deterministically from ``seed + epoch`` whenever
    the epoch changes and cached, so :meth:`__iter__` and :meth:`__len__` always
    agree (Accelerate's ``BatchSamplerShard`` and the trainer's per-pass step
    math both read ``len``). All ranks share ``seed`` and the epoch, so every
    rank builds the identical batch list before Accelerate shards it.

    Args:
        lengths: Per-sample sequence lengths (e.g. codon counts), parallel to the
            dataset; ``len(lengths)`` must equal the dataset size.
        max_tokens_per_batch: Padded-token cap per batch (``max_len * num_rows``).
        shuffle: Shuffle samples each epoch (train). False sorts by length for a
            deterministic, low-padding order (val).
        max_batch_size: Optional hard cap on rows per batch (keeps short-sequence
            batches from ballooning). None = token budget is the only limit.
        mega_batch_factor: When shuffling, indices are processed in mega-batches
            of ``mega_batch_factor * (max_batch_size or 64)`` rows, sorted by
            length within each, to cut padding while keeping approximate
            randomness; the resulting batch order is then reshuffled. Ignored in
            the locality-preserving mode (``row_group_boundaries`` set), where
            the window size is the streaming ``buffer_size`` instead.
        seed: Base RNG seed; the per-epoch seed is ``seed + epoch``.
        drop_last: Drop the final batch if it is the leftover partial batch.
        row_group_boundaries: When set (streaming Parquet datasets), pack within
            the same row-group-locality stream a :class:`ShuffleBufferSampler`
            produces (shuffle row-group order, drain a ``buffer_size`` window),
            length-grouping + token-packing WITHIN each window only. This keeps
            the ``ParquetBackend`` LRU row-group cache warm — a global shuffle
            (the default mode) instead touches every row group per mega-batch
            and thrashes the cache (issue #724 follow-up). ``None`` → the
            global-shuffle mode, correct for in-memory datasets.
        buffer_size: The streaming window size; only used with
            ``row_group_boundaries``. Pass the source ``ShuffleBufferSampler``'s
            ``buffer_size`` so the working set matches the backend's LRU cache.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        max_tokens_per_batch: int,
        shuffle: bool = True,
        max_batch_size: Optional[int] = None,
        mega_batch_factor: int = 100,
        seed: int = 0,
        drop_last: bool = False,
        row_group_boundaries: Optional[Sequence[Tuple[int, int]]] = None,
        buffer_size: int = 8192,
    ) -> None:
        if max_tokens_per_batch <= 0:
            raise ValueError(
                f"max_tokens_per_batch must be positive, got {max_tokens_per_batch}"
            )
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be positive or None, got {max_batch_size}"
            )
        self._lengths = list(lengths)
        self._max_tokens = max_tokens_per_batch
        self._shuffle = shuffle
        self._max_batch_size = max_batch_size
        self._mega_batch_factor = max(1, mega_batch_factor)
        self._seed = seed
        self._drop_last = drop_last
        self._row_group_boundaries = (
            list(row_group_boundaries) if row_group_boundaries is not None else None
        )
        self._buffer_size = max(1, buffer_size)
        self._epoch = 0
        # Cache of (epoch, batches) so __iter__/__len__ never disagree.
        self._cached_epoch: Optional[int] = None
        self._cached_batches: List[List[int]] = []
        self._warned_oversized = False

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed the per-epoch shuffle.

        Mirrors ``torch.utils.data.DistributedSampler.set_epoch``; the trainer's
        ``_set_loader_epoch`` calls this (directly or via Accelerate's wrapper)
        once per data pass so the shuffle advances reproducibly on every rank.
        """
        self._epoch = int(epoch)

    def _greedy_pack(self, ordered: List[int]) -> List[List[int]]:
        """Pack length-ordered indices into token-budget batches."""
        batches: List[List[int]] = []
        cur: List[int] = []
        cur_max = 0
        for i in ordered:
            length = self._lengths[i]
            new_max = max(cur_max, length)
            over_tokens = new_max * (len(cur) + 1) > self._max_tokens
            over_count = (
                self._max_batch_size is not None and len(cur) + 1 > self._max_batch_size
            )
            if cur and (over_tokens or over_count):
                batches.append(cur)
                cur, cur_max = [], 0
                new_max = length
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)
        return batches

    def _build_batches(self) -> List[List[int]]:
        n = len(self._lengths)
        if n == 0:
            return []
        if self._row_group_boundaries is not None:
            batches = self._build_batches_locality(n)
        else:
            batches = self._build_batches_global(n)

        if self._drop_last and len(batches) > 1:
            batches.pop()

        self._maybe_warn_oversized()
        return batches

    def _build_batches_global(self, n: int) -> List[List[int]]:
        """Global-shuffle packing (in-memory datasets; no I/O locality cost)."""
        rng = random.Random(self._seed + self._epoch)
        indices = list(range(n))

        if self._shuffle:
            rng.shuffle(indices)
            # Length-group within mega-batches (already globally shuffled) to cut
            # padding, then pack; reshuffle the batch order so length-sorted
            # batches aren't presented in increasing-length order.
            mega = self._mega_batch_factor * (
                self._max_batch_size or _DEFAULT_NOMINAL_ROWS
            )
            batches: List[List[int]] = []
            for start in range(0, n, mega):
                chunk = indices[start : start + mega]
                chunk.sort(key=lambda i: self._lengths[i])
                batches.extend(self._greedy_pack(chunk))
            rng.shuffle(batches)
        else:
            indices.sort(key=lambda i: self._lengths[i])
            batches = self._greedy_pack(indices)
        return batches

    def _build_batches_locality(self, n: int) -> List[List[int]]:
        """Pack in the row-group-locality stream order (streaming Parquet).

        The index ORDER comes from a :class:`ShuffleBufferSampler` seeded with
        ``seed + epoch`` (so all ranks build the identical batch list before
        Accelerate's ``BatchSamplerShard`` shards it). We greedy-pack token-
        budget batches in that exact stream order — NO global shuffle and NO
        re-sort. This reproduces the access pattern the streaming path was
        designed around (consecutive picks from the few row groups resident in
        the shuffle buffer, advancing forward), so the backend's LRU row-group
        cache stays warm. The only change from the legacy fixed-size path is the
        token-budget cap: when a long sequence appears in the stream, the
        current batch closes early so its padded forward stays bounded — the
        anti-blowup property (issue #724) — without globally reordering and
        thrashing the cache. We do NOT length-group or shuffle batch order, as
        either would scatter a batch's rows across many row groups.
        """
        # Imported here (not at module top) to avoid a heavy import for the
        # in-memory path and any import cycle with the data backend module.
        from dasm2.data.parquet_backend import ShuffleBufferSampler

        # Only reached via _build_batches when boundaries are present; assert so
        # the type narrows for ShuffleBufferSampler (which requires a non-None list).
        assert self._row_group_boundaries is not None
        if self._shuffle:
            gen = torch.Generator().manual_seed(self._seed + self._epoch)
            order = list(
                ShuffleBufferSampler(
                    self._row_group_boundaries,
                    buffer_size=self._buffer_size,
                    generator=gen,
                )
            )
        else:
            # Sequential index order is already perfect locality (row groups
            # read front-to-back); no shuffle for validation.
            order = list(range(n))

        return self._greedy_pack(order)

    def _maybe_warn_oversized(self) -> None:
        if self._warned_oversized:
            return
        n_over = sum(1 for length in self._lengths if length > self._max_tokens)
        if n_over:
            logger.warning(
                "TokenBudgetBatchSampler: %d sample(s) individually exceed "
                "max_tokens_per_batch=%d and are emitted as singleton batches "
                "(unavoidable). Consider raising the budget.",
                n_over,
                self._max_tokens,
            )
        self._warned_oversized = True

    def _batches(self) -> List[List[int]]:
        if self._cached_epoch != self._epoch:
            self._cached_batches = self._build_batches()
            self._cached_epoch = self._epoch
        return self._cached_batches

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())


def sequence_lengths_for_dataset(dataset: Dataset) -> Optional[List[int]]:
    """Return cheap per-sample sequence lengths, or None if unavailable.

    Reads ``aa_parents`` (the SequenceProvider protocol; exposed by
    ``TransformDataset`` and delegated by ``ESM2DatasetWrapper``) without running
    ``__getitem__``/the transform. Recurses through ``Subset`` (e.g. cv_split),
    remapping lengths through the subset indices. Returns None for datasets that
    do not expose ``aa_parents`` (non-evolutionary heads) so the caller can fall
    back to fixed-size batching.
    """
    if isinstance(dataset, Subset):
        base = sequence_lengths_for_dataset(dataset.dataset)
        if base is None:
            return None
        return [base[i] for i in dataset.indices]
    aa_parents = getattr(dataset, "aa_parents", None)
    if aa_parents is None:
        return None
    return [len(seq) for seq in aa_parents]
