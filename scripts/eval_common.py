"""Score a checkpoint on one split of the corpus, in eval mode.

Ported from ``dasm2-experiments``' ``2026-09-21-v12-replicate-training/scripts/
eval_common.py``, essentially unchanged: it is the scoring loop Gate 1 needs,
and the pieces that make it match the trainer are easy to get subtly wrong.

**The loader is built sequentially, and that is load-bearing.** The train
bundle ``load_whichmut_train_val`` returns carries a ``ShuffleBufferSampler``;
the library's own note on the val bundle records that shuffling it ran **20x
slower** than a sequential pass on this corpus, which made evaluation 97% of
training wall clock. A full training pass is ~21 minutes sequentially and would
be most of a day shuffled. Shuffle order is also unobservable in a mean, so
there is nothing to trade away. Hence :func:`build_sequential_loader` reaches
for ``create_whichmut_dataset_from_parquet`` with ``shuffle=False`` rather than
taking the bundle it is handed.

**Censoring is resolved through the trainer's own helpers**, so a pass here
censors exactly what training censored. All three sources have to arrive: the
reversion source alone is roughly 40x the other two on Spike, and dropping any
one of them moves the loss by ~1e-3 and the loss-mask count by thousands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from lib import assert_dasm2_is_this_repo
from torch.utils.data import DataLoader

from dasm2.data.evolutionary_dataset import (
    _resolve_parquet_train_val,
    create_whichmut_dataset_from_parquet,
)
from dasm2.heads.types import HeadType
from dasm2.losses.whichmut_losses import WhichmutLoss
from dasm2.train_command import _resolve_censor_table, _resolve_reference_bases

assert_dasm2_is_this_repo()


Split = Literal["train", "val"]


def build_sequential_loader(
    config,
    parquet: Path,
    cv_split,
    batch_size: int,
    split: Split,
    num_workers: int = 0,
) -> DataLoader:
    """A deterministic single-pass loader over one split, censored as trained.

    Args:
        config: The run's ``TrainingConfig``, read for the censoring settings.
        parquet: The prepared corpus.
        cv_split: Fold spec, or ``None`` to use the Parquet's ``in_train``
            column (what the landed runs trained against).
        batch_size: Rows per batch. Only affects wall clock and the float
            reduction order, not the mean.
        split: Which half of the split to iterate.
        num_workers: DataLoader workers. Matters more than it looks: the
            per-sample whichmut transform chain runs in the worker, so at 0 it
            runs in the main process between GPU calls and the pass is
            dataloader-bound. Measured on a full training split, 0 workers took
            54.5 min against a ~21 min projection from the trainer's own eval
            timings -- the trainer uses ``num_dataloader_workers`` for exactly
            this reason. Each worker forks its own Parquet row-group cache, so
            this trades host RAM for wall clock.

    Returns:
        A ``DataLoader`` yielding every row of ``split`` exactly once, in file
        order.
    """
    paths, nickname, train_filter, val_filter, handler, max_aa_seq_len = (
        _resolve_parquet_train_val(str(parquet), cv_split=cv_split)
    )
    row_filter = train_filter if split == "train" else val_filter

    bundle = create_whichmut_dataset_from_parquet(
        parquet_paths=paths,
        segment_handler=handler,
        max_aa_seq_len=max_aa_seq_len,
        nickname=f"{nickname}_{split}_eval",
        row_filter=row_filter,
        censor_table=_resolve_censor_table(config, print),
        censor_mode=config.censor_mode,
        reference_base_idxs=_resolve_reference_bases(config, print),
        shuffle=False,
    )
    return DataLoader(
        bundle.dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=bundle.sampler,
        collate_fn=bundle.collator,
        worker_init_fn=bundle.worker_init_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def evaluate(
    module, head: HeadType, loader: DataLoader, device: str
) -> tuple[float, int]:
    """Mean whichmut loss per loss-mask position, and the position count.

    The trainer's own denominator. It sums a sum-reduced per-batch loss and
    divides by the summed ``loss_mask`` count across the whole pass
    (``Trainer._evaluate`` / ``_get_n_loss_positions``; whichmut is not a
    ``SelfNormalizingLoss``, so the mask count is what applies). Averaging
    per-batch means instead would weight a short final batch equally with a full
    one and move the fourth decimal -- which is the decimal these readouts are
    about.

    ``module.eval()`` is what makes this comparable to a validation number:
    dropout off, so the training-set loss is the model's actual fit rather than
    the noised objective it was optimized against.
    """
    loss_fn = WhichmutLoss()
    total_loss = 0.0
    total_positions = 0
    module.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            # `head_names` takes HeadType members, not their string values;
            # `Model._compute_head_outputs` reads `.value` off each one.
            predictions = module(batch, head_names=[head])[head.value]
            total_loss += float(loss_fn(predictions, batch).item())
            total_positions += int(batch["loss_mask"].sum().item())
    if total_positions == 0:
        raise ValueError(
            "the pass saw no loss-mask positions, so the mean is undefined. An "
            "empty split or a censoring setting that removed everything would "
            "both land here."
        )
    return total_loss / total_positions, total_positions


def head_key_of(module) -> HeadType:
    """The single prediction head, as the enum member the model expects.

    Raises:
        ValueError: If the checkpoint carries more than one head, where picking
            the first would silently score whichever happened to be registered
            first.
    """
    keys = list(module.prediction_heads)
    if len(keys) != 1:
        raise ValueError(
            f"checkpoint carries {len(keys)} heads ({keys}); these runs train "
            f"exactly one, so scoring would have to guess which loss to report."
        )
    return HeadType(keys[0])


def batch_size_for(config, override: int | None = None) -> int:
    """Rows per eval batch, preferring the config's validation batch size."""
    if override is not None:
        return override
    return config.val_batch_size or config.batch_size
