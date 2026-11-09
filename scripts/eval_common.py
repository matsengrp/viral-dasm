"""Score a checkpoint on one split of the corpus, in eval mode.

A mean whichmut loss over a full deterministic pass with dropout off. The
pieces that make such a pass match the trainer are easy to get subtly wrong,
so they live here rather than inside the one script that calls them.

**The loader is built sequentially, and that is load-bearing.** The bundle
``load_whichmut_train_val`` returns carries a ``ShuffleBufferSampler``, and
shuffling a full pass over a corpus this shape is far slower than reading it
in file order. Shuffle order is also unobservable in a mean, so there is
nothing traded away by dropping it:
:func:`build_sequential_loader` reaches for
``create_whichmut_dataset_from_parquet`` with ``shuffle=False`` rather than
taking the bundle it is handed.

**Censoring is resolved through the trainer's own helpers**, so a pass here
censors exactly what training censored. Applying it to one split and not the
other would make the two losses incomparable for a reason unrelated to
generalization, which is the whole point of the comparison.

GISAID: reads the corpus, returns scalars only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` is what puts this repository's root on `sys.path`, so the `dasm2`
# imports below resolve to the vendored fork rather than to whichever
# checkout the venv's editable install points at. `tests/test_import_order.py`
# reads that order off this file.
from lib import assert_dasm2_is_this_repo

assert_dasm2_is_this_repo()

from dasm2.data.evolutionary_dataset import (
    _resolve_parquet_train_val,
    create_whichmut_dataset_from_parquet,
)
from dasm2.heads.types import HeadType
from dasm2.losses.whichmut_losses import WhichmutLoss
from dasm2.train_command import _resolve_censor_table, _resolve_reference_bases

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
            column.
        batch_size: Rows per batch. Only affects wall clock and the float
            reduction order, not the mean.
        split: Which half of the split to iterate.
        num_workers: DataLoader workers. Matters more than it looks: the
            per-sample whichmut transform chain runs in the worker, so at 0 it
            runs in the main process between GPU calls and the pass becomes
            dataloader-bound rather than GPU-bound. The trainer uses
            ``num_dataloader_workers`` for exactly this reason. Each worker
            forks its own Parquet row-group cache, so this trades host RAM for
            wall clock.

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
    per-batch means instead would weight a short final batch equally with a
    full one and move the fourth decimal, which is the decimal these readouts
    are about.

    ``module.eval()`` is what makes this comparable to a validation number:
    dropout off, so the training-set loss is the model's actual fit rather
    than the noised objective it was optimized against.
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
        ValueError: If the checkpoint carries more than one head, where
            picking the first would silently score whichever happened to be
            registered first.
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
