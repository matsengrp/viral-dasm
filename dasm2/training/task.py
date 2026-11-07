"""Task abstraction for task-based training."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset, Sampler

from dasm2.config import HeadConfig, LossConfig
from dasm2.data.evolutionary_dataset import DatasetBundle
from dasm2.heads.types import HeadType
from dasm2.losses.types import LossType
from dasm2.training.hooks.base import TrainingHook


def compute_task_loss_keys(
    head_types: Sequence[HeadType], loss_types: Sequence[LossType]
) -> List[str]:
    """State-dict keys for ``model.task_losses`` from parallel task descriptors.

    Disambiguation rules, applied per task:

    * Head type appears once in ``head_types`` → bare ``head_type.value``.
    * Head type duplicated but ``(head_type, loss_type)`` is unique → append
      ``__<loss_type>`` (this is the AAH gauge-coupled pair: WhichmutLoss +
      AAOWhichmutLoss on a shared head).
    * ``(head_type, loss_type)`` itself appears multiple times → also append
      a zero-based occurrence index ``__<i>`` (joint multi-dataset training
      on the same objective).

    The resulting list is guaranteed unique. Centralising disambiguation here
    removes the per-task footgun where a caller passing only a subset of the
    task list got a different key for the same task.
    """
    if len(head_types) != len(loss_types):
        raise ValueError(
            f"head_types and loss_types must be parallel: "
            f"{len(head_types)} != {len(loss_types)}"
        )
    counts_ht: Counter[HeadType] = Counter(head_types)
    pairs = list(zip(head_types, loss_types))
    counts_pair: Counter[Tuple[HeadType, LossType]] = Counter(pairs)
    seen: Dict[Tuple[HeadType, LossType], int] = {}

    keys: List[str] = []
    for ht, lt in pairs:
        if counts_ht[ht] == 1:
            keys.append(ht.value)
            continue
        if counts_pair[(ht, lt)] == 1:
            keys.append(f"{ht.value}__{lt.value}")
            continue
        idx = seen.get((ht, lt), 0)
        seen[(ht, lt)] = idx + 1
        keys.append(f"{ht.value}__{lt.value}__{idx}")

    # Defensive: the rules above are constructively unique, so a duplicate
    # here would indicate a logic regression in this function rather than
    # a caller error.
    assert len(set(keys)) == len(keys), f"non-unique task loss keys: {keys}"
    return keys


def build_task_loss_module_dict(tasks: Sequence["Task"]) -> nn.ModuleDict:
    """Build ``model.task_losses`` from a Task list, asserting key uniqueness.

    Computes keys from the full ``tasks`` list (via :func:`compute_task_loss_keys`)
    and stamps each ``Task.loss_state_dict_key`` in place so downstream
    consumers (logs, codon-metric dicts) can read the same key without
    reconstructing the task list themselves.
    """
    loss_ids = {id(t.loss) for t in tasks}
    if len(loss_ids) != len(tasks):
        raise ValueError(
            "Each Task must own a distinct loss instance "
            "(create_loss returns fresh modules per call)."
        )
    keys = compute_task_loss_keys(
        [t.head_type for t in tasks],
        [t.loss_config.loss_type for t in tasks],
    )
    for task, key in zip(tasks, keys):
        task.loss_state_dict_key = key
    return nn.ModuleDict({key: t.loss for key, t in zip(keys, tasks)})


@dataclass
class Task:
    """A training task: data + head + loss + hooks.

    Stores either a ``DatasetBundle`` (unified pipeline) or a raw ``Dataset``
    (eager pipeline) for train and validation. Computed properties extract
    the dataset, collator, and worker_init_fn so that the trainer and hooks
    can access them uniformly without knowing whether a bundle is present.

    The Task owns the loss module and its config: the head is a
    pure forward, and ``Loss(predictions, batch)`` is what the trainer
    invokes. ``loss_config`` is required so the trainer can build a
    state-dict key naming scheme and round-trip loss metadata to checkpoints.

    Attributes:
        train_data: Training data — ``DatasetBundle`` or raw ``Dataset``.
        val_data: Validation data — same type convention.
        head_config: Configuration for the prediction head.
        loss: Loss module (``nn.Module``) — must not be shared across tasks.
        loss_config: Configuration for the loss (discriminated union).
        loss_weight: Per-task scaling applied to the loss before summation.
        batch_size: Per-device training batch size for this task's loader
            only. None falls back to the trainer's shared batch_size. Does
            not affect validation batch size.
        pre_cycle_hooks: Hooks to run before each training cycle.
        post_cycle_hooks: Hooks to run after each training cycle.
        eval_hooks: Hooks to run after each evaluation pass (their ``on_step``
            also runs after every optimizer step).
        loss_state_dict_key: Stamped by :func:`build_task_loss_module_dict`;
            ``None`` until the helper has run. Used by per-task log entries
            and codon-metric dicts so log keys and checkpoint keys agree.
    """

    train_data: Union[Dataset, DatasetBundle]
    val_data: Union[Dataset, DatasetBundle]
    head_config: HeadConfig
    loss: nn.Module
    loss_config: LossConfig
    loss_weight: float = 1.0
    batch_size: Optional[int] = None
    pre_cycle_hooks: List[TrainingHook] = field(default_factory=list)
    post_cycle_hooks: List[TrainingHook] = field(default_factory=list)
    eval_hooks: List[TrainingHook] = field(default_factory=list)
    loss_state_dict_key: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate that train_data and val_data are the same pipeline type."""
        train_is_bundle = isinstance(self.train_data, DatasetBundle)
        val_is_bundle = isinstance(self.val_data, DatasetBundle)
        if train_is_bundle != val_is_bundle:
            raise TypeError(
                f"train_data and val_data must be the same type: "
                f"train is {'DatasetBundle' if train_is_bundle else 'Dataset'}, "
                f"val is {'DatasetBundle' if val_is_bundle else 'Dataset'}"
            )

    @property
    def head_type(self) -> HeadType:
        """Head type derived from configuration."""
        return self.head_config.head_type

    @property
    def log_key(self) -> str:
        """Per-task key for log entries and codon-metric dicts.

        Returns the state-dict key stamped by
        :func:`build_task_loss_module_dict` so log keys and checkpoint keys
        agree. Falls back to ``head_type.value`` for unregistered tasks (e.g.
        test mocks that bypass the trainer constructor).
        """
        return self.loss_state_dict_key or self.head_type.value

    @property
    def train_dataset(self) -> Dataset:
        """The underlying training dataset (unwrapped from bundle if present)."""
        if isinstance(self.train_data, DatasetBundle):
            return self.train_data.dataset
        return self.train_data  # type: ignore[no-any-return]

    @property
    def val_dataset(self) -> Dataset:
        """The underlying validation dataset (unwrapped from bundle if present)."""
        if isinstance(self.val_data, DatasetBundle):
            return self.val_data.dataset
        return self.val_data  # type: ignore[no-any-return]

    @property
    def collate_fn(
        self,
    ) -> Optional[Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]]:
        """Collation function from the training bundle, or None for auto-detect."""
        if isinstance(self.train_data, DatasetBundle):
            return self.train_data.collator
        return None

    @property
    def worker_init_fn(self) -> Optional[Callable[[int], None]]:
        """Worker init function from the training bundle, or None."""
        if isinstance(self.train_data, DatasetBundle):
            return self.train_data.worker_init_fn
        return None

    @property
    def sampler(self) -> Optional[Sampler[int]]:
        """Sampler from the training bundle (e.g. ShuffleBufferSampler), or None."""
        if isinstance(self.train_data, DatasetBundle):
            return self.train_data.sampler
        return None

    @property
    def val_sampler(self) -> Optional[Sampler[int]]:
        """Sampler from the validation bundle (e.g. ShuffleBufferSampler), or None.

        Without this, token-budget batching (``max_tokens_per_batch``) has no
        row-group geometry to pack within for val and falls back to a global
        length sort that touches every row group of the underlying Parquet
        file — harmless at small scale (fits the row-group LRU cache) but a
        catastrophic cache-thrashing slowdown at large scale, since val's own
        row count doesn't bound the cost (the shared physical file's size
        does, e.g. a 4.79M-row full corpus vs a 258k-row val split).
        """
        if isinstance(self.val_data, DatasetBundle):
            return self.val_data.sampler
        return None
