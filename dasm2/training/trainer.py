"""Protein evolution trainer with task-based multi-objective training.

Drives training as a step-based loop: a single ``while True``
over optimizer steps, with validation on a configurable cadence and all
checkpoint/best-model bookkeeping pinned to evaluation boundaries. The
multi-cycle wrapper is preserved for cycle-level concerns (branch-length
re-fitting, geometric-mean LR adjustment), but each cycle now runs exactly
one step loop per task (sequential mode) or one step loop total (joint mode).

Supports single-task and joint multi-task training with pluggable heads,
per-cycle hook architecture, BF16 mixed precision, gradient accumulation,
and configurable LR scheduling (ReduceLROnPlateau or warmup+polynomial decay).
"""

import contextlib
import copy
import faulthandler
import gc
import hashlib
import itertools
import math
import pickle
import time
import warnings
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    PolynomialLR,
    ReduceLROnPlateau,
    SequentialLR,
)
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    Subset,
    default_collate,
)
from tqdm import tqdm

from dasm2.artifacts import (
    ComputationCache,
    TrainingArtifact,
)
from dasm2.config import (
    CosineWarmRestartsSchedulerConfig,
    HeadConfig,
    OptimizerConfig,
    ReduceOnPlateauSchedulerConfig,
    SchedulerConfig,
    TaskConfig,
    TrainingConfig,
    WarmupPolynomialSchedulerConfig,
)
from dasm2.data.evolutionary_dataset import neutral_rates_source_of
from dasm2.data.neutral_rates_source import (
    NeutralRatesSource,
    combine_neutral_rates_sources,
)
from dasm2.data.parquet_backend import ShuffleBufferSampler
from dasm2.data.token_budget_sampler import (
    TokenBudgetBatchSampler,
    sequence_lengths_for_dataset,
)
from dasm2.heads.types import HeadType
from dasm2.losses.protocols import SelfNormalizingLoss
from dasm2.metadata import (
    BaseTrainingMetadata,
    SourceCheckpointInfo,
    SourceLaunchInfo,
    create_training_metadata_info,
)
from dasm2.models.core.base import CoreModel
from dasm2.models.core.transformer import LOG_DISTANCE_EPS
from dasm2.neutral.base import AbstractNeutralModel
from dasm2.sequence.constants import AMBIGUOUS_CODON_IDX
from dasm2.serialization.checkpoints import save_model, save_resumable_checkpoint
from dasm2.serialization.naming import (
    build_checkpoint_filename,
    run_token_from_run_id,
)
from dasm2.serialization.training_state import TrainingState, as_cpu_byte_tensor
from dasm2.training.compute_backend import ComputeBackend, compute_effective_batch_size
from dasm2.training.diagnostics import TrainerDiagnostics
from dasm2.training.head_optimizer_state import (
    HeadOptimizerState,
    assert_head_optimizer_supported_by_backend,
)
from dasm2.training.hooks.base import GradientCombineHook
from dasm2.training.logging import LoggerProtocol, NullLogger
from dasm2.training.metric_providers import ActiveMetric, build_active_metrics
from dasm2.training.optimizer import create_optimizer, reconstruct_optimizer_config
from dasm2.training.optimizer_step_metrics import OptimizerStepMetrics
from dasm2.training.task import build_task_loss_module_dict
from dasm2.training.types import HookPhase, StepBudget
from dasm2.utils.batch_operations import move_batch_to_device
from dasm2.utils.device import mem_available_bytes

if TYPE_CHECKING:
    from dasm2.training.hooks.base import TrainingHook
    from dasm2.training.task import Task


@dataclass
class StepWindowAccumulator:
    """Position-weighted rolling-window mean for train loss.

    Holds two parallel windows over the same per-step events:

    * **Step window**: flushed every optimizer step for the per-step
      ``batch`` log row. Resets immediately after flush.
    * **Eval window**: flushed every ``eval_every_steps`` for the
      ``eval_end.avg_train_loss`` field. Resets after each eval.

    The accumulator is local to ``_train_for_steps`` and does not escape
    the method. On resume the first eval window is partially filled
    (cosmetic, not snapshotted in TrainingState).
    """

    _step_loss_sum: float = 0.0
    _step_n_positions: int = 0
    _step_per_task_loss: Dict[str, float] = field(default_factory=dict)
    _step_per_task_n: Dict[str, int] = field(default_factory=dict)
    _eval_loss_sum: float = 0.0
    _eval_n_positions: int = 0
    _eval_per_task_loss: Dict[str, float] = field(default_factory=dict)
    _eval_per_task_n: Dict[str, int] = field(default_factory=dict)

    def add(
        self,
        loss_sum: float,
        n_positions: int,
        per_task: Optional[Dict[str, Tuple[float, int]]] = None,
    ) -> None:
        """Record one accumulation window's totals toward the next optimizer step.

        Called once per closed window by :meth:`Trainer._process_window` with the
        window's summed loss and label count (not per micro-batch).
        """
        self._step_loss_sum += loss_sum
        self._step_n_positions += n_positions
        self._eval_loss_sum += loss_sum
        self._eval_n_positions += n_positions
        if per_task:
            for k, (loss, n) in per_task.items():
                self._step_per_task_loss[k] = (
                    self._step_per_task_loss.get(k, 0.0) + loss
                )
                self._step_per_task_n[k] = self._step_per_task_n.get(k, 0) + n
                self._eval_per_task_loss[k] = (
                    self._eval_per_task_loss.get(k, 0.0) + loss
                )
                self._eval_per_task_n[k] = self._eval_per_task_n.get(k, 0) + n

    def flush_step_metrics(self) -> Tuple[float, Dict[str, float]]:
        """Return the per-step mean loss and per-task means, then reset the step window."""
        mean_loss = (
            self._step_loss_sum / self._step_n_positions
            if self._step_n_positions > 0
            else 0.0
        )
        per_task = {
            k: (
                self._step_per_task_loss[k] / self._step_per_task_n[k]
                if self._step_per_task_n.get(k, 0) > 0
                else 0.0
            )
            for k in self._step_per_task_loss
        }
        self._step_loss_sum = 0.0
        self._step_n_positions = 0
        self._step_per_task_loss = {}
        self._step_per_task_n = {}
        return mean_loss, per_task

    def flush_eval_window(self) -> Tuple[float, Dict[str, float]]:
        """Return the mean loss over the just-completed eval window, then reset."""
        mean_loss = (
            self._eval_loss_sum / self._eval_n_positions
            if self._eval_n_positions > 0
            else 0.0
        )
        per_task = {
            k: (
                self._eval_per_task_loss[k] / self._eval_per_task_n[k]
                if self._eval_per_task_n.get(k, 0) > 0
                else 0.0
            )
            for k in self._eval_per_task_loss
        }
        self._eval_loss_sum = 0.0
        self._eval_n_positions = 0
        self._eval_per_task_loss = {}
        self._eval_per_task_n = {}
        return mean_loss, per_task


@dataclass(frozen=True)
class _PassPosition:
    """A saved position inside a data pass, ready to replay."""

    draws: List[int]
    pass_start_rng: Dict[str, Any]
    save_time_rng: Dict[str, Any]


def _base_loader(loader: DataLoader) -> DataLoader:
    """The torch ``DataLoader`` under an Accelerate wrapper, or ``loader`` itself."""
    return getattr(loader, "base_dataloader", loader)


def _has_persistent_iterator(loader: DataLoader) -> bool:
    """Whether ``iter(loader)`` would reset a live persistent-worker iterator.

    A fresh torch iterator draws ``_base_seed`` from the global RNG before its
    sampler draws; a reset one does not (``_BaseDataLoaderIter``), so the two
    give different batch orders from the same RNG state.
    """
    return getattr(_base_loader(loader), "_iterator", None) is not None


def _create_persistent_iterators(loaders: List[DataLoader]) -> None:
    """Give each persistent-worker loader a live iterator, so its next ``iter`` resets it.

    Builds the base loader's iterator directly, because an Accelerate
    wrapper's ``iter`` does nothing until its first batch is drawn.
    """
    for loader in loaders:
        base = _base_loader(loader)
        if base.persistent_workers and not _has_persistent_iterator(base):
            iter(base)


def _sampler_chain(loader: DataLoader) -> List[Any]:
    """The loader, its Accelerate base loader, and every sampler beneath them."""
    chain: List[Any] = []
    stack: List[Any] = [loader]
    while stack:
        node = stack.pop()
        if node is None or any(node is seen for seen in chain):
            continue
        chain.append(node)
        stack.extend(
            getattr(node, attr, None)
            for attr in ("base_dataloader", "batch_sampler", "sampler")
        )
    return chain


def _order_draws_global_rng(loader: DataLoader) -> bool:
    """Whether a loader's batch order comes from the global torch RNG at pass start.

    True for an unseeded shuffling ``ShuffleBufferSampler`` or ``RandomSampler``
    anywhere in its sampler chain. False for token-budget batching (ordered by
    ``data_seed + loader_cycle``) and for a sampler Accelerate gave a generator
    it broadcasts from rank 0.
    """
    return any(
        (isinstance(node, ShuffleBufferSampler) and node.draws_global_rng)
        or (isinstance(node, RandomSampler) and node.generator is None)
        for node in _sampler_chain(loader)
    )


def _is_token_budget(loader: DataLoader) -> bool:
    """Whether a loader batches by token budget, so ranks may hold unequal batch counts."""
    return any(
        isinstance(node, TokenBudgetBatchSampler) for node in _sampler_chain(loader)
    )


def _capture_pass_start_rng(loaders: List[DataLoader]) -> Dict[str, Any]:
    """Snapshot what a pass's batch order can come from.

    That is the global streams, plus the generator Accelerate gives a
    shuffled loader under DDP (``synchronized_generator``, broadcast from
    rank 0 at each ``iter``), which no other checkpoint field holds, plus
    whether each loader's ``iter`` will reset a persistent-worker iterator
    rather than build a fresh one (see :func:`_has_persistent_iterator`).
    Call it just before the pass's ``iter`` calls.
    """
    states = TrainingState.capture_rng_states()
    states["loader_generators"] = [
        gen.get_state() if gen is not None else None
        for gen in (getattr(ldr, "synchronized_generator", None) for ldr in loaders)
    ]
    states["reset_iterators"] = [_has_persistent_iterator(ldr) for ldr in loaders]
    return states


def _restore_pass_start_rng(states: Dict[str, Any], loaders: List[DataLoader]) -> None:
    """Restore a :func:`_capture_pass_start_rng` snapshot onto freshly built loaders."""
    TrainingState.restore_rng_states(states)
    for ldr, gen_state in zip(loaders, states["loader_generators"], strict=True):
        if gen_state is not None:
            ldr.synchronized_generator.set_state(  # type: ignore[attr-defined]
                as_cpu_byte_tensor(gen_state)
            )


def _set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    """Best-effort ``set_epoch`` on the loader and its batch-sampler chain.

    Accelerate's prepared loader and PyTorch's :class:`DistributedSampler`
    expose ``set_epoch`` on the loader itself (standard fixed-size path). A
    custom ``batch_sampler`` (e.g. ``TokenBudgetBatchSampler``) is NOT reached by
    Accelerate's ``DataLoaderShard.set_epoch`` — that only forwards to ``.sampler``
    attributes, while Accelerate wraps our batch sampler as
    ``DataLoaderShard.batch_sampler.batch_sampler`` (a ``BatchSamplerShard`` with
    no ``set_epoch`` around ours). So walk the ``batch_sampler`` chain and call
    ``set_epoch`` wherever it exists. All targets are idempotent (``set_epoch``
    just records the epoch), so calling several is safe; samplers without it are
    no-ops (single-GPU, buffer samplers).
    """
    targets: List[Any] = [loader]
    node = getattr(loader, "batch_sampler", None)
    for _ in range(
        2
    ):  # DataLoaderShard.batch_sampler -> BatchSamplerShard.batch_sampler
        if node is None:
            break
        targets.append(node)
        node = getattr(node, "batch_sampler", None)
    for target in targets:
        set_epoch = getattr(target, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)


def _effective_batch_size(task_batch_size: Optional[int], default: int) -> int:
    """A task's own batch size override, or ``default`` when unset."""
    return task_batch_size if task_batch_size is not None else default


#: Fraction of ``MemAvailable`` the in-flight prefetched batches may occupy
#: before the window-covering prefetch depth is refused. Deep
#: queues of long padded batches are a bursty host-RAM footprint that has
#: OOM-killed workers, so the raise is conditional on this headroom.
_PREFETCH_RAM_FRACTION = 0.25


def _nbytes(obj: Any) -> int:
    """Total tensor bytes in a collated batch (dicts, lists and tuples walked)."""
    if isinstance(obj, torch.Tensor):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(v) for v in obj)
    return 0


def _batch_bytes_upper_bound(
    dataset: Dataset,
    collate_fn: Optional[Callable[[List[Any]], Any]],
    lengths: Optional[List[int]],
    batch_size: int,
    max_tokens_per_batch: Optional[int],
) -> int:
    """Upper-bound host bytes of one collated micro-batch.

    Collates sample 0 alone to measure bytes per padded token, then scales by
    the batch's padded-token ceiling: ``batch_size`` × the longest sequence,
    tightened to ``max_tokens_per_batch`` under token-budget batching. Datasets without
    per-sample lengths scale sample 0's bytes by ``batch_size``.
    """
    collate = collate_fn if collate_fn is not None else default_collate
    sample_bytes = _nbytes(collate([dataset[0]]))
    if not lengths:
        return batch_size * sample_bytes
    bytes_per_token = sample_bytes / max(1, lengths[0])
    # Token-budget batches are also row-capped at ``batch_size``, so the
    # fixed-size ceiling bounds both modes.
    max_tokens = batch_size * max(lengths)
    if max_tokens_per_batch is not None:
        max_tokens = min(max_tokens, max_tokens_per_batch)
    return int(max_tokens * bytes_per_token)


def _window_max_seqlen(window_microbatches: List[List[Any]]) -> int:
    """Max padded sequence length over an accumulation window's collated batches.

    Reads ``tensor.shape[1]`` (the padded length dim) across the dict-batch
    tensors; padded length, not true length, is what drives activation memory.
    """
    max_len = 0
    for batches in window_microbatches:
        for batch in batches:
            if isinstance(batch, dict):
                for value in batch.values():
                    if isinstance(value, torch.Tensor) and value.ndim >= 2:
                        max_len = max(max_len, value.shape[1])
    return max_len


def _window_microbatch_detail(
    window_microbatches: List[List[Any]],
    valid_codon_threshold: int,
    n_frac_threshold: float = 0.5,
) -> List[Dict[str, float]]:
    """Per-sequence fingerprint of the suspect rows in an accumulation window.

    For every micro-batch's DASM-style batch (one carrying ``codon_parents_idxs``)
    returns one record per sequence (row) that is either long (valid length ≥
    ``valid_codon_threshold``) or high-N (parent ambiguous-codon fraction ≥
    ``n_frac_threshold``) — the suspects for a cycle-1 slow batch.

    Each record carries the micro-batch index, row index, padded length, valid
    (non-pad) codon count, informative ``loss_mask`` count, parent N-fraction
    over the valid region, and branch length. The parent N-fraction is computed
    *within the mask* because ``codon_parents_idxs`` pads with
    ``AMBIGUOUS_CODON_IDX`` itself — counting N over the full padded row would
    conflate padding with genuine ambiguity. Rows are skipped (N-fraction
    reported as ``-1``) when no ``mask`` is present to delimit the valid region.
    """
    records: List[Dict[str, float]] = []
    for micro_idx, batches in enumerate(window_microbatches):
        for batch in batches:
            if not isinstance(batch, dict):
                continue
            parents = batch.get("codon_parents_idxs")
            if not isinstance(parents, torch.Tensor) or parents.ndim < 2:
                continue
            padded = int(parents.shape[1])
            mask = batch.get("mask")
            loss_mask = batch.get("loss_mask")
            branch_length = batch.get("branch_length")
            has_mask = isinstance(mask, torch.Tensor) and mask.ndim >= 2
            is_amb = parents == AMBIGUOUS_CODON_IDX
            for row in range(int(parents.shape[0])):
                if has_mask:
                    assert isinstance(mask, torch.Tensor)  # narrowed by has_mask
                    valid_mask = mask[row].bool()
                    valid = int(valid_mask.sum().item())
                    n_amb = int((is_amb[row] & valid_mask).sum().item())
                    n_frac = (n_amb / valid) if valid > 0 else 0.0
                else:
                    valid = padded
                    n_frac = -1.0
                if valid < valid_codon_threshold and not (n_frac >= n_frac_threshold):
                    continue
                loss_n = (
                    int(loss_mask[row].sum().item())
                    if isinstance(loss_mask, torch.Tensor) and loss_mask.ndim >= 2
                    else -1
                )
                bl_v = (
                    float(branch_length[row].item())
                    if isinstance(branch_length, torch.Tensor)
                    and branch_length.ndim >= 1
                    else -1.0
                )
                records.append(
                    {
                        "micro": float(micro_idx),
                        "row": float(row),
                        "padded": float(padded),
                        "valid": float(valid),
                        "loss": float(loss_n),
                        "parent_n_frac": n_frac,
                        "bl": bl_v,
                    }
                )
    return records


class Trainer:
    """Protein evolution trainer with task-based multi-head support."""

    # netam compatibility: retry up to 5 times on NaN gradients before failing
    _MAX_GRADIENT_RETRIES = 5

    # A window whose loss is non-finite on ANY rank is skipped (no optimizer
    # step) so a single degenerate sample on one rank cannot desync the
    # gradient all-reduce. If this many windows are skipped *consecutively*,
    # training is genuinely diverging and we fail loudly instead.
    _MAX_NONFINITE_WINDOWS = 10

    # A run can also rot slowly: intermittent skips, each followed by a finite
    # window, never trip the consecutive check. Fail when more than
    # _MAX_SKIPS_PER_RATE_WINDOW (1%) of the last _SKIP_RATE_WINDOW windows were
    # skipped. Skips are rank-synchronized, so every rank raises
    # together.
    _SKIP_RATE_WINDOW = 1000
    _MAX_SKIPS_PER_RATE_WINDOW = 10

    # Warn when task datasets differ in size by more than this ratio (50%)
    _DATASET_SIZE_IMBALANCE_RATIO = 1.5

    # Warn when per-task losses differ by more than this factor (10x)
    _LOSS_IMBALANCE_THRESHOLD = 10

    def __init__(
        self,
        model: CoreModel,
        neutral_model: Optional[AbstractNeutralModel],
        tasks: List["Task"],
        backend: ComputeBackend,
        batch_size: int = 32,
        val_batch_size: int = TrainingConfig.model_fields["val_batch_size"].default,
        optimizer: Optional[torch.optim.Optimizer] = None,
        min_learning_rate: float = 1e-6,
        cache_dir: Optional[str] = ".dasm2_cache",
        config_name: Optional[str] = None,
        logger: Optional[LoggerProtocol] = None,
        gradient_clip_val: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        monitor_task: Optional[str] = None,
        source_checkpoint: Optional[SourceCheckpointInfo] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
        use_mixed_precision: bool = False,
        gradient_accumulation_steps: int = 1,
        step_budget: Optional[StepBudget] = None,
        num_dataloader_workers: int = 0,
        max_tokens_per_batch: Optional[int] = None,
        dataloader_prefetch_factor: int = 4,
        dataloader_pin_memory: bool = True,
        data_seed: int = 0,
        source_task_configs: Optional[List[TaskConfig]] = None,
        source_launch_info: Optional[SourceLaunchInfo] = None,
        source_neutral_rates_source: Optional[NeutralRatesSource] = None,
        combine_gradient_hooks: Optional[List[GradientCombineHook]] = None,
        head_optimizer: Optional[torch.optim.Optimizer] = None,
        head_optimizer_module: Optional[torch.nn.Module] = None,
        head_optimizer_head_type: Optional[HeadType] = None,
    ) -> None:
        """Initialize trainer with task-based configuration.

        Args:
            model: CoreModel with pluggable heads.
            neutral_model: Domain knowledge for computing rates/CSPs.
            tasks: List of training tasks (single-task training uses [task]).
            backend: Compute backend for device management and distributed training.
            batch_size: Per-device batch size for training only. Together
                with ``gradient_accumulation_steps`` and world size this
                sets the effective batch size that drives gradient dynamics.
            val_batch_size: Per-device batch size for validation only (no
                backward pass, no accumulation), so this typically admits a
                larger value than ``batch_size`` on the same device. The
                default reads from ``TrainingConfig.val_batch_size`` so the
                two cannot drift.
            optimizer: Optimizer for training.
            min_learning_rate: Minimum learning rate for early stopping (default: 1e-6).
                Training stops when LR falls below this threshold (plateau scheduler only).
            cache_dir: Directory for cached artifacts. If None, caching is disabled.
                Defaults to ".dasm2_cache".
            config_name: Optional name of config used to create this trainer.
            logger: Optional training logger. If None, uses NullLogger.
            gradient_clip_val: Maximum gradient norm for clipping. If None, no clipping.
            early_stopping_patience: Number of *evaluations* without validation loss
                improvement before stopping. If None, only LR-based early stopping is used.
            early_stopping_min_delta: Minimum validation loss improvement required to
                reset the patience counter. An improvement counts when
                ``val_loss < best_val_loss - min_delta`` (strict less-than).
            monitor_task: Log key (``Task.log_key``) of the task whose own
                validation loss drives early stopping, best-model tracking and the
                reduce-on-plateau scheduler. Its ``loss_weight`` is divided out.
                None monitors the sum of every task's weighted validation loss.
            source_checkpoint: Optional provenance info for fine-tuned models.
            scheduler_config: Scheduler configuration. If None, defaults to ReduceLROnPlateau.
            use_mixed_precision: Enable BF16 mixed precision for forward pass.
            gradient_accumulation_steps: Number of micro-batches to accumulate gradients
                over before performing an optimizer step. NaN gradients at window
                boundaries trigger a replay-and-retry loop.
            step_budget: Per-invocation step budget (max_train_steps, eval cadence,
                eval cap). Required when scheduler is ``warmup_polynomial`` because
                the polynomial schedule spans exactly one ``max_train_steps`` cycle.
            num_dataloader_workers: Worker process count for DataLoader.
            max_tokens_per_batch: If set, use token-budget batching (see
                :class:`dasm2.data.token_budget_sampler.TokenBudgetBatchSampler`)
                instead of fixed-size batches. None keeps fixed-size batching.
            dataloader_prefetch_factor: Per-worker DataLoader prefetch depth
                (only used when ``num_dataloader_workers > 0``). Lower on
                memory-constrained nodes.
            dataloader_pin_memory: Whether to pin DataLoader host memory
                (CUDA only). Disable under host-RAM pressure.
            data_seed: Base seed of the token-budget batch order (the run's
                ``--seed``); carried in :class:`TrainingState` so a resumed
                run replays the same order.
            source_task_configs: Task descriptors to emit from
                :meth:`export_metadata` when ``tasks`` is empty.
                ``loss_config`` and ``loss_weight`` are per-task facts that no
                amount of inspecting the model can recover, so a Trainer built
                without live Tasks — ``load_checkpoint`` is the canonical case
                — has to be handed them explicitly or it re-saves a checkpoint
                with an empty ``tasks: [...]`` block. Ignored when ``tasks``
                is non-empty, which is then the single source of truth.
            source_launch_info: World size and backend precision mode to emit
                from :meth:`export_metadata` when this Trainer's ``backend``
                is not the one the run actually trained under.
                Same carry-through role as ``source_task_configs``: a
                ``load_checkpoint`` Trainer is always handed a fabricated
                ``SingleGPUBackend`` reporting world size ``1`` and
                ``"no"``, so without this a load → save round-trip would
                rewrite a world-size-4 bf16 checkpoint's metadata to those
                literals. Leave None whenever the backend is live —
                ``create_trainer`` and ``resume_training`` both are, so
                neither sets it.
            source_neutral_rates_source: Where the source checkpoint's
                training rates came from, for a Trainer built
                without live Tasks. Same carry-through role as
                ``source_task_configs``: with no training datasets to inspect,
                :meth:`export_metadata` re-emits this. Ignored when ``tasks``
                is non-empty. Also what evaluation compares against the
                evaluation file.
            combine_gradient_hooks: Top-level (not per-task) hooks controlling
                how multiple tasks' gradients on shared parameters are
                combined before the optimizer step. None/empty
                (the default) reproduces today's plain-sum combination
                exactly — every existing config exercises this path. At most
                one entry is supported; two combination strategies cannot
                both apply to the same gradient. Not supported under
                multi-process backends (see the ``backend.num_processes``
                check below): ``torch.autograd.grad`` bypasses the DDP
                reducer that ``self.backend.backward()`` normally drives.
            head_optimizer: Optional second optimizer scoped to exactly one
                head's own parameters, built by the caller
                with ``create_optimizer(head_module, head_optimizer_config,
                ...)`` over a ``model`` whose main optimizer already
                excludes that same head's parameters via
                ``create_optimizer(..., exclude_module=head_module)``. Every
                other parameter, including the shared encoder, stays on the
                single main ``optimizer`` above — this never touches shared
                weights. None (default) is fully inert: every existing
                config trains exactly as before. Not supported under
                multi-process backends (see the ``backend.num_processes``
                check below). ``head_optimizer`` is never attached to
                ``self.scheduler`` — instead, when ``self.scheduler_config``
                is ``ReduceOnPlateauSchedulerConfig``, it gets its own
                ``self.head_scheduler`` (see :meth:`_create_head_scheduler`),
                a separate ``ReduceLROnPlateau`` with the same
                factor/eval_patience, stepped on
                ``head_optimizer_head_type``'s own raw validation loss
                (:meth:`_step_head_scheduler`) rather than the combined loss
                ``self.scheduler`` watches — the two plateau schedules and
                the two optimizers' LRs never mix. Under a per-step scheduler
                (``WarmupPolynomialSchedulerConfig``,
                ``CosineWarmRestartsSchedulerConfig``) there is no per-eval
                plateau signal to mirror, so ``head_scheduler`` is None and
                ``head_optimizer``'s LR stays flat within a cycle. Either
                way, its LR also moves once at every cycle boundary, via the
                same reset ``_reset_optimizer_with_adjusted_lr`` applies to
                the main optimizer, seeded from its own initial LR.
            head_optimizer_module: The same head module ``head_optimizer``
                was built over — kept so cycle-boundary LR resets and
                checkpoint round-tripping can rebuild this optimizer the
                same way ``unwrapped_model`` lets the main optimizer rebuild
                itself. Required iff ``head_optimizer`` is set.
            head_optimizer_head_type: Which head ``head_optimizer_module``
                is, purely for :meth:`export_metadata` to round-trip
                ``TrainingConfig.head_optimizer_head_type``. Required iff
                ``head_optimizer`` is set.
        """
        self.backend = backend
        self.combine_gradient_hooks: List[GradientCombineHook] = (
            combine_gradient_hooks or []
        )
        if len(self.combine_gradient_hooks) > 1:
            raise ValueError(
                "At most one combine_gradient_hooks entry is supported "
                f"(got {len(self.combine_gradient_hooks)}): two gradient-"
                "combination strategies cannot both apply to the same "
                "shared-parameter gradient in one micro-batch."
            )
        if self.combine_gradient_hooks and backend.num_processes > 1:
            raise ValueError(
                "combine_gradient_hooks is not supported under multi-process "
                f"backends (backend.num_processes={backend.num_processes}): "
                "torch.autograd.grad bypasses the DDP gradient-sync reducer "
                "that self.backend.backward() normally drives. Run single-"
                "process (SingleGPUBackend) until DDP support is added."
            )
        if self.combine_gradient_hooks and backend.mixed_precision != "no":
            raise ValueError(
                "combine_gradient_hooks is not supported with mixed precision "
                f"(backend.mixed_precision={backend.mixed_precision!r}): "
                "torch.autograd.grad bypasses the loss-scaling/unscaling that "
                "self.backend.backward() normally drives under AccelerateBackend, "
                "which would silently feed scaled gradients into PCGrad's cosine "
                "similarity. Run without --mixed_precision until this is supported."
            )

        if not (
            (head_optimizer is None)
            == (head_optimizer_module is None)
            == (head_optimizer_head_type is None)
        ):
            raise ValueError(
                "head_optimizer, head_optimizer_module, and "
                "head_optimizer_head_type must all be provided or all be "
                f"None (head_optimizer={head_optimizer!r}, "
                f"head_optimizer_module={head_optimizer_module!r}, "
                f"head_optimizer_head_type={head_optimizer_head_type!r})."
            )
        if head_optimizer is not None:
            assert_head_optimizer_supported_by_backend(backend)
        # Consolidates head_optimizer/head_optimizer_module/
        # head_optimizer_head_type/head_scheduler's state and behavior in
        # one object (DESIGN.md D5) — see
        # HeadOptimizerState's own docstring. None reproduces today's
        # single-optimizer behavior exactly; the presence checks below
        # (``self.head_optimizer_state is not None``) already establish
        # head_optimizer_module/head_optimizer_head_type are non-None too.
        self.head_optimizer_state: Optional[HeadOptimizerState] = (
            HeadOptimizerState(
                optimizer=head_optimizer,
                module=head_optimizer_module,  # type: ignore[arg-type]
                head_type=head_optimizer_head_type,  # type: ignore[arg-type]
            )
            if head_optimizer is not None
            else None
        )

        self.model = model
        self.neutral_model = neutral_model
        self.tasks = tasks
        self.source_task_configs = source_task_configs
        self.source_launch_info = source_launch_info
        self.source_neutral_rates_source = source_neutral_rates_source
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.optimizer = optimizer
        self.min_learning_rate = min_learning_rate
        self.config_name = config_name
        self.logger: LoggerProtocol = logger if logger is not None else NullLogger()

        # Step-based counters. ``_global_step`` and ``_data_passes`` persist
        # across cycles and are snapshotted in TrainingState. ``_cycle_step``,
        # ``_cycle_task_index``, and ``_eval_count`` reset at the start of
        # each cycle (and ``_cycle_step`` resets per ``_train_for_steps``
        # invocation). ``_loader_cycle`` is the authoritative DDP shuffle seed
        # and persists across cycles so freshly-built loaders pick up where
        # the prior shuffle left off.
        self._global_step: int = 0
        self._cycle_step: int = 0
        self._cycle_task_index: int = 0
        self._eval_count: int = 0
        self._data_passes: float = 0.0
        self._loader_cycle: int = 0
        self._current_cycle: int = 0
        # Position inside the current data pass, snapshotted in
        # TrainingState: micro-batches each live train iterator has yielded,
        # and the global RNG state just before those iterators were built.
        # ``_resume_pass_position`` holds the restored pair until the first
        # ``_train_for_steps`` after a resume consumes it.
        self._train_iter_draws: List[int] = []
        self._pass_start_rng_states: Optional[Dict[str, Any]] = None
        self._resume_pass_position: Optional[
            Tuple[Optional[List[int]], Optional[Dict[str, Any]]]
        ] = None

        self.gradient_clip_val = gradient_clip_val
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.monitor_task = monitor_task
        self.source_checkpoint = source_checkpoint

        # Performance features
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        self.num_dataloader_workers = num_dataloader_workers
        self.max_tokens_per_batch = max_tokens_per_batch
        self.dataloader_prefetch_factor = dataloader_prefetch_factor
        self.dataloader_pin_memory = dataloader_pin_memory
        self._data_seed = data_seed
        self._is_cuda_device = torch.device(self.device).type == "cuda"
        if use_mixed_precision and not self._is_cuda_device:
            raise ValueError(
                f"Mixed precision requires CUDA device, got '{self.device}'. "
                "Set use_mixed_precision=False for non-CUDA devices."
            )
        self._autocast_ctx: contextlib.AbstractContextManager = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_mixed_precision
            else contextlib.nullcontext()
        )

        # Scheduler config (stored for _create_scheduler and _reset_optimizer)
        self.scheduler_config = scheduler_config or ReduceOnPlateauSchedulerConfig()
        self.step_budget = step_budget
        # Polynomial scheduler reads ``total_training_steps``; under step-based
        # training this is the per-cycle ``max_train_steps``.
        self.total_training_steps: Optional[int] = (
            step_budget.max_train_steps if step_budget is not None else None
        )

        # Store initial learning rate for netam's weighted geometric mean LR adjustment
        self.learning_rate = (
            self._get_current_learning_rate() if optimizer is not None else 0.0
        )
        # head_optimizer's own initial LR is captured by
        # HeadOptimizerState itself at construction; see the head_learning_rate
        # property below.

        # Checkpoint-specific identifier for cache disambiguation.
        self.checkpoint_id: Optional[str] = None

        self.cache: Optional[ComputationCache] = (
            ComputationCache(cache_dir=cache_dir) if cache_dir is not None else None
        )

        # Best model tracking (reset per ``_train_for_steps`` invocation —
        # per-task in sequential mode, per-cycle in joint mode). Each
        # invocation is self-contained: reset at start, restore at end.
        # Post-cycle hooks therefore see the best model from the just-completed
        # invocation. Matches netam's behavior (framework.py:756-786).
        self.best_val_loss: float = float("inf")
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.evals_without_improvement: int = 0

        # Scheduler creation (dispatches on scheduler_config.scheduler_type)
        self.scheduler = self._create_scheduler()
        # Per-step schedulers (warmup_polynomial, cosine_warm_restarts) step
        # after each optimizer.step()
        # Per-eval schedulers (reduce_on_plateau) step after each evaluation with val_loss
        self._is_per_step_scheduler = isinstance(
            self.scheduler_config,
            (WarmupPolynomialSchedulerConfig, CosineWarmRestartsSchedulerConfig),
        )
        # head_optimizer's own plateau scheduler — see
        # _create_head_scheduler for when this is/isn't built.
        if self.head_optimizer_state is not None:
            self.head_optimizer_state.replace_scheduler(self._create_head_scheduler())

        # Track if we've warned about loss imbalance (warn only once per training run)
        self._has_warned_loss_imbalance = False

        # Fail-fast: validate distance encoding consistency between model and datasets
        self._validate_distance_encoding_consistency()

        # Initialize distance bin boundaries from training data if needed
        self._initialize_distance_bins()

        # Per-eval-window optimizer-step metrics: updated
        # from ``_clip_and_step`` each optimizer step, flushed at every
        # eval boundary by ``_run_eval_boundary``. Loss/label accumulation is
        # handled by the ``StepWindowAccumulator`` threaded through the step
        # loop (no per-trainer window buffers).
        self._optimizer_step_metrics = OptimizerStepMetrics()

        # Count of consecutive windows skipped for a non-finite loss (see
        # ``_MAX_NONFINITE_WINDOWS`` and ``_process_window``). Reset on any
        # successfully-stepped window.
        self._consecutive_nonfinite_windows = 0
        # Skip flags of the most recent windows, and skips since this process
        # started (logged on each ``batch`` row). Neither is checkpointed, so
        # a resumed run starts both afresh.
        self._recent_window_skips: deque[bool] = deque(maxlen=self._SKIP_RATE_WINDOW)
        self._skipped_steps = 0

        # Register per-task loss modules on the (unwrapped) model so their
        # parameters live in `model.parameters()`. The optimizer is built
        # before the trainer is constructed (see `dasm2/training/factory.py`
        # and `dasm2/serialization/checkpoints.py:resume_training`), so the
        # caller is expected to register losses BEFORE `create_optimizer` —
        # we re-register here for idempotence and to assert uniqueness. A
        # ``ModuleDict`` rebuilt from the same Loss instances does not change
        # parameter ``id()`` values, so the optimizer's param group remains
        # valid.
        self._register_task_losses()
        # After registration: ``Task.log_key`` reads the key it stamps.
        self._validate_monitor_task()

    def _validate_monitor_task(self) -> None:
        """Fail at construction, not hours into a run, if ``monitor_task`` cannot work.

        The monitored task must exist, carry a positive ``loss_weight`` (its raw
        loss is recovered by dividing the weighted value by it), and not be the
        target of an ``OverfitGuardHook``, which zeroes ``loss_weight`` mid-run.
        """
        if self.monitor_task is None:
            return
        by_key = {task.log_key: task for task in self.tasks}
        if self.monitor_task not in by_key:
            raise ValueError(
                f"monitor_task={self.monitor_task!r} is not the log key of any "
                f"task; this trainer's tasks have log keys {sorted(by_key)}."
            )
        if by_key[self.monitor_task].loss_weight <= 0.0:
            raise ValueError(
                f"monitor_task={self.monitor_task!r} has loss_weight "
                f"{by_key[self.monitor_task].loss_weight}; the monitored task needs "
                "a positive loss_weight."
            )
        for task in self.tasks:
            for hook in task.pre_cycle_hooks + task.post_cycle_hooks + task.eval_hooks:
                if hook.zeroed_loss_weight_task_key() == self.monitor_task:
                    raise ValueError(
                        f"monitor_task={self.monitor_task!r} is also the target of "
                        f"{type(hook).__name__}, which zeroes the task's loss_weight "
                        "mid-run and would leave nothing to monitor. Monitor the "
                        "other task, or drop the hook."
                    )

    def _register_task_losses(self) -> None:
        """Attach a ``task_losses`` ``ModuleDict`` to the (unwrapped) model.

        ``build_task_loss_module_dict`` asserts each Task carries a distinct
        loss instance (sharing one across tasks would put the same parameters
        into two optimizer groups) and stamps each ``Task.loss_state_dict_key``
        in place so log / codon-metric dicts can read the same key.
        """
        if not self.tasks:
            return
        module_dict = build_task_loss_module_dict(self.tasks)
        # ``unwrapped_model`` strips the DDP/FSDP wrapper; setting attributes
        # on the wrapper would not register parameters under the inner module.
        self.unwrapped_model.task_losses = module_dict
        # Move loss params to the model's device so forward stays device-uniform.
        module_dict.to(self.device)

    def _model_uses_distance(self) -> bool:
        """Check if any component of the model requires distance input."""
        # Go through unwrapped_model — DDP/FSDP do not forward custom attrs.
        model = self.unwrapped_model
        encoder_uses = (
            getattr(model.encoder, "distance_encoding_mode", "none") != "none"
        )
        head_uses = any(
            getattr(h, "use_distance", False) for h in model.prediction_heads.values()
        )
        return encoder_uses or head_uses

    def _validate_distance_encoding_consistency(self) -> None:
        """Validate that dataset distance availability matches model config.

        Ensures that if the model (encoder or head) uses distance, all
        evolutionary datasets provide 'distance' in their batch output, and
        vice versa.

        Raises:
            ValueError: If model expects distance but any dataset doesn't provide it,
                        or if a dataset provides distance but model doesn't use it.
        """
        model_uses_distance = self._model_uses_distance()
        for task in self.tasks:
            for dataset, split_name in [
                (task.train_dataset, "train"),
                (task.val_dataset, "val"),
            ]:
                base = dataset.dataset if isinstance(dataset, Subset) else dataset
                # Check if the dataset provides distance by inspecting a sample
                if len(base) == 0:  # type: ignore[arg-type]
                    continue
                sample = base[0]
                has_distance = "distance" in sample
                if model_uses_distance and not has_distance:
                    raise ValueError(
                        f"Model uses distance encoding but "
                        f"task {split_name} dataset does not provide 'distance' "
                        f"in its batch output. Ensure the dataset was created "
                        f"with use_distance=True."
                    )
                if not model_uses_distance and has_distance:
                    raise ValueError(
                        f"Task {split_name} dataset provides 'distance' "
                        f"but model does not use distance encoding. "
                        f"Set distance_encoding_mode in EncoderConfig or "
                        f"use_distance in the head config, or remove distance "
                        f"from the dataset."
                    )

    def _initialize_distance_bins(self) -> None:
        """Compute quantile bin boundaries from training data and set on model.

        Only runs for binned/context_token encoder modes. Collects all
        distance values from training datasets, computes log-transformed
        quantile boundaries, and stores them as registered buffers on the
        encoder. Skips if the loaded checkpoint already has non-default
        bin boundaries (i.e. the model was previously trained and the
        buffer was restored from state_dict).
        """
        # Go through unwrapped_model — DDP/FSDP do not forward custom attrs.
        encoder = self.unwrapped_model.encoder
        mode = getattr(encoder, "distance_encoding_mode", "none")

        needs_encoder_bins = mode in ("binned", "context_token")

        if not needs_encoder_bins:
            return

        # If the buffer was already populated by a checkpoint load (any
        # non-zero entry), trust it and skip recomputation — at eval time
        # there is no training data to bin from.
        boundaries = encoder.distance_bin_boundaries
        assert isinstance(boundaries, torch.Tensor)
        if torch.any(boundaries != 0):
            return

        # Collect all distance values from training datasets
        all_distances: List[float] = []
        for task in self.tasks:
            dataset = task.train_dataset
            base = dataset.dataset if isinstance(dataset, Subset) else dataset
            for i in range(len(base)):  # type: ignore[arg-type]
                sample = base[i]
                if "distance" in sample:
                    d = sample["distance"]
                    all_distances.append(d.item() if isinstance(d, torch.Tensor) else d)

        if not all_distances:
            raise ValueError(
                "No distance values found in training data for bin computation."
            )

        log_distances = torch.log(
            torch.tensor(all_distances, dtype=torch.float32) + LOG_DISTANCE_EPS
        )

        n_bins: int = encoder.num_distance_bins  # type: ignore[assignment]
        quantiles = torch.linspace(0, 1, n_bins + 1)[1:-1]
        boundaries = torch.quantile(log_distances, quantiles)
        encoder.set_distance_bins(boundaries)  # type: ignore[operator]

    def _get_peak_memory_mb(self) -> Optional[float]:
        """Return peak GPU memory allocated in MB, or None if not on CUDA."""
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            return torch.cuda.max_memory_allocated(self.device) / (1024**2)
        return None

    def _create_scheduler(
        self,
    ) -> Optional[
        Union[
            ReduceLROnPlateau,
            SequentialLR,
            PolynomialLR,
            CosineAnnealingWarmRestarts,
        ]
    ]:
        """Create learning rate scheduler based on scheduler_config.

        Note: This scheduler is intentionally *not* passed through
        ``accelerator.prepare()``.  ``ReduceLROnPlateau`` only inspects the
        metric value supplied to ``scheduler.step(val_loss)`` — it has no
        interaction with the gradient scaler or optimizer internals that
        Accelerate wraps.  If a future scheduler *does* need scaler
        integration, it should be prepared via the compute backend.

        Returns:
            Scheduler instance, or None if no optimizer configured.
            - ``ReduceOnPlateauSchedulerConfig``: ReduceLROnPlateau (stepped per
              evaluation boundary with val_loss)
            - ``WarmupPolynomialSchedulerConfig``: SequentialLR(LinearLR warmup +
              PolynomialLR decay), stepped per optimizer step. Requires
              total_training_steps.
            - ``CosineWarmRestartsSchedulerConfig``: CosineAnnealingWarmRestarts,
              stepped per optimizer step.
        """
        if self.optimizer is None:
            return None

        if isinstance(self.scheduler_config, ReduceOnPlateauSchedulerConfig):
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.scheduler_config.factor,
                patience=self.scheduler_config.eval_patience,
            )

        if isinstance(self.scheduler_config, CosineWarmRestartsSchedulerConfig):
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.scheduler_config.t_0_steps,
                T_mult=self.scheduler_config.t_mult,
                eta_min=self.scheduler_config.eta_min,
            )

        # WarmupPolynomialSchedulerConfig
        assert isinstance(self.scheduler_config, WarmupPolynomialSchedulerConfig)
        warmup_steps = self.scheduler_config.warmup_steps
        total_steps = self.total_training_steps
        if total_steps is None:
            raise ValueError(
                "total_training_steps is required for warmup_polynomial scheduler"
            )

        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be less than "
                f"total_training_steps ({total_steps}). "
                "Reduce warmup_steps or increase max_train_steps."
            )
        decay_steps = total_steps - warmup_steps

        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=self.scheduler_config.warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            decay_scheduler = PolynomialLR(
                self.optimizer,
                total_iters=decay_steps,
                power=self.scheduler_config.polynomial_power,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps],
            )
        else:
            return PolynomialLR(
                self.optimizer,
                total_iters=decay_steps,
                power=self.scheduler_config.polynomial_power,
            )

    def _create_head_scheduler(self) -> Optional[ReduceLROnPlateau]:
        """Create head_optimizer's own plateau scheduler, if any.

        Reuses ``scheduler_config``'s own ``factor``/``eval_patience`` — the
        head optimizer doesn't get an independently-tunable schedule, just
        the same reduce-on-plateau behavior the main optimizer gets, applied
        to the head's own validation signal (see :meth:`_step_head_scheduler`).
        Only built when both a ``head_optimizer`` exists and the configured
        scheduler is ``ReduceOnPlateauSchedulerConfig`` — under a per-step
        scheduler (``WarmupPolynomialSchedulerConfig``,
        ``CosineWarmRestartsSchedulerConfig``) there is no per-eval plateau
        signal to mirror, so ``head_optimizer``'s LR stays flat within a
        cycle (moved only by the cycle-boundary reset), same
        as before this scheduler existed.

        Returns:
            A fresh ``ReduceLROnPlateau`` over ``head_optimizer``, or None.
        """
        if self.head_optimizer_state is None:
            return None
        if not isinstance(self.scheduler_config, ReduceOnPlateauSchedulerConfig):
            return None
        return ReduceLROnPlateau(
            self.head_optimizer_state.optimizer,
            mode="min",
            factor=self.scheduler_config.factor,
            patience=self.scheduler_config.eval_patience,
        )

    def _step_head_scheduler(
        self, tasks: List["Task"], per_task_val: Dict[str, float]
    ) -> None:
        """Step head_optimizer's plateau scheduler, if any.

        Steps on the *raw* (unscaled) validation loss of the task whose
        ``head_type`` matches ``head_optimizer_head_type`` — the same
        unscaling ``OverfitGuardHook`` applies to its own comparison (see
        that class's docstring): ``per_task_val`` is
        ``loss * task.loss_weight``, and this repo's own lambda sweeps vary
        ``loss_weight`` by design, so comparing the scaled value directly
        would make the plateau threshold lambda-dependent. No-op if
        ``head_scheduler`` was never built, the task isn't found (mismatched
        config), or the task's ``loss_weight`` has already been zeroed (e.g.
        by ``OverfitGuardHook`` freezing it) — a zeroed task's own val loss
        is no longer being trained against, so it has nothing left to say
        about the head optimizer's LR.
        """
        if (
            self.head_optimizer_state is None
            or self.head_optimizer_state.scheduler is None
        ):
            return
        task = next(
            (t for t in tasks if t.head_type == self.head_optimizer_state.head_type),
            None,
        )
        if task is None or task.loss_weight == 0.0:
            return
        weighted = per_task_val.get(task.log_key)
        if weighted is None:
            return
        self.head_optimizer_state.step_scheduler(weighted / task.loss_weight)

    @property
    def head_optimizer(self) -> Optional[torch.optim.Optimizer]:
        """The head-scoped optimizer, delegated to head_optimizer_state."""
        return (
            self.head_optimizer_state.optimizer
            if self.head_optimizer_state is not None
            else None
        )

    @property
    def head_optimizer_module(self) -> Optional[torch.nn.Module]:
        """The head module head_optimizer covers, delegated to head_optimizer_state."""
        return (
            self.head_optimizer_state.module
            if self.head_optimizer_state is not None
            else None
        )

    @property
    def head_optimizer_head_type(self) -> Optional[HeadType]:
        """Which head head_optimizer covers, delegated to head_optimizer_state."""
        return (
            self.head_optimizer_state.head_type
            if self.head_optimizer_state is not None
            else None
        )

    @property
    def head_scheduler(self) -> Optional[ReduceLROnPlateau]:
        """head_optimizer's own plateau scheduler, delegated to head_optimizer_state."""
        return (
            self.head_optimizer_state.scheduler
            if self.head_optimizer_state is not None
            else None
        )

    @property
    def head_learning_rate(self) -> Optional[float]:
        """head_optimizer's initial LR, delegated to head_optimizer_state."""
        return (
            self.head_optimizer_state.learning_rate
            if self.head_optimizer_state is not None
            else None
        )

    @property
    def device(self) -> str:
        """Device string for this process, delegated to the compute backend."""
        return self.backend.device

    @property
    def unwrapped_model(self) -> CoreModel:
        """Get the unwrapped model (strips DDP/FSDP wrapper if present)."""
        return self.backend.unwrap_model(self.model)  # type: ignore[return-value]

    @property
    def is_main_process(self) -> bool:
        """True on the main process (rank 0), or always True without accelerate."""
        return self.backend.is_main_process

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size across all parallelism dimensions.

        Equals per_device_batch_size × gradient_accumulation_steps × num_processes.
        """
        return compute_effective_batch_size(
            self.batch_size,
            self.gradient_accumulation_steps,
            self.backend.num_processes,
        )

    def _gather_val_loss(self, loss: float) -> float:
        """Aggregate a scalar validation loss across ranks.

        On SingleGPUBackend this is a no-op. On AccelerateBackend it averages
        the per-rank mean losses. This is correct because DistributedSampler
        ensures each rank processes the same number of batches (via padding),
        so mean-of-per-rank-averages == global average.
        """
        return self.backend.gather_mean(torch.tensor(loss, device=self.device)).item()

    def _get_current_learning_rate(self) -> float:
        """Get current learning rate from optimizer.

        Returns:
            Current learning rate

        Raises:
            ValueError: If no optimizer is configured
        """
        if self.optimizer is None:
            raise ValueError("Cannot get learning rate: no optimizer configured")
        lr: float = self.optimizer.param_groups[0]["lr"]
        return lr

    def _should_stop_early(self) -> Optional[str]:
        """Check if training should stop early.

        Returns:
            A message describing why training should stop, or None to continue.
            Patience-based stopping takes precedence over LR-based stopping.
            LR-based stopping only applies to reduce_on_plateau scheduler.
        """
        if self.optimizer is None:
            return None

        # Check patience-based early stopping (takes precedence)
        if (
            self.early_stopping_patience is not None
            and self.evals_without_improvement >= self.early_stopping_patience
        ):
            threshold_note = (
                f", min_delta={self.early_stopping_min_delta}"
                if self.early_stopping_min_delta > 0
                else ""
            )
            return (
                f"no improvement for {self.evals_without_improvement} evaluations "
                f"(patience={self.early_stopping_patience}{threshold_note})"
            )

        # Check LR-based early stopping (only for plateau scheduler).
        # Polynomial schedulers follow a predetermined decay to near-zero, so
        # LR-based stopping would always trigger near the end of each cycle.
        # Use patience-based early stopping instead.
        if isinstance(self.scheduler, ReduceLROnPlateau):
            current_lr = self._get_current_learning_rate()
            if current_lr < self.min_learning_rate:
                return f"LR {current_lr:.2e} < {self.min_learning_rate:.2e}"

        return None

    def _reset_best_model_tracking(self) -> None:
        """Reset best-model and patience tracking for a fresh ``_train_for_steps`` invocation.

        Called at the start of every step-loop invocation so each task
        (sequential mode) or each cycle (joint mode) tracks its own best
        model and patience independently. This preserves today's per-task
        scoping.
        """
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.evals_without_improvement = 0

    def _update_best_model(self, val_loss: float) -> bool:
        """Update best model state if validation loss improved by more than min_delta.

        An improvement is defined as ``val_loss < best_val_loss - min_delta``
        (strict less-than). When ``min_delta=0.0`` this reduces to
        ``val_loss < best_val_loss``.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if this evaluation qualified as an improvement, False otherwise.
        """
        if val_loss < self.best_val_loss - self.early_stopping_min_delta:
            self.best_val_loss = val_loss
            # Must unwrap DDP model to get clean state_dict
            self.best_model_state = copy.deepcopy(self.unwrapped_model.state_dict())
            self.evals_without_improvement = 0
            return True
        else:
            self.evals_without_improvement += 1
            return False

    def _restore_best_model(self) -> None:
        """Restore best model state from current cycle.

        Loads the model state with the best validation loss observed
        during the current cycle. No-op if no best model was recorded.
        """
        if self.best_model_state is not None:
            self.unwrapped_model.load_state_dict(self.best_model_state)

    def _window_prefetch_factor(
        self,
        dataset: Dataset,
        collate_fn: Optional[Callable[[List[Any]], Any]],
        lengths: Optional[List[int]],
        batch_size: int,
        max_tokens_per_batch: Optional[int],
        window: int,
        requested: Optional[int],
    ) -> int:
        """Per-worker prefetch depth that keeps an accumulation window in flight.

        The training loop draws ``window`` micro-batches back to back before any
        forward pass, while the workers hold only ``workers × prefetch_factor``
        ready batches; a shallower queue stalls the GPU on every window. The
        depth is raised to ``ceil(window / workers)`` unless the
        caller passed ``prefetch_factor`` in ``dataloader_kwargs`` (an explicit
        override wins) or the in-flight batches' upper-bound footprint would
        exceed ``_PREFETCH_RAM_FRACTION`` of ``MemAvailable``, which risks an OOM.
        A queue left shorter than the window triggers a warning.

        Args:
            dataset: The loader's dataset; sample 0 sizes the RAM estimate.
            collate_fn: The loader's collator (None: torch's default).
            lengths: Per-sample sequence lengths, or None if the dataset
                exposes none.
            batch_size: Per-batch row count (row cap under token budgets).
            max_tokens_per_batch: Token-budget cap, or None for fixed batches.
            window: Micro-batches drawn per optimizer step from this loader.
            requested: ``prefetch_factor`` from ``dataloader_kwargs``, if any.

        Returns:
            The ``prefetch_factor`` to pass to the DataLoader.
        """
        configured = (
            requested if requested is not None else self.dataloader_prefetch_factor
        )
        if window <= 1:
            return configured
        workers = self.num_dataloader_workers
        needed = math.ceil(window / workers)
        batch_bytes = _batch_bytes_upper_bound(
            dataset,
            collate_fn,
            lengths,
            batch_size,
            max_tokens_per_batch,
        )
        prefetch = configured
        guard = ""
        if requested is None and needed > configured:
            available = mem_available_bytes()
            estimate = workers * needed * batch_bytes
            if available is None or estimate <= _PREFETCH_RAM_FRACTION * available:
                prefetch = needed
            else:
                guard = (
                    f"; covering it would hold ~{estimate / 1e9:.1f} GB of batches, "
                    f"over {_PREFETCH_RAM_FRACTION:.0%} of the "
                    f"{available / 1e9:.1f} GB MemAvailable"
                )
        if not self.is_main_process:
            return prefetch
        print(
            f"  DataLoader prefetch: workers={workers} "
            f"configured_prefetch={configured} effective_prefetch={prefetch} "
            f"gradient_accumulation_steps={window} "
            f"in_flight<={workers * prefetch * batch_bytes / 1e9:.2f}GB"
        )
        if workers * prefetch < window:
            warnings.warn(
                f"DataLoader holds {workers} workers x prefetch {prefetch} = "
                f"{workers * prefetch} batches, fewer than the {window} "
                f"micro-batches each optimizer step draws{guard}. Every window "
                "will stall the GPU on data loading; raise num_dataloader_workers "
                "if host RAM allows.",
                stacklevel=2,
            )
        return prefetch

    def _create_dataloaders_with_collators(
        self,
        datasets: List[Dataset],
        shuffle: bool,
        batch_size: int,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        collate_fns: Optional[List[Optional[Any]]] = None,
        worker_init_fns: Optional[List[Optional[Any]]] = None,
        samplers: Optional[List[Optional[Any]]] = None,
        max_tokens_per_batch: Optional[int] = None,
        batch_sizes: Optional[List[Optional[int]]] = None,
        prefetch_window: int = 1,
    ) -> List[DataLoader]:
        """Create dataloaders with appropriate collators for each dataset.

        For unified ``TransformDataset`` datasets, uses the collator from
        the Task's ``collate_fn`` field. Falls back to
        ``DataCollatorFromLayoutTensor`` when no explicit collator is provided.

        When a sampler is provided for a dataset, it replaces the ``shuffle``
        flag (``shuffle`` and ``sampler`` are mutually exclusive in DataLoader).

        Args:
            datasets: List of datasets to create loaders for.
            shuffle: Whether to shuffle data (ignored when sampler is provided).
            batch_size: Per-device batch size applied to every loader built
                by this call, unless overridden per-dataset by
                ``batch_sizes``. Callers pass ``self.batch_size`` for
                training and ``self.val_batch_size`` for validation so the
                two paths can diverge independently (validation has no
                backward pass, so it typically admits a larger batch on the
                same device).
            dataloader_kwargs: Optional kwargs for DataLoader.
            collate_fns: Optional per-dataset collators (from Task.collate_fn).
            worker_init_fns: Optional per-dataset worker init functions.
            samplers: Optional per-dataset samplers (e.g. ShuffleBufferSampler).
            batch_sizes: Optional per-dataset batch size overrides (from
                ``Task.batch_size``). ``None`` entries (the default for
                every task) fall back to ``batch_size``. Also used as the
                per-dataset ``max_batch_size`` cap when
                ``max_tokens_per_batch`` is set.
            prefetch_window: Micro-batches drawn back to back from each
                loader before any forward pass (``gradient_accumulation_steps``
                for training, 1 for validation). With workers, the prefetch
                depth is raised so the window is in flight.

        Returns:
            List of configured DataLoaders.
        """
        kwargs = dataloader_kwargs or {}
        loaders = []
        _collate_fns = collate_fns or [None] * len(datasets)
        _worker_init_fns = worker_init_fns or [None] * len(datasets)
        _samplers = samplers or [None] * len(datasets)
        _batch_sizes = batch_sizes or [None] * len(datasets)

        for (
            dataset,
            task_collate_fn,
            task_worker_init_fn,
            task_sampler,
            task_batch_size,
        ) in zip(datasets, _collate_fns, _worker_init_fns, _samplers, _batch_sizes):
            # Not `batch_size = ... else batch_size`: that would rebind the
            # outer parameter and leak this task's override into the next
            # task's fallback.
            effective_batch_size = _effective_batch_size(task_batch_size, batch_size)
            task_kwargs = kwargs.copy()
            if task_collate_fn is not None:
                task_kwargs["collate_fn"] = task_collate_fn
            if task_worker_init_fn is not None:
                task_kwargs["worker_init_fn"] = task_worker_init_fn
            # pin_memory enables async CPU→GPU DMA transfers. This locks
            # page memory, so on memory-constrained nodes callers can disable
            # it via dataloader_kwargs={"pin_memory": False} or the
            # ``dataloader_pin_memory`` config field.
            if self._is_cuda_device and "pin_memory" not in task_kwargs:
                task_kwargs["pin_memory"] = self.dataloader_pin_memory
            # Per-sample lengths (a pass over ``aa_parents``), computed once
            # for both the prefetch RAM estimate and token-budget batching.
            prefetch_raise = self.num_dataloader_workers > 0 and prefetch_window > 1
            lengths = (
                sequence_lengths_for_dataset(dataset)
                if max_tokens_per_batch is not None or prefetch_raise
                else None
            )
            if self.num_dataloader_workers > 0:
                task_kwargs["num_workers"] = self.num_dataloader_workers
                task_kwargs["persistent_workers"] = True
                task_kwargs["prefetch_factor"] = self._window_prefetch_factor(
                    dataset,
                    task_collate_fn,
                    lengths,
                    batch_size=effective_batch_size,
                    max_tokens_per_batch=max_tokens_per_batch,
                    window=prefetch_window,
                    requested=task_kwargs.get("prefetch_factor"),
                )
                # Explicitly use fork so DataLoader workers inherit the
                # parent's warm LRU caches (copy-on-write) and aren't
                # poisoned by force_spawn() calls elsewhere (e.g. BL
                # optimization).
                task_kwargs["multiprocessing_context"] = "fork"
            # Token-budget batching: form variable-size micro-batches
            # bounded by padded tokens so a long-sequence cluster holds fewer rows
            # and no rank's padded forward blows up. Needs cheap per-sample lengths
            # (evolutionary heads expose ``aa_parents``); other datasets fall back
            # to fixed-size batching.
            if max_tokens_per_batch is not None and lengths is not None:
                # batch_sampler is mutually exclusive with batch_size/shuffle/
                # sampler, so drop any task sampler. For a streaming Parquet
                # dataset the dropped sampler is a ShuffleBufferSampler whose
                # row-group order keeps the backend's LRU cache warm; hand its
                # row-group geometry to the budget sampler so it packs WITHIN
                # that locality stream instead of globally shuffling (which
                # touches every row group per mega-batch and thrashes the cache
                # — a large data-load slowdown). The index
                # spaces must line up (no cv_split Subset between sampler and
                # lengths); fall back to global shuffling otherwise.
                rg_boundaries = None
                rg_buffer = 8192
                if isinstance(task_sampler, ShuffleBufferSampler) and len(
                    task_sampler
                ) == len(lengths):
                    rg_boundaries = task_sampler.row_group_boundaries
                    rg_buffer = task_sampler.buffer_size
                task_kwargs.pop("sampler", None)
                batch_sampler = TokenBudgetBatchSampler(
                    lengths=lengths,
                    max_tokens_per_batch=max_tokens_per_batch,
                    shuffle=shuffle,
                    max_batch_size=effective_batch_size,
                    row_group_boundaries=rg_boundaries,
                    buffer_size=rg_buffer,
                    # The base seed; ``_set_loader_epoch`` adds ``_loader_cycle``
                    # per pass. Carried in TrainingState because the run's
                    # --seed is not otherwise saved in the checkpoint.
                    seed=self._data_seed,
                )
                loader = DataLoader(dataset, batch_sampler=batch_sampler, **task_kwargs)
            else:
                if max_tokens_per_batch is not None and self.is_main_process:
                    warnings.warn(
                        f"max_tokens_per_batch={max_tokens_per_batch} set but "
                        f"{type(dataset).__name__} exposes no per-sample lengths; "
                        "falling back to fixed-size batching for this dataset.",
                        stacklevel=2,
                    )
                if task_sampler is not None:
                    task_kwargs["sampler"] = task_sampler
                    use_shuffle = False
                else:
                    use_shuffle = shuffle
                loader = DataLoader(
                    dataset,
                    batch_size=effective_batch_size,
                    shuffle=use_shuffle,
                    **task_kwargs,
                )
            # Backend adds DistributedSampler if needed (no-op for SingleGPU)
            loaders.append(self.backend.prepare_dataloader(loader))

        return loaders

    def _run_hook_with_logging(
        self,
        hook: "TrainingHook",
        phase: HookPhase,
        task: "Task",
        cycle_count: int,
        num_cycles: int,
        start_cycle: int = 0,
    ) -> None:
        """Run a single hook with timing and logging.

        Args:
            hook: TrainingHook instance to run
            phase: Hook execution phase (pre_cycle or post_cycle)
            task: Task providing datasets for the hook
            cycle_count: Current cycle number (0-indexed)
            num_cycles: Total number of cycles
            start_cycle: Cycle the trainer started from (0 unless resuming).
                Threaded into the hook context so resume-aware hooks (e.g.,
                ``BranchLengthOptimizer``) can detect the resume entry point.
        """
        hook_name = type(hook).__name__
        self.logger.log_hook_start(
            cycle_count, hook_name, phase.value, task.head_type.value
        )

        hook_start = time.perf_counter()
        hook_method = getattr(hook, phase.value)
        hook_method(
            self,
            train_data=task.train_data,
            val_data=task.val_data,
            cycle_count=cycle_count,
            num_cycles=num_cycles,
            start_cycle=start_cycle,
            head_type=task.head_type,
        )
        hook_duration = time.perf_counter() - hook_start

        self.logger.log_hook_end(
            cycle_count, hook_name, phase.value, task.head_type.value, hook_duration
        )

    @contextmanager
    def _eval_mode(self) -> Iterator[None]:
        """Context manager that switches the model to eval and restores train.

        Mirrors the existing ``_autocast_ctx`` pattern. Restoration runs from
        ``finally`` so a raised exception inside ``_evaluate`` cannot leave
        the model stuck in eval mode.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            self.model.train(was_training)

    def _gather_active_metric(self, active: ActiveMetric) -> Dict[str, float]:
        """DDP-reduce one accumulated metric group to its final metric dict.

        Codon counters are integers; SF counts carry fractional histogram
        mass. The provider declares which via ``counts_are_int`` so the
        gather round-trip uses the right dtype.
        """
        return self._gather_counts_then_reduce(
            active.accumulator.counts(),
            active.provider.metrics_from_counts,
            value_to_int=active.provider.counts_are_int,
        )

    def _gather_counts_then_reduce(
        self,
        local_counts: Dict[str, Any],
        metrics_from_counts: "Callable[[Dict[str, Any]], Dict[str, float]]",
        *,
        value_to_int: bool,
    ) -> Dict[str, float]:
        """Shared DDP gather-then-reduce for per-batch metrics accumulators.

        Per-rank ratios cannot be averaged correctly across unequal-partition
        shards (e.g. one rank could see no mutated positions while another
        sees many). Gathering raw counts first and computing the ratios from
        the global totals avoids that bias entirely. ``backend.gather_mean``
        only operates on float scalars, so for an int count
        ``mean(count_per_rank) * num_processes`` recovers the global sum
        exactly. SF metrics carry float counts (histograms include
        fractional contributions after dtype promotion) so the int round-trip
        is gated by ``value_to_int``.
        """
        if self.backend.num_processes == 1:
            return metrics_from_counts(local_counts)
        num_processes = self.backend.num_processes
        global_counts: Dict[str, Any] = {}
        for name, value in local_counts.items():
            local = torch.tensor(float(value), device=self.device)
            mean = self.backend.gather_mean(local).item()
            total = mean * num_processes
            global_counts[name] = int(round(total)) if value_to_int else total
        return metrics_from_counts(global_counts)

    @staticmethod
    def _get_n_loss_positions(batch: Dict[str, Any]) -> int:
        """Get the number of positions contributing to the loss from a batch.

        Resolution order:

        1. ``loss_mask`` — evolutionary / whichmut heads.
        2. ``mask`` — StructuralGuidanceHead (`HERMES valid-position mask`) and
           PerSiteChildHead use this key for the same semantic. Falling
           through to the ``batch_size`` fallback would undercount by a factor
           of ``seq_len``, giving the head roughly ``seq_len`` times more
           gradient weight than its ``loss_weight`` implies and producing
           ``seq_len``-scale values in the logged per-task loss.
        3. ``batch_size`` — DMS / DeltaAffinity heads where each sample is one
           loss-contributing measurement.

        Args:
            batch: Batch dictionary from a dataset's DataLoader.

        Returns:
            Number of loss-contributing positions as a Python int.
        """
        if "loss_mask" in batch:
            return int(batch["loss_mask"].sum().item())
        if "mask" in batch:
            return int(batch["mask"].sum().item())
        # Fallback: batch_size (each sample = one loss position)
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v.shape[0]
        raise ValueError("Cannot determine n_loss_positions: batch has no tensors")

    def _compute_batch_loss(
        self,
        batch: Dict[str, Any],
        task: "Task",
        active_metrics: Optional[List[ActiveMetric]] = None,
    ) -> torch.Tensor:
        """Compute weighted sum loss for a specific task.

        All heads return sum-reduced losses. The trainer normalizes by total
        ``loss_mask`` positions across the accumulation window.

        When mixed precision is enabled, the forward pass runs under BF16 autocast.
        BF16 has the same exponent range as FP32, so no GradScaler is needed.

        When the model is in eval mode, the forward runs under
        ``torch.inference_mode()`` so no autograd graph is built.

        Args:
            batch: Batch data from task's dataset.
            task: Task specifying which head to use.
            active_metrics: Validation metric accumulators eligible for this
                task, built by
                :func:`metric_providers.build_active_metrics`. Each is fed
                the per-batch tensor its extractor pulls from this forward's
                predictions. ``None`` on the training path (metrics are
                validation-only).

        Returns:
            Sum loss tensor (not mean), multiplied by task's loss_weight
        """
        batch = move_batch_to_device(batch, self.device)
        grad_ctx: contextlib.AbstractContextManager[Any] = (
            torch.inference_mode()
            if not self.model.training
            else contextlib.nullcontext()
        )
        with grad_ctx, self._autocast_ctx:
            outputs = self.model(batch, head_names=[task.head_type])
            head_key = task.head_type.value
            predictions = outputs[head_key]
            loss: torch.Tensor = task.loss(predictions, batch)
            for active in active_metrics or ():
                active.observe(predictions, batch)
        return loss * task.loss_weight

    def _find_nan_in_parameters(self) -> Optional[Tuple[str, str]]:
        """Check model parameters for non-finite weights or gradients.

        Uses ``isfinite`` (not ``isnan``) so an Inf gradient is caught as well
        as a NaN one: the degenerate near-all-N PCPs this guard defends against
        (branch-length MLE saturating) are at least as likely to overflow to
        ``+/-inf`` as to underflow to ``NaN``. Letting an
        Inf gradient through to ``optimizer.step()`` would silently corrupt
        weights for one step, defeating the graceful-skip mechanism.

        Returns:
            None if all clean, or ``(param_name, "weights" | "gradients")``
            for the first non-finite value found.
        """
        for param_name, param in self.model.named_parameters():
            if not torch.isfinite(param).all():
                return (param_name, "weights")
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return (param_name, "gradients")
        return None

    def _check_gradients(self) -> bool:
        """Check model parameters and gradients for non-finite values.

        Returns True if gradients are valid. Raises on non-finite weights
        (unrecoverable). Returns False on non-finite gradients (recoverable via
        retry).

        Returns:
            True if gradients are valid, False if non-finite gradients detected.

        Raises:
            ValueError: If a non-finite value is detected in parameter weights
                (unrecoverable).
        """
        nonfinite_found = self._find_nan_in_parameters()
        if nonfinite_found is None:
            return True
        param_name, location = nonfinite_found
        if location == "weights":
            raise ValueError(
                f"Non-finite value detected in parameter weights: {param_name}"
            )
        return False

    def _gradients_finite_all_ranks(self) -> bool:
        """Rank-synchronized version of :meth:`_check_gradients`.

        DDP all-reduces gradients at the final micro-batch's syncing backward, so
        a local ``_check_gradients`` verdict is normally already rank-consistent.
        We all-reduce the flag anyway (mirroring the non-finite-loss guard in
        :meth:`_forward_backward_window`) so the retry/skip decision is provably
        identical on every rank — a per-rank divergence here would deadlock the
        next collective exactly as the old per-rank loss raise did.

        ``_check_gradients`` still raises locally on NaN *weights* (unrecoverable
        and rank-consistent, since weights only diverge if gradients do, which the
        all-reduce prevents). Returns True iff every rank reports finite gradients.
        """
        local_nonfinite = torch.tensor(
            0.0 if self._check_gradients() else 1.0, device=self.device
        )
        return self.backend.gather_mean(local_nonfinite).item() <= 0.0

    def _clip_and_step(self) -> None:
        """Clip gradients and step optimizer.

        Also records the per-step pre-clip gradient norm and clip-fire flag
        into :attr:`_optimizer_step_metrics`, which is
        flushed at every eval boundary by ``_run_eval_boundary``.

        Args: None — reads ``self.gradient_clip_val`` and ``self.optimizer``.

        Raises:
            ValueError: If optimizer is None.
        """
        if self.optimizer is None:
            raise ValueError("Optimizer is required for training but was not provided")
        # ``clip_grad_norm_`` returns the *pre-clip* total norm, so it doubles
        # as the §7 grad-norm probe and needs no separate norm pass. When
        # clipping is disabled we clip to ``inf`` — a no-op that still measures
        # the norm — so ``grad_norm_pre_clip_mean`` stays meaningful even with
        # ``gradient_clip_val`` unset (``record_step`` treats a ``None``
        # threshold as "record the norm, but clipping cannot have fired").
        max_norm = (
            self.gradient_clip_val if self.gradient_clip_val is not None else math.inf
        )
        pre_clip_norm = float(
            self.backend.clip_grad_norm_(self.model.parameters(), max_norm).item()
        )
        self._optimizer_step_metrics.record_step(pre_clip_norm, self.gradient_clip_val)
        self.optimizer.step()
        # head_optimizer shares this same clipped-gradient
        # snapshot — clip_grad_norm_ above already covers every model
        # parameter, head-optimizer-owned or not, as a single global clip;
        # this just applies the already-clipped head gradients with the
        # head's own optimizer settings.
        if self.head_optimizer_state is not None:
            self.head_optimizer_state.step()
        self._step_per_step_scheduler()

    def _task_n_labels(self, task: "Task", batch: Dict[str, Any]) -> int:
        """Resolve a task's label count for per-task loss normalization.

        Defaults to ``_get_n_loss_positions`` (``loss_mask.sum()``). A loss
        implementing :class:`SelfNormalizingLoss` overrides this with its own
        label unit — e.g. CSP counts substituted sites while sharing
        DASM's broad ``loss_mask``.
        """
        loss = task.loss
        if isinstance(loss, SelfNormalizingLoss):
            return loss.n_loss_labels(batch)
        return self._get_n_loss_positions(batch)

    def _count_window_labels(
        self,
        window_microbatches: List[List[Any]],
        tasks: List["Task"],
    ) -> List[float]:
        """Per-task window-total label counts, reduced across ranks.

        Counts use masks only — no forward pass and no autograd graph — so the
        window can be sized before any micro-batch graph is built. Each batch
        first goes through the model's ``prepare_training_batch``, which may
        narrow its ``loss_mask`` in the dict (the site-masked DASM). This is what
        lets :meth:`_process_window` divide each task by its *own* window total
        ``N_t = Σ_i n_{t,i}`` instead of the pooled ``Σ_t Σ_i n``.
        The per-task label *count* (``_task_n_labels``) reads only ``loss_mask``
        / ``n_loss_labels`` and does not call ``move_batch_to_device``; the
        cross-rank *reduction* below is the only device-touching step here.

        ``gather_mean`` (not sum) mirrors the DDP gradient mean-reduce: with a
        sum-reduced loss pre-scaled by ``1/N_t`` and DDP averaging gradients
        across ranks, dividing by the cross-rank *mean* of each task's per-rank
        label count makes the final per-parameter update equal the single-GPU
        ``(1/N_t)·Σ_global ∇`` — exactly as the previous global ``1/mean_n``
        scaling did for the pooled total.
        """
        local = [0] * len(tasks)
        for batches in window_microbatches:
            # ``tasks`` and ``batches`` are parallel lists, so ``task_idx`` is
            # the task's stable position across every micro-batch in the window.
            for task_idx, (task, batch) in enumerate(zip(tasks, batches)):
                # Before the count: a model may narrow ``loss_mask`` here.
                self.unwrapped_model.prepare_training_batch(batch)
                local[task_idx] += self._task_n_labels(task, batch)
        return [
            self.backend.gather_mean(torch.tensor(float(n), device=self.device)).item()
            for n in local
        ]

    def _shared_parameters(self) -> List[torch.nn.Parameter]:
        """Every trainable model parameter not owned by a single task's own
        prediction head — i.e. the shared encoder plus any other non-head
        submodule (e.g. ``task_losses``, if it carries learnable parameters).

        Mirrors the head/non-head split
        ``dasm2.training.hooks.overfit_guard.freeze_task_head`` already makes
        for one task at a time; this returns the complementary set across all
        tasks at once, for ``combine_gradient_hooks`` dispatch.

        Not every entry here is genuinely *contended* between tasks — e.g. a
        ``task_losses`` submodule only one task owns still lands in this
        list (it's outside ``prediction_heads``), and every other task
        contributes an explicit zero gradient for it (see
        ``_backward_with_gradient_combine``). A conflict-resolving hook
        (e.g. PCGrad) will see spurious "agreement" there — real for now,
        but worth knowing if a future hook's behavior looks off on such a
        parameter.
        """
        return [
            p
            for name, p in self.unwrapped_model.named_parameters()
            if not name.startswith("prediction_heads.")
        ]

    def _combine_gradients(
        self,
        per_task_grads: Dict[str, List[torch.Tensor]],
        shared_params: List[torch.nn.Parameter],
    ) -> List[torch.Tensor]:
        """Combine per-task gradients on ``shared_params`` into one gradient.

        With no ``combine_gradient_hooks`` attached, this is a plain sum —
        reproducing today's ``torch.stack(terms).sum()`` combination exactly.
        With one attached (``Trainer.__init__`` enforces at most one), the
        hook's own (possibly modified) per-task gradients are summed instead.
        """
        combined_per_task = per_task_grads
        if self.combine_gradient_hooks:
            (hook,) = self.combine_gradient_hooks
            combined_per_task = hook.combine_gradients(
                self, per_task_grads, shared_params
            )
        task_keys = list(combined_per_task.keys())
        return [
            torch.stack([combined_per_task[k][i] for k in task_keys]).sum(dim=0)
            for i in range(len(shared_params))
        ]

    def _backward_with_gradient_combine(
        self, task_terms: List[Tuple["Task", torch.Tensor]]
    ) -> None:
        """Per-task gradient extraction + combination for one micro-batch.

        Only called when 2+ tasks contribute to the micro-batch and
        ``self.combine_gradient_hooks`` is non-empty — see
        ``_forward_backward_window``. Each task's scaled loss has its own,
        independent forward graph (D2: the encoder runs once per task's own
        batch), so a single ``torch.autograd.grad`` call per task — over that
        task's shared parameters plus its own head parameters — needs no
        ``retain_graph``. Head parameters accumulate directly (no other task
        ever touches them); shared parameters go through
        ``_combine_gradients`` first. Both accumulate into ``.grad`` via
        explicit ``+=``, matching the additive accumulation
        ``self.backend.backward()`` normally performs across a window's
        micro-batches — this method never calls ``optimizer.zero_grad()``.
        """
        shared_params = self._shared_parameters()
        per_task_shared_grads: Dict[str, List[torch.Tensor]] = {}
        for task, scaled_loss in task_terms:
            head_params = list(
                self.unwrapped_model.prediction_heads[task.head_type.value].parameters()
            )
            grads = torch.autograd.grad(
                scaled_loss, shared_params + head_params, allow_unused=True
            )
            shared_grads = grads[: len(shared_params)]
            head_grads = grads[len(shared_params) :]
            per_task_shared_grads[task.log_key] = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(shared_grads, shared_params)
            ]
            for p, g in zip(head_params, head_grads):
                # torch.autograd.grad's stub return type doesn't reflect
                # allow_unused=True's real per-element Optional[Tensor] —
                # mypy sees `g: Tensor` and calls this unreachable, but a
                # head param genuinely can go untouched by a given task.
                if g is None:
                    continue  # type: ignore[unreachable]
                p.grad = g.clone() if p.grad is None else p.grad + g
        combined = self._combine_gradients(per_task_shared_grads, shared_params)
        for p, g in zip(shared_params, combined):
            p.grad = g.clone() if p.grad is None else p.grad + g

    def _forward_backward_window(
        self,
        window_microbatches: List[List[Any]],
        tasks: List["Task"],
        n_labels: List[float],
    ) -> Optional[Dict[str, float]]:
        """Restartable forward+backward of one window, each micro-batch per-task pre-scaled.

        Zeros gradients on entry (so ``_process_window`` can call this again on
        NaN-gradient retry), then forward+backwards every micro-batch. Task
        ``t``'s contribution to each micro-batch backward is
        ``loss_weight_t · task_sum / N_t``, so the accumulated gradient realizes
        ``Σ_t (loss_weight_t / N_t) · ∇ Σ_i task_sum_{t,i}`` — each task divided
        by its own window-total label count. Only one micro-batch *graph* is held
        at a time (backward frees it before the next), so peak autograd/activation
        memory matches the previous per-micro-batch path; the window's input
        batches are buffered up front by the caller (as the old replay cache did).

        ``n_labels[task_idx]`` is the task's *window total* ``N_t``, not a
        per-micro-batch count. A task with ``N_t <= 0`` (no labels anywhere in
        the window, e.g. CSP on a substitution-free window) is skipped for every
        micro-batch: its term would be ``0/0``. The caller guarantees at least
        one task has ``N_t > 0``, and that task is therefore included in *every*
        micro-batch, so ``terms`` is never empty.

        Returns the per-task ``loss_weight``-scaled sum-loss accumulated over the
        window, keyed by ``task.log_key``, for the step/eval logging windows, or
        ``None`` if the window's loss was non-finite on **any** rank (the caller
        then skips the optimizer step on every rank — see ``_process_window``).

        The finite-loss check is **rank-synchronized**: DDP shards the global
        batch, so a single degenerate sample (e.g. a near-all-N ancestor PCP,
        a long padded batch) can make exactly one rank's loss non-finite. A per-rank
        ``raise``/skip here would let the healthy ranks reach the final
        micro-batch's syncing backward while the bad rank does not, deadlocking
        them on the gradient all-reduce. all-reducing a non-finite flag makes
        every rank reach the same decision at the same micro-batch.

        When ``self.combine_gradient_hooks`` is non-empty, a
        micro-batch with 2+ contributing tasks routes through
        ``_backward_with_gradient_combine`` instead of
        ``self.backend.backward(micro_loss)`` — ``micro_loss`` is still built
        for the non-finite check below either way, just not backwarded
        directly in that branch. A run with no ``combine_gradient_hooks``
        attached (every existing config) never takes this branch; the code
        path below it is byte-for-byte what it was before this hook category
        existed.

        A task whose ``loss_weight`` has been zeroed out at runtime (e.g.
        ``OverfitGuardHook`` firing, which also freezes that task's own head
        parameters — ``requires_grad=False``) is left out of ``task_terms``
        entirely, not included with an inert zero term: its head parameters
        no longer require grad, and ``torch.autograd.grad`` (unlike
        ``.backward()``) raises if asked to differentiate with respect to a
        tensor that doesn't require grad at all, regardless of
        ``allow_unused``. Dropping it here also means that once only one
        task remains alive, the ``len(task_terms) >= 2`` check below falls
        back to the plain ``self.backend.backward(micro_loss)`` path — no
        gradient-combine hook runs at all, which is correct: a
        conflict-resolution hook like PCGrad has nothing to reconcile a
        single task's gradient against.
        """
        if self.optimizer is None:
            raise ValueError("Optimizer is required for training but was not provided")
        self.optimizer.zero_grad()
        if self.head_optimizer_state is not None:
            self.head_optimizer_state.zero_grad()
        per_task_loss_sum: Dict[str, float] = {}
        num_micro = len(window_microbatches)
        for micro_idx, batches in enumerate(window_microbatches):
            terms: List[torch.Tensor] = []
            task_terms: List[Tuple["Task", torch.Tensor]] = []
            for task_idx, (task, batch) in enumerate(zip(tasks, batches)):
                if n_labels[task_idx] <= 0:
                    continue
                loss_t = self._compute_batch_loss(batch, task)
                per_task_loss_sum[task.log_key] = (
                    per_task_loss_sum.get(task.log_key, 0.0) + loss_t.item()
                )
                scaled = loss_t / n_labels[task_idx]
                terms.append(scaled)
                if self.combine_gradient_hooks and task.loss_weight > 0:
                    task_terms.append((task, scaled))
            # Invariant from the docstring: ``_process_window`` rejects windows
            # where every task has ``N_t <= 0``, so at least one task survives
            # the ``continue`` above for *every* micro-batch.
            assert terms, (
                f"no terms for micro-batch {micro_idx}: all tasks have "
                "n_labels <= 0 for this micro-batch despite a positive window "
                "total"
            )
            micro_loss = torch.stack(terms).sum()
            # Rank-synchronized non-finite (NaN/Inf) guard. ``gather_mean`` is a
            # collective every rank calls once per micro-batch, so the loop stays
            # in lockstep across ranks. If the mean of the per-rank flags is > 0,
            # at least one rank saw a non-finite loss this micro-batch and ALL
            # ranks abort the window together (no partial gradient all-reduce).
            local_nonfinite = torch.tensor(
                0.0 if bool(torch.isfinite(micro_loss)) else 1.0,
                device=self.device,
            )
            if self.backend.gather_mean(local_nonfinite).item() > 0.0:
                return None
            # Suppress gradient sync for non-boundary micro-batches (DDP perf);
            # the final micro-batch's backward triggers the cross-rank reduce.
            is_last_micro = micro_idx == num_micro - 1
            if self.combine_gradient_hooks and len(task_terms) >= 2:
                self._backward_with_gradient_combine(task_terms)
            elif is_last_micro:
                self.backend.backward(micro_loss)
            else:
                with self.backend.no_sync(self.model):
                    self.backend.backward(micro_loss)
        return per_task_loss_sum

    def _skip_nonfinite_window(
        self, tasks: List["Task"], reason: str = "loss"
    ) -> Tuple[Dict[str, Tuple[float, int]], float, int]:
        """Drop a window whose loss OR gradient was non-finite on at least one rank.

        Reached identically on every rank (both the non-finite-loss decision in
        ``_forward_backward_window`` and the non-finite-gradient decision in
        ``_gradients_finite_all_ranks`` are rank-synchronized), so the ranks stay
        in lockstep — no optimizer step, partial gradients discarded, and a
        zero-label logging record returned (``n=0`` keeps it out of the loss
        mean). Fails loudly — on every rank at once — if windows keep coming
        back non-finite, which means training is genuinely diverging rather than
        hitting a rare degenerate sample.

        ``reason`` ("loss" or "gradient") is surfaced in the skip warning so the
        log distinguishes which guard fired — a key signal when relaunching to
        diagnose a hang.
        """
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        if self.head_optimizer_state is not None:
            self.head_optimizer_state.zero_grad()
        self._consecutive_nonfinite_windows += 1
        if self._consecutive_nonfinite_windows > self._MAX_NONFINITE_WINDOWS:
            raise ValueError(
                f"Non-finite training {reason} for "
                f"{self._consecutive_nonfinite_windows} consecutive windows "
                f"(> _MAX_NONFINITE_WINDOWS={self._MAX_NONFINITE_WINDOWS}); "
                "training is diverging, not just hitting a rare bad sample."
            )
        self._record_window_outcome(skipped=True, reason=reason)
        warnings.warn(
            f"Skipping optimizer step for a window with non-finite {reason} on at "
            f"least one rank (consecutive skips: "
            f"{self._consecutive_nonfinite_windows}).",
            stacklevel=2,
        )
        return ({task.log_key: (0.0, 0) for task in tasks}, 0.0, 0)

    def _record_window_outcome(self, skipped: bool, reason: str = "loss") -> None:
        """Track the skip rate; raise past ``_MAX_SKIPS_PER_RATE_WINDOW`` recent skips.

        Every rank reaches this with the same ``skipped`` (the skip decision is
        rank-synchronized), so every rank raises together.
        """
        self._recent_window_skips.append(skipped)
        if not skipped:
            return
        self._skipped_steps += 1
        recent_skips = sum(self._recent_window_skips)
        if recent_skips > self._MAX_SKIPS_PER_RATE_WINDOW:
            raise ValueError(
                f"Non-finite training {reason}: {recent_skips} of the last "
                f"{len(self._recent_window_skips)} windows skipped "
                f"(> _MAX_SKIPS_PER_RATE_WINDOW={self._MAX_SKIPS_PER_RATE_WINDOW} "
                f"per {self._SKIP_RATE_WINDOW}); training is degrading."
            )

    def _process_window(
        self,
        window_microbatches: List[List[Any]],
        tasks: List["Task"],
    ) -> Tuple[Dict[str, Tuple[float, int]], float, int]:
        """Per-task-normalized backward + optimizer step for one accumulation window.

        Each task ``t`` is normalized by its own window-total label count
        ``N_t`` rather than the pooled ``Σ_t N_t``:

            grad = Σ_t (loss_weight_t / N_t) · Σ_i ∇ task_sum_{t,i}

        Single-task reduces to ``(loss_weight / N) · ∇ Σ_i task_sum_i`` —
        the same gradient the previous global ``1/mean_n`` scaling produced.
        A ``loss_weight=0`` task contributes a ``0/N`` term, so it is inert
        (it no longer dilutes its partners through a shared denominator). A task
        with ``N_t == 0`` (no labels this window) is silently skipped (see
        :meth:`_forward_backward_window`); only a fully-zero-label window raises.

        The window is **counted first** (masks only) so each ``N_t`` is known
        before any forward, then forward+backwarded pre-scaled by
        ``loss_weight_t / N_t``. Non-finite gradients trigger a replay up to
        ``_MAX_GRADIENT_RETRIES`` times (counts reused on replay); a gradient that
        is still non-finite after the last retry skips the window on all ranks
        (``_skip_nonfinite_window``), exactly like a non-finite loss, rather than
        raising — see :meth:`_gradients_finite_all_ranks`.

        Args:
            window_microbatches: ``gradient_accumulation_steps`` entries; each
                entry is one batch per task (parallel to ``tasks``).
            tasks: Tasks processed this window (``[task]`` single-task,
                ``self.tasks`` joint).

        A window whose loss is non-finite on any rank takes no optimizer step
        (see :meth:`_skip_nonfinite_window`) and returns a zero-label record;
        only ``_MAX_NONFINITE_WINDOWS`` consecutive such skips, or more than
        ``_MAX_SKIPS_PER_RATE_WINDOW`` in the last ``_SKIP_RATE_WINDOW``, raise.

        Returns:
            ``(per_task, total_loss_sum, total_n)`` for logging — ``per_task``
            maps ``task.log_key`` to ``(loss_weight_scaled_sum_loss, N_t)``.
            A skipped (non-finite loss or gradient) window returns
            ``({log_key: (0.0, 0)}, 0.0, 0)``.

        Raises:
            ValueError: On zero total labels, or a non-finite loss/gradient for
                ``_MAX_NONFINITE_WINDOWS`` consecutive windows (genuine
                divergence) or too many of the last ``_SKIP_RATE_WINDOW``. NaN *weights* are caught by ``_check_gradients``; a
                missing optimizer by ``_forward_backward_window``.
        """
        n_labels = self._count_window_labels(window_microbatches, tasks)
        if all(n <= 0 for n in n_labels):
            raise ValueError(
                "accumulation window has zero loss labels across all tasks and "
                f"all {self.backend.num_processes} rank(s) — check that "
                "loss_mask (or each loss's n_loss_labels) is non-empty"
            )

        # Forward+backward once, then retry on NaN gradients. Every attempt's
        # gradients are checked before the next runs, so the final retry is not
        # wasted: 1 initial pass + up to _MAX_GRADIENT_RETRIES replays.
        per_task_loss_sum = self._forward_backward_window(
            window_microbatches, tasks, n_labels
        )
        if per_task_loss_sum is None:
            return self._skip_nonfinite_window(tasks)
        retries = 0
        while not self._gradients_finite_all_ranks():
            if retries >= self._MAX_GRADIENT_RETRIES:
                # A non-finite gradient that survives every retry is treated like
                # a non-finite loss: skip the window on all ranks together rather
                # than raising per-rank. A per-rank raise here
                # would deadlock DDP exactly as the old loss raise did — DDP
                # shards the batch, so a degenerate sample can make one rank's
                # accumulated gradient non-finite. The consecutive-skip backstop
                # in _skip_nonfinite_window still fails loudly if non-finite
                # windows keep coming (genuine divergence, not a rare bad sample).
                return self._skip_nonfinite_window(tasks, reason="gradient")
            retries += 1
            per_task_loss_sum = self._forward_backward_window(
                window_microbatches, tasks, n_labels
            )
            if per_task_loss_sum is None:
                return self._skip_nonfinite_window(tasks)
        self._consecutive_nonfinite_windows = 0
        self._record_window_outcome(skipped=False)
        self._clip_and_step()

        per_task: Dict[str, Tuple[float, int]] = {}
        total_loss_sum = 0.0
        total_n = 0
        for task_idx, task in enumerate(tasks):
            # ``build_task_loss_module_dict`` already asserts log_keys are
            # distinct, so this assertion is defense-in-depth against future
            # callers building ``tasks`` outside that path: without it, a
            # duplicate key would silently overwrite ``per_task[key]`` while
            # ``total_loss_sum`` accumulated both, so logged per-task and total
            # losses would disagree.
            assert task.log_key not in per_task, (
                f"duplicate task.log_key {task.log_key!r} in window — log keys "
                "must be unique across tasks (see build_task_loss_module_dict)"
            )
            # The float ``n_labels[task_idx]`` is the real gradient divisor (used
            # in _forward_backward_window); this rounded int is only the label
            # count surfaced to the logging accumulator.
            n_t = int(round(n_labels[task_idx]))
            loss_sum_t = per_task_loss_sum.get(task.log_key, 0.0)
            per_task[task.log_key] = (loss_sum_t, n_t)
            total_loss_sum += loss_sum_t
            total_n += n_t
        return per_task, total_loss_sum, total_n

    def multi_cycle_train(
        self,
        num_cycles: int = 2,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        start_cycle: int = 0,
        start_cycle_task_index: int = 0,
        start_cycle_step: int = 0,
    ) -> CoreModel:
        """Sequential multi-cycle, multi-task training.

        Each cycle iterates through ``self.tasks``: for every task it runs
        ``task.pre_cycle_hooks``, executes one ``_train_for_steps`` invocation
        with ``self.step_budget``, then runs ``task.post_cycle_hooks`` on the
        restored best model. Best-model + patience tracking is per-task
        (matches today's per-task scoping).

        Resume: ``start_cycle`` skips full cycles. In the resumed cycle,
        tasks with index ``< start_cycle_task_index`` are skipped (including
        their hooks); the task at that index resumes from
        ``StepBudget(start_step=start_cycle_step, ...)``; subsequent tasks
        in the same cycle start from step 0.

        Args:
            num_cycles: Number of training cycles.
            dataloader_kwargs: Optional kwargs for DataLoader.
            start_cycle: Cycle to start from when resuming (0-indexed).
            start_cycle_task_index: Task index to resume in ``start_cycle``
                (0 except for sequential mid-task resume).
            start_cycle_step: Optimizer step to resume at within the resumed
                task (0 except for mid-task resume).

        Returns:
            Trained model (restored to best validation loss from final task).
        """
        if self.monitor_task is not None:
            raise ValueError(
                f"monitor_task={self.monitor_task!r} needs joint training, where every "
                "task is evaluated together at each eval. Sequential multi-task "
                "training evaluates one task at a time, so it cannot follow one "
                "task's loss. Unset monitor_task, or train jointly."
            )
        budget = self._require_step_budget()
        self._warn_if_dataset_sizes_imbalanced()
        training_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.is_main_process:
            self.logger.log_training_start(
                num_cycles,
                budget.max_train_steps,
                budget.eval_every_steps,
                batch_size=self.effective_batch_size,
                per_device_batch_size=self.batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                num_processes=self.backend.num_processes,
            )

        # On a clean cycle boundary at resume, apply the optimizer reset that
        # would have happened between (start_cycle-1) and start_cycle. Skip
        # for mid-cycle/mid-task resume so we don't double-reset.
        mid_cycle_resume = start_cycle_task_index > 0 or start_cycle_step > 0
        if start_cycle > 0 and not mid_cycle_resume:
            self._reset_optimizer_with_adjusted_lr(start_cycle, num_cycles)

        for cycle in range(start_cycle, num_cycles):
            self._current_cycle = cycle
            cycle_start = time.perf_counter()
            self._eval_count = 0
            if self.is_main_process:
                self.logger.log_cycle_start(cycle, self._get_current_learning_rate())
                print(f"\n### Cycle {cycle + 1}/{num_cycles}")

            for task_idx, task in enumerate(self.tasks):
                # Resume: skip earlier tasks within the resumed cycle.
                if cycle == start_cycle and task_idx < start_cycle_task_index:
                    continue

                self._cycle_task_index = task_idx

                if self.is_main_process:
                    print(f"\n  Task: {task.head_type.value}")
                    num_train_batches = math.ceil(
                        len(task.train_dataset) / self.batch_size  # type: ignore[arg-type]
                    )
                    num_val_batches = math.ceil(
                        len(task.val_dataset) / self.val_batch_size  # type: ignore[arg-type]
                    )
                    self.logger.log_task_start(
                        cycle,
                        task.head_type.value,
                        num_train_batches,
                        num_val_batches,
                    )

                for hook in task.pre_cycle_hooks:
                    self._run_hook_with_logging(
                        hook,
                        HookPhase.PRE_CYCLE,
                        task,
                        cycle,
                        num_cycles,
                        start_cycle=start_cycle,
                    )

                # Mid-task resume only applies to the resumed (cycle, task) pair.
                start_step = (
                    start_cycle_step
                    if cycle == start_cycle and task_idx == start_cycle_task_index
                    else 0
                )
                # If the resumed start is at or past the budget, this task's
                # work is already done — skip ``_train_for_steps`` entirely
                # rather than entering an immediate-terminate replay.
                # Entering the loop would call ``iter(loader)`` to build
                # dataloader iterators which consumes RNG, diverging from an
                # uninterrupted run that only consumes RNG when actually
                # training. Post-cycle hooks still run so checkpoints save.
                if start_step >= budget.max_train_steps:
                    for hook in task.post_cycle_hooks:
                        self._run_hook_with_logging(
                            hook, HookPhase.POST_CYCLE, task, cycle, num_cycles
                        )
                    continue

                task_budget = StepBudget(
                    max_train_steps=budget.max_train_steps,
                    eval_every_steps=budget.eval_every_steps,
                    max_eval_steps=budget.max_eval_steps,
                    start_step=start_step,
                )
                self._train_for_steps(
                    [task],
                    task_budget,
                    cycle=cycle,
                    num_cycles=num_cycles,
                    dataloader_kwargs=dataloader_kwargs,
                )

                for hook in task.post_cycle_hooks:
                    self._run_hook_with_logging(
                        hook,
                        HookPhase.POST_CYCLE,
                        task,
                        cycle,
                        num_cycles,
                        start_cycle=start_cycle,
                    )

            cycle_time_s = time.perf_counter() - cycle_start
            if self.is_main_process:
                self.logger.log_cycle_end(cycle, cycle_time_s, self.best_val_loss)

            # Clean-boundary invariant: after a cycle completes cleanly,
            # ``cycle_step`` and ``cycle_task_index`` must read as 0 so
            # ``resume_training`` can tell "cycle just finished, resume at
            # the next one" from "mid-task, resume where we stopped". Any
            # save between cycles observes a clean (0, 0) snapshot.
            self._cycle_step = 0
            self._cycle_task_index = 0

            # Reset optimizer between cycles.
            if cycle < num_cycles - 1:
                self._reset_optimizer_with_adjusted_lr(cycle + 1, num_cycles)

        total_time_s = time.perf_counter() - training_start
        if self.is_main_process:
            self.logger.log_training_end(total_time_s, self.best_val_loss)

        return self.model

    def _create_dataloaders_for_tasks(
        self,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        tasks: Optional[List["Task"]] = None,
    ) -> Tuple[List[DataLoader], List[DataLoader]]:
        """Create train and validation DataLoaders for the given tasks.

        Called once per ``_train_for_steps`` invocation to build fresh
        loaders for the task list. Train loaders use ``self.batch_size``,
        overridden per task by ``Task.batch_size`` when set; val loaders
        always use ``self.val_batch_size`` for every task (the replay
        effect ``Task.batch_size`` addresses is training-side only).

        Args:
            dataloader_kwargs: Optional kwargs for DataLoader.
            tasks: Tasks to create loaders for. Defaults to ``self.tasks``.

        Returns:
            Tuple of (train_loaders, val_loaders), each a list parallel to tasks
        """
        tasks = tasks if tasks is not None else self.tasks
        kwargs = dataloader_kwargs or {}

        train_loaders = self._create_dataloaders_with_collators(
            [task.train_dataset for task in tasks],
            shuffle=True,
            batch_size=self.batch_size,
            dataloader_kwargs=kwargs,
            collate_fns=[task.collate_fn for task in tasks],
            worker_init_fns=[task.worker_init_fn for task in tasks],
            samplers=[task.sampler for task in tasks],
            max_tokens_per_batch=self.max_tokens_per_batch,
            batch_sizes=[task.batch_size for task in tasks],
            prefetch_window=self.gradient_accumulation_steps,
        )
        if self.is_main_process and any(task.batch_size is not None for task in tasks):
            for task, loader in zip(tasks, train_loaders):
                if hasattr(loader, "__len__"):
                    effective_batch_size = _effective_batch_size(
                        task.batch_size, self.batch_size
                    )
                    print(
                        f"  {task.log_key}: batch_size={effective_batch_size}, "
                        f"~{len(loader)} steps/epoch"
                    )

        val_loaders = self._create_dataloaders_with_collators(
            [task.val_dataset for task in tasks],
            shuffle=False,
            batch_size=self.val_batch_size,
            dataloader_kwargs=kwargs,
            collate_fns=[task.collate_fn for task in tasks],
            worker_init_fns=[task.worker_init_fn for task in tasks],
            samplers=[task.val_sampler for task in tasks],
            max_tokens_per_batch=self.max_tokens_per_batch,
        )

        return train_loaders, val_loaders

    def multi_cycle_train_joint(
        self,
        num_cycles: int = 2,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        start_cycle: int = 0,
        start_cycle_step: int = 0,
    ) -> CoreModel:
        """Multi-cycle joint training across all tasks.

        Each cycle runs ``pre_cycle`` hooks across every task, executes one
        ``_train_for_steps(self.tasks, ...)`` invocation, then runs
        ``post_cycle`` hooks. Best-model + patience tracking is per-cycle.
        ``cycle_task_index`` is always 0 for joint mode (the whole task list
        is processed per step).

        Note: joint mode has no ``start_cycle_task_index`` parameter by
        design — task indices are not meaningful here. If you loaded a
        :class:`ResumeResult` from a sequential-mode checkpoint and call
        this method instead of :meth:`multi_cycle_train`, the saved
        ``start_cycle_task_index`` is silently ignored. Always resume in
        the same mode the checkpoint was saved in.

        Args:
            num_cycles: Number of training cycles.
            dataloader_kwargs: Optional kwargs for DataLoader.
            start_cycle: Cycle to start from when resuming (0-indexed).
            start_cycle_step: Optimizer step to resume at within
                ``start_cycle`` (0 for resume at cycle boundary).

        Returns:
            Trained model (restored to best validation loss).
        """
        self._warn_if_dataset_sizes_imbalanced()
        budget = self._require_step_budget()

        training_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.is_main_process:
            self.logger.log_training_start(
                num_cycles,
                budget.max_train_steps,
                budget.eval_every_steps,
                batch_size=self.effective_batch_size,
                per_device_batch_size=self.batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                num_processes=self.backend.num_processes,
            )

        # Skip the cycle-boundary optimizer reset on mid-cycle resume.
        if start_cycle > 0 and start_cycle_step == 0:
            self._reset_optimizer_with_adjusted_lr(start_cycle, num_cycles)

        for cycle in range(start_cycle, num_cycles):
            self._current_cycle = cycle
            self._cycle_task_index = 0
            self._eval_count = 0
            cycle_start = time.perf_counter()
            if self.is_main_process:
                self.logger.log_cycle_start(cycle, self._get_current_learning_rate())
                print(f"\n### Cycle {cycle + 1}/{num_cycles} (Joint Training)")

            self._run_all_task_hooks(
                phase=HookPhase.PRE_CYCLE,
                cycle_count=cycle,
                num_cycles=num_cycles,
                start_cycle=start_cycle,
            )

            start_step = start_cycle_step if cycle == start_cycle else 0
            cycle_budget = StepBudget(
                max_train_steps=budget.max_train_steps,
                eval_every_steps=budget.eval_every_steps,
                max_eval_steps=budget.max_eval_steps,
                start_step=start_step,
            )
            self._train_for_steps(
                self.tasks,
                cycle_budget,
                cycle=cycle,
                num_cycles=num_cycles,
                dataloader_kwargs=dataloader_kwargs,
            )

            self._run_all_task_hooks(
                phase=HookPhase.POST_CYCLE,
                cycle_count=cycle,
                num_cycles=num_cycles,
                start_cycle=start_cycle,
            )

            cycle_time_s = time.perf_counter() - cycle_start
            if self.is_main_process:
                self.logger.log_cycle_end(cycle, cycle_time_s, self.best_val_loss)

            # Clean-boundary invariant (see ``multi_cycle_train``): reset so
            # a post-cycle save reads as ``cycle_step=0`` and
            # ``resume_training`` correctly identifies it as a cycle
            # boundary rather than a mid-cycle interrupt.
            self._cycle_step = 0
            self._cycle_task_index = 0

            if cycle < num_cycles - 1:
                self._reset_optimizer_with_adjusted_lr(cycle + 1, num_cycles)

        total_time_s = time.perf_counter() - training_start
        if self.is_main_process:
            self.logger.log_training_end(total_time_s, self.best_val_loss)

        return self.model

    def _step_per_step_scheduler(self) -> None:
        """Step the scheduler if it is a per-step scheduler.

        Per-step schedulers (SequentialLR, PolynomialLR,
        CosineAnnealingWarmRestarts) take no required args.
        """
        if self._is_per_step_scheduler and self.scheduler is not None:
            assert not isinstance(self.scheduler, ReduceLROnPlateau)
            self.scheduler.step()

    def _require_step_budget(self) -> StepBudget:
        """Return ``self.step_budget`` or raise a clear error if unset."""
        if self.step_budget is None:
            raise ValueError(
                "Trainer was constructed without a step_budget. "
                "Pass step_budget=StepBudget(...) to Trainer.__init__ "
                "(see dasm2.training.factory.create_trainer)."
            )
        return self.step_budget

    def _train_for_steps(
        self,
        tasks: List["Task"],
        budget: StepBudget,
        cycle: int,
        num_cycles: int,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Step-based inner loop shared by sequential and joint training.

        Drives a single ``while True`` over optimizer steps. Termination
        fires at the top of every iteration, so the optimizer state on exit
        is always at a clean step boundary. Validation runs every
        ``budget.eval_every_steps`` plus a terminal eval at the final step;
        best-model bookkeeping, plateau scheduler stepping, patience update,
        early-stopping check, and ``on_eval`` hook dispatch all happen
        inside that boundary.

        Args:
            tasks: ``[task]`` for sequential mode, ``self.tasks`` for joint.
            budget: Step budget for this invocation (per-task in sequential
                mode, per-cycle in joint mode).
            cycle: Current cycle (0-indexed) for log/hook context.
            num_cycles: Total cycles for log/hook context.
            dataloader_kwargs: Optional DataLoader kwargs.
        """
        is_joint = len(tasks) > 1
        # On mid-task resume (budget.start_step > 0), best-model / patience
        # tracking was just rehydrated from the training state dict; zeroing
        # it would drop a global best observed before the save point and
        # break determinism against an uninterrupted run.
        if budget.start_step == 0:
            self._reset_best_model_tracking()

        # Building loaders can draw from the global RNG (Accelerate seeds a
        # shuffled loader's generator from it), so a pending mid-pass resume
        # is taken, with the save-time RNG state, before they exist.
        resume_position = self._take_resume_pass_position(budget)
        train_loaders, val_loaders = self._create_dataloaders_for_tasks(
            dataloader_kwargs, tasks=tasks
        )
        # Apply the current loader_cycle as DDP shuffle seed on freshly-built
        # loaders so per-task / per-cycle shuffling stays monotonic.
        for loader in train_loaders:
            _set_loader_epoch(loader, self._loader_cycle)

        window = StepWindowAccumulator()
        # Cache the longest-loader index once per invocation: the loader list is
        # fixed for the lifetime of this call, and _draw_micro_batch fires on
        # every micro-batch. When sizes are unknown (iterable datasets without
        # __len__) argmax falls back to index 0, so the first-listed loader
        # drives _loader_cycle advancement — ordering matters in that case.
        loader_lens = [
            len(loader) if hasattr(loader, "__len__") else 0  # type: ignore[arg-type]
            for loader in train_loaders
        ]
        longest_idx = max(range(len(train_loaders)), key=lambda i: loader_lens[i])
        train_iters: List[Any] = []
        self._cycle_step = budget.start_step
        optimizer_steps_per_dataset_pass = self._estimate_optimizer_steps_per_pass(
            train_loaders[longest_idx]
        )

        task_names = ", ".join(t.head_type.value for t in tasks)
        desc = f"Step-train [{task_names}]" if is_joint else f"Train {task_names}"
        self.model.train()

        # Diagnostics: enable faulthandler so a fatal abort — e.g.
        # the NCCL watchdog's SIGABRT on a collective timeout — dumps every
        # thread's stack, and optionally arm a per-step stall watchdog when
        # DASM2_STEP_WATCHDOG_SECONDS is set. Both are no-ops in the common case.
        if not faulthandler.is_enabled():
            faulthandler.enable()
        diagnostics = TrainerDiagnostics.from_env()
        watchdog_seconds = diagnostics.step_watchdog_seconds
        long_batch_codons = diagnostics.long_batch_log_codons
        capture_detail_codons = diagnostics.capture_window_detail_codons
        capture_steps = diagnostics.capture_window_steps
        if self.is_main_process:
            if watchdog_seconds is not None:
                print(f"  Step stall watchdog armed: {watchdog_seconds:g}s/step")
            if long_batch_codons is not None:
                print(
                    f"  Long-batch forensics: log windows ≥ {long_batch_codons} codons"
                )
            if capture_detail_codons is not None:
                gate = (
                    f" (cycle {capture_steps[0]} steps "
                    f"{capture_steps[1]}-{capture_steps[2]})"
                    if capture_steps is not None
                    else ""
                )
                print(
                    "  Window-detail capture: per-row dump for sequences ≥ "
                    f"{capture_detail_codons} valid codons or high parent-N{gate}"
                )
            if self.num_dataloader_workers > 0:
                print(
                    f"  DataLoader: workers={self.num_dataloader_workers} "
                    f"pin_memory={self._is_cuda_device and self.dataloader_pin_memory}"
                )

        try:
            train_iters = (
                self._replay_pass_position(
                    train_loaders, val_loaders, longest_idx, resume_position
                )
                if resume_position is not None
                else self._begin_pass(train_loaders)
            )
            # ``_data_passes`` is the float fractional analogue of the legacy
            # epoch counter. We approximate it as ``loader_cycle +
            # steps_into_current_pass / optimizer_steps_per_dataset_pass``.
            # After a mid-pass resume the longest loader has already yielded
            # ``_train_iter_draws[longest_idx]`` micro-batches of this pass;
            # round up because a window the pass began inside counts as one.
            steps_into_current_pass = math.ceil(
                self._train_iter_draws[longest_idx] / self.gradient_accumulation_steps
            )
            with tqdm(
                total=budget.max_train_steps,
                initial=self._cycle_step,
                desc=desc,
                dynamic_ncols=True,
                disable=not self.is_main_process,
            ) as pbar:
                while True:
                    if self._cycle_step >= budget.max_train_steps:
                        break
                    stop_reason = self._should_stop_early()
                    if stop_reason is not None:
                        if self.is_main_process:
                            self.logger.log_early_stopping(
                                cycle,
                                self._cycle_step,
                                self._global_step,
                                stop_reason,
                                self._get_current_learning_rate(),
                            )
                            print(f"  Early stopping: {stop_reason}")
                        break

                    # Draw one full accumulation window (G micro-batches) *before*
                    # any forward pass, so each task's window-total label count is
                    # known before its gradient is computed — the precondition for
                    # per-task normalization. The window always closes
                    # at exactly ``gradient_accumulation_steps`` micro-batches, so
                    # every iteration steps the optimizer (no partial windows).
                    # Arm the per-step stall watchdog around the whole iteration —
                    # data draw included, since a freeze can sit in data loading
                    #. Cancelled once the window completes below.
                    if watchdog_seconds is not None:
                        faulthandler.dump_traceback_later(watchdog_seconds, repeat=True)
                    window_start_time = time.perf_counter()
                    window_microbatches: List[List[Any]] = []
                    for _ in range(self.gradient_accumulation_steps):
                        prev_loader_cycle = self._loader_cycle
                        batches, train_iters = self._draw_micro_batch(
                            train_iters, train_loaders, longest_idx
                        )
                        if self._loader_cycle != prev_loader_cycle:
                            # Loader exhausted: restart fractional counter so
                            # _data_passes = loader_cycle + steps_since_exhaustion / pass.
                            steps_into_current_pass = 0
                        window_microbatches.append(batches)

                    # Data-load time for this window (the _draw_micro_batch loop above),
                    # measured separately from forward+backward so the forensics can
                    # tell a slow data draw apart from slow compute. When a
                    # stall leaves GPU memory flat, whether the time is
                    # host-side (data) or device-side (forward) is decisive.
                    data_load_ms = (time.perf_counter() - window_start_time) * 1000

                    # Long-batch forensics: flushed, per-rank, BEFORE the
                    # forward, so a heavy window that SIGSEGVs one rank still leaves a
                    # record of its padded length + that rank's CUDA memory.
                    if long_batch_codons is not None:
                        max_seqlen = _window_max_seqlen(window_microbatches)
                        if max_seqlen >= long_batch_codons:
                            mem = ""
                            if self._is_cuda_device:
                                mem = (
                                    f" cuda_alloc={torch.cuda.memory_allocated() / 1e6:.0f}MB"
                                    f" reserved={torch.cuda.memory_reserved() / 1e6:.0f}MB"
                                    f" max_alloc={torch.cuda.max_memory_allocated() / 1e6:.0f}MB"
                                )
                            print(
                                f"[long-batch] rank={self.backend.process_index} "
                                f"cycle={cycle} step={self._cycle_step + 1} "
                                f"max_seqlen={max_seqlen} data_load_ms={data_load_ms:.0f}"
                                f"{mem}",
                                flush=True,
                            )

                    # Per-sequence window detail: one flushed line per
                    # suspect row (long or high-parent-N), pre-forward, so the exact
                    # PCPs in a crashing window are captured even if the step dies.
                    # Gated to the crash window (when DASM2_CAPTURE_WINDOW_STEPS is
                    # set) so the flushed I/O does not perturb the timing it measures.
                    step_1indexed = self._cycle_step + 1
                    in_capture_window = capture_steps is None or (
                        cycle == capture_steps[0]
                        and capture_steps[1] <= step_1indexed <= capture_steps[2]
                    )
                    if capture_detail_codons is not None and in_capture_window:
                        for rec in _window_microbatch_detail(
                            window_microbatches, capture_detail_codons
                        ):
                            print(
                                f"[window-row] rank={self.backend.process_index} "
                                f"cycle={cycle} step={self._cycle_step + 1} "
                                f"micro={int(rec['micro'])} row={int(rec['row'])} "
                                f"padded={int(rec['padded'])} valid={int(rec['valid'])} "
                                f"loss={int(rec['loss'])} "
                                f"parent_n_frac={rec['parent_n_frac']:.3f} "
                                f"bl={rec['bl']:.4f}",
                                flush=True,
                            )

                    per_task, total_loss_sum, total_n = self._process_window(
                        window_microbatches, tasks
                    )
                    window.add(total_loss_sum, total_n, per_task)
                    if watchdog_seconds is not None:
                        faulthandler.cancel_dump_traceback_later()

                    self._cycle_step += 1
                    self._global_step += 1
                    steps_into_current_pass += 1
                    self._data_passes = (
                        float(self._loader_cycle)
                        + steps_into_current_pass / optimizer_steps_per_dataset_pass
                    )
                    pbar.update(1)

                    step_time_ms = (time.perf_counter() - window_start_time) * 1000
                    mean_loss, per_task_mean = window.flush_step_metrics()
                    if self.is_main_process:
                        self._log_step_metrics(
                            cycle,
                            tasks,
                            is_joint,
                            mean_loss,
                            per_task_mean,
                            step_time_ms,
                        )

                    is_cadence = self._cycle_step % budget.eval_every_steps == 0
                    is_terminal = self._cycle_step == budget.max_train_steps
                    if is_cadence or is_terminal:
                        self._run_eval_boundary(
                            cycle,
                            num_cycles,
                            budget,
                            tasks,
                            is_joint,
                            val_loaders,
                            window,
                            pbar,
                        )
                    self._dispatch_step_hooks(cycle, num_cycles, tasks)

            self._restore_best_model()
        finally:
            # Shut down this cycle's DataLoader workers and drop every
            # reference to the loaders/iterators before the next cycle
            # rebuilds them, even if the loop raised. Without this,
            # persistent-worker loaders survive across cycles (the persistent
            # Accelerator retains each prepared loader, so the iterators are
            # never GC'd and their workers never exit), leaking
            # ~workers-per-rank each cycle until the host OOMs.
            self._teardown_loaders(train_loaders, val_loaders, train_iters)

    def _take_resume_pass_position(self, budget: StepBudget) -> Optional[_PassPosition]:
        """Consume the pass position ``restore_training_state`` left.

        Returns it, with the current (save-time) RNG state, only for the first
        invocation after a mid-task resume (``budget.start_step > 0``). Every
        other invocation starts its loaders fresh, exactly as an uninterrupted
        run does, so a position restored from a cycle- or task-boundary save
        is dropped. A checkpoint saved before mid-pass resume existed has
        no position, so
        its pass restarts from batch 0, as every resume did before; this warns.
        """
        pending = self._resume_pass_position
        self._resume_pass_position = None
        if pending is None or budget.start_step == 0:
            return None
        draws, pass_start_rng = pending
        if draws is None or pass_start_rng is None:
            warnings.warn(
                "Checkpoint predates mid-pass resume: it holds "
                "no position inside its data pass, so the pass restarts "
                "from its first batch and the resumed run will not match "
                "an uninterrupted one.",
                stacklevel=2,
            )
            return None
        return _PassPosition(draws, pass_start_rng, TrainingState.capture_rng_states())

    def _begin_pass(self, train_loaders: List[DataLoader]) -> List[Any]:
        """Start a data pass: snapshot its RNG, zero the draw counts, build iterators.

        The snapshot must precede ``iter``, because samplers draw their order
        from these streams when iteration begins.
        """
        self._pass_start_rng_states = _capture_pass_start_rng(train_loaders)
        self._check_ranks_share_pass_rng(train_loaders)
        self._train_iter_draws = [0] * len(train_loaders)
        return [iter(loader) for loader in train_loaders]

    def _check_ranks_share_pass_rng(self, train_loaders: List[DataLoader]) -> None:
        """Under DDP, raise if ranks disagree on an RNG that orders this pass.

        Accelerate deals one batch list out round-robin, so every rank must
        build the same list. A loader ordered by the global torch RNG (an
        unseeded ``ShuffleBufferSampler``) builds it from each rank's own CPU
        stream, which Accelerate never synchronizes: if those streams differ,
        ranks overlap and drop rows, and a resume could not reproduce them.
        Called at every pass start. The all-gather needs every rank there in
        step, which holds when every loader is fixed-size (Accelerate pads
        each rank to the same batch count). A token-budget loader can give
        ranks unequal counts, so the longest loader can run out at different
        steps on different ranks, and the check is skipped.

        Raises:
            RuntimeError: If the ranks' CPU torch RNG states differ.
        """
        if (
            self.backend.num_processes == 1
            or not any(_order_draws_global_rng(loader) for loader in train_loaders)
            or any(_is_token_budget(loader) for loader in train_loaders)
        ):
            return
        digest = hashlib.sha256(torch.random.get_rng_state().numpy().tobytes())
        local = torch.tensor(
            [int.from_bytes(digest.digest()[:8], "little", signed=True)],
            device=self.device,
        )
        gathered = self.backend.all_gather(local)
        if not bool((gathered == gathered[0]).all()):
            raise RuntimeError(
                "Ranks disagree on the CPU torch RNG state at the start of a "
                "data pass, but a train loader takes its batch order from that "
                "stream (an unseeded ShuffleBufferSampler or RandomSampler). "
                "Each rank would shuffle differently, so Accelerate's "
                "round-robin sharding would overlap and drop rows. Something "
                "drew from the CPU RNG a different number of times per rank "
                "(e.g. CPU dropout on unequal batch shapes)."
            )

    def _replay_pass_position(
        self,
        train_loaders: List[DataLoader],
        val_loaders: List[DataLoader],
        longest_idx: int,
        position: _PassPosition,
    ) -> List[Any]:
        """Rebuild the train iterators a saved run had live, partway through a pass.

        Builds them under the saved pass-start RNG state, draws and discards
        each loader's saved number of micro-batches, then puts back the
        save-time RNG state, so dropout and collate randomness continue from
        the save point. Drawing, rather than slicing indices, works for every
        sampler, including those that generate their order as they are
        iterated. No forward pass, hook, eval or step counter sees the
        discarded batches.

        Under DDP every rank discards its own saved count, at its own
        ``loader_cycle`` (``TrainingState.rank_pass_positions``), and restores
        rank 0's RNG snapshot. Accelerate deals one batch list out round-robin,
        which shards correctly only if every rank builds the same list, and
        each rank builds it from state rank 0's snapshot holds:
        ``data_seed + loader_cycle`` under token-budget batching, rank 0's
        broadcast generator for a shuffled loader, and the CPU torch stream
        for ``ShuffleBufferSampler``. That last one must agree across ranks
        for the uninterrupted run to shard correctly, and
        :meth:`_check_ranks_share_pass_rng` raises at a pass start where it
        does not. So each rank lands on its own next batch.

        A persistent-worker loader whose saved iterator had been reset, not
        built, is given a live iterator first, so the rebuild is a reset too:
        a fresh iterator would make a global-RNG draw the saved pass did not.
        Persistent val iterators are created for the same reason, since the
        saved run had evaluated. A save at the very end of a pass skips the
        discard and starts the next pass, as the uninterrupted run would.

        Raises:
            ValueError: If this run built a different number of train loaders
                than the checkpoint saved, or a loader runs out before its
                saved count.
        """
        if len(position.draws) != len(train_loaders):
            raise ValueError(
                f"Checkpoint holds a pass position for {len(position.draws)} "
                f"train loader(s) but this run built {len(train_loaders)}; "
                "resume with the tasks the checkpoint was trained on."
            )
        # The saved run had evaluated, so its persistent val iterators exist
        # and its next eval resets them without a global-RNG draw.
        _create_persistent_iterators(val_loaders)
        longest_len = (
            len(train_loaders[longest_idx])  # type: ignore[arg-type]
            if hasattr(train_loaders[longest_idx], "__len__")
            else 0
        )
        if (
            longest_idx == 0
            and longest_len > 0
            and position.draws[longest_idx] == longest_len
            and all(g is None for g in position.pass_start_rng["loader_generators"])
        ):
            # Saved at the end of a pass: the uninterrupted run's next draw
            # finds the longest loader exhausted and starts the next pass
            # (see _draw_micro_batch), so do that instead of discarding a
            # whole pass. Only when no loader is drawn before the longest,
            # since those would draw once more from their old iterators, and
            # no loader has an Accelerate generator, whose end-of-pass state
            # only the replay reaches.
            _create_persistent_iterators(train_loaders)
            TrainingState.restore_rng_states(position.save_time_rng)
            self._loader_cycle += 1
            for loader in train_loaders:
                _set_loader_epoch(loader, self._loader_cycle)
            if self.is_main_process:
                print("  Pass-boundary resume: starting the next pass")
            return self._begin_pass(train_loaders)
        discard_start = time.perf_counter()
        # Iterators the saved pass reset must be reset here too, not built
        # fresh: a fresh one draws from the global RNG before its sampler does.
        _create_persistent_iterators(
            [
                loader
                for loader, reset in zip(
                    train_loaders, position.pass_start_rng["reset_iterators"]
                )
                if reset
            ]
        )
        _restore_pass_start_rng(position.pass_start_rng, train_loaders)
        try:
            train_iters = [iter(loader) for loader in train_loaders]
            for i, (it, n) in enumerate(zip(train_iters, position.draws)):
                for drawn in range(n):
                    try:
                        next(it)
                    except StopIteration:
                        raise ValueError(
                            f"Train loader {i} ran out after {drawn} micro-batches "
                            f"while replaying the {n} its saved pass had consumed; "
                            "resume with the data the checkpoint was trained on."
                        ) from None
        finally:
            TrainingState.restore_rng_states(position.save_time_rng)
        if self.is_main_process:
            print(
                f"  Mid-pass resume: discarded {sum(position.draws)} micro-batches "
                f"({position.draws} per loader) in "
                f"{time.perf_counter() - discard_start:.2f}s"
            )
        self._train_iter_draws = list(position.draws)
        self._pass_start_rng_states = position.pass_start_rng
        return train_iters

    def _teardown_loaders(
        self,
        train_loaders: List[DataLoader],
        val_loaders: List[DataLoader],
        train_iters: List[Any],
    ) -> None:
        """Shut down per-cycle DataLoader workers and drop all references.

        Called at the end of every ``_train_for_steps`` invocation so the
        next cycle starts clean. Three references must be dropped for the
        workers to actually exit:

        1. The dataloader iterators (``train_iters`` plus each loader's
           ``_iterator`` — val loaders acquire one on first eval, and
           ``persistent_workers=True`` keeps it alive after iteration ends).
           Calling ``_shutdown_workers()`` joins the worker processes.
        2. The loader's own ``_iterator`` back-reference.
        3. Accelerate's retained references (``Accelerator._dataloaders``),
           via the backend, so the loader objects can be collected.

        ``gc.collect()`` then reaps the now-unreferenced loaders. This is the
        fix for the cross-cycle worker leak.
        """
        all_loaders = list(train_loaders) + list(val_loaders)
        # Collect every live iterator: the explicit train iterators and any
        # iterator a loader is still holding (deduplicated by identity, since
        # a persistent-worker loader's _iterator IS its train_iters entry).
        iterators: List[Any] = list(train_iters)
        for loader in all_loaders:
            it = getattr(loader, "_iterator", None)
            if it is not None:
                iterators.append(it)
            if hasattr(loader, "_iterator"):
                loader._iterator = None

        seen: set = set()
        for it in iterators:
            if it is None or id(it) in seen:
                continue
            seen.add(id(it))
            shutdown = getattr(it, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()

        # Drop Accelerate's retained references so the loaders (and their now
        # idle workers) are no longer reachable. No-op on single-GPU.
        self.backend.release_dataloaders(all_loaders)
        gc.collect()

    def _run_eval_boundary(
        self,
        cycle: int,
        num_cycles: int,
        budget: StepBudget,
        tasks: List["Task"],
        is_joint: bool,
        val_loaders: List[DataLoader],
        window: StepWindowAccumulator,
        pbar: "tqdm",
    ) -> None:
        """Evaluate, update best-model + patience, log, fire ``on_eval`` hooks.

        Called once per evaluation boundary inside the step loop. Side
        effects (model restore on improvement, scheduler step, eval-count
        bump) are deliberately bundled here so ``_train_for_steps`` reads
        as a sequence of named operations rather than 60 lines of inline
        bookkeeping.
        """
        eval_start = time.perf_counter()
        with self._eval_mode():
            (
                val_loss,
                per_task_val,
                eval_steps,
                per_task_metrics,
            ) = self._evaluate(val_loaders, tasks, max_eval_steps=budget.max_eval_steps)
        eval_time_s = time.perf_counter() - eval_start

        if is_joint:
            self._warn_if_losses_imbalanced(per_task_val)

        monitored_val_loss = self._monitored_val_loss(val_loss, per_task_val, tasks)
        was_best = self._process_eval_metrics(monitored_val_loss)
        self._step_head_scheduler(tasks, per_task_val)
        # Read first, then bump: hooks see the 0-indexed eval number that
        # just finished (eval_count == 0 for the first eval of a cycle).
        current_eval = self._eval_count
        self._eval_count += 1
        avg_train_loss, avg_per_task_train = window.flush_eval_window()
        # Snapshot per-step optimizer metrics for the window we just closed,
        # then reset so the next window starts clean. The accumulator is
        # only populated on optimizer steps (training mode), so an
        # all-validation entrypoint to ``_run_eval_boundary`` would see
        # ``n_steps == 0`` and the metrics default to zeros.
        optimizer_step_metrics = self._optimizer_step_metrics.compute()
        self._optimizer_step_metrics.reset()
        # Merged into the same dict rather than threaded through as its own
        # logger-signature parameter: ``_log_eval_metrics``
        # already forwards ``optimizer_step_metrics`` to the logger as a
        # flat per-key merge, so any ``combine_gradient_hooks`` diagnostics
        # (e.g. PCGradHook's conflict fraction) ride along for free. Empty
        # when no hook is attached (every existing config).
        if self.combine_gradient_hooks:
            (hook,) = self.combine_gradient_hooks
            optimizer_step_metrics.update(hook.pop_diagnostics())

        # Label the value that drives early stopping, which is not the logged
        # ``val_loss`` (the sum over tasks) when ``monitor_task`` is set.
        val_label = "val" if self.monitor_task is None else "val_monitored"
        pbar.set_postfix(
            {
                "train": f"{avg_train_loss:.4g}",
                val_label: f"{monitored_val_loss:.4g}",
                "lr": self._get_current_learning_rate(),
            }
        )

        if self.is_main_process:
            self._log_eval_metrics(
                cycle,
                tasks,
                is_joint,
                avg_train_loss,
                avg_per_task_train,
                val_loss,
                per_task_val,
                eval_steps,
                eval_time_s,
                was_best,
                per_task_metrics,
                optimizer_step_metrics,
                monitored_val_loss if self.monitor_task is not None else None,
            )

        self._dispatch_eval_hooks(
            cycle,
            num_cycles,
            val_loss,
            monitored_val_loss,
            was_best,
            tasks,
            current_eval,
            per_task_val,
        )

    def _estimate_optimizer_steps_per_pass(self, loader: DataLoader) -> int:
        """Approximate the number of optimizer steps in one pass through ``loader``.

        Used as the denominator of the diagnostic ``_data_passes`` counter.
        Floor division is deliberate: under step-based training every
        accumulation window closes at exactly ``gradient_accumulation_steps``
        micro-batches, so any tail micro-batches at the end of a loader pass
        do not trigger an optimizer step. The fractional ``_data_passes``
        is therefore accurate to within ``1 / optimizer_steps_per_pass``,
        which is cosmetic for a diagnostic field.
        """
        if not hasattr(loader, "__len__"):
            return 1
        loader_len = len(loader)  # type: ignore[arg-type]
        return max(1, loader_len // self.gradient_accumulation_steps)

    def _log_step_metrics(
        self,
        cycle: int,
        tasks: List["Task"],
        is_joint: bool,
        mean_loss: float,
        per_task_mean: Dict[str, float],
        step_time_ms: float,
    ) -> None:
        """Emit the per-step ``batch`` (single-task) or ``joint_batch`` event."""
        if is_joint:
            self.logger.log_joint_batch(
                cycle,
                self._cycle_step,
                self._global_step,
                self._data_passes,
                self._loader_cycle,
                mean_loss,
                per_task_mean,
                step_time_ms,
                self.effective_batch_size,
                skipped_steps=self._skipped_steps,
            )
        else:
            self.logger.log_batch(
                cycle,
                self._cycle_step,
                self._global_step,
                self._data_passes,
                self._loader_cycle,
                mean_loss,
                step_time_ms,
                self.effective_batch_size,
                tasks[0].head_type.value,
                skipped_steps=self._skipped_steps,
            )

    def _log_eval_metrics(
        self,
        cycle: int,
        tasks: List["Task"],
        is_joint: bool,
        avg_train_loss: float,
        avg_per_task_train: Dict[str, float],
        val_loss: float,
        per_task_val: Dict[str, float],
        eval_steps: int,
        eval_time_s: float,
        was_best: bool,
        per_task_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        optimizer_step_metrics: Optional[Dict[str, float]] = None,
        monitored_val_loss: Optional[float] = None,
    ) -> None:
        """Emit the per-eval ``eval_end`` (single-task) or ``joint_eval_end`` event.

        ``val_loss`` is the sum of the tasks' weighted validation losses;
        ``monitored_val_loss`` (set only when the trainer has a ``monitor_task``,
        and written to joint events) is the loss early stopping follows, which
        ``was_best`` and the logged ``best_val_loss`` refer to.

        ``common`` carries only the fields shared by both logger methods. The
        ``task`` keyword (single-task only) and the per-task dicts
        (joint only) are passed separately at each call site so a future
        addition to ``common`` cannot silently shadow a branch-specific
        argument.

        ``per_task_metrics`` maps task key to one flat dict
        merging every validation metric group eligible for that task
        (child-codon + selection-factor). In single-task mode
        the sole entry is unwrapped and forwarded as ``val_metrics``; in
        joint mode the per-task mapping is forwarded as-is.
        ``optimizer_step_metrics`` carries the per-window grad-norm /
        clip-fraction summary, forwarded verbatim.
        """
        common: Dict[str, Any] = dict(
            cycle=cycle,
            cycle_step=self._cycle_step,
            global_step=self._global_step,
            data_passes=self._data_passes,
            loader_cycle=self._loader_cycle,
            avg_train_loss=avg_train_loss,
            val_loss=val_loss,
            eval_steps=eval_steps,
            eval_time_s=eval_time_s,
            lr=self._get_current_learning_rate(),
            is_best=was_best,
            best_val_loss=self.best_val_loss,
            peak_memory_mb=self._get_peak_memory_mb(),
        )
        if is_joint:
            self.logger.log_joint_eval_end(
                per_task_train=avg_per_task_train,
                per_task_val=per_task_val,
                per_task_val_metrics=per_task_metrics,
                optimizer_step_metrics=optimizer_step_metrics,
                monitored_val_loss=monitored_val_loss,
                **common,
            )
        else:
            task_key = tasks[0].head_type.value
            val_metrics: Optional[Dict[str, float]] = None
            if per_task_metrics is not None:
                val_metrics = per_task_metrics.get(task_key)
            self.logger.log_eval_end(
                task=task_key,
                val_metrics=val_metrics,
                optimizer_step_metrics=optimizer_step_metrics,
                **common,
            )

    def _draw_micro_batch(
        self,
        train_iters: List[Any],
        train_loaders: List[DataLoader],
        longest_idx: int,
    ) -> Tuple[List[Any], List[Any]]:
        """Pull one micro-batch from every train loader, re-iterating on exhaustion.

        Tracks the longest loader: when it exhausts, ``_loader_cycle``
        advances and every loader receives a fresh ``set_epoch`` so DDP
        shuffles deterministically. Shorter loaders re-iterate mid-pass
        without bumping the counter; they are intentionally allowed to
        complete multiple passes per ``_loader_cycle`` because their
        sample distribution is already uniform across the run, so
        synchronizing their reshuffles with the long-loader boundary
        would buy nothing and would diverge from today's joint-loop
        semantics.

        Side effects: when the longest loader exhausts this method
        mutates ``self._loader_cycle`` and calls ``_set_loader_epoch``
        on every loader. The mutation is batched so all loaders advance
        to the same shuffle seed in lock-step.

        Simultaneous-exhaust corner case: if a shorter loader (lower
        index than ``longest_idx``) hits ``StopIteration`` on the same
        outer call where the longest loader will also exhaust, the
        shorter loader is re-iter'd via the ``else`` branch *before*
        ``longest_idx`` triggers the cycle bump, so its first batch of
        the new pass is drawn under the previous ``_loader_cycle``
        epoch. This is benign in practice (DDP shuffle skew on at most
        one micro-batch per shorter loader per cycle); the next call to
        ``_draw_micro_batch`` reads from the fresh iterators built in
        the longest-loader branch and is back in lock-step.
        """
        batches: List[Any] = []
        for i, (it, loader) in enumerate(zip(train_iters, train_loaders)):
            try:
                batch = next(it)
            except StopIteration:
                if i == longest_idx:
                    self._loader_cycle += 1
                    for ldr in train_loaders:
                        _set_loader_epoch(ldr, self._loader_cycle)
                    train_iters = self._begin_pass(train_loaders)
                    batch = next(train_iters[i])
                else:
                    train_iters[i] = iter(loader)
                    self._train_iter_draws[i] = 0
                    batch = next(train_iters[i])
            self._train_iter_draws[i] += 1
            batches.append(batch)
        return batches, train_iters

    def _evaluate(
        self,
        val_loaders: List[DataLoader],
        tasks: List["Task"],
        *,
        max_eval_steps: Optional[int],
    ) -> Tuple[
        float,
        Dict[str, float],
        int,
        Optional[Dict[str, Dict[str, float]]],
    ]:
        """Run evaluation, returning
        ``(val_loss, per_task_val, eval_steps, per_task_metrics)``.

        Each task is evaluated over its own val loader, independently of the
        others: up to ``max_eval_steps`` batches, or the whole loader if that is
        shorter. A task with a short val set therefore never cuts another task's
        eval short. Each call iterates a *fresh* iterator, so
        when ``max_eval_steps`` caps the eval, the same prefix is sampled every
        time (val samplers are non-shuffled). This keeps the metric
        deterministic across cycles. The loader is **not** re-iterated when it
        ends before the cap.

        ``eval_steps`` (the third return value) is the largest number of batches
        any task evaluated. In joint training the tasks can evaluate different
        numbers of batches, so it is not each task's own count.

        ``per_task_metrics`` maps task key to a single flat dict merging
        every validation metric group eligible for that task:
        child-codon prediction and selection-factor distribution.
        Eligibility is resolved by the capability-driven
        provider registry (:mod:`dasm2.training.metric_providers`) — the
        head's declared output space for SF, the loss's ``predict_codons``
        for codon. The return is ``None`` when no task emits any group.
        """
        # Single source of truth for the per-task dict key. The loss-tally
        # loop and the post-loop metrics gather loop both index into this
        # list so a future rename can never desynchronize the two output
        # dicts. Keys match the state-dict layout in ``model.task_losses``
        # (set by ``build_task_loss_module_dict``).
        task_keys = [task.log_key for task in tasks]

        per_task_loss_sums: Dict[str, float] = {}
        per_task_n_positions: Dict[str, int] = {}
        # Validation metric accumulators per task, gated by head capability
        # / loss extractor. Empty list ⇒ this task emits no metric groups.
        active_per_task: List[List[ActiveMetric]] = [
            build_active_metrics(task, self.device) for task in tasks
        ]
        steps_per_task: Dict[str, int] = {}

        for key, task, loader, actives in zip(
            task_keys, tasks, val_loaders, active_per_task, strict=True
        ):
            steps_per_task[key] = 0
            for batch in itertools.islice(loader, max_eval_steps):
                # Count first, then forward — matches the training-path
                # ``_count_window_labels`` ordering and keeps the
                # ``SelfNormalizingLoss`` CPU-tensor contract affirmative even
                # if ``_compute_batch_loss`` ever starts mutating its input
                # (today ``move_batch_to_device`` returns a *new* dict, but
                # this ordering doesn't depend on that).
                n_pos = self._task_n_labels(task, batch)
                # Same per-task denominator as training: a
                # SelfNormalizingLoss reports its own label unit, so logged
                # train and val losses for the task stay on one scale (and the
                # val loss that drives early stopping is not miscalibrated).
                loss = self._compute_batch_loss(batch, task, actives)
                per_task_loss_sums[key] = per_task_loss_sums.get(key, 0.0) + loss.item()
                per_task_n_positions[key] = per_task_n_positions.get(key, 0) + n_pos
                steps_per_task[key] += 1
        steps = max(steps_per_task.values(), default=0)

        self._require_monitored_labels(steps_per_task, per_task_n_positions)

        per_task_avg = {
            k: (
                per_task_loss_sums[k] / per_task_n_positions[k]
                if per_task_n_positions.get(k, 0) > 0
                else 0.0
            )
            for k in per_task_loss_sums
        }
        per_task_avg_gathered = {
            k: self._gather_val_loss(v) for k, v in per_task_avg.items()
        }
        total_val_loss = sum(per_task_avg_gathered.values())

        # Merge each task's metric groups into one flat dict. Group key
        # spaces are disjoint (``codon_*`` / ``aa_*`` / ``mut*`` vs
        # ``sf_*``), so the merge can never collide.
        per_task_metrics: Optional[Dict[str, Dict[str, float]]] = None
        collected: Dict[str, Dict[str, float]] = {}
        for i, actives in enumerate(active_per_task):
            if not actives:
                continue
            merged: Dict[str, float] = {}
            for active in actives:
                merged.update(self._gather_active_metric(active))
            collected[task_keys[i]] = merged
        if collected:
            per_task_metrics = collected

        return (
            total_val_loss,
            per_task_avg_gathered,
            steps,
            per_task_metrics,
        )

    def _require_monitored_labels(
        self,
        steps_per_task: Dict[str, int],
        per_task_n_positions: Dict[str, int],
    ) -> None:
        """Raise if the monitored task saw no validation labels in this eval.

        Such a task would report a mean of 0.0, which early stopping would read as
        a new best at every eval.
        """
        task_key = self.monitor_task
        if task_key is None or task_key not in steps_per_task:
            return
        if per_task_n_positions.get(task_key, 0) == 0:
            raise ValueError(
                f"monitor_task={task_key!r} had no validation labels in "
                f"{steps_per_task[task_key]} evaluated batches."
            )

    def _monitored_val_loss(
        self,
        val_loss: float,
        per_task_val: Dict[str, float],
        tasks: List["Task"],
    ) -> float:
        """The validation loss that early stopping, best-model tracking and the
        plateau scheduler follow.

        ``monitor_task=None`` keeps ``val_loss``, the sum over tasks of
        ``loss_weight`` x each task's per-label mean, which adds means over
        different units. Otherwise it is that one task's own mean with its
        ``loss_weight`` divided out (``per_task_val`` holds the weighted value; see
        ``OverfitGuardHook.on_eval`` and ``_step_head_scheduler``, which unscale
        the same way), so a change of weight does not rescale
        ``early_stopping_min_delta``. ``_validate_monitor_task`` guarantees a
        positive weight.
        """
        if self.monitor_task is None:
            return val_loss
        task = next((t for t in tasks if t.log_key == self.monitor_task), None)
        if task is None:
            raise ValueError(
                f"monitor_task={self.monitor_task!r} was not evaluated: this eval "
                f"covers only {[t.log_key for t in tasks]}. monitor_task needs joint "
                "training, where every task is evaluated together at each eval; "
                "sequential multi-task training evaluates one task at a time."
            )
        if self.monitor_task not in per_task_val:
            raise ValueError(
                f"monitor_task={self.monitor_task!r} has no validation loss: its "
                "validation loader yielded no batches."
            )
        return per_task_val[self.monitor_task] / task.loss_weight

    def _process_eval_metrics(self, val_loss: float) -> bool:
        """Update best model, step plateau scheduler, return ``was_best``."""
        was_best = self._update_best_model(val_loss)
        if self.scheduler is not None and not self._is_per_step_scheduler:
            assert isinstance(self.scheduler, ReduceLROnPlateau)
            self.scheduler.step(val_loss)
        return was_best

    @staticmethod
    def _distinct_eval_hooks(
        tasks: List["Task"],
    ) -> Iterator[Tuple["Task", "TrainingHook"]]:
        """Each distinct eval hook once, with the first task that carries it."""
        seen_hook_ids: Set[int] = set()
        for task in tasks:
            for hook in task.eval_hooks:
                if id(hook) not in seen_hook_ids:
                    seen_hook_ids.add(id(hook))
                    yield task, hook

    def _dispatch_step_hooks(
        self, cycle: int, num_cycles: int, tasks: List["Task"]
    ) -> None:
        """Fire each distinct eval hook's ``on_step`` once, on every rank.

        Called after the step's evaluation (if any), so a hook that saves here
        captures a state whose evaluation has already run.
        """
        for _, hook in self._distinct_eval_hooks(tasks):
            hook.on_step(
                self,
                cycle_count=cycle,
                num_cycles=num_cycles,
                cycle_step=self._cycle_step,
            )

    def _dispatch_eval_hooks(
        self,
        cycle: int,
        num_cycles: int,
        val_loss: float,
        monitored_val_loss: float,
        was_best: bool,
        tasks: List["Task"],
        eval_count: int,
        per_task_val: Dict[str, float],
    ) -> None:
        """Run every task's eval hooks once per evaluation.

        ``val_loss`` is the logged sum over tasks. ``monitored_val_loss`` is the
        loss that ``was_best`` and early stopping follow (equal to ``val_loss``
        unless ``monitor_task`` is set).

        ``eval_count`` is the 0-indexed sequence number of the just-completed
        evaluation within the current cycle (caller passes the value before
        bumping ``self._eval_count``). ``per_task_val`` is each task's own
        (loss_weight-scaled) validation loss, keyed by ``task.log_key`` — a
        generic, always-available piece of context (not per-task branching
        in the trainer itself) that a hook needing a task's own signal, such
        as ``OverfitGuardHook``, reads directly.
        """
        for task, hook in self._distinct_eval_hooks(tasks):
            hook.on_eval(
                self,
                train_data=task.train_data,
                val_data=task.val_data,
                cycle_count=cycle,
                num_cycles=num_cycles,
                cycle_step=self._cycle_step,
                eval_count=eval_count,
                val_loss=val_loss,
                monitored_val_loss=monitored_val_loss,
                was_best=was_best,
                tasks=tasks,
                per_task_val=per_task_val,
            )

    def _warn_if_dataset_sizes_imbalanced(self) -> None:
        """Warn if task dataset sizes differ by more than 50%.

        When datasets have significantly different sizes, data points from smaller
        datasets will have proportionally more influence on training due to
        resampling. This is a known trade-off, not a bug.
        """
        if len(self.tasks) < 2:
            return

        # Get sizes for datasets that support len()
        sizes: List[int] = []
        for task in self.tasks:
            if hasattr(task.train_dataset, "__len__"):
                sizes.append(len(task.train_dataset))  # type: ignore[arg-type]

        if len(sizes) < 2:
            return

        max_size = max(sizes)
        min_size = min(sizes)

        if min_size > 0 and max_size / min_size > self._DATASET_SIZE_IMBALANCE_RATIO:
            task_sizes = ", ".join(
                f"{task.head_type.value}: {len(task.train_dataset)}"  # type: ignore[arg-type]
                for task in self.tasks
                if hasattr(task.train_dataset, "__len__")
            )
            warnings.warn(
                f"Task dataset sizes differ by more than "
                f"{(self._DATASET_SIZE_IMBALANCE_RATIO - 1) * 100:.0f}% ({task_sizes}). "
                f"Smaller dataset samples will have proportionally more influence "
                f"on training due to resampling.",
                stacklevel=3,
            )

    def _warn_if_losses_imbalanced(self, per_task_losses: Dict[str, float]) -> None:
        """Warn once if per-task loss values differ significantly.

        Large differences in loss magnitudes may indicate that one task is
        dominating the gradient signal. Consider adjusting loss_weight values.

        Args:
            per_task_losses: Dict mapping task keys to loss values
        """
        if self._has_warned_loss_imbalance:
            return

        losses = [v for v in per_task_losses.values() if v > 0]
        if len(losses) < 2:
            return

        max_loss = max(losses)
        min_loss = min(losses)

        if max_loss / min_loss > self._LOSS_IMBALANCE_THRESHOLD:
            self._has_warned_loss_imbalance = True
            warnings.warn(
                f"Per-task losses differ by more than "
                f"{self._LOSS_IMBALANCE_THRESHOLD}× ({per_task_losses}). "
                f"One task may dominate gradient signal. Consider adjusting "
                f"loss_weight values.",
                stacklevel=4,
            )

    def _run_all_task_hooks(
        self,
        phase: HookPhase,
        cycle_count: int,
        num_cycles: int,
        start_cycle: int = 0,
    ) -> None:
        """Run hooks for all tasks.

        Each task's hooks are run with that task's datasets. No deduplication is
        performed - callers are responsible for ensuring hooks are set up correctly.

        Args:
            phase: Hook execution phase (pre_cycle or post_cycle)
            cycle_count: Current cycle number (0-indexed)
            num_cycles: Total number of cycles
            start_cycle: Cycle the trainer started from. Threaded through so
                resume-aware hooks can detect the resume entry point.

        Raises:
            ValueError: If phase is not a known HookPhase.
        """
        hook_attr = {
            HookPhase.PRE_CYCLE: "pre_cycle_hooks",
            HookPhase.POST_CYCLE: "post_cycle_hooks",
        }
        if phase not in hook_attr:
            raise ValueError(f"Unknown hook phase: {phase!r}")

        for task in self.tasks:
            for hook in getattr(task, hook_attr[phase]):
                self._run_hook_with_logging(
                    hook,
                    phase,
                    task,
                    cycle_count,
                    num_cycles,
                    start_cycle=start_cycle,
                )

    def _reset_optimizer_with_adjusted_lr(self, cycle: int, total_cycles: int) -> None:
        """Reset optimizer and scheduler with adjusted learning rate for next cycle.

        Under ReduceLROnPlateau, uses netam's weighted geometric mean approach:
        progressively shifts from equal weighting between current and original
        LR to favoring current LR as cycles progress. This gives a partial LR bump at each cycle boundary,
        helping the model adapt to the new loss landscape after branch length
        re-optimization.

        IMPORTANT: Must reset BOTH optimizer AND scheduler. The scheduler needs to
        point to the new optimizer, otherwise scheduler.step() will adjust the old
        optimizer's learning rate and early stopping will never trigger.

        For per-step schedulers (polynomial, cosine warm restarts), the blend
        is skipped: the full schedule restarts at each cycle at the configured
        LR.
        Such a cycle anneals itself, and one that runs its schedule to the end
        leaves LR at exactly 0, which is not a training outcome to blend from
        total_training_steps represents one cycle's worth of
        optimizer steps (set once at training start), so the schedule spans
        exactly one cycle.

        Args:
            cycle: Target cycle number that will train with the computed LR (0-indexed).
                   Called after cycle N-1 completes to set LR for cycle N.
            total_cycles: Total number of cycles
        """
        if self.optimizer is None:
            return

        new_lr = (
            self.learning_rate
            if self._is_per_step_scheduler
            else self._compute_adjusted_learning_rate(cycle, total_cycles)
        )
        self.optimizer = self._rebuild_optimizer_at_lr(
            optimizer_config=self._reconstruct_optimizer_config(),
            new_lr=new_lr,
            module=self.unwrapped_model,
            # Excludes head_optimizer's own parameters, if any.
            exclude_module=(
                self.head_optimizer_state.module
                if self.head_optimizer_state is not None
                else None
            ),
        )

        # Create new scheduler pointing to the prepared optimizer
        self.scheduler = self._create_scheduler()

        if self.is_main_process:
            print(f"  Cycle {cycle + 1}: Reset optimizer with lr={new_lr:.3e}")

        # Mirror the same LR reset for head_optimizer, seeded
        # from its own initial LR rather than the main optimizer's — the two
        # optimizers' LRs never mix.
        if self.head_optimizer_state is not None:
            head_state = self.head_optimizer_state
            new_head_lr = (
                head_state.learning_rate
                if self._is_per_step_scheduler
                else self._compute_adjusted_learning_rate(
                    cycle,
                    total_cycles,
                    current_lr=head_state.current_lr(),
                    initial_lr=head_state.learning_rate,
                )
            )
            new_head_optimizer = self._rebuild_optimizer_at_lr(
                optimizer_config=head_state.to_optimizer_config(self.backend),
                new_lr=new_head_lr,
                module=head_state.module,
            )
            head_state.replace_optimizer(
                new_head_optimizer, new_learning_rate=new_head_lr
            )
            # Rebuilt fresh (like self.scheduler above): must point at the
            # new head_optimizer, and a new cycle's plateau count should
            # start clean rather than carrying over the last cycle's.
            head_state.replace_scheduler(self._create_head_scheduler())

            if self.is_main_process:
                print(
                    f"  Cycle {cycle + 1}: Reset head_optimizer with "
                    f"lr={new_head_lr:.3e}"
                )

    def _rebuild_optimizer_at_lr(
        self,
        *,
        optimizer_config: OptimizerConfig,
        new_lr: float,
        module: torch.nn.Module,
        exclude_module: Optional[torch.nn.Module] = None,
    ) -> torch.optim.Optimizer:
        """Rebuild an optimizer over ``module``'s params at ``new_lr``, backend-prepared.

        Shared rebuild sequence for both the main-optimizer and
        head_optimizer branches of
        :meth:`_reset_optimizer_with_adjusted_lr`: apply the adjusted LR to
        an otherwise-unchanged config, rebuild the optimizer, and hand it to
        the backend (DDP/accelerate wrap; no-op for ``SingleGPUBackend``).

        Args:
            optimizer_config: Config to adjust and rebuild from (preserves
                weight_decay/betas/optimizer_name).
            new_lr: The adjusted learning rate to apply.
            module: Module whose parameters the new optimizer covers.
            exclude_module: Passed through to :func:`create_optimizer` —
                None for every call except the main optimizer's, which
                excludes head_optimizer's own parameters.

        Returns:
            The new, backend-prepared optimizer.
        """
        adjusted_config = optimizer_config.model_copy(update={"learning_rate": new_lr})
        new_optimizer = create_optimizer(
            module, adjusted_config, exclude_module=exclude_module
        )
        return self.backend.prepare_optimizer(new_optimizer)

    def _compute_adjusted_learning_rate(
        self,
        current_cycle: int,
        total_cycles: int,
        current_lr: Optional[float] = None,
        initial_lr: Optional[float] = None,
    ) -> float:
        """Compute adjusted learning rate using weighted geometric mean.

        Uses netam's approach: progressively shifts from equal weighting
        between current and original LR to favoring current LR as cycles progress.

        Args:
            current_cycle: Cycle number that will use this LR (0-indexed)
            total_cycles: Total number of cycles
            current_lr: Current LR to blend from. Defaults to the main
                optimizer's current LR (``_get_current_learning_rate()``).
                Pass ``head_optimizer``'s current LR here to reuse this same
                geometric-mean schedule for it without mixing
                the two optimizers' LR values together.
            initial_lr: Initial LR to blend toward. Defaults to
                ``self.learning_rate`` (the main optimizer's initial LR).
                Pass ``self.head_learning_rate`` alongside ``current_lr``
                above for the same reason.

        Returns:
            Adjusted learning rate for next cycle
        """
        if current_lr is None:
            current_lr = self._get_current_learning_rate()
        if initial_lr is None:
            initial_lr = self.learning_rate
        weight = 0.5 + current_cycle / (2 * total_cycles)

        assert current_lr > 0, f"Found non-positive current learning rate {current_lr}"
        assert initial_lr > 0, "Found non-positive initial learning rate"

        # Geometric mean: exp(weight * log(current) + (1 - weight) * log(original))
        adjusted_lr: float = np.exp(
            weight * np.log(current_lr) + (1 - weight) * np.log(initial_lr)
        )
        return adjusted_lr

    def shutdown(self) -> None:
        """Release distributed resources held by the trainer's backend.

        Safe to call multiple times and on non-distributed runs. Intended to
        run from the caller's ``finally`` block so NCCL process-group cleanup
        fires on both the success and exception paths.
        """
        self.backend.shutdown()

    def get_model_identifier(self) -> str:
        """Generate model identifier for cache filenames.

        Returns config_name if available, otherwise generates architecture description.

        Returns:
            Model identifier string (e.g., "77k" or "8L256D" or "ESM2-640D")
        """
        from dasm2.serialization.naming import generate_model_identifier

        encoder_config = self.unwrapped_model.to_config().encoder
        return generate_model_identifier(self.config_name, encoder_config)

    @property
    def neutral_rates_source(self) -> Optional[NeutralRatesSource]:
        """Where this run's training rates come from.

        Read off the live training datasets when there are tasks. A Trainer
        built without them (``load_checkpoint``) returns the carried
        ``source_neutral_rates_source``, which is None for a checkpoint that
        predates the field.
        """
        if not self.tasks:
            return self.source_neutral_rates_source
        return combine_neutral_rates_sources(
            neutral_rates_source_of(task.train_dataset) for task in self.tasks
        )

    def _resolve_metadata_launch_info(self) -> SourceLaunchInfo:
        """Launch facts to stamp into the checkpoint's training metadata.

        Normally these are just read off ``self.backend``. The exception is a
        Trainer whose backend was fabricated rather than launched under —
        ``load_checkpoint`` always hands one a ``SingleGPUBackend``, whose
        ``num_processes`` is a literal ``1`` and whose ``mixed_precision`` is
        a literal ``"no"`` — where reading the backend would rewrite a
        world-size-4 bf16 checkpoint's metadata on a load → save round-trip.
        Those callers pass the parsed values as ``source_launch_info`` and it
        wins wholesale — the same carry-through shape
        ``source_task_configs`` uses.

        Carried ``None`` fields are re-emitted as ``None``. A checkpoint
        written before these were recorded says nothing, and "the source
        didn't say" is
        the honest answer; substituting the fabricated backend's ``1``/``"no"``
        would manufacture a fact the source never claimed, and the resume
        validators would then trust it.

        Raises:
            ValueError: If a carried fact contradicts a live multi-process
                backend. No current caller can reach this —
                ``load_checkpoint`` is the only one that carries values and
                it never has a live backend — so this is a trip-wire for a
                future caller that wires both up, kept for the same reason
                :meth:`_validate_metadata_matches_model` keeps its
                (today tautological) heads check: a contradiction should
                fail at write time, not at some later resume.
        """
        live = SourceLaunchInfo(
            num_processes=self.backend.num_processes,
            mixed_precision=self.backend.mixed_precision,
        )
        carried = self.source_launch_info
        if carried is None:
            return live
        # num_processes != 1 is the only available proof that the backend was
        # launched under rather than fabricated: load_checkpoint's stand-in
        # is single-process by construction.
        if live.num_processes != 1:
            contradictions = [
                f"{field}={getattr(carried, field)!r} vs live {getattr(live, field)!r}"
                for field in ("num_processes", "mixed_precision")
                if getattr(carried, field) is not None
                and getattr(carried, field) != getattr(live, field)
            ]
            if contradictions:
                raise ValueError(
                    f"export_metadata was given source_launch_info that "
                    f"contradicts the trainer's live distributed backend "
                    f"({'; '.join(contradictions)}); refusing to guess which "
                    f"launch this run actually ran under. Pass "
                    f"source_launch_info only for a Trainer built with a "
                    f"fabricated backend (load_checkpoint), not one with a "
                    f"live distributed backend."
                )
        return carried

    def _validate_metadata_matches_model(
        self, head_configs: List[HeadConfig], task_configs: List[TaskConfig]
    ) -> None:
        """Refuse to export metadata that contradicts the model being saved.

        The bug class this closes is ``export_metadata`` writing
        a ``heads`` / ``tasks`` view that disagrees with the weights
        ``save_model`` puts next to it. Two shapes of that were live: a head
        owning parameters produced a loud ``RuntimeError: Unexpected key(s)``
        on reload, and ``SelectionFactorHead`` — whose parameters live in the
        shared ``sf_projection`` — reloaded *successfully* into a model with
        no heads and no losses, i.e. a different function, silently.

        Only the ``tasks`` checks can actually fire, because ``heads`` is now
        read off the model and so cannot disagree with it; the ``heads``
        trip-wire is a plain ``assert`` for a future writer who reintroduces
        a hand-built head list. ``loss_config``/``loss_weight`` have no such
        source of truth to be read through from, which is what leaves the
        ``tasks`` block needing validation.

        Raises:
            ValueError: If a task names a head the model doesn't have, or if
                a model with heads would be saved with an empty
                ``tasks: [...]`` block.
        """
        actual = set(self.unwrapped_model.prediction_heads.keys())
        declared = {hc.head_type.value for hc in head_configs}
        assert declared == actual, (
            f"export_metadata would write heads={sorted(declared)} but the "
            f"model has prediction_heads={sorted(actual)}; heads must be "
            f"derived from the model."
        )
        unknown_heads = sorted(
            tc.head_type.value
            for tc in task_configs
            if not self.unwrapped_model.has_head(tc.head_type)
        )
        if unknown_heads:
            raise ValueError(
                f"export_metadata would write tasks for heads "
                f"{unknown_heads} but the model has "
                f"prediction_heads={sorted(actual)}; the saved checkpoint "
                f"would build losses for heads that do not exist."
            )
        if actual and not task_configs:
            raise ValueError(
                f"export_metadata would write an empty tasks: [] block for a "
                f"model with prediction_heads={sorted(actual)}; the saved "
                f"checkpoint would lose every loss_config and loss_weight, "
                f"and any loss-owned parameters in its state_dict would fail "
                f"to reload. Build the Trainer with tasks=[...], or pass "
                f"source_task_configs=[...] if it has no live Tasks (this is "
                f"what load_checkpoint does)."
            )

    def export_metadata(
        self,
        command: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> BaseTrainingMetadata:
        """Export complete metadata for caching and checkpointing.

        Reconstructs TrainingConfig from runtime state and adds training metadata
        (git info, system info, etc.). Returns Pydantic model for type-safe
        serialization.

        Everything that *can* be read off the model is (``encoder``,
        ``heads``, ``sf_projection``); the rest comes from the Trainer's own
        fields, with ``tasks: [...]`` falling back to
        ``source_task_configs`` (see :class:`Trainer`'s constructor for why
        that fallback exists). :meth:`_validate_metadata_matches_model` then
        refuses to emit a ``heads``/``tasks`` view that contradicts the
        model. ``training_metadata.num_processes`` and
        ``training_metadata.mixed_precision`` are two more such non-model
        facts — launch-environment ones — and get the same treatment via
        :meth:`_resolve_metadata_launch_info`.

        Args:
            command: Command line arguments (defaults to sys.argv)
            notes: Optional notes about this training run
                reconstructed TrainingConfig. Only needed for checkpoint metadata

        Returns:
            BaseTrainingMetadata with:
            - training_config: TrainingConfig (model, neutral_model, optimizer, etc.)
            - config_name: Optional[str] - config name if available
            - training_metadata: Git, system, command, and notes
        """
        # Reconstruct the model config from the *model*, not from
        # ``self.tasks``. ``heads`` used to be collected from
        # ``task.head_config`` with a manual ``seen_head_types`` dedup pass,
        # which silently emitted ``heads: []`` for any Trainer built without
        # live Tasks — ``load_checkpoint`` being the one that matters, so a
        # load → save round-trip corrupted the checkpoint with no mutation
        # at all. The model is the single source of truth here and cannot
        # drift from what ``save_model`` writes; ``CoreModel`` also rejects
        # duplicate head types at construction (DESIGN D1), so
        # ``to_config().heads`` is inherently deduplicated and the manual
        # dedup a two-tasks-one-head shape needed is gone. This is the
        # same treatment ``sf_projection`` gets below — and ``loss_weight``
        # is no longer carried on ``BaseHeadConfig``, so the head config
        # carries architecture only either way.
        # ``to_config()`` already returns a ModelConfig of exactly
        # (encoder, heads, sf_projection), so use it as-is rather than
        # rebuilding a field-for-field identical copy. Carrying
        # ``sf_projection`` (e.g. wiggle_beta) matters: it's a model-level
        # attribute, not per-task, and unlike encoder/head weights it's a
        # plain Python float (not a state_dict tensor), so a checkpoint
        # reloaded via create_model_from_config(training_config.model)
        # silently loses it if it isn't in this block. The failure is silent
        # and downstream: scoring code rebuilds the model without wiggle, and
        # the calibration it produces is then wrong rather than absent, so the
        # checkpoint has to carry the value -- the scoring path cannot recover
        # it.
        model_config = self.unwrapped_model.to_config()
        head_configs: List[HeadConfig] = model_config.heads
        # Not a Trainer field: read off ``CoreModel``'s own tracking of
        # whether ``freeze_encoder()`` was called, rather than inspecting
        # ``requires_grad`` directly. requires_grad alone cannot distinguish
        # "this run asked for head-only fine-tuning" from "this encoder
        # always freezes itself": an encoder that freezes at construction
        # time would otherwise get ``freeze_encoder=True`` written into its
        # checkpoint even though ``--freeze_encoder`` was never passed,
        # spuriously conflicting with a later ``--resume
        # --freeze_encoder=False``.
        freeze_encoder = self.unwrapped_model.encoder_frozen_via_freeze_call
        # ``tasks: [...]`` block: one entry per task with its declarative
        # ``loss_config`` and ``loss_weight``. A head with two losses yields
        # two entries sharing one head_type pointer.
        task_configs: List[TaskConfig] = (
            [
                TaskConfig(
                    head_type=task.head_config.head_type,
                    loss_config=task.loss_config,
                    loss_weight=task.loss_weight,
                    batch_size=task.batch_size,
                    eval_hooks=[h.to_config() for h in task.eval_hooks],
                )
                for task in self.tasks
            ]
            if self.tasks
            else list(self.source_task_configs or [])
        )
        self._validate_metadata_matches_model(head_configs, task_configs)
        neutral_config = (
            self.neutral_model.to_config()  # type: ignore[attr-defined]
            if self.neutral_model is not None
            else None
        )
        optimizer_config = self._reconstruct_optimizer_config()

        # Collect all hooks from tasks
        all_pre_hooks = []
        all_post_hooks = []
        for task in self.tasks:
            all_pre_hooks.extend([hook.to_config() for hook in task.pre_cycle_hooks])
            all_post_hooks.extend([hook.to_config() for hook in task.post_cycle_hooks])

        # Build TrainingConfig (top-level). Prefer the configured step budget,
        # but fall back to placeholder values when ``export_metadata`` is
        # called outside a real training run — typically ``load_checkpoint``
        # builds a Trainer for inference with ``tasks=[]`` (its task configs
        # ride along in ``source_task_configs`` instead) and never sets a
        # step_budget. WARNING: the fallback emits ``eval_every_steps ==
        # max_train_steps``, so any consumer reading the saved metadata
        # YAML to drive new training would think the model was evaluated
        # exactly once at the end. For inference-only checkpoints this is
        # harmless because the ``training_config`` block is reconstructed
        # from runtime state for traceability, not used to drive a loop.
        if self.step_budget is not None:
            cfg_max_train_steps = self.step_budget.max_train_steps
            cfg_eval_every_steps = self.step_budget.eval_every_steps
            cfg_max_eval_steps = self.step_budget.max_eval_steps
        else:
            cfg_max_train_steps = max(self._global_step, 1)
            cfg_eval_every_steps = max(self._global_step, 1)
            cfg_max_eval_steps = None
        training_config = TrainingConfig(
            model=model_config,
            neutral_model=neutral_config,
            optimizer=optimizer_config,
            head_optimizer=self._reconstruct_head_optimizer_config(),
            head_optimizer_head_type=(
                self.head_optimizer_state.head_type
                if self.head_optimizer_state is not None
                else None
            ),
            freeze_encoder=freeze_encoder,
            scheduler=self.scheduler_config,
            batch_size=self.batch_size,
            val_batch_size=self.val_batch_size,
            max_train_steps=cfg_max_train_steps,
            eval_every_steps=cfg_eval_every_steps,
            max_eval_steps=cfg_max_eval_steps,
            gradient_clip_val=self.gradient_clip_val,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta,
            monitor_task=self.monitor_task,
            use_mixed_precision=self.use_mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_dataloader_workers=self.num_dataloader_workers,
            max_tokens_per_batch=self.max_tokens_per_batch,
            dataloader_prefetch_factor=self.dataloader_prefetch_factor,
            dataloader_pin_memory=self.dataloader_pin_memory,
            pre_hooks=all_pre_hooks,
            post_hooks=all_post_hooks,
            tasks=task_configs,
        )

        return BaseTrainingMetadata(
            training_config=training_config,
            config_name=self.config_name,
            training_metadata=create_training_metadata_info(
                command=command,
                notes=notes,
                launch_info=self._resolve_metadata_launch_info(),
            ),
            created_at=str(pd.Timestamp.now()),
            source_checkpoint=self.source_checkpoint,
            neutral_rates_source=self.neutral_rates_source,
        )

    def save_checkpoint(
        self,
        output_dir: str,
        notes: Optional[str] = None,
        suffix: Optional[str] = None,
        filename_override: Optional[str] = None,
    ) -> Optional[Path]:
        """Save model checkpoint with rich descriptive filename.

        Generates checkpoint filename from model configuration, head-dataset
        associations, this run's token, a metadata hash, and date. The
        filename format is:
        {config_name}_{HeadType(dataset)-...}[_{suffix}][_{run_token}]_{hash}_{date}.zip

        No-op on non-main processes (only rank 0 saves checkpoints).

        Args:
            output_dir: Directory to save checkpoint (not full filepath)
            notes: Optional notes about this checkpoint
            suffix: Optional suffix marking which save this is, e.g.
                "cycle0_best" for a post-cycle save or "cycle0_step500" for an
                on-eval one
            filename_override: Optional explicit filename (escape hatch for
                backward compatibility or special cases)

        Returns:
            Path to saved checkpoint file, or None on non-main processes

        Examples:
            >>> # Unsuffixed save from a logged run
            >>> path = trainer.save_checkpoint("checkpoints")
            >>> # checkpoints/dasm_8m_SelectionFactorHead-v1tang_run20260127-101530-a1b2c3_a3b2c1d4_20260127.zip

            >>> # Override filename for special cases
            >>> path = trainer.save_checkpoint("checkpoints", filename_override="custom.zip")
            >>> # checkpoints/custom.zip

        Note:
            Configs are reconstructed from actual runtime objects (model, optimizer, etc.)
            rather than being stored redundantly. This ensures single source of truth.
        """
        if not self.is_main_process:
            return None

        metadata = self.export_metadata(notes=notes)
        output_path = self._resolve_checkpoint_path(
            output_dir, metadata, suffix, filename_override
        )
        save_model(self.unwrapped_model, metadata, output_path)
        self._log_checkpoint_saved("Checkpoint", output_path)
        return output_path

    def get_training_state(
        self, rank_pass_positions: Optional[List[Dict[str, Any]]] = None
    ) -> TrainingState:
        """Capture current mutable training state for checkpoint resume.

        Args:
            rank_pass_positions: Every rank's pass position under DDP, from
                :meth:`_gather_rank_pass_positions` (a collective, so the
                caller gathers it on every rank before saving on rank 0).

        Returns:
            TrainingState carrying optimizer, scheduler, step-loop counters
            (``global_step``, ``cycle_step``, ``cycle_task_index``,
            ``eval_count``, ``data_passes``, ``loader_cycle``), patience,
            RNG, best-model tracking state, the pass position, and
            ``rank_pass_positions`` when given.

        Raises:
            ValueError: If no optimizer is configured (nothing to resume).
        """
        if self.optimizer is None:
            raise ValueError("Cannot capture training state: no optimizer configured")

        return TrainingState(
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=(
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            global_step=self._global_step,
            current_cycle=self._current_cycle,
            cycle_step=self._cycle_step,
            cycle_task_index=self._cycle_task_index,
            eval_count=self._eval_count,
            data_passes=self._data_passes,
            loader_cycle=self._loader_cycle,
            data_seed=self._data_seed,
            best_val_loss=self.best_val_loss,
            evals_without_improvement=self.evals_without_improvement,
            best_model_state=(
                copy.deepcopy(self.best_model_state)
                if self.best_model_state is not None
                else None
            ),
            rng_states=TrainingState.capture_rng_states(),
            learning_rate=self.learning_rate,
            head_optimizer_state_dict=(
                self.head_optimizer_state.state_dict()
                if self.head_optimizer_state is not None
                else None
            ),
            head_learning_rate=(
                self.head_optimizer_state.learning_rate
                if self.head_optimizer_state is not None
                else None
            ),
            head_scheduler_state_dict=(
                self.head_optimizer_state.scheduler_state_dict()
                if self.head_optimizer_state is not None
                else None
            ),
            head_optimizer_head_type=(
                self.head_optimizer_state.head_type.value
                if self.head_optimizer_state is not None
                else None
            ),
            train_iter_draws=list(self._train_iter_draws),
            pass_start_rng_states=self._pass_start_rng_states,
            rank_pass_positions=rank_pass_positions,
        )

    def restore_training_state(self, state: TrainingState) -> None:
        """Restore mutable training state from a checkpoint.

        Restores optimizer, scheduler, step-loop counters, RNG states, and
        best-model tracking. The model weights must already be loaded before
        calling this.

        Args:
            state: TrainingState from :meth:`get_training_state` or
                :func:`load_resumable_checkpoint`.

        Raises:
            ValueError: If no optimizer is configured, scheduler state
                mismatches, or the checkpoint was saved at a different world
                size (per its pass positions).
        """
        if self.optimizer is None:
            raise ValueError("Cannot restore training state: no optimizer configured")

        # Fail fast on scheduler mismatch
        if (self.scheduler is None) != (state.scheduler_state_dict is None):
            raise ValueError(
                f"Scheduler mismatch: trainer has scheduler="
                f"{self.scheduler is not None}, checkpoint has scheduler_state="
                f"{state.scheduler_state_dict is not None}"
            )

        # Restore counters
        self._global_step = state.global_step
        self._current_cycle = state.current_cycle
        self._cycle_step = state.cycle_step
        self._cycle_task_index = state.cycle_task_index
        self._eval_count = state.eval_count
        self._data_passes = state.data_passes
        self._loader_cycle = state.loader_cycle
        self._data_seed = state.data_seed
        draws, pass_start_rng = state.train_iter_draws, state.pass_start_rng_states
        if (
            self.backend.num_processes > 1
            and state.rank_pass_positions is None
            and state.train_iter_draws is not None
        ):
            # Saved with a pass position but with no
            # per-rank entries, so by a single process.
            raise ValueError(
                "Checkpoint was saved by a single process but this run has "
                f"{self.backend.num_processes}; resume at the world size the "
                "checkpoint was trained at."
            )
        if state.rank_pass_positions is not None:
            # Each rank takes its own position, which under token-budget
            # batching can differ from rank 0's. The rest of the
            # snapshot is rank 0's: batch order does not depend on it there,
            # and fixed-size ranks stay in step.
            if len(state.rank_pass_positions) != self.backend.num_processes:
                raise ValueError(
                    f"Checkpoint holds pass positions for "
                    f"{len(state.rank_pass_positions)} ranks but this run has "
                    f"{self.backend.num_processes}; resume at the world size "
                    "the checkpoint was trained at."
                )
            mine = state.rank_pass_positions[self.backend.process_index]
            self._loader_cycle = mine["loader_cycle"]
            draws = mine["train_iter_draws"]
            if pass_start_rng is not None:
                pass_start_rng = {
                    **pass_start_rng,
                    "reset_iterators": mine["reset_iterators"],
                }
        self._resume_pass_position = (draws, pass_start_rng)

        # Restore initial learning rate (for geometric mean LR computation)
        self.learning_rate = state.learning_rate

        # head_optimizer: fail fast on presence mismatch, same
        # pattern as the scheduler check above, then restore state/LR.
        if (self.head_optimizer_state is None) != (
            state.head_optimizer_state_dict is None
        ):
            raise ValueError(
                f"head_optimizer mismatch: trainer has head_optimizer="
                f"{self.head_optimizer_state is not None}, checkpoint has "
                f"head_optimizer_state={state.head_optimizer_state_dict is not None}"
            )
        # head_scheduler: same presence-mismatch pattern.
        # Independent of the head_optimizer check above — head_scheduler is
        # None whenever the configured scheduler isn't reduce_on_plateau,
        # even with a head_optimizer present (see _create_head_scheduler).
        current_head_scheduler = (
            self.head_optimizer_state.scheduler
            if self.head_optimizer_state is not None
            else None
        )
        if (current_head_scheduler is None) != (
            state.head_scheduler_state_dict is None
        ):
            raise ValueError(
                f"head_scheduler mismatch: trainer has head_scheduler="
                f"{current_head_scheduler is not None}, checkpoint has "
                f"head_scheduler_state={state.head_scheduler_state_dict is not None}"
            )

        if self.head_optimizer_state is not None:
            # The presence-mismatch checks above already guarantee
            # state.head_optimizer_state_dict is not None here, and that
            # state.head_scheduler_state_dict's presence matches
            # current_head_scheduler's.
            assert state.head_optimizer_state_dict is not None
            # Same parameter shapes don't imply the same head (e.g. two
            # single-linear heads of the same width) — checked explicitly
            # here rather than left to load_state_dict, which would apply
            # the wrong head's optimizer state silently whenever shapes
            # happen to match. None on either side (a
            # checkpoint saved before this field existed) skips the check
            # rather than treating "unknown" as a mismatch.
            checkpoint_head_type = state.head_optimizer_head_type
            trainer_head_type = self.head_optimizer_state.head_type.value
            if (
                checkpoint_head_type is not None
                and checkpoint_head_type != trainer_head_type
            ):
                raise ValueError(
                    "head_optimizer_head_type mismatch: trainer's "
                    f"head_optimizer is configured for {trainer_head_type!r}, "
                    f"but the checkpoint's head_optimizer was trained for "
                    f"{checkpoint_head_type!r}."
                )
            assert state.head_learning_rate is not None
            self.head_optimizer_state.restore(
                optimizer_state_dict=state.head_optimizer_state_dict,
                learning_rate=state.head_learning_rate,
                scheduler_state_dict=state.head_scheduler_state_dict,
            )

        # Restore optimizer state (momentum buffers, second moments for AdamW)
        self.optimizer.load_state_dict(state.optimizer_state_dict)

        # Restore scheduler state
        if self.scheduler is not None and state.scheduler_state_dict is not None:
            self.scheduler.load_state_dict(state.scheduler_state_dict)

        # Restore best model tracking
        self.best_val_loss = state.best_val_loss
        self.evals_without_improvement = state.evals_without_improvement
        self.best_model_state = (
            copy.deepcopy(state.best_model_state)
            if state.best_model_state is not None
            else None
        )

        # Restore RNG states (capture_rng_states always produces a non-empty
        # dict). Under DDP each rank takes its own saved streams when the
        # checkpoint holds them, else rank 0's.
        assert state.rng_states, "TrainingState.rng_states must not be empty"
        rng_states = state.rng_states
        if state.rank_pass_positions is not None:
            rng_states = state.rank_pass_positions[self.backend.process_index][
                "rng_states"
            ]
        TrainingState.restore_rng_states(rng_states)

    def save_resumable(
        self,
        output_dir: str,
        notes: Optional[str] = None,
        suffix: Optional[str] = None,
        filename_override: Optional[str] = None,
    ) -> Optional[Path]:
        """Save resumable checkpoint with full training state.

        Like save_checkpoint() but includes optimizer, scheduler, counters,
        RNG states, and best model tracking so training can resume.

        Under DDP this is a collective: every rank must call it in step,
        because it gathers each rank's pass position and RNG streams before
        the main process writes.

        Args:
            output_dir: Directory to save checkpoint (not full filepath)
            notes: Optional notes about this checkpoint
            suffix: Optional suffix marking which save this is (e.g.,
                "cycle0_best")
            filename_override: Optional explicit filename

        Returns:
            Path to saved checkpoint file, or None on non-main processes
        """
        # Collective: every rank calls this (it is reached from hooks, which
        # run on every rank), before the main-process guard below.
        rank_pass_positions = self._gather_rank_pass_positions()
        if not self.is_main_process:
            return None

        metadata = self.export_metadata(notes=notes)
        training_state = self.get_training_state(rank_pass_positions)
        output_path = self._resolve_checkpoint_path(
            output_dir, metadata, suffix, filename_override
        )
        save_resumable_checkpoint(
            self.unwrapped_model, metadata, training_state, output_path
        )
        self._log_checkpoint_saved("Resumable checkpoint", output_path)
        return output_path

    def _gather_rank_pass_positions(self) -> Optional[List[Dict[str, Any]]]:
        """Under DDP, every rank's pass position and RNG streams, in rank order.

        A token-budget pass can deal ranks unequal batch counts, so ranks run
        out at different steps and their ``loader_cycle``, draw counts and
        RNG streams drift apart; rank 0's are not theirs. Every rank must
        call this in step. Returns None on a single process.
        """
        if self.backend.num_processes == 1:
            return None
        resets = (self._pass_start_rng_states or {}).get(
            "reset_iterators", [False] * len(self._train_iter_draws)
        )
        return self._all_gather_objects(
            {
                "loader_cycle": self._loader_cycle,
                "train_iter_draws": list(self._train_iter_draws),
                "reset_iterators": list(resets),
                "rng_states": TrainingState.capture_rng_states(),
            }
        )

    def _all_gather_objects(self, obj: Any) -> List[Any]:
        """Gather one picklable object from every rank, in rank order.

        Collective: every rank must call it in step. Pickles to bytes and
        pads to the longest rank's length, because ``all_gather`` needs equal
        shapes on every rank.
        """
        payload = np.frombuffer(pickle.dumps(obj), dtype=np.uint8)
        sizes = (
            self.backend.all_gather(torch.tensor([payload.size], device=self.device))
            .cpu()
            .tolist()
        )
        padded = torch.zeros(max(sizes), dtype=torch.uint8)
        padded[: payload.size] = torch.from_numpy(payload.copy())
        rows = (
            self.backend.all_gather(padded.unsqueeze(0).to(self.device)).cpu().numpy()
        )
        return [pickle.loads(row[:size].tobytes()) for row, size in zip(rows, sizes)]

    def _log_checkpoint_saved(self, label: str, path: Path) -> None:
        """Print checkpoint save confirmation (main process only)."""
        print(f"{label} saved: {path}")

    def _resolve_checkpoint_path(
        self,
        output_dir: str,
        metadata: BaseTrainingMetadata,
        suffix: Optional[str] = None,
        filename_override: Optional[str] = None,
    ) -> Path:
        """Build checkpoint output path from metadata and naming config.

        The filename carries this run's token (from the logger's run ID) so a
        second attempt writing into the same directory cannot produce a file
        indistinguishable from the first's.

        Args:
            output_dir: Directory for the checkpoint
            metadata: Training metadata (used for filename hash)
            suffix: Optional suffix (e.g., "cycle0_best")
            filename_override: Explicit filename, bypassing auto-generation

        Returns:
            Full output path for the checkpoint file
        """
        if filename_override is not None:
            filename = filename_override
        else:
            head_dataset_pairs = self._get_head_dataset_pairs()
            filename = build_checkpoint_filename(
                config_name=self.config_name,
                encoder_config=self.unwrapped_model.to_config().encoder,
                head_dataset_pairs=head_dataset_pairs,
                training_config=metadata.training_config,
                suffix=suffix,
                run_token=run_token_from_run_id(self.logger.run_id),
            )
        return Path(output_dir) / filename

    def _get_head_dataset_pairs(self) -> List[Tuple[str, str]]:
        """Extract head-dataset pairs from tasks for checkpoint naming.

        Returns:
            List of (head_type_value, dataset_nickname) tuples
        """
        pairs: List[Tuple[str, str]] = []
        for task in self.tasks:
            head_type_value = task.head_type.value
            nickname = _nickname_of_dataset(task.train_dataset)
            pairs.append((head_type_value, nickname))

        return pairs

    def _reconstruct_optimizer_config(self) -> OptimizerConfig:
        """Reconstruct OptimizerConfig from training configuration.

        Uses the initial learning rate stored at trainer initialization,
        not the current (possibly decayed) learning rate from the optimizer.

        Returns:
            OptimizerConfig matching the original training configuration
        """
        if self.optimizer is None:
            return OptimizerConfig()  # Default config
        return reconstruct_optimizer_config(
            self.backend, self.optimizer, self.learning_rate
        )

    def _reconstruct_head_optimizer_config(self) -> Optional[OptimizerConfig]:
        """Reconstruct head_optimizer's OptimizerConfig.

        Delegates to :meth:`HeadOptimizerState.to_optimizer_config`. Returns
        None when no head_optimizer is configured.

        Returns:
            OptimizerConfig matching head_optimizer's original configuration,
            or None if head_optimizer is not set.
        """
        if self.head_optimizer_state is None:
            return None
        return self.head_optimizer_state.to_optimizer_config(self.backend)

    def try_load_cached(
        self,
        artifact_class: type[TrainingArtifact[Any]],
        dataset: Any,
        split: str,
        head_type: Optional[HeadType] = None,
    ) -> Optional[torch.Tensor]:
        """Try to load cached artifact, return None if not found or invalid.

        Delegates to artifact class's try_load_for_dataset method.

        Args:
            artifact_class: TrainingArtifact subclass (e.g., BranchLengthArtifact)
            dataset: Dataset to load artifact for
            split: Split identifier ("train", "val", "test")
            head_type: Optional HeadType for head-specific artifacts

        Returns:
            Cached tensor if found and valid, None otherwise
        """
        if self.cache is None:
            return None

        return artifact_class.try_load_for_dataset(
            cache_dir=self.cache.cache_dir,
            trainer=self,
            dataset=dataset,
            split=split,
            head_type=head_type,
        )

    def save_cached(
        self,
        artifact_class: type[TrainingArtifact[Any]],
        result: torch.Tensor,
        dataset: Any,
        split: str,
        head_type: Optional[HeadType] = None,
    ) -> None:
        """Save computed result to cache.

        Delegates to artifact class's save_for_dataset method.

        Args:
            artifact_class: TrainingArtifact subclass (e.g., BranchLengthArtifact)
            result: Computed tensor to cache
            dataset: Dataset artifact was computed for
            split: Split identifier ("train", "val", "test")
            head_type: Optional HeadType for head-specific artifacts
        """
        if self.cache is None:
            return

        artifact_class.save_for_dataset(
            cache_dir=self.cache.cache_dir,
            result=result,
            trainer=self,
            dataset=dataset,
            split=split,
            head_type=head_type,
        )


def _nickname_of_dataset(dataset: Any) -> str:
    """Extract nickname from dataset, unwrapping Subset if needed.

    Args:
        dataset: Dataset instance, possibly wrapped in Subset

    Returns:
        Dataset nickname, or "unknown" if not available
    """
    # Unwrap Subset to get underlying dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset

    return getattr(dataset, "nickname", "unknown")
