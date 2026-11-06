"""Compute backend abstraction for single-GPU and distributed training.

The Trainer delegates device management, backward passes, gradient clipping,
and model unwrapping to a ComputeBackend. Implementations hide whether training
runs on a single GPU or across multiple GPUs via accelerate. The Trainer holds
no knowledge of which backend is active — no conditionals, no isinstance checks.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator, List, Literal, Optional

import torch
import torch.distributed as dist

#: Valid mixed precision modes for AccelerateBackend.
#: ``None`` means no mixed precision. ``"fp16"`` and ``"bf16"`` enable
#: half-precision training with the respective floating-point format.
MixedPrecision = Literal["fp16", "bf16"]


def compute_effective_batch_size(
    batch_size: int,
    gradient_accumulation_steps: int,
    num_processes: int,
) -> int:
    """Compute effective batch size across all parallelism dimensions.

    Equals per_device_batch_size x gradient_accumulation_steps x num_processes.

    Args:
        batch_size: Per-device batch size (must be positive).
        gradient_accumulation_steps: Number of micro-batches accumulated per step
            (must be positive).
        num_processes: Number of distributed processes (must be positive;
            1 for single-GPU).

    Returns:
        Effective batch size seen by the optimizer per step.

    Raises:
        ValueError: If any argument is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            f"gradient_accumulation_steps must be positive, got {gradient_accumulation_steps}"
        )
    if num_processes <= 0:
        raise ValueError(f"num_processes must be positive, got {num_processes}")
    return batch_size * gradient_accumulation_steps * num_processes


class ComputeBackend(ABC):
    """Abstract base for compute backends (single-GPU, accelerate, etc.).

    Every operation that differs between single-GPU and distributed training
    lives here. The Trainer calls these methods uniformly.
    """

    @property
    @abstractmethod
    def device(self) -> str:
        """Device string for this process (e.g., 'cuda:0')."""
        ...

    @property
    @abstractmethod
    def model_init_device(self) -> str:
        """Device to use when creating the model before backend preparation.

        For single-GPU, this is the training device itself. For accelerate,
        models should be created on CPU — accelerate.prepare() handles
        device placement.
        """
        ...

    @property
    @abstractmethod
    def is_main_process(self) -> bool:
        """True on rank 0, or always True for single-GPU."""
        ...

    @property
    @abstractmethod
    def process_index(self) -> int:
        """Zero-based rank of this process (0 for single-GPU)."""
        ...

    @property
    @abstractmethod
    def num_processes(self) -> int:
        """Number of processes participating in training."""
        ...

    @property
    @abstractmethod
    def mixed_precision(self) -> str:
        """Backend precision mode, in Accelerate's spelling.

        ``"no"``, ``"fp16"``, or ``"bf16"``. Recorded in checkpoint metadata
        so that resuming at a different precision than the run was trained
        at can be rejected instead of silently changing the numerics
        (issue #1027). Returns ``"no"`` rather than ``None`` when disabled,
        so "off" and "not recorded" stay distinguishable on disk.

        This is *not* ``TrainingConfig.use_mixed_precision``, the trainer's
        own BF16 autocast around the forward pass: that is config-declared
        and rides in ``training_config``, and applies on single-GPU too.

        Typed ``str`` rather than the :data:`MixedPrecision` alias used for
        the ``AccelerateBackend`` constructor parameter, deliberately: that
        alias is the set of *requestable* modes and excludes ``"no"``, which
        is spelled as ``None`` on the way in and ``"no"`` on the way out.
        Widening it to include ``"no"`` still would not fit, because
        ``AccelerateBackend`` reads this back off the live ``Accelerator``,
        where a mode this version doesn't know (e.g. ``"fp8"``) must survive
        rather than need a cast — the same forward-compatibility reason
        ``TrainingMetadataInfo.mixed_precision`` is a plain ``str``.
        """
        ...

    @abstractmethod
    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Strip DDP/FSDP wrapper if present, returning the raw module."""
        ...

    @abstractmethod
    def unwrap_optimizer(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer:
        """Strip any backend wrapper and return the underlying torch optimizer.

        Single-GPU returns the optimizer unchanged. Accelerate wraps optimizers
        in ``AcceleratedOptimizer``; callers that inspect the class name or
        param groups need the inner optimizer to avoid seeing the wrapper.
        """
        ...

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        """Run backward pass (handles loss scaling for mixed precision)."""
        ...

    @abstractmethod
    def clip_grad_norm_(self, parameters: Any, max_norm: float) -> torch.Tensor:
        """Clip gradient norms (handles unscaling for mixed precision)."""
        ...

    @abstractmethod
    def prepare_model_and_optimizer(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        """Prepare model and optimizer for training (DDP wrap, device placement)."""
        ...

    @abstractmethod
    def prepare_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> torch.utils.data.DataLoader:
        """Prepare dataloader (adds DistributedSampler if needed)."""
        ...

    @abstractmethod
    def release_dataloaders(
        self,
        dataloaders: List[torch.utils.data.DataLoader],
    ) -> None:
        """Drop any backend-held references to the given prepared dataloaders.

        The multi-cycle trainer rebuilds its dataloaders every cycle. A
        distributed backend may retain each prepared loader internally (e.g.
        Accelerate appends them to ``Accelerator._dataloaders`` so it can call
        ``set_epoch``), which prevents the prior cycle's loaders — and their
        persistent worker processes — from being garbage-collected. Releasing
        those references here lets the loaders die and their workers shut down,
        avoiding a per-cycle worker leak (issue #724 follow-up).

        Single-GPU holds no such references, so this is a no-op there.
        Implementations must release ONLY the dataloaders, never the prepared
        model/optimizer/scheduler the trainer still needs across cycles.
        """
        ...

    @abstractmethod
    def prepare_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        """Prepare a new optimizer for an already-prepared model.

        Called when resetting the learning rate between training cycles. The
        model is already prepared (DDP-wrapped, device-placed) from the initial
        ``prepare_model_and_optimizer`` call and does not need re-preparation.
        Only the new optimizer needs to be registered with the backend.
        """
        ...

    @abstractmethod
    @contextmanager
    def no_sync(self, model: torch.nn.Module) -> Iterator[None]:
        """Context manager to suppress gradient sync (for accumulation)."""
        ...

    @abstractmethod
    def broadcast(self, tensor: torch.Tensor) -> torch.Tensor:
        """Broadcast a tensor from rank 0 to all processes.

        On single-GPU this is the identity. On multi-GPU, rank 0's tensor
        is sent to every other rank (overwriting their local value).

        All ranks must call this collectively with a tensor of the same shape.

        Args:
            tensor: Tensor to broadcast (rank 0's value is authoritative).

        Returns:
            The broadcast tensor (same on every rank after the call).
        """
        ...

    @abstractmethod
    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather a tensor from every rank and return the concatenation.

        All ranks must call this collectively with tensors of the **same
        shape**: ``accelerate.gather`` requires equal per-rank shapes. For
        unequal-length workloads, the caller pads each rank's tensor to the
        same length with a sentinel and trims after gather. The post-cycle
        BL re-opt uses this pattern at world_size 4 with a 1003-row N.

        On single-GPU this is the identity. On multi-GPU, the returned tensor
        concatenates ranks in rank order along dim 0: ``[rank0..., rank1...,
        ..., rankK-1...]``.

        Args:
            tensor: Per-rank tensor to gather; same shape on every rank.

        Returns:
            Concatenated tensor of shape ``(num_processes * tensor.shape[0],
            *tensor.shape[1:])``.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Release any distributed resources held by this backend.

        Single-GPU is a no-op. Distributed backends must destroy the torch
        process group so NCCL doesn't warn about leaked resources at exit.
        """
        ...

    @abstractmethod
    def gather_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather a scalar (0-d) tensor from all processes and return the mean.

        Used for validation loss aggregation across ranks so that early stopping
        sees a consistent loss regardless of the number of processes.

        The tensor must be 0-dimensional (a scalar produced by e.g.
        ``torch.tensor(value)``).

        On single-GPU, returns the input tensor unchanged. On multi-GPU, uses
        ``accelerator.gather`` (plain all-gather, not ``gather_for_metrics``)
        because the inputs are already per-rank averages, not per-sample
        metrics. ``gather_for_metrics`` applies DistributedSampler padding
        correction that would incorrectly truncate rank entries for scalars.

        Since DistributedSampler ensures each rank processes the same number
        of batches (via padding), the mean-of-per-rank-averages equals the
        global average.

        Raises:
            ValueError: If the tensor is not 0-dimensional.
        """
        ...


class SingleGPUBackend(ComputeBackend):
    """Single-GPU backend — zero-overhead wrapper over plain PyTorch."""

    def __init__(self, device: str) -> None:
        self._device = device

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_init_device(self) -> str:
        return self._device

    @property
    def is_main_process(self) -> bool:
        return True

    @property
    def process_index(self) -> int:
        return 0

    @property
    def num_processes(self) -> int:
        return 1

    @property
    def mixed_precision(self) -> str:
        # No backend-level mixed precision exists here: ``--mixed_precision``
        # is rejected without ``--multi_gpu``, and the trainer's BF16 autocast
        # is the separate, config-declared ``use_mixed_precision``.
        return "no"

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def unwrap_optimizer(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer:
        return optimizer

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def clip_grad_norm_(self, parameters: Any, max_norm: float) -> torch.Tensor:
        total_norm: torch.Tensor = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        return total_norm

    def prepare_model_and_optimizer(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        return model, optimizer

    def prepare_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> torch.utils.data.DataLoader:
        return dataloader

    def release_dataloaders(
        self,
        dataloaders: List[torch.utils.data.DataLoader],
    ) -> None:
        # Single-GPU holds no references to prepared loaders; the trainer's
        # own teardown shuts the workers down. Nothing to release here.
        pass

    def prepare_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        return optimizer

    @contextmanager
    def no_sync(self, model: torch.nn.Module) -> Iterator[None]:
        yield

    def broadcast(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def gather_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 0:
            raise ValueError(
                f"gather_mean expects a scalar tensor, got shape {tensor.shape}"
            )
        return tensor

    def shutdown(self) -> None:
        pass


class AccelerateBackend(ComputeBackend):
    """Multi-GPU backend wrapping HuggingFace Accelerate.

    The ``Accelerator`` is instantiated lazily in ``__init__`` (not at import
    time) to avoid initialization overhead when importing the module.

    Note: We do NOT pass ``gradient_accumulation_steps`` to the Accelerator.
    Our trainer has its own accumulation logic with partial-window scaling and
    NaN fail-fast checks. Instead, we expose ``no_sync()`` so the trainer can
    suppress gradient all-reduce on non-boundary micro-batches.
    """

    def __init__(self, mixed_precision: Optional[MixedPrecision] = None) -> None:
        from datetime import timedelta

        from accelerate import (
            Accelerator,
            DistributedDataParallelKwargs,
            InitProcessGroupKwargs,
        )

        # Accelerate expects the string "no" for disabled mixed precision.
        # Type narrowing via the MixedPrecision Literal enforces valid values
        # at static analysis time; Accelerator validates at runtime.
        #
        # Pass cpu=True when CUDA is unavailable so Accelerate's PartialState
        # correctly detects MULTI_CPU distributed mode (e.g. gloo backend).
        # Without this, PartialState falls through to DistributedType.NO and
        # collective ops like gather become no-ops.
        #
        # broadcast_buffers=False: QDASM/Hybrid with q_variant=global_13mer
        # registers a 1.34 GB read-only Thrifty lookup table as a non-persistent
        # buffer. Default broadcast_buffers=True would sync this from rank 0 on
        # every forward pass, hanging DDP init (initial sync) and adding ~30ms+
        # of NCCL traffic per step. All ranks load the table from the same file
        # so the broadcast is redundant. Models with mutable buffers (e.g.,
        # BatchNorm running stats) should use SyncBatchNorm or restore the flag.
        #
        # find_unused_parameters=True was tried here (issue #1096) to let
        # OverfitGuardHook freeze a co-trained task's parameters
        # (requires_grad=False) mid-training under DDP without DDP raising
        # "parameters were not used in producing loss" on the next backward.
        # Reverted: it broke DDP for any loss-owned parameter used outside the
        # wrapped model's own forward() — e.g. DMSFunctionalScoreLoss.scale,
        # attached to the model via Trainer._register_task_losses but applied
        # in a second, separate call (task.loss(...)) after model.forward()
        # returns. find_unused_parameters=True's traversal only looks
        # backward from forward()'s own output tensors, so it can't see that
        # later usage, eagerly marks the parameter "unused", and then its real
        # backward hook collides with that when the actual gradient arrives
        # ("Expected to mark a variable ready only once" / "marked as ready
        # twice") — broke test_loss_params_ddp and test_resume_ddp, neither of
        # which involves OverfitGuardHook at all. requires_grad=False itself
        # is unaffected and still correctly freezes a task (see
        # OverfitGuardHook); it's just that DDP + OverfitGuardHook firing
        # together is unsupported for now — this repo does not yet run joint
        # training under DDP, so nothing currently depends on that
        # combination. A real fix (e.g. static_graph=True, or something more
        # surgical) is needed before combining the two.
        ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
        # Widen the NCCL/process-group collective timeout to 8h. Two reasons,
        # the first of which sets the floor:
        #   * QDASM/Hybrid re-optimize branch lengths solo on rank 0 between
        #     cycles (and at cycle end); on the 4-dataset combo this pass runs
        #     for hours while ranks 1..N block at the next collective. The
        #     default ~10-min timeout aborts those ranks mid-BL-opt and loses
        #     the cycle's work (observed on the 4ds QDASM-v4 run, 2026-05-08).
        #   * A legitimately slow micro-batch on one rank — e.g. a heavy padded
        #     forward in the long-sequence band — must not be aborted mid-step
        #     by a collective timeout (issue #724).
        # 8h comfortably covers the longest BL-opt pass and subsumes the
        # per-step case. Diagnostics still surface a true stall via the per-step
        # faulthandler watchdog (DASM2_STEP_WATCHDOG_SECONDS).
        pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=8))
        self._accelerator = Accelerator(
            mixed_precision=mixed_precision or "no",
            cpu=not torch.cuda.is_available(),
            kwargs_handlers=[ddp_kwargs, pg_kwargs],
        )

    @property
    def device(self) -> str:
        return str(self._accelerator.device)

    @property
    def model_init_device(self) -> str:
        # accelerate.prepare() handles device placement
        return "cpu"

    @property
    def is_main_process(self) -> bool:
        return bool(self._accelerator.is_main_process)

    @property
    def process_index(self) -> int:
        return int(self._accelerator.process_index)

    @property
    def num_processes(self) -> int:
        return int(self._accelerator.num_processes)

    @property
    def mixed_precision(self) -> str:
        # Read back off the Accelerator rather than stored from __init__: the
        # Accelerator normalizes (None -> "no") and can override the request
        # from the environment/accelerate config, so this is the mode actually
        # in force, which is what belongs in checkpoint metadata.
        return str(self._accelerator.mixed_precision)

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        result: torch.nn.Module = self._accelerator.unwrap_model(model)
        return result

    def unwrap_optimizer(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer:
        # accelerate.utils.AcceleratedOptimizer exposes the inner torch
        # optimizer as ``.optimizer``; fall back to the input for any
        # already-unwrapped optimizer that slips through.
        inner = getattr(optimizer, "optimizer", optimizer)
        return inner  # type: ignore[no-any-return]

    def backward(self, loss: torch.Tensor) -> None:
        self._accelerator.backward(loss)

    def clip_grad_norm_(self, parameters: Any, max_norm: float) -> torch.Tensor:
        result: torch.Tensor = self._accelerator.clip_grad_norm_(parameters, max_norm)
        return result

    def prepare_model_and_optimizer(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        # broadcast_buffers=False (set in __init__) skips DDP buffer sync — a
        # safe optimisation only when every persistent buffer is loaded
        # identically on all ranks. BatchNorm/LayerNorm running stats violate
        # this contract; surface that loudly here rather than letting per-rank
        # statistics silently diverge.
        for name, module in model.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.modules.batchnorm._BatchNorm,
                    torch.nn.modules.instancenorm._InstanceNorm,
                ),
            ):
                raise RuntimeError(
                    f"AccelerateBackend uses broadcast_buffers=False so the "
                    f"QDASM/Hybrid Thrifty lookup table is not re-synced every "
                    f"step. Module {name!r} ({type(module).__name__}) carries "
                    f"running statistics that DDP would normally sync; running "
                    f"stats will diverge across ranks. Wrap with SyncBatchNorm "
                    f"or restore broadcast_buffers=True for this model."
                )
        prepared_model, prepared_optimizer = self._accelerator.prepare(model, optimizer)
        return prepared_model, prepared_optimizer

    def prepare_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> torch.utils.data.DataLoader:
        # A variable-size batch sampler (e.g. TokenBudgetBatchSampler, issue
        # #724) has no fixed ``batch_size`` attribute. Accelerate's
        # ``BatchSamplerShard`` refuses to shard such a sampler unless
        # ``even_batches=False`` — with no per-batch row count it cannot pad the
        # trailing batch to equalize counts across ranks.
        #
        # Set the flag *explicitly per loader* rather than only ever flipping it
        # to False: it lives on the shared, run-long ``Accelerator``, so a
        # one-way flip would leak into every later loader. In joint training one
        # task can use token-budget batching while another falls back to
        # fixed-size on the same accelerator (issue #724 review finding #7) —
        # without restoring the flag, that fixed-size loader would silently lose
        # its trailing-batch padding. ``True`` matches Accelerate's default.
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        needs_uneven = batch_sampler is not None and not hasattr(
            batch_sampler, "batch_size"
        )
        self._accelerator.even_batches = not needs_uneven
        # accelerate.prepare() returns Any; narrow to DataLoader explicitly
        result: torch.utils.data.DataLoader = self._accelerator.prepare(dataloader)  # type: ignore[assignment]
        return result

    def release_dataloaders(
        self,
        dataloaders: List[torch.utils.data.DataLoader],
    ) -> None:
        # ``Accelerator.prepare(dataloader)`` appends every prepared loader to
        # ``self._accelerator._dataloaders`` (it keeps them to call set_epoch).
        # Because the accelerator is created once and lives for the whole run,
        # those references outlive each training cycle, so a persistent-worker
        # loader's iterator is never collected and its workers never shut down
        # — a per-cycle worker leak (issue #724 follow-up).
        #
        # Drop the references for exactly the loaders handed back. We do NOT
        # use ``Accelerator.free_memory()``/``clear()``: in Accelerate 1.10.1
        # those also wipe ``_models``/``_optimizers``/``_schedulers`` and reset
        # ``step``, which would discard the prepared model+optimizer the trainer
        # reuses across cycles.
        #
        # Fail loudly if the internal we depend on has disappeared: a silent
        # no-op here would resurrect the ~340 GB/cycle worker leak with nothing
        # to flag it (issue #724 review). An *empty* list is fine (nothing
        # prepared yet); a *missing* attribute means Accelerate restructured it.
        if not hasattr(self._accelerator, "_dataloaders"):
            import accelerate

            raise RuntimeError(
                "Accelerator no longer exposes '_dataloaders'; the per-cycle "
                "DataLoader release (issue #724) relies on this internal to "
                "prevent a ~340 GB/cycle worker leak in multi-cycle training. "
                f"The installed accelerate ({accelerate.__version__}) appears "
                "to have restructured it — update release_dataloaders() to "
                "match before running multi-cycle training."
            )
        retained = self._accelerator._dataloaders
        if not retained:
            return
        targets = {id(dl) for dl in dataloaders}
        self._accelerator._dataloaders = [
            dl for dl in retained if id(dl) not in targets
        ]

    def prepare_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        result: torch.optim.Optimizer = self._accelerator.prepare(optimizer)
        return result

    @contextmanager
    def no_sync(self, model: torch.nn.Module) -> Iterator[None]:
        with self._accelerator.no_sync(model):
            yield

    def broadcast(self, tensor: torch.Tensor) -> torch.Tensor:
        """Broadcast ``tensor`` from the main rank to all ranks.

        The tensor is moved to the accelerator device for the collective and
        restored to its original device on return. This round-trip is
        required because NCCL (the default multi-GPU backend) cannot
        broadcast CPU tensors, while callers such as ``BranchLengthHook``
        legitimately hold their tensors on CPU as dataset state. ``.to`` is
        a no-op when source and target devices already match, so no
        same-device guard is needed.
        """
        from accelerate.utils import broadcast as accel_broadcast

        original_device = tensor.device
        device_tensor = tensor.to(self._accelerator.device)
        broadcast_tensor: torch.Tensor = accel_broadcast(device_tensor)
        return broadcast_tensor.to(original_device)

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather a per-rank tensor from every rank, concatenated in rank order.

        Mirrors ``broadcast``'s device round-trip: NCCL cannot collect CPU
        tensors, but ``BranchLengthOptimizer.post_cycle`` holds its per-rank
        BL slices on CPU as dataset state. The tensor is moved to the
        accelerator device for the collective and restored to its original
        device on return.

        ``accelerate.gather`` requires the input shape to match across ranks.
        For unequal per-rank workloads (post-cycle BL re-opt slices uneven
        N across ranks), the caller pads each rank's tensor to the common
        length with a sentinel and trims after gather.
        """
        original_device = tensor.device
        device_tensor = tensor.to(self._accelerator.device)
        gathered: torch.Tensor = self._accelerator.gather(device_tensor)
        return gathered.to(original_device)

    def gather_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather a scalar tensor from all processes and return the mean.

        Uses plain ``gather`` (not ``gather_for_metrics``) because the input is
        a per-rank average, not a per-sample metric. ``gather_for_metrics``
        applies DistributedSampler padding correction that would incorrectly
        truncate rank entries when applied to scalars.

        The scalar is unsqueezed to shape ``[1]`` before gathering (``gather``
        requires at least 1-d input), yielding a ``[num_processes]`` tensor
        whose mean is the cross-rank average.
        """
        if tensor.ndim != 0:
            raise ValueError(
                f"gather_mean expects a scalar tensor, got shape {tensor.shape}"
            )
        gathered: torch.Tensor = self._accelerator.gather(tensor.unsqueeze(0))
        return gathered.mean()

    def shutdown(self) -> None:
        """Destroy the torch distributed process group if still initialized.

        Without this, NCCL emits a warning at interpreter exit about a
        leaked process group. Guarded by ``is_available()``/``is_initialized()``
        so repeated calls and non-distributed runs are safe.
        """
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
