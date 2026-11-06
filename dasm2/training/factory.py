"""Trainer factory for Task-based training."""

from typing import TYPE_CHECKING, List, Optional

from dasm2.config import OptimizerConfig, SchedulerConfig, TaskConfig
from dasm2.heads.types import HeadType
from dasm2.metadata import SourceCheckpointInfo
from dasm2.models.core.base import CoreModel
from dasm2.neutral.base import AbstractNeutralModel
from dasm2.training.compute_backend import ComputeBackend, SingleGPUBackend
from dasm2.training.head_optimizer_state import (
    assert_head_optimizer_supported_by_backend,
)
from dasm2.training.hooks.base import GradientCombineHook
from dasm2.training.optimizer import create_optimizer
from dasm2.training.task import Task, build_task_loss_module_dict
from dasm2.training.trainer import Trainer
from dasm2.training.types import StepBudget

if TYPE_CHECKING:
    from dasm2.training.logging import TrainingLogger


def create_trainer(
    model: CoreModel,
    neutral_model: Optional[AbstractNeutralModel],
    tasks: List[Task],
    batch_size: int = 32,
    val_batch_size: int = 32,
    optimizer_config: Optional[OptimizerConfig] = None,
    backend: Optional[ComputeBackend] = None,
    cache_dir: Optional[str] = ".dasm2_cache",
    config_name: Optional[str] = None,
    logger: Optional["TrainingLogger"] = None,
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
    combine_gradient_hooks: Optional[List[GradientCombineHook]] = None,
    head_optimizer_config: Optional[OptimizerConfig] = None,
    head_optimizer_head_type: Optional[HeadType] = None,
) -> Trainer:
    """Create a Trainer for the given tasks.

    Validates that the model contains all required heads before creating
    the trainer. Prepares model and optimizer for the given backend
    (DDP wrapping + device placement for multi-GPU, no-op for single-GPU).

    Args:
        model: CoreModel with prediction heads.
        neutral_model: Neutral evolution model.
        tasks: List of training tasks (single-task training uses ``[task]``).
        batch_size: Per-device batch size for training.
        val_batch_size: Per-device batch size for validation. Defaults to 32;
            change ``TrainingConfig.val_batch_size`` in lockstep if you
            change the default here.
        optimizer_config: Optimizer configuration (defaults to Adam, lr=1e-3).
        backend: Compute backend. If None, defaults to ``SingleGPUBackend("cuda")``.
        cache_dir: Directory for cached artifacts.
        config_name: Optional config name for logging.
        logger: Optional training logger for metrics streaming.
        gradient_clip_val: Maximum gradient norm for clipping. None disables clipping.
        early_stopping_patience: Stop if validation loss doesn't improve for this
            many *evaluations*. None disables patience-based early stopping.
        early_stopping_min_delta: Minimum validation loss improvement required to
            reset the patience counter.
        monitor_task: Log key of the task whose validation loss drives early
            stopping, best-model tracking and the plateau scheduler. None monitors
            the sum over all tasks. See ``TrainingConfig.monitor_task``.
        source_checkpoint: Optional provenance info for fine-tuned models.
        scheduler_config: Scheduler configuration. If None, defaults to ReduceLROnPlateau.
        use_mixed_precision: Enable BF16 mixed precision for forward pass.
        gradient_accumulation_steps: Number of micro-batches to accumulate gradients
            over before performing an optimizer step.
        step_budget: Step budget passed through to :class:`Trainer`. Required for
            the warmup-polynomial scheduler (the schedule spans
            ``step_budget.max_train_steps``) and for the step-based loop entry
            points ``multi_cycle_train`` / ``multi_cycle_train_joint``.
        source_task_configs: Task descriptors for :meth:`Trainer.export_metadata`
            to emit when ``tasks`` is empty (issue #974). Only meaningful for
            the task-less checkpoint-handling paths; see :class:`Trainer`.
        combine_gradient_hooks: Top-level (not per-task) hooks controlling how
            multiple tasks' gradients on shared parameters are combined
            before the optimizer step (issue #1144). None/empty (the
            default) reproduces today's plain-sum combination exactly.
        head_optimizer_config: Optional config for a second optimizer scoped
            to exactly one head's own parameters (issue #1220) — e.g.
            slowing ``BindingClassificationHead``'s own convergence in
            joint training without touching the shared encoder or any
            other head. Requires ``head_optimizer_head_type``. Only that
            head's parameters move under this optimizer's own
            learning_rate/weight_decay; every other parameter (including
            the shared encoder) stays on ``optimizer_config``'s optimizer,
            unchanged. Not supported under multi-process backends. None
            (default) reproduces today's single-optimizer behavior exactly.
            When the run's scheduler is ``ReduceOnPlateauSchedulerConfig``,
            this optimizer also gets its own ``ReduceLROnPlateau`` (same
            factor/eval_patience, applied independently), stepped on this
            head's own raw validation loss rather than the combined loss the
            main scheduler watches. Under a per-step scheduler
            (``WarmupPolynomialSchedulerConfig``,
            ``CosineWarmRestartsSchedulerConfig``) there is no per-eval
            plateau signal to mirror, so this optimizer's LR stays flat
            within a cycle instead. Either way, its LR is also reset once at
            every cycle boundary the same way the main optimizer's is (a
            geometric-mean blend under plateau, a restart at the configured
            LR under a per-step scheduler), seeded from its own initial LR.
        head_optimizer_head_type: Which head ``head_optimizer_config``
            applies to. Required iff ``head_optimizer_config`` is set; the
            named head must exist on ``model``.

    Returns:
        Configured Trainer instance.

    Raises:
        ValueError: If model is missing any head required by the tasks, or
            if exactly one of ``head_optimizer_config``/
            ``head_optimizer_head_type`` is set, or if
            ``head_optimizer_config`` is set under a multi-process backend.
    """
    # Validate all required heads exist
    for task in tasks:
        if not model.has_head(task.head_type):
            available = list(model.prediction_heads.keys())
            raise ValueError(
                f"Model missing required head: {task.head_type.value}. "
                f"Available heads: {available}"
            )

    if (head_optimizer_config is None) != (head_optimizer_head_type is None):
        raise ValueError(
            "head_optimizer_config and head_optimizer_head_type must both "
            "be provided or both be None "
            f"(head_optimizer_config={head_optimizer_config!r}, "
            f"head_optimizer_head_type={head_optimizer_head_type!r})."
        )

    if backend is None:
        backend = SingleGPUBackend("cuda")

    if head_optimizer_config is not None:
        assert_head_optimizer_supported_by_backend(backend)

    # Attach per-task loss modules to the model so their parameters are
    # picked up by ``create_optimizer(model, ...)``. The Trainer constructor
    # re-attaches the same instances inside an idempotent ``ModuleDict``;
    # rebuilding does not change parameter ``id()`` values, so the optimizer
    # group remains valid across the two registrations.
    model.task_losses = build_task_loss_module_dict(tasks)

    if optimizer_config is None:
        optimizer_config = OptimizerConfig()

    # head_optimizer_module is resolved before backend.prepare_model_and_optimizer
    # below, while ``model`` is still the plain (unwrapped) CoreModel — the
    # multi-process check above guarantees that wrap is a no-op anyway, but
    # resolving here keeps the module reference obviously correct regardless.
    head_optimizer_module = (
        model.require_head(head_optimizer_head_type)
        if head_optimizer_head_type is not None
        else None
    )
    optimizer = create_optimizer(
        model, optimizer_config, exclude_module=head_optimizer_module
    )
    head_optimizer = (
        create_optimizer(head_optimizer_module, head_optimizer_config)
        if head_optimizer_module is not None and head_optimizer_config is not None
        else None
    )

    # Prepare model and optimizer together (DDP wrap + device placement).
    # No-op for SingleGPUBackend.
    model, optimizer = backend.prepare_model_and_optimizer(model, optimizer)  # type: ignore[assignment]
    if head_optimizer is not None:
        head_optimizer = backend.prepare_optimizer(head_optimizer)

    return Trainer(
        model=model,
        neutral_model=neutral_model,
        tasks=tasks,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        optimizer=optimizer,
        backend=backend,
        cache_dir=cache_dir,
        config_name=config_name,
        logger=logger,
        gradient_clip_val=gradient_clip_val,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        monitor_task=monitor_task,
        source_checkpoint=source_checkpoint,
        gradient_accumulation_steps=gradient_accumulation_steps,
        scheduler_config=scheduler_config,
        use_mixed_precision=use_mixed_precision,
        step_budget=step_budget,
        num_dataloader_workers=num_dataloader_workers,
        max_tokens_per_batch=max_tokens_per_batch,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        dataloader_pin_memory=dataloader_pin_memory,
        data_seed=data_seed,
        source_task_configs=source_task_configs,
        combine_gradient_hooks=combine_gradient_hooks,
        head_optimizer=head_optimizer,
        head_optimizer_module=head_optimizer_module,
        head_optimizer_head_type=head_optimizer_head_type,
    )
