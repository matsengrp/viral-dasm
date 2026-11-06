"""Per-task overfit guard hook (issue #1096)."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from dasm2.config import HookConfig
from dasm2.training.hooks.base import TrainingHook

if TYPE_CHECKING:
    from dasm2.training.task import Task
    from dasm2.training.trainer import Trainer


def freeze_task_head(trainer: "Trainer", task: "Task", tasks: List["Task"]) -> bool:
    """Freeze ``task``'s head parameters (``requires_grad=False``), unless
    another task shares the same head (e.g. AAH's whichmut/aao_whichmut
    pair), in which case freezing would also stop that other task's own
    gradient from updating the shared head.

    Shared by :meth:`OverfitGuardHook.on_eval` (the initial freeze) and
    ``train_command.py`` (re-deriving the freeze at task-construction time
    for a task whose ``loss_weight`` is already ``0.0`` — e.g. on
    ``--resume`` of a run this hook already froze, since ``requires_grad``
    isn't part of ``state_dict`` and doesn't survive the reload otherwise).

    Returns:
        True if parameters were frozen, False if skipped due to head sharing.
    """
    sharing_tasks = [t for t in tasks if t.head_type == task.head_type]
    if len(sharing_tasks) > 1:
        return False
    head = trainer.unwrapped_model.prediction_heads[task.head_type.value]
    for p in head.parameters():
        p.requires_grad = False
    return True


class OverfitGuardHook(TrainingHook):
    """Zeroes ``task_key``'s ``loss_weight`` and freezes its head's own
    parameters once its own val loss stops improving for ``patience``
    consecutive evals.

    Compares the *raw* (unweighted) per-task loss, not the value stored in
    the ``per_task_val`` eval context — that value is
    ``loss * task.loss_weight`` (``Trainer._compute_batch_loss``), so
    comparing it directly against a fixed ``min_delta`` would make the
    guard's sensitivity to the same underlying loss trend scale with the
    task's own ``loss_weight``. Unscaling keeps behavior consistent across
    a lambda sweep, which varies ``loss_weight`` by design.

    Independent of ``TrainingConfig.early_stopping_patience``, which
    watches the combined loss across all tasks.

    ``loss_weight=0.0`` alone zeroes this task's *gradient* contribution but
    does not stop the optimizer from continuing to touch its parameters.
    Under ``torch.optim.RMSprop`` (this repo's default), ``weight_decay`` is
    not decoupled — it's added directly into the gradient before RMSprop's
    adaptive per-parameter normalization — so a parameter with a genuinely
    zero true gradient still receives a ``weight_decay * param`` "gradient"
    every step, and RMSprop's own adaptive normalization amplifies that into
    a large *relative* decay rate regardless of how small ``weight_decay``
    is: a real, trained parameter decays to exact numerical zero within
    roughly 1,000-2,000 further optimizer steps. Setting
    ``requires_grad=False`` on the head's parameters when this hook fires
    stops that: a parameter with no gradient at all is skipped by every
    optimizer's own ``if p.grad is None: continue`` check, so weight_decay
    never touches it either.

    Freezing under DDP/accelerate is currently unsupported: DDP's reducer is
    built once at construction from whichever parameters have
    ``requires_grad=True`` at that time, and expects every one of them to
    produce a gradient on every subsequent backward pass unless told
    otherwise. ``find_unused_parameters=True`` on the
    ``DistributedDataParallelKwargs`` (``dasm2/training/compute_backend.py``)
    would tell it otherwise, but that flag breaks DDP for any loss-owned
    parameter used outside the wrapped model's own ``forward()`` (e.g.
    ``DMSFunctionalScoreLoss.scale``) — see that file's comment for the full
    explanation. Nothing in this repo currently runs joint training under
    DDP, so this hook is exercised single-device/single-process only for
    now; combining it with DDP needs a real fix (e.g. ``static_graph=True``)
    first.

    ``requires_grad=False`` is not part of a checkpoint's ``state_dict``, so
    it does not survive a reload on its own. ``train_command.py`` re-derives
    it for any task whose (checkpoint-sourced) ``loss_weight`` is already
    ``0.0`` when tasks are constructed — see :func:`freeze_task_head` and its
    call site in ``train_model_from_datasets`` — so a ``--resume`` of a run
    this hook already froze comes back frozen rather than re-arming for
    another ``patience`` evals.
    """

    def __init__(self, task_key: str, patience: int, min_delta: float = 1e-4) -> None:
        self.task_key = task_key
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf")
        self._no_improve = 0
        self._last_cycle_count: Any = None

    def zeroed_loss_weight_task_key(self) -> Optional[str]:
        """This hook zeroes ``task_key``'s ``loss_weight`` once it fires."""
        return self.task_key

    def pre_cycle(self, trainer: "Trainer", **context: Any) -> None:
        """No-op: ``train_command.py`` attaches this hook only to
        ``Task.eval_hooks``, never ``Task.pre_cycle_hooks``, so
        ``Trainer``'s per-cycle hook dispatch never calls this. The
        per-cycle reset of ``_best``/``_no_improve`` happens in ``on_eval``
        instead, by detecting a change in ``cycle_count`` — see there."""

    def post_cycle(self, trainer: "Trainer", **context: Any) -> None:
        pass

    def on_eval(self, trainer: "Trainer", **context: Any) -> None:
        """No-op if this task isn't found, is already zeroed, or
        ``per_task_val`` has no entry for it yet.

        Resets ``_best``/``_no_improve`` on the first eval of every new
        cycle (detected via ``context["cycle_count"]``, since ``pre_cycle``
        is never dispatched for an eval-only hook — see ``pre_cycle``),
        matching ``Trainer._reset_best_model_tracking``'s per-cycle scoping
        for the combined-loss early-stopping counters this guard is
        independent of.
        """
        cycle_count = context["cycle_count"]
        if cycle_count != self._last_cycle_count:
            self._best, self._no_improve = float("inf"), 0
            self._last_cycle_count = cycle_count

        tasks: List["Task"] = context["tasks"]
        per_task_val: Dict[str, float] = context["per_task_val"]
        task = next((t for t in tasks if t.log_key == self.task_key), None)
        if task is None or task.loss_weight == 0.0:
            return
        weighted = per_task_val.get(self.task_key)
        if weighted is None:
            return
        raw = weighted / task.loss_weight  # unscale — see class docstring
        if raw < self._best - self.min_delta:
            self._best, self._no_improve = raw, 0
            return
        self._no_improve += 1
        if self._no_improve >= self.patience:
            # Unlike every other hook in this package, this writes directly
            # into a Task field rather than only instance/dataset/model
            # state — Task.loss_weight stops matching the value TaskConfig
            # produced it from once this fires. Trainer.export_metadata()
            # (trainer.py) round-trips this live value back into a
            # TaskConfig on every post-cycle save (e.g. BranchLengthOptimizer's
            # cache write), so TaskConfig.loss_weight's Field is ge=0.0 rather
            # than gt=0.0 to match. Any other code reading task.loss_weight
            # post-construction should not assume it is immutable.
            task.loss_weight = 0.0

            # loss_weight=0.0 alone does not stop weight_decay from decaying
            # this head's parameters to zero over subsequent training (see
            # class docstring) — freeze them via requires_grad too.
            frozen = freeze_task_head(trainer, task, tasks)
            if trainer.is_main_process:
                if frozen:
                    print(
                        f"OverfitGuardHook: {self.task_key} loss_weight -> 0.0, "
                        "parameters frozen (requires_grad=False)"
                    )
                else:
                    sharing_tasks = [t for t in tasks if t.head_type == task.head_type]
                    print(
                        f"OverfitGuardHook: {self.task_key} loss_weight -> 0.0 "
                        f"(head shared by {len(sharing_tasks)} tasks — not freezing "
                        "parameters, only zeroing this task's loss_weight)"
                    )

    def to_config(self) -> HookConfig:
        return HookConfig(  # type: ignore[call-arg]
            type="OverfitGuardHook",
            task_key=self.task_key,
            patience=self.patience,
            min_delta=self.min_delta,
        )
