"""Type definitions for training module."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class HookPhase(str, Enum):
    """Enumeration of hook execution phases.

    Hooks can run at different points during the training cycle.
    Using an enum instead of string literals catches typos at type-check time.

    Note:
        The enum values ("pre_cycle", "post_cycle", "on_eval") match the
        corresponding method names on TrainingHook classes. ``PRE_CYCLE``
        and ``POST_CYCLE`` are dispatched generically by
        ``Trainer._run_all_task_hooks`` via ``getattr(hook, phase.value)``.
        ``ON_EVAL`` is dispatched separately by
        ``Trainer._dispatch_eval_hooks`` because the eval-boundary call
        site needs the eval payload (val_loss, monitored_val_loss, eval_count,
        was_best) and runs at a different cadence than cycle hooks.
        If you add a new phase, also add a matching method to the
        TrainingHook protocol and wire it into the appropriate dispatcher.

    Examples:
        >>> self._run_hook_with_logging(hook, HookPhase.PRE_CYCLE, task, cycle, num_cycles)
        >>> if phase == HookPhase.POST_CYCLE:
        >>>     # post-cycle logic
    """

    PRE_CYCLE = "pre_cycle"
    POST_CYCLE = "post_cycle"
    ON_EVAL = "on_eval"


@dataclass(frozen=True)
class StepBudget:
    """Per-invocation step budget for the step-based training loop.

    One ``_train_for_steps`` invocation consumes up to ``max_train_steps``
    optimizer steps. Scoping depends on mode:

    - **Joint multi-task**: one invocation per cycle, so ``max_train_steps``
      scopes the whole cycle.
    - **Sequential single-task**: one invocation per task per cycle, so
      each task gets its own ``max_train_steps``. Total per cycle is
      ``num_tasks * max_train_steps``.

    Attributes:
        max_train_steps: Maximum optimizer steps for this invocation.
        eval_every_steps: Validation cadence in optimizer steps.
        max_eval_steps: Maximum micro-batches per evaluation, per task: each
            task is evaluated over its own validation loader, for at most this
            many batches or its whole loader if shorter. ``None`` exhausts
            every task's validation loader.
        start_step: Starting step (>0 only when resuming mid-invocation).
            Only eval-boundary values (multiples of ``eval_every_steps``)
            are exercised by ``TestMidCycleResume``: a save inside an
            eval window — i.e. between optimizer steps that are not yet
            an eval boundary — will produce a cosmetically incorrect
            ``avg_train_loss`` for the first post-resume eval row
            (the partial pre-save window contribution is lost) but
            otherwise preserves model weights, optimizer state, RNG
            streams, and best-model tracking. Save points emitted by
            ``CheckpointSaveHook.on_eval`` always land on eval
            boundaries, so the CLI default never hits this case.
    """

    max_train_steps: int
    eval_every_steps: int
    max_eval_steps: Optional[int] = None
    start_step: int = 0
