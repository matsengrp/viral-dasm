"""Encapsulates a second, head-scoped optimizer's own state.

``HeadOptimizerState`` is deliberately **not** a ``TrainingHook`` or
``GradientCombineHook`` (see ``dasm2/training/hooks/``), and is not
registered in ``HOOK_REGISTRY``/``GRADIENT_COMBINE_HOOK_REGISTRY`` or wired
through ``create_hook()`` — DESIGN.md D5 requires that of anything actually
plugged into the training loop that way, and this class doesn't fit either
shape: it isn't per-task (``TrainingHook``'s ``pre_cycle``/``post_cycle``/
``on_eval`` lifecycle) or cross-task-once-per-microbatch
(``GradientCombineHook``). It lives next to ``optimizer.py`` rather than
under ``dasm2/training/hooks/`` for exactly this reason — an earlier version
of this class was misnamed ``HeadOptimizerHook`` and placed in that
directory, which invited exactly the confusion this paragraph is now
avoiding.

What it actually is: ``Trainer`` supports giving exactly one head its own
optimizer, separate from the shared main optimizer every other parameter
(including the shared encoder) trains under. Before this module existed,
that support was a ``head_optimizer``/``head_optimizer_module``/
``head_optimizer_head_type``/``head_scheduler`` attribute quartet with
``if self.head_optimizer is not None: ...`` branches repeated across
roughly eight places in the training loop (``__init__``, ``_optimizer_step``,
both ``zero_grad`` call sites, ``_reset_optimizer_with_adjusted_lr``,
``export_metadata``, and checkpoint save/restore) — exactly the spread-out,
head-specific-conditional pattern D5 says the training loop should not
contain.

``HeadOptimizerState`` does not eliminate the presence check itself (whether
this lever is configured at all is legitimately a training-loop-level
question, decided once via ``Trainer.head_optimizer_state is not None``) —
it consolidates the *behavior* inside each branch (the actual optimizer/
scheduler manipulation) into one object, so each of those eight call sites
shrinks to a single delegated method call instead of several lines of
head-specific logic.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dasm2.config import OptimizerConfig
from dasm2.heads.types import HeadType
from dasm2.training.optimizer import reconstruct_optimizer_config


def assert_head_optimizer_supported_by_backend(backend: "ComputeBackend") -> None:
    """Raise if a configured head_optimizer can't run under ``backend``.

    Only ``SingleGPUBackend`` has been exercised with a second, head-scoped
    optimizer. Call this once head_optimizer/head_optimizer_config
    is already known to be set (the caller decides *whether* it's configured;
    this only decides whether ``backend`` can support it), before any
    expensive model/dataset setup — the three call sites
    (``create_trainer``, ``Trainer.__init__``, ``resume_training``) each want
    to fail fast at their own entry point rather than share one call site,
    which is why this is a shared function, not a single shared call.

    Raises:
        ValueError: If ``backend.num_processes > 1``.
    """
    if backend.num_processes > 1:
        raise ValueError(
            "head_optimizer is not supported under multi-process backends "
            f"(backend.num_processes={backend.num_processes}): only "
            "SingleGPUBackend has been exercised with a second, head-scoped "
            "optimizer. Run single-process (SingleGPUBackend) "
            "until DDP support is added."
        )


if TYPE_CHECKING:
    from dasm2.training.compute_backend import ComputeBackend


class HeadOptimizerState:
    """Owns the optimizer/scheduler/LR state for one head's own optimizer.

    Args:
        optimizer: Optimizer scoped to exactly ``module``'s own parameters.
        module: The head module ``optimizer`` covers.
        head_type: Which head ``module`` is.
        scheduler: This optimizer's own ``ReduceLROnPlateau``, or None under
            a scheduler config with no per-eval plateau signal to mirror
            (see ``Trainer._create_head_scheduler``).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        module: torch.nn.Module,
        head_type: HeadType,
        scheduler: Optional[ReduceLROnPlateau] = None,
    ) -> None:
        self.optimizer = optimizer
        self.module = module
        self.head_type = head_type
        self.scheduler = scheduler
        # Initial LR, for the same cycle-boundary reset the main optimizer
        # gets (Trainer._reset_optimizer_with_adjusted_lr),
        # kept distinct from the main optimizer's own initial LR so the two
        # never mix.
        self.learning_rate: float = optimizer.param_groups[0]["lr"]

    def current_lr(self) -> float:
        """This optimizer's current LR (param_groups[0])."""
        return float(self.optimizer.param_groups[0]["lr"])

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step_scheduler(self, val_loss: float) -> None:
        """Step this optimizer's own plateau scheduler, if any. No-op otherwise."""
        if self.scheduler is not None:
            self.scheduler.step(val_loss)

    def state_dict(self) -> Dict[str, Any]:
        return dict(self.optimizer.state_dict())

    def scheduler_state_dict(self) -> Optional[Dict[str, Any]]:
        return self.scheduler.state_dict() if self.scheduler is not None else None

    def restore(
        self,
        *,
        optimizer_state_dict: Dict[str, Any],
        learning_rate: float,
        scheduler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Restore this optimizer's (and, if present, scheduler's) state
        from a checkpoint. All mutation of this state during
        checkpoint restore goes through this one method — the caller
        (``Trainer.restore_training_state``) only decides *whether* to call
        it (the head-type mismatch / presence checks it already does),
        never pokes ``self.optimizer``/``self.learning_rate`` directly.

        Args:
            optimizer_state_dict: From ``TrainingState.head_optimizer_state_dict``.
            learning_rate: From ``TrainingState.head_learning_rate`` — the
                initial LR the next cycle-boundary reset blends from.
            scheduler_state_dict: From ``TrainingState.head_scheduler_state_dict``,
                or None if this run has no head scheduler (``self.scheduler
                is None``). Restoring one when ``self.scheduler`` is None,
                or omitting one when it isn't, is a caller bug — asserted
                here rather than silently ignored.
        """
        self.learning_rate = learning_rate
        self.optimizer.load_state_dict(optimizer_state_dict)
        assert (self.scheduler is None) == (scheduler_state_dict is None), (
            "HeadOptimizerState.restore(): scheduler_state_dict presence must "
            "match self.scheduler presence — caller must check this first "
            f"(self.scheduler is None={self.scheduler is None}, "
            f"scheduler_state_dict is None={scheduler_state_dict is None})."
        )
        if self.scheduler is not None and scheduler_state_dict is not None:
            self.scheduler.load_state_dict(scheduler_state_dict)

    def to_optimizer_config(self, backend: "ComputeBackend") -> OptimizerConfig:
        """Reconstruct this optimizer's OptimizerConfig, for export_metadata
        and for :meth:`Trainer._reset_optimizer_with_adjusted_lr`'s rebuild.

        Reads from ``self.optimizer``/``self.learning_rate`` (the initial
        LR) rather than the main optimizer's own config, since the two can
        legitimately differ.

        Args:
            backend: Strips any backend-specific optimizer wrapper — same
                role as in :func:`dasm2.training.optimizer.reconstruct_optimizer_config`.
        """
        return reconstruct_optimizer_config(backend, self.optimizer, self.learning_rate)

    def replace_optimizer(
        self, optimizer: torch.optim.Optimizer, *, new_learning_rate: float
    ) -> None:
        """Swap in a freshly rebuilt optimizer (cycle-boundary LR reset).

        ``new_learning_rate`` becomes the "initial LR" the next cycle-boundary
        reset uses: the blend target under ReduceLROnPlateau, the restart LR
        under a per-step scheduler (warmup_polynomial, cosine_warm_restarts).
        The main optimizer's ``Trainer.learning_rate``
        instead stays at its configured value across resets.
        """
        self.optimizer = optimizer
        self.learning_rate = new_learning_rate

    def replace_scheduler(self, scheduler: Optional[ReduceLROnPlateau]) -> None:
        self.scheduler = scheduler
