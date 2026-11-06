"""Base classes for training hooks."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from dasm2.config import HookConfig

if TYPE_CHECKING:
    from dasm2.training.trainer import Trainer


class TrainingHook(ABC):
    """Abstract base class for training hooks with flexible context.

    Subclasses extract their specific requirements from **context.
    """

    @abstractmethod
    def pre_cycle(self, trainer: "Trainer", **context: Any) -> None:
        """Pre-cycle hook implementation.

        Args:
            trainer: Trainer instance with model, neutral_model, and training state
            **context: Flexible context data (e.g., train_dataset, val_dataset, cycle_count)
        """
        pass

    @abstractmethod
    def post_cycle(self, trainer: "Trainer", **context: Any) -> None:
        """Post-cycle hook implementation.

        Args:
            trainer: Trainer instance with model, neutral_model, and training state
            **context: Flexible context data (e.g., train_dataset, val_dataset, cycle_count)
        """
        pass

    def on_eval(self, trainer: "Trainer", **context: Any) -> None:
        """Per-evaluation hook. Default: no-op. Override to react to evals.

        Fires once per evaluation pass inside ``_train_for_steps`` after
        best-model bookkeeping but before the next training step. Use this
        for per-eval checkpointing, custom early-stop signals, or metric
        reporting that should track evaluation cadence.

        Args:
            trainer: Trainer instance with model, neutral_model, and training state.
            **context: Includes ``cycle_count``, ``num_cycles``, ``cycle_step``,
                ``eval_count``, ``val_loss`` (the sum over tasks of each task's
                loss_weight-scaled validation loss), ``monitored_val_loss`` (the
                loss that ``was_best`` and early stopping follow: equal to
                ``val_loss`` unless ``monitor_task`` is set), ``was_best``,
                ``train_data``, ``val_data``, ``tasks``, ``per_task_val`` (each
                task's own loss_weight-scaled validation loss, keyed by
                ``task.log_key``).
        """

    def on_step(self, trainer: "Trainer", **context: Any) -> None:
        """Per-optimizer-step hook. Default: no-op.

        Fires after every optimizer step inside ``_train_for_steps``, after
        that step's evaluation (if one fell due) and its ``on_eval`` hooks.

        Args:
            trainer: Trainer instance with model, neutral_model, and training state.
            **context: Includes ``cycle_count``, ``num_cycles`` and
                ``cycle_step``.
        """

    def zeroed_loss_weight_task_key(self) -> Optional[str]:
        """Log key of the task whose ``loss_weight`` this hook zeroes mid-run.

        Default ``None``. A hook that sets a task's ``loss_weight`` to 0.0 (as
        ``OverfitGuardHook`` does) overrides this, so the trainer can refuse to
        monitor a task that would end up with nothing to monitor.
        """
        return None

    @abstractmethod
    def to_config(self) -> HookConfig:
        """Convert hook back to configuration for serialization.

        Returns:
            HookConfig with hook type and parameters
        """
        pass


class GradientCombineHook(ABC):
    """Abstract base for pluggable multi-task gradient combination.

    Unlike :class:`TrainingHook`'s three per-task categories
    (``pre_cycle_hooks``/``post_cycle_hooks``/``eval_hooks``, each attached to
    one ``Task`` and reacting to that task's own events), this hook category
    is inherently cross-task: it decides how every task's own gradient on the
    *shared* (non-head) model parameters gets combined into one gradient for
    the optimizer step, once per micro-batch. Attached at the top level of
    ``TrainingConfig`` (``combine_gradient_hooks``), not per-task, since
    combination is not any single task's behavior.

    ``Trainer`` calls :meth:`combine_gradients` only when 2+ tasks
    contribute to a micro-batch and this hook is attached — a run with no
    hook attached never reaches this class at all, and gets exactly today's
    plain-sum combination (see ``Trainer._combine_gradients``'s own no-hook
    branch, which is what every existing config exercises).
    """

    @abstractmethod
    def combine_gradients(
        self,
        trainer: "Trainer",
        per_task_grads: Dict[str, List[torch.Tensor]],
        shared_params: List[torch.nn.Parameter],
    ) -> Dict[str, List[torch.Tensor]]:
        """Return possibly-modified per-task gradients on ``shared_params``.

        Args:
            trainer: The Trainer instance (model/optimizer access, e.g. for a
                combiner that needs the run's own seed for reproducible
                random task ordering).
            per_task_grads: ``task.log_key -> gradients``, one tensor per
                entry of ``shared_params`` in the same order. Every task
                contributing to the current micro-batch has an entry, and
                every entry has the same length as ``shared_params`` (a task
                that never touches a given shared parameter contributes an
                explicit zero tensor there, not a missing entry) — safe to
                index positionally without a per-task None check.
            shared_params: The model's shared (non-head) parameters, in the
                order ``per_task_grads``' gradient lists are indexed against.

        Returns:
            A dict with the same keys and per-key list length as
            ``per_task_grads`` — the (possibly modified) per-task gradients
            that ``Trainer`` then sums into the combined gradient applied to
            ``shared_params``. Returning ``per_task_grads`` unchanged is a
            valid, correct implementation (the identity/no-op case).
        """
        pass

    def pop_diagnostics(self) -> Dict[str, float]:
        """Per-eval-window diagnostic summary, then reset.

        Default: no diagnostics. Override to report reset-able aggregate
        stats (e.g. ``PCGradHook``'s conflict fraction / mean cosine
        similarity) accumulated since the last call — read once per eval
        boundary the same way ``OptimizerStepMetrics.compute()`` +
        ``.reset()`` already is (``Trainer._run_eval_boundary``), and merged
        into that same logged dict rather than needing its own new
        logger-signature parameter.
        """
        return {}

    @abstractmethod
    def to_config(self) -> HookConfig:
        """Convert hook back to configuration for serialization.

        Returns:
            HookConfig with hook type and parameters
        """
        pass
