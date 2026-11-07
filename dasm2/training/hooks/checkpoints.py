"""Checkpoint saving hooks for training."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from dasm2.config import HookConfig
from dasm2.training.hooks.base import TrainingHook

if TYPE_CHECKING:
    from dasm2.training.trainer import Trainer


class CheckpointSaveHook(TrainingHook):
    """Hook to save model checkpoints after each training cycle."""

    #: Marks step-cadence saves in the filename; only these are rotated.
    #:
    #: Upstream this came from ``training/checkpoint_selection.py``, which
    #: shared it with the readers that pick a final checkpoint out of a
    #: finished run's directory. This fork has no such reader -- it names the
    #: checkpoint it wants -- so the module is not vendored and the writer
    #: owns the marker.
    PERIODIC_TAG = "periodic"
    #: Step-cadence saves kept: the newest, plus one in case it is truncated.
    PERIODIC_KEEP = 2

    def __init__(
        self,
        output_dir: str,
        notes: Optional[str] = None,
        checkpoint_every_n_evals: Optional[int] = None,
        save_training_state: bool = False,
        checkpoint_every_n_steps: Optional[int] = None,
    ) -> None:
        """Initialize checkpoint saving hook.

        Args:
            output_dir: Directory to save checkpoints
            notes: Optional notes to embed in checkpoint metadata
            checkpoint_every_n_evals: If set, save checkpoints every N
                evaluations via on_eval(). None disables intra-cycle
                checkpointing (the post_cycle save still fires once per
                cycle on the restored best model).
            save_training_state: If True, save resumable checkpoints that include
                optimizer, scheduler, counters, and RNG state for training resume.
            checkpoint_every_n_steps: If set, also save every N optimizer steps
                via on_step() (issue #1640: D₂'s evals are hours apart), as
                ``periodic_step{N}`` files of which only the newest
                ``PERIODIC_KEEP`` are kept. The save follows the step's eval, if
                one ran.
        """
        if checkpoint_every_n_steps is not None and checkpoint_every_n_steps < 1:
            raise ValueError(
                f"checkpoint_every_n_steps must be >= 1, got {checkpoint_every_n_steps}"
            )
        self.output_dir = Path(output_dir)
        self.notes = notes
        self.checkpoint_every_n_evals = checkpoint_every_n_evals
        self.save_training_state = save_training_state
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        # Periodic saves on disk, oldest first; read from disk at the first.
        self._periodic: Optional[list[Path]] = None

    def pre_cycle(self, trainer: "Trainer", **context: Any) -> None:
        """Pre-cycle hook - no action needed for checkpoint saving.

        Args:
            trainer: Trainer instance
            **context: Flexible context data
        """
        pass

    def post_cycle(self, trainer: "Trainer", **context: Any) -> None:
        """Post-cycle hook - save checkpoint after cycle completes.

        The trainer calls ``_restore_best_model()`` at the end of every
        cycle's step loop, so *every* post-cycle save holds that cycle's
        best-validation weights rather than its last-step weights. The
        ``_best`` marker in the suffix says so on the file itself: before
        issue #1011 the final cycle's save was the one file with neither a
        cycle nor a step suffix, which read as a generic "last state"
        artifact and, with two attempts in one directory, produced two
        equally anonymous files that selection heuristics picked between at
        random.

        Call on every rank: only the main process writes, but a resumable
        save first gathers every rank's pass position (``save_resumable``).

        Args:
            trainer: Trainer instance with model to save
            **context: Must include ``cycle_count`` (int): current cycle
                number (0-indexed).
        """
        if "cycle_count" not in context:
            raise ValueError("Context must include 'cycle_count' for checkpoint naming")

        cycle_count = context["cycle_count"]

        self._save(trainer, f"cycle{cycle_count}_best")

    def on_eval(self, trainer: "Trainer", **context: Any) -> None:
        """Save checkpoint at evaluation cadence.

        Fires every ``checkpoint_every_n_evals`` evaluations. No-op when
        ``checkpoint_every_n_evals`` is None or when the rank guard
        suppresses the underlying save.

        Args:
            trainer: Trainer instance with model to save.
            **context: Must include ``cycle_count``, ``num_cycles``,
                ``cycle_step``, ``eval_count``.
        """
        if self.checkpoint_every_n_evals is None:
            return
        cycle_count = context["cycle_count"]
        num_cycles = context["num_cycles"]
        cycle_step = context["cycle_step"]
        eval_count = context["eval_count"]
        if (eval_count + 1) % self.checkpoint_every_n_evals != 0:
            return
        if num_cycles > 1:
            suffix = f"cycle{cycle_count}_step{cycle_step}"
        else:
            suffix = f"step{cycle_step}"
        self._save(trainer, suffix)

    def on_step(self, trainer: "Trainer", **context: Any) -> None:
        """Save every ``checkpoint_every_n_steps`` steps, keeping the newest few.

        Every rank reaches the same step, so the collective save needs no
        coordination. It runs after the step's eval, so a save at an eval step
        holds the post-eval state. Older periodic files are deleted only after
        the new one is written, so a kill mid-write leaves an intact earlier one.

        Args:
            trainer: Trainer instance with model to save.
            **context: Must include ``cycle_count``, ``num_cycles``,
                ``cycle_step``.
        """
        n = self.checkpoint_every_n_steps
        cycle_step = context["cycle_step"]
        if n is None or cycle_step % n != 0:
            return
        prefix = f"cycle{context['cycle_count']}_" if context["num_cycles"] > 1 else ""
        if self._periodic is None:
            # An earlier attempt's periodic saves (before a resume) are older
            # than any this process writes.
            self._periodic = sorted(
                self.output_dir.glob(f"*_{self.PERIODIC_TAG}_step*.zip"),
                key=lambda p: p.stat().st_mtime_ns,
            )
        path = self._save(trainer, f"{prefix}{self.PERIODIC_TAG}_step{cycle_step}")
        if path is None:
            return  # not the main process
        self._periodic.append(Path(path))
        while len(self._periodic) > self.PERIODIC_KEEP:
            self._periodic.pop(0).unlink(missing_ok=True)

    def _save(self, trainer: "Trainer", suffix: str) -> Optional[Path]:
        """Save under ``suffix``; the written path on the main process, else None."""
        if self.save_training_state:
            return trainer.save_resumable(
                str(self.output_dir),
                suffix=suffix,
                notes=self.notes,
            )
        return trainer.save_checkpoint(
            str(self.output_dir),
            suffix=suffix,
            notes=self.notes,
        )

    def to_config(self) -> HookConfig:
        """Convert hook back to configuration for serialization.

        Returns:
            HookConfig with hook type and parameters
        """
        return HookConfig(  # type: ignore[call-arg]
            type="CheckpointSaveHook",
            output_dir=str(self.output_dir),
            notes=self.notes,
            checkpoint_every_n_evals=self.checkpoint_every_n_evals,
            save_training_state=self.save_training_state,
            checkpoint_every_n_steps=self.checkpoint_every_n_steps,
        )
