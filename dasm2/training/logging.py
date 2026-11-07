"""Training logger for streaming metrics to JSONL files with YAML metadata."""

import enum
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import torch
import yaml

from dasm2.training.log_reader import LOG_EXTENSION, META_EXTENSION

# Bounded retry budget for a metrics write; see ``TrainingLogger._append``
# for why a failed write is never allowed to reach the training loop.
_LOG_WRITE_MAX_ATTEMPTS = 6
_LOG_WRITE_BACKOFF_SEC = 0.5


class _MetadataDumper(yaml.SafeDumper):
    """YAML dumper that serializes enums as plain strings.

    ``yaml.SafeDumper`` cannot represent Python enum instances.  This
    subclass adds a representer for all ``enum.Enum`` subtypes so that
    values like ``HeadType.SELECTION_FACTOR_HEAD`` are written as
    ``SelectionFactorHead`` instead of raising ``RepresenterError``.
    """


def _represent_enum(dumper: yaml.SafeDumper, data: enum.Enum) -> yaml.Node:
    return dumper.represent_str(str(data.value))


def _represent_path(dumper: yaml.SafeDumper, data: Path) -> yaml.Node:
    """Serialize pathlib.Path (including PosixPath) as a string for YAML."""
    return dumper.represent_str(str(data))


_MetadataDumper.add_multi_representer(enum.Enum, _represent_enum)
_MetadataDumper.add_multi_representer(Path, _represent_path)


@runtime_checkable
class LoggerProtocol(Protocol):
    """Protocol defining the training logger interface.

    Both TrainingLogger and NullLogger implement this protocol,
    enabling clean type annotations without inheritance.
    """

    @property
    def run_id(self) -> Optional[str]:
        """Identifier of the run being logged, or None when logging is off."""
        ...

    def log_training_start(
        self,
        num_cycles: int,
        max_train_steps: int,
        eval_every_steps: int,
        batch_size: int,
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
        num_processes: int,
    ) -> None: ...
    def log_cycle_start(self, cycle: int, lr: float) -> None: ...
    def log_task_start(
        self, cycle: int, task: str, num_train_batches: int, num_val_batches: int
    ) -> None: ...
    def log_batch(
        self,
        cycle: int,
        cycle_step: int,
        global_step: int,
        data_passes: float,
        loader_cycle: int,
        loss: float,
        batch_time_ms: float,
        batch_size: int,
        task: str,
        skipped_steps: Optional[int] = None,
    ) -> None: ...
    def log_joint_batch(
        self,
        cycle: int,
        cycle_step: int,
        global_step: int,
        data_passes: float,
        loader_cycle: int,
        total_loss: float,
        per_task_losses: Dict[str, float],
        batch_time_ms: float,
        batch_size: int,
        skipped_steps: Optional[int] = None,
    ) -> None: ...
    def log_eval_end(
        self,
        cycle: int,
        cycle_step: int,
        global_step: int,
        data_passes: float,
        loader_cycle: int,
        task: str,
        avg_train_loss: float,
        val_loss: float,
        eval_steps: int,
        eval_time_s: float,
        lr: float,
        is_best: bool,
        best_val_loss: float,
        peak_memory_mb: Optional[float] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        optimizer_step_metrics: Optional[Dict[str, float]] = None,
    ) -> None: ...
    def log_joint_eval_end(
        self,
        cycle: int,
        cycle_step: int,
        global_step: int,
        data_passes: float,
        loader_cycle: int,
        avg_train_loss: float,
        val_loss: float,
        per_task_train: Dict[str, float],
        per_task_val: Dict[str, float],
        eval_steps: int,
        eval_time_s: float,
        lr: float,
        is_best: bool,
        best_val_loss: float,
        peak_memory_mb: Optional[float] = None,
        per_task_val_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        optimizer_step_metrics: Optional[Dict[str, float]] = None,
        monitored_val_loss: Optional[float] = None,
    ) -> None: ...
    def log_hook_start(self, cycle: int, hook: str, phase: str, task: str) -> None: ...
    def log_hook_end(
        self, cycle: int, hook: str, phase: str, task: str, duration_s: float
    ) -> None: ...
    def log_early_stopping(
        self,
        cycle: int,
        cycle_step: int,
        global_step: int,
        reason: str,
        final_lr: float,
    ) -> None: ...
    def log_cycle_end(
        self, cycle: int, cycle_time_s: float, best_val_loss: float
    ) -> None: ...
    def log_checkpoint_saved(self, cycle: int, path: str) -> None: ...
    def log_training_end(
        self, total_time_s: float, final_best_val_loss: float
    ) -> None: ...


class NullLogger:
    """No-op logger implementing Null Object Pattern.

    Uses __getattr__ to dynamically provide all logger methods as no-ops.
    This avoids implementing 18+ protocol methods explicitly while maintaining
    compatibility with LoggerProtocol through duck typing.
    """

    #: Declared explicitly so it shadows ``__getattr__``, which would
    #: otherwise hand back a no-op *callable* for ``run_id`` and make
    #: "is there a run to name checkpoints after?" read as True.
    run_id: Optional[str] = None

    def __getattr__(self, name: str) -> Any:
        """Return no-op callable for any attribute."""

        def noop(*args: Any, **kwargs: Any) -> None:
            pass

        return noop


def _json_serializer(obj: Any) -> Any:
    """Handle special types for JSON serialization.

    Supported: torch.Tensor, pathlib.Path, numpy arrays.
    Raises TypeError for unsupported types (fail-fast).
    """
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist") and hasattr(obj, "__array__"):  # numpy
        return obj.tolist()

    raise TypeError(
        f"Cannot serialize {type(obj).__name__} to JSON. "
        f"Add explicit handling in _json_serializer(). Value: {obj!r}"
    )


@dataclass
class TrainingLogger:
    """Streams training metrics to JSONL with linked YAML metadata.

    Step-based counters (``global_step``, ``cycle_step``, ``data_passes``,
    ``loader_cycle``) are passed in by the trainer rather than maintained
    as mutable state on the logger.
    """

    log_dir: Path
    run_id: str
    meta_path: Path = field(init=False)
    log_path: Path = field(init=False)
    _start_time: float = field(default_factory=time.perf_counter)
    #: Metrics lines discarded after exhausting the write retry budget.
    dropped_lines: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)
        self.meta_path = self.log_dir / f"{self.run_id}{META_EXTENSION}"
        self.log_path = self.log_dir / f"{self.run_id}{LOG_EXTENSION}"

    @classmethod
    def create(
        cls,
        output_dir: str,
        config_name: Optional[str],
        metadata: Dict[str, Any],
    ) -> "TrainingLogger":
        """Create logger with auto-generated run ID."""
        log_dir_path = Path(output_dir) / "logs"
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # Generate run ID with timestamp + random suffix (no metadata dependency)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        random_suffix = os.urandom(3).hex()
        # Use stem (basename without extension) to avoid path separators in run_id
        config_prefix = Path(config_name).stem if config_name else "run"
        run_id = f"{config_prefix}_{timestamp}_{random_suffix}"

        logger = cls(log_dir=log_dir_path, run_id=run_id)

        # Write metadata atomically (run_id serves as unique identifier)
        full_metadata = {
            "run_id": run_id,
            "log_file": f"{run_id}{LOG_EXTENSION}",
            "started_at": cls._timestamp(),
            **metadata,
        }
        temp_path = logger.meta_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            yaml.dump(
                full_metadata,
                f,
                Dumper=_MetadataDumper,
                default_flow_style=False,
                sort_keys=False,
            )
        temp_path.rename(logger.meta_path)

        return logger

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append(self, entry: Dict[str, Any]) -> None:
        """Append entry to the log file, best-effort.

        This is telemetry: a transient filesystem error (a shared-scratch
        mount blipping out under a routine write) must never abort a
        multi-day training run. Retry a few times with backoff, recreating
        the parent directory in case it vanished with the mount, then warn
        and drop the line rather than raising into the training loop.
        """
        entry["timestamp"] = self._timestamp()
        line = json.dumps(entry, default=_json_serializer) + "\n"
        last_err: Optional[OSError] = None
        for attempt in range(_LOG_WRITE_MAX_ATTEMPTS):
            try:
                with open(self.log_path, "a") as f:
                    f.write(line)
                    f.flush()
                return
            except OSError as e:
                last_err = e
                # The parent dir may have momentarily disappeared with a
                # shared-FS blip; try to recreate it before retrying.
                try:
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass
                if attempt < _LOG_WRITE_MAX_ATTEMPTS - 1:
                    time.sleep(_LOG_WRITE_BACKOFF_SEC * (2**attempt))
        # The running count keeps each message distinct, so the default
        # warning filter (first occurrence per location) can't hide how much
        # telemetry a long outage actually cost.
        self.dropped_lines += 1
        warnings.warn(
            f"Dropping metrics log line after {_LOG_WRITE_MAX_ATTEMPTS} failed "
            f"writes to {self.log_path}: {last_err}. Training continues; "
            f"{self.dropped_lines} line(s) dropped so far this run.",
            stacklevel=2,
        )

    # --- Event logging methods ---

    def log_training_start(
        self,
        num_cycles: int,
        max_train_steps: int,
        eval_every_steps: int,
        batch_size: int,
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
        num_processes: int,
    ) -> None:
        """Log training start event with batch size and step-budget breakdown."""
        if num_cycles <= 0 or max_train_steps <= 0 or eval_every_steps <= 0:
            raise ValueError(
                "num_cycles, max_train_steps, and eval_every_steps must all be positive"
            )
        self._append(
            {
                "type": "training_start",
                "num_cycles": num_cycles,
                "max_train_steps": max_train_steps,
                "eval_every_steps": eval_every_steps,
                "batch_size": batch_size,
                "per_device_batch_size": per_device_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_processes": num_processes,
            }
        )

    def log_cycle_start(self, cycle: int, lr: float) -> None:
        """Log cycle start event."""
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"Learning rate must be positive and finite, got {lr}")
        self._append({"type": "cycle_start", "cycle": cycle, "lr": lr})

    def log_task_start(
        self, cycle: int, task: str, num_train_batches: int, num_val_batches: int
    ) -> None:
        """Log task start event."""
        if num_train_batches <= 0 or num_val_batches <= 0:
            raise ValueError("Batch counts must be positive")
        self._append(
            {
                "type": "task_start",
                "cycle": cycle,
                "task": task,
                "num_train_batches": num_train_batches,
                "num_val_batches": num_val_batches,
            }
        )

    def log_batch(
        self,
        cycle: int,
        cycle_step: int,
        global_step: int,
        data_passes: float,
        loader_cycle: int,
        loss: float,
        batch_time_ms: float,
        batch_size: int,
        task: str,
        skipped_steps: Optional[int] = None,
    ) -> None:
        """Log per-step batch metrics with fail-fast validation.

        ``skipped_steps`` is the trainer's count of non-finite windows skipped
        since its process started, so per attempt, not per run: a scan of a
        resumed run adds the last count of each of its logs. A skipped step
        itself logs ``loss`` 0.0.
        """
        if not math.isfinite(loss):
            raise ValueError(f"Loss is not finite: {loss}")
        if batch_time_ms < 0:
            raise ValueError(f"Batch time cannot be negative: {batch_time_ms}")
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive: {batch_size}")

        self._append(
            {
                "type": "batch",
                "cycle": cycle,
                "cycle_step": cycle_step,
                "global_step": global_step,
                "data_passes": data_passes,
                "loader_cycle": loader_cycle,
                "loss": loss,
                "batch_time_ms": batch_time_ms,
                "batch_size": batch_size,
                "task": task,
                **({} if skipped_steps is None else {"skipped_steps": skipped_steps}),
            }
        )

    def log_joint_batch(
        self,
        cycle: int,
        cycle_step: int,
        global_step: int,
        data_passes: float,
        loader_cycle: int,
        total_loss: float,
        per_task_losses: Dict[str, float],
        batch_time_ms: float,
        batch_size: int,
        skipped_steps: Optional[int] = None,
    ) -> None:
        """Log per-step joint batch metrics (``skipped_steps`` as in :meth:`log_batch`)."""
        if not math.isfinite(total_loss):
            raise ValueError(f"Total loss is not finite: {total_loss}")
        entry = {
            "type": "joint_batch",
            "cycle": cycle,
            "cycle_step": cycle_step,
            "global_step": global_step,
            "data_passes": data_passes,
            "loader_cycle": loader_cycle,
            "total_loss": total_loss,
            "batch_time_ms": batch_time_ms,
            "batch_size": batch_size,
            **({} if skipped_steps is None else {"skipped_steps": skipped_steps}),
        }
        # Flatten per-task losses to top level
        entry.update(per_task_losses)
        self._append(entry)

    def log_eval_end(
        self,
        cycle: int,
        cycle_step: int,
        global_step: int,
        data_passes: float,
        loader_cycle: int,
        task: str,
        avg_train_loss: float,
        val_loss: float,
        eval_steps: int,
        eval_time_s: float,
        lr: float,
        is_best: bool,
        best_val_loss: float,
        peak_memory_mb: Optional[float] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        optimizer_step_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log evaluation-boundary metrics.

        ``avg_train_loss`` is the position-weighted mean over the
        ``eval_every_steps`` optimizer steps preceding this eval — a rolling
        window, not a per-epoch summary. The field is renamed from the old
        ``train_loss`` deliberately to prevent silent quantitative shifts in
        downstream dashboards.

        ``val_metrics`` is one flat dict merging every
        validation metric group eligible for this task — child-codon
        prediction and selection-factor distribution (#678 §7).
        Keys are emitted at the top level prefixed with ``val_`` (e.g.
        ``val_codon_acc_mutated``, ``val_sf_frac_positive``), exactly as the
        previously separate ``val_codon_metrics`` / ``val_sf_metrics`` did,
        so the on-disk schema is unchanged. ``optimizer_step_metrics`` is a
        per-window summary of per-step grad norm / clip rate written
        verbatim.
        """
        if not (math.isfinite(avg_train_loss) and math.isfinite(val_loss)):
            raise ValueError("Train/val losses must be finite")
        entry: Dict[str, Any] = {
            "type": "eval_end",
            "cycle": cycle,
            "cycle_step": cycle_step,
            "global_step": global_step,
            "data_passes": data_passes,
            "loader_cycle": loader_cycle,
            "task": task,
            "avg_train_loss": avg_train_loss,
            "val_loss": val_loss,
            "eval_steps": eval_steps,
            "eval_time_s": eval_time_s,
            "lr": lr,
            "is_best": is_best,
            "best_val_loss": best_val_loss,
        }
        if peak_memory_mb is not None:
            entry["peak_memory_mb"] = peak_memory_mb
        if val_metrics is not None:
            for key, value in val_metrics.items():
                entry[f"val_{key}"] = value
        if optimizer_step_metrics is not None:
            for key, value in optimizer_step_metrics.items():
                entry[key] = value
        self._append(entry)

    def log_joint_eval_end(
        self,
        cycle: int,
        cycle_step: int,
        global_step: int,
        data_passes: float,
        loader_cycle: int,
        avg_train_loss: float,
        val_loss: float,
        per_task_train: Dict[str, float],
        per_task_val: Dict[str, float],
        eval_steps: int,
        eval_time_s: float,
        lr: float,
        is_best: bool,
        best_val_loss: float,
        peak_memory_mb: Optional[float] = None,
        per_task_val_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        optimizer_step_metrics: Optional[Dict[str, float]] = None,
        monitored_val_loss: Optional[float] = None,
    ) -> None:
        """Log joint evaluation-boundary metrics with per-task losses.

        ``per_task_val_metrics`` maps task_key to one flat dict
        merging every validation metric group eligible for that task —
        child-codon prediction and selection-factor distribution
        (#678 §7). Keys are flattened as ``val_{task_key}_{metric_name}`` to
        mirror the per-task loss flatten pattern and stay queryable from the
        JSONL log; the on-disk schema is unchanged from the previously
        separate codon / sf payloads. ``optimizer_step_metrics`` is a
        per-window summary written verbatim.

        ``val_loss`` is the sum of the tasks' weighted validation losses.
        ``monitored_val_loss`` (written only when given, i.e. when the trainer has
        a ``monitor_task``) is the single task's loss that early stopping follows;
        ``is_best`` and ``best_val_loss`` then refer to it, not to ``val_loss``.
        """
        if not (math.isfinite(avg_train_loss) and math.isfinite(val_loss)):
            raise ValueError("Train/val losses must be finite")
        entry: Dict[str, Any] = {
            "type": "joint_eval_end",
            "cycle": cycle,
            "cycle_step": cycle_step,
            "global_step": global_step,
            "data_passes": data_passes,
            "loader_cycle": loader_cycle,
            "avg_train_loss": avg_train_loss,
            "val_loss": val_loss,
            "eval_steps": eval_steps,
            "eval_time_s": eval_time_s,
            "lr": lr,
            "is_best": is_best,
            "best_val_loss": best_val_loss,
        }
        if peak_memory_mb is not None:
            entry["peak_memory_mb"] = peak_memory_mb
        if monitored_val_loss is not None:
            entry["monitored_val_loss"] = monitored_val_loss
        # Flatten per-task losses with train_/val_ prefixes
        for key, val in per_task_train.items():
            entry[f"train_{key}"] = val
        for key, val in per_task_val.items():
            entry[f"val_{key}"] = val
        if per_task_val_metrics is not None:
            for task_key, task_metrics in per_task_val_metrics.items():
                for metric_name, metric_value in task_metrics.items():
                    entry[f"val_{task_key}_{metric_name}"] = metric_value
        if optimizer_step_metrics is not None:
            for key, value in optimizer_step_metrics.items():
                entry[key] = value
        self._append(entry)

    def log_hook_start(self, cycle: int, hook: str, phase: str, task: str) -> None:
        """Log hook start event."""
        self._append(
            {
                "type": "hook_start",
                "cycle": cycle,
                "hook": hook,
                "phase": phase,
                "task": task,
            }
        )

    def log_hook_end(
        self, cycle: int, hook: str, phase: str, task: str, duration_s: float
    ) -> None:
        """Log hook end event with duration."""
        self._append(
            {
                "type": "hook_end",
                "cycle": cycle,
                "hook": hook,
                "phase": phase,
                "task": task,
                "duration_s": duration_s,
            }
        )

    def log_early_stopping(
        self,
        cycle: int,
        cycle_step: int,
        global_step: int,
        reason: str,
        final_lr: float,
    ) -> None:
        """Log early stopping event."""
        self._append(
            {
                "type": "early_stopping",
                "cycle": cycle,
                "cycle_step": cycle_step,
                "global_step": global_step,
                "reason": reason,
                "final_lr": final_lr,
            }
        )

    def log_cycle_end(
        self, cycle: int, cycle_time_s: float, best_val_loss: float
    ) -> None:
        """Log cycle end event.

        ``best_val_loss`` is on the scale that picked the best checkpoint: the
        monitored task's own loss when the trainer has a ``monitor_task``, the
        logged ``val_loss`` otherwise.
        """
        self._append(
            {
                "type": "cycle_end",
                "cycle": cycle,
                "cycle_time_s": cycle_time_s,
                "best_val_loss": best_val_loss,
            }
        )

    def log_checkpoint_saved(self, cycle: int, path: str) -> None:
        """Log checkpoint saved event."""
        self._append({"type": "checkpoint_saved", "cycle": cycle, "path": path})

    def log_training_end(self, total_time_s: float, final_best_val_loss: float) -> None:
        """Log training end event.

        ``final_best_val_loss`` is on the same scale as ``cycle_end``'s
        ``best_val_loss`` (see :meth:`log_cycle_end`).
        """
        self._append(
            {
                "type": "training_end",
                "total_time_s": total_time_s,
                "final_best_val_loss": final_best_val_loss,
            }
        )
