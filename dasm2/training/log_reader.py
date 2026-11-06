"""Utilities for reading and analyzing training logs."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Canonical filename markers shared with TrainingLogger and the dashboard.
LOG_STEM_MARKER = "_log"
META_STEM_MARKER = "_meta"
LOG_EXTENSION = f"{LOG_STEM_MARKER}.jsonl"
META_EXTENSION = f"{META_STEM_MARKER}.yaml"


def read_log(log_path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate over log entries with error handling.

    Args:
        log_path: Path to the JSONL log file.

    Yields:
        Dictionary entries from the log file.

    Raises:
        json.JSONDecodeError: If a line contains malformed JSON.
    """
    with open(log_path) as f:
        for line_num, line in enumerate(f, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Malformed JSON at line {line_num} in {log_path}: {line.strip()}",
                    e.doc,
                    e.pos,
                ) from e


def read_metadata(meta_path: str | Path) -> dict[str, Any]:
    """Read YAML metadata file.

    Args:
        meta_path: Path to the YAML metadata file.

    Returns:
        Dictionary containing the metadata.

    Raises:
        ValueError: If metadata file does not contain a YAML dict.
    """
    with open(meta_path) as f:
        result = yaml.safe_load(f)
        if not isinstance(result, dict):
            raise ValueError(
                f"Metadata file {meta_path} must contain a YAML dict, "
                f"got {type(result).__name__}"
            )
        return result


def log_to_dataframe(
    log_path: str | Path, event_type: str | tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Convert log to DataFrame, optionally filtering by event type.

    Args:
        log_path: Path to the JSONL log file.
        event_type: Optional event type, or tuple of event types, to filter
            by (e.g., "batch", ("eval_end", "joint_eval_end")).

    Returns:
        DataFrame containing log entries.
    """
    entries = list(read_log(log_path))
    if event_type:
        allowed = (event_type,) if isinstance(event_type, str) else event_type
        entries = [e for e in entries if e.get("type") in allowed]
    return pd.DataFrame(entries)


def get_batch_losses(log_path: str | Path) -> pd.DataFrame:
    """Extract batch-level losses for analysis.

    Args:
        log_path: Path to the JSONL log file.

    Returns:
        DataFrame containing batch-level metrics.
    """
    return log_to_dataframe(log_path, event_type="batch")


def get_eval_losses(log_path: str | Path) -> pd.DataFrame:
    """Extract evaluation-boundary losses for analysis.

    Reads ``eval_end`` events (single-task runs) emitted at every
    ``eval_every_steps`` optimizer-step boundary plus the terminal eval at
    cycle end. Also reads ``joint_eval_end`` events (multi-task/joint
    training runs, ``TrainingLogger.log_joint_eval_end``) -- both event
    types carry the same ``cycle``/``cycle_step``/``global_step``/``is_best``
    fields consumers like :func:`dasm2.analysis.results_table.lookup_eval_at`
    and :func:`dasm2.training.checkpoint_manifest.build_manifest` key off of,
    just with different (per-task-flattened, for joint) metric columns
    alongside them. A single-task run's log never contains a
    ``joint_eval_end`` row and vice versa, so including both event types is a
    no-op for existing single-task callers.

    Args:
        log_path: Path to the JSONL log file.

    Returns:
        DataFrame containing per-evaluation metrics.
    """
    return log_to_dataframe(log_path, event_type=("eval_end", "joint_eval_end"))


def find_logs(
    output_dir: str | Path,
    config_name: str | None = None,
    max_depth: int | None = None,
) -> list[Path]:
    """Find all log files under output_dir.

    Searches for log files inside ``logs/`` directories at various nesting
    levels under *output_dir*.

    Args:
        output_dir: Base output directory containing log files.
        config_name: Optional config name to filter logs by prefix.
        max_depth: Maximum number of directory levels to descend below
            *output_dir* before the ``logs/`` directory.  ``0`` searches only
            ``output_dir/logs/``, ``1`` also searches ``output_dir/*/logs/``,
            etc.  *None* (default) searches all levels (``**/logs/``).

    Returns:
        List of paths to log files, sorted by modification time (oldest first).
    """
    base = Path(output_dir)
    filename_pattern = (
        f"{config_name}_*{LOG_EXTENSION}" if config_name else f"*{LOG_EXTENSION}"
    )
    if max_depth is None:
        patterns = [f"**/logs/{filename_pattern}"]
    else:
        patterns = []
        for d in range(max_depth + 1):
            prefix = "/".join(["*"] * d) + "/" if d > 0 else ""
            patterns.append(f"{prefix}logs/{filename_pattern}")

    matches: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for p in base.glob(pattern):
            if p not in seen:
                seen.add(p)
                matches.append(p)

    return sorted(matches, key=lambda p: p.stat().st_mtime)


def find_training_logs(output_dir: str | Path) -> list[Path]:
    """Every training log one run left under ``output_dir/logs``, oldest first.

    ``TrainingLogger.create`` mints a fresh run id per ``dasm2 train``
    invocation, so a run that the Snakefile's ``--resume`` path restarted
    after a preemption or a host crash leaves two or more ``.jsonl`` files
    behind. A caller that wants the run's whole history has to read all of
    them; one that reads only the newest sees a history starting at the resume
    point.

    Args:
        output_dir: A run's checkpoint directory.

    Returns:
        The run's logs, oldest-first by mtime, which is training order.

    Raises:
        RuntimeError: If the run wrote no log at all.
    """
    logs = find_logs(str(output_dir), max_depth=0)
    if not logs:
        raise RuntimeError(f"No training log under {output_dir}/logs")
    return logs


def get_run_eval_losses(output_dir: str | Path) -> pd.DataFrame:
    """One run's ``eval_end`` events, read across every log it wrote.

    A resumed run replays the boundaries between the checkpoint it resumed
    from and wherever the interrupted attempt had got to, so each of those
    steps is logged twice with different losses. The later log wins: those are
    the evals the surviving weights actually passed through, and the earlier
    log's tail belongs to an abandoned branch of training. Deduplicating is
    not optional -- a replayed boundary left in place silently turns one curve
    row into two.

    Args:
        output_dir: A run's checkpoint directory.

    Returns:
        One row per evaluation boundary, sorted by ``(cycle, cycle_step)``.
        Empty (and columnless) if the run logged no evals at all.

    Raises:
        RuntimeError: If the run wrote no log at all.
    """
    logs = find_training_logs(output_dir)
    events = pd.concat([get_eval_losses(log) for log in logs], ignore_index=True)
    # Checked before the dedupe, which keys on columns an empty frame does not
    # have -- a run that logged no evals would otherwise raise KeyError
    # instead of returning the empty frame its caller can report on.
    if events.empty:
        return events
    return (
        events.drop_duplicates(subset=["cycle", "cycle_step"], keep="last")
        .sort_values(["cycle", "cycle_step"], kind="stable")
        .reset_index(drop=True)
    )
