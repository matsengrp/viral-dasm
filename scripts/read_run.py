# ruff: noqa: I001 -- import order in this file is load-bearing; see below.
"""Read one finished run out of its own log, as one row.

Everything here comes from what the trainer wrote. Nothing is recomputed and
nothing is scored: ``data_passes`` is a logged field, ``wall_clock_s`` is the
sum of the ``training_end`` records that exist, and the checkpoint is
identified by its sha256 rather than described.

**The eval series goes through ``dasm2.training.log_reader``**, which is
written against exactly the resume behaviour ``scripts/train.py`` relies on.
``TrainingLogger.create`` mints a fresh run id per invocation, so a run that
was restarted after a preemption leaves two or more ``.jsonl`` files, and the
later one replays the evaluations between its checkpoint and wherever the
interrupted attempt had got to. A reader that takes the newest log starts at
the resume point; one that concatenates counts the replay twice.
``get_run_eval_losses`` dedupes on ``(cycle, cycle_step)``, later wins. It
also matters at scale: one full-depth run's log is about 150 MB, nearly all
of it per-batch events.

**A step-budget stop leaves no record**, which is why ``stop_reason`` has a
literal fallback. ``trainer.py`` breaks on ``self._cycle_step >=
budget.max_train_steps`` *before* ``_should_stop_early`` is called, and
``log_early_stopping`` fires only inside the branch that call opens, so there
is no ``"type": "early_stopping"`` record and no step-budget reason anywhere
in ``logging.py``. The signature is the absence of that event together with
a last step at the budget. ``scripts/check_run.py`` is what fails on it.

**Three stopping thresholds, one run.** ``early_stopping_min_delta`` is 0 for
every run here, which makes ``best_val_loss`` the validation curve's own
minimum. It reaches ``Trainer._update_best_model`` and the patience counter
and nothing else -- the plateau scheduler steps on the raw ``val_loss`` -- so
a coarser threshold would have produced the same descent and only a different
answer to "which evaluation counted". Those answers are recovered here by
replaying the trainer's own rule over the recorded series, which is why the
two reference campaigns' values (1e-3 for the sampling-depth ladder, 1e-4 for
the capacity scan) appear as derived columns rather than as a choice this
pipeline had to make.

GISAID: per-evaluation aggregate scalars only. No sequences, no per-row data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first: it is what puts this repository's root on `sys.path`.
from lib import ARTIFACT_ROOT, select_checkpoint
from runs import MAX_TRAIN_STEPS, MODELS

# `dasm2` next, above anything that pulls pyarrow in -- pandas does.
import dasm2  # noqa: F401  (imported for its side effects)

from dasm2.training.log_reader import (
    find_training_logs,
    get_run_eval_losses,
    read_log,
    read_metadata,
)

#: The thresholds replayed over this run's own evaluations, and the column
#: suffix each gets. ``logged`` is the value every run here trained under, so
#: those columns describe what actually happened; the other two describe what
#: the capacity scan's and the sampling-depth ladder's rules would have kept.
REPLAY_DELTAS = {
    "logged": 0.0,
    "at_min_delta_1e4": 1e-4,
    "at_min_delta_1e3": 1e-3,
}

#: ``early_stopping_patience`` in every template. The gate checked *before*
#: the learning-rate floor, so a coarser threshold can end a run earlier by
#: advancing this counter faster even though the descent is identical.
EARLY_STOPPING_PATIENCE = 30

#: What ``read_run`` writes when the run left no stop event at all. There is
#: no such reason string in ``dasm2/training/logging.py``; this names the
#: condition for a reader of ``data/runs.csv``.
BUDGET_STOP_REASON = "max_train_steps budget"


def replay(losses: list[float], steps: list[int], delta: float) -> dict:
    """``Trainer._update_best_model``'s rule, run again at another threshold.

    An improvement is ``val_loss < best - delta``, strictly; anything else
    increments the patience counter. Returns the best this threshold would
    have kept, and the step at which patience 30 would have ended the run --
    None when it never would have, which is the case for a run that reached
    the learning-rate floor.
    """
    best = float("inf")
    best_step = None
    without_improvement = 0
    patience_stop_step = None
    for loss, step in zip(losses, steps, strict=True):
        if loss < best - delta:
            best, best_step, without_improvement = loss, step, 0
        else:
            without_improvement += 1
            if (
                patience_stop_step is None
                and without_improvement >= EARLY_STOPPING_PATIENCE
            ):
                patience_stop_step = step
    return {
        "best_val_loss": None if best_step is None else best,
        "best_step": best_step,
        "patience_stop_step": patience_stop_step,
    }


def end_events(run_dir: Path) -> tuple[dict | None, list[dict]]:
    """The run's ``early_stopping`` record, if any, and every ``training_end``.

    Read across every attempt's log, because which attempt reached the end is
    not knowable in advance. Both events are written only by an attempt that
    *finished*: an attempt killed by a preemption or a walltime logs neither,
    and the one that resumes from its checkpoint logs both. So a resumed run
    normally yields one stop and one end, and the wall clock below is the
    time of the attempts that completed rather than the total this run spent
    on a GPU. Nothing records the latter.
    """
    stop = None
    ends = []
    for path in find_training_logs(run_dir):
        for record in read_log(path):
            kind = record.get("type")
            if kind == "early_stopping":
                stop = record
            elif kind == "training_end":
                ends.append(record)
    return stop, ends


def file_sha256(path: Path, chunk_bytes: int = 1 << 24) -> str:
    """sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def read_run(model: str, corpus_name: str, run_dir: Path) -> dict:
    """One row: what this run did, entirely from what it wrote."""
    frame = get_run_eval_losses(run_dir)
    if frame.empty:
        raise SystemExit(
            f"{run_dir} logged no evaluations, so there is no curve to read. "
            f"A run that reached its first `eval_every_steps` boundary has "
            f"at least one."
        )
    frame = frame.sort_values("global_step")
    losses = [float(v) for v in frame["val_loss"]]
    steps = [int(v) for v in frame["global_step"]]

    stop, ends = end_events(run_dir)
    checkpoint = select_checkpoint(run_dir)
    # The first attempt's metadata: the geometry, rate and split sizes are
    # the same in every attempt of one run, and the first is the one that
    # started from scratch.
    first_log = find_training_logs(run_dir)[0]
    meta = read_metadata(
        first_log.with_name(first_log.name.replace("_log.jsonl", "_meta.yaml"))
    )

    row = {
        "model": model,
        "corpus": corpus_name,
        "effective_params": MODELS[model],
        "learning_rate": meta["optimizer_config"]["learning_rate"],
        "n_train": meta["tasks"][0]["train_size"],
        "n_val": meta["tasks"][0]["val_size"],
        "evaluations": len(steps),
        "steps_to_stop": steps[-1],
        "stop_reason": stop["reason"] if stop else BUDGET_STOP_REASON,
        "final_lr": float(frame["lr"].iloc[-1]),
        "data_passes": float(frame["data_passes"].max()),
        # Summed over the attempts that finished; see `end_events`.
        "wall_clock_s": sum(float(e["total_time_s"]) for e in ends),
        "completed_attempts": len(ends),
        # What the saved checkpoint holds, as the trainer recorded it. Not
        # the same number as the log's argmin in general; at min_delta 0 the
        # two collapse, and emitting both is how that is shown rather than
        # asserted.
        "best_val_loss_saved": float(ends[-1]["final_best_val_loss"]) if ends else None,
        "checkpoint": str(checkpoint.relative_to(ARTIFACT_ROOT)),
        "checkpoint_sha256": file_sha256(checkpoint),
    }
    for suffix, delta in REPLAY_DELTAS.items():
        replayed = replay(losses, steps, delta)
        row[f"best_val_loss_{suffix}"] = replayed["best_val_loss"]
        row[f"best_step_{suffix}"] = replayed["best_step"]
        row[f"patience_stop_step_{suffix}"] = replayed["patience_stop_step"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    row = read_run(args.model, args.corpus, args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2, default=str))

    print(f"{row['model']} on {row['corpus']}")
    print(f"  stopped at step {row['steps_to_stop']:,}: {row['stop_reason']}")
    print(
        f"  best val {row['best_val_loss_logged']:.6f} at step "
        f"{row['best_step_logged']:,}; saved {row['best_val_loss_saved']:.6f}"
    )
    print(
        f"  {row['data_passes']:.1f} data passes, "
        f"{row['wall_clock_s'] / 3600:.1f} h over "
        f"{row['completed_attempts']} completed attempt(s)"
    )
    print(f"  budget was {MAX_TRAIN_STEPS:,} steps")


if __name__ == "__main__":
    main()
