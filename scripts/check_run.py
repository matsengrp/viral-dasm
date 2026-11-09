"""Fail a run that did not stop the way every run in this paper stops.

One stopping rule for all seven: train until ``ReduceLROnPlateau`` has driven
the learning rate below ``min_learning_rate``. The step is then a reading
rather than a setting -- it follows from the initial rate, the plateau factor
and that floor -- which is what makes steps-to-stop comparable across the
model-size supplement and the sampling-depth ladder.

Two other endings are possible, and both have to be loud.

**The step budget.** ``trainer.py`` breaks on ``self._cycle_step >=
budget.max_train_steps`` *before* ``_should_stop_early`` is called, so a run
that exhausts its budget writes no stop event at all -- there is no
``"type": "early_stopping"`` record and no step-budget reason anywhere in
``dasm2/training/logging.py``. ``scripts/read_run.py`` writes the literal
``max_train_steps budget`` when the event is absent, and this fails on it.
A capped run's best validation loss is a floor rather than a converged value,
and three arms of the capacity scan ended that way.

**Patience.** ``_should_stop_early`` checks ``evals_without_improvement >=
early_stopping_patience`` before it looks at the learning rate. Every
template sets that to ``null``, so no run here can stop that way and the two
remaining outcomes are the floor and the budget. The check still reports a
patience stop rather than assuming it away: a template edited back to a
number would otherwise end an arm by a rule the size comparison does not
share. Its message can contain the letters ``LR`` (``min_delta=...`` is
appended to it), which is why the test below is on the ``LR `` prefix rather
than on a substring.

This is a gate, not a readout: the row is already written when this runs, so
a run that fails here stops its own branch of the DAG with its curve in hand
while the other six finish. Raising the budget, or accepting a patience stop,
is then a decision someone makes rather than one a pipeline absorbs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runs import MAX_TRAIN_STEPS

#: The learning-rate floor message's prefix, as ``trainer.py`` formats it:
#: ``f"LR {current_lr:.2e} < {self.min_learning_rate:.2e}"``.
FLOOR_PREFIX = "LR "


def failures_for(row: dict) -> list[str]:
    """Why this run's ending is not the one the paper reports, if it is not."""
    reason = str(row.get("stop_reason", ""))
    steps = row.get("steps_to_stop")
    if reason.startswith(FLOOR_PREFIX):
        return []
    if steps == MAX_TRAIN_STEPS:
        return [
            (
                f"this run reached the {MAX_TRAIN_STEPS:,}-step budget without "
                f"stopping on the learning-rate floor, so its best validation "
                f"loss of {row.get('best_val_loss_logged')!r} is a floor rather "
                f"than a converged value and is not comparable to the other six. "
                f"Raise the budget for this arm, or report it as capped."
            )
        ]
    return [
        (
            f"this run stopped for {reason!r}, not on the learning-rate floor. "
            f"Patience is checked before the floor is, so a run that stops "
            f"improving ends after about three plateau reductions rather than "
            f"ten -- a different stopping rule, reached by accident."
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    row = json.loads(args.row.read_text())
    failures = failures_for(row)

    print(f"{row['model']} on {row['corpus']}")
    print(f"  stop_reason:   {row['stop_reason']!r}")
    print(f"  steps_to_stop: {row['steps_to_stop']:,} of {MAX_TRAIN_STEPS:,}")
    print(f"  final_lr:      {row['final_lr']:.6g}")

    report = {
        "model": row["model"],
        "corpus": row["corpus"],
        "stop_reason": row["stop_reason"],
        "steps_to_stop": row["steps_to_stop"],
        "final_lr": row["final_lr"],
        "passed": not failures,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))

    if failures:
        raise SystemExit(
            "run check failed:\n" + "\n".join(f"  - {f}" for f in failures)
        )
    print("run check passed: stopped on the learning-rate floor")


if __name__ == "__main__":
    main()
