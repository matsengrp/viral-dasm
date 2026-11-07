# ruff: noqa: I001 -- import order in this file is load-bearing; see below.
"""Gate 3: compare a full run's endpoint against the reference run's.

The reference stopped at step 436,000 with a best validation loss of
7.766710 at step 412,000, after 38.9 data passes. It stopped on the
**learning-rate floor**, not on patience: 1e-3 x 0.5^10 = 9.77e-7 < 1e-6,
and patience 30 never fires -- the last improvement was 12 evaluations
before the stop, nowhere near 30.

So this reports three things, and steps-to-stop matters as much as the loss.
A run that matches the loss but stops 100,000 steps earlier is telling you
something, and after `scripts/check_rendered_config.py` we know exactly which
three library defaults it would be telling you about.

**No tolerance gate on the loss.** The two 20,000-step canaries tracked the
reference to within 0.012 nats with no drift, and their seed-to-seed spread
was wider than either one's gap to the reference -- so a threshold here would
be measuring the seed, not the port. The numbers are reported and compared;
judging them is a human's job. What *is* gated is the stop reason, because
"stopped on the LR floor" is a structural fact about the schedule rather than
a draw from a distribution.

GISAID: aggregate per-evaluation scalars only.
"""

import dasm2  # noqa: F401  (imported first for its side effects)

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import reference_run

from dasm2.training.log_reader import get_run_eval_losses

#: What the reference's stop reason has to say. The schedule reaching its
#: floor is the designed ending; anything else means the run ended for a
#: reason the reference did not have.
EXPECTED_STOP_SUBSTRING = "LR"


def read_stop_event(run_dir: Path) -> dict | None:
    """The ``early_stopping`` record, if the run logged one."""
    for path in sorted(Path(run_dir, "logs").glob("*_log.jsonl")):
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("type") == "early_stopping":
                return record
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model", default="dasm-85k")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    reference = reference_run(args.model)
    frame = get_run_eval_losses(str(args.run_dir)).sort_values("global_step")
    best = frame.loc[frame["val_loss"].idxmin()]

    measured = {
        "best_val_loss": float(best["val_loss"]),
        "best_step": int(best["global_step"]),
        "last_step": int(frame["global_step"].max()),
        "evaluations": len(frame),
        "data_passes": float(frame["data_passes"].max()),
        "final_lr": float(frame["lr"].iloc[-1]),
        "best_avg_train_loss": float(best["avg_train_loss"]),
        "best_val_aa_acc_mutated": float(best["val_aa_acc_mutated"]),
    }
    stop = read_stop_event(args.run_dir)
    measured["stop_reason"] = stop.get("reason") if stop else None
    measured["stop_step"] = stop.get("global_step") if stop else None

    rows = [
        ("best val_loss", measured["best_val_loss"], reference.best_val_loss),
        ("best step", measured["best_step"], reference.best_step),
        ("stop step", measured["stop_step"], reference.stop_step),
    ]
    print(f"run:       {args.run_dir}")
    print(f"reference: {reference.model} / {reference.checkpoints_dir.parent.parent}")
    print()
    print(f"{'quantity':>18} {'this run':>14} {'reference':>14} {'delta':>12}")
    print("-" * 62)
    for name, mine, theirs in rows:
        if mine is None:
            print(f"{name:>18} {'(none)':>14} {theirs:>14} {'-':>12}")
            continue
        delta = mine - theirs
        fmt = ".6f" if isinstance(theirs, float) else ",d"
        print(f"{name:>18} {mine:>14{fmt}} {theirs:>14{fmt}} {delta:>+12{fmt}}")
    print()
    print(f"  stop reason:  {measured['stop_reason']!r}")
    print(f"                reference: {reference.stop_reason!r}")
    print(f"  final lr:     {measured['final_lr']:.6g}")
    print(f"  data passes:  {measured['data_passes']:.1f}  (reference 38.9)")
    print(f"  evaluations:  {measured['evaluations']}")

    failures = []
    if measured["stop_reason"] is None:
        failures.append(
            "the run logged no early_stopping record, so it did not stop on "
            "its own -- it was killed, hit the step cap, or is unfinished. "
            "The reference stopped on the learning-rate floor, and an endpoint "
            "comparison against a run that was interrupted compares nothing."
        )
    elif EXPECTED_STOP_SUBSTRING not in str(measured["stop_reason"]):
        failures.append(
            f"the run stopped for {measured['stop_reason']!r}, not on the "
            f"learning-rate floor. The reference's stop step is set entirely "
            f"by three library defaults -- min_learning_rate 1e-6, plateau "
            f"factor 0.5, plateau eval_patience 10 -- so a different ending "
            f"means one of them differs here."
        )

    report = {
        "run_dir": str(args.run_dir),
        "model": args.model,
        "measured": measured,
        "reference": {
            "best_val_loss": reference.best_val_loss,
            "best_step": reference.best_step,
            "stop_step": reference.stop_step,
            "stop_reason": reference.stop_reason,
        },
        "passed": not failures,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))

    if failures:
        raise SystemExit("\nGate 3 failed:\n" + "\n".join(f"  - {f}" for f in failures))
    print(
        "\nGate 3 passed: the run stopped on the LR floor; the losses above "
        "are reported for judgement, not thresholded."
    )


if __name__ == "__main__":
    main()
