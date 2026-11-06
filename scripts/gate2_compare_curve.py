"""Gate 2: compare the canary's first ten evaluations against the reference.

The gate that runs on every port change. What it catches, in hours rather
than days: a dropped censoring source, a lost mask, the wrong reference
sequence, a fold off by one, a wrong ``min_delta``. All of those move the
curve from the very first evaluation.

**What it cannot catch, despite looking as though it should: the LR
schedule.** The reference's learning rate is a constant 1e-3 through step
130,000, so a constant-LR run -- or a plateau scheduler with a different
``factor`` or ``eval_patience`` -- produces an identical first ten
evaluations. ``scripts/check_rendered_config.py`` asserts the scheduler
directly instead. What this script does check is that the canary's own LR
stayed at 1e-3, which is the part of the schedule that is visible this early.

**The tolerance is measured, not argued.** Neither this run nor the full one
can be bit-exact: the reference trained under flash attention, and trajectory
chaos compounds over tens of thousands of optimizer steps. So ``--tolerance``
has no default. Run the canary twice, differing only in ``--seed``, take the
per-row spread, and pass that. Until then this script reports the deltas and
checks only what does not depend on a tolerance -- row count, step alignment,
and the constant learning rate.

Two existing anchors bound the answer from above, both from
``2026-09-21-v12-replicate-training/README.md``: the replicate best-loss
standard deviation is 0.0055 nats, and the smallest effect this corpus has
resolved is 0.0033 nats. A tolerance looser than 0.0055 is not measuring
anything.

``is_best`` is reported but not gated, for the same reason. It is the only
column that sees ``early_stopping_min_delta`` -- at 1e-4 the reference's
pattern is T T T T F T T T T T, and at the sampling-depth ladder's 1e-3 the
step-16,000 row flips to F -- but the F at step 10,000 is there because that
evaluation came out *worse* than the one before it, which is a property of
the reference's trajectory and not of the configuration. A different seed may
legitimately not reproduce it. A mismatch is printed loudly and left for a
human; a mismatch at step 16,000 specifically is the ``min_delta`` signature.

GISAID: per-evaluation aggregate scalars. No sequence, no per-sample data.
"""

# ruff: noqa: I001 -- import order in this file is load-bearing; see below.
from __future__ import annotations

# `dasm2` first, above anything that pulls pyarrow in -- pandas does. Arrow
# fixes its C++ default memory pool at first import, and `dasm2/__init__.py`
# calls `use_system_arrow_pool()` to steer it off mimalloc, which under
# pyarrow 25 requests transparent huge pages. Whichever import lands first
# wins, and dasm2's complaint about losing goes to a Snakemake log nobody
# reads. `tests/test_import_order.py` keeps this honest.
import dasm2  # noqa: F401  (imported first for its side effects)

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import CANARY_STEPS, EVAL_EVERY_STEPS

from dasm2.training.log_reader import get_run_eval_losses

#: Columns compared numerically.
NUMERIC_COLUMNS = ("avg_train_loss", "val_loss", "val_aa_acc_mutated")

#: The learning rate the reference holds through this whole window.
EXPECTED_LR = 1e-3

#: The step whose ``is_best`` flips if ``early_stopping_min_delta`` is the
#: ladder's 1e-3 rather than this experiment's 1e-4.
MIN_DELTA_SIGNATURE_STEP = 16000


def wall_clock(measured: pd.DataFrame) -> dict:
    """Seconds to the last evaluation, split into training and evaluation.

    Taken from the log's own timestamps rather than from a stopwatch around
    the rule, so a queued or resumed run does not inflate it. The first
    evaluation's timestamp is the end of the first ``eval_every_steps``
    window, so the span it covers is every step after the first one; the
    per-step figure is reported against that count rather than the full
    budget.
    """
    stamps = pd.to_datetime(measured["timestamp"])
    total_s = float((stamps.iloc[-1] - stamps.iloc[0]).total_seconds())
    eval_s = float(measured["eval_time_s"].iloc[1:].sum())
    steps = int(measured["global_step"].iloc[-1] - measured["global_step"].iloc[0])
    train_s = total_s - eval_s
    return {
        "total_s": total_s,
        "eval_s": eval_s,
        "train_s": train_s,
        "eval_fraction": eval_s / total_s if total_s else 0.0,
        "steps_measured": steps,
        "ms_per_step": 1000.0 * train_s / steps if steps else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-dir", required=True, type=Path)
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help=(
            "per-row absolute tolerance on the loss columns, measured from "
            "two canary runs differing only in --seed. Omitted, the loss "
            "comparison is reported but not judged."
        ),
    )
    parser.add_argument("--through-step", type=int, default=CANARY_STEPS)
    args = parser.parse_args()

    reference = pd.read_csv(args.reference_csv)
    measured = get_run_eval_losses(str(args.canary_dir))
    measured = (
        measured[measured["global_step"] <= args.through_step]
        .sort_values("global_step")
        .reset_index(drop=True)
    )

    failures: list[str] = []
    warnings: list[str] = []

    expected_rows = args.through_step // EVAL_EVERY_STEPS
    if len(measured) != expected_rows:
        raise SystemExit(
            f"the canary logged {len(measured)} evaluations at or before step "
            f"{args.through_step:,}, not {expected_rows}. The run did not "
            f"reach its budget, or its logs were not concatenated as expected."
        )
    if list(measured["global_step"]) != list(reference["global_step"]):
        raise SystemExit(
            f"the canary evaluated at steps {list(measured['global_step'])}, "
            f"the reference at {list(reference['global_step'])}. These have to "
            f"line up before any column can be compared."
        )

    merged = reference.merge(
        measured[["global_step", *NUMERIC_COLUMNS, "is_best", "lr"]].rename(
            columns={"lr": "lr_canary"}
        ),
        on="global_step",
        suffixes=("_ref", "_canary"),
    )
    for column in NUMERIC_COLUMNS:
        merged[f"{column}_delta"] = merged[f"{column}_canary"] - merged[f"{column}_ref"]

    print(f"canary: {args.canary_dir}")
    print(f"reference: {args.reference_csv}")
    print()
    header = (
        f"{'step':>7} {'val_ref':>10} {'val_canary':>11} {'delta':>10} "
        f"{'train_delta':>12} {'aa_acc_delta':>13} {'is_best':>16} {'lr':>8}"
    )
    print(header)
    print("-" * len(header))
    for _, row in merged.iterrows():
        best = f"{row['is_best_ref']!s:>5} / {row['is_best_canary']!s:<5}"
        flag = "" if row["is_best_ref"] == row["is_best_canary"] else "  <-"
        print(
            f"{int(row['global_step']):>7} {row['val_loss_ref']:>10.6f} "
            f"{row['val_loss_canary']:>11.6f} {row['val_loss_delta']:>+10.6f} "
            f"{row['avg_train_loss_delta']:>+12.6f} "
            f"{row['val_aa_acc_mutated_delta']:>+13.6f} "
            f"{best:>16}{flag} {row['lr_canary']:>8.1e}"
        )
    print()

    worst = {
        column: float(merged[f"{column}_delta"].abs().max())
        for column in NUMERIC_COLUMNS
    }
    for column, value in worst.items():
        print(f"  max |delta| {column}: {value:.6f}")

    # The wall clock, which is the other thing this run is for. The reference
    # reached step 20,000 in 50.6 minutes *with flash attention*; flash-off is
    # slower, and how much slower sets the nine-run budget in matsengrp/
    # viral-dasm#1, which currently costs M3 at flash-on rates.
    timing = wall_clock(measured)
    print()
    print(
        f"  wall clock to step {args.through_step:,}: {timing['total_s'] / 3600:.2f} h"
    )
    print(
        f"    of which evaluation: {timing['eval_s'] / 3600:.2f} h "
        f"({timing['eval_fraction']:.0%})"
    )
    print(
        f"    training only:       {timing['train_s'] / 3600:.2f} h "
        f"= {timing['ms_per_step']:.1f} ms/step"
    )

    # Checks that do not need a tolerance.
    off_lr = merged[merged["lr_canary"] != EXPECTED_LR]
    if not off_lr.empty:
        failures.append(
            f"the canary's learning rate left {EXPECTED_LR:.0e} at step(s) "
            f"{list(off_lr['global_step'])}. The reference holds 1e-3 through "
            f"step 130,000, so a reduction this early means the plateau "
            f"scheduler is configured differently."
        )

    best_mismatches = merged[merged["is_best_ref"] != merged["is_best_canary"]]
    if not best_mismatches.empty:
        steps = list(best_mismatches["global_step"])
        message = (
            f"is_best differs from the reference at step(s) {steps}. This "
            f"column is the only one that sees early_stopping_min_delta, but "
            f"it is also a property of the trajectory, so a different seed "
            f"may legitimately differ. Judge it against the per-row spread of "
            f"the two-seed measurement."
        )
        if MIN_DELTA_SIGNATURE_STEP in steps:
            message += (
                f" Note step {MIN_DELTA_SIGNATURE_STEP:,} is in the list, "
                f"which is exactly the row that flips when min_delta is the "
                f"ladder's 1e-3 rather than 1e-4. Check the rendered config "
                f"before blaming the seed."
            )
        warnings.append(message)

    # The loss comparison, only once there is a number to judge it by.
    if args.tolerance is None:
        warnings.append(
            "no --tolerance given, so the loss columns above are reported but "
            "not judged. Run the canary a second time at a different --seed "
            "and pass the per-row spread. A tolerance looser than 0.0055 nats "
            "-- the replicate best-loss standard deviation -- is not measuring "
            "anything."
        )
    else:
        for column in NUMERIC_COLUMNS:
            over = merged[merged[f"{column}_delta"].abs() > args.tolerance]
            if not over.empty:
                failures.append(
                    f"{column} is outside {args.tolerance:.6f} at step(s) "
                    f"{list(over['global_step'])}; worst |delta| "
                    f"{worst[column]:.6f}."
                )

    report = {
        "canary_dir": str(args.canary_dir),
        "reference_csv": str(args.reference_csv),
        "tolerance": args.tolerance,
        "rows": merged.to_dict(orient="records"),
        "max_abs_delta": worst,
        "wall_clock": timing,
        "warnings": warnings,
        "passed": not failures,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))

    for warning in warnings:
        print(f"\nNOTE: {warning}")
    if failures:
        raise SystemExit("\nGate 2 failed:\n" + "\n".join(f"  - {f}" for f in failures))
    verdict = (
        "Gate 2 passed"
        if args.tolerance is not None
        else ("Gate 2 structural checks passed; the loss comparison is unjudged")
    )
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
