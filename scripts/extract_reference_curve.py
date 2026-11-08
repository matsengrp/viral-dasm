"""Write the reference run's evaluation curve to a tracked CSV.

Gate 2 compares the canary's ten evaluations against this file, so the
comparison is a rule rather than a hand check against a table in an issue.
The file is derived here rather than transcribed, so it cannot drift from the
run it describes.

Five columns, and the fifth is in it on purpose:

- ``avg_train_loss``, ``val_loss`` and ``val_aa_acc_mutated`` are what the
  curve is.
- ``is_best`` is the only column that sees ``early_stopping_min_delta``. At
  the replicate experiment's 1e-4 the pattern is T T T T F T T T T T; at the
  sampling-depth ladder's 1e-3 the step-16,000 row flips to F. One extra
  column, one extra flip, free -- and ``min_delta`` gates best-model tracking
  as well as stopping, so getting it wrong saves the wrong weights.
- ``lr`` is recorded but **cannot** be checked by this gate: the reference's
  learning rate is a constant 1e-3 through step 130,000, so a constant-LR run,
  or a plateau scheduler with a different ``factor`` or ``eval_patience``,
  produces an identical first ten evaluations. That is
  ``scripts/check_rendered_config.py``'s job.

GISAID: per-evaluation aggregate scalars over the held-out fold. No sequence,
no per-sample or per-site data. Publishable, hence tracked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import CANARY_STEPS, EVAL_EVERY_STEPS, assert_dasm2_is_this_repo, reference_run

from dasm2.training.log_reader import get_run_eval_losses

assert_dasm2_is_this_repo()

#: What Gate 2 compares. Narrow on purpose: the run logs 26 columns, most of
#: them diagnostics whose run-to-run spread nobody has measured, and a gate
#: that compared all of them would fail on noise.
COLUMNS = (
    "global_step",
    "avg_train_loss",
    "val_loss",
    "val_aa_acc_mutated",
    "is_best",
    "lr",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dasm-85k")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--through-step",
        type=int,
        default=CANARY_STEPS,
        help="last step to include; the canary's budget by default",
    )
    args = parser.parse_args()

    reference = reference_run(args.model)
    frame = get_run_eval_losses(str(reference.checkpoints_dir))

    expected_rows = args.through_step // EVAL_EVERY_STEPS
    curve = frame[frame["global_step"] <= args.through_step][list(COLUMNS)]
    curve = curve.sort_values("global_step").reset_index(drop=True)
    if len(curve) != expected_rows:
        raise SystemExit(
            f"the reference log has {len(curve)} evaluations at or before step "
            f"{args.through_step:,}, not {expected_rows}. At one evaluation "
            f"every {EVAL_EVERY_STEPS:,} steps those should agree; a gap means "
            f"a resumed run's logs were not concatenated as expected."
        )

    # Sanity against the endpoint config.yaml pins, so a reference run that was
    # retrained under different settings cannot quietly become the baseline.
    full = frame.sort_values("global_step")
    best_row = full.loc[full["val_loss"].idxmin()]
    if abs(float(best_row["val_loss"]) - reference.best_val_loss) > 1e-6:
        raise SystemExit(
            f"this run's best val_loss is {float(best_row['val_loss']):.6f}, "
            f"but config.yaml records {reference.best_val_loss:.6f}."
        )
    if int(full["global_step"].max()) != reference.stop_step:
        raise SystemExit(
            f"this run's last evaluation is at step "
            f"{int(full['global_step'].max()):,}, but config.yaml records a "
            f"stop at {reference.stop_step:,}."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(args.output, index=False, float_format="%.6f")
    print(f"{len(curve)} rows -> {args.output}")
    print(curve.to_string(index=False))


if __name__ == "__main__":
    main()
