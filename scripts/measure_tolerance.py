# ruff: noqa: I001 -- import order in this file is load-bearing; see below.
"""Measure Gate 2's tolerance from two canary runs that differ only in seed.

Neither the canary nor the full run can be bit-exact against the reference:
it trained under flash attention, and trajectory chaos compounds over tens of
thousands of optimizer steps. So the tolerance is measured rather than
argued, by running the same configuration at two seeds and taking the spread.

**Per row, not one number.** Early evaluations are far noisier than late
ones, so a single scalar would be set by step 2,000 and be far too loose by
step 20,000 -- it would wave through a real regression at the end of the
window in order to accommodate noise at the start.

**Two seeds is a thin estimator, and this script does not pretend
otherwise.** |a - b| for two draws is one sample of a random variable, not
its standard deviation; on average it understates the spread of the
underlying distribution. So the emitted tolerance is the observed spread
times ``--safety``, and the factor is written into the output file next to
the numbers rather than buried here. Raising the seed count is the honest
way to tighten it; two is what the GPU budget bought.

For orientation, two anchors from the reference experiment's README: the
replicate best-loss standard deviation is 0.0055 nats, and the smallest
effect this corpus has resolved is 0.0033 nats. Those describe *converged*
losses, so an early-step tolerance above them is expected and is not
evidence of a problem.

GISAID: per-evaluation aggregate scalars only.
"""

import dasm2  # noqa: F401  (imported first for its side effects)

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import assert_dasm2_is_this_repo, CANARY_STEPS, EVAL_EVERY_STEPS

from dasm2.training.log_reader import get_run_eval_losses

assert_dasm2_is_this_repo()

#: Columns Gate 2 compares, and so the columns a tolerance is needed for.
COLUMNS = ("avg_train_loss", "val_loss", "val_aa_acc_mutated")

#: Multiple of the observed two-seed spread that becomes the tolerance.
#: Above 1 because |a - b| from two draws understates the spread it estimates.
DEFAULT_SAFETY = 2.0

#: Floor, so a row where the two seeds happen to agree almost exactly does not
#: produce a tolerance no rerun could ever meet.
MIN_TOLERANCE = 1.0e-3


def curve(canary_dir: Path, through_step: int) -> pd.DataFrame:
    frame = get_run_eval_losses(str(canary_dir))
    frame = frame[frame["global_step"] <= through_step]
    return frame.sort_values("global_step").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canary-dir", action="append", required=True, help="give twice, once per seed"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--safety", type=float, default=DEFAULT_SAFETY)
    parser.add_argument("--through-step", type=int, default=CANARY_STEPS)
    args = parser.parse_args()

    if len(args.canary_dir) < 2:
        raise SystemExit("a spread needs at least two runs; pass --canary-dir twice.")

    curves = [curve(Path(d), args.through_step) for d in args.canary_dir]
    steps = list(curves[0]["global_step"])
    expected = args.through_step // EVAL_EVERY_STEPS
    for directory, frame in zip(args.canary_dir, curves):
        if list(frame["global_step"]) != steps:
            raise SystemExit(
                f"{directory} evaluated at {list(frame['global_step'])}, which "
                f"does not line up with {steps}."
            )
        if len(frame) != expected:
            raise SystemExit(
                f"{directory} has {len(frame)} evaluations, not {expected}."
            )

    rows = []
    for position, step in enumerate(steps):
        row = {"global_step": step}
        for column in COLUMNS:
            values = [float(frame[column].iloc[position]) for frame in curves]
            spread = max(values) - min(values)
            row[f"{column}_spread"] = spread
            row[f"{column}_tolerance"] = max(spread * args.safety, MIN_TOLERANCE)
        rows.append(row)
    table = pd.DataFrame(rows)

    header = (
        f"# Gate 2 tolerance, measured from {len(curves)} canary runs "
        f"differing only in --seed.\n"
        f"# runs: {', '.join(str(Path(d).name) for d in args.canary_dir)}\n"
        f"# tolerance = observed spread x {args.safety}, floored at "
        f"{MIN_TOLERANCE}.\n"
        f"# Two seeds understate the spread they estimate; raising the seed "
        f"count is how to tighten this.\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + table.to_csv(index=False, float_format="%.6f"))

    print(header, end="")
    print(table.to_string(index=False))
    print()
    for column in COLUMNS:
        series = table[f"{column}_spread"]
        print(
            f"  {column}: spread {series.min():.6f}..{series.max():.6f}, "
            f"first row {series.iloc[0]:.6f}, last row {series.iloc[-1]:.6f}"
        )
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
