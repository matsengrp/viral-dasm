"""Gather the seven per-run rows into the one table this repository commits.

``data/runs.csv`` is aggregate summary statistics -- one row per run, every
number read from that run's own log -- so it is publishable and tracked. The
checkpoints it names stay under the artifact root until M7 decides what is
released; the table records each one's path and sha256 so that decision has
something to act on.

Row order is :data:`runs.RUNS`, not whatever order the DAG finished in, so a
diff of this file shows what changed rather than which run was slowest.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runs import RUNS, run_id

#: Column order, written out rather than taken from the first row's dict, so
#: adding a field to `read_run.py` is a visible change here too.
COLUMNS = (
    "model",
    "corpus",
    "effective_params",
    "learning_rate",
    "n_train",
    "n_val",
    "evaluations",
    "steps_to_stop",
    "stop_reason",
    "final_lr",
    "data_passes",
    "wall_clock_s",
    "completed_attempts",
    "best_val_loss_logged",
    "best_step_logged",
    "patience_stop_step_logged",
    "best_val_loss_saved",
    "best_val_loss_at_min_delta_1e4",
    "best_step_at_min_delta_1e4",
    "patience_stop_step_at_min_delta_1e4",
    "best_val_loss_at_min_delta_1e3",
    "best_step_at_min_delta_1e3",
    "patience_stop_step_at_min_delta_1e3",
    "train_loss_eval_mode",
    "train_positions",
    "gap",
    "checkpoint",
    "checkpoint_sha256",
)

#: Keys `train_eval.py` repeats from the log so its file stands alone. They
#: are checked against `read_run`'s row rather than written twice.
TRAIN_EVAL_ECHOED = ("best_val_loss_logged", "best_step_logged")

#: What `train_eval.py` contributes that `read_run.py` cannot.
TRAIN_EVAL_COLUMNS = ("train_loss_eval_mode", "train_positions", "gap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows-dir",
        required=True,
        type=Path,
        help=(
            "directory holding one <model>_<corpus>.json and one "
            "<model>_<corpus>_train_eval.json per run"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for model, corpus_name in RUNS:
        run = run_id(model, corpus_name)
        path = args.rows_dir / f"{run}.json"
        row = json.loads(path.read_text())

        scored_path = args.rows_dir / f"{run}_train_eval.json"
        scored = json.loads(scored_path.read_text())
        for key in TRAIN_EVAL_ECHOED:
            if scored[key] != row[key]:
                raise SystemExit(
                    f"{scored_path} was scored against {key}={scored[key]!r} "
                    f"but {path} records {row[key]!r}. The two read the same "
                    f"log, so they disagree only if one is stale -- rerun "
                    f"whichever is older rather than joining them."
                )
        # The losses can match while the files describe different runs: a
        # retrained arm that reached the same best would pass the check above.
        # The checkpoint name carries the run id, so it does not.
        if scored["scored_checkpoint"] != Path(row["checkpoint"]).name:
            raise SystemExit(
                f"{scored_path} scored {scored['scored_checkpoint']!r} but "
                f"{path} names {Path(row['checkpoint']).name!r}. One of them "
                f"predates a retrain; rerun it rather than joining them."
            )
        # `min_delta` is 0 for every run, which is what makes the saved best
        # and the curve's argmin the same number. If that ever stops holding,
        # `gap` is computed against a loss no checkpoint has.
        saved = row["best_val_loss_saved"]
        if saved is not None and saved != row["best_val_loss_logged"]:
            raise SystemExit(
                f"{path}: the saved best {saved!r} is not the logged argmin "
                f"{row['best_val_loss_logged']!r}. At "
                f"`early_stopping_min_delta: 0` they are the same number, so "
                f"this run did not train under the stopping rule the table "
                f"assumes and its `gap` compares two different checkpoints."
            )
        row.update({key: scored[key] for key in TRAIN_EVAL_COLUMNS})

        unexpected = sorted(set(row) - set(COLUMNS))
        missing = sorted(set(COLUMNS) - set(row))
        if unexpected or missing:
            raise SystemExit(
                f"{path} has columns this table does not write "
                f"({unexpected}) and lacks ones it does ({missing}). The "
                f"column list here is deliberate, so reconcile the two "
                f"rather than letting a field reach the committed table "
                f"unannounced."
            )
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
