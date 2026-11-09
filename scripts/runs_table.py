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
    "checkpoint",
    "checkpoint_sha256",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows-dir",
        required=True,
        type=Path,
        help="directory holding one <model>_<corpus>.json per run",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for model, corpus_name in RUNS:
        path = args.rows_dir / f"{run_id(model, corpus_name)}.json"
        row = json.loads(path.read_text())
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
