"""Score one run's best checkpoint on its own training split, dropout off.

The training curve in a run's log is not comparable to its validation curve.
``avg_train_loss`` is a running mean over training batches taken **with
dropout active** and with the weights moving across the averaging window,
while ``val_loss`` is a clean eval-mode pass at a fixed point. Subtracting
them gives a number that mixes a generalization gap with a dropout penalty
and a stale-weights artifact, and the dropout term is not small on a model
with ``dropout: 0.1``.

This recomputes the training side the way the validation side is already
computed: ``module.eval()``, a full deterministic pass, the trainer's own
denominator. The gap it reports is then a generalization gap and nothing
else, which is what makes "the larger model fits its own training data worse"
a statement about optimization rather than about capacity.

**Each run is scored on its own training rows**, the split its own config and
``config.yaml``'s ``cv_split`` define. The thinned arms train on different
corpora, so each one's training set is its own.

**The validation side is read from the log, not recomputed.** The trainer
already evaluated in eval mode at every ``eval_every_steps``, and
``early_stopping_min_delta`` is 0 for every run here, so the saved best
checkpoint is the validation curve's argmin and the paired number is that
minimum. Recomputing it would cost a second pass to reproduce a recorded
value.

**One checkpoint, not every checkpoint.** ``lib.select_checkpoint`` picks the
cycle's best; the periodic ``_stepNNNNNN_`` checkpoints are not scored, since
the readout is about where each run ended up.

GISAID: reads the corpus, emits per-run scalars only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import (
    batch_size_for,
    build_sequential_loader,
    evaluate,
    head_key_of,
)
from lib import (
    CV_FOLD_INDEX,
    CV_NUM_FOLDS,
    CV_SEED,
    select_checkpoint,
)

from dasm2.config import CvSplitConfig, load_config_from_yaml
from dasm2.serialization.checkpoints import load_model_only
from dasm2.training.log_reader import get_run_eval_losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help=(
            "DataLoader workers for the training pass. Matches the templates' "
            "`num_dataloader_workers`; 0 makes the pass dataloader-bound."
        ),
    )
    args = parser.parse_args()

    config = load_config_from_yaml(args.config)
    batch_size = batch_size_for(config, args.batch_size)
    checkpoint = select_checkpoint(args.run_dir)

    frame = get_run_eval_losses(args.run_dir)
    if frame.empty:
        raise SystemExit(
            f"{args.run_dir} logged no evaluations, so there is no validation "
            f"number to pair the training pass with."
        )
    best = frame.loc[frame["val_loss"].idxmin()]
    best_val_loss = float(best["val_loss"])
    best_step = int(best["global_step"])

    cv_split = CvSplitConfig(
        num_folds=CV_NUM_FOLDS, fold_index=CV_FOLD_INDEX, seed=CV_SEED
    )

    print(
        f"{args.model}/{args.corpus}: scoring {checkpoint.name} on the full "
        f"training split of fold {CV_FOLD_INDEX} of {CV_NUM_FOLDS}, batch "
        f"{batch_size}, {args.num_workers} worker(s)",
        flush=True,
    )

    loader = build_sequential_loader(
        config, args.parquet, cv_split, batch_size, "train", args.num_workers
    )

    module = load_model_only(str(checkpoint), device=args.device)
    try:
        train_loss, positions = evaluate(
            module, head_key_of(module), loader, args.device
        )
    finally:
        del module
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    row = {
        "model": args.model,
        "corpus": args.corpus,
        "train_loss_eval_mode": train_loss,
        "train_positions": positions,
        # Repeated from the log so this file stands on its own, and so a
        # mismatch against `read_run`'s row is visible rather than silent.
        "best_val_loss_logged": best_val_loss,
        "best_step_logged": best_step,
        # Positive means the model fits its training rows better than the
        # held-out ones, which is the ordinary direction.
        "gap": best_val_loss - train_loss,
        "scored_checkpoint": checkpoint.name,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    print(
        f"  train {train_loss:.6f} over {positions:,} positions   "
        f"val {best_val_loss:.6f} @ {best_step:,}   gap {row['gap']:+.6f}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
