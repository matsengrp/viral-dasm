"""Train one model, over the flags this pipeline actually passes.

**Deliberately a rewrite, not a port.** `dasm2`'s own entry point is
`dasm2/cli.py`, a Fire-dispatched surface over a dozen commands -- preprocess,
oe-plot, functional-eval, dashboards, checkpoint selection -- that this
repository does not have and should not pretend to. Carrying it in would mean
either vendoring those commands too or shipping a CLI whose help text lists
subcommands that raise on import.

So this wrapper calls `dasm2.train_command.train_model` directly, over the
eleven arguments the pipeline passes, and nothing else. Which also means the
pipeline cannot reach a knob the paper never turned. `dasm2.training.
train_argv.build_train_argv` is carried but unused for the same reason: it
builds argv for a CLI that is not here.

Resuming is idempotent. A run that dies partway -- preemption, a full disk, a
walltime -- picks up from its own latest checkpoint instead of starting over,
which on a multi-day run is the difference between a bad afternoon and a bad
week. That needs `save_training_state: true` in the config; every template
under `configs/` sets it.

**This script claims its own card.** `--device cuda` means "pick one", not
"card 0": `pick_device` treats any string preference as explicit and would
return `torch.device("cuda")` unexamined, so seven parallel runs would land
on the same GPU. `scripts/gpu_slot.py` picks a free card this user is alone
on and holds a lock file for as long as this process runs, and the resolved
`cuda:N` is what reaches the trainer. `CUDA_VISIBLE_DEVICES` cannot do this
job here: `dasm2.train_command` is imported at module scope, so torch has
already enumerated the devices before any line of `main` runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_slot import claim_card
from lib import (
    ARTIFACT_ROOT,
    CHECKPOINT_EVERY_N_EVALS,
    CV_FOLD_INDEX,
    CV_NUM_FOLDS,
    CV_SEED,
    NUM_CYCLES,
    TRAIN_SEED,
    assert_dasm2_is_this_repo,
)

from dasm2.train_command import train_model

assert_dasm2_is_this_repo()

#: The head each architecture trains under. The three transformers consume
#: the shared selection factor; `ci-484k`'s is its own per-site table. The
#: key names the dataset `train_model` loads, so it has to match the config's
#: `tasks` entry or the run trains nothing.
HEADS = ("SelectionFactorHead", "PerSiteParentChildHead")

#: Where a claimed card's lock file lives. Under the artifact root rather
#: than the repository, so two checkouts of this pipeline -- the arrangement
#: a multi-day run forces -- contend for the same cards rather than each
#: believing every card is free.
GPU_LOCK_DIR = ARTIFACT_ROOT / ".gpu_locks"


def latest_resumable(output_dir: Path) -> Path | None:
    """The newest checkpoint in ``output_dir``, or None if there is none.

    By modification time, not by name: checkpoint filenames carry a
    config-metadata hash, so sorting them lexically returns an arbitrary one
    when a directory holds more than one attempt. That is how a 16M-step run
    once lost its best-validation weights permanently.

    Each run attempt still gets its own ``output_dir``; this protects against
    one attempt dying, not against two running at once.
    """
    checkpoints = sorted(
        output_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return checkpoints[0] if checkpoints else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--device",
        default="cuda",
        help=(
            "`cuda` claims a free card this user is alone on and trains on "
            "it; `cuda:N` takes that card without a claim; `cpu` is the "
            "wiring smoke"
        ),
    )
    parser.add_argument(
        "--head",
        default=HEADS[0],
        choices=HEADS,
        help="the head whose dataset this run loads; must match the config",
    )
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--cycles", type=int, default=NUM_CYCLES)
    parser.add_argument(
        "--checkpoint-every-n-evals", type=int, default=CHECKPOINT_EVERY_N_EVALS
    )
    parser.add_argument("--cv-fold", type=int, default=CV_FOLD_INDEX)
    parser.add_argument("--cv-num-folds", type=int, default=CV_NUM_FOLDS)
    parser.add_argument("--cv-seed", type=int, default=CV_SEED)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="start fresh even if the output directory holds a checkpoint",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resume = None if args.no_resume else latest_resumable(args.output_dir)
    if resume is not None:
        print(f"resuming from {resume.name}", flush=True)

    def run(device: str) -> None:
        print(f"training on {device}", flush=True)
        train_model(
            config_name=str(args.config),
            data={args.head: str(args.parquet)},
            output_dir=str(args.output_dir),
            device=device,
            seed=args.seed,
            cycles=args.cycles,
            checkpoint_every_n_evals=args.checkpoint_every_n_evals,
            cv_fold=args.cv_fold,
            cv_num_folds=args.cv_num_folds,
            cv_seed=args.cv_seed,
            resume=str(resume) if resume is not None else None,
        )

    # The lock is held for exactly as long as training runs: the `with` block
    # releases it on a raise and on a normal return alike, which a wrapper
    # script around this process could not promise.
    if args.device == "cuda":
        with claim_card(GPU_LOCK_DIR, args.output_dir.name) as card:
            run(f"cuda:{card}")
    else:
        run(args.device)


if __name__ == "__main__":
    main()
