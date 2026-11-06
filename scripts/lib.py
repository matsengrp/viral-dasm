"""config.yaml, read once, with the invariants checked at import.

Every path and every pinned number the pipeline uses comes through here.
Import-time rather than rule-time validation: the alternative is discovering a
missing corpus or a contradictory split partway into a multi-day DAG.

Nothing in this module resolves anything. If a path is not in config.yaml it
does not exist, which is what makes config.yaml a complete description of the
pipeline's inputs for a reader who has their own mat-pcp run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

#: Repository root -- the directory holding config.yaml.
REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = REPO_ROOT / "config.yaml"

with CONFIG_PATH.open() as handle:
    CONFIG = yaml.safe_load(handle)

ARTIFACT_ROOT = Path(CONFIG["artifact_root"])
GENE = CONFIG["mat_pcp"]["gene"]
TRAIN_SEED = int(CONFIG["train_seed"])

CV_NUM_FOLDS = int(CONFIG["cv_split"]["num_folds"])
CV_FOLD_INDEX = int(CONFIG["cv_split"]["fold_index"])
CV_SEED = int(CONFIG["cv_split"]["seed"])
CV_N_TRAIN = int(CONFIG["cv_split"]["n_train"])
CV_N_VAL = int(CONFIG["cv_split"]["n_val"])

#: The one architecture M1 trains, and the reference run it is checked against.
CANARY_MODEL = "dasm-85k"

#: Steps in the cheap gate. Ten evals at the reference's 2,000-step cadence.
CANARY_STEPS = 20000
EVAL_EVERY_STEPS = 2000

#: Evals between intermediate checkpoint saves, matching the reference run.
CHECKPOINT_EVERY_N_EVALS = 10

#: Cycles per run. These heads need no branch-length re-optimization, so
#: ``global_step`` and ``cycle_step`` agree and a step budget is a step budget.
NUM_CYCLES = 1

#: DataLoader workers. The per-sample whichmut transform chain runs in the
#: worker, so at 0 a scoring pass is dataloader-bound and roughly 4x slower for
#: an identical answer.
DATALOADER_WORKERS = 6

#: Agreement required between a loss recomputed here and the value the
#: reference run logged for itself. Loose enough to absorb a different card's
#: float reduction order, tight enough that a wrong denominator, a missing
#: censoring source or a shuffled split could not slip through: all three move
#: the loss by 1e-3 or more against a quantity of ~7.77.
SELF_CHECK_TOLERANCE = 1.0e-4


def mat_pcp_path(key: str) -> Path:
    """One of the five mat-pcp files the training config names.

    Raises:
        KeyError: If ``key`` is not one of them -- so a typo fails here rather
            than rendering a config with a path that silently censors nothing.
    """
    block = CONFIG["mat_pcp"]
    allowed = {
        "censored_mutations_file",
        "censored_clades_file",
        "masked_nt_sites_file",
        "reference_sequence_file",
    }
    if key not in allowed:
        raise KeyError(
            f"{key!r} is not a mat-pcp file; expected one of {sorted(allowed)}"
        )
    return Path(block["root"]) / block[key]


@dataclass(frozen=True)
class Corpus:
    """A prepared Parquet, with the integers that identify it."""

    name: str
    path: Path
    md5: str
    n_rows: int
    row_order_digest: str
    parquet_cache_budget_gb: float


def corpus(name: str) -> Corpus:
    """The named corpus from ``config.yaml``."""
    entry = CONFIG["corpora"][name]
    return Corpus(
        name=name,
        path=Path(entry["path"]),
        md5=entry["md5"],
        n_rows=int(entry["n_rows"]),
        row_order_digest=entry["row_order_digest"],
        parquet_cache_budget_gb=float(entry["parquet_cache_budget_gb"]),
    )


FULL_DEPTH = corpus("full_depth")

FAST_TEST_PARQUET = Path(CONFIG["fast_test_corpus"]["path"])
FAST_TEST_MD5 = CONFIG["fast_test_corpus"]["md5"]


@dataclass(frozen=True)
class ReferenceRun:
    """A ``dasm2-experiments`` run the port is checked against."""

    model: str
    checkpoints_dir: Path
    best_checkpoint: Path
    best_checkpoint_md5: str
    best_val_loss: float
    best_step: int
    stop_step: int
    stop_reason: str
    n_loss_positions: int
    effective_params: int


def reference_run(model: str) -> ReferenceRun:
    """The reference run for ``model``."""
    entry = CONFIG["reference_runs"][model]
    checkpoints_dir = Path(entry["checkpoints_dir"])
    return ReferenceRun(
        model=model,
        checkpoints_dir=checkpoints_dir,
        best_checkpoint=checkpoints_dir / entry["best_checkpoint"],
        best_checkpoint_md5=entry["best_checkpoint_md5"],
        best_val_loss=float(entry["best_val_loss"]),
        best_step=int(entry["best_step"]),
        stop_step=int(entry["stop_step"]),
        stop_reason=entry["stop_reason"],
        n_loss_positions=int(entry["n_loss_positions"]),
        effective_params=int(entry["effective_params"]),
    )


def file_md5(path: Path, chunk_bytes: int = 1 << 24) -> str:
    """md5 of a file, read in chunks so a 1.8 GB Parquet does not go in memory."""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _check_config() -> None:
    """Fail at import on a config that cannot describe a run.

    Only invariants internal to the file are checked. Whether a path's contents
    are what the file says they are is the corpus guard's job
    (``scripts/check_corpus.py``), because that costs a read of 1.8 GB.
    """
    if not 0 <= CV_FOLD_INDEX < CV_NUM_FOLDS:
        raise ValueError(
            f"cv_split holds out fold {CV_FOLD_INDEX} of {CV_NUM_FOLDS}; "
            f"CvSplitConfig rejects a fold_index >= num_folds, hours into training."
        )
    if CV_N_TRAIN + CV_N_VAL != FULL_DEPTH.n_rows:
        raise ValueError(
            f"cv_split's halves sum to {CV_N_TRAIN + CV_N_VAL}, but full_depth "
            f"has {FULL_DEPTH.n_rows} rows. The split is a partition of every "
            f"row, so these must agree: one of the three numbers is stale."
        )
    # The held-out share has to be 1/num_folds to within the draw's own
    # randomness. A fold that is 20% of the corpus means num_folds moved and
    # the recorded sizes did not.
    share = CV_N_VAL / FULL_DEPTH.n_rows
    expected = 1.0 / CV_NUM_FOLDS
    if abs(share - expected) > 0.01:
        raise ValueError(
            f"fold {CV_FOLD_INDEX} is {share:.4f} of the corpus, but a "
            f"{CV_NUM_FOLDS}-fold split draws about {expected:.4f}. The "
            f"recorded split sizes do not come from this num_folds."
        )
    if CANARY_STEPS % EVAL_EVERY_STEPS:
        raise ValueError(
            f"the canary runs {CANARY_STEPS} steps at one eval every "
            f"{EVAL_EVERY_STEPS}, which does not divide evenly, so its last "
            f"row would not line up with a reference row."
        )
    if len(FULL_DEPTH.row_order_digest) != 32:
        raise ValueError(
            f"row_order_digest is {len(FULL_DEPTH.row_order_digest)} hex "
            f"characters; blake2b-128 gives 32."
        )


_check_config()
