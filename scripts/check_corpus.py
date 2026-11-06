"""Prove the corpus is the file the paper's 90/10 split was drawn from.

The paper holds out fold 0 of a 10-fold ``cv_seed`` 42 split, and that split is
a **row-positional draw**. With no ``group_col``, ``dasm2``'s Parquet path
(``_cv_split_train_filter``, ``dasm2/data/evolutionary_dataset.py``) does:

    rng = np.random.default_rng(cv_split.seed)
    fold_per_row = rng.integers(cv_split.num_folds, size=n_rows)
    train_filter = fold_per_row != cv_split.fold_index

with ``n_rows`` taken from the Parquet footers. So the *index set* held out is
fixed by ``(seed, num_folds, n_rows)`` alone, and **which PCP is held out is
fixed by which row sits at each index**. A file with the same rows in a
different order is a different 90/10 split.

This also overrides the Parquet's own ``in_train`` column, which holds a
different partition (1,435,844 / 158,930) and is discarded whenever
``--cv-fold`` is passed.

**A content-keyed pin could not replace row order.** ``(clade, parent, child)``
is not unique here -- 911,927 distinct triples across 1,594,774 rows, because
mat-pcp counts are per-branch events and the same transition recurs on many
branches. The positional draw scatters a triple's copies across folds, so the
split is not recoverable from a key set. Hence an order-sensitive digest: a
sorted or set-based one would pass on a reordered file, which is the single
failure this guard exists to catch.

Three checks, all cheap relative to a training run, plus the md5:

1. row count
2. fold-0 split sizes
3. the row-order digest

Then it writes the fold-0 membership beside the Parquet so a later reader can
check the split without re-deriving it.

GISAID: the membership array is per-row derived data and goes to the artifact
root on the cluster. The three integers and the digest are aggregate and are
pinned in ``config.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import (
    CV_FOLD_INDEX,
    CV_N_TRAIN,
    CV_N_VAL,
    CV_NUM_FOLDS,
    CV_SEED,
    corpus,
    file_md5,
)

#: Columns the digest is taken over, in this order. The three that identify a
#: PCP; `era` is derived from `clade` and `in_train` is the column the fold
#: draw replaces.
DIGEST_COLUMNS = ("clade", "parent", "child")

#: Bytes per inner digest. 16 = blake2b-128, the same width as the outer one.
DIGEST_SIZE = 16


def fold_assignment(n_rows: int, num_folds: int, seed: int) -> np.ndarray:
    """The per-row fold index, exactly as ``dasm2`` draws it.

    Reimplemented here rather than imported so the guard states the draw it is
    guarding. ``tests/test_corpus_guard.py`` asserts the two agree; if ``dasm2``
    ever changed its draw, that test fails rather than this guard silently
    blessing a new partition.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(num_folds, size=n_rows)


def row_order_digest(parquet: Path) -> tuple[str, int]:
    """blake2b-128 over per-row key digests, in row order.

    Returns:
        The hex digest and the number of rows it covered.
    """
    outer = hashlib.blake2b(digest_size=DIGEST_SIZE)
    n_rows = 0
    reader = pq.ParquetFile(parquet)
    for batch in reader.iter_batches(columns=list(DIGEST_COLUMNS)):
        columns = [batch.column(name).to_pylist() for name in DIGEST_COLUMNS]
        for row in zip(*columns):
            key = "\t".join(row).encode()
            outer.update(hashlib.blake2b(key, digest_size=DIGEST_SIZE).digest())
            n_rows += 1
    return outer.hexdigest(), n_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="full_depth")
    parser.add_argument(
        "--membership-output",
        type=Path,
        default=None,
        help="where to write the fold-0 membership array; skipped if omitted",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="skip the md5 of a 1.8 GB file when only the split is in question",
    )
    args = parser.parse_args()

    spec = corpus(args.corpus)
    failures: list[str] = []

    print(f"corpus: {spec.path}")

    # 1. Row count, from the footers -- the same number the fold draw reads.
    footer_rows = pq.ParquetFile(spec.path).metadata.num_rows
    print(f"  rows (footer):   {footer_rows:,}  expected {spec.n_rows:,}")
    if footer_rows != spec.n_rows:
        failures.append(
            f"row count is {footer_rows:,}, not {spec.n_rows:,}. The fold draw "
            f"sizes itself from this number, so a different count is a "
            f"different held-out index set before any row is even read."
        )

    # 2. Fold sizes, from the draw itself.
    folds = fold_assignment(footer_rows, CV_NUM_FOLDS, CV_SEED)
    is_val = folds == CV_FOLD_INDEX
    n_val = int(is_val.sum())
    n_train = int(footer_rows - n_val)
    print(
        f"  fold-{CV_FOLD_INDEX} split:    {n_train:,} train / {n_val:,} val"
        f"  expected {CV_N_TRAIN:,} / {CV_N_VAL:,}"
    )
    if (n_train, n_val) != (CV_N_TRAIN, CV_N_VAL):
        failures.append(
            f"fold {CV_FOLD_INDEX} of {CV_NUM_FOLDS} at seed {CV_SEED} gives "
            f"{n_train:,}/{n_val:,}, not {CV_N_TRAIN:,}/{CV_N_VAL:,}. A wrong "
            f"cv_seed, a wrong num_folds, an off-by-one fold or a non-numpy "
            f"RNG all land here."
        )

    # 3. The row-order digest: which PCP sits at each held-out index.
    digest, digest_rows = row_order_digest(spec.path)
    print(f"  row-order digest: {digest}  expected {spec.row_order_digest}")
    if digest_rows != footer_rows:
        failures.append(
            f"the digest pass saw {digest_rows:,} rows where the footers "
            f"promise {footer_rows:,}."
        )
    if digest != spec.row_order_digest:
        failures.append(
            f"row-order digest is {digest}, not {spec.row_order_digest}. The "
            f"file's rows are not in the order the paper's fold 0 was drawn "
            f"from, so its held-out set is a different 10% of the corpus. "
            f"Rebuilding this corpus is a decision to retrain every reference, "
            f"not a prep detail."
        )

    md5 = None
    if not args.skip_md5:
        md5 = file_md5(spec.path)
        print(f"  md5:              {md5}  expected {spec.md5}")
        if md5 != spec.md5:
            failures.append(f"md5 is {md5}, not {spec.md5}.")

    if args.membership_output is not None and not failures:
        args.membership_output.parent.mkdir(parents=True, exist_ok=True)
        # Packed bits: 1.59M booleans in 200 KB rather than 1.6 MB.
        np.savez_compressed(
            args.membership_output,
            is_val_packed=np.packbits(is_val),
            n_rows=np.int64(footer_rows),
            num_folds=np.int64(CV_NUM_FOLDS),
            fold_index=np.int64(CV_FOLD_INDEX),
            seed=np.int64(CV_SEED),
        )
        print(f"  membership written to {args.membership_output}")

    report = {
        "corpus": args.corpus,
        "path": str(spec.path),
        "n_rows": footer_rows,
        "n_train": n_train,
        "n_val": n_val,
        "row_order_digest": digest,
        "md5": md5,
        "passed": not failures,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))

    if failures:
        raise SystemExit(
            "corpus guard failed:\n" + "\n".join(f"  - {f}" for f in failures)
        )
    print("corpus guard passed")


if __name__ == "__main__":
    main()
