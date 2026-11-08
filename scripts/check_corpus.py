"""Prove the corpus gives the paper's 90/10 split.

The paper holds out fold 0 of a 10-fold ``cv_seed`` 42 split, and that split is
a **row-positional draw**. With no ``group_col``, ``dasm2``'s Parquet path
(``_cv_split_train_filter``, ``dasm2/data/evolutionary_dataset.py``) does:

    rng = np.random.default_rng(cv_split.seed)
    fold_per_row = rng.integers(cv_split.num_folds, size=n_rows)
    train_filter = fold_per_row != cv_split.fold_index

with ``n_rows`` taken from the Parquet footers. So the *index set* held out is
fixed by ``(seed, num_folds, n_rows)`` alone, and **which PCP is held out is
fixed by which row sits at each index**.

**The corpus can be rebuilt and still give the same fold 0.** The builder
copies the mat-pcp PCP file's row order verbatim -- nothing in it sorts,
samples or shuffles. So what has to be pinned is that input file, not the
Parquet built from it, and this guard checks the input's md5.

The Parquet's own md5 is reported but **not** a gate. A rebuild recomputes
``nt_rates`` and ``nt_csps`` and picks its own row-group boundaries and
compression, so the file's bytes may differ while every row stays where it
was. Failing on that would block a legitimate rebuild and would say nothing
about the split.

**A content-keyed pin could not replace row order either.** ``(clade, parent,
child)`` is not unique here -- 911,927 distinct triples across 1,594,774 rows,
because mat-pcp counts are per-branch events and the same transition recurs on
many branches. The positional draw scatters a triple's copies across folds, so
the split is not recoverable from a key set. Hence an order-sensitive digest: a
sorted or set-based one would pass on a reordered file, which is the single
failure this guard exists to catch.

The checks, in order of what they would cost you to get wrong:

1. The source PCP file's md5. Row order is inherited from it.
2. Row count.
3. Fold-0 split sizes.
4. The row-order digest.

Then it writes the fold-0 membership beside the Parquet so a later reader can
check the split without re-deriving it.

GISAID: the membership array is per-row derived data and goes to the artifact
root on the cluster. The counts and the digest are aggregate and are pinned in
``config.yaml``.
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
        "--parquet",
        type=Path,
        default=None,
        help=(
            "check this file instead of the corpus's configured path, against "
            "that corpus's recorded numbers. For verifying a rebuild at a "
            "scratch path: without it the only way to check one is to edit "
            "config.yaml between runs, which is the mutation this guard exists "
            "to catch. The Parquet's own md5 is not compared under an override."
        ),
    )
    parser.add_argument(
        "--expect-digest",
        default=None,
        help=(
            "require this row-order digest instead of the corpus's recorded "
            "one. For checking a file before anything is pinned to it, and for "
            "the negative control that confirms the digest is order-sensitive."
        ),
    )
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
        help="skip both md5s when only the split itself is in question",
    )
    args = parser.parse_args()

    spec = corpus(args.corpus)
    parquet_path = args.parquet if args.parquet is not None else spec.path
    # `or` rather than a None test: a digest is 32 hex characters, so the
    # empty string is not a value either side can legitimately hold.
    expected_digest = args.expect_digest or spec.row_order_digest
    failures: list[str] = []

    print(f"corpus: {parquet_path}")
    if args.parquet is not None:
        print(f"  (override; {spec.name}'s recorded numbers still apply)")
    if args.expect_digest is not None:
        print("  (digest expectation overridden on the command line)")

    # 1. Row count, from the footers -- the same number the fold draw reads.
    footer_rows = pq.ParquetFile(parquet_path).metadata.num_rows
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
        f"  expected {spec.n_train:,} / {spec.n_val:,}"
    )
    if (n_train, n_val) != (spec.n_train, spec.n_val):
        failures.append(
            f"fold {CV_FOLD_INDEX} of {CV_NUM_FOLDS} at seed {CV_SEED} gives "
            f"{n_train:,}/{n_val:,}, not {spec.n_train:,}/{spec.n_val:,}. A "
            f"wrong cv_seed, a wrong num_folds, an off-by-one fold or a "
            f"non-numpy RNG all land here."
        )

    # 3. The row-order digest: which PCP sits at each held-out index.
    digest, digest_rows = row_order_digest(parquet_path)
    print(f"  row-order digest: {digest}  expected {expected_digest}")
    if digest_rows != footer_rows:
        failures.append(
            f"the digest pass saw {digest_rows:,} rows where the footers "
            f"promise {footer_rows:,}."
        )
    if digest != expected_digest:
        failures.append(
            f"row-order digest is {digest}, not {expected_digest}. The "
            f"file's rows are not in the order the paper's fold 0 was drawn "
            f"from, so its held-out set is a different 10% of the corpus. "
            f"Rebuilding this corpus is a decision to retrain every reference, "
            f"not a prep detail."
        )

    source_md5 = None
    md5 = None
    if not args.skip_md5:
        # The pin that matters. Row order is inherited from this file, so a
        # different one here is a different fold 0 no matter what the builder
        # does.
        source_md5 = file_md5(spec.source_pcp)
        print(f"  source PCP md5:   {source_md5}  expected {spec.source_pcp_md5}")
        if source_md5 != spec.source_pcp_md5:
            failures.append(
                f"the source PCP file {spec.source_pcp} has md5 {source_md5}, "
                f"not {spec.source_pcp_md5}. The corpus inherits its row order "
                f"from this file, so a different one gives a different 90/10 "
                f"split. mat-pcp rebuilds its results tree in place, which is "
                f"exactly how this changes without anyone deciding to change it."
            )

        # Reported, not gated: a rebuild recomputes nt_rates/nt_csps and picks
        # its own row-group boundaries, so these bytes can move while every row
        # stays where it was. The digest above is what the split depends on.
        #
        # A corpus this pipeline builds records no md5 at all, and under
        # `--parquet` the recorded one belongs to a different file, so in both
        # cases the number is printed without a comparison rather than compared
        # against something it does not describe.
        md5 = file_md5(parquet_path)
        if spec.md5 is None or args.parquet is not None:
            print(f"  parquet md5:      {md5}  (nothing recorded to compare against)")
        else:
            note = "matches" if md5 == spec.md5 else "DIFFERS from"
            print(
                f"  parquet md5:      {md5}  ({note} the recorded {spec.md5}; "
                f"not a gate)"
            )

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

    md5_comparable = md5 is not None and spec.md5 is not None and args.parquet is None
    report = {
        "corpus": args.corpus,
        "path": str(parquet_path),
        "path_is_override": args.parquet is not None,
        "n_rows": footer_rows,
        "n_train": n_train,
        "n_val": n_val,
        "row_order_digest": digest,
        "expected_row_order_digest": expected_digest,
        "digest_expectation_is_override": args.expect_digest is not None,
        "md5": md5,
        "md5_matches_recorded": (md5 == spec.md5) if md5_comparable else None,
        "source_pcp": str(spec.source_pcp),
        "source_pcp_md5": source_md5,
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
