"""The guard's fold draw is `dasm2`'s fold draw.

`scripts/check_corpus.py` reimplements the positional draw rather than
importing it, so the guard states the thing it is guarding. That is only safe
while the two agree, which is what these assert. If `dasm2` ever changed its
draw, these fail rather than the guard silently blessing a new partition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_corpus import DIGEST_COLUMNS, fold_assignment, row_order_digest
from lib import CORPUS_NAMES, CV_FOLD_INDEX, CV_NUM_FOLDS, CV_SEED, FULL_DEPTH, corpus


def dasm2_fold_per_row(n_rows: int, num_folds: int, seed: int) -> np.ndarray:
    """The draw as `dasm2/data/evolutionary_dataset.py` writes it."""
    rng = np.random.default_rng(seed)
    return rng.integers(num_folds, size=n_rows)


@pytest.mark.parametrize(
    ("n_rows", "num_folds", "seed"),
    [(1000, 10, 42), (1_594_774, 10, 42), (7, 3, 0), (1_594_774, 5, 7)],
)
def test_guard_draw_matches_the_library(n_rows, num_folds, seed):
    np.testing.assert_array_equal(
        fold_assignment(n_rows, num_folds, seed),
        dasm2_fold_per_row(n_rows, num_folds, seed),
    )


@pytest.mark.parametrize("name", CORPUS_NAMES)
def test_every_corpus_split_is_reproducible_from_three_numbers(name):
    """Each arm's recorded fold sizes come out of (seed, num_folds, n_rows).

    1,435,362 / 159,412 out of (42, 10, 1,594,774) for the paper's corpus, and
    the same three numbers for every thinned arm. The held-out *index set*
    needs only these three; which PCP sits at each index is the corpus file's
    row order, which the digest covers.

    Parametrized over every corpus because the sizes are per arm: a copied
    number would otherwise sit in ``config.yaml`` unchallenged until a build
    ran, which is hours later and on a different machine.
    """
    spec = corpus(name)
    folds = fold_assignment(spec.n_rows, CV_NUM_FOLDS, CV_SEED)
    n_val = int((folds == CV_FOLD_INDEX).sum())
    assert (spec.n_rows - n_val, n_val) == (spec.n_train, spec.n_val)


def test_the_arms_hold_out_different_pcps_at_the_same_indices():
    """Two arms' fold 0 are different held-out *sets*, not a prefix of each other.

    The draw is positional, so a shallower arm's fold 0 is not the deeper
    arm's restricted to the rows that survived thinning -- the rows moved. A
    test that only checked sizes would miss this, and it is the reason each arm
    carries its own digest rather than inheriting ``full_depth``'s.
    """
    digests = {name: corpus(name).row_order_digest for name in CORPUS_NAMES}
    assert len(set(digests.values())) == len(digests), (
        f"two corpora share a row-order digest, which cannot happen for files "
        f"with different row counts: {digests}"
    )


def test_the_draw_is_sensitive_to_every_one_of_the_three():
    """A change to any of seed, num_folds or n_rows gives a different fold 0."""
    base = set(
        np.flatnonzero(
            fold_assignment(FULL_DEPTH.n_rows, CV_NUM_FOLDS, CV_SEED) == CV_FOLD_INDEX
        )
    )
    for label, folds in (
        ("seed", fold_assignment(FULL_DEPTH.n_rows, CV_NUM_FOLDS, CV_SEED + 1)),
        ("num_folds", fold_assignment(FULL_DEPTH.n_rows, CV_NUM_FOLDS + 1, CV_SEED)),
        ("n_rows", fold_assignment(FULL_DEPTH.n_rows - 1, CV_NUM_FOLDS, CV_SEED)),
    ):
        other = set(np.flatnonzero(folds == CV_FOLD_INDEX))
        assert other != base, (
            f"changing {label} left fold {CV_FOLD_INDEX} unchanged, so the "
            f"guard's three numbers do not in fact pin the held-out set."
        )


def test_an_off_by_one_fold_is_a_different_held_out_set():
    folds = fold_assignment(FULL_DEPTH.n_rows, CV_NUM_FOLDS, CV_SEED)
    fold0 = set(np.flatnonzero(folds == 0))
    fold1 = set(np.flatnonzero(folds == 1))
    assert fold0.isdisjoint(fold1)
    assert len(fold0) == FULL_DEPTH.n_val


def _write_parquet(path, rows):
    """A Parquet with just the three columns the digest reads."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table(
            {
                "clade": [r[0] for r in rows],
                "parent": [r[1] for r in rows],
                "child": [r[2] for r in rows],
            }
        ),
        path,
    )


def test_the_digest_is_order_sensitive(tmp_path):
    """Reversing the rows changes the digest, and nothing else about the file.

    This is the single failure the guard exists to catch: a rebuilt corpus
    holding exactly the paper's PCPs, in a different order, draws a different
    held-out 10%. A sorted or set-based digest would pass on it. The row count
    and the multiset of rows are identical here, so only order can be what
    moved.
    """
    rows = [(f"cl{i % 7}", f"AAA{i}", f"AAC{i}") for i in range(200)]
    forward = tmp_path / "forward.parquet"
    reversed_ = tmp_path / "reversed.parquet"
    _write_parquet(forward, rows)
    _write_parquet(reversed_, list(reversed(rows)))

    forward_digest, forward_rows = row_order_digest(forward)
    reversed_digest, reversed_rows = row_order_digest(reversed_)

    assert (forward_rows, reversed_rows) == (len(rows), len(rows))
    assert forward_digest != reversed_digest
    assert len(forward_digest) == 32


def test_the_digest_is_reproducible_and_covers_all_three_columns(tmp_path):
    """Rewriting the same rows gives the same digest; changing any column does not.

    The first half is what lets a rebuild be checked at all. The second is why
    the digest reads three columns rather than one: `child` alone would miss a
    corpus whose parents were rebuilt against a different clade founder.
    """
    rows = [(f"cl{i % 7}", f"AAA{i}", f"AAC{i}") for i in range(200)]
    again = tmp_path / "again.parquet"
    _write_parquet(again, rows)
    _write_parquet(tmp_path / "original.parquet", rows)
    assert (
        row_order_digest(again)[0] == row_order_digest(tmp_path / "original.parquet")[0]
    )

    for column in range(3):
        changed = list(rows)
        altered = list(changed[0])
        altered[column] = altered[column] + "X"
        changed[0] = tuple(altered)
        path = tmp_path / f"changed{column}.parquet"
        _write_parquet(path, changed)
        assert row_order_digest(path)[0] != row_order_digest(again)[0], (
            f"a change to column {DIGEST_COLUMNS[column]!r} left the digest "
            f"unchanged, so that column is not in fact covered."
        )
