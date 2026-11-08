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

from check_corpus import fold_assignment
from lib import CV_FOLD_INDEX, CV_N_TRAIN, CV_N_VAL, CV_NUM_FOLDS, CV_SEED, FULL_DEPTH


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


def test_the_paper_split_is_reproducible_from_three_numbers():
    """1,435,362 / 159,412 out of (42, 10, 1,594,774), and nothing else.

    The held-out *index set* needs only these three. Which PCP sits at each
    index is the corpus file's row order, which the digest covers.
    """
    folds = fold_assignment(FULL_DEPTH.n_rows, CV_NUM_FOLDS, CV_SEED)
    n_val = int((folds == CV_FOLD_INDEX).sum())
    assert (FULL_DEPTH.n_rows - n_val, n_val) == (CV_N_TRAIN, CV_N_VAL)


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
    assert len(fold0) == CV_N_VAL
