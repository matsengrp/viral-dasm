"""Shared helpers for sequence-based dataset factories.

Hosts utilities used by more than one head's data pipeline so the duplicate
copies (chain concatenation, group-aware train/val split) live in exactly
one place — see Constitution VI ("Extend, Don't Duplicate").
"""

import warnings
from typing import TYPE_CHECKING, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from dasm2.sequence.constants import AA_SEPARATOR

if TYPE_CHECKING:
    from dasm2.config import CvSplitConfig


def clean_chain_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return ``df[col]`` as a NaN-safe, whitespace-stripped string series.

    NaN becomes ``""`` (``astype(str)`` alone produces the literal string
    ``"nan"`` and silently corrupts the sequence). Leading/trailing
    whitespace is dropped.
    """
    return df[col].fillna("").astype(str).str.strip()


def concat_chains_with_separator(
    df: pd.DataFrame, sequence_cols: Sequence[str]
) -> List[str]:
    """Concatenate per-row chain columns with ``AA_SEPARATOR``.

    Each column is cleaned via :func:`clean_chain_series` so NaN becomes
    ``""`` and whitespace is stripped before joining. Rows with an empty
    chain on one side still carry the separator, so ``num_segments`` stays
    consistent across the whole dataset and downstream layout/mask
    computation produces a valid layout with zero-length segments.
    """
    if not sequence_cols:
        raise ValueError("sequence_cols must not be empty.")
    parts = [clean_chain_series(df, c) for c in sequence_cols]
    combined = parts[0]
    for chunk in parts[1:]:
        combined = combined + AA_SEPARATOR + chunk
    return combined.tolist()


def split_df_train_val(
    df: pd.DataFrame,
    seed: int,
    train_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/val.

    When a ``group_id`` column is present, the split is group-aware (whole
    groups land in one side) to avoid leakage. Otherwise rows are shuffled
    and split at the requested fraction. Coarse group counts can make the
    realized fraction drift from the requested one — when that drift is
    >5pp, a ``UserWarning`` is emitted.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}.")

    generator = torch.Generator().manual_seed(seed)

    if "group_id" in df.columns:
        unique_groups = sorted(df["group_id"].unique())
        perm = torch.randperm(len(unique_groups), generator=generator).tolist()
        split_idx = int(train_fraction * len(unique_groups))
        train_groups = {unique_groups[perm[i]] for i in range(split_idx)}
        train_mask = df["group_id"].isin(train_groups)
        train_df = df[train_mask].reset_index(drop=True)
        val_df = df[~train_mask].reset_index(drop=True)
        actual_frac = len(train_df) / len(df) if len(df) else 0.0
        if abs(actual_frac - train_fraction) > 0.05:
            warnings.warn(
                f"Group-aware split produced {actual_frac:.0%} train / "
                f"{1 - actual_frac:.0%} val (requested {train_fraction:.0%}). "
                f"Only {len(unique_groups)} groups available.",
                stacklevel=2,
            )
        return train_df, val_df

    n = len(df)
    indices = torch.randperm(n, generator=generator).tolist()
    split = int(train_fraction * n)
    train_df = df.iloc[indices[:split]].reset_index(drop=True)
    val_df = df.iloc[indices[split:]].reset_index(drop=True)
    return train_df, val_df


def group_kfold_in_train(
    groups: "np.ndarray",
    cv_split: "CvSplitConfig",
) -> "np.ndarray":
    """Run GroupKFold over ``groups`` and return the per-row in_train mask.

    Validates that ``num_folds <= n_unique_groups`` with a clear error
    (sklearn's own message points users at ``n_splits``, which is the
    internal name and easy to miss).

    ``cv_split.group_col`` is read only to make the error message
    actionable; the actual partitioning is determined entirely by
    ``num_folds``, ``fold_index``, and the unique values in ``groups``.

    Promoted from ``dasm2.data.evolutionary_dataset`` (issue #923) so both
    the evolutionary/whichmut CV path and ``dasm2.data.mlm`` share one
    implementation — see Constitution VI ("Extend, Don't Duplicate").
    """
    n_unique_groups = len(set(groups))
    if cv_split.num_folds > n_unique_groups:
        raise ValueError(
            f"cv_split.num_folds ({cv_split.num_folds}) exceeds the number of "
            f"unique groups in '{cv_split.group_col}' ({n_unique_groups}). "
            "Reduce num_folds or pick a column with more unique values."
        )
    gkf = GroupKFold(n_splits=cv_split.num_folds)
    folds = list(gkf.split(np.arange(len(groups)), groups=groups))
    train_idx, _val_idx = folds[cv_split.fold_index]
    in_train = np.zeros(len(groups), dtype=bool)
    in_train[train_idx] = True
    return in_train


def nested_group_kfold_split(
    groups: "np.ndarray",
    cv_split: "CvSplitConfig",
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Split row indices into group-disjoint ``(train, val, test)`` sets.

    Outer ``GroupKFold(cv_split.num_folds)``: fold ``cv_split.fold_index`` is
    the test set. Inner ``GroupKFold(cv_split.val_num_folds)`` over the
    groups that remain: its fold 0 is the validation set and the rest is
    train. No group appears in more than one of the three sets, and the test
    rows never reach training or model selection.

    Both splits go through :func:`group_kfold_in_train`, so they are exactly
    sklearn's ``GroupKFold`` (deterministic given the group labels and their
    counts; ``cv_split.seed`` is unused).

    ``GroupKFold`` fills fold 0 first with the largest group, so inner fold 0
    (validation) is dominated by the largest remaining group. Cap very large
    groups upstream (a data-preparation step) so validation is not one
    cluster.

    Args:
        groups: Per-row group labels, one entry per row of the dataset.
        cv_split: Must have ``val_num_folds`` set.

    Returns:
        ``(train_idx, val_idx, test_idx)``: sorted row positions into the
        array ``groups`` was built from.
    """
    if cv_split.val_num_folds is None:
        raise ValueError(
            "nested_group_kfold_split requires cv_split.val_num_folds; use "
            "group_kfold_in_train for a plain train/val split."
        )
    in_pool = group_kfold_in_train(groups, cv_split)
    pool_idx = np.flatnonzero(in_pool)
    test_idx = np.flatnonzero(~in_pool)
    n_pool_groups = len(set(groups[pool_idx]))
    if cv_split.val_num_folds > n_pool_groups:
        raise ValueError(
            f"cv_split.val_num_folds ({cv_split.val_num_folds}) exceeds the "
            f"{n_pool_groups} groups left in '{cv_split.group_col}' after the "
            "test fold is removed. Reduce val_num_folds or num_folds."
        )
    inner_split = cv_split.model_copy(
        update={
            "num_folds": cv_split.val_num_folds,
            "fold_index": 0,
            "val_num_folds": None,
        }
    )
    in_train = group_kfold_in_train(groups[pool_idx], inner_split)
    return pool_idx[in_train], pool_idx[~in_train], test_idx


__all__ = [
    "clean_chain_series",
    "concat_chains_with_separator",
    "group_kfold_in_train",
    "nested_group_kfold_split",
    "split_df_train_val",
]
