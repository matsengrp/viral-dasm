"""Data validation functions for parent-child pair datasets.

Functions for validating and filtering parent-child pair (PCP) data,
ensuring sequences are suitable for evolutionary modeling.
"""

from collections.abc import Sequence
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import pyarrow as pa

from dasm2.data.arrow_strings import iter_segment_blocks
from dasm2.sequence.constants import BASES, CODON_TO_IDX, NT_SEPARATOR, STOP_CODONS
from dasm2.sequence.conversion import contains_stop_codon

_SENSE_CODONS = frozenset(CODON_TO_IDX) - frozenset(STOP_CODONS)
"""Parent codons the loss scores: ``ACGT`` codons that are not stops."""

_RADIX = len(BASES) + 1
"""A codon's key is its three digits in base 5: a base's ``BASES`` index, or 4."""

_DIGIT = np.full(256, len(BASES), dtype=np.uint8)
for _idx, _base in enumerate(BASES):
    _DIGIT[ord(_base)] = _idx
"""Byte -> its index in ``BASES``, or 4 if it is not ``ACGT``."""

_PLACE_VALUES = tuple(
    (_DIGIT.astype(np.uint16) * _RADIX**place).astype(np.uint8) for place in (2, 1, 0)
)
"""Per codon position, byte -> its digit times its place value. A key is at
most ``4 * 25 + 4 * 5 + 4 = 124``, so ``uint8`` sums never overflow."""


def _codon_key(codon: str) -> int:
    return sum(int(table[ord(base)]) for table, base in zip(_PLACE_VALUES, codon))


_IS_SCORED_PARENT_KEY = np.zeros(_RADIX**3, dtype=bool)
_IS_SCORED_PARENT_KEY[[_codon_key(codon) for codon in _SENSE_CODONS]] = True
"""Codon key -> whether the loss scores it as a parent: a sense ``ACGT`` codon."""

_IS_SCORED_CHILD_KEY = np.zeros(_RADIX**3, dtype=bool)
_IS_SCORED_CHILD_KEY[[_codon_key(codon) for codon in CODON_TO_IDX]] = True
"""Codon key -> whether the loss scores it as a child: any ``ACGT`` codon."""


def _codon_keys(nt: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Keys of the codons starting at byte positions ``starts`` of ``nt``."""
    first_table, second_table, third_table = _PLACE_VALUES
    keys: np.ndarray = (
        first_table[nt[starts]]
        + second_table[nt[starts + 1]]
        + third_table[nt[starts + 2]]
    )
    return keys


def assert_pcp_valid(parent: str, child: str) -> None:
    """Assert that a parent-child pair is valid for evolutionary modeling.

    A valid PCP must have:
    - Both sequences non-empty
    - Equal length parent and child
    - Length divisible by 3 (codon structure)
    - No stop codons in either sequence
    - No ambiguous nucleotides (N) in either sequence

    Args:
        parent: Parent nucleotide sequence
        child: Child nucleotide sequence

    Raises:
        AssertionError: If any validation check fails
    """
    assert len(parent) > 0, "Parent sequence is empty"
    assert len(child) > 0, "Child sequence is empty"
    assert len(parent) == len(child), (
        f"Parent length ({len(parent)}) != child length ({len(child)})"
    )
    assert len(parent) % 3 == 0, f"Sequence length ({len(parent)}) not divisible by 3"
    assert not contains_stop_codon(parent), f"Parent has stop codon: {parent}"
    assert not contains_stop_codon(child), f"Child has stop codon: {child}"
    assert "N" not in parent.upper(), f"Parent has ambiguous nucleotides: {parent}"
    assert "N" not in child.upper(), f"Child has ambiguous nucleotides: {child}"


def is_pcp_valid(parent: str, child: str) -> bool:
    """Check if a parent-child pair is valid for evolutionary modeling.

    Args:
        parent: Parent nucleotide sequence
        child: Child nucleotide sequence

    Returns:
        True if the PCP passes all validation checks, False otherwise
    """
    try:
        assert_pcp_valid(parent, child)
        return True
    except AssertionError:
        return False


def filter_valid_pcps(
    pcp_df: pd.DataFrame,
    parent_columns: Tuple[str, ...],
    child_columns: Tuple[str, ...],
) -> pd.DataFrame:
    """Filter DataFrame to keep only rows with valid parent-child pairs.

    Validates all specified parent/child column pairs and keeps rows where
    ALL segment pairs are valid. A pair is valid if:
    - Both sequences are non-empty
    - Parent and child have equal length
    - Length is divisible by 3
    - No stop codons in either sequence
    - No ambiguous nucleotides (N)

    Args:
        pcp_df: DataFrame containing PCP data
        parent_columns: Column names for parent sequences (e.g., ("parent_heavy", "parent_light"))
        child_columns: Column names for child sequences (e.g., ("child_heavy", "child_light"))

    Returns:
        Filtered DataFrame containing only valid PCPs

    Example:
        >>> df = pd.DataFrame({
        ...     "parent_heavy": ["ATGATG", "ATGNNN"],
        ...     "child_heavy": ["ATGATC", "ATGNNN"],
        ...     "parent_light": ["GGGCCC", ""],
        ...     "child_light": ["GGGCCT", ""],
        ... })
        >>> valid_df = filter_valid_pcps(
        ...     df,
        ...     ("parent_heavy", "parent_light"),
        ...     ("child_heavy", "child_light"),
        ... )
        >>> len(valid_df)
        1
    """
    if len(parent_columns) != len(child_columns):
        raise ValueError(
            f"parent_columns length ({len(parent_columns)}) != "
            f"child_columns length ({len(child_columns)})"
        )

    valid_mask = pd.Series([True] * len(pcp_df), index=pcp_df.index)
    validated_segments = 0

    for parent_col, child_col in zip(parent_columns, child_columns):
        # Skip column pairs that don't exist (supports heavy-only, light-only data)
        if parent_col not in pcp_df.columns or child_col not in pcp_df.columns:
            continue

        validated_segments += 1

        # Check each row for this segment pair
        segment_valid = pcp_df.apply(
            lambda row, p=parent_col, c=child_col: _is_segment_valid(row[p], row[c]),
            axis=1,
        )
        valid_mask = valid_mask & segment_valid

    if validated_segments == 0:
        raise ValueError(
            f"No valid column pairs found. Requested parent columns: {parent_columns}, "
            f"child columns: {child_columns}. DataFrame columns: {list(pcp_df.columns)}"
        )

    return _report_filtered(pcp_df, valid_mask, "invalid PCPs")


def _report_filtered(
    pcp_df: pd.DataFrame,
    valid_mask: "pd.Series[bool]",
    description: str,
) -> pd.DataFrame:
    """Apply a boolean mask to a DataFrame and report how many rows were removed.

    Args:
        pcp_df: DataFrame to filter
        valid_mask: Boolean Series where True = keep row
        description: What was filtered (for the log message)

    Returns:
        Filtered copy of the DataFrame
    """
    initial_count = len(pcp_df)
    filtered_df = pcp_df[valid_mask].copy()
    final_count = len(filtered_df)

    if initial_count != final_count:
        print(
            f"Filtered {description}: {initial_count} → {final_count} "
            f"(removed {initial_count - final_count} rows)"
        )

    return filtered_df


def filter_naive_pcps(pcp_df: pd.DataFrame) -> pd.DataFrame:
    """Remove PCPs where the parent is the naive/root sequence.

    Handles both ``parent_is_naive`` (real PCP pipelines) and
    ``parent_is_root`` (tree-based simulation output).

    Args:
        pcp_df: DataFrame of parent-child pairs.

    Returns:
        Filtered copy with naive/root-parent rows removed.

    Raises:
        KeyError: If neither column is present.
    """
    if "parent_is_naive" in pcp_df.columns:
        mask = ~pcp_df["parent_is_naive"]
    elif "parent_is_root" in pcp_df.columns:
        mask = ~pcp_df["parent_is_root"]
    else:
        raise KeyError(
            "Expected 'parent_is_naive' or 'parent_is_root' column in DataFrame"
        )
    return _report_filtered(pcp_df, mask, "naive/root-parent PCPs")


def _is_segment_valid(parent: object, child: object) -> bool:
    """Check if a single segment pair is valid, handling empty segments.

    Empty segments (both parent and child empty) are considered valid
    to support datasets with optional segments (e.g., heavy-only antibody data).

    Args:
        parent: Parent nucleotide sequence (may be str or NA)
        child: Child nucleotide sequence (may be str or NA)

    Returns:
        True if the segment pair is valid
    """
    # Handle pandas NA/NaN values
    # Cast to Any for pd.isna compatibility with pandas-stubs
    parent_str = "" if pd.isna(cast(Any, parent)) else str(parent)
    child_str = "" if pd.isna(cast(Any, child)) else str(child)

    # Both empty is valid (optional segment)
    if len(parent_str) == 0 and len(child_str) == 0:
        return True

    # One empty but not both is invalid
    if len(parent_str) == 0 or len(child_str) == 0:
        return False

    return is_pcp_valid(parent_str, child_str)


def filter_identical_pcps(
    pcp_df: pd.DataFrame,
    segment_suffixes: Tuple[str, ...] = ("_heavy", "_light"),
) -> pd.DataFrame:
    """Remove PCPs where parent and child are identical across all segments.

    Checks string equality for each segment column pair. Columns that don't
    exist in the DataFrame are silently skipped (supports heavy-only data).

    Args:
        pcp_df: DataFrame with PCP data.
        segment_suffixes: Segment suffixes (e.g., ``("_heavy", "_light")``).

    Returns:
        Filtered DataFrame with identical parent-child pairs removed.
    """
    parent_cols, child_cols = get_pcp_column_pairs(segment_suffixes)
    has_mutation = pd.Series(False, index=pcp_df.index)
    for parent_col, child_col in zip(parent_cols, child_cols):
        if parent_col in pcp_df.columns and child_col in pcp_df.columns:
            has_mutation = has_mutation | (pcp_df[parent_col] != pcp_df[child_col])
    return _report_filtered(pcp_df, has_mutation, "identical parent-child pairs")


def has_scored_substitution(parent: str, child: str) -> bool:
    """Whether a PCP has at least one codon substitution the DASM loss scores.

    A codon position is a scored substitution when the parent codon is a
    sense codon over ``ACGT``, the child codon is over ``ACGT``, and the two
    differ. Under ``_evolutionary_transforms`` this is exactly
    ``(loss_mask & (codon_parents_idxs != codon_children_idxs)).any()``:
    ``ComputeMasks`` excludes ambiguous parent and child codons, parent stop
    codons (which translate to ``X``), and the ``^^^`` separator. Any
    character other than ``ACGT``, a gap included, makes its codon ambiguous.

    Under ``_whichmut_transforms`` it is necessary but not sufficient. That
    chain also narrows the mask to single-nucleotide codon changes, and with
    a censor table or reference bases it censors sites too. So a PCP whose
    only scored substitutions are multi-nucleotide or censored passes here
    but has an empty Whichmut loss mask. Issue #1243 leaves such rows out of
    scope.

    A PCP without one carries no information: its branch length fits to
    zero, where the target is the parent codon at every scored site
    (issue #1243).

    Args:
        parent: Parent NT sequence, segments joined by ``^^^``.
        child: Child NT sequence, aligned to ``parent``.

    Returns:
        True if any segment has a scored substitution.

    Raises:
        ValueError: If the segments do not line up, or a segment's length is
            not a multiple of 3.
    """
    parent_segments = parent.split(NT_SEPARATOR)
    child_segments = child.split(NT_SEPARATOR)
    if len(parent_segments) != len(child_segments):
        raise ValueError(
            f"Parent has {len(parent_segments)} segments but child has "
            f"{len(child_segments)}"
        )
    for parent_segment, child_segment in zip(parent_segments, child_segments):
        if len(parent_segment) != len(child_segment):
            raise ValueError(
                f"Parent segment length ({len(parent_segment)}) != child "
                f"segment length ({len(child_segment)})"
            )
        if len(parent_segment) % 3:
            raise ValueError(
                f"Segment length ({len(parent_segment)}) not divisible by 3"
            )
        for pos in range(0, len(parent_segment), 3):
            parent_codon = parent_segment[pos : pos + 3]
            child_codon = child_segment[pos : pos + 3]
            if (
                parent_codon != child_codon
                and parent_codon in _SENSE_CODONS
                and child_codon in CODON_TO_IDX
            ):
                return True
    return False


def scored_substitution_flags_from_arrow(
    parent_columns: Sequence[pa.ChunkedArray],
    child_columns: Sequence[pa.ChunkedArray],
) -> np.ndarray:
    """Per-row :func:`has_scored_substitution`, straight off Arrow buffers.

    Takes one Arrow string column per segment, as the Parquet files store
    them, and reads their byte buffers through
    :func:`dasm2.data.arrow_strings.iter_segment_blocks` (issue #992), so it
    builds no Python strings. A null is an absent segment, as
    ``make_parquet_row_converter`` reads it.
    Every segment is a whole number of codons, so a ``^^^`` joined into one
    column lands on its own codon, which is not ``ACGT`` and never scores.

    Args:
        parent_columns: Parent sequences, one column per segment.
        child_columns: Child sequences, same segments and rows.

    Returns:
        Boolean array with one entry per row.

    Raises:
        ValueError: If the segment counts differ, or a row's parent and child
            segments are not equal multiples of 3 in length.
    """
    if len(parent_columns) != len(child_columns):
        raise ValueError(
            f"Segment count mismatch: {len(parent_columns)} parent columns, "
            f"{len(child_columns)} child columns"
        )
    num_rows = len(parent_columns[0]) if parent_columns else 0
    flags = np.zeros(num_rows, dtype=bool)
    for segment, (parent_column, child_column) in enumerate(
        zip(parent_columns, child_columns)
    ):
        for block, offsets, parent_nt, child_offsets, child_nt in iter_segment_blocks(
            parent_column, child_column
        ):
            lengths = offsets[1:] - offsets[:-1]
            child_lengths = child_offsets[1:] - child_offsets[:-1]
            bad = np.flatnonzero((lengths != child_lengths) | (lengths % 3 != 0))
            if bad.size:
                first = int(bad[0])
                raise ValueError(
                    f"Segment {segment}, row {block.start + first}: parent "
                    f"length {int(lengths[first])} and child length "
                    f"{int(child_lengths[first])} must be equal multiples of 3"
                )
            # Only substituted codons can score, and they are a small fraction,
            # so key just those. Three 1-D gathers and uint8 table sums are
            # what keep this pass under the mutation-frequency pass (#1243
            # criterion 3); a row gather of (n, 3) codons was 30% over.
            differs = (parent_nt != child_nt).reshape(-1, 3)
            codons = np.flatnonzero(differs[:, 0] | differs[:, 1] | differs[:, 2])
            codon_starts = 3 * codons
            scored = (
                _IS_SCORED_PARENT_KEY[_codon_keys(parent_nt, codon_starts)]
                & _IS_SCORED_CHILD_KEY[_codon_keys(child_nt, codon_starts)]
            )
            # Codon c of the chunk belongs to the row whose codon range holds it.
            rows = np.searchsorted(offsets // 3, codons[scored], side="right") - 1
            flags[block.start + rows] = True
    return flags


def trim_pcp_segments_to_codon_multiple(
    pcp_df: pd.DataFrame,
    segment_suffixes: Tuple[str, ...] = ("_heavy", "_light"),
    max_len: Optional[int] = None,
) -> pd.DataFrame:
    """Trim each present parent/child segment to a whole number of codons.

    Missing segment pairs are skipped, matching :func:`filter_identical_pcps`,
    so the helper works for heavy-only and single-segment tables as well as
    paired antibody data.

    Args:
        pcp_df: DataFrame with PCP data.
        segment_suffixes: Segment suffixes (e.g., ``("_heavy", "_light")``).
        max_len: Optional cap on the number of nucleotides retained per
            sequence, itself rounded down to a codon boundary. ``None`` (the
            default) keeps every whole codon.

    Returns:
        A copy of ``pcp_df`` with the present segment columns trimmed.

    Raises:
        ValueError: If ``max_len`` is negative.
    """
    if max_len is not None and max_len < 0:
        raise ValueError(f"max_len must be non-negative, got {max_len}")

    parent_cols, child_cols = get_pcp_column_pairs(segment_suffixes)
    trimmed = pcp_df.copy()
    max_codon_len = None if max_len is None else max_len - max_len % 3

    def _trim(sequence: object) -> str:
        value = str(sequence)
        keep = len(value) - len(value) % 3
        if max_codon_len is not None:
            keep = min(keep, max_codon_len)
        return value[:keep]

    for parent_col, child_col in zip(parent_cols, child_cols):
        if parent_col in trimmed.columns and child_col in trimmed.columns:
            trimmed[parent_col] = trimmed[parent_col].map(_trim)
            trimmed[child_col] = trimmed[child_col].map(_trim)
    return trimmed


def get_pcp_column_pairs(
    segment_suffixes: Tuple[str, ...],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Get parent/child column name pairs from segment suffixes.

    Args:
        segment_suffixes: Segment suffixes from a SegmentHandler
            (e.g., ("_heavy", "_light") or ("",))

    Returns:
        Tuple of (parent_columns, child_columns)

    Example:
        >>> get_pcp_column_pairs(("_heavy", "_light"))
        (('parent_heavy', 'parent_light'), ('child_heavy', 'child_light'))
        >>> get_pcp_column_pairs(("",))
        (('parent',), ('child',))
    """
    parent_columns: List[str] = []
    child_columns: List[str] = []

    for suffix in segment_suffixes:
        parent_columns.append(f"parent{suffix}")
        child_columns.append(f"child{suffix}")

    return tuple(parent_columns), tuple(child_columns)
