"""Utilities for handling replicate measurements in functional data.

Functional datasets often contain multiple measurements (replicates) for the
same unique sequence. This module provides utilities for:
- Detecting replicates based on sequence columns
- Averaging replicate measurements
- Reporting replicate statistics
"""

from typing import List, Optional

import pandas as pd


def average_replicates(
    df: pd.DataFrame,
    sequence_cols: List[str],
    y_cols: Optional[List[str]] = None,
    count_col: str = "replicate_count",
) -> pd.DataFrame:
    """Average replicate measurements for identical sequences.

    Groups rows by the specified sequence columns and computes the mean of
    numeric y columns. Useful for datasets with technical or biological
    replicates where you want one measurement per unique sequence.

    Args:
        df: DataFrame with sequence and measurement columns.
        sequence_cols: Columns that define unique sequences (grouping keys).
            Rows with identical values in these columns are considered replicates.
        y_cols: Columns to average. If None, averages all columns starting with
            "y_" (standard functional data convention).
        count_col: Name for the column storing replicate counts. Set to None
            to skip adding this column.

    Returns:
        DataFrame with one row per unique sequence combination, where y columns
        contain the mean across replicates. Sequence columns define the grouping,
        and other non-y columns are preserved using the first value from each group.

    Raises:
        ValueError: If sequence_cols are not found in DataFrame.

    Example:
        >>> # Dataset with 3 replicates per sequence
        >>> df = pd.DataFrame({
        ...     "sequence_heavy": ["AAA", "AAA", "AAA", "BBB", "BBB"],
        ...     "y_binding": [1.0, 1.2, 0.8, 2.0, 2.2],
        ... })
        >>> averaged = average_replicates(df, ["sequence_heavy"])
        >>> len(averaged)
        2
        >>> averaged["y_binding"].tolist()
        [1.0, 2.1]
    """

    # Validate sequence columns exist
    missing_cols = [col for col in sequence_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Sequence columns not found: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Determine which columns to average
    if y_cols is None:
        y_cols = [col for col in df.columns if col.startswith("y_")]

    # Validate y columns exist
    missing_y = [col for col in y_cols if col not in df.columns]
    if missing_y:
        raise ValueError(
            f"Y columns not found: {missing_y}. Available columns: {list(df.columns)}"
        )

    # Group by sequence columns
    grouped = df.groupby(sequence_cols, as_index=False)

    # Compute means for y columns
    result = grouped[y_cols].mean()

    # Add replicate count if requested
    if count_col is not None:
        counts = df.groupby(sequence_cols, as_index=False).size()
        counts = counts.rename(columns={"size": count_col})  # type: ignore[call-overload]
        result = result.merge(counts, on=sequence_cols)

    # Preserve additional non-y, non-sequence columns (e.g., is_reference)
    # by merging the first value from each group
    preserve_cols = [
        col
        for col in df.columns
        if col not in y_cols and col not in sequence_cols and col != count_col
    ]

    if preserve_cols:
        first_values = grouped[preserve_cols].first()
        result = result.merge(first_values, on=sequence_cols)

    return result


def replicate_summary(
    df: pd.DataFrame,
    sequence_cols: List[str],
) -> pd.DataFrame:
    """Compute summary statistics for replicates.

    Args:
        df: DataFrame with sequence columns.
        sequence_cols: Columns that define unique sequences.

    Returns:
        DataFrame with summary statistics:
        - total_rows: Total number of rows in input
        - unique_sequences: Number of unique sequence combinations
        - mean_replicates: Mean number of replicates per unique sequence
        - min_replicates: Minimum replicates
        - max_replicates: Maximum replicates

    Raises:
        ValueError: If sequence_cols are not found in DataFrame.
    """
    # Validate sequence columns exist
    missing_cols = [col for col in sequence_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Sequence columns not found: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    counts = df.groupby(sequence_cols).size()

    return pd.DataFrame(
        {
            "total_rows": [len(df)],
            "unique_sequences": [len(counts)],
            "mean_replicates": [counts.mean()],
            "min_replicates": [counts.min()],
            "max_replicates": [counts.max()],
        }
    )
