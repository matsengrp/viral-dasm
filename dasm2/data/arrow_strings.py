"""Row-aligned numpy views of parent/child NT columns, read off Arrow buffers.

The columnar passes over Parquet PCPs (the branch-length cold start and the
scored-substitution guard) compare parent and child
sequences byte for byte without building Python strings. They share the
reading: null-fill each segment column, split both columns at the union of
their chunk boundaries, take zero-copy ``(offsets, data)`` views of each
slice, and stream a backend's row groups under its ``row_filter``. This module
holds that reading; each caller keeps its own row checks and reduction.
"""

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pyarrow as pa

if TYPE_CHECKING:
    from dasm2.data.parquet_backend import ParquetBackend


def offset_dtype(arrow_type: pa.DataType) -> np.dtype:
    """Numpy dtype of an Arrow variable-width type's offset buffer.

    Arrow's ``string``/``binary`` carry int32 offsets and
    ``large_string``/``large_binary`` int64 ones. This must be read off the
    array's own type and never assumed: mismatching the width does not fail
    loudly. Reading int64 offsets as int32 yields the interleaved low and high
    words of each offset, and any offset whose low word has its top bit set
    comes back negative — the ``IndexError: index -2147481961 out-of-bounds``
    seen while probing the v10 corpus, whose NT columns are ``large_string``.

    Raises:
        TypeError: If the type has no 32- or 64-bit offset buffer.
    """
    if pa.types.is_string(arrow_type) or pa.types.is_binary(arrow_type):
        return np.dtype(np.int32)
    if pa.types.is_large_string(arrow_type) or pa.types.is_large_binary(arrow_type):
        return np.dtype(np.int64)
    raise TypeError(
        f"Expected an Arrow string or binary column with an offsets buffer, "
        f"got {arrow_type}"
    )


def string_array_views(arr: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """Zero-copy ``(offsets, data)`` numpy views of one Arrow string array.

    Returns offsets rebased to zero (int64, length ``len(arr) + 1``) and the
    exact byte range they index, so ``data[offsets[i]:offsets[i + 1]]`` is row
    ``i``'s bytes. Four ways to read a plausible-but-wrong answer off Arrow
    string buffers are closed here rather than assumed away;
    none of them would crash on its own, they would all just shift or pad the
    numbers that become branch lengths:

    1. **Offset width** — taken from the array's type, see ``offset_dtype``.
    2. **Array offset** — a sliced array shares its parent's buffers and
       records the shift in ``arr.offset``, so the offsets buffer must be read
       from there, not from its start.
    3. **Offset base** — a slice's first offset is not 0, so offsets are
       rebased and the data view starts at ``offsets[0]``.
    4. **Buffer capacity** — the data buffer is allocated to a padded capacity,
       so its length is an upper bound on ``offsets[-1]``, not equal to it.
       Trailing capacity bytes are zeros, which pass the ambiguity mask, so
       including them would silently add phantom mutable sites to the last row.

    None of those is then trusted: the offsets buffer must be large enough for
    the requested rows at the chosen width, and the offsets themselves must be
    non-negative, non-decreasing, and within the data buffer. Any residual
    overflow — including a hypothetical int32 ``string`` column built past
    2**31 bytes — lands here as a ``ValueError`` instead of as wrong branch
    lengths.

    Raises:
        ValueError: If the offsets buffer is too small, or the offsets are
            negative, decreasing, or run past the data buffer.
        TypeError: If ``arr`` is not an Arrow string or binary array.
    """
    dtype = offset_dtype(arr.type)
    if len(arr) == 0:
        return np.zeros(1, dtype=np.int64), np.empty(0, dtype=np.uint8)

    _, offsets_buffer, data_buffer = arr.buffers()[:3]
    if offsets_buffer is None:
        raise ValueError(f"Arrow array of {len(arr)} rows has no offsets buffer")
    needed = (arr.offset + len(arr) + 1) * dtype.itemsize
    if len(offsets_buffer) < needed:
        raise ValueError(
            f"Arrow {arr.type} offsets buffer holds {len(offsets_buffer)} bytes "
            f"but {len(arr)} rows at offset {arr.offset} need {needed} at "
            f"{dtype.itemsize * 8}-bit width"
        )
    offsets = np.frombuffer(
        offsets_buffer,
        dtype=dtype,
        offset=arr.offset * dtype.itemsize,
        count=len(arr) + 1,
    ).astype(np.int64)

    if offsets[0] < 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(
            f"Arrow {arr.type} offsets are not non-negative and non-decreasing, "
            f"which means they overflowed or were read at the wrong width. "
            f"First offsets: {offsets[:8].tolist()}"
        )
    first, last = int(offsets[0]), int(offsets[-1])
    available = 0 if data_buffer is None else len(data_buffer)
    if last > available:
        raise ValueError(
            f"Arrow {arr.type} offsets end at {last} but the data buffer holds "
            f"only {available} bytes"
        )
    data = (
        np.empty(0, dtype=np.uint8)
        if data_buffer is None
        else np.frombuffer(
            data_buffer, dtype=np.uint8, offset=first, count=last - first
        )
    )
    return offsets - first, data


def iter_aligned_chunks(
    columns: Sequence[pa.ChunkedArray],
) -> Iterator[tuple[pa.Array, ...]]:
    """Yield row-aligned single-chunk slices of several equal-length columns.

    Arrow may split a column into chunks at will — and *must* once a
    32-bit-offset ``string`` column passes 2**31 bytes — and two columns of the
    same table need not share chunk boundaries. Flattening with
    ``combine_chunks()`` first is exactly what overflows in that case, so this
    instead walks the union of every column's chunk boundaries and slices each
    column to each interval. Every yielded tuple therefore holds one contiguous
    chunk per column, all covering the same rows, so their byte buffers line up
    row for row and can be compared position-wise.

    Slices are zero-copy views that carry a non-zero ``arr.offset``, which
    ``string_array_views`` accounts for.

    Raises:
        ValueError: If the columns do not all have the same number of rows.
    """
    lengths = {len(column) for column in columns}
    if len(lengths) > 1:
        raise ValueError(
            f"Columns must have equal lengths to align, got {sorted(lengths)}"
        )

    cuts = {0}
    for column in columns:
        position = 0
        for chunk in column.iterchunks():
            position += len(chunk)
            cuts.add(position)
    boundaries = sorted(cuts)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        sliced = []
        for column in columns:
            chunks = [
                c for c in column.slice(start, end - start).iterchunks() if len(c)
            ]
            if len(chunks) != 1:
                raise AssertionError(
                    f"Rows [{start}, {end}) span {len(chunks)} chunks after "
                    f"slicing at a chunk boundary; this should be impossible"
                )
            sliced.append(chunks[0])
        yield tuple(sliced)


class SegmentBlock(NamedTuple):
    """One row-aligned block of a parent/child segment column pair.

    ``parent_nt[parent_offsets[i]:parent_offsets[i + 1]]`` is row
    ``rows.start + i``'s parent bytes, and likewise for the child. Nothing
    here checks that the two agree in length; each caller does, with its own
    rule and message. Unpacks in field order.
    """

    rows: slice
    """Rows of the whole column this block covers."""
    parent_offsets: np.ndarray
    parent_nt: np.ndarray
    child_offsets: np.ndarray
    child_nt: np.ndarray


def iter_segment_blocks(
    parent_column: pa.ChunkedArray, child_column: pa.ChunkedArray
) -> Iterator[SegmentBlock]:
    """Yield byte views of one segment's parent and child columns, block by block.

    Nulls are the Parquet spelling of an absent segment, which the row
    converter reads as ``""``. Filling them makes a null row's offsets
    unambiguously zero-length, rather than relying on Arrow's unspecified
    offsets for null slots.

    Raises:
        ValueError: If the columns differ in row count, or on any of the
            buffer errors :func:`string_array_views` rejects.
    """
    if parent_column.null_count:
        parent_column = parent_column.fill_null("")
    if child_column.null_count:
        child_column = child_column.fill_null("")
    row = 0
    for parent_chunk, child_chunk in iter_aligned_chunks([parent_column, child_column]):
        parent_offsets, parent_nt = string_array_views(parent_chunk)
        child_offsets, child_nt = string_array_views(child_chunk)
        rows = slice(row, row + len(parent_chunk))
        row = rows.stop
        yield SegmentBlock(rows, parent_offsets, parent_nt, child_offsets, child_nt)


def iter_selected_row_group_values(
    backend: "ParquetBackend",
    parent_names: Sequence[str],
    child_names: Sequence[str],
    per_row: Callable[[list[pa.ChunkedArray], list[pa.ChunkedArray]], np.ndarray],
) -> Iterator[np.ndarray]:
    """Apply ``per_row`` to each row group's segment columns, keeping selected rows.

    Streams ``parent_names + child_names`` one row group at a time through
    ``ParquetBackend.iter_row_group_columns``, calls ``per_row`` with the
    parent and child columns in segment order, and yields its result restricted
    to the rows the backend's ``row_filter`` selects. Concatenating the yields
    gives one value per selected row, in filtered index order.

    Args:
        backend: Parquet backend to stream from.
        parent_names: Parent segment columns, in segment order.
        child_names: Child segment columns, same segments.
        per_row: Maps one row group's columns to an array with one entry per
            row of that row group.
    """
    for table, selection in backend.iter_row_group_columns(
        list(parent_names) + list(child_names)
    ):
        values = per_row(
            [table.column(name) for name in parent_names],
            [table.column(name) for name in child_names],
        )
        yield values if selection is None else values[selection]
