"""Chunked parquet writer with schema-level metadata.

Writes parquet files incrementally by accepting DataFrames or Arrow tables
one chunk at a time, avoiding the need to hold the full dataset in memory.
Schema-level metadata is embedded in the Arrow schema on first write and
persisted when the file is finalized.

Designed as the write counterpart to :mod:`dasm2.data.parquet_backend`.

Usage::

    with ChunkedParquetWriter(output_path, metadata={"model": "GTR"}) as writer:
        for chunk_df in process_chunks(data):
            writer.write_chunk(chunk_df)

    # File is finalized when the context manager exits.
    # On exception, the partial file is removed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ChunkedParquetWriter:
    """Incrementally write a parquet file from DataFrame chunks.

    Each call to :meth:`write_chunk` converts the chunk to an Arrow table
    and writes it as one or more row groups.  Schema-level metadata is
    embedded in the Arrow schema when the underlying ``ParquetWriter`` is
    created (on the first :meth:`write_chunk` call).

    The Arrow schema is inferred from the first chunk.  All subsequent
    chunks must have the same columns and compatible types.

    When used as a context manager, the file is finalized on normal exit.
    If an exception propagates, the partial file is deleted to avoid
    leaving truncated data on disk.

    Args:
        output_path: Destination parquet file path.
        metadata: String key-value pairs to store as schema-level metadata.
        row_group_size: Maximum rows per row group.
        compression: Parquet compression codec (default ``"zstd"``).
    """

    def __init__(
        self,
        output_path: Path,
        metadata: dict[str, str] | None = None,
        row_group_size: int = 8192,
        compression: str = "zstd",
    ) -> None:
        self._output_path = Path(output_path)
        self._metadata = metadata or {}
        self._row_group_size = row_group_size
        self._compression = compression
        self._writer: pq.ParquetWriter | None = None
        self._rows_written = 0

    @property
    def rows_written(self) -> int:
        """Total rows written so far."""
        return self._rows_written

    def write_chunk(self, chunk: pd.DataFrame | pa.Table) -> None:
        """Write a chunk of data as one or more row groups.

        Args:
            chunk: Data to write.  If a DataFrame, it is converted to an
                Arrow table (index is not preserved).
        """
        if isinstance(chunk, pd.DataFrame):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
        else:
            table = chunk

        if self._writer is None:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            schema_metadata: dict[bytes, bytes] = table.schema.metadata or {}
            schema_metadata.update(
                {k.encode(): v.encode() for k, v in self._metadata.items()}
            )
            schema = table.schema.with_metadata(schema_metadata)
            self._writer = pq.ParquetWriter(
                self._output_path, schema, compression=self._compression
            )

        self._writer.write_table(table, row_group_size=self._row_group_size)
        self._rows_written += len(table)

    def add_key_value_metadata(self, metadata: dict[str, str]) -> None:
        """Add metadata known only after the rows are written (e.g. counts).

        The pairs land in the file footer but not in the Arrow schema, so
        readers must use
        :func:`dasm2.data.parquet_backend.read_parquet_file_metadata`.

        Raises:
            RuntimeError: If no chunk has been written yet, or the file is
                already closed.
        """
        if self._writer is None:
            raise RuntimeError(
                "add_key_value_metadata needs an open writer: write a chunk first"
            )
        self._writer.add_key_value_metadata(metadata)

    def close(self) -> None:
        """Finalize the parquet file.

        Closes the underlying writer so the parquet footer is written.
        Safe to call multiple times.
        """
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "ChunkedParquetWriter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            # Exception during writing — close and remove partial file.
            self.close()
            if self._output_path.exists():
                self._output_path.unlink()
                logger.warning(
                    "Removed partial parquet file %s due to exception",
                    self._output_path,
                )
        else:
            self.close()
