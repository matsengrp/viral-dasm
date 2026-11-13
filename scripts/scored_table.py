"""Concatenate the per-run scored tables into the one the figures read.

Streamed row group by row group rather than concatenated in pandas. Seven runs
at 1,425,760 rows each is 9,980,320 rows over 24 columns, and a pandas concat
holds the inputs and the result at once; writing through
``pyarrow.parquet.ParquetWriter`` holds one row group.

The schemas are required to match exactly. A column present in one run's table
and absent from another would otherwise concatenate to nulls that look like
"this run predicted nothing here", which is indistinguishable in the output
from a mutation mat-pcp never estimated.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2`; then `dasm2` before pyarrow, for the Arrow memory pool.
from lib import assert_dasm2_is_this_repo

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import argparse

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    readers = [pq.ParquetFile(path) for path in args.inputs]
    schema = readers[0].schema_arrow
    for path, reader in zip(args.inputs, readers):
        if reader.schema_arrow != schema:
            raise SystemExit(
                f"{path} has a different schema from {args.inputs[0]}. Every "
                f"run's table must carry the same columns and types: a column "
                f"missing from one run concatenates to nulls that read as a "
                f"prediction of nothing rather than as an absent column.\n"
                f"  expected: {schema}\n  found:    {reader.schema_arrow}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with pq.ParquetWriter(args.output, schema) as writer:
        for path, reader in zip(args.inputs, readers):
            for batch in reader.iter_batches():
                writer.write_batch(batch)
                rows += batch.num_rows
            print(f"  {path.name}: {reader.metadata.num_rows:,} rows", flush=True)

    print(f"wrote {args.output}: {rows:,} rows from {len(args.inputs)} runs")


if __name__ == "__main__":
    main()
