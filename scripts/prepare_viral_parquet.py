"""Prepare a viral training Parquet with per-era precomputed neutral rates.

DASM factors substitution as ``p ∝ λ·f``: a neutral mutation rate times a
selection factor. The viral training path currently serves **one** λ to every
PCP in a run, via a single ``neutral_model`` archive named in the training YAML.
That is wrong whenever the corpus spans a period across which the mutational
spectrum changed -- for SARS-CoV-2 it demonstrably did, and mat-pcp's own
synonymous residuals separate cleanly by pre-Omicron / Omicron era.

This script gives each PCP an era-appropriate λ **without touching the training
path**, by computing ``nt_rates``/``nt_csps`` at data-prep time with the archive
that row's clade belongs to and storing them in the Parquet. The trainer then
reads rates from the file and never learns that more than one neutral model
exists.

It is deliberately *not* specialized to two eras. An era is just a named group
of clades with an archive, so:

- **one era covering every clade** reproduces pooled-λ behaviour, which is what
  makes the per-era change measurable in isolation: same rates, different
  transport, so nothing should move;
- **two eras** is the pre-Omicron / Omicron split this was written for;
- **N eras** is per-clade, if that is ever wanted.

Nothing here fits a rate model. It consumes already-built ``TabularRates``
archives, so it can be exercised against perturbed copies of a rate table
without rebuilding anything.

Usage::

    python scripts/prepare_viral_parquet.py \\
        --pcp-csv <mat-pcp run>/pcps/S/all_pcps_dasm_ready.csv.gz \\
        --eras <artifact root>/era_maps/S.yaml \\
        --output <artifact root>/parquet/S/full_depth.parquet

The era file maps each era to an archive and the clades it covers::

    neutral_model_name: per-era-TabularRates-v10
    eras:
      pre_omicron:
        archive: /path/to/TabularRates-...-E-pre_omicron.zip
        clades: [19A, 19B, 20A, ...]
      omicron:
        archive: /path/to/TabularRates-...-E-omicron.zip
        clades: [21K, 21L, ...]

**Every clade in the corpus must be named.** An unmapped clade raises rather
than falling back to a default era -- a silent default is exactly the
well-formed-archive-with-wrong-numbers failure this whole route has to avoid,
and it would be invisible in the loss curve. The reverse is checked too: an era
matching zero rows raises, which is what catches a typo'd clade name in the YAML
(it would otherwise produce a perfectly valid file in which one archive was
simply never used).

Both checks run in :func:`scan_corpus`, **before** the writer is opened, so a
coverage failure never leaves an output file on disk. Doing them during the
write instead would mean the zero-matched-era case is only detectable once the
Parquet is already complete and valid-looking, and a pipeline gating on file
existence rather than exit status would consume it.

Precision: ``TabularRatesNeutralModel.compute_rates_and_csps`` returns
``torch.float32``, and this writes ``list<float>`` (fp32) under an explicit Arrow
schema. That is lossless -- the stored values are bit-identical to what the
streaming path computes, not merely close -- and it halves the file against
letting Arrow infer from ``.tolist()``, which yields Python floats and therefore
``list<double>``.

GISAID: reads a mat-pcp PCP table and writes a training corpus. Both stay on
the cluster.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import assert_dasm2_is_this_repo

# Then `dasm2`, before pandas, pyarrow or torch. `dasm2/arrow_pool.py` moves
# Arrow off the bundled mimalloc, whose arenas are marked `MADV_HUGEPAGE`, and
# it can only do that if pyarrow has not been imported yet. Importing pandas
# first imports pyarrow, and the switch silently becomes a no-op -- which on a
# 1.6M-row write is the difference between the allocator this pipeline was
# measured with and one asking for transparent huge pages.
import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import argparse
import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd
import pyarrow as pa
import torch
import yaml

from dasm2.data.local import compute_sha256
from dasm2.data.parquet_backend import ParquetBackend, make_parquet_row_converter
from dasm2.data.parquet_writer import ChunkedParquetWriter
from dasm2.data.registry import seam_identity
from dasm2.data.segment_handlers import SegmentHandler
from dasm2.neutral.tabular_rates import TabularRatesNeutralModel

logger = logging.getLogger(__name__)

# Small row groups cost a little size and buy random-access locality; these
# files are read by ShuffleBufferSampler, which reads whole row groups. This is
# also the geometry `scripts/lib.py::PARQUET_CACHE_ROW_GROUPS` is sized
# against, so changing it means re-deriving every corpus's cache budget.
DEFAULT_ROW_GROUP_SIZE = 2048

# Rows held in memory at once. The footprint is dominated by the rate/CSP
# arrays: 5 floats per nucleotide per row, fp32, so 8192 rows of ORF1a (13,215
# nt) is about 2.2 GB before the dedup below claws most of it back. Genes an
# order of magnitude shorter are correspondingly cheaper; the default is sized
# for the worst case rather than the common one.
DEFAULT_ROWS_PER_CHUNK = 8192

# Columns this script requires from the source PCP table. `clade` is what makes
# the era assignment possible at all, and mat-pcp has emitted it on the
# all-clade aggregates since v9.
REQUIRED_PCP_COLUMNS = frozenset({"clade", "parent", "child", "in_train"})

# Carried through to the output when present in the source. `branch_length` is
# read by the training path as the initial BL, and by BranchLengthOptimizer if
# hooks are enabled. `clade` and `era` are always written (see
# `scalar_columns`), because without them a clade-stratified re-score has to
# re-join against the source CSV.
#
# `parent_name`/`child_name` are deliberately NOT here. Measured on the real v9
# E corpus, `child_name` alone was 1.067 MB of a 2.1 MB file -- 51% -- while
# `clade` and `era` cost 4 KB each. It is not a bare accession but a
# `strain|EPI_ISL_xxxxxxx|date` composite averaging 34.7 characters, with 77,977
# distinct values over 79,577 rows, so dictionary encoding is useless (Parquet
# falls back to plain encoding past its 1 MB dictionary-page cap) and it
# compresses only ~2.9x. Carrying strain name *and* collection date also makes
# it the most GISAID-identifying field in the row. `parent_name` is likewise
# high-cardinality (64,117 distinct). Nothing in the training or scoring path
# reads either. Pass `--extra-columns child_name` if an analysis needs them.
OPTIONAL_PCP_COLUMNS = ("branch_length",)


class EraMappingError(ValueError):
    """Raised when the era map and the corpus disagree.

    Distinct from a bare ValueError so a caller can tell "this configuration is
    wrong" apart from an unrelated pandas or IO failure. Every raise site guards
    an invariant whose violation would otherwise yield a well-formed Parquet
    holding rates from the wrong neutral model.
    """


@dataclass(frozen=True)
class Era:
    """One named group of clades and the neutral archive that serves them.

    The archive's SHA-256 goes into the provenance blob (via
    :func:`dasm2.data.local.compute_sha256`) rather than its path alone,
    because a path is repointable and a checkpoint trained against the
    resulting Parquet keeps no other record of which λ produced its rates.
    """

    name: str
    archive: Path
    clades: frozenset[str]


def load_eras(path: Path) -> tuple[list[Era], str]:
    """Read and validate the era map.

    Validates everything decidable without the corpus: the file's shape, that
    each archive exists, that no era is empty, and that no clade is claimed by
    two eras. Coverage in the other direction -- every clade in the corpus is
    named, every era matches rows -- needs the data, and is checked by
    :func:`scan_corpus` before the writer opens.

    Args:
        path: YAML file with a top-level ``eras`` mapping and an optional
            ``neutral_model_name``.

    Returns:
        ``(eras, neutral_model_name)``, eras sorted by name for stable output.

    Raises:
        EraMappingError: On any structural problem, naming the offending era.
    """
    spec = yaml.safe_load(path.read_text())
    if not isinstance(spec, dict) or "eras" not in spec:
        raise EraMappingError(
            f"{path}: expected a mapping with a top-level `eras` key, got "
            f"{type(spec).__name__}."
        )
    raw_eras = spec["eras"]
    if not isinstance(raw_eras, dict) or not raw_eras:
        raise EraMappingError(f"{path}: `eras` must be a non-empty mapping.")

    eras: list[Era] = []
    for name, entry in sorted(raw_eras.items()):
        if not isinstance(entry, dict):
            raise EraMappingError(f"{path}: era {name!r} must be a mapping.")
        missing = {"archive", "clades"} - set(entry)
        if missing:
            raise EraMappingError(f"{path}: era {name!r} is missing {sorted(missing)}.")
        archive = Path(entry["archive"]).expanduser()
        if not archive.is_file():
            raise EraMappingError(
                f"{path}: era {name!r} names archive {archive}, which does not "
                "exist. Build it before preparing the Parquet -- a missing "
                "archive here would otherwise surface as an opaque failure "
                "partway through a multi-hour write."
            )
        clades = entry["clades"]
        if not isinstance(clades, list) or not clades:
            raise EraMappingError(
                f"{path}: era {name!r} has no clades. An era matching nothing "
                "means the map is wrong, not that the era is optional."
            )
        duplicated = {c for c in clades if clades.count(c) > 1}
        if duplicated:
            raise EraMappingError(
                f"{path}: era {name!r} lists {sorted(duplicated)} more than once."
            )
        eras.append(Era(str(name), archive, frozenset(str(c) for c in clades)))

    for i, first in enumerate(eras):
        for second in eras[i + 1 :]:
            if overlap := first.clades & second.clades:
                raise EraMappingError(
                    f"{path}: clades {sorted(overlap)} appear in both "
                    f"{first.name!r} and {second.name!r}. Eras must partition "
                    "the clade set; an overlap makes a row's λ depend on "
                    "iteration order."
                )

    model_name = str(spec.get("neutral_model_name", "per-era-TabularRates"))
    return eras, model_name


def clade_to_era(eras: list[Era]) -> dict[str, str]:
    """Invert the era map into a clade lookup.

    Safe to build flat because :func:`load_eras` has already rejected overlaps.
    """
    return {clade: era.name for era in eras for clade in era.clades}


def resolve_pcp_source(csv_path: Path) -> tuple[Path, dict[str, str]]:
    """Check the source PCP table exists, and describe its provenance.

    One way in. Upstream also took a data-central manifest name and resolved
    the path and md5 from it; this fork resolves nothing, so the caller passes
    the path ``config.yaml`` names and the md5 is checked where it is recorded
    -- by ``scripts/check_corpus.py``, against the corpus entry, rather than
    here against a manifest a reader does not have.

    Args:
        csv_path: Path to a PCP ``.csv`` or ``.csv.gz``.

    Returns:
        ``(path, provenance)`` where provenance goes verbatim into the Parquet
        metadata.

    Raises:
        ValueError: If the CSV does not exist.
    """
    if not csv_path.is_file():
        raise ValueError(f"--pcp-csv {csv_path} does not exist.")
    return csv_path, {"source_pcp_path": str(csv_path)}


def iter_pcp_chunks(
    source: Path, columns: list[str], rows_per_chunk: int, limit: int | None
) -> Iterator[pd.DataFrame]:
    """Stream the PCP table in row chunks, honouring an optional row cap.

    Reads only the requested columns. pandas infers gzip from the suffix, which
    matters because mat-pcp ships its PCP tables gzipped.

    Args:
        source: Path to a ``.csv`` or ``.csv.gz``.
        columns: Columns to read.
        rows_per_chunk: Rows per yielded frame.
        limit: Stop after this many rows in total; None for all.

    Yields:
        Frames with a contiguous 0-based index, so a caller can align computed
        arrays back by position without worrying about the source's index.
    """
    yielded = 0
    reader = pd.read_csv(source, usecols=columns, chunksize=rows_per_chunk)
    for chunk in reader:
        if limit is not None and yielded + len(chunk) > limit:
            chunk = chunk.iloc[: limit - yielded]
        if chunk.empty:
            break
        yield chunk.reset_index(drop=True)
        yielded += len(chunk)
        if limit is not None and yielded >= limit:
            break


def rates_for_sequences(
    model: TabularRatesNeutralModel, sequences: list[str]
) -> tuple[list[list[float]], list[list[float]]]:
    """Compute rates and flattened CSPs for a list of parent sequences.

    Deduplicates first. Viral parents repeat heavily -- the same clade founder
    fathers many PCPs, from 4.8x on ORF1a to 185x on E -- so computing once per
    distinct sequence and fanning the result back out is a large constant-factor
    saving on a job whose whole cost is this call.

    CSPs are flattened from ``(L, 4)`` to ``4L`` because Parquet stores 1-D
    lists; ``ParquetRowConverter`` reshapes on read.

    Args:
        model: The era's neutral model.
        sequences: Parent nucleotide sequences, one per output row.

    Returns:
        ``(rates, csps)``, one entry per input sequence in input order.
    """
    unique = list(dict.fromkeys(sequences))
    with torch.no_grad():
        rates_list, csps_list = model.compute_rates_and_csps(unique)
    cache = {
        seq: (rates.tolist(), csps.reshape(-1).tolist())
        for seq, rates, csps in zip(unique, rates_list, csps_list)
    }
    rates_out: list[list[float]] = []
    csps_out: list[list[float]] = []
    for seq in sequences:
        rates, csps = cache[seq]
        rates_out.append(rates)
        csps_out.append(csps)
    return rates_out, csps_out


@dataclass(frozen=True)
class CorpusScan:
    """What the cheap pre-pass learns before any rate is computed."""

    num_rows: int
    max_aa_seq_len: int
    per_era_rows: dict[str, int]


def scan_corpus(
    source: Path,
    lookup: dict[str, str],
    era_names: list[str],
    rows_per_chunk: int,
    limit: int | None,
) -> CorpusScan:
    """Validate era coverage and collect file metadata, reading two columns.

    This exists so that **no coverage failure can leave an output file behind**.
    Checking coverage during the write instead would mean the zero-matched-era
    case is only detectable at the end, by which point the Parquet is complete
    and valid -- correct schema, real rates, just missing one archive's
    contribution -- so a pipeline gating on file existence rather than exit
    status would consume it. Failing before the writer is opened removes that
    mode entirely.

    It also supplies ``max_aa_seq_len`` and ``num_rows`` up front. Both depend
    only on ``parent`` lengths and the row count, never on the neutral models,
    so computing them here lets the metadata be complete at the first write.
    The alternative -- learning them during the pass and patching the footer
    afterwards -- costs a full read and rewrite of the finished file, which at
    ORF1a's projected 42 GB is not affordable.

    Cost is one streaming read of two columns.

    Args:
        source: PCP CSV (optionally gzipped).
        lookup: clade -> era name, from :func:`clade_to_era`.
        era_names: Every era's name, so unmatched ones can be reported.
        rows_per_chunk: Rows per read chunk.
        limit: Optional cap, matching the write pass so the counts agree.

    Returns:
        The scan result.

    Raises:
        EraMappingError: If any clade is unmapped, or any era matches no rows.
        ValueError: If the corpus is empty, or a parent is not codon-aligned.
    """
    per_era_rows = {name: 0 for name in era_names}
    num_rows = 0
    max_nt_len = 0
    unmapped: set[str] = set()

    for chunk in iter_pcp_chunks(source, ["clade", "parent"], rows_per_chunk, limit):
        clades = chunk["clade"].astype(str)
        eras = clades.map(lookup)
        unmapped.update(clades[eras.isna()].unique())
        for era_name, count in eras.value_counts().items():
            per_era_rows[str(era_name)] += int(count)
        lengths = chunk["parent"].astype(str).str.len()
        max_nt_len = max(max_nt_len, int(lengths.max()))
        num_rows += len(chunk)

    if unmapped:
        raise EraMappingError(
            f"{source}: clade(s) {sorted(unmapped)} are not named by any era. "
            "Add them to the era map -- there is no default era, because a row "
            "silently served the wrong λ trains without error and yields wrong "
            "selection factors."
        )
    if num_rows == 0:
        raise ValueError(f"{source} has no rows to process.")
    if empty := sorted(name for name, count in per_era_rows.items() if count == 0):
        raise EraMappingError(
            f"Era(s) {empty} matched no rows. Their archives were built and "
            "never used, which almost always means a clade name in the era map "
            "is misspelled. Refusing to start a write whose λ scheme would not "
            "be the one it claims."
        )
    if max_nt_len % 3 != 0:
        raise ValueError(
            f"{source}: longest parent is {max_nt_len} nt, not a multiple of 3. "
            "max_aa_seq_len would round down, and tokenization silently "
            "truncates amino-acid sequences longer than it."
        )

    logger.info(
        "Scanned %d rows, longest parent %d nt (%d codons), per era: %s",
        num_rows,
        max_nt_len,
        max_nt_len // 3,
        per_era_rows,
    )
    return CorpusScan(num_rows, max_nt_len // 3, per_era_rows)


def build_arrow_table(
    frame: pd.DataFrame,
    rates: list[list[float]],
    csps: list[list[float]],
) -> pa.Table:
    """Assemble one output chunk with an explicit fp32 schema.

    The explicit schema is the point of this function. ``pa.Table.from_pandas``
    on a column of Python lists infers ``list<double>``, because ``.tolist()``
    on a float32 tensor yields Python floats -- so the file silently stores
    doubles at twice the size. ``compute_rates_and_csps`` returns float32, so declaring fp32 here is
    lossless: the stored values are bit-identical to the streaming path's, not
    approximately equal.

    Args:
        frame: The chunk's non-array columns, in output order.
        rates: Per-row rate lists, aligned to ``frame``.
        csps: Per-row flattened CSP lists, aligned to ``frame``.

    Returns:
        An Arrow table whose column order is ``frame``'s, then ``nt_rates`` and
        ``nt_csps``.
    """
    arrays = [pa.array(frame[column]) for column in frame.columns]
    fields = [
        pa.field(column, array.type) for column, array in zip(frame.columns, arrays)
    ]
    list_f32 = pa.list_(pa.float32())
    arrays.append(pa.array(rates, type=list_f32))
    fields.append(pa.field("nt_rates", list_f32))
    arrays.append(pa.array(csps, type=list_f32))
    fields.append(pa.field("nt_csps", list_f32))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def prepare_parquet(
    source: Path,
    eras: list[Era],
    output: Path,
    *,
    neutral_model_name: str,
    provenance: dict[str, str],
    rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK,
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
    limit: int | None = None,
    extra_columns: tuple[str, ...] = (),
) -> dict[str, int]:
    """Compute per-era rates over the whole corpus and write the Parquet.

    One streaming pass. Within each chunk the rows are grouped by era, each
    group's rates are computed with that era's own model, and the results are
    scattered back to the chunk's original row order before writing -- so the
    output row order matches the input's and no row's λ depends on where it
    happened to fall.

    Args:
        source: PCP CSV (optionally gzipped).
        eras: Validated era list from :func:`load_eras`.
        output: Destination Parquet path.
        neutral_model_name: Recorded in metadata; names the λ scheme.
        provenance: Source description from :func:`resolve_pcp_source`.
        rows_per_chunk: Rows held in memory at once.
        row_group_size: Parquet row group size.
        limit: Optional cap on rows processed, for smoke tests.
        extra_columns: Additional source columns to carry through.

    Returns:
        Rows written per era name.

    Raises:
        EraMappingError: If a clade is unmapped, or an era matches no rows.
        ValueError: If the source is missing a required column.
    """
    header = pd.read_csv(source, nrows=0)
    available = set(header.columns)
    if missing := REQUIRED_PCP_COLUMNS - available:
        raise ValueError(
            f"{source} is missing required column(s) {sorted(missing)}. "
            f"Available: {sorted(available)}."
        )
    carried = [c for c in OPTIONAL_PCP_COLUMNS if c in available]
    if unknown := [c for c in extra_columns if c not in available]:
        raise ValueError(f"{source} has no column(s) {unknown}.")
    carried += [c for c in extra_columns if c not in carried]
    read_columns = sorted(REQUIRED_PCP_COLUMNS | set(carried))

    lookup = clade_to_era(eras)
    models = {
        era.name: TabularRatesNeutralModel.from_pretrained(era.archive) for era in eras
    }
    logger.info("Loaded %d neutral archive(s): %s", len(models), sorted(models))

    # Output column order: the identity columns first, then the arrays. `era` is
    # derived here rather than carried, so the file records the assignment that
    # was actually used rather than one a reader would have to recompute. `era`
    # is in the exclusion set as well as the prefix: a source column literally
    # named `era`, requested via --extra-columns, would otherwise be appended
    # twice and turn `frame["era"]` into a two-column frame inside
    # `build_arrow_table`, failing with an opaque Arrow TypeError.
    scalar_columns = ["parent", "child", "clade", "era", "in_train"] + [
        c for c in carried if c not in {"parent", "child", "clade", "era", "in_train"}
    ]

    scan = scan_corpus(
        source, lookup, [era.name for era in eras], rows_per_chunk, limit
    )
    start = time.time()

    handler = SegmentHandler()
    metadata = {
        "neutral_model_name": neutral_model_name,
        # Derived from the handler rather than hardcoded, so the two cannot
        # drift apart if SegmentHandler's suffixes ever change.
        "segment_handler": type(handler).__name__,
        "segment_suffixes": ",".join(handler.segment_suffixes),
        "num_rows": str(scan.num_rows),
        "max_aa_seq_len": str(scan.max_aa_seq_len),
        "eras": json.dumps(
            [
                {
                    "name": era.name,
                    "archive": str(era.archive),
                    "sha256": compute_sha256(era.archive),
                    "clades": sorted(era.clades),
                }
                for era in eras
            ],
            sort_keys=True,
        ),
        # Row-preprocessing seam stamp. This writer reads the
        # PCP CSV directly and never reaches
        # ``registry.preprocess_rows_for_split``; viral corpora have no
        # manifest nickname carrying a naughty list either, so the identity is
        # the vacuous one and ``source_dataset`` is empty. The stamp therefore
        # records that this writer asserts seam compatibility, not that these
        # rows passed through the seam. Without it every load of a viral
        # Parquet would warn, on a path where the warning is unactionable.
        "source_dataset": "",
        **seam_identity(None),
        # ``--limit`` head-truncates the corpus, so it records the key the
        # load-time check rejects a prefix on.
        # Recorded whenever the flag is set, without checking whether it
        # actually bit: ``--limit`` is the smoke-test switch, and a smoke-test
        # corpus must never drive a CV fold map even if it happened to cover
        # every row.
        **({"max_rows": str(limit)} if limit is not None else {}),
        **provenance,
    }

    with ChunkedParquetWriter(
        output, metadata=metadata, row_group_size=row_group_size
    ) as writer:
        for chunk in iter_pcp_chunks(source, read_columns, rows_per_chunk, limit):
            chunk["era"] = chunk["clade"].astype(str).map(lookup)
            # `scan_corpus` already proved every clade maps, so reaching this is
            # only possible if the source changed under us mid-run. Cheap enough
            # to keep as a second line, and raising inside the writer's context
            # means the partial file is deleted rather than left behind.
            if chunk["era"].isna().any():
                offenders = sorted(
                    chunk.loc[chunk["era"].isna(), "clade"].astype(str).unique()
                )
                raise EraMappingError(
                    f"{source}: clade(s) {offenders} are not named by any era, "
                    "but the pre-pass over this same file saw none. Did the "
                    "source change while the job was running?"
                )

            rates_col: list[list[float]] = [[]] * len(chunk)
            csps_col: list[list[float]] = [[]] * len(chunk)
            for era_name, group in chunk.groupby("era", sort=True):
                sequences = group["parent"].astype(str).tolist()
                rates, csps = rates_for_sequences(models[str(era_name)], sequences)
                for position, rate, csp in zip(group.index, rates, csps):
                    rates_col[position] = rate
                    csps_col[position] = csp

            writer.write_chunk(
                build_arrow_table(chunk[scalar_columns], rates_col, csps_col)
            )
            logger.info(
                "  %d / %d rows written (%.1f min)",
                writer.rows_written,
                scan.num_rows,
                (time.time() - start) / 60,
            )

        total = writer.rows_written

    if total != scan.num_rows:
        raise ValueError(
            f"Wrote {total} rows but the pre-pass counted {scan.num_rows}. "
            "The two passes disagree about the corpus; do not use this file."
        )

    logger.info(
        "Wrote %s (%.1f MB, %d rows, %.1f min): %s",
        output,
        output.stat().st_size / 1e6,
        total,
        (time.time() - start) / 60,
        scan.per_era_rows,
    )
    return scan.per_era_rows


def validate_output(output: Path, num_check: int = 32) -> None:
    """Read the finished file back the way training will, and check its shape.

    A multi-hour write deserves a cheap end-to-end check that the artifact is
    actually loadable, rather than discovering at the start of a training run
    that a metadata key is missing or an array column has the wrong length.
    This deliberately re-reads through the real ``ParquetBackend`` and the real
    ``make_parquet_row_converter``, so it exercises the consumer rather than
    re-asserting the writer's own view.

    Narrow on purpose: the per-row properties a sampling check would look at
    -- CSP rows summing to one, positive rates at every site -- are enforced by
    ``TabularRatesNeutralModel.compute_rates_and_csps``, which raises on a zero
    site rate and normalizes the CSPs itself. What is *not* guaranteed by
    construction, and so is checked here, is that the file round-trips through
    the loader with rate and CSP lengths agreeing with each row's parent.

    Args:
        output: The written Parquet.
        num_check: How many rows to convert, evenly spaced across the file.

    Raises:
        ValueError: If metadata is missing or a row's arrays are misshapen.
    """
    backend = ParquetBackend([output])
    missing = {"neutral_model_name", "eras", "max_aa_seq_len"} - set(backend.metadata)
    if missing:
        raise ValueError(f"{output}: metadata is missing {sorted(missing)}.")

    converter = make_parquet_row_converter(
        SegmentHandler(), available_columns=set(backend.column_names)
    )
    total = backend.total_rows
    step = max(1, total // max(num_check, 1))
    for index in range(0, total, step):
        row = backend.get_row(index)
        sample = converter(row)
        expected = len(row["parent"])
        if sample["nt_rates"].shape[0] != expected:
            raise ValueError(
                f"{output}: row {index} has {sample['nt_rates'].shape[0]} rates "
                f"for a {expected} nt parent."
            )
        if tuple(sample["nt_csps"].shape) != (expected, 4):
            raise ValueError(
                f"{output}: row {index} has CSPs of shape "
                f"{tuple(sample['nt_csps'].shape)}, expected ({expected}, 4)."
            )
    logger.info(
        "Validated %s: %d rows sampled, loader round-trip clean", output, num_check
    )


def build_argparser() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Write a viral training Parquet whose neutral rates come from a "
            "per-era set of TabularRates archives."
        )
    )
    parser.add_argument(
        "--pcp-csv",
        type=Path,
        required=True,
        help="Path to the mat-pcp PCP table, as config.yaml's source_pcp names it",
    )
    parser.add_argument(
        "--eras", type=Path, required=True, help="YAML era -> {archive, clades} map"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-chunk", type=int, default=DEFAULT_ROWS_PER_CHUNK)
    parser.add_argument("--row-group-size", type=int, default=DEFAULT_ROW_GROUP_SIZE)
    parser.add_argument(
        "--limit", type=int, default=None, help="Process at most this many rows"
    )
    parser.add_argument(
        "--extra-columns",
        nargs="*",
        default=[],
        help="Additional source columns to carry into the Parquet",
    )
    parser.add_argument(
        "--summary", type=Path, default=None, help="Write per-era row counts as JSON"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the post-write loader round-trip check",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing output file"
    )
    return parser


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_argparser().parse_args()

    if args.output.exists() and not args.force:
        print(f"{args.output} exists; pass --force to overwrite.", file=sys.stderr)
        raise SystemExit(1)

    eras, model_name = load_eras(args.eras)
    source, provenance = resolve_pcp_source(args.pcp_csv)
    provenance["era_map"] = str(args.eras)

    counts = prepare_parquet(
        source,
        eras,
        args.output,
        neutral_model_name=model_name,
        provenance=provenance,
        rows_per_chunk=args.rows_per_chunk,
        row_group_size=args.row_group_size,
        limit=args.limit,
        extra_columns=tuple(args.extra_columns),
    )
    if not args.no_validate:
        validate_output(args.output)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
