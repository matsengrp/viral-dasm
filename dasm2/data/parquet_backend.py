"""Parquet-backed storage and shuffle-buffer sampler for streaming data loading.

Provides two components for from-disk streaming of precomputed training data:

1. **ParquetBackend**: Row-group-cached random access to one or more Parquet
   files.  Maintains an LRU cache of decompressed row groups so that only a
   bounded working set is held in memory.  By default the cache is sized in
   row groups from the ``ShuffleBufferSampler`` buffer size and row-group
   geometry; an explicit budget bounds it in decoded bytes instead (see
   ``plan_cache``).

2. **ShuffleBufferSampler**: A ``torch.utils.data.Sampler`` that shuffles
   row-group order, streams row groups into a buffer, and yields random
   indices from the buffer. Produces the same I/O pattern as a true
   shuffle-buffer ``IterableDataset`` while working with map-style Datasets.

Usage::

    backend = ParquetBackend(
        [Path("train_part1.parquet"), Path("train_part2.parquet")],
        buffer_size=8192,
    )
    sampler = ShuffleBufferSampler(backend.row_group_boundaries, buffer_size=8192)

    dataset = TransformDataset(parquet_backend=backend, ...)
    loader = DataLoader(dataset, sampler=sampler, batch_size=256)
"""

import bisect
import hashlib
import logging
import os
import random
import statistics
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Sampler

from dasm2.data.transforms import (
    NT_SEPARATOR_LEN,
    SEPARATOR_NT_CSP_FILL,
    SEPARATOR_NT_RATE_FILL,
)
from dasm2.sequence.constants import NT_SEPARATOR
from dasm2.utils.device import mem_total_bytes

if TYPE_CHECKING:
    from dasm2.data.segment_handlers import SegmentHandler

logger = logging.getLogger(__name__)

#: The file-level key under which pyarrow stores the serialized Arrow schema.
_ARROW_SCHEMA_KEY = b"ARROW:schema"


def read_parquet_file_metadata(source: "pq.ParquetFile | Path") -> dict[str, str]:
    """A Parquet file's key-value metadata, decoded, from the file footer.

    Reads the footer's key-value metadata, not the Arrow schema's. The footer
    holds every key the schema carries plus any a writer added at close with
    ``ParquetWriter.add_key_value_metadata``, which the schema never sees
    (``pf.schema_arrow.metadata`` and ``pq.read_schema`` both miss them). The
    streaming converter (:mod:`dasm2.data.pcp_streaming`) stamps its
    whole-corpus counts that way, so every metadata reader goes through here.
    pyarrow's own serialized schema is left out.

    Args:
        source: An open ParquetFile, or a path to open.

    Returns:
        Dict of metadata key-value pairs.
    """
    pf = source if isinstance(source, pq.ParquetFile) else pq.ParquetFile(source)
    return {
        k.decode(): v.decode()
        for k, v in (pf.metadata.metadata or {}).items()
        if k != _ARROW_SCHEMA_KEY
    }


#: The default cache bound, in parquet ``total_byte_size`` bytes: as many row
#: groups as fit 2 GiB by that measure, capped at the working set.
#:
#: A bound exists at all because the cache is per-worker: each forked
#: DataLoader worker builds its own, so an uncapped cache multiplies by the
#: worker count and can exhaust host RAM. Bounding row groups rather than
#: sizing from host RAM is deliberate -- a row group's decoded size varies
#: with the corpus, so a RAM-derived bound is not more predictable, only
#: harder to reason about. Override per run with
#: ``DASM2_PARQUET_CACHE_BUDGET_GB``.
_CACHE_FLOOR_PARQUET_BYTES = 2 * 1024**3

#: Environment override for the cache budget in GB (a positive float). Takes
#: precedence over the ``parquet_cache_budget_gb`` config field.
CACHE_BUDGET_ENV_VAR = "DASM2_PARQUET_CACHE_BUDGET_GB"

#: Multiplier applied to ``buffer_size // median_rg_rows`` to estimate the
#: working set of a randomly-drained shuffle buffer. Random eviction leaves
#: stragglers from many already-streamed row groups live in the buffer, so the
#: distinct-row-group working set runs well above the naive buffer/row-group
#: ratio — empirically ~10x for buffer 8192 over ~1531-row groups. 16x is a
#: thrash-safe floor the RAM cap is never allowed to undercut.
_STRAGGLER_WORKING_SET_MARGIN = 16

#: Row-group cache size for a *sequential* sweep (evaluation, oe-plot). A
#: forward-only walk needs the group it is in plus the one streaming in; the
#: slack covers DataLoader workers, which take whole batches round-robin and so
#: step forward in strides rather than one group at a time. Sized explicitly
#: rather than derived from ``buffer_size``, because the buffer is what makes
#: the working set large and a sequential sampler never fills it.
SEQUENTIAL_SWEEP_CACHE_SIZE = 8


@dataclass(frozen=True)
class CacheBudgetSpec:
    """Inputs for one backend's row-group cache budget.

    Attributes:
        budget_gb: The ``parquet_cache_budget_gb`` config value; None keeps
            the default bound.
        num_dataloader_workers: Workers per DataLoader, assumed the same for
            the train and the val loader (the trainer uses one setting for
            both). Each worker fills its own copy of its loader's cache; used
            to warn when an explicit budget's copies would crowd host RAM.
    """

    budget_gb: Optional[float] = None
    num_dataloader_workers: int = 0


def _positive_gb_to_bytes(gb: float, what: str) -> int:
    """Bytes for an explicit GB budget, which must be positive."""
    if not gb > 0:
        raise ValueError(f"{what} must be a positive number of GB, got {gb!r}")
    return int(gb * 1e9)


def _explicit_budget(spec: CacheBudgetSpec) -> Optional[tuple[int, str]]:
    """The env var's budget, else the config's, else None; in bytes."""
    raw = os.environ.get(CACHE_BUDGET_ENV_VAR)
    if raw is not None:
        try:
            gb = float(raw)
        except ValueError:
            raise ValueError(
                f"{CACHE_BUDGET_ENV_VAR} must be a positive number of GB, got {raw!r}"
            ) from None
        return _positive_gb_to_bytes(gb, CACHE_BUDGET_ENV_VAR), "env"
    if spec.budget_gb is not None:
        return _positive_gb_to_bytes(spec.budget_gb, "parquet_cache_budget_gb"), (
            "config"
        )
    return None


def default_cache_row_groups(rg_parquet_bytes: Sequence[int]) -> int:
    """Row groups the default cache bound allows: 2 GiB of ``total_byte_size``.

    The older rule, kept as the default: as many median-size row groups as
    fit ``_CACHE_FLOOR_PARQUET_BYTES`` of parquet ``total_byte_size``, and at
    most the number of row groups there are. The median keeps one short final
    row group from inflating the count.

    Args:
        rg_parquet_bytes: Parquet ``total_byte_size`` per row group.
    """
    median_parquet = max(1, int(statistics.median(rg_parquet_bytes or [1])))
    return min(
        max(1, len(rg_parquet_bytes)),
        max(1, _CACHE_FLOOR_PARQUET_BYTES // median_parquet),
    )


def typical_decoded_row_group_bytes(
    files: Sequence[pq.ParquetFile], rg_parquet_bytes: Sequence[int]
) -> int:
    """Decoded (arrow ``nbytes``) size of the median-size row group.

    Converts a row-group count into the decoded bytes a cache budget is
    measured in; see ``ParquetBackend._get_row`` for why arrow ``nbytes`` is
    the measure. Picked by parquet ``total_byte_size``, not row count: on the
    v10 corpus nearly every row group has 8192 rows but sizes range 37-230 MB,
    while decoded / parquet bytes stays within 5.9-7.5. Decodes
    that one row group.

    Args:
        files: The Parquet files, in order.
        rg_parquet_bytes: Parquet ``total_byte_size`` per row group, in file
            then row-group order.
    """
    locations = [
        (f, rg)
        for f, pf in enumerate(files)
        for rg in range(pf.metadata.num_row_groups)
    ]
    median_bytes = statistics.median(rg_parquet_bytes)
    f, rg = locations[
        min(
            range(len(locations)),
            key=lambda k: abs(rg_parquet_bytes[k] - median_bytes),
        )
    ]
    return int(files[f].read_row_group(rg).nbytes)


#: Share of ``MemTotal`` past which an explicit budget's copies earn a warning.
_EXPLICIT_BUDGET_WARN_FRACTION = 0.25


@dataclass(frozen=True)
class CachePlan:
    """Bounds for one backend's row-group LRU cache.

    Attributes:
        max_entries: Row-group count bound.
        max_bytes: Decoded-byte bound for an explicit budget; None bounds the
            default cache by count only.
        source: ``"env"``, ``"config"`` or ``"default"``.
    """

    max_entries: int
    max_bytes: Optional[int]
    source: str


def plan_cache(
    spec: CacheBudgetSpec,
    rg_rows: list[int],
    rg_parquet_bytes: list[int],
    buffer_size: int,
    rg_decoded_bytes: int,
) -> CachePlan:
    """Bound one backend's row-group cache, and log the bound.

    By default the cache holds ``min(working set, floor)`` row groups: the
    working set is ``_working_set_row_groups``, the floor is as many row groups
    as fit ``_CACHE_FLOOR_PARQUET_BYTES`` of parquet ``total_byte_size`` (the
    older rule). An explicit budget (the ``DASM2_PARQUET_CACHE_BUDGET_GB``
    env var, then ``spec.budget_gb``) bounds it instead in decoded bytes,
    measured per row group at insert, up to the working set's count. It is not
    clamped, but a warning names the host-RAM cost when the main process and
    train workers holding the budget, plus the val workers' sequential caches,
    would exceed ``_EXPLICIT_BUDGET_WARN_FRACTION`` of ``MemTotal``: a
    generous budget times a worker count is how a shared host gets filled.

    Args:
        spec: Config budget and worker count.
        rg_rows: Rows per row group (after any row filter).
        rg_parquet_bytes: Parquet ``total_byte_size`` per row group.
        buffer_size: The shuffle buffer size.
        rg_decoded_bytes: Measured decoded (arrow ``nbytes``) size of one
            typical row group, to report the bounds in bytes.

    Returns:
        The cache bounds.

    Raises:
        ValueError: If the env var or ``spec.budget_gb`` is not a positive
            number.
    """
    floor_rgs = default_cache_row_groups(rg_parquet_bytes)
    ws_rgs = _working_set_row_groups(rg_rows, buffer_size)
    per_rg = max(1, rg_decoded_bytes)

    explicit = _explicit_budget(spec)
    if explicit is None:
        plan = CachePlan(
            max_entries=min(ws_rgs, floor_rgs), max_bytes=None, source="default"
        )
        budget_bytes = plan.max_entries * per_rg
    else:
        budget_bytes, source = explicit
        plan = CachePlan(max_entries=ws_rgs, max_bytes=budget_bytes, source=source)
        _warn_if_explicit_budget_crowds_ram(spec, budget_bytes, per_rg)
    logger.info(
        "Parquet row-group cache: budget %.1f GB = %d row groups (source=%s); "
        "working set %d row groups = %.1f GB; floor %d row groups = %.1f GB; "
        "decoded row group %.2f GB",
        budget_bytes / 1e9,
        budget_bytes // per_rg,
        plan.source,
        ws_rgs,
        ws_rgs * per_rg / 1e9,
        floor_rgs,
        floor_rgs * per_rg / 1e9,
        per_rg / 1e9,
    )
    return plan


def _warn_if_explicit_budget_crowds_ram(
    spec: CacheBudgetSpec, budget_bytes: int, rg_decoded_bytes: int
) -> None:
    """Warn when an explicit budget's cache copies exceed a share of MemTotal.

    The main process and each train worker can fill ``budget_bytes``; each val
    worker holds a ``SEQUENTIAL_SWEEP_CACHE_SIZE``-row-group sweep cache.
    """
    total = mem_total_bytes()
    if total is None:
        return
    workers = spec.num_dataloader_workers
    projected = (1 + workers) * budget_bytes + (
        workers * SEQUENTIAL_SWEEP_CACHE_SIZE * rg_decoded_bytes
    )
    limit = _EXPLICIT_BUDGET_WARN_FRACTION * total
    if projected > limit:
        logger.warning(
            "Parquet cache budget %.1f GB x (main + %d train workers) + %d val "
            "workers' sweep caches = %.0f GB of decoded row groups, over %.0f%% "
            "of the host's %.0f GB MemTotal (%.0f GB). The budget is not clamped; "
            "lower it or num_dataloader_workers on a shared host.",
            budget_bytes / 1e9,
            workers,
            workers,
            projected / 1e9,
            100 * _EXPLICIT_BUDGET_WARN_FRACTION,
            total / 1e9,
            limit / 1e9,
        )


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------


_MISSING: Any = object()
"""Sentinel for LRU cache misses (distinct from any real cached value)."""


class _LRUCache:
    """LRU cache bounded by entry count and, optionally, by total bytes.

    Each entry's size is given at insert (measured once by the caller), so the
    byte bound tracks real memory rather than an estimate made before decoding.

    Args:
        maxsize: Maximum number of entries.
        max_bytes: Maximum total of the entries' sizes; None bounds by count
            only. The newest entry is always kept, even if alone it exceeds
            ``max_bytes``.
    """

    def __init__(self, maxsize: int, max_bytes: Optional[int] = None) -> None:
        self._maxsize = maxsize
        self._max_bytes = max_bytes
        self._data: OrderedDict[tuple[int, int], tuple[Any, int]] = OrderedDict()
        self._nbytes = 0
        self._warned_byte_bound = False

    @property
    def nbytes(self) -> int:
        """Total size of the cached entries, as measured at insert."""
        return self._nbytes

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: tuple[int, int]) -> Any:
        """Return cached value or ``_MISSING`` sentinel if not present."""
        entry = self._data.get(key, _MISSING)
        if entry is _MISSING:
            return _MISSING
        self._data.move_to_end(key)
        return entry[0]

    def put(self, key: tuple[int, int], value: Any, nbytes: int = 0) -> None:
        """Insert value, evicting oldest entries past the count or byte bound."""
        if key in self._data:
            self._nbytes -= self._data.pop(key)[1]
        while self._data and len(self._data) >= self._maxsize:
            self._nbytes -= self._data.popitem(last=False)[1][1]
        while (
            self._data
            and self._max_bytes is not None
            and self._nbytes + nbytes > self._max_bytes
        ):
            self._warn_byte_bound_once()
            self._nbytes -= self._data.popitem(last=False)[1][1]
        self._data[key] = (value, nbytes)
        self._nbytes += nbytes

    def _warn_byte_bound_once(self) -> None:
        if self._warned_byte_bound:
            return
        self._warned_byte_bound = True
        logger.warning(
            "Parquet row-group cache reached its %.1f GB RAM budget holding %d "
            "row groups, fewer than its %d-row-group working set; evicting by "
            "size from here on. Data loading may thrash; raise the budget if "
            "host RAM allows.",
            (self._max_bytes or 0) / 1e9,
            len(self._data),
            self._maxsize,
        )


# ---------------------------------------------------------------------------
# ParquetBackend
# ---------------------------------------------------------------------------


def _working_set_row_groups(rg_sizes: list[int], buffer_size: int) -> int:
    """Distinct row groups a randomly drained shuffle buffer keeps live.

    The cache's entry-count bound. Random eviction leaves stragglers from many
    already-streamed row groups live, so the working set runs well above
    buffer/median rows. Uses the MEDIAN row-group size, not the minimum: a
    single 16-row group among 1531-row groups would otherwise push the bound
    to ~every row group. Times a straggler margin.
    """
    if not rg_sizes:
        return 1
    median_rows = max(1, int(statistics.median(rg_sizes)))
    return min(
        len(rg_sizes),
        (buffer_size // median_rows + 2) * _STRAGGLER_WORKING_SET_MARGIN,
    )


class ParquetBackend:
    """Row-group-cached random access to one or more Parquet files.

    Maintains an LRU cache of decompressed row groups (as pandas DataFrames)
    so that only a bounded working set is held in memory at any time.  For
    datasets too large to fit in RAM this is essential.

    The cache is sized from ``buffer_size`` toward the row groups whose indices
    can coexist in the ``ShuffleBufferSampler`` buffer, within the bound
    ``plan_cache`` sets (a row-group count by default, decoded bytes for an
    explicit budget). An undersized cache causes eviction/re-read thrashing
    (measured at 50-70x slowdown).

    Args:
        paths: Parquet file paths.
        buffer_size: The ``ShuffleBufferSampler`` buffer size, used to size
            the LRU cache.  Ignored when ``cache_size`` is provided explicitly.
        cache_size: Explicit LRU cache size override (number of row groups
            to keep decompressed).  When given, ``buffer_size`` is not used
            for cache sizing.  Mainly useful for tests.
        row_filter: Optional boolean list selecting which global rows to
            include.  When provided, ``total_rows``, ``get_row``,
            ``read_column``, and ``row_group_boundaries`` operate over
            the filtered subset.  Length must equal the unfiltered total.
        cache_budget: How to size the cache's RAM budget when ``cache_size``
            is not given; see ``plan_cache``.
    """

    def __init__(
        self,
        paths: list[Path],
        buffer_size: int = 8192,
        *,
        cache_size: Optional[int] = None,
        row_filter: Optional[list[bool]] = None,
        cache_budget: CacheBudgetSpec = CacheBudgetSpec(),
    ) -> None:
        if not paths:
            raise ValueError("ParquetBackend requires at least one Parquet file path")

        # Store paths for lazy re-opening after pickling (multiprocessing)
        self._paths = list(paths)

        self._files: Optional[list[pq.ParquetFile]] = []
        self._file_num_rows: list[int] = []
        self._file_cumulative: list[int] = []  # cumulative row count per file

        # Per-file row group metadata: list of (num_rows_in_rg,) per file
        self._file_rg_sizes: list[list[int]] = []
        # Per-file cumulative row offsets within each file for row-group lookup
        self._file_rg_cumulative: list[list[int]] = []
        # Parquet total_byte_size per row group, for the cache floor.
        rg_parquet_bytes: list[int] = []

        cumulative = 0
        for path in paths:
            pf = pq.ParquetFile(path)
            self._files.append(pf)
            num_rgs = pf.metadata.num_row_groups
            file_rows = 0
            rg_sizes: list[int] = []
            rg_cumulative: list[int] = [0]
            for rg_idx in range(num_rgs):
                rg_meta = pf.metadata.row_group(rg_idx)
                rg_num_rows = rg_meta.num_rows
                rg_sizes.append(rg_num_rows)
                rg_parquet_bytes.append(rg_meta.total_byte_size)
                file_rows += rg_num_rows
                rg_cumulative.append(file_rows)

            self._file_rg_sizes.append(rg_sizes)
            self._file_rg_cumulative.append(rg_cumulative)
            self._file_num_rows.append(file_rows)
            self._file_cumulative.append(cumulative)
            cumulative += file_rows

        self._unfiltered_total = cumulative

        # Row filter: maps filtered indices → global indices
        if row_filter is not None:
            if len(row_filter) != self._unfiltered_total:
                raise ValueError(
                    f"row_filter length ({len(row_filter)}) != "
                    f"unfiltered total rows ({self._unfiltered_total})"
                )
            self._index_map: Optional[list[int]] = [
                i for i, keep in enumerate(row_filter) if keep
            ]
            self._total_rows = len(self._index_map)
            self._row_filter_digest: Optional[str] = hashlib.sha256(
                np.asarray(self._index_map, dtype=np.int64).tobytes()
            ).hexdigest()[:8]
        else:
            self._index_map = None
            self._total_rows = cumulative
            self._row_filter_digest = None

        # Compute effective RG sizes for cache sizing (filtered if applicable)
        if self._index_map is not None:
            all_rg_sizes = self._filtered_rg_sizes()
        else:
            all_rg_sizes = [s for sizes in self._file_rg_sizes for s in sizes]
        # An explicit cache_size (sequential sweeps, tests) bounds by count
        # only; otherwise plan_cache sets the bounds.
        cache_max_bytes: Optional[int] = None
        if cache_size is None:
            plan = plan_cache(
                cache_budget,
                all_rg_sizes,
                rg_parquet_bytes,
                buffer_size,
                typical_decoded_row_group_bytes(
                    self._ensure_files_open(), rg_parquet_bytes
                ),
            )
            cache_size = plan.max_entries
            cache_max_bytes = plan.max_bytes
        self._cache_size = cache_size
        self._cache_max_bytes = cache_max_bytes
        self._cache = _LRUCache(maxsize=cache_size, max_bytes=cache_max_bytes)

        # Cache column names and file metadata (plain Python, pickle-safe)
        self._file_column_names: list[list[str]] = [
            list(pf.schema_arrow.names) for pf in self._files
        ]
        self._column_names: list[str] = self._file_column_names[0]
        self._file_metadata = [read_parquet_file_metadata(pf) for pf in self._files]
        self._metadata = {
            key: val for meta in self._file_metadata for key, val in meta.items()
        }

        # Precompute row group boundaries (derived from immutable state)
        self._row_group_boundaries = self._compute_row_group_boundaries()

        logger.info(
            "ParquetBackend: %d files, %d total rows (%d unfiltered), "
            "%d row groups, cache_size=%d",
            len(paths),
            self._total_rows,
            self._unfiltered_total,
            len(self._row_group_boundaries),
            cache_size,
        )

    def _ensure_files_open(self) -> list[pq.ParquetFile]:
        """Return open ParquetFile handles, reopening if needed after unpickling."""
        if self._files is None:
            self._files = [pq.ParquetFile(p) for p in self._paths]
        return self._files

    def __getstate__(self) -> dict:
        """Exclude unpicklable ParquetFile handles and LRU cache."""
        state = self.__dict__.copy()
        state["_files"] = None
        state["_cache"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state and create a fresh LRU cache."""
        self.__dict__.update(state)
        self._cache = _LRUCache(
            maxsize=self._cache_size, max_bytes=self._cache_max_bytes
        )

    def sequential_sweep_copy(self) -> "ParquetBackend":
        """Return a copy of this backend with an empty sequential-sweep cache.

        For a forward-only walk split across forked DataLoader workers (the
        post-cycle BL pass). Each worker fills its own copy of
        the cache, so a copy of this backend would let every worker grow to
        the shuffle-sized bound set at construction. The copy's cache holds
        ``SEQUENTIAL_SWEEP_CACHE_SIZE`` row groups instead, and it reopens its
        file handles lazily in whichever process first reads. If the parent
        reads through the copy before forking (the ESM2 BL path's
        ``aa_parents`` preload does), the workers inherit those open handles;
        pyarrow reads local files by offset, so sharing them is safe, as the
        training loader's forked workers already do with the bundle's backend.
        """
        state = self.__getstate__()
        state["_cache_size"] = SEQUENTIAL_SWEEP_CACHE_SIZE
        state["_cache_max_bytes"] = None
        clone = object.__new__(type(self))
        clone.__setstate__(state)
        return clone

    @property
    def paths(self) -> list[Path]:
        """The backend's Parquet file paths, in construction order."""
        return self._paths

    @property
    def total_rows(self) -> int:
        """Total number of rows (filtered if row_filter was provided)."""
        return self._total_rows

    @property
    def row_filter_digest(self) -> Optional[str]:
        """8-hex-char SHA-256 digest of the kept row indices, or ``None``.

        ``None`` only when no ``row_filter`` was given; an all-True filter
        still gets a digest. The artifact cache keys on it, so datasets built
        from the same files with different filters don't share cached
        branch lengths or predictions. It covers the indices
        only; which files they index is the nickname's job.
        """
        return self._row_filter_digest

    @property
    def metadata(self) -> dict[str, str]:
        """Each file's key-value metadata, merged (later files win)."""
        return self._metadata

    @property
    def file_metadata(self) -> list[dict[str, str]]:
        """Each file's footer key-value metadata, in ``paths`` order (unmerged)."""
        return self._file_metadata

    @property
    def column_names(self) -> list[str]:
        """Column names from the first file's Parquet schema."""
        return self._column_names

    @property
    def file_column_names(self) -> list[list[str]]:
        """Each file's schema column names, in ``paths`` order."""
        return self._file_column_names

    def _resolve_index(self, global_idx: int) -> tuple[int, int]:
        """Map physical global index to (file_idx, local_idx).

        Args:
            global_idx: Physical row index across all files (unfiltered).

        Returns:
            Tuple of (file_index, local_row_index_within_file).

        Raises:
            IndexError: If global_idx is out of range.
        """
        if global_idx < 0 or global_idx >= self._unfiltered_total:
            raise IndexError(
                f"Global index {global_idx} out of range [0, {self._unfiltered_total})"
            )
        # bisect_right on cumulative offsets gives the file index
        file_idx = bisect.bisect_right(self._file_cumulative, global_idx) - 1
        local_idx = global_idx - self._file_cumulative[file_idx]
        return file_idx, local_idx

    def _resolve_row_group(self, file_idx: int, local_idx: int) -> tuple[int, int]:
        """Map local index within a file to (row_group_idx, row_within_rg).

        Args:
            file_idx: Index of the Parquet file.
            local_idx: Row index within that file.

        Returns:
            Tuple of (row_group_index, row_index_within_row_group).
        """
        rg_cum = self._file_rg_cumulative[file_idx]
        rg_idx = bisect.bisect_right(rg_cum, local_idx) - 1
        row_within_rg = local_idx - rg_cum[rg_idx]
        return rg_idx, row_within_rg

    def get_row(self, global_idx: int) -> dict[str, Any]:
        """Read a single row by global index.

        Uses the row-group LRU cache to amortize decompression cost.
        Row groups are cached as pandas DataFrames for fast per-row access
        (much faster than per-cell ``.as_py()`` on Arrow tables, especially
        for list columns like ``nt_rates`` and ``nt_csps``).

        When a ``row_filter`` is active, ``global_idx`` is in the filtered
        index space and is mapped to the physical row before lookup.

        Args:
            global_idx: Global row index (filtered space if row_filter active).

        Returns:
            Dict mapping column names to Python values. Array columns are
            returned as numpy arrays (``torch.tensor()`` accepts both).
        """
        if self._index_map is not None:
            if global_idx < 0 or global_idx >= self._total_rows:
                raise IndexError(
                    f"Filtered index {global_idx} out of range [0, {self._total_rows})"
                )
            physical_idx = self._index_map[global_idx]
        else:
            physical_idx = global_idx
        file_idx, local_idx = self._resolve_index(physical_idx)
        rg_idx, row_within_rg = self._resolve_row_group(file_idx, local_idx)

        cache_key = (file_idx, rg_idx)
        rg_df = self._cache.get(cache_key)
        if rg_df is _MISSING:
            rg_table = self._ensure_files_open()[file_idx].read_row_group(rg_idx)
            rg_df = rg_table.to_pandas()
            # The DataFrame's list cells are views into the arrow buffers, so
            # the table's nbytes is the decoded footprint: within ~12% of the
            # RSS delta on the v10 corpus, where memory_usage(deep=True) reads
            # 22x low and total_byte_size 7.6x low.
            self._cache.put(cache_key, rg_df, rg_table.nbytes)

        # Fast row access via pandas iloc
        row_series = rg_df.iloc[row_within_rg]
        result: dict[str, Any] = row_series.to_dict()  # type: ignore[assignment]
        return result

    def _iter_filtered_rg_counts(self) -> Iterator[int]:
        """Yield per-RG filtered row counts, skipping empty row groups.

        Only valid when ``_index_map`` is set.  Iterates the filtered index
        map once (O(filtered_count)) rather than scanning every physical row
        in every row group.
        """
        assert self._index_map is not None
        # Build cumulative row-group boundaries as (file_idx, rg_global_start)
        # pairs, then count filtered indices per row group via bisect.
        rg_boundaries: list[int] = []
        for file_idx in range(len(self._paths)):
            file_offset = self._file_cumulative[file_idx]
            rg_cum = self._file_rg_cumulative[file_idx]
            for rg_idx in range(len(self._file_rg_sizes[file_idx])):
                rg_boundaries.append(file_offset + rg_cum[rg_idx])
        # Sentinel after the last row group
        rg_boundaries.append(self._unfiltered_total)

        # Count filtered indices per row group
        counts = [0] * (len(rg_boundaries) - 1)
        for phys_idx in self._index_map:
            rg = bisect.bisect_right(rg_boundaries, phys_idx) - 1
            counts[rg] += 1

        for c in counts:
            if c > 0:
                yield c

    def _filtered_rg_sizes(self) -> list[int]:
        """Compute per-RG filtered row counts for cache sizing.

        Only called when ``_index_map`` is set.
        """
        return list(self._iter_filtered_rg_counts())

    def _compute_row_group_boundaries(self) -> list[tuple[int, int]]:
        """Compute (start, end) pairs for all row groups in filtered space.

        When no filter is active, these are global physical boundaries.
        When a filter is active, boundaries are in the filtered index space
        and empty row groups are skipped.
        """
        if self._index_map is None:
            # Unfiltered: physical boundaries
            boundaries: list[tuple[int, int]] = []
            for file_idx in range(len(self._paths)):
                file_offset = self._file_cumulative[file_idx]
                rg_cum = self._file_rg_cumulative[file_idx]
                for rg_idx in range(len(self._file_rg_sizes[file_idx])):
                    start = file_offset + rg_cum[rg_idx]
                    end = file_offset + rg_cum[rg_idx + 1]
                    boundaries.append((start, end))
            return boundaries

        # Filtered: map physical RG boundaries to filtered index space.
        # _index_map preserves global order, so filtered indices from the
        # same physical RG are contiguous.
        boundaries = []
        filtered_offset = 0
        for count in self._iter_filtered_rg_counts():
            boundaries.append((filtered_offset, filtered_offset + count))
            filtered_offset += count
        return boundaries

    @property
    def row_group_boundaries(self) -> list[tuple[int, int]]:
        """Global (start, end) pairs for all row groups across all files.

        Used by ``ShuffleBufferSampler`` to define the chunked I/O pattern.
        """
        return self._row_group_boundaries

    def read_column(self, column_name: str, default: Any = None) -> list[Any]:
        """Read a single column from all files via PyArrow (no row-group cache).

        Efficiently reads one column across all files without loading full
        DataFrames into the row-group cache. If the column is absent from a
        file, fills with ``default`` for that file's rows.

        When a ``row_filter`` is active, only filtered rows are returned.

        Args:
            column_name: Name of the column to read.
            default: Value to use when the column is absent from a file.
                If ``None`` (the default) and the column is absent from ALL
                files, a ``ValueError`` is raised to catch typos early.

        Returns:
            List of column values in global (or filtered) row order.

        Raises:
            ValueError: If ``default`` is ``None`` and ``column_name`` is
                not found in any file's schema.
        """
        # Check column exists in at least one file when no explicit default
        files = self._ensure_files_open()
        if default is None:
            found = any(column_name in pf.schema_arrow.names for pf in files)
            if not found:
                raise ValueError(
                    f"Column '{column_name}' not found in any Parquet file. "
                    f"Available columns: {self.column_names}"
                )

        all_values: list[Any] = []
        for pf in files:
            if column_name in pf.schema_arrow.names:
                table = pf.read(columns=[column_name])
                all_values.extend(table.column(column_name).to_pylist())
            else:
                all_values.extend([default] * pf.metadata.num_rows)

        if self._index_map is not None:
            return [all_values[i] for i in self._index_map]
        return all_values

    def read_source_paths(self) -> list[str]:
        """``str(path)`` of the file each row came from, in row order.

        Filtered like ``read_column``. Backs the ``SOURCE_FILE_FEATURE`` edge
        feature that ``PerSourceStrategy`` dispatches on.
        """
        all_sources: list[str] = []
        for path, num_rows in zip(self._paths, self._file_num_rows):
            all_sources.extend([str(path)] * num_rows)
        if self._index_map is not None:
            return [all_sources[i] for i in self._index_map]
        return all_sources

    def _row_group_selection(self, start: int, end: int) -> Optional[np.ndarray]:
        """Row-group-local indices passing the filter for physical ``[start, end)``.

        Returns ``None`` when no ``row_filter`` is active (every row is kept).
        ``_index_map`` is ascending, so the kept physical indices for one row
        group are a contiguous slice of it, locatable by bisection without
        scanning the whole map.
        """
        if self._index_map is None:
            return None
        lo = bisect.bisect_left(self._index_map, start)
        hi = bisect.bisect_left(self._index_map, end)
        return np.asarray(self._index_map[lo:hi], dtype=np.int64) - start

    def iter_row_group_columns(
        self, column_names: Sequence[str]
    ) -> Iterator[tuple[pa.Table, Optional[np.ndarray]]]:
        """Stream the requested columns as Arrow data, one row group at a time.

        Complements ``read_column`` rather than replacing it: ``read_column``
        materializes a whole column as a Python list (and, under a
        ``row_filter``, reads every physical row before subsetting), which is
        what a caller wants for small metadata columns and what makes it
        unaffordable for multi-gigabyte string columns. This yields raw Arrow
        arrays with a working set of one row group, so a caller can reduce over
        a column it could never hold.

        Row groups are read directly, bypassing the row-group LRU cache: a full
        scan touches every row group exactly once, so caching them would evict
        the training working set for no reuse.

        Args:
            column_names: Columns to read. Every name must be present in every
                file's schema — unlike ``read_column``, there is no per-file
                default fill, because a row group's Arrow type is not known
                without the column.

        Yields:
            ``(table, selection)`` per non-empty row group in physical order.
            ``table`` holds ``column_names`` for that row group; ``selection``
            is the ascending array of indices *within that table* that pass
            ``row_filter``, or ``None`` when no filter is active (all rows
            kept). Row groups with no selected rows are skipped, so
            concatenating each yield's selected rows in order reproduces
            filtered index order — the same order ``read_column``,
            ``get_row``, and ``row_group_boundaries`` use.

        Raises:
            ValueError: If a requested column is missing from any file.
        """
        files = self._ensure_files_open()
        for name in column_names:
            absent = [
                str(self._paths[i])
                for i, pf in enumerate(files)
                if name not in pf.schema_arrow.names
            ]
            if absent:
                raise ValueError(
                    f"Column '{name}' not found in {absent}. "
                    f"Available columns: {self.column_names}"
                )

        columns = list(column_names)
        for file_idx, pf in enumerate(files):
            file_offset = self._file_cumulative[file_idx]
            rg_cumulative = self._file_rg_cumulative[file_idx]
            for rg_idx in range(len(self._file_rg_sizes[file_idx])):
                start = file_offset + rg_cumulative[rg_idx]
                end = file_offset + rg_cumulative[rg_idx + 1]
                if start == end:
                    continue
                selection = self._row_group_selection(start, end)
                if selection is not None and selection.size == 0:
                    continue
                yield pf.read_row_group(rg_idx, columns=columns), selection


# ---------------------------------------------------------------------------
# ShuffleBufferSampler
# ---------------------------------------------------------------------------


class ShuffleBufferSampler(Sampler[int]):
    """Shuffle-buffer sampler for chunked on-disk data.

    Shuffles chunk (row group) order, streams chunks into a buffer,
    and yields random indices from the buffer. Produces the same I/O
    pattern as a true shuffle-buffer IterableDataset while working
    with map-style Datasets.

    Args:
        row_group_boundaries: List of (start, end) index pairs for each
            row group, as returned by ``ParquetBackend.row_group_boundaries``.
        buffer_size: Target buffer size. The buffer is drained to this size
            after each row group is loaded. Larger buffers give better
            randomization at the cost of more memory.
        generator: Optional random generator for reproducibility.
        shuffle: When False, yield indices in file order and never build the
            buffer. Evaluation sweeps every sample exactly once, so their order
            cannot change any metric -- but shuffling them is not free: a
            drained buffer keeps indices from ``buffer_size // rows_per_group``
            row groups live at once, and once that working set exceeds the LRU
            cache every access re-decompresses a whole row group. Measured on
            the v11 S corpus (157,669 val rows over a 1.58M-row file), a
            shuffled val sweep ran at 18.5 samples/s against 372.8 sequential
            -- 20x, for randomization that validation does not use. The
            ``row_group_boundaries`` and ``buffer_size`` properties stay
            populated either way, so ``TokenBudgetBatchSampler`` keeps the
            locality geometry it reads off this sampler.
    """

    def __init__(
        self,
        row_group_boundaries: list[tuple[int, int]],
        buffer_size: int = 8192,
        generator: Optional[torch.Generator] = None,
        shuffle: bool = True,
    ) -> None:
        self._row_group_boundaries = row_group_boundaries
        self._buffer_size = buffer_size
        self._generator = generator
        self._shuffle = shuffle
        self._total = sum(end - start for start, end in row_group_boundaries)

    @property
    def draws_global_rng(self) -> bool:
        """Whether each pass's order is drawn from torch's global RNG."""
        return self._shuffle and self._generator is None

    def __iter__(self) -> Iterator[int]:
        """Yield indices in shuffle-buffer order, or in file order."""
        if not self._shuffle:
            for start, end in self._row_group_boundaries:
                yield from range(start, end)
            return
        # Shuffle row group order. Both draws use ``self._generator``, or
        # torch's global RNG when it is None, so a seeded run replays the same
        # order; the buffer RNG gets a fresh seed each epoch as state advances.
        rg_order = torch.randperm(
            len(self._row_group_boundaries), generator=self._generator
        ).tolist()
        epoch_seed = int(torch.randint(0, 2**31, (1,), generator=self._generator))
        rng = random.Random(epoch_seed)

        buffer: list[int] = []

        for rg_idx in rg_order:
            start, end = self._row_group_boundaries[rg_idx]
            buffer.extend(range(start, end))
            # Drain buffer to target size
            while len(buffer) > self._buffer_size:
                idx = rng.randint(0, len(buffer) - 1)
                yield buffer.pop(idx)

        # Drain remaining
        rng.shuffle(buffer)
        yield from buffer

    def __len__(self) -> int:
        """Total number of samples across all row groups."""
        return self._total

    @property
    def row_group_boundaries(self) -> list[tuple[int, int]]:
        """The ``(start, end)`` index pairs this sampler streams over.

        Exposed so a wrapping batch sampler (e.g. ``TokenBudgetBatchSampler``'s
        locality-preserving mode) can replicate the same row-group streaming
        order, keeping the ``ParquetBackend`` LRU cache warm.
        """
        return self._row_group_boundaries

    @property
    def buffer_size(self) -> int:
        """The shuffle-buffer size (governs the live row-group working set)."""
        return self._buffer_size


# ---------------------------------------------------------------------------
# Row converter builders
# ---------------------------------------------------------------------------


class ParquetRowConverter:
    """Picklable callable that converts a raw Parquet row to a seed sample dict.

    Uses precomputed segment column names to map segment-specific Parquet
    columns (e.g. ``parent_heavy``, ``nt_rates_heavy``) to the generic keys
    expected by the transform chain (``raw_parent``, ``nt_rates``).

    For multi-segment data (e.g. paired antibody heavy+light):
    - NT sequences are joined with ``^^^`` separators
    - ``nt_rates`` tensors are combined with separator fill values (1.0)
    - ``nt_csps`` tensors are combined with separator fill values (0.0)
    - Rates are assumed to be pre-adjusted at write time (no per-segment scaling)

    For single-segment data, this reduces to simple column renames.

    This is a class (not a closure) so it can be pickled by PyTorch
    DataLoader workers.
    """

    def __init__(
        self,
        parent_cols: list[str],
        child_cols: list[str],
        rates_cols: list[str],
        csps_cols: list[str],
        scalar_cols: tuple[str, ...] = (),
    ) -> None:
        self.parent_cols = parent_cols
        self.child_cols = child_cols
        self.rates_cols = rates_cols
        self.csps_cols = csps_cols
        # Non-segmented per-row columns copied through verbatim (e.g. ``clade``,
        # which per-clade censoring keys on). Opt-in rather than always-on: this
        # converter is shared with the DASM/Compara path, whose Parquet files
        # need not carry them.
        self.scalar_cols = scalar_cols
        self.seq_col_pairs = [
            ("raw_parent", parent_cols),
            ("raw_child", child_cols),
        ]

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        sample: dict[str, Any] = {}

        for col in self.scalar_cols:
            sample[col] = row[col]

        # Combine parent/child NT sequences from segment columns.
        # Must include ALL segments (even empty/missing ones) to match
        # SegmentHandler._combine_segments behavior: heavy-only antibody
        # data becomes "ACGT...^^^" with trailing separator for empty light.
        for sample_key, cols in self.seq_col_pairs:
            segments = []
            for col in cols:
                val = row.get(col)
                segments.append(str(val) if val is not None else "")
            sample[sample_key] = NT_SEPARATOR.join(segments)

        # Combine nt_rates from segment columns with separator fills.
        # Rates are assumed to be pre-adjusted (rate adjustments baked in at
        # write time), so no per-segment scaling is applied here.
        rate_parts: list[torch.Tensor] = []
        for seg_idx, col in enumerate(self.rates_cols):
            if seg_idx > 0:
                rate_parts.append(
                    torch.full((NT_SEPARATOR_LEN,), SEPARATOR_NT_RATE_FILL)
                )
            val = row.get(col)
            if val is not None and len(val) > 0:
                seg_rates = torch.tensor(val, dtype=torch.float32)
                rate_parts.append(seg_rates)

        if rate_parts:
            sample["nt_rates"] = torch.cat(rate_parts)

        # Combine nt_csps from segment columns with separator fills.
        csp_parts: list[torch.Tensor] = []
        for seg_idx, col in enumerate(self.csps_cols):
            if seg_idx > 0:
                csp_parts.append(
                    torch.full((NT_SEPARATOR_LEN, 4), SEPARATOR_NT_CSP_FILL)
                )
            val = row.get(col)
            if val is not None and len(val) > 0:
                seg_csps = torch.tensor(val, dtype=torch.float32).reshape(-1, 4)
                csp_parts.append(seg_csps)

        if csp_parts:
            sample["nt_csps"] = torch.cat(csp_parts)

        return sample


def make_parquet_row_converter(
    segment_handler: "SegmentHandler",
    available_columns: Optional[set[str]] = None,
    scalar_cols: tuple[str, ...] = (),
) -> ParquetRowConverter:
    """Build a callable that converts a raw Parquet row to a seed sample dict.

    Args:
        segment_handler: Handler defining segment suffixes and rate adjustments.
        available_columns: Optional set of column names available in the
            Parquet file(s). When provided, validates all expected columns
            exist at construction time. Pass ``set(backend.column_names)``.
        scalar_cols: Non-segmented columns to carry through verbatim, e.g.
            ``("clade",)`` for per-clade censoring. Validated against
            ``available_columns`` when that is given, so a missing one fails at
            construction rather than in a DataLoader worker's first row.

    Returns:
        Picklable callable that takes a row dict and returns a seed sample dict.

    Raises:
        ValueError: If ``available_columns`` is provided and no complete
            segment is found, or a requested scalar column is absent.
    """
    # Precompute column names via the authoritative SegmentHandler method
    parent_cols = segment_handler.get_column_names("parent")
    child_cols = segment_handler.get_column_names("child")
    rates_cols = segment_handler.get_column_names("nt_rates")
    csps_cols = segment_handler.get_column_names("nt_csps")

    # Validate columns using the same logic as the CSV path's
    # validate_segment_format: at least one complete segment must be
    # present; absent segments are handled gracefully via row.get().
    if available_columns is not None:
        segment_handler.validate_segment_columns(available_columns, ("parent", "child"))
        if missing := sorted(set(scalar_cols) - available_columns):
            raise ValueError(
                f"Parquet is missing requested scalar column(s) {missing}. "
                f"Available: {sorted(available_columns)}."
            )

    return ParquetRowConverter(
        parent_cols, child_cols, rates_cols, csps_cols, scalar_cols
    )
