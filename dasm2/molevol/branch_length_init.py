"""Branch length initialization from mutation frequency and closed-form estimates.

This module provides fast branch length initialization for DASM training using:

1. **Slope-based initialization**: Uses an empirical linear relationship between
   mutation frequency and optimal DASM branch lengths (t = slope * f_mut).

2. **Selection-adjusted closed-form estimate**: MLE approximation at the codon
   level separating mutated and unmutated codons, with nucleotide rates scaled
   by per-codon selection factors:

   t = (1 / r̄_M) * log(1 + |M| * r̄_M / (|U| * r̄_U))

   where r̄_M, r̄_U are mean selection-adjusted nt_rates over nucleotide sites
   belonging to mutated/unmutated codons. For mutated codons, rates are scaled
   by sel(child_aa); for unmutated codons, by mean(sel(non-parent_aas)).
   Vectorized (no Python loops over codons), ~0.15ms/seq.

Background (slope-based):
    For antibody heavy chains (rate adjustment 1.0) trained with v1jaffeCC+v1tangCC:
    - slope = 1.76, R² = 0.974, RMSE = 0.0065

    For Compara single-segment proteins (phyz ASR branch lengths, 50K sample):
    - slope = 1.24, stored in parquet metadata as "branch_length_slope"
    - R² = 0.58 (origin-centered, no-intercept)
    - Lower R² reflects heterogeneous gene families spanning broad divergence
    - Recalibrated with mutable-site denominator after phyz 1043 gap-preserving
      PCP rebuild (issue #512); gaps mapped to N and excluded from denominator
    - Leaf edges (slope=1.60) vs internal edges (slope=0.80) differ 2x

    The slope of 1.76 reflects purifying selection: observed mutations are ~57% as
    likely as neutral expectation.

Multi-segment sequences:
    For sequences with multiple segments (e.g., heavy and light chains), we compute
    a single branch length using rate-adjusted mutation frequency:

    f_adjusted = Σ(n_i / c_i) / Σ(m_i)

    where for segment i:
    - n_i = number of mutations (excluding ambiguous N and gap positions)
    - m_i = number of mutable sites (positions where neither parent nor child
      has N or '-')
    - c_i = rate adjustment coefficient (heavy: 1.0, light: 0.63)

    Then: t = slope * f_adjusted

    This normalizes mutation counts to a common reference rate. A mutation on a
    light chain (c=0.63) contributes 1/0.63 ≈ 1.59 effective mutations because it
    took more evolutionary time to accumulate given the slower mutation rate.

Pluggable strategies:
    The ``BranchLengthInitStrategy`` ABC allows different initialization approaches.
    ``ProportionalStrategy`` implements the classic ``slope * mut_freq`` model.
    ``JCStratifiedStrategy`` applies Jukes-Cantor correction and per-stratum slopes,
    reaching R² ≈ 0.75 on Compara data by correcting for saturation and
    stratifying by edge type (leaf vs internal). ``PerSourceStrategy`` routes
    each row of a multiname Parquet spec to its own file's strategy.

Three implementations of the same mutation frequency:
    ``_mut_freq_from_segments`` works per edge over NT strings and is the
    fallback for in-memory and precomputed datasets. ``_mut_freqs_from_arrow``
    reduces whole blocks of rows straight off Parquet's Arrow string buffers
    and is the cold-start path for Parquet-backed datasets, where materializing
    the NT columns as Python strings dominated the hook (issue #992).
    ``_compute_tensor_mut_freq`` (in the BL hook) works from codon index
    tensors during re-optimization. The first two agree bit for bit, pinned by
    exact-equality tests against a retained scalar oracle; the third differs
    only in dropping a whole codon per ambiguous nucleotide.

See Also:
    - Issue #140 for slope-based derivation
    - Issue #499 for closed-form estimates and warm-started optimization
    - Issue #534 for pluggable strategy design
    - Issue #992 for the columnar Arrow cold-start path
    - AntibodySegmentHandler.bl_init_strategy for the calibrated slope value
"""

import json
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import torch

from dasm2.codon_table import CODON_AA_INDICATOR_MATRIX, STOP_CODON_INDICATOR
from dasm2.data.arrow_strings import iter_segment_blocks, iter_selected_row_group_values
from dasm2.sequence.constants import AMBIGUOUS_CODON_IDX, NT_SEPARATOR

logger = logging.getLogger(__name__)

# ASCII byte values of the two characters that make a site unusable for
# mutation-frequency estimation. Used by the vectorized site mask in
# ``_mut_freq_from_segments``, which compares raw bytes rather than
# characters. Uppercase only, deliberately: a lowercase ``n`` is a real,
# distinct nucleotide as far as this function is concerned, and widening the
# ambiguity set would shrink the denominator on any soft-masked corpus and
# silently shift every initialized branch length.
_N_BYTE = ord("N")
_GAP_BYTE = ord("-")

# Lookup table: codon index (0-63) → amino acid index (0-19).
# Stop codons (indices 48, 50, 56) have all-zero indicator rows so argmax
# returns 0 for them. This is fine because stop codons are excluded by the
# valid mask in selection_adjusted_bl_estimate (via _is_stop_codon).
# CPU tensor — callers must pass CPU tensors for indexing (this runs in
# scipy worker processes, never on GPU).
_CODON_TO_AA = CODON_AA_INDICATOR_MATRIX.argmax(dim=1)  # [64]
_IS_STOP_CODON = STOP_CODON_INDICATOR.bool()  # [64]

if TYPE_CHECKING:
    from dasm2.data.evolutionary_dataset import BranchLengthDatasetBundle
    from dasm2.data.parquet_backend import ParquetBackend
    from dasm2.data.segment_handlers import SegmentHandler


class BranchLengthInitStrategy(ABC):
    """Computes initial branch length from mutation frequency and edge features.

    Subclasses implement a mapping from observed mutation frequency (and
    optional per-edge features) to an initial branch length estimate.
    Strategies are serialized to/from parquet metadata as JSON, and
    deserialized via STRATEGY_REGISTRY.
    """

    @property
    @abstractmethod
    def required_features(self) -> frozenset[str]:
        """Edge feature column names this strategy needs at call time."""
        ...

    @abstractmethod
    def __call__(self, mut_freq: float, edge_features: dict[str, Any]) -> float:
        """Compute initial branch length.

        Args:
            mut_freq: Adjusted mutation frequency from
                compute_adjusted_mutation_frequency().
            edge_features: Per-edge metadata from parquet columns
                (e.g., {"child_is_leaf": True, "depth": 3}).

        Returns:
            Estimated branch length (non-negative float).

        Raises:
            ValueError: If a required edge feature is missing or has
                an unknown stratum value.
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for parquet metadata."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BranchLengthInitStrategy":
        """Deserialize from a dict parsed from parquet metadata JSON."""
        ...

    def validate_features(self, available: set[str]) -> None:
        """Raise ValueError if required features are missing from available set."""
        missing = self.required_features - available
        if missing:
            raise ValueError(
                f"Strategy {type(self).__name__} requires features {missing} "
                f"but only {available} available"
            )


STRATEGY_REGISTRY: dict[str, type[BranchLengthInitStrategy]] = {}


class ProportionalStrategy(BranchLengthInitStrategy):
    """BL = slope * mut_freq. Current behavior.

    Args:
        slope: Proportional slope relating mutation frequency to branch length.
    """

    def __init__(self, slope: float) -> None:
        self.slope = slope

    @property
    def required_features(self) -> frozenset[str]:
        return frozenset()

    def __call__(self, mut_freq: float, edge_features: dict[str, Any]) -> float:
        return self.slope * mut_freq

    def to_dict(self) -> dict[str, Any]:
        return {"type": "ProportionalStrategy", "slope": self.slope}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ProportionalStrategy":
        return cls(slope=raw["slope"])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ProportionalStrategy) and self.slope == other.slope

    def __repr__(self) -> str:
        return f"ProportionalStrategy(slope={self.slope})"


STRATEGY_REGISTRY["ProportionalStrategy"] = ProportionalStrategy


JC_SATURATION_THRESHOLD = 0.75
"""Maximum mutation frequency for which Jukes-Cantor correction is defined.

At p = 3/4, the argument to ln becomes zero: 1 - 4/3 * 3/4 = 0.
Above this, the observed mutation frequency has saturated and the
JC model cannot estimate the true evolutionary distance.
"""

JC_SATURATION_CLAMP = JC_SATURATION_THRESHOLD - 1e-6
"""Clamp value used when mut_freq reaches JC saturation.

Values at or above 0.75 make jc_correct() undefined (log of zero).
JCStratifiedStrategy clamps to just below the threshold rather than
raising, because long Compara sequences occasionally produce mut_freq
at the mathematical boundary. The resulting branch length estimate is
a large-but-finite upper bound (~4.4 JC distance).
"""


def jc_correct(mut_freq: float) -> float:
    """Apply Jukes-Cantor correction: d = -3/4 * ln(1 - 4/3 * p).

    Args:
        mut_freq: Observed mutation frequency, must be in [0, 0.75).

    Returns:
        JC-corrected evolutionary distance.

    Raises:
        ValueError: If mut_freq < 0 or mut_freq >= JC_SATURATION_THRESHOLD.
    """
    if mut_freq < 0:
        raise ValueError(f"Mutation frequency must be non-negative, got {mut_freq}")
    if mut_freq >= JC_SATURATION_THRESHOLD:
        raise ValueError(
            f"Mutation frequency {mut_freq} >= {JC_SATURATION_THRESHOLD} "
            f"(JC saturation threshold). Cannot compute JC correction."
        )
    if mut_freq == 0.0:
        return 0.0
    return -0.75 * math.log1p(-4 / 3 * mut_freq)


class JCStratifiedStrategy(BranchLengthInitStrategy):
    """BL = slope[stratum] * jc_correct(mut_freq).

    Applies Jukes-Cantor correction to linearize the saturating
    mut_freq-to-BL relationship, then uses a per-stratum slope
    looked up from a specified edge feature column.

    Args:
        slopes: Mapping from stratum value (as string) to slope.
        stratify_by: Name of edge feature column to stratify on.
    """

    def __init__(self, slopes: dict[str, float], stratify_by: str) -> None:
        self.slopes = slopes
        self.stratify_by = stratify_by

    @property
    def required_features(self) -> frozenset[str]:
        return frozenset({self.stratify_by})

    def __call__(self, mut_freq: float, edge_features: dict[str, Any]) -> float:
        if mut_freq >= JC_SATURATION_THRESHOLD:
            logger.warning(
                "mut_freq %.4f at or above JC saturation; clamping to %.6f",
                mut_freq,
                JC_SATURATION_CLAMP,
            )
            mut_freq = JC_SATURATION_CLAMP
        jc_dist = jc_correct(mut_freq)
        if self.stratify_by not in edge_features:
            raise ValueError(
                f"Required edge feature '{self.stratify_by}' not provided. "
                f"Available: {list(edge_features)}"
            )
        stratum = str(edge_features[self.stratify_by])
        if stratum not in self.slopes:
            raise ValueError(
                f"Unknown stratum '{stratum}' for feature '{self.stratify_by}'. "
                f"Known strata: {list(self.slopes)}"
            )
        return self.slopes[stratum] * jc_dist

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "JCStratifiedStrategy",
            "slopes": self.slopes,
            "stratify_by": self.stratify_by,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "JCStratifiedStrategy":
        return cls(slopes=raw["slopes"], stratify_by=raw["stratify_by"])

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, JCStratifiedStrategy)
            and self.slopes == other.slopes
            and self.stratify_by == other.stratify_by
        )

    def __repr__(self) -> str:
        return (
            f"JCStratifiedStrategy(slopes={self.slopes}, "
            f"stratify_by={self.stratify_by!r})"
        )


STRATEGY_REGISTRY["JCStratifiedStrategy"] = JCStratifiedStrategy


SOURCE_FILE_FEATURE = "parquet_source_file"
"""Edge feature naming the Parquet file a row was read from.

Not a Parquet column: the Parquet dataset factory fills it from
``ParquetBackend.read_source_paths`` when a strategy requires it.
"""


class PerSourceStrategy(BranchLengthInitStrategy):
    """Apply each row's own source file's strategy (issue #1289).

    A multiname Parquet spec joins files whose writers each fit their own
    strategy. This dispatches every edge to the strategy stamped on the file
    it came from, keyed by ``str(path)`` and looked up through the
    ``SOURCE_FILE_FEATURE`` edge feature.

    Built in memory by the Parquet resolver, never stamped on a file: its keys
    are machine-local paths, so it is left out of ``STRATEGY_REGISTRY`` and
    ``deserialize_strategy`` rejects it. Like any strategy it lives on a shared
    ``STANDARD_SEGMENT_HANDLERS`` singleton, so a later load of another spec
    replaces it; a bundle whose paths it does not know raises on first use.

    Args:
        strategies: Mapping from ``str(path)`` of each Parquet file to the
            strategy stamped on it.
    """

    def __init__(self, strategies: dict[str, BranchLengthInitStrategy]) -> None:
        self.strategies = strategies

    @property
    def required_features(self) -> frozenset[str]:
        features = {SOURCE_FILE_FEATURE}
        for strategy in self.strategies.values():
            features |= strategy.required_features
        return frozenset(features)

    def __call__(self, mut_freq: float, edge_features: dict[str, Any]) -> float:
        if SOURCE_FILE_FEATURE not in edge_features:
            raise ValueError(
                f"Required edge feature '{SOURCE_FILE_FEATURE}' not provided. "
                f"Available: {list(edge_features)}"
            )
        strategy = self.strategy_for(edge_features[SOURCE_FILE_FEATURE])
        return strategy(mut_freq, edge_features)

    def strategy_for(self, source: str) -> BranchLengthInitStrategy:
        """The strategy stamped on file ``source`` (a ``str(path)``)."""
        if source not in self.strategies:
            raise ValueError(
                f"No BL init strategy for source file {source!r}. "
                f"Known: {list(self.strategies)}"
            )
        return self.strategies[source]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "PerSourceStrategy",
            "strategies": {
                source: strategy.to_dict()
                for source, strategy in self.strategies.items()
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PerSourceStrategy":
        return cls(
            strategies={
                source: STRATEGY_REGISTRY[sub["type"]].from_dict(sub)
                for source, sub in raw["strategies"].items()
            }
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PerSourceStrategy) and self.strategies == other.strategies
        )

    def __repr__(self) -> str:
        return f"PerSourceStrategy(strategies={self.strategies})"


def deserialize_strategy(
    metadata: dict[str, str],
) -> BranchLengthInitStrategy | None:
    """Deserialize a BranchLengthInitStrategy from parquet metadata JSON.

    Checks for 'bl_init_strategy' (new format) first, then falls back to
    'branch_length_slope' (legacy single-float format) for convenience.

    Returns:
        Strategy instance, or None if neither key is present.

    Raises:
        ValueError: If strategy type is not in STRATEGY_REGISTRY.
    """
    if "bl_init_strategy" in metadata:
        raw = json.loads(metadata["bl_init_strategy"])
        strategy_type = raw["type"]
        if strategy_type not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown BL init strategy: {strategy_type}. "
                f"Known: {list(STRATEGY_REGISTRY)}"
            )
        return STRATEGY_REGISTRY[strategy_type].from_dict(raw)
    if "branch_length_slope" in metadata:
        return ProportionalStrategy(slope=float(metadata["branch_length_slope"]))
    return None


def compute_adjusted_mutation_frequency(
    parent: str,
    child: str,
    segment_handler: "SegmentHandler",
) -> float:
    """Compute mutation frequency adjusted for segment-specific mutation rates.

    For sequences with multiple segments (joined by ^^^), normalizes mutation
    counts by dividing each segment's count by its rate adjustment coefficient
    before summing. This accounts for segments with different baseline mutation
    rates. Single-segment sequences are handled uniformly (as one-segment case).

    Formula: f_adjusted = Σ(n_i / c_i) / Σ(m_i)
    where:
        n_i = mutations in segment i (excluding N and gap positions)
        c_i = rate adjustment for segment i (e.g., heavy=1.0, light=0.63)
        m_i = mutable sites in segment i (positions where neither parent nor
              child has N or '-')

    Args:
        parent: Parent nucleotide sequence (segments joined by ^^^)
        child: Child nucleotide sequence (segments joined by ^^^)
        segment_handler: Handler providing rate adjustments per segment

    Returns:
        Adjusted mutation frequency: Σ(n_i/c_i) / Σ(m_i)

    Raises:
        ValueError: If a non-empty sequence has zero mutable sites (all N or gap),
            or on segment-count, segment-length, or rate-adjustment violations.
        UnicodeEncodeError: If a sequence contains non-ASCII characters, which are
            not nucleotides. See :func:`_mut_freq_from_segments`.
    """
    return _mut_freq_from_segments(
        parent.split(NT_SEPARATOR),
        child.split(NT_SEPARATOR),
        segment_handler.get_rate_adjustments().tolist(),
    )


def _mut_freq_from_segments(
    parent_segments: Sequence[str],
    child_segments: Sequence[str],
    rate_adjustments: Sequence[float],
) -> float:
    """Rate-adjusted mutation frequency from pre-split segments.

    The body of :func:`compute_adjusted_mutation_frequency`, split out so that
    per-edge loops can hoist the ``split`` and ``get_rate_adjustments()`` calls
    out of the loop (issue #992). The site mask is computed on raw ASCII bytes
    via ``np.frombuffer`` rather than by iterating characters, which is what
    makes full-corpus cold-start initialization tractable.

    Segments must be pre-split on ``NT_SEPARATOR``: a single ``frombuffer``
    over a joined sequence would count each separator character as a valid
    unmutated site (``ord('^') == 94`` is neither N nor gap) and apply one rate
    divisor instead of per-segment ones.

    Args:
        parent_segments: Parent sequence split on ``NT_SEPARATOR``.
        child_segments: Child sequence split on ``NT_SEPARATOR``.
        rate_adjustments: Per-segment rate adjustment coefficients.

    Returns:
        Adjusted mutation frequency: Σ(n_i/c_i) / Σ(m_i)

    Raises:
        ValueError: On segment-count mismatch, per-segment length mismatch,
            non-positive rate adjustment, or a non-empty sequence with zero
            mutable sites (all N or gap).
        UnicodeEncodeError: If a segment contains non-ASCII characters, which
            are not nucleotides and whose byte length would not match their
            character length.
    """
    if len(parent_segments) != len(rate_adjustments):
        raise ValueError(
            f"Segment count mismatch: parent has {len(parent_segments)} segments "
            f"but rate_adjustments has {len(rate_adjustments)}"
        )
    if len(child_segments) != len(rate_adjustments):
        raise ValueError(
            f"Segment count mismatch: child has {len(child_segments)} segments "
            f"but rate_adjustments has {len(rate_adjustments)}"
        )

    effective_mutations = 0.0
    total_mutable_sites = 0

    for idx, (p_seg, c_seg, rate_adj) in enumerate(
        zip(parent_segments, child_segments, rate_adjustments)
    ):
        if p_seg:  # Non-empty segment
            if len(p_seg) != len(c_seg):
                raise ValueError(
                    f"Parent and child segment lengths must match: "
                    f"segment {idx} has parent={len(p_seg)}, child={len(c_seg)}"
                )
            if rate_adj <= 0:
                raise ValueError(
                    f"Rate adjustment must be positive, got {rate_adj} for segment {idx}"
                )
            p_nt = np.frombuffer(p_seg.encode("ascii"), dtype=np.uint8)
            c_nt = np.frombuffer(c_seg.encode("ascii"), dtype=np.uint8)
            mutable = (
                (p_nt != _N_BYTE)
                & (p_nt != _GAP_BYTE)
                & (c_nt != _N_BYTE)
                & (c_nt != _GAP_BYTE)
            )
            num_mutable_sites = int(np.count_nonzero(mutable))
            num_valid_mutations = int(np.count_nonzero(mutable & (p_nt != c_nt)))
            effective_mutations += num_valid_mutations / float(rate_adj)
            total_mutable_sites += num_mutable_sites

    if total_mutable_sites == 0:
        # All segments are empty (legitimate in multi-segment sequences where
        # one chain is absent) — zero mutation frequency is correct.
        # But non-empty sequences that are entirely gaps/Ns are a data quality
        # error: we can't estimate a mutation frequency from zero usable sites.
        total_length = sum(len(s) for s in parent_segments)
        if total_length > 0:
            raise ValueError(
                f"Sequence has {total_length} nucleotides but zero mutable sites "
                f"(all positions are N or gap). Cannot compute mutation frequency."
            )
        return 0.0
    return effective_mutations / total_mutable_sites


def _row_sums(flags: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Per-row sums of a byte-level boolean array delimited by ``offsets``.

    ``np.add.reduceat`` is the memory-frugal reduction for this — unlike a
    ``cumsum``, it needs no widened array over the whole data buffer — but two
    of its edge behaviours would silently produce wrong counts:

    - a zero-length row returns ``flags[start]`` rather than 0, borrowing the
      next row's first nucleotide;
    - a start index equal to ``flags.size`` raises ``IndexError``, which
      trailing empty rows produce. Clamping such an index is *not* the fix:
      each row's end boundary is the next row's start, so clamping truncates
      the preceding row's range instead.

    Both are handled by reducing only over the rows that begin inside the
    buffer — the rest are empty by construction — and then zeroing every
    zero-length row.
    """
    lengths = offsets[1:] - offsets[:-1]
    counts = np.zeros(lengths.size, dtype=np.int64)
    if lengths.size == 0 or flags.size == 0:
        return counts
    # Rows starting at ``flags.size`` are trailing empty rows: since offsets are
    # non-decreasing and end at ``flags.size``, dropping them leaves the last
    # reduced row ending exactly at the end of ``flags``, which is its true end.
    inside = int(np.searchsorted(offsets[:-1], flags.size))
    counts[:inside] = np.add.reduceat(flags, offsets[:inside], dtype=np.int64)
    counts[lengths == 0] = 0
    return counts


def _mut_freqs_from_arrow(
    parent_columns: Sequence["pa.ChunkedArray"],
    child_columns: Sequence["pa.ChunkedArray"],
    rate_adjustments: Sequence[float],
) -> np.ndarray:
    """Rate-adjusted mutation frequency per row, straight off Arrow buffers.

    The columnar counterpart of :func:`_mut_freq_from_segments`, computing the
    same quantity for a whole block of rows at once with no Python string
    objects. Per-segment sequences arrive as separate Arrow columns, so there
    is no ``NT_SEPARATOR`` to split and no separator hazard on this path.

    Per-segment contributions are accumulated in segment order in float64, the
    same order and precision the scalar path uses, so results agree bit for bit
    rather than approximately.

    Args:
        parent_columns: One Arrow string column per segment, parent sequences.
        child_columns: One Arrow string column per segment, child sequences.
        rate_adjustments: Per-segment rate adjustment coefficients.

    Returns:
        Float64 array of length ``len(parent_columns[0])``.

    Raises:
        ValueError: On segment-count mismatch, per-row/per-segment length
            mismatch, non-positive rate adjustment, or a non-empty sequence
            with zero mutable sites — the same conditions
            :func:`_mut_freq_from_segments` rejects, plus the degenerate
            empty-parent-with-non-empty-child row it tolerates (see below).

    Note:
        One input the scalar path silently skips is rejected here: a row whose
        parent segment is empty while its child segment is not. The scalar path
        keys the "absent segment" test off the parent alone and drops the pair;
        columnar comparison cannot, because unequal lengths misalign every
        later row's bytes in the shared buffer. Such a row is a corrupt
        parent/child pair in any case, and no valid corpus contains one.
    """
    if len(parent_columns) != len(rate_adjustments):
        raise ValueError(
            f"Segment count mismatch: parent has {len(parent_columns)} columns "
            f"but rate_adjustments has {len(rate_adjustments)}"
        )
    if len(child_columns) != len(rate_adjustments):
        raise ValueError(
            f"Segment count mismatch: child has {len(child_columns)} columns "
            f"but rate_adjustments has {len(rate_adjustments)}"
        )

    num_rows = len(parent_columns[0]) if parent_columns else 0
    effective_mutations = np.zeros(num_rows, dtype=np.float64)
    mutable_sites = np.zeros(num_rows, dtype=np.int64)
    parent_lengths = np.zeros(num_rows, dtype=np.int64)

    for idx, (parent_column, child_column, rate_adj) in enumerate(
        zip(parent_columns, child_columns, rate_adjustments)
    ):
        # A null segment reads as "" here, as it does on the scalar path after
        # ``read_column``'s ``str(value or "")``.
        for block, offsets, parent_nt, child_offsets, child_nt in iter_segment_blocks(
            parent_column, child_column
        ):
            lengths = offsets[1:] - offsets[:-1]
            child_lengths = child_offsets[1:] - child_offsets[:-1]
            mismatched = np.flatnonzero(lengths != child_lengths)
            if mismatched.size:
                bad = int(mismatched[0])
                if lengths[bad] == 0:
                    raise ValueError(
                        f"Parent segment {idx} is empty but the child segment "
                        f"has {child_lengths[bad]} nucleotides. Cannot align "
                        f"parent and child columns."
                    )
                raise ValueError(
                    f"Parent and child segment lengths must match: "
                    f"segment {idx} has parent={lengths[bad]}, "
                    f"child={child_lengths[bad]}"
                )
            parent_lengths[block] += lengths
            if not np.any(lengths):
                # Every row's segment is absent: nothing to count, and no
                # rate adjustment is applied, so it is not validated either.
                continue
            if rate_adj <= 0:
                raise ValueError(
                    f"Rate adjustment must be positive, got {rate_adj} for segment {idx}"
                )

            mutable = (
                (parent_nt != _N_BYTE)
                & (parent_nt != _GAP_BYTE)
                & (child_nt != _N_BYTE)
                & (child_nt != _GAP_BYTE)
            )
            mutable_sites[block] += _row_sums(mutable, offsets)
            effective_mutations[block] += _row_sums(
                mutable & (parent_nt != child_nt), offsets
            ) / float(rate_adj)

    no_mutable_sites = mutable_sites == 0
    unusable = np.flatnonzero(no_mutable_sites & (parent_lengths > 0))
    if unusable.size:
        bad = int(unusable[0])
        raise ValueError(
            f"Sequence has {parent_lengths[bad]} nucleotides but zero mutable "
            f"sites (all positions are N or gap). Cannot compute mutation "
            f"frequency."
        )
    # Rows with no mutable sites at all are empty sequences, which the scalar
    # path reports as zero frequency.
    frequencies = np.zeros(num_rows, dtype=np.float64)
    np.divide(
        effective_mutations, mutable_sites, out=frequencies, where=~no_mutable_sites
    )
    return frequencies


def mutation_frequencies_from_parquet(
    backend: "ParquetBackend",
    segment_handler: "SegmentHandler",
) -> np.ndarray:
    """Rate-adjusted mutation frequency for every row, read off Arrow buffers.

    Streams the parent and child segment columns one row group at a time and
    reduces each row group with numpy over the raw Arrow string buffers. This
    is the cold-start branch-length path for Parquet-backed datasets (issue
    #992): it replaces materializing both full NT columns as Python strings,
    whose cost is independent of how many rows the split actually keeps — a
    258k-row validation split paid a 5.05M-row read.

    Args:
        backend: Parquet backend to stream from. Under a row filter, only the
            selected rows are returned, in filtered index order.
        segment_handler: Handler naming the per-segment columns and supplying
            rate adjustments.

    Returns:
        Float64 array of length ``backend.total_rows``, aligned with
        ``read_column`` / ``get_row`` / ``edge_features`` index order.

    Raises:
        ValueError: If a segment column is missing, or on any of the data
            errors :func:`_mut_freqs_from_arrow` rejects.
    """
    rate_adjustments = segment_handler.get_rate_adjustments().tolist()
    parent_names = segment_handler.get_column_names("parent")
    child_names = segment_handler.get_column_names("child")

    frequencies = np.empty(backend.total_rows, dtype=np.float64)
    filled = 0
    for row_group_frequencies in iter_selected_row_group_values(
        backend,
        parent_names,
        child_names,
        lambda parents, children: _mut_freqs_from_arrow(
            parents, children, rate_adjustments
        ),
    ):
        frequencies[filled : filled + row_group_frequencies.size] = (
            row_group_frequencies
        )
        filled += row_group_frequencies.size
    if filled != backend.total_rows:
        raise RuntimeError(
            f"Row-group scan produced {filled} mutation frequencies but the "
            f"backend reports {backend.total_rows} rows"
        )
    return frequencies


def _mutation_frequencies(dataset: "BranchLengthDatasetBundle") -> np.ndarray:
    """Per-edge rate-adjusted mutation frequency for a whole bundle.

    Prefers the columnar Arrow path when the bundle is Parquet-backed and its
    NT sequences have not already been materialized; otherwise falls back to
    the per-edge string path over ``nt_parents``/``nt_children``, which is what
    in-memory and precomputed bundles use. Both compute the same quantity — bit
    for bit, pinned by tests — so the choice is purely about cost.
    """
    backend = dataset.parquet_backend
    if backend is not None and not dataset.nt_sequences_are_loaded:
        return mutation_frequencies_from_parquet(backend, dataset.segment_handler)

    # get_rate_adjustments() allocates a fresh tensor on every call, which at
    # multi-million-edge scale is pure overhead (#992); same for the split.
    rate_adjustments = dataset.segment_handler.get_rate_adjustments().tolist()
    nt_parents, nt_children = dataset.nt_parents, dataset.nt_children
    return np.array(
        [
            _mut_freq_from_segments(
                nt_parents[i].split(NT_SEPARATOR),
                nt_children[i].split(NT_SEPARATOR),
                rate_adjustments,
            )
            for i in range(len(nt_parents))
        ],
        dtype=np.float64,
    )


def initialize_branch_lengths(
    dataset: "BranchLengthDatasetBundle",
    strategy: BranchLengthInitStrategy,
    feature_columns: Mapping[str, Sequence[Any]],
) -> torch.Tensor:
    """Initialize branch lengths using a pluggable strategy.

    Computes adjusted mutation frequency for each parent-child pair — off the
    Parquet Arrow buffers where possible, otherwise per-edge over the NT
    strings (see :func:`_mutation_frequencies`) — then applies the strategy
    with per-edge features to produce initial branch length estimates.

    Args:
        dataset: BranchLengthDatasetBundle with nt_parents, nt_children,
            and segment_handler attributes.
        strategy: Strategy instance to compute branch lengths.
        feature_columns: Per-edge feature columns keyed by name.
            Each value is a sequence of length len(dataset), in the same
            (row-filtered) order as the mutation frequencies.

    Returns:
        Tensor of initialized branch lengths.

    Raises:
        ValueError: If a feature column's length does not match the number of
            edges, which would misalign per-edge features against frequencies.
    """
    strategy.validate_features(set(feature_columns))
    mut_freqs = _mutation_frequencies(dataset)
    misaligned = {
        name: len(values)
        for name, values in feature_columns.items()
        if len(values) != len(mut_freqs)
    }
    if misaligned:
        raise ValueError(
            f"Edge feature columns {misaligned} do not have one value per "
            f"edge ({len(mut_freqs)}). Per-edge features must be in the same "
            f"order as the dataset's rows, or strata would be swapped."
        )
    branch_lengths = [
        strategy(
            float(mut_freq),
            {name: values[i] for name, values in feature_columns.items()},
        )
        for i, mut_freq in enumerate(mut_freqs)
    ]
    return torch.tensor(branch_lengths)


def initialize_branch_lengths_from_mutation_frequency(
    dataset: "BranchLengthDatasetBundle",
) -> torch.Tensor | None:
    """Initialize branch lengths using the dataset's bl_init_strategy.

    Reads the strategy from the segment handler's ``bl_init_strategy``
    attribute. For strategies requiring edge features, reads them from
    ``dataset.edge_features``.

    Args:
        dataset: BranchLengthDatasetBundle with nt_parents, nt_children,
            segment_handler, and edge_features attributes.

    Returns:
        Tensor of initialized branch lengths, or None if segment handler
        lacks a configured bl_init_strategy.
    """
    strategy = dataset.segment_handler.bl_init_strategy

    if strategy is None:
        return None

    return initialize_branch_lengths(dataset, strategy, dataset.edge_features)


def selection_adjusted_bl_estimate(
    codon_parents_idxs: torch.Tensor,
    codon_children_idxs: torch.Tensor,
    nt_rates: torch.Tensor,
    mask: torch.Tensor,
    log_selection_factors: torch.Tensor,
    min_bl: float = 1e-6,
    max_bl: float = 5.0,
) -> float:
    """Estimate branch length using selection-adjusted nucleotide rates.

    Applies the codon-level MLE formula with nucleotide rates scaled by
    per-codon selection factors.

    For mutated codons (parent ≠ child), rates are scaled by the selection
    factor of the child amino acid — the mutation was *to* that AA.
    For unmutated codons (parent == child), rates are scaled by the mean
    selection factor across the 19 non-parent amino acids — any of those
    substitutions could have occurred but didn't.

    Vectorized: all tensor ops, no Python loops over codons.

    Args:
        codon_parents_idxs: Parent codon indices [aa_len].
        codon_children_idxs: Child codon indices [aa_len].
        nt_rates: Nucleotide mutation rates [nt_len] where nt_len = 3 * aa_len.
        mask: Loss mask [aa_len] excluding separators and padding.
        log_selection_factors: Log AA selection factors [aa_len, 20].
        min_bl: Minimum branch length.
        max_bl: Maximum branch length.

    Returns:
        Selection-adjusted branch length estimate, clamped to [min_bl, max_bl].
    """
    aa_len = mask.shape[0]
    assert log_selection_factors.shape[0] >= aa_len, (
        f"log_selection_factors length {log_selection_factors.shape[0]} < aa_len {aa_len}"
    )
    # Exclude ambiguous codons and stop codons from valid positions.
    parent_is_stop = _IS_STOP_CODON[codon_parents_idxs.clamp(max=63)]
    child_is_stop = _IS_STOP_CODON[codon_children_idxs.clamp(max=63)]
    valid = (
        mask.bool()
        & (codon_parents_idxs != AMBIGUOUS_CODON_IDX)
        & ~parent_is_stop
        & ~child_is_stop
    )
    mutated = valid & (codon_parents_idxs != codon_children_idxs)
    unmutated = valid & (codon_parents_idxs == codon_children_idxs)

    n_mut_nt = 3 * int(mutated.sum().item())
    n_unmut_nt = 3 * int(unmutated.sum().item())

    if n_mut_nt == 0:
        return min_bl
    if n_unmut_nt == 0:
        return max_bl

    # Compute per-codon selection scale factors
    sf = log_selection_factors[:aa_len].exp()  # [aa_len, 20]

    # Mutated codons: use selection factor of the child AA
    child_aa = _CODON_TO_AA[codon_children_idxs[:aa_len].clamp(max=63)]  # [aa_len]
    child_sf = sf.gather(1, child_aa.unsqueeze(1)).squeeze(1)  # [aa_len]

    # Unmutated codons: mean selection factor across 19 non-parent AAs
    parent_aa = _CODON_TO_AA[codon_parents_idxs[:aa_len].clamp(max=63)]  # [aa_len]
    # Sum all 20 AAs, subtract parent, divide by 19
    sf_sum_all = sf.sum(dim=1)  # [aa_len]
    sf_parent = sf.gather(1, parent_aa.unsqueeze(1)).squeeze(1)  # [aa_len]
    nonparent_mean_sf = (sf_sum_all - sf_parent) / 19.0  # [aa_len]

    # Combine: mutated positions get child_sf, unmutated get nonparent_mean_sf
    codon_scale = torch.where(mutated, child_sf, nonparent_mean_sf)
    codon_scale = codon_scale.clamp(min=1e-6)

    # Expand codon-level scale to nucleotide level and apply
    rates = nt_rates[: aa_len * 3].reshape(aa_len, 3) * codon_scale.unsqueeze(1)

    # Apply the same MLE formula on the selection-adjusted rates
    mut_nt_mask = mutated.unsqueeze(1).expand(-1, 3)
    unmut_nt_mask = unmutated.unsqueeze(1).expand(-1, 3)

    r_bar_m = rates[mut_nt_mask].mean().item()
    r_bar_u = rates[unmut_nt_mask].mean().item()

    if r_bar_m <= 0 or r_bar_u <= 0:
        return min_bl

    t = (1.0 / r_bar_m) * np.log1p(n_mut_nt * r_bar_m / (n_unmut_nt * r_bar_u))
    return float(np.clip(t, min_bl, max_bl))
