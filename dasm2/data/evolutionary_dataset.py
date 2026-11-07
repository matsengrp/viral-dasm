"""Transform-based dataset with precomputed and streaming modes.

Single dataset class (``TransformDataset``) that supports two execution modes:
- **precomputed**: All transforms run at init, ``__getitem__`` is pure index
  lookup. Use for datasets that fit in memory (up to ~100k samples).
- **streaming**: All transforms run per-sample in ``__getitem__``. Use for
  very large datasets (>1M samples) where precomputed tensors exceed memory.

Both modes use the same transform definitions. Factory functions handle
mode-specific setup (e.g., batched neutral model for precomputed, deep-copied
per-worker model for streaming).

Adding a new head type requires only a new transform list, output keys, and
collator specs — no new dataset class needed.

Heads supported:
    - SelectionFactorHead + WhichmutLoss: create_whichmut_dataset() from a
      DataFrame, create_whichmut_dataset_from_parquet() from a Parquet corpus.

That is the whole list. Upstream this module also built datasets for the DASM
and AAO loss families and for the DMS, HERMES, MPNN and structural heads;
viral-DASM trains none of them, so those factories are not in this fork. See
``PROVENANCE.md``.

Usage:
    from dasm2.data.evolutionary_dataset import create_whichmut_dataset

    # Precomputed (fast __getitem__, higher init cost + memory)
    bundle = create_whichmut_dataset(df, neutral_model, mode="precomputed")

    # Streaming (zero init cost + memory, transforms in __getitem__)
    bundle = create_whichmut_dataset(df, neutral_model, mode="streaming")

    loader = DataLoader(
        bundle.dataset,
        batch_size=256,
        collate_fn=bundle.collator,
        num_workers=8,
        worker_init_fn=bundle.worker_init_fn,
    )
"""

import copy
import logging
import math
import time
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from dasm2.constants import SMALL_PROBABILITY
from dasm2.data.arrow_strings import iter_selected_row_group_values
from dasm2.data.collators import ESM2_COLLATOR_SPECS, TensorScale, TransformCollator
from dasm2.data.neutral_rates_source import (
    NeutralRatesSource,
    parquet_carries_neutral_columns,
)
from dasm2.data.parquet_backend import (
    SEQUENTIAL_SWEEP_CACHE_SIZE,
    CacheBudgetSpec,
    ParquetBackend,
    ShuffleBufferSampler,
    make_parquet_row_converter,
    read_parquet_file_metadata,
)
from dasm2.data.protocols import SequenceProvider
from dasm2.data.registry import (
    _require_holdout_samples,
    determine_nickname_from_data_spec,
    is_parquet_data_spec,
    load_dataframe_with_split_markers,
    parse_multiname,
    registered_nickname_or_none,
    resolve_data_path,
    split_dataframe,
    verify_parquet_seam_stamp,
)
from dasm2.data.segment_handlers import (
    SegmentHandler,
    detect_compatible_handler,
)
from dasm2.data.transforms import (
    NT_SEPARATOR_LEN,
    SEPARATOR_NT_CSP_FILL,
    SEPARATOR_NT_RATE_FILL,
    CensorMaskedMutations,
    CensorReferenceReversions,
    CodonIndex,
    ComputeCodonMutationRates,
    ComputeLayout,
    ComputeMasks,
    ComputeMutIndicator,
    ComputeNeutralCodonProbs,
    ComputeNtRatesAndCsps,
    ComputeParentNtWindows,
    FusedTranslateTokenizeCodonIndex,
    NarrowLossMask,
    TranslateAndTokenize,
)
from dasm2.data.validation import (
    has_scored_substitution,
    scored_substitution_flags_from_arrow,
)
from dasm2.molevol.branch_length_init import (
    BranchLengthInitStrategy,
    PerSourceStrategy,
    deserialize_strategy,
)
from dasm2.molevol.codon_neutral_rates import (
    NUM_CODON_INDICES,
    compute_codon_mutation_rates,
)
from dasm2.sequence.constants import (
    AA_AMBIG_IDX,
    AMBIGUOUS_CODON_IDX,
    NT_SEPARATOR,
)
from dasm2.sequence.conversion import (
    fused_translate_tokenize_codon_index,
    translate_sequence,
)
from dasm2.sequence.layout import padding_sentinel_for
from dasm2.sequence.masking import CensorMode, CensorSpec
from dasm2.tokenizers import ESM2Tokenizer
from dasm2.utils.tensor import clamp_probability, tensors_to_numpy
from dasm2.utils.torch_utils import pin_single_thread

if TYPE_CHECKING:
    from dasm2.config import CvSplitConfig
    from dasm2.neutral.hit_class import HitClassModel

logger = logging.getLogger(__name__)

# Dedicated logger for on-the-fly CV split events so CLI entry points can
# route just these messages to stdout (via a _print handler) without surfacing
# every other logger.info call in this module. Library users keep the standard
# Python logging silence-by-default behavior.
cv_split_logger = logger.getChild("cv_split")

DEFAULT_BRANCH_LENGTH = 0.1
Mode = Literal["precomputed", "streaming"]


def warn_if_parquet_dataset_mode_mismatch(dataset_mode: str) -> None:
    """Warn only when ``dataset_mode`` is silently violated for Parquet data.

    "streaming" matches what Parquet actually does, so only the
    "precomputed" request is silently violated -- warning on "streaming"
    instead (the inverted pre-#501 condition) fired on the case that
    already matches reality and stayed silent on the one config knob a
    user would reach for to get precomputed behavior, which Parquet can't
    provide.

    Lives here rather than in ``train_command`` because the eval path
    (``dasm_evaluation.load_trainer_and_dataset``) takes the same knob and
    has the same Parquet blind spot (issue #988); importing it from
    ``train_command`` would be circular.

    Args:
        dataset_mode: The requested mode. ``stacklevel=3`` below assumes every
            caller is itself the function a user called with this knob — one
            frame for this function, one for the caller, landing the warning on
            the user's call. That holds for all three call sites
            (``train_command`` twice, ``dasm_evaluation.load_trainer_and_dataset``
            once); a caller that wraps it in a helper would need its own level.
    """
    if dataset_mode == "precomputed":
        warnings.warn(
            f"dataset_mode='{dataset_mode}' is ignored for Parquet data "
            f"(Parquet always uses streaming transforms).",
            stacklevel=3,
        )


#: Cap on worker processes for a parallel fetch over a dataset. Reached only
#: through :func:`max_fetch_worker_count`, which both fetch paths call —
#: :func:`fetch_worker_count` and
#: ``branch_lengths._calculate_optimal_worker_count`` (netam's
#: ``find_optimal_branch_lengths`` strategy) — so the two cannot disagree about
#: how much of a shared host one job may take.
MAX_FETCH_WORKERS = 10








def fork_worker_kwargs(num_workers: int) -> Dict[str, Any]:
    """``num_workers`` and, when positive, the ``fork`` start method.

    The one place that pins ``fork`` for a DataLoader (see
    :func:`fetch_dataloader_kwargs` for why). Callers pass ``worker_init_fn``
    themselves, normally ``make_worker_init_fn()``. ``multiprocessing_context``
    is omitted for 0 workers, since ``DataLoader`` rejects it without
    multiprocessing.
    """
    if num_workers == 0:
        return {"num_workers": 0}
    return {"num_workers": num_workers, "multiprocessing_context": "fork"}


# ---------------------------------------------------------------------------
# Fused translate + tokenize + codon index
# ---------------------------------------------------------------------------


def _precompute_sequence_tensors(
    raw_parents: List[str],
    raw_children: List[str],
) -> Tuple[
    List[str],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
]:
    """Fused translate+tokenize+codon_index for all parents and children.

    Replaces the separate ``translate_sequence`` loop in the factory +
    ``TranslateAndTokenize`` and ``CodonIndex`` transforms in the per-sample
    loop. One pass per sequence instead of three.

    Args:
        raw_parents: Parent NT sequences.
        raw_children: Child NT sequences.

    Returns:
        Tuple of (aa_parents, parent_aa_tokens_list, parent_codon_idxs_list,
        child_aa_tokens_list, child_codon_idxs_list).
    """
    aa_parents: List[str] = []
    parent_aa_tokens_list: List[torch.Tensor] = []
    parent_codon_idxs_list: List[torch.Tensor] = []

    for p in raw_parents:
        aa_str, aa_tok, codon_idx = fused_translate_tokenize_codon_index(p)
        aa_parents.append(aa_str)
        parent_aa_tokens_list.append(aa_tok)
        parent_codon_idxs_list.append(codon_idx)

    child_aa_tokens_list: List[torch.Tensor] = []
    child_codon_idxs_list: List[torch.Tensor] = []

    for c in raw_children:
        _, aa_tok, codon_idx = fused_translate_tokenize_codon_index(c)
        child_aa_tokens_list.append(aa_tok)
        child_codon_idxs_list.append(codon_idx)

    return (
        aa_parents,
        parent_aa_tokens_list,
        parent_codon_idxs_list,
        child_aa_tokens_list,
        child_codon_idxs_list,
    )


# ---------------------------------------------------------------------------
# Worker init
# ---------------------------------------------------------------------------


def make_worker_init_fn() -> Callable[[int], None]:
    """Return worker_init_fn for DataLoader workers.

    The workers run neutral model forward passes, so each is pinned to one
    torch thread to avoid oversubscription.

    Returns:
        Callable suitable for DataLoader(worker_init_fn=...).
    """
    return pin_single_thread


# ---------------------------------------------------------------------------
# Validation and sequence preparation
# ---------------------------------------------------------------------------


def _validate_codon_mutations_exist(
    raw_parents: List[str],
    raw_children: List[str],
) -> None:
    """Validate every sequence pair has at least one scored codon substitution.

    Uses :func:`dasm2.data.validation.has_scored_substitution`, the definition
    the Parquet converter filters on (issue #1243): a codon position where
    the parent is a sense codon over ``ACGT``, the child is over ``ACGT``, and
    they differ. Such a pair's branch length fits to zero, so it has no
    evolutionary signal.

    Args:
        raw_parents: List of parent NT sequences (may contain ^^^ separators).
        raw_children: List of child NT sequences (same length as raw_parents).

    Raises:
        ValueError: If any sequence pair has no scored substitution, or if
            parent/child lengths don't match.
    """
    zero_mutation_indices = []
    for idx, (parent, child) in enumerate(zip(raw_parents, raw_children)):
        if len(parent) != len(child):
            raise ValueError(
                f"Sequence {idx}: parent length ({len(parent)}) != "
                f"child length ({len(child)})"
            )
        if not has_scored_substitution(parent, child):
            zero_mutation_indices.append(idx)

    if zero_mutation_indices:
        raise ValueError(
            f"Sequences at indices {zero_mutation_indices} have no scored codon "
            f"substitution (a sense parent codon and an ACGT child codon that "
            f"differ). Remove these sequences before creating the dataset."
        )


def _validate_parquet_scored_substitutions(
    backend: ParquetBackend,
    segment_handler: SegmentHandler,
    parquet_paths: List[Path],
) -> None:
    """Parquet counterpart of :func:`_validate_codon_mutations_exist`.

    Streams the parent/child segment columns one row group at a time through
    :func:`dasm2.data.validation.scored_substitution_flags_from_arrow`,
    without materializing the NT strings (#992). Checks only the rows the
    backend's ``row_filter`` selects. A segment whose columns the files lack
    (a heavy-only file under an antibody handler) is absent, as in
    ``make_parquet_row_converter``.

    Raises:
        ValueError: If any selected row has no scored substitution. The
            converter drops these rows (issue #1243), so the file was built
            without it.
    """
    absent = segment_handler.validate_segment_columns(
        set(backend.column_names), ("parent", "child")
    )
    parent_names, child_names = (
        [name for name in segment_handler.get_column_names(base) if name not in absent]
        for base in ("parent", "child")
    )
    num_inert = sum(
        int(flags.size - np.count_nonzero(flags))
        for flags in iter_selected_row_group_values(
            backend, parent_names, child_names, scored_substitution_flags_from_arrow
        )
    )
    if num_inert:
        raise ValueError(
            f"{num_inert} of {backend.total_rows} selected rows in "
            f"{[str(path) for path in parquet_paths]} have no scored codon "
            f"substitution. Rebuild the file with "
            f"scripts/prepare_compara_parquet.py or "
            f"scripts/prepare_superfamily_parquet.py, which drop these rows "
            f"(issue #1243)."
        )




def _reject_context_model_on_csv(neutral_model: "RatesCSPNeutralModel") -> None:
    """Refuse a neutral model that needs per-row context on the CSV path.

    Only the Parquet loader carries a model's ``required_context`` columns
    into each sample (issue #1299). Without this check, a CSV streaming
    bundle would fail inside a DataLoader worker's first ``__getitem__``.

    Raises:
        ValueError: If ``neutral_model.required_context`` is non-empty.
    """
    if neutral_model.required_context:
        raise ValueError(
            f"Neutral model {neutral_model.name} needs per-row context "
            f"{list(neutral_model.required_context)}, which only the Parquet "
            "path supplies. Package the corpus as a rate-free Parquet file "
            "(--omit-neutral-columns), or use a context-free neutral model."
        )




def _prepare_pcp_sequences(
    sequences_df: pd.DataFrame,
    segment_handler: Optional[SegmentHandler],
) -> Tuple[SegmentHandler, List[str], List[str]]:
    """Detect handler, prepare sequences, extract raw parent/child NT strings.

    Args:
        sequences_df: DataFrame with segment-specific columns.
        segment_handler: Optional handler override. Auto-detected if None.

    Returns:
        Tuple of (handler, raw_parents, raw_children).
    """
    handler = segment_handler or detect_compatible_handler(
        set(sequences_df.columns), ("parent", "child")
    )
    prepared_df = handler.prepare_sequences(
        sequences_df, base_column_names=("parent", "child")
    )
    raw_parents = prepared_df["parent"].tolist()
    raw_children = prepared_df["child"].tolist()
    return handler, raw_parents, raw_children


# ---------------------------------------------------------------------------
# Neutral model precomputation
# ---------------------------------------------------------------------------


def _precompute_neutral_outputs_vectorized(
    raw_parents: List[str],
    neutral_model: "RatesCSPNeutralModel",
    segment_handler: SegmentHandler,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Precompute neutral model outputs using vectorized encoding (~100x faster).

    Collects all segments across all sequences, runs the neutral model in
    one batched call per segment index using ``compute_rates_and_csps_vectorized``,
    then reassembles per-sequence tensors with separator fill values and rate
    adjustments.

    Args:
        raw_parents: List of parent NT sequences (may contain ``^^^`` separators).
        neutral_model: Neutral mutation model.
        segment_handler: Handler defining segment suffixes and rate adjustments.

    Returns:
        Tuple of (nt_rates_list, nt_csps_list), each a list of tensors
        with one entry per sequence.
    """
    rate_adjustments = segment_handler.get_rate_adjustments()
    num_segments = segment_handler.get_num_segments()

    # Split all sequences into segments
    all_segments_per_seq = [p.split("^^^") for p in raw_parents]

    # Batch-compute per segment index using vectorized encoding
    seg_results: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [
        [] for _ in range(num_segments)
    ]
    for seg_idx in range(num_segments):
        segments_at_idx = [segs[seg_idx] for segs in all_segments_per_seq]
        with torch.no_grad():
            rates_list, csps_list = neutral_model.compute_rates_and_csps_vectorized(
                segments_at_idx
            )
        adj = rate_adjustments[seg_idx]
        for r, c in zip(rates_list, csps_list):
            seg_results[seg_idx].append((r * adj, c))

    # Reassemble per-sequence tensors with separator fill values
    nt_rates_list: List[torch.Tensor] = []
    nt_csps_list: List[torch.Tensor] = []
    for seq_idx in range(len(raw_parents)):
        all_rates: List[torch.Tensor] = []
        all_csps: List[torch.Tensor] = []
        for seg_idx in range(num_segments):
            if seg_idx > 0:
                all_rates.append(
                    torch.full((NT_SEPARATOR_LEN,), SEPARATOR_NT_RATE_FILL)
                )
                all_csps.append(
                    torch.full((NT_SEPARATOR_LEN, 4), SEPARATOR_NT_CSP_FILL)
                )
            rates, csps = seg_results[seg_idx][seq_idx]
            all_rates.append(rates)
            all_csps.append(csps)

        nt_rates_list.append(torch.cat(all_rates))
        nt_csps_list.append(torch.cat(all_csps))

    return nt_rates_list, nt_csps_list


# ---------------------------------------------------------------------------
# Unified Dataset
# ---------------------------------------------------------------------------


class TransformDataset(torch.utils.data.Dataset):
    """Transform-based dataset with precomputed and streaming modes.

    Single dataset class for all heads. In **precomputed** mode, all transform
    results are stored at init as ``List[Dict[str, Tensor]]``.
    ``__getitem__`` is a pure index lookup. In precomputed mode, ``output_keys``
    is not applied at ``__getitem__`` time — precomputed samples are already
    filtered by the factory that built them.

    In **streaming** mode, raw sequences and transforms are stored. Each
    ``__getitem__`` call builds a seed dict from raw data, runs the full
    transform chain, then strips keys not in ``output_keys``.

    Satisfies the SequenceProvider protocol (``aa_parents``, ``max_aa_seq_len``).

    Args:
        aa_parents: Pre-computed AA parent sequences for SequenceProvider protocol.
            If None, defaults to a copy of raw_parents (correct for AA-only datasets).
            For NT datasets, pass [translate_sequence(p) for p in raw_parents].
        nickname: Human-readable identifier.
        precomputed_samples: For precomputed mode — list of output dicts, one per
            sample. If provided, streaming-mode fields are ignored.
        raw_parents: For streaming mode — list of parent NT sequences.
        raw_children: For streaming mode — list of child NT sequences.
        transforms: For streaming mode — list of per-sample transforms.
        output_keys: Keys to retain in streaming ``__getitem__`` output. Required.
        extra_fields: Optional dict mapping field names to lists of float values
            (one per sequence). Injected as scalar tensors AFTER transforms in
            streaming mode. Used e.g. for DeltaAffinity target values.
        extra_tensor_fields: For streaming mode — pre-computed tensors injected
            BEFORE transforms (e.g., nt_rates/nt_csps from batched neutral model).
        extra_string_fields: For streaming mode — per-sample string values
            injected into the seed dict BEFORE transforms (e.g., an
            ``antigen_key`` that a downstream transform resolves to a
            pre-computed embedding tensor). Not stacked or tensorized.
        branch_lengths: Shared-memory branch lengths for DASM head. Used by both
            modes (streaming reads per-sample in ``__getitem__``, precomputed stores
            reference for ``recompute_neutral_probs``).
        codon_pref_log: Optional log-space codon preference vector (64,)
            (issue #747). Shared-memory, same live-update pattern as
            ``branch_lengths``: streaming/Parquet ``__getitem__`` injects it
            into the sample dict for ``ComputeNeutralCodonProbs`` to read;
            ``CodonStrengthOptimizer`` refits it in-place (issue #1039).
        parquet_backend: For Parquet streaming mode — reads rows from disk with
            row-group LRU cache. When present, replaces raw_parents/raw_children
            and extra_tensor_fields for per-sample data access.
        parquet_row_converter: Callable that converts a raw Parquet row dict
            (column name → Python value) into a seed sample dict for the
            transform chain. Must produce at least ``raw_parent`` and
            ``raw_child``, and should combine segment-specific columns
            (e.g. ``parent_heavy`` → ``raw_parent``) and precomputed
            neutral outputs (``nt_rates_heavy`` → ``nt_rates`` tensor)
            using the appropriate ``SegmentHandler``.
        max_aa_seq_len_override: Override for max_aa_seq_len (from Parquet metadata)
            when aa_parents are lazy-loaded and not available at init time.
    """

    def __init__(
        self,
        *,
        aa_parents: Optional[List[str]] = None,
        nickname: str = "unnamed",
        # Precomputed mode
        precomputed_samples: Optional[List[Dict[str, torch.Tensor]]] = None,
        # Streaming mode
        raw_parents: Optional[List[str]] = None,
        raw_children: Optional[List[str]] = None,
        transforms: Optional[List[Callable[[Dict[str, Any]], Dict[str, Any]]]] = None,
        output_keys: Optional[FrozenSet[str]] = None,
        extra_fields: Optional[Dict[str, List[float]]] = None,
        extra_tensor_fields: Optional[Dict[str, List[torch.Tensor]]] = None,
        extra_string_fields: Optional[Dict[str, List[str]]] = None,
        # Shared
        branch_lengths: Optional[torch.Tensor] = None,
        codon_pref_log: Optional[torch.Tensor] = None,
        # Parquet streaming mode
        parquet_backend: Optional["ParquetBackend"] = None,
        parquet_row_converter: Optional[
            Callable[[Dict[str, Any]], Dict[str, Any]]
        ] = None,
        max_aa_seq_len_override: Optional[int] = None,
    ) -> None:
        self.nickname = nickname
        self._branch_lengths = branch_lengths
        # Whether ``branch_lengths`` came from the input (an explicit tensor
        # or a ``branch_length`` column on every row) rather than
        # ``DEFAULT_BRANCH_LENGTH``. The evolutionary factories set it;
        # ``BranchLengthOptimizer(init_from_input=True)`` refuses a dataset
        # without it.
        self.branch_lengths_from_input = False
        # Codon-optimality log-preference vector (issue #747), shared-memory
        # so a between-cycle refit (CodonStrengthOptimizer) can update it
        # in-place and have streaming/Parquet workers see the change on their
        # next __getitem__ call — the same live-dataset-state pattern as
        # ``branch_lengths`` (DESIGN.md D6; verified by probe, issue #1039).
        self._codon_pref_log = (
            codon_pref_log.clone().share_memory_()
            if codon_pref_log is not None
            else None
        )

        # Precomputed mode
        self._precomputed = precomputed_samples

        # Streaming mode
        self._raw_parents = raw_parents
        self._raw_children = raw_children
        self._transforms = transforms
        self._output_keys = output_keys
        self._extra_fields: Dict[str, List[float]] = extra_fields or {}
        self._extra_tensor_fields = extra_tensor_fields or {}
        self._extra_string_fields: Dict[str, List[str]] = extra_string_fields or {}

        # Parquet streaming mode
        self._parquet_backend = parquet_backend
        self._parquet_row_converter = parquet_row_converter

        # Determine dataset size
        if precomputed_samples is not None:
            self._n = len(precomputed_samples)
        elif parquet_backend is not None:
            self._n = parquet_backend.total_rows
        elif raw_parents is not None:
            self._n = len(raw_parents)
        else:
            raise ValueError(
                "Must provide precomputed_samples, parquet_backend, or raw_parents"
            )

        # aa_parents: derive from raw_parents if not provided (for AA-only datasets).
        # For Parquet mode, aa_parents is populated lazily on first access.
        if aa_parents is not None:
            self._aa_parents: Optional[List[str]] = aa_parents
        elif raw_parents is not None:
            self._aa_parents = list(raw_parents)
        elif parquet_backend is not None:
            # Lazy: populated on first access via _load_aa_parents_from_parquet()
            self._aa_parents = None
        else:
            self._aa_parents = []

        # Validate field lengths match dataset size
        for field_key, field_values in self._extra_fields.items():
            if len(field_values) != self._n:
                raise ValueError(
                    f"extra_fields[{field_key!r}] has {len(field_values)} values "
                    f"but dataset has {self._n} sequences"
                )
        for field_key, field_strings in self._extra_string_fields.items():
            if len(field_strings) != self._n:
                raise ValueError(
                    f"extra_string_fields[{field_key!r}] has {len(field_strings)} "
                    f"values but dataset has {self._n} sequences"
                )
        for field_key, field_tensors in self._extra_tensor_fields.items():
            if len(field_tensors) != self._n:
                raise ValueError(
                    f"extra_tensor_fields[{field_key!r}] has {len(field_tensors)} "
                    f"tensors but dataset has {self._n} sequences"
                )
        if self._raw_children is not None and len(self._raw_children) != self._n:
            raise ValueError(
                f"raw_children has {len(self._raw_children)} sequences "
                f"but dataset has {self._n}"
            )

        if max_aa_seq_len_override is not None:
            self._max_aa_seq_len = max_aa_seq_len_override
        elif self._aa_parents:
            self._max_aa_seq_len = max(len(s) for s in self._aa_parents)
        elif parquet_backend is not None:
            # Read from Parquet file-level metadata if present, default to 0
            meta = parquet_backend.metadata
            self._max_aa_seq_len = int(meta.get("max_aa_seq_len", "0"))
        else:
            self._max_aa_seq_len = 0

    @property
    def row_filter_digest(self) -> Optional[str]:
        """The Parquet backend's ``row_filter_digest``; ``None`` without one."""
        if self._parquet_backend is None:
            return None
        return self._parquet_backend.row_filter_digest

    @property
    def is_precomputed(self) -> bool:
        """Whether this dataset is in precomputed mode."""
        return self._precomputed is not None

    @property
    def neutral_rates_source(self) -> NeutralRatesSource:
        """Where this dataset's per-site ``nt_rates``/``nt_csps`` come from.

        A Parquet-backed dataset without ``ComputeNtRatesAndCsps`` in its
        chain reads the files' columns. Every other dataset computes rates
        from its neutral model: a rate-free Parquet file, or CSV in either
        mode (precomputed mode runs the model up front).
        """
        computes = any(
            isinstance(t, ComputeNtRatesAndCsps) for t in self._transforms or []
        )
        if self._parquet_backend is None or computes:
            return NeutralRatesSource(kind="neutral_model")
        stamps = {
            meta["neutral_model_name"]
            for meta in self._parquet_backend.file_metadata
            if "neutral_model_name" in meta
        }
        return NeutralRatesSource(
            kind="parquet_columns", neutral_model_names=sorted(stamps)
        )

    def __len__(self) -> int:
        """Return number of sequences in the dataset."""
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single processed sample.

        Precomputed mode: pure index lookup (output_keys not applied — samples
        are already filtered by the factory).
        Streaming mode: build seed dict, run transforms, inject extra fields,
        strip to output_keys.
        Parquet mode: read row from disk via ParquetBackend, build seed dict,
        run transforms, strip to output_keys.

        Args:
            idx: Index of the sequence (0 to len-1).

        Returns:
            Dict of tensors ready for the collator.
        """
        if self._precomputed is not None:
            return self._precomputed[idx]

        if self._parquet_backend is not None:
            return self._getitem_parquet(idx)

        # Streaming mode: build seed and run transforms
        if self._raw_parents is None or self._transforms is None:
            raise RuntimeError(
                "Streaming __getitem__ requires raw_parents and transforms"
            )

        sample: Dict[str, Any] = {"raw_parent": self._raw_parents[idx]}
        if self._raw_children is not None:
            sample["raw_child"] = self._raw_children[idx]
        if self._branch_lengths is not None:
            sample["branch_length"] = self._branch_lengths[idx]
        if self._codon_pref_log is not None:
            sample["codon_pref_log"] = self._codon_pref_log

        # Inject extra string fields BEFORE transforms (transforms that
        # resolve keys like ``antigen_key`` → pre-computed tensors read them
        # from the seed dict).
        for key, strings in self._extra_string_fields.items():
            sample[key] = strings[idx]

        # Inject extra tensor fields BEFORE transforms
        for key, tensors in self._extra_tensor_fields.items():
            sample[key] = tensors[idx]

        # Run transform chain
        for t in self._transforms:
            sample = t(sample)

        # Inject extra scalar fields AFTER transforms
        for key, values in self._extra_fields.items():
            sample[key] = torch.tensor(values[idx], dtype=torch.float32)

        # Strip intermediate keys — only retain keys the collator expects
        if self._output_keys is None:
            raise RuntimeError(
                "output_keys is required in streaming mode. "
                "Pass output_keys to TransformDataset constructor."
            )
        return {k: v for k, v in sample.items() if k in self._output_keys}

    def _getitem_parquet(self, idx: int) -> Dict[str, torch.Tensor]:
        """Read a sample from Parquet backend and run transforms.

        Uses the ``parquet_row_converter`` to convert the raw Parquet row
        (with segment-specific columns) into a seed sample dict, then runs
        the transform chain.

        Args:
            idx: Global row index.

        Returns:
            Dict of tensors ready for the collator.
        """
        if self._parquet_backend is None:
            raise RuntimeError("_getitem_parquet called but parquet_backend is None")
        if self._transforms is None:
            raise RuntimeError("_getitem_parquet called but transforms is None")
        if self._parquet_row_converter is None:
            raise RuntimeError(
                "_getitem_parquet called but parquet_row_converter is None"
            )

        row = self._parquet_backend.get_row(idx)
        sample = self._parquet_row_converter(row)

        # Branch length from shared-memory tensor (updated by BranchLengthOptimizer)
        if self._branch_lengths is not None:
            sample["branch_length"] = self._branch_lengths[idx]
        # Codon preference from shared-memory tensor (updated by CodonStrengthOptimizer)
        if self._codon_pref_log is not None:
            sample["codon_pref_log"] = self._codon_pref_log

        # Run transform chain
        for t in self._transforms:
            sample = t(sample)

        # Inject extra scalar fields AFTER transforms (matches streaming path)
        for key, values in self._extra_fields.items():
            sample[key] = torch.tensor(values[idx], dtype=torch.float32)

        # Strip intermediate keys
        if self._output_keys is None:
            raise RuntimeError(
                "output_keys is required in Parquet mode. "
                "Pass output_keys to TransformDataset constructor."
            )
        return {k: v for k, v in sample.items() if k in self._output_keys}

    @property
    def aa_parents(self) -> List[str]:
        """Return amino acid parent sequences (SequenceProvider protocol).

        For Parquet mode, lazily loads all parent sequences from disk on first
        access. This is acceptable because ``aa_parents`` is only used during
        branch length optimization, which already requires a full data scan.
        """
        if self._aa_parents is None:
            self._aa_parents = self._load_aa_parents_from_parquet()
        return self._aa_parents

    def _load_aa_parents_from_parquet(self) -> List[str]:
        """Load and translate all parent sequences from Parquet backend.

        Uses column-oriented ``_read_combined_nt_sequences_from_parquet``
        for efficient bulk reading, then translates each to AA.
        Called lazily on first access to ``aa_parents``.

        Returns:
            List of AA parent sequences.
        """
        assert self._parquet_backend is not None
        logger.info(
            "Lazily loading aa_parents from Parquet (%d rows)...",
            self._parquet_backend.total_rows,
        )
        # Detect handler from column names
        handler = detect_compatible_handler(
            set(self._parquet_backend.column_names), ("parent", "child")
        )
        nt_parents = _read_combined_nt_sequences_from_parquet(
            self._parquet_backend, handler, "parent"
        )
        aa_parents = [translate_sequence(p) for p in nt_parents]
        logger.info("Loaded %d aa_parents from Parquet", len(aa_parents))
        return aa_parents

    @property
    def max_aa_seq_len(self) -> int:
        """Return maximum amino acid sequence length (SequenceProvider protocol)."""
        return self._max_aa_seq_len

    @property
    def branch_lengths(self) -> Optional[torch.Tensor]:
        """Branch lengths tensor (shared memory), or None if not set."""
        return self._branch_lengths

    @property
    def codon_pref_log(self) -> Optional[torch.Tensor]:
        """Codon preference log-vector (shared memory), or None if not set."""
        return self._codon_pref_log

    def set_codon_pref_log(self, new_codon_pref_log: torch.Tensor) -> None:
        """Establish or update the shared-memory codon preference tensor.

        Three cases, dispatched on current state:

        - No channel yet, precomputed dataset: establishes a new
          ``share_memory_()`` tensor (see "may establish the channel late"
          below).
        - No channel yet, streaming/Parquet dataset: raises — there is no
          shared tensor for a between-cycle refit to reach.
        - Channel already exists (either mode): updates it in place via
          ``copy_``.

        Always updates ``self._codon_pref_log`` — the single source of truth
        ``BranchLengthDatasetBundle.codon_pref_log`` reads through to (issue
        #1039; mirrors ``branch_lengths``, which is exactly the same
        read-through relationship). For streaming/Parquet mode, this alone
        makes the update visible: the per-sample ``ComputeNeutralCodonProbs``
        transform reads ``sample["codon_pref_log"]`` fresh on every
        ``__getitem__`` call (injected from ``self._codon_pref_log``), so an
        in-place update here reaches DataLoader workers on their next fetch —
        the same shared-memory live-dataset-state pattern
        ``update_branch_lengths``/``recompute_neutral_probs`` use for branch
        lengths (DESIGN.md D6). Verified by probe (issue #1039): a between-
        cycle mutation is immediately and exactly visible starting with the
        next cycle's DataLoader iterator (the actual usage pattern for
        ``CodonStrengthOptimizer.post_cycle``); only an in-flight mutation
        concurrent with an active iterator has a transient staleness window
        bounded by the prefetch queue depth, which does not apply here.

        For precomputed mode, this call alone does NOT refresh
        ``log_neutral_codon_probs`` — callers must also call
        ``update_branch_lengths``, which passes ``self.codon_pref_log``
        (already updated by this call) to ``recompute_neutral_probs`` to bake
        it into the cache. ``CodonStrengthOptimizer._rewrite_cache`` does both.

        Precomputed mode may establish the channel late (first call with no
        prior ``codon_pref_log``): nothing reads it via ``__getitem__``
        injection there, only through this property, so there is no worker-
        visibility requirement to enforce. Streaming/Parquet cannot: workers
        may already hold a reference to the (nonexistent) tensor, so late
        establishment would be silently invisible to them — raises instead.

        Raises:
            RuntimeError: If no codon preference channel was established at
                construction (``codon_pref_log=None`` was passed to the
                factory) on a streaming/Parquet dataset — there is no shared
                tensor for a between-cycle refit to reach.
        """
        if self._codon_pref_log is None:
            if self._precomputed is not None:
                self._codon_pref_log = new_codon_pref_log.clone().share_memory_()
                return
            raise RuntimeError(
                "Cannot update codon_pref_log: no shared-memory channel was "
                "established at dataset construction (codon_pref_log was "
                "None there). A between-cycle refit (fit_codon_strengths) "
                "requires an initial codon-optimality table to be configured "
                "so the shared tensor exists to update in place."
            )
        self._codon_pref_log.copy_(new_codon_pref_log)

    def recompute_neutral_probs(
        self,
        new_branch_lengths: torch.Tensor,
        nt_rates_list: List[torch.Tensor],
        nt_csps_list: List[torch.Tensor],
        multihit_model: Optional["HitClassModel"] = None,
        codon_pref_log: Optional[torch.Tensor] = None,
    ) -> None:
        """Update branch lengths in-place and recompute log_neutral_codon_probs.

        For precomputed mode, updates ``_branch_lengths`` shared-memory tensor
        and recomputes ``log_neutral_codon_probs`` for each precomputed sample,
        using ``codon_pref_log`` if given. For streaming mode, only updates
        ``_branch_lengths`` — transforms read the updated value on next
        ``__getitem__`` call. This method's ``codon_pref_log`` parameter is
        silently ignored for streaming mode — the per-sample
        ``ComputeNeutralCodonProbs`` transform reads its own live value from
        ``_codon_pref_log`` on every call, so there is nothing for this
        branch-length-triggered recompute to change. Use
        ``set_codon_pref_log`` to update the streaming/Parquet table
        in place (issue #1039). ``multihit_model`` below is ignored for
        streaming/Parquet for the same reason: the transform holds the very
        object the bundle exposes as ``multihit_model``, and its
        ``log_hit_class_factors`` are shared memory, so a refit mutates it in
        place rather than routing through here (issue #1041; see
        ``_prepare_transform_multihit``).

        Args:
            new_branch_lengths: New branch length values, shape (N,).
            nt_rates_list: Per-sequence neutral rates from vectorized precompute.
            nt_csps_list: Per-sequence neutral CSPs from vectorized precompute.
            multihit_model: Optional multihit correction model. Only applied
                in precomputed mode; ignored in streaming mode (see above).
            codon_pref_log: Optional log-space codon preference vector (64,)
                for codon-optimality adjustment (issue #747). Only applied in
                precomputed mode; ignored in streaming mode (see above).

        Raises:
            ValueError: If length doesn't match dataset size or non-finite values found.
            RuntimeError: If dataset has no branch lengths or no precomputed samples.
        """
        if len(new_branch_lengths) != self._n:
            raise ValueError(
                f"Branch length count mismatch: expected {self._n}, "
                f"got {len(new_branch_lengths)}"
            )
        if self._branch_lengths is None:
            raise RuntimeError("Dataset has no branch lengths to update")

        # Update shared-memory tensor
        self._branch_lengths.copy_(new_branch_lengths)

        # Precomputed mode: recompute log_neutral_codon_probs in-place
        if self._precomputed is not None:
            if len(nt_rates_list) != self._n:
                raise ValueError(
                    f"nt_rates_list has {len(nt_rates_list)} entries "
                    f"but dataset has {self._n}"
                )
            if len(nt_csps_list) != self._n:
                raise ValueError(
                    f"nt_csps_list has {len(nt_csps_list)} entries "
                    f"but dataset has {self._n}"
                )
            for i in range(self._n):
                sample = self._precomputed[i]
                bl = float(new_branch_lengths[i].item())

                codon_probs = neutral_codon_probs_with_padding(
                    sample["codon_parents_idxs"],
                    sample["loss_mask"],
                    nt_rates_list[i],
                    nt_csps_list[i],
                    bl,
                    multihit_model=multihit_model,
                    target_length=sample["codon_parents_idxs"].shape[0],
                    codon_pref_log=codon_pref_log,
                )
                # Validate + clamp before nan_to_num + log
                valid_mask = sample["loss_mask"][: codon_probs.shape[0]]
                if (
                    valid_mask.any()
                    and not torch.isfinite(codon_probs[valid_mask]).all()
                ):
                    raise ValueError(
                        f"Sample {i}: neutral_probs has non-finite values "
                        "at valid positions during branch length update"
                    )
                codon_probs[valid_mask] = clamp_probability(codon_probs[valid_mask])
                codon_probs = torch.nan_to_num(codon_probs, nan=SMALL_PROBABILITY)
                sample["log_neutral_codon_probs"] = torch.log(codon_probs)
                sample["branch_length"] = new_branch_lengths[i]


# ---------------------------------------------------------------------------
# Bundle types
# ---------------------------------------------------------------------------


def neutral_rates_source_of(dataset: Dataset) -> Optional[NeutralRatesSource]:
    """The rate source of a training dataset, looking through wrappers.

    Wrappers such as ``ESM2DatasetWrapper`` hold
    the ``TransformDataset`` they wrap as ``base_dataset``.

    Returns:
        The wrapped ``TransformDataset``'s ``neutral_rates_source``, or None
        for a dataset that consumes no per-site rates (e.g. MLM).
    """
    while not isinstance(dataset, TransformDataset) and hasattr(
        dataset, "base_dataset"
    ):
        dataset = dataset.base_dataset
    if isinstance(dataset, TransformDataset):
        return dataset.neutral_rates_source
    return None


class DatasetBundle:
    """Bundle for dataset + collator + worker_init_fn + optional sampler.

    Provides everything needed to construct a DataLoader in one object.

    Attributes:
        dataset: The unified dataset instance.
        collator: TransformCollator configured with matching tensor_specs.
        worker_init_fn: Callable to pass to DataLoader(worker_init_fn=...).
        sampler: Optional sampler (e.g. ``ShuffleBufferSampler`` for Parquet
            streaming). When present, replaces ``shuffle=True`` in the
            DataLoader. Existing bundles without a sampler behave unchanged.
    """

    def __init__(
        self,
        dataset: "torch.utils.data.Dataset",
        collator: TransformCollator,
        worker_init_fn: Callable[[int], None],
        sampler: Optional[Sampler[int]] = None,
    ) -> None:
        self.dataset = dataset
        self.collator = collator
        self.worker_init_fn = worker_init_fn
        self.sampler = sampler

    def _resolve_sampler_for_wrap(
        self, new_dataset: "torch.utils.data.Dataset"
    ) -> Optional[Sampler[int]]:
        """Resolve sampler when wrapping with a new dataset.

        Preserves the sampler only when the new dataset has the same index
        space as the original (e.g., a decorator/wrapper that delegates
        ``__len__`` and ``__getitem__`` unchanged). When the new dataset is a
        ``Subset`` (different index space), the sampler is dropped because its
        global indices would bypass the subset's remapping.
        """
        return None if isinstance(new_dataset, Subset) else self.sampler

    def with_esm2_collator(self) -> "DatasetBundle":
        """Return a new bundle whose collator includes ESM2 token_ids/masks specs.

        Creates a new ``TransformCollator`` that merges the existing tensor
        specs with the ESM2-specific ``token_ids`` and ``masks`` specs, so
        batches correctly pad and collate ESM2 tokenization outputs.
        """
        esm2_collator = TransformCollator(
            tensor_specs={**self.collator.tensor_specs, **ESM2_COLLATOR_SPECS}
        )
        return DatasetBundle(
            self.dataset,
            esm2_collator,
            self.worker_init_fn,
            self.sampler,
        )

    def with_wrapped_dataset(
        self, new_dataset: "torch.utils.data.Dataset"
    ) -> "DatasetBundle":
        """Return a new bundle with a different dataset, keeping collator/worker_init."""
        return DatasetBundle(
            new_dataset,
            self.collator,
            self.worker_init_fn,
            self._resolve_sampler_for_wrap(new_dataset),
        )


class _ESM2BLDataset(Dataset):
    """Thin wrapper adding ESM2 tokenization to a BL optimization dataset.

    Reads the parent AA sequence from the base dataset's ``aa_parents``
    property (matching ``ESM2DatasetWrapper``'s approach) and tokenizes
    with the ESM2 tokenizer to produce ``token_ids`` and ``masks``.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        esm2_tokenizer: ESM2Tokenizer,
    ) -> None:
        self._base = base_dataset
        self._esm2_tokenizer = esm2_tokenizer

    def __len__(self) -> int:
        return len(self._base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = self._base[idx]

        # Read sequence from base dataset's aa_parents (same approach as
        # ESM2DatasetWrapper), avoiding the detokenize roundtrip which
        # silently drops ambiguous tokens.
        if not isinstance(self._base, SequenceProvider):
            raise TypeError(
                f"base_dataset must implement SequenceProvider protocol "
                f"(have aa_parents property). Got {type(self._base).__name__}"
            )
        sequence = self._base.aa_parents[idx]

        # Tokenize with ESM2
        esm2_batch = self._esm2_tokenizer.create_encoder_batch([sequence])
        item["token_ids"] = esm2_batch["token_ids"].squeeze(0)
        item["masks"] = esm2_batch["masks"].squeeze(0)
        return item


class BranchLengthDatasetBundle(DatasetBundle):
    """Extended bundle with branch length optimization support.

    Handles BL updates for both modes:
    - **Streaming**: writes to shared-memory tensor; transforms read updated
      value on next ``__getitem__`` call.
    - **Precomputed**: writes to shared-memory tensor AND recomputes
      ``log_neutral_codon_probs`` for all samples using stored intermediates.

    Attributes:
        neutral_model: Reference to neutral model (for BL optimization).
        multihit_model: Reference to multihit model (for BL optimization).
        segment_handler: Reference to segment handler (for BL optimization).
    """

    dataset: TransformDataset  # Narrow type from parent's Dataset

    def __init__(
        self,
        dataset: TransformDataset,
        collator: TransformCollator,
        worker_init_fn: Callable[[int], None],
        neutral_model: Optional["RatesCSPNeutralModel"],
        segment_handler: SegmentHandler,
        multihit_model: Optional["HitClassModel"] = None,
        nt_rates_list: Optional[List[torch.Tensor]] = None,
        nt_csps_list: Optional[List[torch.Tensor]] = None,
        nt_parents: Optional[List[str]] = None,
        nt_children: Optional[List[str]] = None,
        sampler: Optional[Sampler[int]] = None,
        edge_features: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        super().__init__(dataset, collator, worker_init_fn, sampler=sampler)
        self.neutral_model = neutral_model
        self.multihit_model = multihit_model
        self.segment_handler = segment_handler
        self._nt_rates_list = nt_rates_list
        self._nt_csps_list = nt_csps_list
        # Raw NT sequences for mutation-frequency branch length initialization.
        # None means "lazy-load from Parquet on first access".
        self._nt_parents = nt_parents
        self._nt_children = nt_children
        # Per-edge metadata columns for BL init strategies (e.g., child_is_leaf).
        # Eagerly loaded at construction time — small metadata, not large tensors.
        self.edge_features: Dict[str, List[Any]] = edge_features or {}

    @property
    def nickname(self) -> str:
        """Delegate to underlying dataset for CacheableDataset protocol."""
        return self.dataset.nickname

    @property
    def row_filter_digest(self) -> Optional[str]:
        """Delegate to underlying dataset for CacheableDataset protocol."""
        return self.dataset.row_filter_digest

    def __len__(self) -> int:
        """Delegate to underlying dataset for CacheableDataset protocol."""
        return len(self.dataset)

    @property
    def nt_sequences_are_loaded(self) -> bool:
        """Whether ``nt_parents``/``nt_children`` are already materialized.

        False means reading either one triggers a full column read from
        Parquet. Cold-start branch length initialization uses this to decide
        between its columnar Arrow path and the per-edge string path (#992):
        when the strings are already in hand there is nothing to gain by
        re-reading the columns.
        """
        return self._nt_parents is not None and self._nt_children is not None

    @property
    def nt_parents(self) -> List[str]:
        """NT parent sequences, lazy-loaded from Parquet on first access."""
        if self._nt_parents is None:
            backend = self.parquet_backend
            if backend is None:
                raise RuntimeError(
                    "No nt_parents provided and no ParquetBackend for lazy loading"
                )
            self._nt_parents = _read_combined_nt_sequences_from_parquet(
                backend, self.segment_handler, "parent"
            )
        return self._nt_parents

    @property
    def nt_children(self) -> List[str]:
        """NT child sequences, lazy-loaded from Parquet on first access."""
        if self._nt_children is None:
            backend = self.parquet_backend
            if backend is None:
                raise RuntimeError(
                    "No nt_children provided and no ParquetBackend for lazy loading"
                )
            self._nt_children = _read_combined_nt_sequences_from_parquet(
                backend, self.segment_handler, "child"
            )
        return self._nt_children

    def update_branch_lengths(self, new_branch_lengths: torch.Tensor) -> None:
        """Update branch lengths and recompute neutral probs if precomputed.

        Delegates to ``TransformDataset.recompute_neutral_probs()``.
        For streaming mode, workers see updated values on next ``__getitem__``.
        For precomputed mode, ``log_neutral_codon_probs`` is recomputed in-place,
        using ``self.codon_pref_log`` if set. That table is only passed through
        for precomputed mode — streaming/Parquet already has a live view via
        ``TransformDataset.set_codon_pref_log`` and does not need (or use)
        it here.

        Args:
            new_branch_lengths: New branch length values, shape (N,).

        Raises:
            ValueError: If length doesn't match dataset size.
            RuntimeError: If nt_rates/nt_csps not available for precomputed recomputation.
        """
        if self.dataset.is_precomputed:
            if self._nt_rates_list is None or self._nt_csps_list is None:
                raise RuntimeError(
                    "Missing nt_rates_list/nt_csps_list for branch length update"
                )
        self.dataset.recompute_neutral_probs(
            new_branch_lengths,
            nt_rates_list=self._nt_rates_list or [],
            nt_csps_list=self._nt_csps_list or [],
            multihit_model=self.multihit_model,
            codon_pref_log=self.codon_pref_log if self.dataset.is_precomputed else None,
        )

    def with_wrapped_dataset(
        self, new_dataset: "torch.utils.data.Dataset"
    ) -> "BranchLengthDatasetBundle":
        """Return a new bundle with a different dataset, preserving all BL state.

        The new dataset may be a wrapper (e.g., ``ESM2DatasetWrapper``) that
        delegates attribute access to the underlying ``TransformDataset`` via
        ``__getattr__``. BL methods like ``recompute_neutral_probs`` and
        ``is_precomputed`` are accessed through this delegation chain.
        """
        return BranchLengthDatasetBundle(
            dataset=new_dataset,  # type: ignore[arg-type]
            collator=self.collator,
            worker_init_fn=self.worker_init_fn,
            neutral_model=self.neutral_model,
            segment_handler=self.segment_handler,
            multihit_model=self.multihit_model,
            nt_rates_list=self._nt_rates_list,
            nt_csps_list=self._nt_csps_list,
            nt_parents=self._nt_parents,
            nt_children=self._nt_children,
            sampler=self._resolve_sampler_for_wrap(new_dataset),
            edge_features=self.edge_features,
        )

    def with_esm2_collator(self) -> "BranchLengthDatasetBundle":
        """Return a new bundle whose collator includes ESM2 specs, preserving BL state."""
        esm2_collator = TransformCollator(
            tensor_specs={**self.collator.tensor_specs, **ESM2_COLLATOR_SPECS}
        )
        return BranchLengthDatasetBundle(
            dataset=self.dataset,
            collator=esm2_collator,
            worker_init_fn=self.worker_init_fn,
            neutral_model=self.neutral_model,
            segment_handler=self.segment_handler,
            multihit_model=self.multihit_model,
            nt_rates_list=self._nt_rates_list,
            nt_csps_list=self._nt_csps_list,
            nt_parents=self._nt_parents,
            nt_children=self._nt_children,
            sampler=self.sampler,
            edge_features=self.edge_features,
        )

    @property
    def branch_lengths(self) -> torch.Tensor:
        """Current branch lengths (shared memory tensor)."""
        bl = self.dataset.branch_lengths
        if bl is None:
            raise RuntimeError("Dataset has no branch lengths")
        return bl

    @property
    def branch_lengths_from_input(self) -> bool:
        """Whether the initial branch lengths came from the input.

        Read-through to the dataset, which may be a wrapper such as
        ``ESM2DatasetWrapper`` that delegates attribute access.
        """
        return bool(self.dataset.branch_lengths_from_input)

    @property
    def codon_pref_log(self) -> Optional[torch.Tensor]:
        """Codon preference log-vector (issue #747), or None if not configured.

        Read-through to ``self.dataset.codon_pref_log`` — structurally the
        same single-source-of-truth relationship as ``branch_lengths`` above,
        so the branch-length objective's duck-typed
        ``getattr(bundle, "codon_pref_log", None)`` lookup
        (``branch_lengths.py``) and the loss's per-sample transform always
        see the same tensor, not two independently-writable copies that could
        drift (issue #1039).
        """
        return self.dataset.codon_pref_log

    def _resolve_nt_rates_and_csps(
        self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get per-sample nt_rates/nt_csps, from whichever source owns them.

        Three sources, in the order they are preferred:

        1. **Stored** (precomputed mode) — already in hand.
        2. **The Parquet bundle's own chain**, streamed via
           ``create_bl_optimization_dataloader``. That chain is the training
           forward pass's: it reads the file's columns when it has them, and
           runs ``ComputeNtRatesAndCsps`` with the configured model on a
           rate-free file (issue #1299). Preferred over recomputing here,
           because stored columns need not match any configured model —
           Compara's v10 columns hold per-family GTR rates that no single
           model reproduces — so recomputing would fit branch lengths and
           hit-class factors against rates the model was never trained on.
           Streaming columns is also what lets a ``PrecomputedRates`` carrier
           (multihit correction only, no rate model) work here (issue #1041).
        3. **The neutral model**, recomputed from raw parents. The in-memory
           (CSV) path, where there are no stored rates and no Parquet columns.

        Returns:
            Tuple of (nt_rates_list, nt_csps_list) in dataset index order.

        Raises:
            RuntimeError: If none of the three sources is available.
        """
        if self._nt_rates_list is not None and self._nt_csps_list is not None:
            return self._nt_rates_list, self._nt_csps_list

        # Sequential, num_workers=0, shuffle=False — batch order is index
        # order, matching ``branch_lengths[i]`` and ``self.dataset[i]``.
        bl_loader = self.create_bl_optimization_dataloader()
        if bl_loader is not None:
            nt_rates_list: List[torch.Tensor] = []
            nt_csps_list: List[torch.Tensor] = []
            for batch in bl_loader:
                nt_rates_list.extend(batch["nt_rates"])
                nt_csps_list.extend(batch["nt_csps"])
            return nt_rates_list, nt_csps_list

        if self.neutral_model is None or not self.nt_parents:
            raise RuntimeError(
                "Cannot compute BL optimization inputs: no stored "
                "nt_rates/nt_csps, no Parquet backend to stream them from, "
                "and no neutral_model + raw sequences available for "
                "on-the-fly computation."
            )
        return _precompute_neutral_outputs_vectorized(
            self.nt_parents, self.neutral_model, self.segment_handler
        )

    def get_branch_length_optimization_inputs(self) -> Dict[str, Any]:
        """Return per-sequence inputs for branch length optimization.

        Collects unpadded per-sample tensors from the dataset and wraps
        them in ``PerSequenceBLInputs`` objects. No padding or stacking
        is performed — each sample stores its actual-length tensors.

        ``nt_rates``/``nt_csps`` come from ``_resolve_nt_rates_and_csps``;
        see there for why a Parquet-backed bundle reads its own columns rather
        than recomputing from a configured neutral model.

        Returns:
            Dict with keys: ``samples`` (list of ``PerSequenceBLInputs``),
            ``multihit_model``.

        Raises:
            RuntimeError: If nt_rates/nt_csps are not available and cannot
                be obtained (not Parquet-backed, and no neutral model or raw
                sequences for on-the-fly computation).
        """

        nt_rates_list, nt_csps_list = self._resolve_nt_rates_and_csps()

        n = len(self.dataset)
        branch_lengths = self.branch_lengths

        samples: List[PerSequenceBLInputs] = []
        for i in range(n):
            sample = self.dataset[i]
            samples.append(
                PerSequenceBLInputs(
                    mask=sample["loss_mask"],
                    codon_children_idxs=sample["codon_children_idxs"],
                    codon_parents_idxs=sample["codon_parents_idxs"],
                    layout=sample["layout"],
                    branch_length=float(branch_lengths[i].item()),
                    nt_rates=nt_rates_list[i],
                    nt_csps=nt_csps_list[i],
                )
            )

        return {
            "samples": samples,
            "multihit_model": self.multihit_model,
        }

    @property
    def parquet_backend(self) -> Optional["ParquetBackend"]:
        """The underlying dataset's Parquet backend, or None if in-memory.

        ``self.dataset`` may be a wrapper (e.g. ``ESM2DatasetWrapper``) that
        forwards unknown attributes to the ``TransformDataset`` it wraps, so
        this reads through whatever is there rather than assuming the type.
        """
        backend: Optional["ParquetBackend"] = getattr(
            self.dataset, "_parquet_backend", None
        )
        return backend

    @property
    def is_parquet_backed(self) -> bool:
        """Whether the underlying dataset reads from Parquet."""
        return self.parquet_backend is not None

    def create_bl_optimization_dataloader(
        self, batch_size: int = 32, num_workers: int = 0
    ) -> Optional[DataLoader]:
        """Create a DataLoader for streaming BL optimization inputs from Parquet.

        Returns ``None`` if the dataset is not Parquet-backed (caller should
        fall back to ``get_branch_length_optimization_inputs`` for in-memory
        data).

        The DataLoader uses sequential ordering (matching ``branch_lengths[i]``)
        and an expanded ``output_keys`` that includes ``nt_rates``/``nt_csps``
        as list-collated fields.  With ``num_workers=0`` (the default) the
        caller's process reads the rows, sharing the bundle backend's LRU
        cache.  With ``num_workers > 0`` the rows are read and transformed in
        forked workers, each through its own
        ``ParquetBackend.sequential_sweep_copy``: the serial read was the cap
        on post-cycle BL throughput (issue #1276), and a sweep needs only a
        few row groups per worker.  Batches still arrive in index order, but
        with numpy arrays in place of tensors; pass each through
        :func:`bl_batch_to_torch`.

        When the bundle's dataset is an ``ESM2DatasetWrapper``, the returned
        DataLoader also includes ESM2-tokenized ``token_ids`` and ``masks``
        fields required by ``ESM2CoreModel.forward()``.

        Args:
            batch_size: Batch size for model forward pass.
            num_workers: DataLoader worker processes (0: read in the caller).

        Returns:
            DataLoader yielding batches with ``nt_rates``/``nt_csps`` as
            ``List[Tensor]``, or ``None`` if not Parquet-backed.
        """
        # Reuse the existing backend so its LRU row-group cache (and the
        # cache_size chosen at construction time) is shared with the BL
        # dataloader. Constructing a fresh ParquetBackend here would
        # re-auto-size the cache and re-introduce the residual memory
        # that callers like oe-plot explicitly lowered via cache_size.
        backend = self.parquet_backend
        if backend is None:
            return None
        if num_workers > 0:
            backend = backend.sequential_sweep_copy()

        num_segments = self.segment_handler.get_num_segments()

        # Build a lightweight TransformDataset sharing the same backend,
        # converter, and transforms, but with BL output keys. It drops
        # ComputeNeutralCodonProbs: the BL objective recomputes neutral probs
        # from nt_rates/nt_csps at each trial length, and nothing reads
        # log_neutral_codon_probs from these batches, yet it cost ~36% of the
        # serial loader's time on v10 (issue #1276). Keep this an exclusion of
        # that one type, not an allow-list: every other transform, including a
        # ComputeNtRatesAndCsps for rate-free files (#1266), must stay in the
        # BL chain, since the BL objective needs nt_rates/nt_csps.
        # If the training dataset includes "distance", carry it through so the
        # model's head can access it during BL optimization forward passes
        # (issue #199).
        output_keys = BL_OPTIMIZATION_OUTPUT_KEYS
        has_distance = (
            self.dataset._output_keys is not None
            and "distance" in self.dataset._output_keys
        )
        if has_distance:
            output_keys = output_keys | {"distance"}

        transform_dataset = TransformDataset(
            parquet_backend=backend,
            parquet_row_converter=self.dataset._parquet_row_converter,
            transforms=[
                t
                for t in self.dataset._transforms or []
                if not isinstance(t, ComputeNeutralCodonProbs)
            ],
            output_keys=output_keys,
            branch_lengths=self.dataset._branch_lengths,
            codon_pref_log=self.dataset.codon_pref_log,
            max_aa_seq_len_override=self.dataset._max_aa_seq_len,
            nickname=self.dataset.nickname,
            extra_fields=self.dataset._extra_fields or None,
        )

        bl_dataset: torch.utils.data.Dataset = transform_dataset

        padding_sentinel = padding_sentinel_for(num_segments)
        # Detect from the dataset's transform list whether ComputeParentNtWindows
        # is active so the collator's tensor_specs match what __getitem__ produces.
        include_parent_nt_windows = any(
            isinstance(t, ComputeParentNtWindows)
            for t in (self.dataset._transforms or [])
        )
        collator_specs = dict(
            _bl_optimization_collator_specs(
                padding_sentinel, include_parent_nt_windows=include_parent_nt_windows
            )
        )
        # Issue #199: carry the per-sample distance scalar through BL optimization
        # when the training dataset exposes it.
        if has_distance:
            collator_specs["distance"] = (0.0, "scalar")

        # If the bundle wraps an ESM2DatasetWrapper, augment the BL dataset
        # with ESM2 tokenization so model.forward() gets token_ids/masks.
        if isinstance(self.dataset, ESM2DatasetWrapper):
            if num_workers > 0:
                # Load the corpus-wide parents once here, so forked workers
                # inherit them rather than each re-reading every row.
                _ = transform_dataset.aa_parents
            bl_dataset = _ESM2BLDataset(bl_dataset, self.dataset.tokenizer)
            collator_specs.update(ESM2_COLLATOR_SPECS)

        bl_collator: Callable[[List[Dict[str, Any]]], Dict[str, Any]] = (
            TransformCollator(tensor_specs=collator_specs)
        )
        if num_workers > 0:
            bl_collator = _NumpyBatchCollator(bl_collator)

        # One torch thread per worker beside the BL scipy pool (issue #1276).
        return DataLoader(
            bl_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=bl_collator,
            worker_init_fn=make_worker_init_fn(),
            **fork_worker_kwargs(num_workers),
        )


class _NumpyBatchCollator:
    """Collate wrapper that hands a batch's tensors to the parent as numpy.

    A DataLoader worker sends each tensor to the parent through its own
    shared-memory file descriptor, one socket round trip apiece. A BL batch
    carries two list-collated tensors per row (``nt_rates``, ``nt_csps``), so
    32-row batches in flight from a few workers exhausted a 1024-fd limit
    ("received 0 items of ancdata"), and the hand-offs capped the parent's
    receive rate (issue #1276). Numpy arrays are pickled by value instead, as
    ``PerSequenceBLInputs`` does for the spawn pool. The parent restores the
    batch with :func:`bl_batch_to_torch`.
    """

    def __init__(self, collate_fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]]):
        self._collate_fn = collate_fn

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self._collate_fn(samples)
        return {key: tensors_to_numpy(value) for key, value in batch.items()}




# ---------------------------------------------------------------------------
# Factory: Whichmut
# ---------------------------------------------------------------------------


def _check_corpus_clades_covered(
    clades: Iterable[str],
    censor_table: Dict[str, CensorSpec],
    source: str,
) -> None:
    """Fail at construction if the corpus names a clade the table does not.

    Checked here, in the parent process, rather than left to the per-sample
    transform: a clade appearing only in the tail of a large corpus would
    otherwise surface as a worker-side exception an hour into a multi-hour run.

    Only this direction is fatal. A table clade absent from the corpus is
    legitimate and common -- the table is built over the whole tree while a
    training corpus may be filtered -- so it is not an error.

    Raises:
        ValueError: If any corpus clade is missing from the table.
    """
    unknown = sorted({c for c in clades if c not in censor_table})
    if unknown:
        raise ValueError(
            f"{source}: clade(s) {unknown} appear in the corpus but not in the "
            f"censor table, so whether they censor anything is unknown. A clade "
            f"that censors nothing is present with an empty spec, so this is a "
            f"table/corpus mismatch -- most likely the table was built from a "
            f"different tree. Table covers {len(censor_table)} clades: "
            f"{sorted(censor_table)}."
        )


def _whichmut_transforms(
    num_segments: int,
    censor_table: Optional[Dict[str, "CensorSpec"]] = None,
    censor_mode: CensorMode = "both",
    reference_base_idxs: Optional[torch.Tensor] = None,
) -> List[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """Build per-sample transform chain for Whichmut head.

    These transforms are shared between precomputed and streaming modes.
    They assume nt_rates and nt_csps are already in the sample dict
    (injected by batched precomputation or ComputeNtRatesAndCsps transform).

    Args:
        num_segments: Number of protein segments.
        censor_table: Optional ``{clade: CensorSpec}`` enabling UShER censoring
            (#1079). Appended last, since it must run after *both*
            ``ComputeCodonMutationRates`` and ``NarrowLossMask``.
        censor_mode: Which half of censoring to apply; shared by both censoring
            transforms. See
            :class:`~dasm2.data.transforms.CensorMaskedMutations`.
        reference_base_idxs: Optional reference base indices enabling
            parent -> reference reversion censoring (#1099). Appended after the
            table's transform; order does not matter, since the two compose.

    Returns:
        List of per-sample transforms.

    Note:
        The appended censor transforms are correct for the **streaming and
        Parquet** paths, which run this whole chain per sample. The
        **precomputed** path splits ``ComputeCodonMutationRates`` out into a
        later batched pass, so ``codon_neutral_rates`` does not exist while this
        chain runs and censoring is applied as a separate pass instead — see
        :func:`_build_precomputed_dataset`.
    """
    transforms: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = [
        FusedTranslateTokenizeCodonIndex(
            "raw_parent", "aa_parents_idxs", "codon_parents_idxs"
        ),
        FusedTranslateTokenizeCodonIndex(
            "raw_child", "aa_children_idxs", "codon_children_idxs"
        ),
        ComputeLayout(num_segments),
        ComputeMasks(
            num_segments,
            exclude_parent_codon_ambiguity=True,
            exclude_child_codon_ambiguity=True,
        ),
        ComputeCodonMutationRates(),
        ComputeMutIndicator(
            "codon_parents_idxs",
            "codon_children_idxs",
            "codon_mut_indicator",
            AMBIGUOUS_CODON_IDX,
            single_nt_only=True,
        ),
        NarrowLossMask("codon_mut_indicator"),
    ]
    if censor_table is not None:
        transforms.append(
            CensorMaskedMutations(
                censor_table=censor_table,
                num_segments=num_segments,
                mode=censor_mode,
            )
        )
    if reference_base_idxs is not None:
        transforms.append(
            CensorReferenceReversions(
                reference_base_idxs=reference_base_idxs,
                num_segments=num_segments,
                mode=censor_mode,
            )
        )
    return transforms


WHICHMUT_OUTPUT_KEYS = frozenset(
    {
        "aa_parents_idxs",
        "aa_children_idxs",
        "codon_parents_idxs",
        "codon_children_idxs",
        "codon_neutral_rates",
        "codon_mut_indicator",
        "layout",
        "mask",
        "loss_mask",
    }
)


def _whichmut_collator_specs(
    padding_sentinel: int,
) -> Dict[str, Tuple[float, TensorScale]]:
    """Build collator tensor_specs for Whichmut head."""
    return {
        "aa_parents_idxs": (AA_AMBIG_IDX, "codon"),
        "aa_children_idxs": (AA_AMBIG_IDX, "codon"),
        "codon_parents_idxs": (AMBIGUOUS_CODON_IDX, "codon"),
        "codon_children_idxs": (AMBIGUOUS_CODON_IDX, "codon"),
        "codon_neutral_rates": (0.0, "codon"),
        "codon_mut_indicator": (False, "codon"),
        "layout": (padding_sentinel, "codon"),
        "mask": (False, "codon"),
        "loss_mask": (False, "codon"),
    }


def create_whichmut_dataset(
    sequences_df: pd.DataFrame,
    neutral_model: "RatesCSPNeutralModel",
    segment_handler: Optional[SegmentHandler] = None,
    mode: Mode = "precomputed",
    nickname: str = "unnamed_whichmut",
    censor_table: Optional[Dict[str, CensorSpec]] = None,
    censor_mode: CensorMode = "both",
    reference_base_idxs: Optional[torch.Tensor] = None,
) -> DatasetBundle:
    """Create dataset for WhichmutHead / WhichmutResidualHead / PerSiteChildHead.

    Same transform definitions for both modes. In precomputed mode, the neutral
    model runs batched at init. In streaming mode, neutral model outputs are
    computed per-sample in ``__getitem__``.

    Args:
        sequences_df: DataFrame with parent/child NT sequence columns.
        neutral_model: Neutral mutation model.
        segment_handler: Handler for multi-segment sequences. Auto-detected if None.
        mode: ``"precomputed"`` for init-time computation (fast __getitem__),
            ``"streaming"`` for per-sample computation (zero memory overhead).
        nickname: Human-readable identifier.
        censor_table: Optional ``{clade: CensorSpec}`` enabling UShER censoring
            (#1079). Requires a ``clade`` column in ``sequences_df``.
        censor_mode: Which half of censoring to apply, for both sources.
        reference_base_idxs: Optional reference base indices enabling
            parent -> reference reversion censoring (#1099). Needs **no**
            ``clade`` column: the rule is clade-independent, which is why it is
            usable on its own.

    Returns:
        DatasetBundle with dataset, collator, and worker_init_fn.

    Raises:
        ValueError: If sequences have no codon mutations, or if ``censor_table``
            is given without a ``clade`` column.
    """
    clades: Optional[List[str]] = None
    if censor_table is not None:
        if "clade" not in sequences_df.columns:
            raise ValueError(
                "censor_table was given but the corpus has no 'clade' column, so "
                "there is no way to know which clade's mask applies to each PCP. "
                f"Columns: {sorted(sequences_df.columns)}."
            )
        clades = sequences_df["clade"].astype(str).tolist()
        _check_corpus_clades_covered(set(clades), censor_table, nickname)
    _reject_context_model_on_csv(neutral_model)
    t_factory_start = time.perf_counter()

    t0 = time.perf_counter()
    handler, raw_parents, raw_children = _prepare_pcp_sequences(
        sequences_df, segment_handler
    )
    _validate_codon_mutations_exist(raw_parents, raw_children)
    logger.info(
        "Whichmut prepare+validate: %.1fs (%d sequences)",
        time.perf_counter() - t0,
        len(raw_parents),
    )

    num_seg = handler.get_num_segments()
    # Precomputed mode applies censoring as a separate pass (the chain runs
    # before codon_neutral_rates exists), so the transform is only built into
    # the chain for streaming.
    streaming = mode != "precomputed"
    transforms = _whichmut_transforms(
        num_seg,
        censor_table=censor_table if streaming else None,
        censor_mode=censor_mode,
        reference_base_idxs=reference_base_idxs if streaming else None,
    )
    # Precomputed mode gets the same transforms as pass 4 instead. Both sources
    # are gated on `streaming` above and rebuilt here, so a source added to one
    # branch and not the other trains uncensored in exactly one dataset mode --
    # which is why the bit-exactness test is parameterized over all three.
    censor_transforms: List[
        Union[CensorMaskedMutations, CensorReferenceReversions]
    ] = []
    if not streaming:
        if censor_table is not None:
            censor_transforms.append(
                CensorMaskedMutations(
                    censor_table=censor_table, num_segments=num_seg, mode=censor_mode
                )
            )
        if reference_base_idxs is not None:
            censor_transforms.append(
                CensorReferenceReversions(
                    reference_base_idxs=reference_base_idxs,
                    num_segments=num_seg,
                    mode=censor_mode,
                )
            )

    if mode == "precomputed":
        # Fused translate+tokenize+codon_index: single pass per sequence
        t0 = time.perf_counter()
        (
            aa_parents,
            parent_aa_tokens,
            parent_codon_idxs,
            child_aa_tokens,
            child_codon_idxs,
        ) = _precompute_sequence_tensors(raw_parents, raw_children)
        logger.info(
            "Whichmut fused translate+tokenize+codon_index: %.1fs",
            time.perf_counter() - t0,
        )

        dataset = _build_precomputed_dataset(
            raw_parents=raw_parents,
            raw_children=raw_children,
            neutral_model=neutral_model,
            handler=handler,
            transforms=transforms,
            output_keys=WHICHMUT_OUTPUT_KEYS,
            aa_parents=aa_parents,
            nickname=nickname,
            per_sample_tensors={
                "aa_parents_idxs": parent_aa_tokens,
                "aa_children_idxs": child_aa_tokens,
                "codon_parents_idxs": parent_codon_idxs,
                "codon_children_idxs": child_codon_idxs,
            },
            censor_transforms=censor_transforms,
            clades=clades,
        )
    else:
        aa_parents = [translate_sequence(p) for p in raw_parents]
        transform_neutral = copy.deepcopy(neutral_model)
        all_transforms: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = [
            ComputeNtRatesAndCsps(transform_neutral, handler),
            *transforms,
        ]
        dataset = TransformDataset(
            aa_parents=aa_parents,
            raw_parents=raw_parents,
            raw_children=raw_children,
            transforms=all_transforms,
            output_keys=WHICHMUT_OUTPUT_KEYS,
            nickname=nickname,
            # Same mechanism contrastive_binding.py uses for antigen_key: the
            # streaming chain runs per sample, so the clade must ride along in
            # the sample dict rather than being applied out of band.
            extra_string_fields=None if clades is None else {"clade": clades},
        )

    padding_sentinel = padding_sentinel_for(num_seg)
    collator = TransformCollator(
        tensor_specs=_whichmut_collator_specs(padding_sentinel)
    )

    logger.info("Whichmut factory total: %.1fs", time.perf_counter() - t_factory_start)
    return DatasetBundle(
        dataset=dataset,
        collator=collator,
        worker_init_fn=make_worker_init_fn(),
    )


# ---------------------------------------------------------------------------
# Factory: MPNN Whichmut (wrapper + combined collator)
# ---------------------------------------------------------------------------






EVOLUTIONARY_OUTPUT_KEYS = frozenset(
    {
        "aa_parents_idxs",
        "aa_children_idxs",
        "codon_parents_idxs",
        "codon_children_idxs",
        "layout",
        "mask",
        "loss_mask",
        "log_neutral_codon_probs",
        "branch_length",
        "codon_rates",
        "codon_csps",
        "codon_parent_nt_window",
        "codon_parent_boundary_mask",
        "codon_boundary_features",
        "codon_rate_adjustments",
    }
)


BL_OPTIMIZATION_OUTPUT_KEYS = frozenset(
    (EVOLUTIONARY_OUTPUT_KEYS - {"log_neutral_codon_probs"}) | {"nt_rates", "nt_csps"}
)
"""Output keys for branch length optimization DataLoader.

Extends ``EVOLUTIONARY_OUTPUT_KEYS`` with ``nt_rates`` and ``nt_csps`` so
that the BL optimizer can stream these from Parquet instead of bulk-loading
them into memory, and drops ``log_neutral_codon_probs``, which the BL
objective recomputes at each trial branch length (issue #1276).
"""


def _bl_optimization_collator_specs(
    padding_sentinel: int,
    include_parent_nt_windows: bool = False,
) -> Dict[str, Tuple[float, TensorScale]]:
    """Build collator tensor_specs for BL optimization.

    Extends evolutionary specs with ``nt_rates`` and ``nt_csps`` as
    ``"list"``-scale keys (variable-length tensors kept as ``List[Tensor]``,
    not padded/stacked), minus ``log_neutral_codon_probs``, which the BL
    dataset does not compute (issue #1276).
    """
    specs = dict(
        _evolutionary_collator_specs(
            padding_sentinel, include_parent_nt_windows=include_parent_nt_windows
        )
    )
    del specs["log_neutral_codon_probs"]  # not in BL_OPTIMIZATION_OUTPUT_KEYS
    specs["nt_rates"] = (0.0, "list")
    specs["nt_csps"] = (0.0, "list")
    return specs


def _evolutionary_collator_specs(
    padding_sentinel: int,
    include_parent_nt_windows: bool = False,
) -> Dict[str, Tuple[float, TensorScale]]:
    """Build collator tensor_specs for SelectionFactorHead + DASM loss.

    The ``codon_parent_*``, ``codon_boundary_features`` and
    ``codon_rate_adjustments`` keys are produced by ``ComputeParentNtWindows``;
    pass ``include_parent_nt_windows=False`` to omit them when that transform
    is gated off (plain DASM training).
    """
    specs: Dict[str, Tuple[float, TensorScale]] = {
        "aa_parents_idxs": (AA_AMBIG_IDX, "codon"),
        "aa_children_idxs": (AA_AMBIG_IDX, "codon"),
        "codon_parents_idxs": (AMBIGUOUS_CODON_IDX, "codon"),
        "codon_children_idxs": (AMBIGUOUS_CODON_IDX, "codon"),
        "layout": (padding_sentinel, "codon"),
        "mask": (False, "codon"),
        "loss_mask": (False, "codon"),
        "log_neutral_codon_probs": (math.log(SMALL_PROBABILITY), "codon"),
        "branch_length": (0.0, "scalar"),
        "codon_rates": (0.0, "codon"),
        "codon_csps": (0.0, "codon"),
    }
    if include_parent_nt_windows:
        # Fill value 4 = "N"/other — matches the out-of-range flag in
        # ComputeParentNtWindows so padded codon sites never hit a valid
        # 13-mer in the Option-1a lookup.
        specs["codon_parent_nt_window"] = (4, "codon")
        specs["codon_parent_boundary_mask"] = (True, "codon")
        specs["codon_boundary_features"] = (-1.0, "ragged")
        specs["codon_rate_adjustments"] = (1.0, "codon")
    return specs


DATAFRAME_BL_REMEDIATION = (
    "Drop the column to fall back to "
    f"DEFAULT_BRANCH_LENGTH={DEFAULT_BRANCH_LENGTH}, or fill it in."
)

_PARQUET_BL_REMEDIATION = (
    "Fill in the missing entries, or remove the column from every Parquet "
    f"file to fall back to DEFAULT_BRANCH_LENGTH={DEFAULT_BRANCH_LENGTH}. "
    "Removing it from only some files leaves the remaining files' values in "
    "play, since the default fills per file."
)












# ---------------------------------------------------------------------------
# Factory: Parquet-backed streaming
# ---------------------------------------------------------------------------


def _read_combined_nt_sequences_from_parquet(
    backend: "ParquetBackend",
    segment_handler: SegmentHandler,
    base_name: str,
) -> List[str]:
    """Read segment columns from Parquet and combine into NT sequences.

    Uses ``backend.read_column`` for efficient column-oriented access
    (single scan per column, no row-by-row ``get_row`` overhead).

    Args:
        backend: ParquetBackend to read columns from.
        segment_handler: Handler defining segment suffixes.
        base_name: Base column name (e.g. ``"parent"`` or ``"child"``).

    Returns:
        List of combined NT sequences (segments joined with ``^^^``).
    """
    num_segments = segment_handler.get_num_segments()
    segment_columns: List[List[Any]] = [
        backend.read_column(col_name, default="")
        for col_name in segment_handler.get_column_names(base_name)
    ]

    n = backend.total_rows
    combined: List[str] = []
    for row_idx in range(n):
        segments = [
            str(segment_columns[seg_idx][row_idx] or "")
            for seg_idx in range(num_segments)
        ]
        combined.append(NT_SEPARATOR.join(segments))
    return combined


_LOGGED_UNUSED_NEUTRAL_MODELS: Set[Tuple[Tuple[str, ...], str]] = set()
"""(paths, model name) pairs already logged by ``_parquet_neutral_rate_plan``.

Train and val bundles both read the same spec, so without this the "config's
model is not used for rates" line would print twice per run. Never cleared: it
holds one entry per distinct spec a process loads, which stays small.
"""


def _parquet_neutral_rate_plan(
    backend: "ParquetBackend",
    segment_handler: SegmentHandler,
    neutral_model: Optional["RatesCSPNeutralModel"],
) -> Tuple[List[Callable[[Dict[str, Any]], Dict[str, Any]]], Tuple[str, ...]]:
    """Decide where a Parquet bundle's per-site rates come from (issue #1299).

    The file's columns decide:

    | Columns? | Config ``neutral_model``   | Rates from                  |
    |----------|----------------------------|-----------------------------|
    | yes      | ``None`` or carrier        | the columns                 |
    | yes      | a model that computes them | the columns (logged once)   |
    | no       | a model that computes them | ``ComputeNtRatesAndCsps``   |
    | no       | ``None`` or carrier        | nothing: raises             |

    Args:
        backend: The bundle's backend (its per-file schemas are checked).
        segment_handler: Defines the segment suffixes and rate adjustments.
        neutral_model: The configured model, or None.

    Returns:
        ``(prefix_transforms, context_cols)``. ``prefix_transforms`` is empty
        or a single ``ComputeNtRatesAndCsps``, and goes ahead of the factory's
        own chain. ``context_cols`` holds the model's ``required_context``
        columns (e.g. ``sample_id``), for the row converter to carry into
        each sample. It is empty when rates come from the columns.

    Raises:
        ValueError: If the files are rate-free and no model can compute rates,
            if the model needs a context column the files lack, or (from
            ``parquet_carries_neutral_columns``) if the rate columns are
            partial or disagree across files.
    """
    paths = backend.paths
    has_columns = parquet_carries_neutral_columns(
        paths, backend.file_column_names, segment_handler
    )
    computes = neutral_model is not None and neutral_model.computes_rates
    if has_columns:
        if computes:
            assert neutral_model is not None
            key = (tuple(str(p) for p in paths), neutral_model.name)
            if key not in _LOGGED_UNUSED_NEUTRAL_MODELS:
                _LOGGED_UNUSED_NEUTRAL_MODELS.add(key)
                logger.info(
                    "Parquet file(s) %s carry nt_rates/nt_csps columns, so "
                    "rates come from them; the configured neutral model %s "
                    "is not used for rates (DESIGN.md D8).",
                    [p.name for p in paths],
                    neutral_model.name,
                )
        return [], ()
    if not computes:
        configured = "none" if neutral_model is None else neutral_model.name
        raise ValueError(
            f"Parquet file(s) {[str(p) for p in paths]} carry no nt_rates/"
            "nt_csps columns, so the configured neutral model must compute "
            f"rates at load, but the configured model is {configured}. "
            "Configure a rate model (e.g. GTRCodon variant: jc69), or "
            "package the corpus with its rate columns."
        )
    assert neutral_model is not None
    context_cols = tuple(neutral_model.required_context)
    # Checked on every file here: make_parquet_row_converter checks its
    # scalar columns against the first file's schema only.
    missing = sorted(
        {
            col
            for names in backend.file_column_names
            for col in context_cols
            if col not in names
        }
    )
    if missing:
        raise ValueError(
            f"Neutral model {neutral_model.name} needs per-row context "
            f"column(s) {missing}, which Parquet file(s) "
            f"{[str(p) for p in paths]} lack."
        )
    return [
        ComputeNtRatesAndCsps(copy.deepcopy(neutral_model), segment_handler)
    ], context_cols


def create_whichmut_dataset_from_parquet(
    parquet_paths: List[Path],
    segment_handler: SegmentHandler,
    max_aa_seq_len: int = 0,
    neutral_model: Optional["RatesCSPNeutralModel"] = None,
    buffer_size: int = 8192,
    nickname: str = "unnamed_whichmut_parquet",
    row_filter: Optional[List[bool]] = None,
    censor_table: Optional[Dict[str, CensorSpec]] = None,
    censor_mode: CensorMode = "both",
    reference_base_idxs: Optional[torch.Tensor] = None,
    shuffle: bool = True,
    cache_budget: CacheBudgetSpec = CacheBudgetSpec(),
) -> DatasetBundle:
    """Create Whichmut dataset from Parquet files.

    Reads raw sequences from Parquet files on disk. Uses the
    ``SegmentHandler`` to combine segment-specific columns (e.g.
    ``parent_heavy``, ``nt_rates_heavy``) into the generic keys expected by
    the transform chain. Per-sample transforms run in ``__getitem__``, as in
    in-memory streaming mode.

    Per-site ``nt_rates``/``nt_csps`` come from the files' columns when they
    carry them. A rate-free file has them computed at load by
    ``neutral_model`` through ``ComputeNtRatesAndCsps``. See
    ``_parquet_neutral_rate_plan`` for the full rule (issue #1299).

    A ``ShuffleBufferSampler`` is included in the bundle to provide
    shuffle-buffer semantics while keeping the map-style Dataset interface.

    Args:
        parquet_paths: Paths to Parquet files (one or more).
        segment_handler: Handler for segment-specific column mapping
            and rate adjustments.
        max_aa_seq_len: Maximum amino acid sequence length (from Parquet
            file-level metadata or known in advance).
        neutral_model: The configured neutral model. Computes rates for a
            rate-free file; unused for rates when the file carries columns.
        buffer_size: Shuffle buffer size for the sampler.
        nickname: Human-readable identifier.
        row_filter: Optional boolean list selecting which rows to include.
        censor_table: Optional ``{clade: CensorSpec}`` enabling UShER censoring
            (#1079). Requires a ``clade`` column in the Parquet.
        censor_mode: Which half of censoring to apply, for both sources.
        reference_base_idxs: Optional reference base indices enabling
            parent -> reference reversion censoring (#1099). Needs no ``clade``
            column, so it imposes no requirement on the Parquet's schema.
        shuffle: Whether the bundle's sampler shuffles. Pass False for
            validation and other single-pass sweeps: order cannot affect the
            result, and a sequential walk needs a handful of cached row groups
            where a shuffle buffer needs hundreds.
        cache_budget: Row-group cache RAM budget inputs (issue #1272); see
            ``dasm2.data.parquet_backend.plan_cache``.

    Returns:
        DatasetBundle with dataset, collator, worker_init_fn, and sampler.

    Raises:
        ValueError: If a selected row has no scored codon substitution (see
            ``_validate_parquet_scored_substitutions``), if ``censor_table``
            is given and the Parquet has no ``clade`` column, or if the
            files' rate columns and ``neutral_model`` leave no rate source
            (see ``_parquet_neutral_rate_plan``).
    """
    backend = ParquetBackend(
        parquet_paths,
        buffer_size=buffer_size,
        row_filter=row_filter,
        # A sequential sweep's working set is the group it is walking, not the
        # buffer's. Left to auto-size from ``buffer_size`` it asks for hundreds
        # of row groups, blows the RAM budget, gets capped, and then thrashes.
        cache_size=None if shuffle else SEQUENTIAL_SWEEP_CACHE_SIZE,
        cache_budget=cache_budget,
    )
    _validate_parquet_scored_substitutions(backend, segment_handler, parquet_paths)
    sampler = ShuffleBufferSampler(
        backend.row_group_boundaries, buffer_size=buffer_size, shuffle=shuffle
    )
    rate_transforms, scalar_cols = _parquet_neutral_rate_plan(
        backend, segment_handler, neutral_model
    )
    if censor_table is not None:
        if "clade" not in set(backend.column_names):
            raise ValueError(
                "censor_table was given but this Parquet has no 'clade' column. "
                "prepare_viral_parquet.py writes it as a scalar column, so a "
                "corpus without one predates that or was built by another "
                f"script. Columns: {sorted(backend.column_names)}."
            )
        scalar_cols = (*scalar_cols, "clade")
        # One dictionary-encoded string column, read once in the parent process.
        # Cheap next to the rates columns, and it turns a worker-side crash deep
        # into a multi-hour run into a startup failure.
        _check_corpus_clades_covered(
            set(_read_parquet_column_raw(list(parquet_paths), "clade")),
            censor_table,
            nickname,
        )
    row_converter = make_parquet_row_converter(
        segment_handler,
        available_columns=set(backend.column_names),
        scalar_cols=scalar_cols,
    )

    num_segments = segment_handler.get_num_segments()
    transforms: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = [
        *rate_transforms,
        *_whichmut_transforms(
            num_segments,
            censor_table=censor_table,
            censor_mode=censor_mode,
            reference_base_idxs=reference_base_idxs,
        ),
    ]

    dataset = TransformDataset(
        parquet_backend=backend,
        parquet_row_converter=row_converter,
        transforms=transforms,
        output_keys=WHICHMUT_OUTPUT_KEYS,
        max_aa_seq_len_override=max_aa_seq_len if max_aa_seq_len > 0 else None,
        nickname=nickname,
    )

    padding_sentinel = padding_sentinel_for(num_segments)
    collator = TransformCollator(
        tensor_specs=_whichmut_collator_specs(padding_sentinel)
    )

    return DatasetBundle(
        dataset=dataset,
        collator=collator,
        worker_init_fn=make_worker_init_fn(),
        sampler=sampler,
    )




# ---------------------------------------------------------------------------
# Precomputed mode: batched post-tokenization transforms
# ---------------------------------------------------------------------------
#
# These replace per-sample ComputeLayout/ComputeMasks/ComputeMutIndicator/
# LookupAAONeutralRates/NarrowLossMask with concat→compute→split operations
# on lists of variable-length tensors. Same pattern as
# _batched_compute_codon_mutation_rates. At 1M sequences this saves ~80s.










# ---------------------------------------------------------------------------
# Precomputed mode: batched codon-level computation helpers
# ---------------------------------------------------------------------------


def _batched_compute_codon_mutation_rates(
    samples: List[Dict[str, Any]],
    nt_rates_list: List[torch.Tensor],
    nt_csps_list: List[torch.Tensor],
    codon_batch_size: int = 500_000,
) -> None:
    """Compute codon_neutral_rates for all samples in a batched operation.

    Replaces the per-sample ``ComputeCodonMutationRates`` transform with a
    vectorized version. For each codon, the per-sample transform does
    gather+scatter over 9 single-nucleotide mutations. This batched version
    processes codons in chunks to limit peak memory, performing the
    gather+scatter with advanced indexing on each chunk.

    Modifies *samples* in-place by adding ``codon_neutral_rates``.

    Args:
        samples: List of sample dicts. Each must contain ``codon_parents_idxs``.
        nt_rates_list: Per-sequence neutral rates (from vectorized precompute).
        nt_csps_list: Per-sequence neutral CSPs (from vectorized precompute).
        codon_batch_size: Max codons per chunk to limit peak memory.
    """
    n_samples = len(samples)

    # Collect per-sample codon counts and pre-allocate output tensors.
    codon_counts: List[int] = []
    for i in range(n_samples):
        num_codons = samples[i]["codon_parents_idxs"].shape[0]
        codon_counts.append(num_codons)
        samples[i]["codon_neutral_rates"] = torch.zeros(num_codons, NUM_CODON_INDICES)

    total_codons = sum(codon_counts)
    if total_codons == 0:
        return

    # Process samples in chunks whose cumulative codon count fits in memory.
    chunk_start = 0  # first sample index in current chunk
    chunk_codons = 0  # codons accumulated so far in current chunk

    for i in range(n_samples + 1):
        # Flush the current chunk when we exceed batch size or reach the end.
        if i < n_samples:
            next_count = codon_counts[i]
            # Always include at least one sample per chunk.
            if chunk_codons > 0 and chunk_codons + next_count > codon_batch_size:
                _process_cmr_chunk(
                    samples,
                    nt_rates_list,
                    nt_csps_list,
                    codon_counts,
                    chunk_start,
                    i,
                )
                chunk_start = i
                chunk_codons = 0
            chunk_codons += next_count
        else:
            # Final chunk.
            if chunk_codons > 0:
                _process_cmr_chunk(
                    samples,
                    nt_rates_list,
                    nt_csps_list,
                    codon_counts,
                    chunk_start,
                    i,
                )


def _process_cmr_chunk(
    samples: List[Dict[str, Any]],
    nt_rates_list: List[torch.Tensor],
    nt_csps_list: List[torch.Tensor],
    codon_counts: List[int],
    start: int,
    end: int,
) -> None:
    """Process a chunk of samples for codon mutation rates.

    Concatenates codons from samples[start:end], performs the vectorized
    gather+scatter, and writes results into pre-allocated output tensors.

    Args:
        samples: Full sample list (modified in-place).
        nt_rates_list: Per-sequence neutral rates.
        nt_csps_list: Per-sequence neutral CSPs.
        codon_counts: Pre-computed codon count per sample.
        start: First sample index (inclusive).
        end: Last sample index (exclusive).
    """
    chunk_codon_idxs: List[torch.Tensor] = []
    chunk_nt_rates: List[torch.Tensor] = []

    for i in range(start, end):
        codon_idxs = samples[i]["codon_parents_idxs"]
        num_codons = codon_counts[i]
        assert nt_rates_list[i].shape[0] == codon_idxs.shape[0] * 3, (
            f"Sample {i}: nt_rates has {nt_rates_list[i].shape[0]} entries "
            f"but codon_parents_idxs has {codon_idxs.shape[0]} codons "
            f"(expected {codon_idxs.shape[0] * 3} nt entries)"
        )
        chunk_codon_idxs.append(codon_idxs)

        # combine_nt_rates_and_csps: rates.unsqueeze(-1) * csps
        nt_rates_per_base = nt_rates_list[i].unsqueeze(-1) * nt_csps_list[i]
        chunk_nt_rates.append(nt_rates_per_base[: num_codons * 3])

    cat_codon_idxs = torch.cat(chunk_codon_idxs)
    cat_nt_rates = torch.cat(chunk_nt_rates)
    chunk_total = cat_codon_idxs.shape[0]

    cat_codon_rates = compute_codon_mutation_rates(
        chunk_total, cat_nt_rates, cat_codon_idxs
    )

    # Write results into pre-allocated per-sample tensors.
    offset = 0
    for i in range(start, end):
        n = codon_counts[i]
        samples[i]["codon_neutral_rates"] = cat_codon_rates[offset : offset + n]
        offset += n

    if offset != chunk_total:
        raise ValueError(
            f"Codon count mismatch in chunk: wrote {offset} codons "
            f"but chunk has {chunk_total}"
        )


def _precompute_passes_1_and_2(
    raw_parents: List[str],
    raw_children: List[str],
    neutral_model: "RatesCSPNeutralModel",
    handler: SegmentHandler,
    transforms: List[Callable[[Dict[str, Any]], Dict[str, Any]]],
    output_keys: FrozenSet[str],
    batched_transform_type: Optional[type] = None,
    branch_lengths: Optional[torch.Tensor] = None,
    per_sample_tensors: Optional[Dict[str, List[torch.Tensor]]] = None,
) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor], List[torch.Tensor]]:
    """Run passes 1-2 of precomputed dataset construction (shared by all heads).

    Pass 1: Batched neutral model → nt_rates, nt_csps per sequence.
    Pass 2: Per-sample transforms (layout, masks) with fused and batched
    transforms filtered out.

    Args:
        raw_parents: Parent NT sequences.
        raw_children: Child NT sequences.
        neutral_model: Neutral mutation model (batched, not deep-copied).
        handler: Segment handler.
        transforms: Full per-sample transform chain.
        output_keys: Keys to retain in output.
        batched_transform_type: Transform type to split out for batched
            computation in pass 3 (e.g., ComputeCodonMutationRates or
            ComputeNeutralCodonProbs). None means no transforms are split out.
        branch_lengths: Optional per-sequence branch lengths (DASM only).
        per_sample_tensors: Optional pre-computed tensors from fused
            translate+tokenize+codon_index.

    Returns:
        Tuple of (samples, nt_rates_list, nt_csps_list).
    """
    # Pass 1: Batched neutral model
    t0 = time.perf_counter()
    logger.info(
        "Precomputing neutral model outputs for %d sequences...", len(raw_parents)
    )
    nt_rates_list, nt_csps_list = _precompute_neutral_outputs_vectorized(
        raw_parents, neutral_model, handler
    )
    logger.info("Vectorized neutral model: %.1fs", time.perf_counter() - t0)

    # Split out batched transform (if specified)
    if batched_transform_type is not None:
        per_sample_transforms = [
            t for t in transforms if not isinstance(t, batched_transform_type)
        ]
    else:
        per_sample_transforms = list(transforms)

    # Filter out fused transforms if pre-computed tensors are provided
    if per_sample_tensors is not None:
        per_sample_transforms = [
            t
            for t in per_sample_transforms
            if not isinstance(
                t, (TranslateAndTokenize, CodonIndex, FusedTranslateTokenizeCodonIndex)
            )
        ]

    # Pass 2: Per-sample transforms (layout, masks, mut_indicator)
    t0 = time.perf_counter()
    logger.info("Running per-sample transforms...")
    samples = _run_transforms_on_all_samples(
        raw_parents=raw_parents,
        raw_children=raw_children,
        transforms=per_sample_transforms,
        output_keys=output_keys,
        nt_rates_list=nt_rates_list,
        nt_csps_list=nt_csps_list,
        branch_lengths=branch_lengths,
        per_sample_tensors=per_sample_tensors,
    )
    logger.info("Per-sample transforms: %.1fs", time.perf_counter() - t0)

    return samples, nt_rates_list, nt_csps_list


def _build_precomputed_dataset(
    raw_parents: List[str],
    raw_children: List[str],
    neutral_model: "RatesCSPNeutralModel",
    handler: SegmentHandler,
    transforms: List[Callable[[Dict[str, Any]], Dict[str, Any]]],
    output_keys: FrozenSet[str],
    aa_parents: List[str],
    nickname: str,
    per_sample_tensors: Optional[Dict[str, List[torch.Tensor]]] = None,
    censor_transforms: Sequence[
        Union[CensorMaskedMutations, CensorReferenceReversions]
    ] = (),
    clades: Optional[List[str]] = None,
) -> TransformDataset:
    """Build precomputed dataset for non-BL heads (Whichmut).

    Four-pass construction:
    1. Batched neutral model → nt_rates, nt_csps per sequence
    2. Per-sample transforms (layout, masks — tokenize/codon/mutation-rates
       skipped if fused/batched)
    3. Batched codon mutation rate computation (if ComputeCodonMutationRates
       was in the transform chain)
    4. Censoring, for each transform given (#1079, #1099)

    Pass 4 exists because censoring cannot ride in the pass-2 chain: pass 2
    filters ``ComputeCodonMutationRates`` out and then strips each sample to
    ``output_keys``, so ``codon_neutral_rates`` does not exist yet *and*
    ``clade`` has already been discarded. Appending
    ``CensorMaskedMutations`` to the transform list would raise ``KeyError``
    here while working correctly in streaming and Parquet mode — which is why
    the empty-table equivalence test is parameterized over all three.

    Pass 4 takes a *sequence* rather than one transform because #1099's
    reversion rule is a sibling of the table's, not a mode of it, and the two
    compose. ``clades`` is required only by the clade-keyed transform:
    :class:`~dasm2.data.transforms.CensorReferenceReversions` is
    clade-independent, so demanding clades for it would make a reversion-only
    arm depend on a roster it never reads.

    Args:
        raw_parents: Parent NT sequences.
        raw_children: Child NT sequences.
        neutral_model: Neutral mutation model (batched, not deep-copied).
        handler: Segment handler.
        transforms: Per-sample transforms (run after nt_rates/nt_csps injection).
            If per_sample_tensors is provided, TranslateAndTokenize and
            CodonIndex transforms are filtered out automatically.
            ComputeCodonMutationRates is always split out for batched computation.
        output_keys: Keys to retain in output.
        aa_parents: Pre-computed AA parent sequences.
        nickname: Human-readable identifier.
        per_sample_tensors: Optional pre-computed tensors from fused
            translate+tokenize+codon_index.
        censor_transforms: Censoring transforms applied as pass 4, in order.
            Order does not affect the result -- see
            :func:`~dasm2.data.transforms._apply_censored_entries` -- but it is
            preserved so logs read predictably.
        clades: Per-sample clade names, required when any of
            ``censor_transforms`` is clade-keyed. Passed as a list rather than
            injected into the sample dicts because pass 2 has already stripped
            them to ``output_keys``.

    Returns:
        TransformDataset in precomputed mode.

    Raises:
        ValueError: If a clade-keyed censor transform is given without matching
            ``clades``.
    """
    if any(isinstance(t, CensorMaskedMutations) for t in censor_transforms):
        if clades is None:
            raise ValueError(
                "CensorMaskedMutations requires clades, since that table is "
                "per-clade and the precomputed samples no longer carry a "
                "'clade' key. CensorReferenceReversions does not."
            )
        if len(clades) != len(raw_parents):
            raise ValueError(
                f"clades has {len(clades)} entries but there are "
                f"{len(raw_parents)} sequences. A positional mismatch would "
                f"censor each PCP against another clade's mask."
            )
    has_codon_mutation_rates = any(
        isinstance(t, ComputeCodonMutationRates) for t in transforms
    )

    samples, nt_rates_list, nt_csps_list = _precompute_passes_1_and_2(
        raw_parents=raw_parents,
        raw_children=raw_children,
        neutral_model=neutral_model,
        handler=handler,
        transforms=transforms,
        batched_transform_type=ComputeCodonMutationRates,
        output_keys=output_keys,
        per_sample_tensors=per_sample_tensors,
    )

    # Pass 3: Batched codon mutation rates (Whichmut-specific)
    if has_codon_mutation_rates:
        t0 = time.perf_counter()
        logger.info("Computing codon mutation rates (batched)...")
        _batched_compute_codon_mutation_rates(
            samples=samples,
            nt_rates_list=nt_rates_list,
            nt_csps_list=nt_csps_list,
        )
        logger.info("Batched CMR: %.1fs", time.perf_counter() - t0)

    # Pass 4: censoring (#1079, #1099). Runs here rather than in the pass-2 chain
    # because it needs codon_neutral_rates, which pass 3 has only just written.
    if censor_transforms and not has_codon_mutation_rates:
        raise ValueError(
            "a censor transform was given but the transform chain has no "
            "ComputeCodonMutationRates, so there are no codon_neutral_rates "
            "to censor. Censoring is only defined for whichmut-style heads."
        )
    for censor_transform in censor_transforms:
        t0 = time.perf_counter()
        logger.info(
            "Censoring with %s (mode=%s)...",
            type(censor_transform).__name__,
            censor_transform.mode,
        )
        if isinstance(censor_transform, CensorMaskedMutations):
            assert clades is not None  # validated above
            # `apply` rather than `__call__`: the samples hold tensors only by
            # now, so the clade is passed as an argument instead of being written
            # into the dict and deleted again. A string left behind would reach
            # the collator, which has no tensor_spec for it.
            for sample, clade in zip(samples, clades):
                censor_transform.apply(sample, clade)
        else:
            for sample in samples:
                censor_transform(sample)
        logger.info("Censoring: %.1fs", time.perf_counter() - t0)

    return TransformDataset(
        aa_parents=aa_parents,
        precomputed_samples=samples,
        nickname=nickname,
    )






def _run_transforms_on_all_samples(
    raw_parents: List[str],
    raw_children: List[str],
    transforms: List[Callable[[Dict[str, Any]], Dict[str, Any]]],
    output_keys: FrozenSet[str],
    nt_rates_list: Optional[List[torch.Tensor]] = None,
    nt_csps_list: Optional[List[torch.Tensor]] = None,
    branch_lengths: Optional[torch.Tensor] = None,
    per_sample_tensors: Optional[Dict[str, List[torch.Tensor]]] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Run transforms on all samples at init time (precomputed mode).

    Builds a seed dict for each sample with raw sequences (and optionally
    precomputed neutral model outputs and pre-computed tensors), runs the
    transform chain, and retains only output keys.

    Args:
        raw_parents: Parent NT or AA sequences.
        raw_children: Child NT or AA sequences.
        transforms: Per-sample transforms to run.
        output_keys: Keys to retain in output.
        nt_rates_list: Optional precomputed neutral rates per sequence.
        nt_csps_list: Optional precomputed neutral CSPs per sequence.
        branch_lengths: Optional branch lengths for DASM head.
        per_sample_tensors: Optional dict mapping key names to lists of
            pre-computed tensors (one per sample). Injected into each seed
            before transforms run. Used by the fused translate+tokenize+codon
            index optimization to skip TranslateAndTokenize and CodonIndex.

    Returns:
        List of output dicts, one per sample.
    """
    samples: List[Dict[str, torch.Tensor]] = []
    for i in range(len(raw_parents)):
        seed: Dict[str, Any] = {
            "raw_parent": raw_parents[i],
            "raw_child": raw_children[i],
        }
        if nt_rates_list is not None and nt_csps_list is not None:
            seed["nt_rates"] = nt_rates_list[i]
            seed["nt_csps"] = nt_csps_list[i]
        if branch_lengths is not None:
            seed["branch_length"] = branch_lengths[i]
        if per_sample_tensors is not None:
            for key, tensor_list in per_sample_tensors.items():
                seed[key] = tensor_list[i]

        for t in transforms:
            seed = t(seed)

        sample = {k: v for k, v in seed.items() if k in output_keys}
        samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# Train/val loading helpers
# ---------------------------------------------------------------------------


def _load_and_split(
    data_spec: str,
    cv_split: Optional["CvSplitConfig"] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load DataFrame from data_spec and split into train/val.

    When ``cv_split`` is set, the loaded DataFrame's ``in_train`` column
    is overwritten with on-the-fly CV fold assignments (row-level draw
    or GroupKFold; see :class:`CvSplitConfig`). The original ``in_train``
    column and any registry holdout fallback are ignored.
    """
    df = load_dataframe_with_split_markers(data_spec)
    if cv_split is not None:
        df = _cv_split_dataframe(df, cv_split)
    train_df, val_df = split_dataframe(df)
    nickname = determine_nickname_from_data_spec(data_spec)
    return train_df, val_df, nickname


def _cv_split_dataframe(
    df: pd.DataFrame,
    cv_split: "CvSplitConfig",
) -> pd.DataFrame:
    """Override ``in_train`` with on-the-fly CV fold assignments.

    Two modes:

    - ``cv_split.group_col is None``: row-positional draw. Each row
      independently lands in fold ``rng.integers(num_folds)``.
    - ``cv_split.group_col`` set: ``GroupKFold`` partitions the rows so
      every row sharing the column value ends up in the same fold.
      ``cv_split.seed`` is ignored (sklearn's GroupKFold is deterministic
      given the input group ordering).
    """
    if cv_split.group_col is not None:
        if cv_split.group_col not in df.columns:
            raise ValueError(
                f"cv_split.group_col='{cv_split.group_col}' not found in DataFrame "
                f"columns: {sorted(df.columns)[:20]}"
            )
        in_train_arr = group_kfold_in_train(
            np.asarray(df[cv_split.group_col]), cv_split
        )
    else:
        rng = np.random.default_rng(cv_split.seed)
        fold_per_row = rng.integers(cv_split.num_folds, size=len(df))
        in_train_arr = fold_per_row != cv_split.fold_index
    df = df.copy()
    df["in_train"] = in_train_arr
    train_count = int(in_train_arr.sum())
    val_count = len(df) - train_count
    if val_count == 0:
        raise ValueError(
            f"CV fold {cv_split.fold_index} of {cv_split.num_folds} produced an "
            f"empty validation set on {len(df)} rows. "
            "Use fewer folds or a larger dataset."
        )
    if cv_split.group_col is not None:
        held_out_groups = sorted(
            map(str, df.loc[~in_train_arr, cv_split.group_col].unique())
        )
        cv_split_logger.info(
            "CV split (group, CSV): fold %d of %d on '%s' "
            "(%d train, %d val); held-out groups: %s",
            cv_split.fold_index,
            cv_split.num_folds,
            cv_split.group_col,
            train_count,
            val_count,
            _format_group_preview(held_out_groups),
        )
    else:
        cv_split_logger.info(
            "CV split (row-level, CSV): fold %d of %d (seed=%d, %d train, %d val)",
            cv_split.fold_index,
            cv_split.num_folds,
            cv_split.seed,
            train_count,
            val_count,
        )
    return df


def _format_group_preview(groups: List[str], max_groups: int = 20) -> str:
    """Format a list of group ids for log lines, truncating long lists."""
    if len(groups) <= max_groups:
        return f"[{', '.join(groups)}]"
    head = ", ".join(groups[:max_groups])
    return f"[{head}, ... (+{len(groups) - max_groups} more)]"


def load_whichmut_train_val(
    data_spec: str,
    neutral_model: Optional["RatesCSPNeutralModel"] = None,
    mode: Mode = "precomputed",
    buffer_size: int = 8192,
    cv_split: Optional["CvSplitConfig"] = None,
    censor_table: Optional[Dict[str, CensorSpec]] = None,
    censor_mode: CensorMode = "both",
    reference_base_idxs: Optional[torch.Tensor] = None,
    cache_budget: CacheBudgetSpec = CacheBudgetSpec(),
) -> Tuple[DatasetBundle, DatasetBundle]:
    """Load Whichmut train/val datasets.

    Automatically detects Parquet vs CSV/DataFrame storage and dispatches
    accordingly.  CSV data requires ``neutral_model``. Parquet data requires
    it only when the files carry no ``nt_rates``/``nt_csps`` columns: the
    model then computes rates at load (issue #1299).

    Args:
        data_spec: File path, registered nickname, or multiname.
        neutral_model: Neutral mutation model (required for CSV and for
            rate-free Parquet; unused for rates on column-carrying Parquet).
        mode: ``"precomputed"`` or ``"streaming"`` (CSV only).
        buffer_size: Shuffle buffer size (Parquet only).
        cv_split: Optional cross-validation fold spec. Bypasses the
            ``in_train`` column or registry holdout fallback by computing
            the split on the fly. Supported on both Parquet and CSV
            inputs.
        censor_table: Optional ``{clade: CensorSpec}`` enabling UShER censoring
            (#1079), applied to both splits.
        censor_mode: Which half of censoring to apply, for both sources.
        reference_base_idxs: Optional reference bases for parent -> reference
            reversion censoring (#1099), applied to both splits.
        cache_budget: Row-group cache RAM budget inputs (issue #1272); see
            ``dasm2.data.parquet_backend.plan_cache``.

    Returns:
        Tuple of (train_bundle, val_bundle).

    Raises:
        ValueError: If ``neutral_model`` is None for non-Parquet data.
    """
    if is_parquet_data_spec(data_spec):
        return _load_whichmut_train_val_parquet(
            data_spec,
            neutral_model,
            buffer_size,
            cv_split=cv_split,
            censor_table=censor_table,
            censor_mode=censor_mode,
            reference_base_idxs=reference_base_idxs,
            cache_budget=cache_budget,
        )

    if neutral_model is None:
        raise ValueError(
            "Non-Parquet Whichmut data requires a neutral_model, but got None."
        )

    train_df, val_df, nickname = _load_and_split(data_spec, cv_split=cv_split)

    train_bundle = create_whichmut_dataset(
        train_df,
        neutral_model,
        mode=mode,
        nickname=f"{nickname}_train",
        censor_table=censor_table,
        censor_mode=censor_mode,
        reference_base_idxs=reference_base_idxs,
    )
    val_bundle = create_whichmut_dataset(
        val_df,
        neutral_model,
        mode=mode,
        nickname=f"{nickname}_val",
        censor_table=censor_table,
        censor_mode=censor_mode,
        reference_base_idxs=reference_base_idxs,
    )
    return train_bundle, val_bundle












# ---------------------------------------------------------------------------
# Factory: Parquet train/val loaders
# ---------------------------------------------------------------------------


def _resolve_single_parquet_component(resolved: Path, component: str) -> List[Path]:
    """Resolve a single component path to Parquet file paths.

    Args:
        resolved: Already-resolved filesystem path.
        component: Original component string (for error messages).

    Returns:
        List of Parquet file paths.

    Raises:
        ValueError: If path is neither a .parquet file nor a directory
            containing .parquet files.
    """
    if resolved.is_file() and resolved.suffix == ".parquet":
        return [resolved]

    if resolved.is_dir():
        paths = sorted(resolved.glob("*.parquet"))
        if not paths:
            raise ValueError(f"Directory '{resolved}' contains no .parquet files")
        return paths

    raise ValueError(
        f"Parquet data spec component '{component}' resolved to '{resolved}' "
        f"which is neither a .parquet file nor a directory containing .parquet files"
    )




def _read_parquet_column_raw(parquet_paths: List[Path], col: str) -> List[Any]:
    """Read a single column from Parquet files without caching or filtering.

    Lightweight column reader for metadata extraction. Unlike
    ``ParquetBackend.read_column``, this does not construct an LRU cache,
    row-group index, or filtered index map — it simply concatenates column
    values from each file in order.

    Args:
        parquet_paths: Parquet file paths to read from.
        col: Column name to read.

    Returns:
        List of column values across all files in file order.
    """
    values: List[Any] = []
    for path in parquet_paths:
        table = pq.read_table(path, columns=[col])
        values.extend(table.column(col).to_pylist())
    return values


def _cv_split_train_filter(
    parquet_paths: List[Path],
    cv_split: "CvSplitConfig",
    num_rows: Optional[int] = None,
) -> List[bool]:
    """Build a per-row train filter from a CV fold spec across all paths.

    Two modes (selected by ``cv_split.group_col``):

    - ``group_col=None``: row-positional draw. Sums row counts across
      ``parquet_paths``, draws one ``num_folds``-way fold per row from a
      single seeded RNG, and returns ``fold != fold_index``.
    - ``group_col`` set: reads the column across all paths, runs
      ``GroupKFold`` so every row sharing a group lands in the same fold,
      and returns ``in_train`` (rows whose group went to a non-held-out
      fold). ``cv_split.seed`` is ignored in this mode.

    Multiname callers must pass every component's paths together so a
    single RNG draw / GroupKFold partition spans the concatenated rows.

    Args:
        parquet_paths: Every component's files, concatenated in load order.
        cv_split: The fold spec to realize.
        num_rows: Total rows across ``parquet_paths``, when the caller already
            read the footers for it (``_resolve_parquet_paths_and_schema`` sums
            it for the seam check). Only the ``group_col=None`` mode needs it;
            passing it there avoids a second footer pass over every file.
            ``None`` recomputes, so the helper still works standalone.
    """
    train_filter: List[bool]
    if cv_split.group_col is not None:
        groups = _read_parquet_column_raw(parquet_paths, cv_split.group_col)
        in_train = group_kfold_in_train(np.asarray(groups), cv_split)
        train_filter = [bool(v) for v in in_train]
    else:
        n_rows = (
            num_rows
            if num_rows is not None
            else sum(pq.ParquetFile(p).metadata.num_rows for p in parquet_paths)
        )
        rng = np.random.default_rng(cv_split.seed)
        fold_per_row = rng.integers(cv_split.num_folds, size=n_rows)
        train_filter = [bool(v) for v in (fold_per_row != cv_split.fold_index)]
    train_count = sum(train_filter)
    val_count = len(train_filter) - train_count
    if val_count == 0:
        raise ValueError(
            f"CV fold {cv_split.fold_index} of {cv_split.num_folds} produced an "
            f"empty validation set on {len(train_filter)} rows. "
            "Use fewer folds or a larger dataset."
        )
    if cv_split.group_col is not None:
        held_out_groups = sorted(
            {str(g) for g, t in zip(groups, train_filter) if not t}
        )
        cv_split_logger.info(
            "CV split (group, parquet): fold %d of %d on '%s' "
            "(%d train, %d val); held-out groups: %s",
            cv_split.fold_index,
            cv_split.num_folds,
            cv_split.group_col,
            train_count,
            val_count,
            _format_group_preview(held_out_groups),
        )
    else:
        cv_split_logger.info(
            "CV split (row-level, parquet): fold %d of %d (seed=%d, %d train, %d val)",
            cv_split.fold_index,
            cv_split.num_folds,
            cv_split.seed,
            train_count,
            val_count,
        )
    return train_filter


def _parquet_metadata(
    parquet_paths: List[Path],
    nickname: Optional[str] = None,
) -> Tuple[List[bool], List[bool], SegmentHandler, int]:
    """Read in_train filter, segment handler, and max_aa_seq_len from Parquet.

    Leaves the handler's ``bl_init_strategy`` alone: it is set for the whole
    spec by ``_resolve_parquet_paths_and_schema``, and one component's stamp
    would overwrite it (issue #1289).

    Uses ``pyarrow.parquet`` directly for lightweight metadata + column reads,
    avoiding construction of a full ``ParquetBackend`` (which is created
    later with different cache/filter settings by the factory functions).

    Split priority:

    1. ``in_train`` column on the Parquet files.
    2. Holdout-based fallback via the dataset's ``nickname``.

    The CV-split branch is handled upstream in ``_resolve_parquet_train_val``
    and never reaches this function.

    Args:
        parquet_paths: Parquet file paths.
        nickname: Optional dataset nickname for holdout-based fallback
            when ``in_train`` column is missing.

    Returns:
        Tuple of (train_filter, val_filter, segment_handler, max_aa_seq_len).

    Raises:
        ValueError: If ``in_train`` column is missing and holdout fallback
            fails (no nickname, no holdout_samples, or no sample_id column).
    """
    # Read schema from the first file (all files must share the same schema)
    first_file = pq.ParquetFile(parquet_paths[0])
    column_names = set(first_file.schema_arrow.names)
    file_metadata = read_parquet_file_metadata(first_file)

    # Read in_train column (or fall back to holdout-based splitting)
    if "in_train" in column_names:
        in_train_values = _read_parquet_column_raw(parquet_paths, "in_train")
        train_filter = [bool(v) for v in in_train_values]
    elif "sample_id" in column_names and nickname is not None:
        # Fallback: compute train filter from holdout sample IDs
        try:
            holdout_ids = _require_holdout_samples(nickname)
        except ValueError:
            raise ValueError(
                f"Parquet file(s) missing 'in_train' column and no holdout "
                f"samples defined for dataset '{nickname}'. Either add an "
                f"'in_train' column or define holdout_samples in the YAML registry."
            )
        holdout_set = set(holdout_ids)
        sample_ids = _read_parquet_column_raw(parquet_paths, "sample_id")
        train_filter = [str(sid) not in holdout_set for sid in sample_ids]
        logger.info(
            "Parquet fallback: computed train/val split from holdout_samples "
            "(%d holdout IDs, %d train, %d val)",
            len(holdout_set),
            sum(train_filter),
            sum(not v for v in train_filter),
        )
    else:
        raise ValueError(
            "Parquet file(s) missing required 'in_train' column for train/val "
            "splitting. Ensure the data preparation pipeline writes this column, "
            "or provide a 'sample_id' column with holdout_samples in the YAML registry."
        )
    val_filter = [not v for v in train_filter]

    # Detect segment handler
    handler = detect_compatible_handler(column_names, ("parent", "child"))

    # Read max_aa_seq_len from file metadata (optional, defaults to 0)
    max_aa_seq_len = int(file_metadata.get("max_aa_seq_len", "0"))

    return train_filter, val_filter, handler, max_aa_seq_len


def _combine_component_strategies(
    component_strategies: List[Tuple[str, Optional[BranchLengthInitStrategy]]],
    path_strategies: Dict[str, Optional[BranchLengthInitStrategy]],
) -> Optional[BranchLengthInitStrategy]:
    """One BL init strategy for a multiname spec's rows.

    Returns the components' strategy when they all agree, as before issue
    #1289, and otherwise a ``PerSourceStrategy`` keyed by file path.

    Raises:
        ValueError: If some components carry a strategy and others do not:
            the strategy-less rows would have no initial branch length.
    """
    first = component_strategies[0][1]
    if all(strategy == first for _, strategy in component_strategies):
        return first
    missing = [component for component, s in component_strategies if s is None]
    if missing:
        raise ValueError(
            f"Multiname components {missing} carry no bl_init_strategy while "
            f"others do; their rows would have no initial branch length."
        )
    return PerSourceStrategy(
        {
            path: strategy
            for path, strategy in path_strategies.items()
            if strategy is not None
        }
    )


def _reject_raw_pcp_parquet(
    component: str,
    comp_paths: Sequence[Path],
    comp_files: Sequence[pq.ParquetFile],
    comp_metadata: Sequence[Mapping[str, str]],
    segment_handler: SegmentHandler,
) -> None:
    """Refuse a component file with neither a seam stamp nor rate columns.

    That is what raw PCP output looks like (pcp-pipeline's Parquet, for one):
    no stamping writer has run, so the rows the CSV and MLM paths drop
    (naughty clonal families) are still in it. An unstamped file that does
    carry rates was written by a pre-#1131 writer and only warns (see
    :func:`verify_parquet_seam_stamp`). A stamped rate-free file is fine: the
    configured neutral model computes its rates at load (issue #1182).

    Raises:
        ValueError: If any file lacks both.
    """
    raw = [
        path
        for path, parquet_file, metadata in zip(
            comp_paths, comp_files, comp_metadata, strict=True
        )
        if "row_preprocessing_version" not in metadata
        and not parquet_carries_neutral_columns(
            [path], [parquet_file.schema_arrow.names], segment_handler
        )
    ]
    if raw:
        raise ValueError(
            f"Parquet component '{component}' has file(s) with neither a seam "
            f"stamp nor nt_rates/nt_csps columns: {[str(p) for p in raw]}. "
            "That is raw PCP output, which no stamping writer has "
            "preprocessed, so the naughty clonal families the CSV and MLM "
            "paths drop are still in it. Register it as a dataset and package "
            "it first:\n"
            "  pixi run python scripts/write_parquet.py <dataset> "
            "<output.parquet> --omit-neutral-columns\n"
            "The output stays rate-free, so the training config's neutral "
            "model still computes its rates at load."
        )


def _resolve_parquet_paths_and_schema(
    data_spec: str,
    strict_seam: bool,
) -> Tuple[List[Path], str, SegmentHandler, int, FrozenSet[str], int]:
    """Resolve Parquet data spec to file paths, handler, and max sequence length.

    Reads only Parquet schema metadata (column names and file-level
    key-value pairs) -- does not compute train/val filters, so it works
    for files that lack an ``in_train`` column.

    Also verifies each component's row-preprocessing seam stamp against the
    current registry (issue #1131). This is the load-time counterpart of the
    stamp the Parquet writers emit; it lives here rather than in
    ``train_command`` so every caller that reaches Parquet data -- eval,
    o/e plots, bootstraps, one-off experiment scripts -- is covered.

    I/O cost: this opens every file's footer, where the pre-#1131 version read
    only ``comp_paths[0]``'s schema. On a many-shard corpus that is O(shards)
    footer reads instead of one -- but every caller here goes on to build a
    ``ParquetBackend``, whose ``__init__`` already opens *every* path and walks
    *every* row group's metadata (``parquet_backend.py``), and the train/val
    loaders build two of them. The seam check therefore adds at most one more
    footer read per file on top of a strictly heavier pass that was always
    unconditional, before any row data is streamed. Reading only the first
    shard is not an option: the stamp (``num_rows`` especially) is per file, so
    a directory-backed component checked off one shard leaves the rest
    unverified and compares one file's count against the whole component.

    Args:
        data_spec: File path, registered nickname, or multiname.
        strict_seam: Raise rather than warn when a component's seam stamp is
            absent or disagrees. True only when a ``cv_split`` will split on
            row counts.

    Returns:
        Tuple of (paths, nickname, segment_handler, max_aa_seq_len,
        column_names, total_num_rows). ``total_num_rows`` is summed from the
        same footers the seam check reads, so callers needing a row count
        (``_cv_split_train_filter``'s row-positional mode) take it from here
        rather than paying for a second pass.

    Sets the returned handler's ``bl_init_strategy``: the components' shared
    strategy, or a ``PerSourceStrategy`` giving each file's rows its own
    component's strategy when they differ (issue #1289).

    Raises:
        ValueError: If segment handlers differ across multiname components,
            if some components carry a bl_init_strategy and others do not,
            if a file has neither a seam stamp nor rate columns (raw PCP
            output; :func:`_reject_raw_pcp_parquet`, regardless of
            ``strict_seam``), or if ``strict_seam`` and a component's seam
            stamp does not hold.
    """
    nickname = determine_nickname_from_data_spec(data_spec)
    components = parse_multiname(data_spec)

    all_paths: List[Path] = []
    handler: Optional[SegmentHandler] = None
    max_aa_seq_len = 0
    all_column_names: FrozenSet[str] = frozenset()
    total_num_rows = 0
    component_strategies: List[Tuple[str, Optional[BranchLengthInitStrategy]]] = []
    path_strategies: Dict[str, Optional[BranchLengthInitStrategy]] = {}

    for component in components:
        comp_resolved = resolve_data_path(component)
        comp_paths = _resolve_single_parquet_component(comp_resolved, component)
        all_paths.extend(comp_paths)

        # Read schema metadata only (no data columns). Every file in the
        # component is opened, not just the first: the stamp -- ``num_rows``
        # especially -- is written per file, so a directory-backed component
        # verified off ``comp_paths[0]`` alone would leave its other shards
        # unchecked and compare one file's stamped count against the whole
        # component. Handler and max_aa_seq_len still come from the first
        # file, as before.
        comp_files = [pq.ParquetFile(path) for path in comp_paths]
        comp_metadata = [
            read_parquet_file_metadata(parquet_file) for parquet_file in comp_files
        ]
        comp_num_rows = sum(
            parquet_file.metadata.num_rows for parquet_file in comp_files
        )
        total_num_rows += comp_num_rows
        column_names = set(comp_files[0].schema_arrow.names)
        comp_handler = detect_compatible_handler(column_names, ("parent", "child"))
        # Before the seam check: on a raw file its "no seam stamp" warning
        # would precede this error and say less.
        _reject_raw_pcp_parquet(
            component, comp_paths, comp_files, comp_metadata, comp_handler
        )
        verify_parquet_seam_stamp(
            component,
            list(zip(comp_paths, comp_metadata)),
            comp_num_rows,
            # Per component, not the whole spec's nickname: a multiname joins
            # components with different manifests, so each is checked against
            # the naughty list that actually governs it.
            component_nickname=registered_nickname_or_none(
                determine_nickname_from_data_spec(component)
            ),
            strict=strict_seam,
        )

        # Per-component checking is required rather than incidental: naughty
        # lists are per-dataset, so each component is checked against its own
        # source nickname above before its schema is folded into the combined
        # view below.
        file_metadata = comp_metadata[0]
        comp_max = int(file_metadata.get("max_aa_seq_len", "0"))

        # Collected per component and set on the handler after the loop: the
        # handler is a STANDARD_SEGMENT_HANDLERS singleton shared by every
        # component of its type, so it can hold only one strategy.
        comp_strategy = deserialize_strategy(file_metadata)
        component_strategies.append((component, comp_strategy))
        for path in comp_paths:
            path_strategies[str(path)] = comp_strategy

        if handler is None:
            handler = comp_handler
            all_column_names = frozenset(column_names)
        elif type(handler) is not type(comp_handler):
            raise ValueError(
                f"Incompatible segment handlers across multiname components: "
                f"{type(handler).__name__} vs {type(comp_handler).__name__}"
            )

        max_aa_seq_len = max(max_aa_seq_len, comp_max)

    assert handler is not None, "No parquet components resolved"
    # Assigned unconditionally, including to None: skipping the assignment for
    # an unstamped spec would let a strategy left on the singleton by an
    # earlier load leak into this one (issue #814 review).
    handler.bl_init_strategy = _combine_component_strategies(
        component_strategies, path_strategies
    )
    return (
        all_paths,
        nickname,
        handler,
        max_aa_seq_len,
        all_column_names,
        total_num_rows,
    )


def _resolve_parquet_train_val(
    data_spec: str,
    cv_split: Optional["CvSplitConfig"] = None,
) -> Tuple[List[Path], str, List[bool], List[bool], SegmentHandler, int]:
    """Resolve Parquet data spec and read train/val split metadata.

    Shared helper for ``_load_evolutionary_train_val_parquet`` and
    ``_load_whichmut_train_val_parquet``.

    Delegates path resolution, handler detection, and max_aa_seq_len
    extraction to ``_resolve_parquet_paths_and_schema``, then layers on
    train/val splitting:

    - ``cv_split is None`` (default): per-component splitting via
      ``_parquet_metadata``. Per-component is required for the holdout
      fallback because different datasets may share sample names but
      define disjoint holdout sets.
    - ``cv_split`` set: one RNG draw / GroupKFold partition over the
      concatenated rows of all components. Per-component splitting
      would re-seed the same RNG for each component (biasing the first
      ``min(N_i)`` rows of each toward identical fold ids) and would
      partition each component's groups into its own folds rather than
      the combined dataset's groups.

    A ``cv_split`` also makes the seam-stamp check strict (issue #1131):
    ``GroupKFold`` packs groups by row count, so a Parquet file holding a
    different row set than the CSV/MLM paths would silently realize a
    different fold map. On the ``in_train``/holdout path row counts do not
    drive the split, so a bad stamp only warns.

    Args:
        data_spec: File path, registered nickname, or multiname.
        cv_split: Optional cross-validation fold spec.

    Returns:
        Tuple of (paths, nickname, train_filter, val_filter, handler,
        max_aa_seq_len).
    """
    # Resolve paths, handler, and max_aa_seq_len (shared with schema-only path)
    all_paths, nickname, handler, max_aa_seq_len, _, total_num_rows = (
        _resolve_parquet_paths_and_schema(data_spec, strict_seam=cv_split is not None)
    )

    if cv_split is not None:
        # ``total_num_rows`` is already summed from the footers the seam check
        # read, so the row-positional mode need not reopen every file.
        all_train_filter = _cv_split_train_filter(
            all_paths, cv_split, num_rows=total_num_rows
        )
    else:
        # Per-component splitting (preserves per-dataset holdout semantics).
        components = parse_multiname(data_spec)
        all_train_filter = []
        for component in components:
            comp_resolved = resolve_data_path(component)
            comp_paths = _resolve_single_parquet_component(comp_resolved, component)
            comp_nickname = determine_nickname_from_data_spec(component)
            comp_train, _comp_val, _comp_handler, _comp_max = _parquet_metadata(
                comp_paths, nickname=comp_nickname
            )
            all_train_filter.extend(comp_train)

    all_val_filter = [not v for v in all_train_filter]
    return (
        all_paths,
        nickname,
        all_train_filter,
        all_val_filter,
        handler,
        max_aa_seq_len,
    )






def _load_whichmut_train_val_parquet(
    data_spec: str,
    neutral_model: Optional["RatesCSPNeutralModel"] = None,
    buffer_size: int = 8192,
    cv_split: Optional["CvSplitConfig"] = None,
    censor_table: Optional[Dict[str, CensorSpec]] = None,
    censor_mode: CensorMode = "both",
    reference_base_idxs: Optional[torch.Tensor] = None,
    cache_budget: CacheBudgetSpec = CacheBudgetSpec(),
) -> Tuple[DatasetBundle, DatasetBundle]:
    """Load Whichmut train/val datasets from Parquet.

    Uses ``cv_split`` when provided; otherwise reads the ``in_train``
    column or falls back to holdout-based splitting.

    Censoring is applied to **both** splits. Applying it to train alone would
    make validation loss incomparable to training loss for a reason unrelated to
    generalization.
    """
    paths, nickname, train_filter, val_filter, handler, max_aa_seq_len = (
        _resolve_parquet_train_val(data_spec, cv_split=cv_split)
    )

    train_bundle = create_whichmut_dataset_from_parquet(
        parquet_paths=paths,
        segment_handler=handler,
        max_aa_seq_len=max_aa_seq_len,
        neutral_model=neutral_model,
        buffer_size=buffer_size,
        nickname=f"{nickname}_train",
        row_filter=train_filter,
        censor_table=censor_table,
        censor_mode=censor_mode,
        reference_base_idxs=reference_base_idxs,
        cache_budget=cache_budget,
    )

    val_bundle = create_whichmut_dataset_from_parquet(
        parquet_paths=paths,
        segment_handler=handler,
        max_aa_seq_len=max_aa_seq_len,
        neutral_model=neutral_model,
        buffer_size=buffer_size,
        nickname=f"{nickname}_val",
        row_filter=val_filter,
        censor_table=censor_table,
        censor_mode=censor_mode,
        reference_base_idxs=reference_base_idxs,
        cache_budget=cache_budget,
        # Validation sweeps every row once and reduces to scalars, so its order
        # is not observable -- but shuffling it is, in wall-clock. See
        # ``ShuffleBufferSampler``: on the v11 S corpus the shuffled val loader
        # ran 20x slower than the sequential one, which made evaluation 97% of
        # training wall-clock at ``eval_every_steps: 2000``.
        shuffle=False,
    )

    return train_bundle, val_bundle


# ---------------------------------------------------------------------------
# Factory: AAO Whichmut
# ---------------------------------------------------------------------------






AAO_WHICHMUT_OUTPUT_KEYS = frozenset(
    {
        "aa_parents_idxs",
        "aa_children_idxs",
        "aao_neutral_rates",
        "nonsyn_mut_indicator",
        "layout",
        "mask",
        "loss_mask",
    }
)










# ---------------------------------------------------------------------------
# AAO / AAH train/val loaders
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Factory: DMS
# ---------------------------------------------------------------------------


class _DMSDatasetAdapter(torch.utils.data.Dataset):
    """Lightweight adapter that wraps DMS tensors for the unified pipeline.

    Stores measurement tensors once and builds per-item dicts in __getitem__,
    exactly matching the eager DMSDataset's memory model: shared WT tensor
    references, no per-sample dict allocation at init.

    Implements SequenceProvider protocol for ESM2 compatibility.
    """

    row_filter_digest: Optional[str] = None
    """Never row-filtered; part of the ``CacheableDataset`` protocol (#1672)."""

    def __init__(
        self,
        aa_parents_idxs: torch.Tensor,
        mask: torch.Tensor,
        dms_positions: torch.Tensor,
        dms_wt_aa_idxs: torch.Tensor,
        dms_mut_aa_idxs: torch.Tensor,
        dms_func_scores: torch.Tensor,
        wt_aa_sequence: str,
        nickname: str,
    ) -> None:
        self._aa_parents_idxs = aa_parents_idxs
        self._mask = mask
        self._dms_positions = dms_positions
        self._dms_wt_aa_idxs = dms_wt_aa_idxs
        self._dms_mut_aa_idxs = dms_mut_aa_idxs
        self._dms_func_scores = dms_func_scores
        self._aa_parents = [wt_aa_sequence]
        self._max_aa_seq_len = len(wt_aa_sequence)
        self.nickname = nickname

    def __len__(self) -> int:
        return len(self._dms_positions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "aa_parents_idxs": self._aa_parents_idxs,  # shared reference
            "mask": self._mask,  # shared reference
            "dms_position": self._dms_positions[idx],
            "dms_wt_aa_idx": self._dms_wt_aa_idxs[idx],
            "dms_mut_aa_idx": self._dms_mut_aa_idxs[idx],
            "dms_func_score": self._dms_func_scores[idx],
        }

    @property
    def aa_parents(self) -> List[str]:
        return self._aa_parents

    @property
    def max_aa_seq_len(self) -> int:
        return self._max_aa_seq_len

    @property
    def seq_len(self) -> int:
        """Sequence length, matching eager DMSDataset interface."""
        return self._max_aa_seq_len




# ---------------------------------------------------------------------------
# Factory: HERMES (Structural Guidance)
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# AAH integration
# ---------------------------------------------------------------------------


class DatasetPair:
    """Train and validation dataset bundles for a task."""

    def __init__(self, train: DatasetBundle, val: DatasetBundle) -> None:
        self.train = train
        self.val = val


class AAHDatasets:
    """Dataset bundles for AAH (Amino Acid Hybrid) training.

    AAH combines two data sources:
    - whichmut: Codon-level (NT) data that anchors the gauge
    - aao: Amino-acid-only data that uses the anchored gauge
    """

    def __init__(self, whichmut: DatasetPair, aao: DatasetPair) -> None:
        self.whichmut = whichmut
        self.aao = aao


