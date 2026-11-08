"""Branch length optimization hooks for evolutionary training."""

# mypy: disable-error-code=operator

import math
import time
import tracemalloc
from dataclasses import dataclass, fields
from math import ceil
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    cast,
)

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dasm2.data.evolutionary_dataset import (
    bl_batch_to_torch,
    fetch_dataloader_kwargs,
    fork_worker_kwargs,
    make_worker_init_fn,
    max_fetch_worker_count,
)
from dasm2.heads.types import DASM_LOSS_HEAD_TYPES, HeadType
from dasm2.losses.capability import (
    MissingTaskLossError,
    is_qdasm_family_loss,
    lookup_task_loss_for_head,
    resolve_dasm_normalization,
)
from dasm2.losses.loss_utils import q_family_global_kmer_inputs
from dasm2.models.core import CoreModel
from dasm2.molevol.branch_length_init import (
    initialize_branch_lengths_from_mutation_frequency,
    selection_adjusted_bl_estimate,
)
from dasm2.molevol.branch_optimization import optimize_branch_length
from dasm2.molevol.hybrid import (
    build_hybrid_neutral_q_all,
    compute_hybrid_log_codon_probs,
)
from dasm2.molevol.mutsel import (
    DASMNormalization,
    compute_dasm_log_pcp_fast,
    precompute_pcp_constants,
)
from dasm2.molevol.neutral_evolution import neutral_codon_probs_with_padding
from dasm2.molevol.q_matrix import shared_codon_length
from dasm2.molevol.qdasm import build_qdasm_q_all, compute_qdasm_log_codon_probs
from dasm2.sequence.constants import AMBIGUOUS_CODON_IDX
from dasm2.sequence.conversion import count_codon_nt_differences
from dasm2.training.hooks.base import TrainingHook
from dasm2.utils.batch_operations import move_batch_to_device
from dasm2.utils.device import get_current_rss_mb, get_process_tree_rss_mb
from dasm2.utils.parallel import chunked_async_spawn_map
from dasm2.utils.tensor import (
    csps_to_4d,
    numpy_to_tensors,
    pad_for_gather,
    rank_stride_local_n,
    rank_stride_pad_len,
    reassemble_rank_stride,
    tensors_to_numpy,
)
from dasm2.utils.torch_utils import pin_single_thread

if TYPE_CHECKING:
    from dasm2.config import HookConfig
    from dasm2.molevol.tandem import TandemTerm
    from dasm2.neutral.hit_class import HitClassModel
    from dasm2.training.compute_backend import ComputeBackend
    from dasm2.training.trainer import Trainer


@dataclass
class PerSequenceBLInputs:
    """Per-sequence data for branch length optimization.

    Stores unpadded, variable-length tensors for a single sequence,
    eliminating the need for pad-stack-unstack during BL optimization.

    Attributes:
        mask: Loss mask [actual_aa_len].
        codon_children_idxs: Child codon indices [actual_aa_len].
        codon_parents_idxs: Parent codon indices [actual_aa_len].
        layout: Segment layout [actual_aa_len] (1-based segment IDs, 0=separator).
        branch_length: Current branch length scalar.
        nt_rates: Nucleotide rates [actual_nt_len].
        nt_csps: Nucleotide CSPs [actual_nt_len, 4].
        log_selection_factors: Model output [actual_aa_len, 20], set after forward.
    """

    mask: torch.Tensor
    codon_children_idxs: torch.Tensor
    codon_parents_idxs: torch.Tensor
    layout: torch.Tensor
    branch_length: float
    nt_rates: torch.Tensor
    nt_csps: torch.Tensor
    warm_estimate: Optional[float] = None
    log_selection_factors: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        for name in ("mask", "nt_rates", "nt_csps", "layout"):
            t = getattr(self, name)
            if t.is_cuda:
                raise ValueError(
                    f"PerSequenceBLInputs.{name} must be on CPU for spawn-based "
                    f"multiprocessing, got device {t.device}"
                )

    def __getstate__(self) -> Dict[str, Any]:
        """Convert tensors to numpy for pickling, bypassing PyTorch sharing."""
        return {f.name: tensors_to_numpy(getattr(self, f.name)) for f in fields(self)}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore tensors from numpy arrays after unpickling."""
        for key, val in state.items():
            object.__setattr__(self, key, numpy_to_tensors(val))


@dataclass
class BranchLengthOptimizationInputs:
    """Container for branch length optimization inputs with chunking support.

    Stores per-sequence data as a list of ``PerSequenceBLInputs``, avoiding
    padding overhead. Shared attributes (multihit_model, convergence_tolerance)
    are kept at the container level.

    Attributes:
        samples: List of per-sequence optimization inputs.
        multihit_model: Optional multihit model instance.
        convergence_tolerance: Scipy optimization tolerance.
        normalization: DASM endpoint kernel the objective evaluates
            (issue #1238). Resolved once off the checkpoint's own loss by
            :func:`~dasm2.losses.capability.resolve_dasm_normalization`;
            without it a conditionally-trained checkpoint would get
            natively-fitted branch lengths.
    """

    samples: List[PerSequenceBLInputs]
    multihit_model: Optional["HitClassModel"]
    convergence_tolerance: float
    codon_pref_log: Optional[torch.Tensor] = None
    normalization: DASMNormalization = "parent_absorbing"
    tandem: Optional["TandemTerm"] = None

    def __len__(self) -> int:
        """Return number of sequences in this input collection."""
        return len(self.samples)


# Batch size for computing selection factors during branch length optimization.
# 128 is the smallest batch within 5% of the best producer rate for the 16M,
# 32M and 64M Compara checkpoints, with BF16 autocast off and on, at two v11
# offsets. Its peak allocated memory on the longest rows is at most 9.0 GB, on
# top of whatever the training process already holds (issue #1293,
# experiments/compara-training/results/bench_bl_workers/).
DEFAULT_SELECTION_FACTOR_BATCH_SIZE = 128


def _selection_factor_forward(
    model: CoreModel,
    batch: Dict[str, Any],
    head_type: HeadType,
    device: str,
    use_autocast: bool,
) -> torch.Tensor:
    """Run the BL forward for ``head_type`` and return fp32 log factors on host.

    With ``use_autocast`` the forward runs under BF16 autocast, as the
    trainer's own forward does when ``use_mixed_precision`` is set (issue
    #1293). The head's output is cast back to fp32 so the scipy optimizer
    sees the same dtype either way.
    """
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=use_autocast,
    ):
        outputs = model.forward(batch, head_names=[head_type])
    log_sf: torch.Tensor = outputs[head_type].float().cpu()  # mypy: outputs is Any
    return log_sf


# Evolutionary head types whose BL likelihood is the codon CTMC P = exp(tQ),
# not the post-hoc DASM factor model. These need the GPU-batched BL path.
_QDASM_FAMILY_HEADS = (HeadType.QDASM_EVOL_HEAD, HeadType.HYBRID_EVOL_HEAD)


# Default upper bound on sequences per optimization chunk. Actual chunk size may
# be smaller to ensure at least num_workers chunks exist for good parallelism.
# This is only the default; it is overridable per run via the
# ``bl_optimization_max_chunk_size`` TrainingConfig field, which threads through
# ``BranchLengthOptimizer(max_chunk_size=...)``. A large per-cycle corpus that
# OOM-kills a worker at the cycle boundary (issue #724) should lower it (e.g.
# 250) rather than change this global default, which affects every experiment.
MAX_OPTIMIZATION_CHUNK_SIZE = 1000


# Scipy workers one BL loader worker keeps fed (issue #1276). Before the
# loader workers, the parent read every row itself, capping one rank near 442
# PCPs/s at any scipy worker count. One process reads and transforms a v10 row
# in ~1 ms, and a scipy worker optimizes one in ~23 ms (44 PCPs/s per worker
# on conatus in #1270); 4 keeps the loader well ahead of the pool.
SCIPY_WORKERS_PER_BL_LOADER_WORKER = 4


def bl_loader_worker_count(num_workers: int) -> int:
    """DataLoader workers that read rows for ``num_workers`` scipy workers."""
    return max(1, ceil(num_workers / SCIPY_WORKERS_PER_BL_LOADER_WORKER))


class _RankSlicedBLSource:
    """Per-rank view of a BL source for sharded post-cycle re-optimization.

    Exposes the same surface that ``optimize_branch_lengths_for_dataset`` and
    ``optimize_branch_lengths_for_dataset_qdasm`` read off a BL source
    (``BranchLengthDatasetBundle`` / ``ESM2DatasetWrapper``), but restricted
    to the rows the calling rank owns under a rank-stride mapping
    (``indices[rank::world_size]``).

    The wrapper keeps the optimization functions unchanged: each rank simply
    optimizes ``len(indices)`` sequences as if it owned that shorter dataset.
    Sharding logic — slice → optimize → all_gather → reassemble → update
    — lives in ``BranchLengthOptimizer._optimize_and_update_datasets``.
    """

    def __init__(self, base: Any, indices: torch.Tensor) -> None:
        # indices: 1-d LongTensor of original sample indices owned by this
        # rank, in original order. ``Subset`` accepts a Python list of ints.
        if indices.ndim != 1:
            raise ValueError(
                f"rank indices must be 1-d, got shape {tuple(indices.shape)}"
            )
        self._base = base
        self._indices: List[int] = indices.tolist()
        self._dataset = Subset(base.dataset, self._indices)

    @property
    def indices(self) -> List[int]:
        return self._indices

    @property
    def dataset(self) -> Any:
        return self._dataset

    @property
    def branch_lengths(self) -> torch.Tensor:
        full: torch.Tensor = self._base.branch_lengths
        return full[self._indices]

    @property
    def segment_handler(self) -> Any:
        return self._base.segment_handler

    @property
    def collator(self) -> Any:
        return self._base.collator

    @property
    def multihit_model(self) -> Any:
        return self._base.multihit_model

    @property
    def neutral_model(self) -> Any:
        return self._base.neutral_model

    @property
    def edge_features(self) -> Dict[str, List[Any]]:
        raw: Dict[str, List[Any]] = self._base.edge_features
        return {
            name: [values[i] for i in self._indices] for name, values in raw.items()
        }

    def create_bl_optimization_dataloader(
        self, batch_size: int = 32, num_workers: int = 0
    ) -> Optional[DataLoader]:
        """Return a DataLoader that emits only this rank's stride slice.

        Delegates to the base bundle to build the full BL-optimization
        dataset (with the streaming output keys and BL-aware collator),
        then wraps its dataset in a ``Subset`` indexed by the rank's
        original-sample indices. The resulting DataLoader yields the rank's
        rows in the same order they appear in the full dataset.
        """
        full_loader = self._base.create_bl_optimization_dataloader(
            batch_size=batch_size, num_workers=num_workers
        )
        if full_loader is None:
            return None
        # ``full_loader.dataset`` is a separate ``TransformDataset`` from
        # ``self._dataset`` (different ``output_keys``), but both wrap the
        # base bundle's parquet backend and row converter in original sample
        # order — so ``self._indices`` (computed against ``base.dataset``)
        # indexes ``full_loader.dataset`` to the same underlying rows.
        sliced_dataset = Subset(full_loader.dataset, self._indices)
        return DataLoader(
            sliced_dataset,
            batch_size=full_loader.batch_size,
            shuffle=False,
            collate_fn=full_loader.collate_fn,
            worker_init_fn=make_worker_init_fn(),
            **fork_worker_kwargs(num_workers),
        )

    def get_branch_length_optimization_inputs(self) -> Dict[str, Any]:
        """Slice the in-memory per-sample inputs to this rank's stride.

        Calls the base bundle's in-memory loader (which materializes every
        sample), then keeps only the rank's slice. The full materialization
        is unavoidable here because the bundle doesn't expose an indexed
        accessor — but the in-memory path is only used for small datasets
        where this is a non-issue. Parquet-backed runs (the only ones large
        enough for the 4 MB all_gather to matter) take the streaming path
        via ``create_bl_optimization_dataloader``.
        """
        full = self._base.get_branch_length_optimization_inputs()
        sliced_samples = [full["samples"][i] for i in self._indices]
        return {
            "samples": sliced_samples,
            "multihit_model": self._base.multihit_model,
        }


def _calculate_optimal_worker_count() -> int:
    """Calculate optimal worker count for branch length optimization.

    Follows netam.framework.find_optimal_branch_lengths: use half of available
    CPU cores, capped at ``MAX_FETCH_WORKERS``, to balance parallelization
    against per-process overhead. Delegates to
    ``evolutionary_dataset.max_fetch_worker_count`` — the formula, not just the
    cap, so the two parallel fetch paths cannot drift apart about how much of a
    shared host one job may take.

    Returns:
        Optimal number of worker processes (minimum 1, maximum
        ``MAX_FETCH_WORKERS``)
    """
    return max_fetch_worker_count()


class BranchLengthOptimizer(TrainingHook):
    """Symmetric pre/post cycle hook for branch length optimization with caching support."""

    def __init__(
        self,
        num_workers: Optional[int] = None,
        parallel: bool = True,
        convergence_tolerance: float = 1e-4,
        force_recompute: bool = False,
        max_chunk_size: int = MAX_OPTIMIZATION_CHUNK_SIZE,
        freeze: bool = False,
        freeze_after_first_cycle: bool = False,
        init_from_input: bool = False,
        recompute_on_resume: bool = False,
    ) -> None:
        """Initialize branch length optimization hook with caching support.

        Args:
            num_workers: Number of parallel workers for multiprocessing optimization.
                Only used when parallel=True. If None (default), automatically calculates
                optimal worker count using netam's strategy: min(cpu_count // 2, 10).
                Serial optimization uses single process regardless of this setting.
            parallel: Enable multiprocessing for branch length optimization.
                True: Use multiprocessing with num_workers processes (faster for large datasets).
                False: Use serial optimization (deterministic, easier debugging).
            convergence_tolerance: Convergence tolerance for scipy optimization.
                Default 1e-4 provides equivalent optima with faster convergence
                than the previous 1e-6 default (see issue #499).
            force_recompute: Force recomputation even if cache exists (default: False).
                Cache is still updated with newly computed values.
            max_chunk_size: Upper bound on sequences per optimization chunk
                (default ``MAX_OPTIMIZATION_CHUNK_SIZE``). Bounds the BL-optimization
                host-RAM working set — lower it on a large per-cycle corpus that
                OOM-kills a worker at the cycle boundary (issue #724).
            freeze: If True, ``pre_cycle`` and ``post_cycle`` are no-ops:
                branch lengths already present in the dataset are used
                unchanged for the entire run, with no cold-start init, cache
                lookup, or model-based re-optimization at any cycle boundary
                (default: False, historical behavior). Needed to isolate a
                training comparison from the BL-estimation pathway (issue
                #821) — without this, ``pre_cycle``'s cold-start fallback and
                ``post_cycle``'s unconditional re-optimization both overwrite
                the input branch lengths even in a single-cycle run.
            freeze_after_first_cycle: If True, ``pre_cycle`` runs normally
                (so cycle 0 still gets its cold-start init / cache lookup /
                optimization) but ``post_cycle`` is a no-op at every cycle:
                "optimize once, then freeze" (default: False). This is the
                weaker sibling of ``freeze`` used by the issue #199 /  #713
                frozen-BL retrain, where the model must still train against a
                sensible BL estimate but must not be allowed to co-adapt BL to
                its own suppressed selection across cycles. Mutually exclusive
                with ``freeze``, which pins BL for the whole run including
                cycle 0.
            init_from_input: If True, cycle 0's cold-start ``pre_cycle`` is a
                no-op, so cycle 0 trains at the branch lengths already present
                in the dataset (its ``branch_length`` column), while
                ``post_cycle`` re-optimizes as usual: "start from the input,
                then alternate" (default: False). Without it, a handler with a
                calibrated ``bl_init_strategy`` overwrites the input with its
                mutation-frequency estimate before cycle 0 (issue #1503's
                alternating arm started from codon-CTMC branch lengths). The
                resume entry point (``start_cycle != 0``) still loads the
                cache. Cycle 0 raises ``ValueError`` if the train or val
                data has no ``branch_length`` column (in every Parquet file),
                since its lengths would then be ``DEFAULT_BRANCH_LENGTH``.
                Mutually exclusive with ``freeze`` and
                ``freeze_after_first_cycle``: combining it with either
                raises ``ValueError``.
            recompute_on_resume: If True, ``pre_cycle``'s resume-entry-point
                (hit when starting mid-pipeline at cycle ``start_cycle != 0``)
                falls back to model-based re-optimization on a branch-length
                cache miss, instead of raising (default: False, historical
                behavior — issue #516). Default False guards against
                silently diverging from an *interrupted* run's own BL
                history when its cache is missing or was clobbered. Set
                True only to *extend* a completed run past its originally
                planned final cycle, where no historical cache exists to
                protect (cycle ``start_cycle`` never ran before) and
                recomputing is exactly what an uninterrupted longer run's
                cycle boundary would have done.

        Note:
            Branch length optimization uses scipy.optimize.minimize_scalar with
            warm-started bounds centered on a closed-form estimate (see issue #499).

            Worker count calculation (when num_workers=None):
                Uses netam's strategy: min(cpu_count() // 2, 10)
                This typically provides 8-10 workers on modern machines, offering
                significant speedup compared to the previous default of 4 workers.

            Cache configuration is owned by the Trainer, not the hook. This allows
            cache to be shared across multiple hooks. The hook uses trainer.cache
            if trainer.enable_cache is True.

            For each sequence, the optimizer finds the branch length that minimizes
            the DASM2 evolutionary loss by recomputing neutral codon probabilities
            for trial branch lengths during optimization.

            The hook accesses model and datasets through the trainer and dataset
            parameters passed to pre_cycle() and post_cycle() methods.
        """
        self.num_workers = (
            num_workers
            if num_workers is not None
            else _calculate_optimal_worker_count()
        )
        self.parallel = parallel
        self.convergence_tolerance = convergence_tolerance
        self.force_recompute = force_recompute
        self.max_chunk_size = max_chunk_size
        if freeze and freeze_after_first_cycle:
            raise ValueError(
                "freeze and freeze_after_first_cycle are mutually exclusive: "
                "freeze pins branch lengths for the whole run (pre_cycle and "
                "post_cycle both no-op), while freeze_after_first_cycle lets "
                "cycle 0's pre_cycle optimize and only freezes thereafter."
            )
        if init_from_input and (freeze or freeze_after_first_cycle):
            raise ValueError(
                "init_from_input is mutually exclusive with freeze and "
                "freeze_after_first_cycle: it keeps the input branch lengths "
                "for cycle 0 and re-optimizes after every cycle, so combined "
                "with either freeze it would pin the input for the whole run, "
                "which is freeze."
            )
        self.freeze = freeze
        self.freeze_after_first_cycle = freeze_after_first_cycle
        self.init_from_input = init_from_input
        self.recompute_on_resume = recompute_on_resume

    def to_config(self) -> "HookConfig":
        """Reconstruct HookConfig from hook state.

        Returns:
            HookConfig matching current hook configuration
        """
        from dasm2.config import HookConfig

        return HookConfig(  # type: ignore[call-arg]
            type="BranchLengthOptimizer",
            num_workers=self.num_workers,
            parallel=self.parallel,
            convergence_tolerance=self.convergence_tolerance,
            force_recompute=self.force_recompute,
            max_chunk_size=self.max_chunk_size,
            freeze=self.freeze,
            freeze_after_first_cycle=self.freeze_after_first_cycle,
            init_from_input=self.init_from_input,
            recompute_on_resume=self.recompute_on_resume,
        )

    def _optimize_and_update_datasets(
        self,
        model: CoreModel,
        train_bl_source: Any,
        val_bl_source: Any,
        backend: "ComputeBackend",
        head_type: HeadType,
        use_autocast: bool = False,
    ) -> None:
        """Shard BL re-optimization across ranks and update every rank.

        Each rank optimizes its rank-stride slice (``indices[rank::world_size]``)
        of every BL source via the existing chunked optimizer, then results
        are all-gathered, reassembled into the original sample order, and
        every rank calls ``update_branch_lengths`` with the full N-length
        tensor. Replaces the rank-0-only optimize-then-broadcast path that
        left ranks 1..K-1 idle for the entire wallclock of the re-opt.

        Dispatches by evolutionary head type:
            - SELECTION_FACTOR_HEAD: scipy/spawn-worker path (post-hoc selection LL).
            - QDASM_EVOL_HEAD / HYBRID_EVOL_HEAD: GPU-batched golden-section
              path that evaluates the codon CTMC ``P = exp(tQ)`` likelihood.

        Args:
            model: CoreModel for computing selection factors.
            train_bl_source: BL-capable object (BranchLengthDatasetBundle
                or ESM2DatasetWrapper).
            val_bl_source: BL-capable object for validation.
            backend: Compute backend for rank-awareness and all_gather.
            head_type: Head whose forward pass produces the L×20 log selection
                factors. Sourced from the owning task at hook-invocation time.
            use_autocast: Run the DASM path's forward under BF16 autocast
                (issue #1293). The caller passes the trainer's
                ``use_mixed_precision``.

        Note:
            ``update_branch_lengths`` (at
            ``dasm2/data/evolutionary_dataset.py:730``) requires a full
            N-length tensor — the reassembled gather output. The cache write
            stays rank-0-only (handled at the post_cycle/pre_cycle caller).

            Both sources are optimized before either is updated. This
            preserves the pre-shard semantics for the train==val case
            (e.g. small-dataset tests where the same bundle is passed for
            both): the val optimization must read the original BLs, not
            the freshly-updated train ones, otherwise GS bounds in the
            QDASM/Hybrid path widen on the second pass and the optimum
            walks off to the search ceiling.
        """
        full_n_train = int(len(train_bl_source.branch_lengths))  # type: ignore[attr-defined]
        full_n_val = int(len(val_bl_source.branch_lengths))  # type: ignore[attr-defined]
        rank = backend.process_index
        world_size = backend.num_processes

        optimized_train = self._sharded_optimize(
            model,
            train_bl_source,
            backend,
            head_type,
            full_n_train,
            rank,
            world_size,
            use_autocast=use_autocast,
        )
        optimized_val = self._sharded_optimize(
            model,
            val_bl_source,
            backend,
            head_type,
            full_n_val,
            rank,
            world_size,
            use_autocast=use_autocast,
        )

        train_bl_source.update_branch_lengths(optimized_train)  # type: ignore[attr-defined]
        val_bl_source.update_branch_lengths(optimized_val)  # type: ignore[attr-defined]

    def _sharded_optimize(
        self,
        model: CoreModel,
        bl_source: Any,
        backend: "ComputeBackend",
        head_type: HeadType,
        full_n: int,
        rank: int,
        world_size: int,
        use_autocast: bool = False,
    ) -> torch.Tensor:
        """Run one BL re-opt on this rank's slice; gather and reassemble."""
        # Rank-stride slice in original sample order. Indices are
        # ``[rank, rank + world_size, rank + 2*world_size, ...]`` truncated
        # to ``< full_n``. ``arange + slice`` is deterministic and the
        # stride mapping is reversible (gathered[r*pad_len + j] sits at
        # full[r + j*world_size]), so reassembly never needs to know the
        # per-rank lengths.
        local_n = rank_stride_local_n(full_n, world_size, rank)
        indices = torch.arange(full_n, dtype=torch.long)[rank::world_size]

        if local_n == 0:
            # World size exceeds N (degenerate but well-defined): this rank
            # produces an empty local result. Still participate in the
            # gather so other ranks don't hang.
            local_result = torch.empty(0, dtype=torch.float32)
        else:
            sliced = _RankSlicedBLSource(bl_source, indices)
            local_result = optimize_branch_lengths_dispatch(
                model,
                sliced,
                head_type=head_type,
                convergence_tolerance=self.convergence_tolerance,
                parallel=self.parallel,
                num_workers=self.num_workers,
                max_chunk_size=self.max_chunk_size,
                use_autocast=use_autocast,
            )
            local_result = local_result.detach().cpu().float()

        if world_size == 1:
            return local_result

        pad_len = rank_stride_pad_len(full_n, world_size)
        padded = pad_for_gather(local_result, pad_len)
        gathered = backend.all_gather(padded).detach().cpu().float()
        return reassemble_rank_stride(gathered, full_n, world_size)

    def _extract_and_validate_context(
        self, context: Any
    ) -> Tuple[Any, Any, int, int, HeadType]:
        """Extract BL-capable objects, cycle count, start cycle, and head type from context.

        Reads ``train_data``/``val_data`` from context. These are
        ``BranchLengthDatasetBundle`` or ``ESM2DatasetWrapper``. Both support
        ``branch_lengths``, ``update_branch_lengths()``, ``segment_handler``,
        and ``get_branch_length_optimization_inputs()``. ``head_type`` is
        passed by the trainer from the owning task and identifies which head
        produces the L×20 log selection factors used by BL optimization.

        ``start_cycle`` is optional; defaults to 0 for normal (non-resumed)
        training. When the trainer is resuming mid-pipeline, the BL hook uses
        ``cycle_count == start_cycle != 0`` as the "resume entry point" signal
        and loads BLs from the artifact cache.

        Args:
            context: Hook context dict with ``train_data``, ``val_data``,
                ``cycle_count``, ``head_type`` (and optional ``start_cycle``).

        Returns:
            Tuple of (train_bl_source, val_bl_source, cycle_count,
            start_cycle, head_type).
        """
        required = ("train_data", "val_data", "cycle_count", "head_type")
        missing = [k for k in required if k not in context]
        if missing:
            raise ValueError(f"Hook context missing required keys: {missing}")

        return (
            context["train_data"],
            context["val_data"],
            context["cycle_count"],
            int(context.get("start_cycle", 0)),
            context["head_type"],
        )

    @staticmethod
    def _cacheable_dataset(bl_source: Any) -> Any:
        """Extract cacheable dataset from a BL source.

        For bundles, returns the underlying ``TransformDataset`` (which has
        ``nickname``).
        """
        return bl_source.dataset

    def _save_branch_length_cache(
        self,
        trainer: "Trainer",
        train_bl: Any,
        val_bl: Any,
        head_type: HeadType,
    ) -> None:
        """Save branch lengths to cache for both train and val.

        Only rank 0 should call this (caller is responsible for the guard).

        ``head_type`` is part of the artifact cache key (see
        ``BranchLengthArtifact.try_load_cached`` in ``dasm2/artifacts.py``),
        so it must match the head whose BLs are being saved — otherwise a
        QDASM resume would fail to find the cached BLs that were written
        under a different head's bucket.
        """
        from dasm2.artifacts import BranchLengthArtifact

        train_cache_ds = self._cacheable_dataset(train_bl)
        val_cache_ds = self._cacheable_dataset(val_bl)
        trainer.save_cached(
            BranchLengthArtifact,
            train_bl.branch_lengths,  # type: ignore[attr-defined]
            train_cache_ds,
            "train",
            head_type=head_type,
        )
        trainer.save_cached(
            BranchLengthArtifact,
            val_bl.branch_lengths,  # type: ignore[attr-defined]
            val_cache_ds,
            "val",
            head_type=head_type,
        )

    def _reoptimize_and_cache(
        self,
        trainer: "Trainer",
        train_bl: Any,
        val_bl: Any,
        head_type: HeadType,
    ) -> None:
        """Run model-based BL re-optimization, then cache on rank 0.

        Shared by three call sites that all need the same pair: ``post_cycle``
        (unconditional re-optimization every cycle), ``pre_cycle``'s cycle-0
        cold-start fallback (no calibrated init strategy, cache miss), and
        ``pre_cycle``'s resume-entry-point fallback (``recompute_on_resume``,
        cache miss). ``_optimize_and_update_datasets`` itself shards across
        all DDP ranks; only the cache write is rank-0-gated.
        """
        self._optimize_and_update_datasets(
            trainer.unwrapped_model,
            train_bl,
            val_bl,
            trainer.backend,
            head_type,
            use_autocast=trainer.use_mixed_precision,
        )
        if trainer.is_main_process:
            self._save_branch_length_cache(trainer, train_bl, val_bl, head_type)

    def pre_cycle(
        self,
        trainer: "Trainer",
        **context: Any,
    ) -> None:
        """Pre-cycle hook: initialize branch lengths.

        Three modes:
        - **Cycle 0 cold start (default)**: if a calibrated ``bl_init_strategy``
          is available, use it; otherwise check the cache, then fall back to
          model-based optimization.

          - With ``self.init_from_input``, the cold start is a no-op: cycle 0
            trains at the dataset's input branch lengths.
        - **Cycle ``start_cycle`` resume (``start_cycle > 0``)**: hard-load
          cached BLs (under the active head's key). Raise on cache miss —
          re-running BL optimization or strategy init would silently change
          the resumed run's training distribution — unless
          ``self.recompute_on_resume`` is set, in which case a cache miss
          falls back to model-based optimization instead (issue #516; for
          extending a completed run past its planned final cycle, where no
          historical cache exists to protect).
        - **Subsequent cycles**: no-op; ``post_cycle`` of the previous cycle
          already wrote the BLs.

        When ``self.freeze`` is True, this entire method is a no-op at every
        cycle (issue #821): the branch lengths already present in the
        dataset are left untouched, bypassing cold-start init, cache lookup,
        and model-based optimization alike.
        """
        if self.freeze:
            return

        from dasm2.artifacts import BranchLengthArtifact

        (
            train_bl,
            val_bl,
            cycle_count,
            start_cycle,
            head_type,
        ) = self._extract_and_validate_context(context)

        # --- Resume entry point ----------------------------------------------
        # When resuming at start_cycle > 0, load BLs from cache. The cache key
        # includes head_type, so make sure we're looking it up under the head
        # that was used to write it (same head as the current model).
        #
        # Use LenientBranchLengthArtifact for the resume read: the strict
        # BranchLengthArtifact validates the entire training_config, including
        # hook configs (CheckpointSaveHook output_dir, notes, etc.). Those can
        # legitimately differ between an offline BL-recovery run and the
        # actual resumed training command. Lenient validates only the fields
        # that affect BL semantics: neutral_model and the loss normalization
        # (the DASM endpoint kernel, issue #1238).
        if cycle_count == start_cycle and start_cycle != 0:
            from dasm2.artifacts import LenientBranchLengthArtifact

            train_cache_ds = self._cacheable_dataset(train_bl)
            val_cache_ds = self._cacheable_dataset(val_bl)
            cached_train = trainer.try_load_cached(
                artifact_class=LenientBranchLengthArtifact,
                dataset=train_cache_ds,
                split="train",
                head_type=head_type,
            )
            cached_val = trainer.try_load_cached(
                artifact_class=LenientBranchLengthArtifact,
                dataset=val_cache_ds,
                split="val",
                head_type=head_type,
            )
            if cached_train is None or cached_val is None:
                if not self.recompute_on_resume:
                    raise RuntimeError(
                        f"Resuming at start_cycle={start_cycle} but no cached "
                        f"branch lengths found for head={head_type.value} "
                        f"(train hit={cached_train is not None}, "
                        f"val hit={cached_val is not None}). "
                        "Run the offline BL recovery script before resuming, "
                        "or pass --recompute_branch_lengths_on_resume if this "
                        "resume extends a completed run past its originally "
                        "planned final cycle (no historical cache to protect)."
                    )
                if trainer.is_main_process:
                    print(
                        f"Pre-cycle (resume @cycle={start_cycle}): no cached "
                        f"branch lengths found for head={head_type.value}; "
                        "recompute_on_resume is set, falling back to "
                        "model-based optimization..."
                    )
                self._reoptimize_and_cache(trainer, train_bl, val_bl, head_type)
                return
            if trainer.is_main_process:
                print(
                    f"Pre-cycle (resume @cycle={start_cycle}): "
                    "Loading branch lengths from cache..."
                )
            train_bl.update_branch_lengths(cached_train)  # type: ignore[attr-defined]
            val_bl.update_branch_lengths(cached_val)  # type: ignore[attr-defined]
            return

        # --- Subsequent-cycle no-op ------------------------------------------
        if cycle_count != 0:
            return

        # --- Cycle 0 from the input --------------------------------------------
        if self.init_from_input:
            for split, bl_source in (("train", train_bl), ("val", val_bl)):
                if not bl_source.branch_lengths_from_input:
                    raise ValueError(
                        f"init_branch_lengths_from_input is set, but the {split} "
                        "data has no 'branch_length' column (in every Parquet "
                        "file), so cycle 0 would train at "
                        "DEFAULT_BRANCH_LENGTH instead of the input's lengths. "
                        "Add the column, or unset the flag."
                    )
            if trainer.is_main_process:
                print("Pre-cycle: Keeping the input branch lengths for cycle 0...")
            return

        # --- Cycle 0 cold start ----------------------------------------------
        has_init_strategy = (
            train_bl.segment_handler.bl_init_strategy is not None  # type: ignore[attr-defined]
        )

        if has_init_strategy:
            # Strategy-based initialization: no model inference needed. The cache
            # check is skipped to avoid stale cached values from different model
            # architectures. This is a per-edge pass over every parent/child
            # sequence pair, so it is not free at multi-million-edge scale --
            # hence the elapsed-time log below (#992).
            if trainer.is_main_process:
                print(
                    "Pre-cycle: Initializing branch lengths from mutation frequency..."
                )
            init_start = time.perf_counter()
            train_lengths = initialize_branch_lengths_from_mutation_frequency(
                train_bl  # type: ignore[arg-type]
            )
            val_lengths = initialize_branch_lengths_from_mutation_frequency(
                val_bl  # type: ignore[arg-type]
            )
            assert train_lengths is not None and val_lengths is not None
            if trainer.is_main_process:
                print(
                    f"Pre-cycle: Initialized {len(train_lengths)} train + "
                    f"{len(val_lengths)} val branch lengths in "
                    f"{time.perf_counter() - init_start:.1f}s"
                )
            train_bl.update_branch_lengths(train_lengths)  # type: ignore[attr-defined]
            val_bl.update_branch_lengths(val_lengths)  # type: ignore[attr-defined]
        else:
            # No init strategy - check cache first, then fall back to model-based
            train_cache_ds = self._cacheable_dataset(train_bl)
            val_cache_ds = self._cacheable_dataset(val_bl)
            if not self.force_recompute:
                cached_train = trainer.try_load_cached(
                    artifact_class=BranchLengthArtifact,
                    dataset=train_cache_ds,
                    split="train",
                    head_type=head_type,
                )
                cached_val = trainer.try_load_cached(
                    artifact_class=BranchLengthArtifact,
                    dataset=val_cache_ds,
                    split="val",
                    head_type=head_type,
                )

                if cached_train is not None and cached_val is not None:
                    if trainer.is_main_process:
                        print("Pre-cycle: Loading branch lengths from cache...")
                    train_bl.update_branch_lengths(cached_train)  # type: ignore[attr-defined]
                    val_bl.update_branch_lengths(cached_val)  # type: ignore[attr-defined]
                    return  # Cache hit - done!

            # Cache miss - fall back to model-based optimization (slow).
            # Caches the result (rank-0-only write, only reached for handlers
            # without init strategy -- the has_init_strategy branch above
            # returns before this point via update_branch_lengths directly).
            handler_name = type(train_bl.segment_handler).__name__  # type: ignore[attr-defined]
            if trainer.is_main_process:
                print(
                    f"WARNING: {handler_name} has no calibrated bl_init_strategy. "
                    f"Falling back to model-based branch length optimization."
                )
            self._reoptimize_and_cache(trainer, train_bl, val_bl, head_type)

    def post_cycle(
        self,
        trainer: "Trainer",
        **context: Any,
    ) -> None:
        """Post-cycle hook: always re-optimize and update cache.

        Updates cache after every cycle with the latest optimized values.
        This ensures the cache stays fresh if model architecture or dataset changes.

        In multi-GPU mode, each rank optimizes its rank-stride slice; the
        per-rank results are all-gathered and every rank passes the
        reassembled full tensor to ``update_branch_lengths`` (see
        ``_optimize_and_update_datasets``). Cache write stays rank-0-only.

        Args:
            trainer: Trainer instance with cache support
            **context: Must include train_data and val_data.

        When ``self.freeze`` is True, this is a no-op (issue #821): the
        dataset's branch lengths are left exactly as ``pre_cycle`` left
        them (unchanged, since it was also frozen), and the cache is not
        written.

        When ``self.freeze_after_first_cycle`` is True, this is likewise a
        no-op, but ``pre_cycle`` was *not* frozen — so the branch lengths kept
        here are cycle 0's optimized values rather than the caller's inputs
        (issue #199 / #713 frozen-BL retrain).
        """
        if self.freeze or self.freeze_after_first_cycle:
            return

        train_bl, val_bl, _, _, head_type = self._extract_and_validate_context(context)

        if trainer.is_main_process:
            print("Post-cycle: Re-optimizing branch lengths...")
        self._reoptimize_and_cache(trainer, train_bl, val_bl, head_type)


def _optimize_branch_lengths_chunk(
    inputs: BranchLengthOptimizationInputs,
) -> torch.Tensor:
    """Optimize branch lengths for a chunk of sequences.

    Uses pre-computed warm-start estimates (set on each sample by
    ``optimize_branch_lengths_for_dataset``) for bounded scipy optimization.
    Falls back to selection-adjusted closed-form estimate if no warm estimate
    is set.

    Args:
        inputs: Container with per-sequence optimization inputs including
            pre-cached log selection factors and warm estimates.

    Returns:
        Tensor of optimized branch lengths [chunk_size].
    """
    optimized = []

    for sample in inputs.samples:
        mask = sample.mask
        child_codon_idxs = sample.codon_children_idxs
        parent_codon_idxs = sample.codon_parents_idxs
        if sample.log_selection_factors is None:
            raise RuntimeError(
                "log_selection_factors must be set before optimization. "
                "Call optimize_branch_lengths_for_dataset to compute them."
            )
        log_selection_factors = sample.log_selection_factors
        nt_rate = sample.nt_rates
        nt_csp = sample.nt_csps

        # Use pre-computed warm estimate; fall back to selection-adjusted
        # estimate. Both are deliberately kernel-agnostic (issue #1238):
        # they only seed the bounded search, and the objective below — which
        # *is* kernel-aware — defines the optimum.
        warm_estimate = sample.warm_estimate
        if warm_estimate is None:
            warm_estimate = selection_adjusted_bl_estimate(
                parent_codon_idxs,
                child_codon_idxs,
                nt_rate,
                mask,
                log_selection_factors,
            )

        # Pre-compute branch-length-independent constants once per sequence.
        # This avoids recomputing the selection lift, child index clamping,
        # and parent detection on every scipy iteration (~16 evals).
        pcp_consts = precompute_pcp_constants(
            parent_codon_idxs,
            child_codon_idxs,
            log_selection_factors,
            mask,
            normalization=inputs.normalization,
        )

        # Create objective function for this sequence
        def log_pcp_probability(log_branch_length: torch.Tensor) -> torch.Tensor:
            """Objective: compute log PCP probability for trial branch length."""
            bl = torch.exp(log_branch_length)

            # Compute neutral probs — no padding needed for per-sequence data
            neutral_probs = neutral_codon_probs_with_padding(
                parent_codon_idxs,
                mask,
                nt_rate,
                nt_csp,
                float(bl.item()),
                multihit_model=inputs.multihit_model,
                target_length=None,
                codon_pref_log=inputs.codon_pref_log,
                tandem=inputs.tandem,
            )

            log_prob = compute_dasm_log_pcp_fast(torch.log(neutral_probs), pcp_consts)
            return torch.tensor(log_prob)

        # Optimize with warm-started bounded optimization
        optimized_bl, _ = optimize_branch_length(
            log_pcp_probability,
            warm_estimate=warm_estimate,
            optimization_tol=inputs.convergence_tolerance,
        )
        optimized.append(float(optimized_bl))

    return torch.tensor(optimized)


def _optimize_branch_lengths_chunk_numpy(
    inputs: BranchLengthOptimizationInputs,
) -> np.ndarray:
    """Wrapper that returns numpy, avoiding PyTorch sharing on the return path."""
    return _optimize_branch_lengths_chunk(inputs).numpy()


def _generate_optimization_chunks_streaming(
    model: CoreModel,
    bl_dataloader: Iterable[Dict[str, Any]],
    multihit_model: Optional["HitClassModel"],
    convergence_tolerance: float,
    strategy: Any,
    rate_adjustments: torch.Tensor,
    chunk_size: int,
    head_type: HeadType,
    edge_features_list: Optional[List[dict]] = None,
    codon_pref_log: Optional[torch.Tensor] = None,
    use_autocast: bool = False,
    normalization: DASMNormalization = "parent_absorbing",
    tandem: Optional["TandemTerm"] = None,
) -> Iterator[BranchLengthOptimizationInputs]:
    """Yield optimization chunks by streaming from a DataLoader.

    Runs GPU model forward on the main process, accumulates samples into
    chunks, computes warm estimates, and yields ready-to-optimize chunks.
    Only ``chunk_size`` samples are held in memory at a time (plus one
    DataLoader batch being processed).

    Args:
        model: CoreModel for computing selection factors (GPU).
        bl_dataloader: DataLoader (or an iterator over one) yielding batches
            with list-collated ``nt_rates``/``nt_csps``.
        multihit_model: Optional multihit model for optimization.
        convergence_tolerance: Scipy optimization tolerance.
        strategy: BranchLengthInitStrategy, or None for selection-adjusted fallback.
        rate_adjustments: Per-segment rate multipliers [num_segments].
        chunk_size: Number of samples per optimization chunk.
        use_autocast: Run the forward under BF16 autocast (issue #1293).
        normalization: DASM endpoint kernel for the BL objective
            (issue #1238), carried onto every yielded chunk.

    Yields:
        BranchLengthOptimizationInputs with warm estimates set.
    """
    device = str(model.device)
    buffer: List[PerSequenceBLInputs] = []
    # Track edge features for buffered samples (parallel to buffer).
    ef_buffer: List[dict] = []
    global_idx = 0

    with torch.no_grad():
        for batch in bl_dataloader:
            # Loader workers hand batches over as numpy arrays.
            batch = bl_batch_to_torch(batch)
            # Pop list-collated fields before moving to device
            batch_nt_rates: List[torch.Tensor] = batch.pop("nt_rates")
            batch_nt_csps: List[torch.Tensor] = batch.pop("nt_csps")

            # Per-row inputs are sliced from the host batch, not the device copy,
            # which would cost a device-to-host sync per field and row (#1276).
            log_sf = _selection_factor_forward(
                model,
                move_batch_to_device(batch, device),
                head_type,
                device,
                use_autocast,
            )
            batch_size = log_sf.shape[0]

            for i in range(batch_size):
                actual_aa_len = batch_nt_rates[i].shape[0] // 3

                buffer.append(
                    PerSequenceBLInputs(
                        mask=batch["loss_mask"][i, :actual_aa_len],
                        codon_children_idxs=batch["codon_children_idxs"][
                            i, :actual_aa_len
                        ],
                        codon_parents_idxs=batch["codon_parents_idxs"][
                            i, :actual_aa_len
                        ],
                        layout=batch["layout"][i, :actual_aa_len],
                        branch_length=float(batch["branch_length"][i]),
                        nt_rates=batch_nt_rates[i],
                        nt_csps=batch_nt_csps[i],
                        log_selection_factors=log_sf[i, :actual_aa_len, :],
                    )
                )
                if edge_features_list is not None:
                    ef_buffer.append(edge_features_list[global_idx])
                global_idx += 1

            # Flush full chunks from the buffer.
            while len(buffer) >= chunk_size:
                chunk, buffer = buffer[:chunk_size], buffer[chunk_size:]
                ef_chunk: Optional[List[dict]] = None
                if ef_buffer:
                    ef_chunk, ef_buffer = ef_buffer[:chunk_size], ef_buffer[chunk_size:]
                _precompute_warm_estimates(chunk, strategy, rate_adjustments, ef_chunk)
                yield BranchLengthOptimizationInputs(
                    samples=chunk,
                    multihit_model=multihit_model,
                    convergence_tolerance=convergence_tolerance,
                    codon_pref_log=codon_pref_log,
                    normalization=normalization,
                    tandem=tandem,
                )

    # Flush remaining samples.
    if buffer:
        _precompute_warm_estimates(
            buffer, strategy, rate_adjustments, ef_buffer or None
        )
        yield BranchLengthOptimizationInputs(
            samples=buffer,
            multihit_model=multihit_model,
            convergence_tolerance=convergence_tolerance,
            codon_pref_log=codon_pref_log,
            normalization=normalization,
            tandem=tandem,
        )


def _generate_optimization_chunks_in_memory(
    model: CoreModel,
    bl_source: Any,
    convergence_tolerance: float,
    strategy: Any,
    rate_adjustments: torch.Tensor,
    chunk_size: int,
    head_type: HeadType,
    edge_features_list: Optional[List[dict]] = None,
    codon_pref_log: Optional[torch.Tensor] = None,
    use_autocast: bool = False,
    normalization: DASMNormalization = "parent_absorbing",
    tandem: Optional["TandemTerm"] = None,
) -> Iterator[BranchLengthOptimizationInputs]:
    """Yield optimization chunks from in-memory datasets.

    Two-phase approach: builds ``PerSequenceBLInputs`` from stored data,
    then attaches ``log_selection_factors`` via batched model forward.
    Yields chunks of ``chunk_size`` samples with warm estimates set.

    Args:
        model: CoreModel for computing selection factors.
        bl_source: BL-capable object with in-memory data.
        convergence_tolerance: Scipy optimization tolerance.
        strategy: BranchLengthInitStrategy, or None for selection-adjusted fallback.
        rate_adjustments: Per-segment rate multipliers [num_segments].
        chunk_size: Number of samples per optimization chunk.
        use_autocast: Run the forward under BF16 autocast (issue #1293).
        normalization: DASM endpoint kernel for the BL objective
            (issue #1238), carried onto every yielded chunk.

    Yields:
        BranchLengthOptimizationInputs with warm estimates set.
    """
    device = str(model.device)

    optimization_inputs_dict = bl_source.get_branch_length_optimization_inputs()  # type: ignore[attr-defined]
    samples: List[PerSequenceBLInputs] = optimization_inputs_dict["samples"]
    multihit_model = optimization_inputs_dict["multihit_model"]

    # Compute selection factors for all sequences using batched model forward
    forward_dataset = bl_source.dataset
    collate_fn = bl_source.collator

    with torch.no_grad():
        dataloader = DataLoader(
            forward_dataset,
            batch_size=DEFAULT_SELECTION_FACTOR_BATCH_SIZE,
            shuffle=False,
            **({"collate_fn": collate_fn} if collate_fn is not None else {}),
        )

        sample_idx = 0
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            log_sf = _selection_factor_forward(
                model, batch, head_type, device, use_autocast
            )
            batch_size = log_sf.shape[0]

            for i in range(batch_size):
                actual_len = samples[sample_idx].mask.shape[0]
                samples[sample_idx].log_selection_factors = log_sf[i, :actual_len, :]
                sample_idx += 1

    # Yield in chunks with warm estimates.
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start : start + chunk_size]
        ef_chunk = (
            edge_features_list[start : start + chunk_size]
            if edge_features_list
            else None
        )
        _precompute_warm_estimates(chunk, strategy, rate_adjustments, ef_chunk)
        yield BranchLengthOptimizationInputs(
            samples=chunk,
            multihit_model=multihit_model,
            convergence_tolerance=convergence_tolerance,
            codon_pref_log=codon_pref_log,
            normalization=normalization,
            tandem=tandem,
        )


def _compute_tensor_mut_freq(
    sample: PerSequenceBLInputs,
    rate_adjustments: torch.Tensor,
) -> float:
    """Compute rate-adjusted mutation frequency from codon index tensors.

    Uses the layout tensor to partition codons into segments and adjusts
    mutation counts by per-segment rate factors. This is the tensor-based
    equivalent of ``compute_adjusted_mutation_frequency`` (which works on
    nucleotide strings).

    Args:
        sample: Per-sequence BL inputs with codon indices and layout.
        rate_adjustments: Per-segment rate multipliers [num_segments].

    Returns:
        Rate-adjusted mutation frequency, or 0.0 if no valid codons.
    """
    nt_diffs = count_codon_nt_differences(
        sample.codon_parents_idxs, sample.codon_children_idxs
    )
    layout = sample.layout
    # Exclude ambiguous codons to match compute_adjusted_mutation_frequency,
    # which excludes N/gap positions from both numerator and denominator.
    valid = (sample.codon_parents_idxs != AMBIGUOUS_CODON_IDX) & (
        sample.codon_children_idxs != AMBIGUOUS_CODON_IDX
    )

    # Accumulate rate-adjusted mutation frequency across segments.
    # Layout is 1-based: segment k has layout == k.
    effective_mutations = 0.0
    total_nt_length = 0
    for seg_idx in range(len(rate_adjustments)):
        seg_mask = (layout == (seg_idx + 1)) & valid
        n_seg_codons = int(seg_mask.sum().item())
        if n_seg_codons == 0:
            continue
        seg_nt_muts = int(nt_diffs[seg_mask].sum().item())
        effective_mutations += seg_nt_muts / float(rate_adjustments[seg_idx])
        total_nt_length += 3 * n_seg_codons

    if total_nt_length == 0:
        return 0.0
    return effective_mutations / total_nt_length


def _precompute_warm_estimates(
    samples: List[PerSequenceBLInputs],
    strategy: Any,
    rate_adjustments: torch.Tensor,
    edge_features_list: Optional[List[dict]] = None,
) -> None:
    """Pre-compute warm-start estimates on each sample in-place.

    Priority:
    1. Strategy-based (if strategy is not None): computes tensor-based
       mutation frequency, then applies the pluggable strategy (~0.02ms/seq).
    2. Selection-adjusted closed-form: ~0.15ms/seq (vectorized, uses
       pre-computed log_selection_factors to correct the neutral-rate bias).

    Args:
        samples: List of PerSequenceBLInputs to update in-place.
        strategy: BranchLengthInitStrategy, or None for selection-adjusted fallback.
        rate_adjustments: Per-segment rate multipliers [num_segments].
        edge_features_list: Per-sequence edge feature dicts for the strategy.
            Required when strategy is not None and strategy.required_features
            is non-empty.
    """
    if strategy is not None:
        for i, sample in enumerate(samples):
            mut_freq = _compute_tensor_mut_freq(sample, rate_adjustments)
            edge_features = edge_features_list[i] if edge_features_list else {}
            sample.warm_estimate = max(strategy(mut_freq, edge_features), 1e-6)
    else:
        for sample in samples:
            if sample.log_selection_factors is None:
                raise RuntimeError(
                    "log_selection_factors must be set before warm estimation. "
                    "Call optimize_branch_lengths_for_dataset to compute them."
                )
            sample.warm_estimate = selection_adjusted_bl_estimate(
                sample.codon_parents_idxs,
                sample.codon_children_idxs,
                sample.nt_rates,
                sample.mask,
                sample.log_selection_factors,
            )


def optimize_branch_lengths_for_dataset(
    model: CoreModel,
    bl_source: Any,
    head_type: HeadType,
    loss: Optional[nn.Module] = None,
    convergence_tolerance: float = 1e-4,
    parallel: bool = True,
    num_workers: Optional[int] = None,
    max_chunk_size: int = MAX_OPTIMIZATION_CHUNK_SIZE,
    use_autocast: bool = False,
) -> torch.Tensor:
    """Optimize branch lengths using DASM2's transformer-based selection model.

    Processes data in chunks to bound memory usage: the main process runs GPU
    model forward on batches and accumulates samples into chunks, then each
    chunk is dispatched to a worker pool for scipy optimization. At most
    ``num_workers + 1`` chunks are alive at any time.

    For Parquet-backed datasets, ``nt_rates``/``nt_csps`` are streamed from
    disk — chunks are built incrementally as batches arrive. For in-memory
    datasets, selection factors are computed in one pass, then samples are
    yielded in chunks.

    Uses ``spawn`` context for the worker pool: the parent has CUDA
    initialized from model.forward(), and fork would copy the CUDA context
    into every worker, causing severe contention even for CPU-only scipy
    work. Workers only touch CPU tensors and don't use CUDA.

    Args:
        model: CoreModel instance for computing selection factors.
        bl_source: BL-capable object providing branch length optimization data.
            Can be ``BranchLengthDatasetBundle`` or ``ESM2DatasetWrapper``.
        head_type: Which DASM-loss head produces the L×20 log selection
            factors used by BL optimization. Must be in
            ``DASM_LOSS_HEAD_TYPES``.
        loss: The task loss owning the endpoint kernel, when the caller has
            it. Mirrors ``optimize_branch_lengths_for_dataset_qdasm``'s
            parameter of the same name and exists for the same reason:
            ``lookup_task_loss_for_head`` cannot disambiguate a head
            carrying several losses. Left None, the normalization is
            resolved off ``model.task_losses``, falling back to
            ``"parent_absorbing"`` when the model carries none.
        convergence_tolerance: Scipy optimization tolerance.
        parallel: Enable parallel processing.
        num_workers: Number of workers (None for auto-calculation).
        max_chunk_size: Upper bound on sequences per chunk (default
            ``MAX_OPTIMIZATION_CHUNK_SIZE``); bounds the host-RAM working set.
        use_autocast: Run the selection-factor forward under BF16 autocast
            (issue #1293). Callers pass the checkpoint's
            ``use_mixed_precision``, so training-time and test-time branch
            lengths share numerics.

    Returns:
        Tensor of optimized branch lengths [num_sequences].
    """
    if head_type not in DASM_LOSS_HEAD_TYPES:
        raise ValueError(
            f"optimize_branch_lengths_for_dataset: head_type must be one of "
            f"{[t.value for t in DASM_LOSS_HEAD_TYPES]}, got "
            f"{head_type.value}."
        )

    # Which endpoint kernel the scipy objective evaluates. The DASM BL path
    # does not go through ``loss.log_codon_probs`` — it calls
    # ``compute_dasm_log_pcp_fast`` directly — so without this a
    # conditionally-trained checkpoint would get natively-fitted branch
    # lengths (issue #1238 §4.5).
    normalization = resolve_dasm_normalization(model, head_type, loss)

    if num_workers is None:
        num_workers = _calculate_optimal_worker_count()

    model.eval()

    # Extract segment info for warm estimation (strategy + rate adjustments).
    segment_handler = bl_source.segment_handler  # type: ignore[attr-defined]
    strategy = segment_handler.bl_init_strategy
    rate_adjustments = segment_handler.get_rate_adjustments()

    # Build per-sequence edge feature dicts for strategy-based warm estimation.
    edge_features_list: Optional[List[dict]] = None
    if strategy is not None:
        raw_features = bl_source.edge_features  # type: ignore[attr-defined]
        n = len(bl_source.dataset)  # type: ignore[attr-defined]
        edge_features_list = [
            {name: values[i] for name, values in raw_features.items()} for i in range(n)
        ]

    # Size chunks so there are at least num_workers chunks (good parallelism)
    # but no more than max_chunk_size (bounded memory).
    dataset_size = len(bl_source.dataset)  # type: ignore[attr-defined]
    effective_workers = max(num_workers, 1)
    chunk_size = min(max_chunk_size, max(1, ceil(dataset_size / effective_workers)))

    # Build a chunk generator for the appropriate data backend. The parallel
    # path reads rows in DataLoader workers (issue #1276), forked here by
    # iter(), before the spawn pool starts its handler threads.
    parallel_path = parallel and num_workers > 1
    bl_dataloader = bl_source.create_bl_optimization_dataloader(  # type: ignore[attr-defined]
        batch_size=DEFAULT_SELECTION_FACTOR_BATCH_SIZE,
        num_workers=bl_loader_worker_count(num_workers) if parallel_path else 0,
    )
    bl_batches = iter(bl_dataloader) if bl_dataloader is not None else None

    # Codon preference log-vector from the bundle (issue #747).
    _codon_pref_log = getattr(bl_source, "codon_pref_log", None)
    # The tandem term rides on the neutral model the loss's transform reads.
    _tandem = getattr(bl_source.neutral_model, "tandem", None)  # type: ignore[attr-defined]

    if bl_batches is not None:
        chunks: Iterator[BranchLengthOptimizationInputs] = (
            _generate_optimization_chunks_streaming(
                model,
                bl_batches,
                bl_source.multihit_model,  # type: ignore[attr-defined]
                convergence_tolerance,
                strategy,
                rate_adjustments,
                chunk_size,
                head_type=head_type,
                edge_features_list=edge_features_list,
                codon_pref_log=_codon_pref_log,
                use_autocast=use_autocast,
                normalization=normalization,
                tandem=_tandem,
            )
        )
    else:
        chunks = _generate_optimization_chunks_in_memory(
            model,
            bl_source,
            convergence_tolerance,
            strategy,
            rate_adjustments,
            chunk_size,
            head_type=head_type,
            edge_features_list=edge_features_list,
            codon_pref_log=_codon_pref_log,
            use_autocast=use_autocast,
            normalization=normalization,
            tandem=_tandem,
        )

    # Serial path: process chunks sequentially.
    if not parallel_path:
        results = [
            _optimize_branch_lengths_chunk(chunk)
            for chunk in tqdm(
                chunks,
                total=ceil(dataset_size / chunk_size),
                desc="Optimizing branch lengths (serial)",
                delay=2.0,
            )
        ]
        return torch.cat(results) if results else torch.tensor([])

    # Parallel path: dispatch chunks to spawn workers as the generator yields
    # them, overlapping GPU forward (chunk generation) with CPU scipy work.
    #
    # PerSequenceBLInputs.__getstate__/__setstate__ convert tensors to numpy
    # arrays for pickling, bypassing PyTorch's tensor sharing strategies
    # entirely. This avoids both:
    # - "file_descriptor" strategy: FD exhaustion (EMFILE) at 50K+ samples
    # - "file_system" strategy: mmap ENOMEM under system memory pressure
    all_results = chunked_async_spawn_map(
        _optimize_branch_lengths_chunk_numpy,
        chunks,
        num_workers=num_workers,
        total_items=dataset_size,
        desc=f"Optimizing branch lengths ({num_workers} workers)",
        initializer=pin_single_thread,
    )
    return (
        torch.from_numpy(np.concatenate(all_results))
        if all_results
        else torch.tensor([])
    )


# ---------------------------------------------------------------------------
# GPU-batched BL optimization for QDASM / Hybrid heads
# ---------------------------------------------------------------------------

# Default batch size for GPU-batched BL optimization. Sized to match the
# training batch (typically 128) — the GS inner loop calls
# compute_qdasm_log_codon_probs with the same B×L×61×61 working set the
# training forward pass already validated fits in GPU memory alongside the
# 1.34 GB Thrifty 13-mer table.
_QDASM_BL_CHUNK_SIZE = 128

# Golden-section search iteration count. 25 iterations on a search interval of
# log-width ~6 (e^6 ratio) gives ~5e-6 relative precision in log-space, i.e.
# ~5e-8 absolute precision on a typical BL of ~0.01 — well below any
# downstream sensitivity. 50 iterations would only buy 6 more decimal places
# at 2x the wall-clock cost (the matrix_exp inside each iteration dominates).
_QDASM_BL_GS_ITERATIONS = 25

# log(BL) bounds:
#   log_bl_lower = -14.0  →  bl ≈ 8e-7
#   log_bl_upper =   0.405 →  bl ≈ 1.5
_QDASM_BL_LOG_LOWER = -14.0
_QDASM_BL_LOG_UPPER = 0.405  # log(1.5)


def _qdasm_bl_memory_record(
    chunk_index: int,
    chunk_start: int,
    n_items: int,
    L_aa: int,
    device: torch.device,
) -> Dict[str, float]:
    """Sample and reset the BL-optimization chunk-loop memory trace.

    Not a pure read: this resets the per-chunk peak trackers it reads
    (``tracemalloc.reset_peak`` and, on CUDA,
    ``torch.cuda.reset_peak_memory_stats``) so the next row measures the next
    chunk rather than the run so far. Call it exactly once per chunk. Only the
    caller that opted into tracing starts ``tracemalloc``, so on the default
    path the reset is a no-op on an unstarted tracer.

    Four measurements, because they fail differently and the point of the
    trace is to tell those failures apart (issue #988):

    - ``traced_peak_mb`` (``tracemalloc``) is this chunk's *Python-object*
      working set, reset per chunk. Note the scope: ``tracemalloc`` only sees
      allocations made through Python's allocator, and **torch tensor storage
      is not one of them** — 128 MB of ``torch.zeros`` moves it by 0.0 MB. So
      this field catches retained dicts/lists, not retained tensors.
    - ``rss_mb`` (``/proc/self/statm``) is what the OS actually holds, and is
      therefore the load-bearing number here: it sees both tensor storage and
      glibc holding freed pages in per-thread arenas rather than returning
      them (the #695 fragmentation mode).
    - ``tree_rss_mb`` adds the forked fetch workers, which ``rss_mb`` cannot
      see: ``/proc/self/statm`` (like ``getrusage(RUSAGE_SELF)``) excludes
      children, and the streaming path runs its fetch in up to
      ``MAX_FETCH_WORKERS`` of them. Compare host-footprint claims against
      this one, not ``rss_mb``. It counts copy-on-write pages once per
      process, so for forked workers it is an upper bound.
    - ``device_peak_mb`` separates a GPU working-set problem from a host one.

    Flat ``traced_peak_mb`` alongside rising ``rss_mb`` is either a retained
    tensor or arena fragmentation; the two are told apart by whether the
    growth survives a ``malloc_trim``. Both rising means Python-level state is
    accumulating. Measured on the tang checkpoint at 32,768 PCPs (issue #988),
    none of these rise: RSS moved 0.19 MB/chunk over 256 chunks and was flat
    after the first ~30, which is why this loop has no periodic trim.

    Args:
        chunk_index: 0-based position of this chunk in the sweep.
        chunk_start: Index of this chunk's first sequence in the dataset.
        n_items: Sequences in this chunk.
        L_aa: Padded AA length of this chunk's batch, which drives the
            per-chunk working set and varies chunk to chunk.
        device: Device the search ran on, for the CUDA peak readback.

    Returns:
        Dict of scalar measurements. ``tracemalloc`` fields read 0.0 when the
        caller has not started tracing; the RSS fields are absent off Linux and
        the CUDA field is absent off CUDA.
    """
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    record = {
        "chunk_index": float(chunk_index),
        "chunk_start": float(chunk_start),
        "n_items": float(n_items),
        "L_aa": float(L_aa),
        "traced_current_mb": traced_current / (1024**2),
        "traced_peak_mb": traced_peak / (1024**2),
    }
    rss_mb = get_current_rss_mb()
    if rss_mb is not None:
        record["rss_mb"] = rss_mb
    tree_rss_mb = get_process_tree_rss_mb()
    if tree_rss_mb is not None:
        record["tree_rss_mb"] = tree_rss_mb
    if device.type == "cuda":
        record["device_peak_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
        torch.cuda.reset_peak_memory_stats(device)
    return record


class _QFamilyLossAttrs(Protocol):
    """Attributes ``optimize_branch_lengths_for_dataset_qdasm`` reads off the
    matching QDASM/Hybrid loss. Cast target for narrowing past
    ``nn.Module.__getattr__``'s ``Tensor | Module | Any`` return type.
    """

    q_variant: Any
    kmer_table: Optional[torch.Tensor]
    single_hit_scale: float
    double_hit_scale: float
    triple_hit_scale: float
    use_cubic: bool


class _QDASMLossAttrs(_QFamilyLossAttrs, Protocol):
    """Additional attributes read only from a QDASM loss."""

    neutral_codon_q: Optional[torch.Tensor]
    adaptive: bool
    tq_threshold: float
    long_path: Literal["matrix_exp", "ss_quintic"]
    adaptive_mode: Literal["per_chunk", "per_sequence"]
    matexp_method: Literal["adaptive", "uniformization"]
    n_terms: int
    normalization: Literal["parent_absorbing", "proportional"]


@dataclass
class _QdasmBLChunkCtx:
    """Per-chunk context for _neg_log_p_qdasm_batch.

    Bundles the tensors and head-config scalars that the GS evaluator needs
    so the closure-free function takes one dataclass instead of ~15 args.
    """

    is_hybrid: bool
    codon_parents_idxs: torch.Tensor
    codon_rates: torch.Tensor
    codon_csps_4d: torch.Tensor
    log_sf: torch.Tensor
    use_cubic: bool
    single_hit_scale: float
    double_hit_scale: float
    triple_hit_scale: float
    adaptive: bool
    tq_threshold: float
    adaptive_mode: Literal["per_chunk", "per_sequence"]
    long_path: Literal["matrix_exp", "ss_quintic"]
    matexp_method: Literal["adaptive", "uniformization"]
    n_terms: int
    normalization: Literal["parent_absorbing", "proportional"]
    q_variant: Any
    codon_parent_nt_window: Optional[torch.Tensor]
    codon_parent_boundary_mask: Optional[torch.Tensor]
    codon_boundary_features: Optional[torch.Tensor]
    codon_rate_adjustments: Optional[torch.Tensor]
    kmer_table: Optional[torch.Tensor]
    neutral_codon_q: Optional[torch.Tensor]
    n_idx: torch.Tensor
    l_idx: torch.Tensor
    clamped_child: torch.Tensor
    mask_combined: torch.Tensor
    # Branch-length-independent Q, built once per chunk and reused across all
    # golden-section iterations. Selection-weighted for QDASM (build_qdasm_q_all)
    # and neutral for Hybrid (build_hybrid_neutral_q_all) — which is why the two
    # entry points take it under different names.
    precomputed_q: torch.Tensor


def _neg_log_p_qdasm_batch(
    log_bl: torch.Tensor, *, ctx: "_QdasmBLChunkCtx"
) -> torch.Tensor:
    """Negative log P(child | parent, t) summed per sequence for a chunk.

    Module-level helper that drives the QDASM/Hybrid GS branch-length search.
    """
    bl = torch.exp(log_bl)
    # Note: ``ctx.precomputed_q`` short-circuits the Q-build inside both entry
    # points, so the Q-construction kwargs below (rates, CSPs, selection,
    # hit-scales, q_variant, and the global-kmer tensors) are inert on
    # every call from this loop. They stay in the call for the non-hoisted
    # contract and so the two branches read the same as their direct callers;
    # only the branch length and the child-codon gather actually vary here.
    if ctx.is_hybrid:
        log_probs_64 = compute_hybrid_log_codon_probs(
            ctx.codon_parents_idxs,
            ctx.codon_rates,
            ctx.codon_csps_4d,
            ctx.log_sf,
            bl,
            use_cubic=ctx.use_cubic,
            single_hit_scale=ctx.single_hit_scale,
            double_hit_scale=ctx.double_hit_scale,
            triple_hit_scale=ctx.triple_hit_scale,
            q_variant=ctx.q_variant,
            codon_parent_nt_window=ctx.codon_parent_nt_window,
            codon_parent_boundary_mask=ctx.codon_parent_boundary_mask,
            codon_boundary_features=ctx.codon_boundary_features,
            codon_rate_adjustments=ctx.codon_rate_adjustments,
            kmer_table=ctx.kmer_table,
            precomputed_neutral_q=ctx.precomputed_q,
        )
    else:
        log_probs_64 = compute_qdasm_log_codon_probs(
            ctx.codon_parents_idxs,
            ctx.codon_rates,
            ctx.codon_csps_4d,
            ctx.log_sf,
            bl,
            use_cubic=ctx.use_cubic,
            single_hit_scale=ctx.single_hit_scale,
            double_hit_scale=ctx.double_hit_scale,
            triple_hit_scale=ctx.triple_hit_scale,
            adaptive=ctx.adaptive,
            tq_threshold=ctx.tq_threshold,
            adaptive_mode=ctx.adaptive_mode,
            long_path=ctx.long_path,
            matexp_method=ctx.matexp_method,
            n_terms=ctx.n_terms,
            normalization=ctx.normalization,
            q_variant=ctx.q_variant,
            codon_parent_nt_window=ctx.codon_parent_nt_window,
            codon_parent_boundary_mask=ctx.codon_parent_boundary_mask,
            codon_boundary_features=ctx.codon_boundary_features,
            codon_rate_adjustments=ctx.codon_rate_adjustments,
            kmer_table=ctx.kmer_table,
            neutral_codon_q=ctx.neutral_codon_q,
            precomputed_q=ctx.precomputed_q,
        )
    child_log_probs = log_probs_64[ctx.n_idx, ctx.l_idx, ctx.clamped_child]
    # Hybrid sets every codon at separator, ambiguous and padding parent sites
    # to -inf, and -inf * 0 is NaN. Multiplying by the zero mask would make
    # every ``f1 < f2`` comparison False and pin the BL at the upper bound.
    neg_log_p: torch.Tensor = -child_log_probs.masked_fill(~ctx.mask_combined, 0.0).sum(
        dim=-1
    )
    nonfinite = ~torch.isfinite(neg_log_p)
    if nonfinite.any():
        raise RuntimeError(
            "Non-finite negative log PCP probability in QDASM/Hybrid BL objective: "
            f"chunk_rows={nonfinite.nonzero().flatten().tolist()}, "
            f"neg_log_p={neg_log_p[nonfinite].tolist()}, "
            f"bl={bl[nonfinite].tolist()}"
        )
    return neg_log_p


def optimize_branch_lengths_for_dataset_qdasm(
    model: CoreModel,
    bl_source: Any,
    head_type: HeadType,
    loss: Optional[nn.Module] = None,
    chunk_size: int = _QDASM_BL_CHUNK_SIZE,
    num_iterations: int = _QDASM_BL_GS_ITERATIONS,
    log_bl_lower: float = _QDASM_BL_LOG_LOWER,
    log_bl_upper: float = _QDASM_BL_LOG_UPPER,
    memory_trace: Optional[List[Dict[str, float]]] = None,
) -> torch.Tensor:
    """GPU-batched golden-section search for QDASM/Hybrid branch lengths.

    Mirrors ``optimize_branch_lengths_for_dataset`` (the DASM scipy/spawn-worker
    path) for QDASM and Hybrid evolutionary heads. The QDASM family evaluates
    likelihood through ``P = exp(tQ)`` with the head's own ``q_variant``,
    multi-hit scales, and (if ``q_variant=global_13mer``) the 1.34 GB
    Thrifty 13-mer lookup table — none of which can ride through pickle to a
    spawn worker. Everything therefore stays on the main GPU process.

    For each chunk of ``chunk_size`` sequences:
        1. Forward through the model on GPU → log selection factors.
        2. Run a vectorized golden-section search on ``log(branch_length)``.
           Each GS evaluation calls
           ``compute_qdasm_log_codon_probs`` (or
           ``compute_hybrid_log_codon_probs``) which builds Q from the head's
           configured variant and runs the batched ``matrix_exp``.
        3. Sum log P(child | parent, t) over masked positions per sequence;
           pick the BL that maximizes that sum.

    Args:
        model: ``CoreModel`` with the active QDASM/Hybrid head registered.
        bl_source: BL-capable bundle (``BranchLengthDatasetBundle`` or
            ``ESM2DatasetWrapper``).
        head_type: Which evol head to read outputs from. Must be QDASM or
            Hybrid; DASM uses the existing path.
        chunk_size: Sequences per GS chunk.
        num_iterations: Golden-section iterations per chunk.
        log_bl_lower / log_bl_upper: Hard bounds on the search interval. The
            interval is widened around each sequence's warm start (current
            ``bl_source.branch_lengths[i]``).
        memory_trace: Optional list to append one
            :func:`_qdasm_bl_memory_record` per chunk to. Off by default — the
            measurement itself is cheap, but it is a diagnostic, not something
            every training cycle needs. Start ``tracemalloc`` before the call
            for the traced fields to be populated.

    Returns:
        Optimized branch lengths as a CPU float tensor of shape ``[N]``.
    """
    if head_type not in _QDASM_FAMILY_HEADS:
        raise ValueError(
            f"optimize_branch_lengths_for_dataset_qdasm: head_type must be "
            f"one of {[t.value for t in _QDASM_FAMILY_HEADS]}, got {head_type.value}"
        )

    # Pull Q-matrix construction params off the matching ``Loss`` object
    # (post-#621 Task #4 the head no longer carries them; the trainer
    # registers each task's loss on the unwrapped model so we can find it
    # via ``model.task_losses`` if not explicitly passed).
    if loss is None:
        loss = lookup_task_loss_for_head(model, head_type)

    # ``nn.Module.__getattr__`` types every attribute access as
    # ``Tensor | Module | Any``; the cast narrows to the typed fields the
    # QDASM/Hybrid loss classes declare so the float()/bool() conversions
    # and ``adaptive_mode`` Literal assignment below type-check.
    loss_attrs = cast(_QFamilyLossAttrs, loss)
    q_variant = loss_attrs.q_variant
    kmer_table = loss_attrs.kmer_table
    single_hit_scale = float(loss_attrs.single_hit_scale)
    double_hit_scale = float(loss_attrs.double_hit_scale)
    triple_hit_scale = float(loss_attrs.triple_hit_scale)
    use_cubic = bool(loss_attrs.use_cubic)

    # adaptive/tq_threshold/adaptive_mode are QDASM-only — Hybrid runs
    # matrix_exp on a unit-selection neutral Q with no gradient flow, so the
    # cubic-vs-exact-with-grad distinction collapses to plain ``use_cubic``.
    # Hoisted to top scope (with ignored sentinels for the Hybrid branch) so
    # the closure below has unconditional bindings — the closure only reads
    # them on the QDASM branch.
    adaptive_mode: Literal["per_chunk", "per_sequence"]
    long_path: Literal["matrix_exp", "ss_quintic"]
    matexp_method: Literal["adaptive", "uniformization"]
    normalization: Literal["parent_absorbing", "proportional"]
    if head_type is HeadType.HYBRID_EVOL_HEAD:
        is_hybrid = True
        adaptive = False
        tq_threshold = 0.0
        adaptive_mode = "per_sequence"
        long_path = "ss_quintic"
        matexp_method = "adaptive"
        n_terms = 0
        normalization = "parent_absorbing"
        # The context-free SHM-1 generator is QDASM-only. The global-kmer
        # lookup itself is shared with Hybrid through _QFamilyLossAttrs.
        neutral_codon_q = None
    elif head_type is HeadType.QDASM_EVOL_HEAD:
        qdasm_loss_attrs = cast(_QDASMLossAttrs, loss)
        is_hybrid = False
        adaptive = bool(qdasm_loss_attrs.adaptive)
        tq_threshold = float(qdasm_loss_attrs.tq_threshold)
        adaptive_mode = qdasm_loss_attrs.adaptive_mode
        long_path = qdasm_loss_attrs.long_path
        matexp_method = qdasm_loss_attrs.matexp_method
        n_terms = int(qdasm_loss_attrs.n_terms)
        normalization = qdasm_loss_attrs.normalization
        # SHM-1 codon generator for q_variant=shm1_codon (None otherwise). The BL
        # golden-section search rebuilds P=exp(tQ) and needs this the same way the
        # training forward does — omitting it raised at the post-cycle BL re-opt.
        neutral_codon_q = qdasm_loss_attrs.neutral_codon_q
    else:
        # Already validated above against _QDASM_FAMILY_HEADS, but make the
        # exhaustive dispatch explicit so adding a third Q-family head doesn't
        # silently route through the QDASM path.
        raise AssertionError(
            f"Unhandled Q-family head_type: {head_type!r}; "
            f"_QDASM_FAMILY_HEADS gate above should have rejected it"
        )

    model.eval()
    device = model.device

    dataset = bl_source.dataset
    collate_fn = bl_source.collator

    starting_bls = bl_source.branch_lengths.detach().clone().cpu().float()  # type: ignore[attr-defined]
    n_total = len(dataset)
    optimized = torch.zeros(n_total, dtype=torch.float32)

    # Golden ratio constants
    invphi = (math.sqrt(5) - 1) / 2  # ~0.618
    invphi2 = (3 - math.sqrt(5)) / 2  # ~0.382

    # A sequential ``DataLoader`` at ``batch_size=chunk_size`` yields exactly
    # the chunks the hand-rolled ``[dataset[i] for i in range(...)]`` loop did,
    # in the same order — but it can hand the fetch to worker processes. That
    # matters only for streaming datasets, whose ``__getitem__`` re-runs the
    # neutral model per sample (~10.6 ms); ``fetch_worker_count`` returns 0 for
    # precomputed, which reproduces the old single-process behavior exactly.
    # Chunk contents are unchanged either way, which is what keeps this loop's
    # output invariant to ``chunk_size`` (issue #988).
    loader = DataLoader(
        dataset,
        batch_size=chunk_size,
        shuffle=False,
        collate_fn=collate_fn,
        # Called directly rather than read off ``bl_source``: only
        # ``BranchLengthDatasetBundle`` carries a ``worker_init_fn``, and the
        # other two source types that reach here — ``ESM2DatasetWrapper`` and
        # the per-rank ``_RankSlicedBLSource`` — do not, so a ``getattr``
        # lookup would silently drop the thread-count guard on the sharded
        # training path. Every bundle in the codebase sets this to
        # ``make_worker_init_fn()`` anyway.
        worker_init_fn=make_worker_init_fn(),
        **fetch_dataloader_kwargs(dataset),
    )
    progress = tqdm(
        loader,
        desc=f"Optimizing branch lengths (GPU-batched, {head_type.value})",
        delay=2.0,
    )
    chunk_start = 0
    for batch in progress:
        batch = move_batch_to_device(batch, device)

        (
            codon_parent_nt_window,
            codon_parent_boundary_mask,
            codon_boundary_features,
            codon_rate_adjustments,
        ) = q_family_global_kmer_inputs(
            batch, q_variant, "optimize_branch_lengths_for_dataset_qdasm()"
        )

        with torch.no_grad():
            outputs = model.forward(batch, head_names=[head_type])
            log_sf = outputs[head_type]  # [N, L_aa, 20]

            codon_parents_idxs = batch["codon_parents_idxs"]
            codon_children_idxs = batch["codon_children_idxs"]
            codon_rates = batch["codon_rates"]
            codon_csps = batch["codon_csps"]
            loss_mask = batch["loss_mask"]

            N, L_aa = codon_parents_idxs.shape
            # Read the span off the batch rather than assuming ``chunk_size``:
            # the DataLoader's last batch is short, and deriving it here keeps
            # the two writes below correct without a separate tail case.
            chunk_end = chunk_start + N

            codon_csps_4d = csps_to_4d(codon_csps)

            # Per-sequence GS bounds, anchored on each sequence's warm start.
            chunk_warm = starting_bls[chunk_start:chunk_end].to(device).clamp(min=1e-7)
            log_start = torch.log(chunk_warm)
            a = torch.minimum(
                torch.full_like(log_start, log_bl_lower),
                log_start - 3.0,
            )
            b = torch.maximum(
                torch.full_like(log_start, log_bl_upper),
                log_start + 2.0,
            )

            # Constants for the per-iteration child-codon gather.
            valid_child = codon_children_idxs < 64
            clamped_child = torch.clamp(codon_children_idxs, max=63).long()
            mask_combined = loss_mask & valid_child
            n_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, L_aa)
            l_idx = torch.arange(L_aa, device=device).unsqueeze(0).expand(N, L_aa)

            # Build the branch-length-independent Q once for this chunk and
            # reuse it across every golden-section iteration (Q depends only on
            # rates/CSPs/selection/hit-scales, not on the trial branch length).
            # This hoists the Q-build — including the 1.34 GB 13-mer gather —
            # out of the ~27-evaluation GS loop.
            l_q = shared_codon_length(L_aa, codon_rates)
            if is_hybrid:
                precomputed_q = build_hybrid_neutral_q_all(
                    codon_rates,
                    codon_csps_4d,
                    l_q,
                    single_hit_scale=single_hit_scale,
                    double_hit_scale=double_hit_scale,
                    triple_hit_scale=triple_hit_scale,
                    q_variant=q_variant,
                    codon_parent_nt_window=codon_parent_nt_window,
                    codon_parent_boundary_mask=codon_parent_boundary_mask,
                    codon_boundary_features=codon_boundary_features,
                    codon_rate_adjustments=codon_rate_adjustments,
                    kmer_table=kmer_table,
                )
            else:
                precomputed_q = build_qdasm_q_all(
                    codon_rates,
                    codon_csps_4d,
                    log_sf,
                    l_q,
                    single_hit_scale=single_hit_scale,
                    double_hit_scale=double_hit_scale,
                    triple_hit_scale=triple_hit_scale,
                    q_variant=q_variant,
                    codon_parent_nt_window=codon_parent_nt_window,
                    codon_parent_boundary_mask=codon_parent_boundary_mask,
                    codon_boundary_features=codon_boundary_features,
                    codon_rate_adjustments=codon_rate_adjustments,
                    kmer_table=kmer_table,
                    neutral_codon_q=neutral_codon_q,
                )

            ctx = _QdasmBLChunkCtx(
                is_hybrid=is_hybrid,
                codon_parents_idxs=codon_parents_idxs,
                codon_rates=codon_rates,
                codon_csps_4d=codon_csps_4d,
                log_sf=log_sf,
                use_cubic=use_cubic,
                single_hit_scale=single_hit_scale,
                double_hit_scale=double_hit_scale,
                triple_hit_scale=triple_hit_scale,
                adaptive=adaptive,
                tq_threshold=tq_threshold,
                adaptive_mode=adaptive_mode,
                long_path=long_path,
                matexp_method=matexp_method,
                n_terms=n_terms,
                normalization=normalization,
                q_variant=q_variant,
                codon_parent_nt_window=codon_parent_nt_window,
                codon_parent_boundary_mask=codon_parent_boundary_mask,
                codon_boundary_features=codon_boundary_features,
                codon_rate_adjustments=codon_rate_adjustments,
                kmer_table=kmer_table,
                neutral_codon_q=neutral_codon_q,
                n_idx=n_idx,
                l_idx=l_idx,
                clamped_child=clamped_child,
                mask_combined=mask_combined,
                precomputed_q=precomputed_q,
            )

            def _neg_log_p(log_bl: torch.Tensor) -> torch.Tensor:
                return _neg_log_p_qdasm_batch(log_bl, ctx=ctx)

            # Standard golden-section search.
            x1 = a + invphi2 * (b - a)
            x2 = a + invphi * (b - a)
            f1 = _neg_log_p(x1)
            f2 = _neg_log_p(x2)
            for _ in range(num_iterations):
                left_wins = f1 < f2

                old_x1, old_f1 = x1.clone(), f1.clone()
                old_x2, old_f2 = x2.clone(), f2.clone()

                a = torch.where(left_wins, a, old_x1)
                b = torch.where(left_wins, old_x2, b)

                new_x = torch.where(
                    left_wins,
                    a + invphi2 * (b - a),
                    a + invphi * (b - a),
                )
                new_f = _neg_log_p(new_x)

                # left_wins: x2 ← old_x1 (shifted right), x1 ← new_x (new lower point).
                # ~left_wins: x1 ← old_x2 (shifted left),  x2 ← new_x.
                x1 = torch.where(left_wins, new_x, old_x2)
                f1 = torch.where(left_wins, new_f, old_f2)
                x2 = torch.where(~left_wins, new_x, old_x1)
                f2 = torch.where(~left_wins, new_f, old_f1)

            best_log_bl = (a + b) / 2
            optimized[chunk_start:chunk_end] = torch.exp(best_log_bl).cpu()

        if memory_trace is not None:
            memory_trace.append(
                _qdasm_bl_memory_record(
                    chunk_index=len(memory_trace),
                    chunk_start=chunk_start,
                    n_items=chunk_end - chunk_start,
                    L_aa=L_aa,
                    device=device,
                )
            )

        chunk_start = chunk_end

    return optimized


def needs_qdasm_branch_length_path(model: CoreModel, head_type: HeadType) -> bool:
    """Whether ``head_type`` must use the QDASM golden-section BL optimizer.

    Keyed on the *loss* rather than the head (issue #1092): the choice
    between ``P = exp(tQ)`` and DASM's Poisson-product likelihood is a
    property of the objective, and ``QDASM_FAMILY_LOSS_TYPES``
    (``dasm2/losses/types.py``) documents exactly this decision. Falls back
    to the head-type family when no loss is registered for the head, which
    is what the pre-#1092 dispatch keyed on and keeps loss-less models
    (some tests, some ad-hoc scripts) routing as before.

    Exposed publicly because ``get_or_compute_branch_lengths`` needs the
    same answer for a second reason — QDASM branch lengths are deliberately
    not cached.

    Only :class:`MissingTaskLossError` triggers the head-type fallback.
    :class:`AmbiguousTaskLossError` propagates: falling back for a head with
    two registered losses would pick an optimizer from a head-type set while
    the checkpoint's actual objective is unknown — the silent-wrong-kernel
    failure mode #1092 exists to remove.
    """
    try:
        loss = lookup_task_loss_for_head(model, head_type)
    except MissingTaskLossError:
        return head_type in _QDASM_FAMILY_HEADS
    return is_qdasm_family_loss(loss)


def optimize_branch_lengths_dispatch(
    model: CoreModel,
    bl_source: Any,
    *,
    head_type: HeadType,
    convergence_tolerance: float = 1e-4,
    parallel: bool = True,
    num_workers: Optional[int] = None,
    max_chunk_size: int = MAX_OPTIMIZATION_CHUNK_SIZE,
    use_autocast: bool = False,
) -> torch.Tensor:
    """Optimize branch lengths with the optimizer this checkpoint's loss needs.

    The single dispatch point between the two branch-length optimizers
    (issue #1092). Before this existed the same two-way branch was
    open-coded in ``BranchLengthOptimizer._sharded_optimize`` and
    ``get_or_compute_branch_lengths``, and a third caller
    (``dasm2.simulation.fit_branch_lengths``) called
    ``optimize_branch_lengths_for_dataset`` unconditionally — which would
    have run the DASM scipy optimizer on a QDASM checkpoint *silently*,
    since that function takes ``head_type`` only as a label for the forward
    pass.

    The DASM-only tuning arguments are ignored on the QDASM path:
    ``optimize_branch_lengths_for_dataset_qdasm`` always runs on the main
    GPU process (its Q-matrix state can't ride through pickle to a spawn
    worker) and searches a fixed golden-section interval.

    Args:
        model: ``CoreModel`` with the head and its task loss registered.
        bl_source: BL-capable object (``BranchLengthDatasetBundle``,
            ``ESM2DatasetWrapper``, or a rank slice of one).
        head_type: Which head produces the L×20 log selection factors.
        convergence_tolerance: Scipy tolerance (DASM path only).
        parallel: Enable the spawn worker pool (DASM path only).
        num_workers: Worker count, ``None`` to auto-size (DASM path only).
        max_chunk_size: Sequences per chunk (DASM path only).
        use_autocast: BF16 autocast on the selection-factor forward (DASM
            path only, issue #1293). The QDASM path's ``matrix_exp`` CTMC
            math stays fp32.

    Returns:
        Tensor of optimized branch lengths, shape ``[num_sequences]``.
    """
    if needs_qdasm_branch_length_path(model, head_type):
        return optimize_branch_lengths_for_dataset_qdasm(
            model,
            bl_source,
            head_type=head_type,
        )
    return optimize_branch_lengths_for_dataset(
        model,
        bl_source,
        head_type=head_type,
        convergence_tolerance=convergence_tolerance,
        parallel=parallel,
        num_workers=num_workers,
        max_chunk_size=max_chunk_size,
        use_autocast=use_autocast,
    )
