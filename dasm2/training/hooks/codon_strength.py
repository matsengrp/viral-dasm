"""Codon-optimality strength optimization hook (issue #747).

Fits the free strength scalars ``β_k`` (one per measured-optimality axis, e.g.
CSC and tAI) that scale the within-AA codon-preference references ``ξ̃ᵏ``:

    codon_pref_log = Σ_k β_k · ξ̃ᵏ            (a fixed 64-vector given β)

The strengths cannot be trained by ordinary backprop because the neutral codon
probabilities they modify are **cached dataset state** computed in the dataset
precompute and the branch-length hook, outside the training-step autograd graph
(``compute_log_adjusted_codon_probs`` consumes them as a fixed input). This is
the same situation as the per-PCP branch lengths (``BranchLengthOptimizer``) and
the DASMfit ``alpha`` (``DASMFitAlphaOptimizer``): a between-cycle hook re-fits
the quantity against the current model and triggers a cache recompute.

The ``β_k`` are point-identified by the within-AA synonymous signal
(Proposition 1(i)): the tables vary *within* each amino acid, and that within-AA
variation pins the strengths. The references ``ξ̃ᵏ`` are FIXED tensors carried
on the hook — never added to any optimizer. After fitting, the hook rewrites
``codon_pref_log`` into the cache via the SAME ``neutral_codon_probs`` primitive
both cache-writers call (dataset precompute and the BL hook), so the loss and BL
paths stay consistent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from dasm2.codon_table import NUM_SENSE_CODONS, SENSE_CODON_LIST_TENSOR
from dasm2.losses.capability import resolve_dasm_normalization
from dasm2.molevol.codon_preference import (
    CodonPreferenceTable,
    CodonPrefForm,
    apply_codon_pref_log,
    build_codon_pref_log,
    design_conditioning,
    lift_sense61_to_64,
)
from dasm2.molevol.mutsel import (
    DASMNormalization,
    compute_log_adjusted_codon_probs,
)
from dasm2.molevol.neutral_evolution import neutral_codon_probs_of_seq
from dasm2.training.hooks.base import TrainingHook
from dasm2.utils.batch_operations import move_batch_to_device

if TYPE_CHECKING:
    from dasm2.config import HookConfig
    from dasm2.heads.types import HeadType
    from dasm2.training.trainer import Trainer

logger = logging.getLogger(__name__)

DEFAULT_INFERENCE_BATCH_SIZE = 32
# Warn when materializing this many valid positions for the in-memory fit
# (the cached design tensors are [P, 64] + [P, 20] float32). The large-scale
# joint fit is issue #752; this hook proves the mechanism and serves the
# smoke/test path.
_LARGE_POSITION_WARN = 5_000_000


@dataclass(frozen=True)
class FitInputs:
    """Flattened, branch-length-fixed inputs for the strength fit.

    Every row is one valid in-frame position pooled across all sequences.

    Attributes:
        log_base: Log base (table-free) neutral codon probs [P, 64].
        log_selection_factors: Log AA selection factors [P, 20].
        parent_codon_idxs: Parent codon indices [P] (0-63).
        child_codon_idxs: Observed child codon indices [P] (0-63).
        weights: Optional per-position weights [P] (e.g. observation counts).
            ``None`` weights every position equally.
        pcp_idx: Optional per-position source-sample index [P] into the
            originating bundle's dataset order (0-based). Lets a caller join
            positions back to per-sample metadata (e.g. lineage/family, for a
            whole-lineage bootstrap) without threading extra columns through
            the fit itself. ``None`` if the caller didn't request it.
        normalization: DASM endpoint kernel every objective built on these
            inputs must evaluate (issue #1238). These fits maximize the DASM
            NLL directly — they have no loss object in scope — so the mode
            travels with the shared inputs exactly as
            ``log_selection_factors`` does. Getting it wrong means the
            fitted nuisance parameters are inconsistent with the likelihood
            being trained.
    """

    log_base: Tensor
    log_selection_factors: Tensor
    parent_codon_idxs: Tensor
    child_codon_idxs: Tensor
    weights: Optional[Tensor] = None
    pcp_idx: Optional[Tensor] = None
    normalization: DASMNormalization = "parent_absorbing"

    def __len__(self) -> int:
        return int(self.log_base.shape[0])


@dataclass(frozen=True)
class FitResult:
    """Result of a strength fit.

    ``betas``/``nll``/``n_positions`` are populated by both the fixed-basis
    strength fit (:func:`fit_strengths`) and the free-vector fit
    (:func:`fit_free_codon_pref`); ``betas`` is an empty ``[0]`` tensor for the
    free-vector fit (it has no basis coefficients). ``log_g``/``observed_mask``
    are populated only by the free-vector fit.
    """

    betas: Tensor  # [K]
    n_positions: int
    nll: float
    log_g: Optional[Tensor] = None  # [61], the free codon potential (Rung 1)
    observed_mask: Optional[Tensor] = None  # [61] bool, codons free vs. held at 0


def within_aa_64_design(tables: List[CodonPreferenceTable]) -> Tensor:
    """Stack the per-axis within-AA references lifted to 64-space: [K, 64]."""
    return torch.stack([lift_sense61_to_64(t.within_aa_log) for t in tables], dim=0)


def strength_nll(
    betas: Tensor,
    inputs: FitInputs,
    within64: Tensor,
    form: CodonPrefForm = "divide_z",
) -> Tensor:
    """In-frame DASM negative log-likelihood as a function of the strengths.

    Differentiable in ``betas``. Mirrors the production pipeline: form the
    codon preference ``Σ_k β_k·ξ̃ᵏ``, apply it to the base neutral codon
    probs, combine with selection via ``compute_log_adjusted_codon_probs``,
    and sum ``-log P(child)`` over the pooled valid positions.

    Args:
        betas: Strength vector [K].
        inputs: Pooled valid-position inputs.
        within64: Per-axis within-AA references lifted to 64-space [K, 64].
        form: ``"divide_z"`` (default; the merged #749/#767 softmax-renorm
            form) or ``"ratio"`` (issue #930 Rung 1; the parent-codon-gauge
            ratio ``g_{c'}/g_{x̄_j}``). See :func:`apply_codon_pref_log`.

    Returns:
        Scalar NLL tensor (to minimize).
    """
    codon_pref_log = betas @ within64  # [64], differentiable in betas
    log_neutral = apply_codon_pref_log(
        inputs.log_base,
        codon_pref_log,
        parent_codon_idxs=inputs.parent_codon_idxs,
        form=form,
    )  # [P, 64]
    log_adjusted = compute_log_adjusted_codon_probs(
        inputs.parent_codon_idxs,
        log_neutral,
        inputs.log_selection_factors,
        normalization=inputs.normalization,
    )  # [P, 64]
    child_log_probs = log_adjusted.gather(
        -1, inputs.child_codon_idxs.unsqueeze(-1)
    ).squeeze(-1)  # [P]
    if inputs.weights is not None:
        child_log_probs = child_log_probs * inputs.weights
    return -child_log_probs.sum()


def fit_strengths(
    inputs: FitInputs,
    within64: Tensor,
    init_betas: Tensor,
    max_iter: int = 50,
    form: CodonPrefForm = "divide_z",
) -> FitResult:
    """Fit the strength scalars by maximizing the pooled in-frame likelihood.

    Uses L-BFGS with a strong-Wolfe line search on the (low-dimensional,
    smooth) NLL. The tables ``ξ̃ᵏ`` (carried in ``within64``) are fixed; only
    ``betas`` is optimized.

    Args:
        inputs: Pooled valid-position inputs.
        within64: Per-axis within-AA references lifted to 64-space [K, 64].
        init_betas: Warm-start strengths [K].
        max_iter: Maximum L-BFGS iterations.
        form: ``"divide_z"`` (default, matches the merged #749/#767 training
            path) or ``"ratio"`` (issue #930 Rung 1 CSC/tAI refit).

    Returns:
        A :class:`FitResult` with the fitted strengths.
    """
    # Stay in float32: the downstream DASM primitives (CODON_AA_INDICATOR_MATRIX,
    # STOP_CODON_ZAPPER) are float32, and mixing dtypes errors. The strength fit
    # is well-conditioned in float32 (2 parameters, smooth objective).
    betas = init_betas.detach().clone().float().requires_grad_(True)
    within64 = within64.float()

    optimizer = torch.optim.LBFGS(
        [betas],
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=1e-12,
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        nll = strength_nll(betas, inputs, within64, form=form)
        nll.backward()
        return nll

    optimizer.step(closure)

    with torch.no_grad():
        final_nll = float(strength_nll(betas, inputs, within64, form=form))
    return FitResult(
        betas=betas.detach().float(),
        n_positions=len(inputs),
        nll=final_nll,
    )


def free_codon_child_log_probs(log_g: Tensor, inputs: FitInputs) -> Tensor:
    """Per-position ``log P(child)`` under the ratio-form potential ``log_g``.

    Differentiable in ``log_g``. Always the ratio form (issue #930 Rung 1):
    for a non-parent child ``c'`` at a site with parent codon ``x̄_j``, the
    per-position codon weight is ``p_{j,c'}·(g_{c'}/g_{x̄_j})``. This bypasses
    :func:`apply_codon_pref_log`'s softmax-renorm path entirely (a per-position
    parent subtraction would cancel under that renorm and silently collapse
    back to the ÷Z form) but reuses it in ``form="ratio"`` mode for the actual
    parent-gauge arithmetic, and reuses :func:`compute_log_adjusted_codon_probs`
    (the same target-only injection point ``strength_nll`` calls) for the
    f-lift + stop-zap + parent-absorb.

    Unweighted and unsummed (contrast :func:`free_codon_nll`): the single
    primitive both the L-BFGS objective and any per-position consumer (e.g.
    the held-out lineage bootstrap in :mod:`dasm2.analysis.codon_rung1`, which
    needs per-position values to bucket by family) should build on, so a
    caller that supplies ``inputs.weights`` gets the same weighting either way.

    Args:
        log_g: Free codon potential [61] (sense-codon space); entries for
            codons held out of the fit (unobserved as a child, or singletons)
            should be exactly 0 by construction of the caller.
        inputs: Pooled valid-position inputs.

    Returns:
        Per-position ``log P(child)``, shape [P].
    """
    pref64 = lift_sense61_to_64(log_g)  # [64], differentiable in log_g
    log_neutral = apply_codon_pref_log(
        inputs.log_base,
        pref64,
        parent_codon_idxs=inputs.parent_codon_idxs,
        form="ratio",
    )  # [P, 64], NOT renormalized (the kernel below does that, by whichever
    # normalization ``inputs.normalization`` names)
    log_adjusted = compute_log_adjusted_codon_probs(
        inputs.parent_codon_idxs,
        log_neutral,
        inputs.log_selection_factors,
        normalization=inputs.normalization,
    )  # [P, 64]
    return log_adjusted.gather(-1, inputs.child_codon_idxs.unsqueeze(-1)).squeeze(-1)


def free_codon_nll(log_g: Tensor, inputs: FitInputs) -> Tensor:
    """In-frame DASM NLL as a function of a free 61-vector codon potential.

    See :func:`free_codon_child_log_probs` for the per-position computation;
    this applies ``inputs.weights`` (if any), sums, and negates.

    Args:
        log_g: Free codon potential [61] (sense-codon space).
        inputs: Pooled valid-position inputs.

    Returns:
        Scalar NLL tensor (to minimize).
    """
    child_log_probs = free_codon_child_log_probs(log_g, inputs)
    if inputs.weights is not None:
        child_log_probs = child_log_probs * inputs.weights
    return -child_log_probs.sum()


@dataclass(frozen=True)
class CoverageReport:
    """Per-codon child-observation coverage for the free-vector fit.

    Attributes:
        child_counts: Observed-as-child count per sense codon [61].
        observed_mask: True where the codon is free in the fit (observed as a
            child at least ``min_count`` times) [61].
        excluded_codons: Sense-codon indices held at ``log_g = 0`` because they
            were never observed as a child, or were observed fewer than
            ``min_count`` times (singletons, by default).
        min_count: The threshold used to build ``observed_mask``.
    """

    child_counts: Tensor
    observed_mask: Tensor
    excluded_codons: List[int]
    min_count: int


def compute_child_coverage(inputs: FitInputs, min_count: int = 2) -> CoverageReport:
    """Count per-codon child observations and gate the free-vector fit on them.

    A sense codon receives an upward likelihood gradient only when observed as
    a child; one that is never an observed child (or observed only once, a
    singleton) gets a one-sided or noise-dominated gradient and would run to
    ``-inf`` (or be unstably estimated) if left free. Restrict the free vector
    to codons observed as a child at least ``min_count`` times in the fitting
    split; the rest are held at ``log_g = 0`` and listed (matches #901's
    coverage reporting).

    Args:
        inputs: Pooled valid-position inputs (the fitting split).
        min_count: Minimum observed-as-child count for a codon to be free.
            Default 2 excludes singletons.

    Returns:
        A :class:`CoverageReport`.
    """
    child_counts = torch.zeros(NUM_SENSE_CODONS, dtype=torch.long)
    if len(inputs) > 0:
        # child_codon_idxs are already restricted to sense codons upstream
        # (collect_fit_inputs drops stop/invalid children), so every count
        # lands on a sense codon's 64-index; map 64->61 via the same index
        # list lift_sense61_to_64 uses.
        counts64 = torch.bincount(inputs.child_codon_idxs, minlength=64)
        child_counts = counts64[SENSE_CODON_LIST_TENSOR]
    observed_mask = child_counts >= min_count
    excluded_codons = [c for c in range(NUM_SENSE_CODONS) if not bool(observed_mask[c])]
    return CoverageReport(
        child_counts=child_counts,
        observed_mask=observed_mask,
        excluded_codons=excluded_codons,
        min_count=min_count,
    )


def fit_free_codon_pref(
    inputs: FitInputs,
    init_log_g: Tensor,
    observed_mask: Tensor,
    max_iter: int = 200,
) -> FitResult:
    """Fit a free 61-vector codon potential by maximizing the in-frame likelihood.

    Only the codons in ``observed_mask`` are free parameters; the rest are
    held at their ``init_log_g`` value (typically 0) throughout the fit — the
    coverage guard (:func:`compute_child_coverage`). Uses L-BFGS with a
    strong-Wolfe line search, mirroring :func:`fit_strengths` but with more
    iterations (61 parameters vs. 2).

    Args:
        inputs: Pooled valid-position inputs.
        init_log_g: Warm-start potential [61] (typically all zeros).
        observed_mask: Which codons are free vs. held fixed [61] bool.
        max_iter: Maximum L-BFGS iterations.

    Returns:
        A :class:`FitResult` with ``log_g`` [61] and ``observed_mask`` set;
        ``betas`` is an empty tensor (no basis coefficients for this fit).
    """
    observed_idxs = observed_mask.nonzero(as_tuple=True)[0]
    fixed_log_g = init_log_g.detach().clone().float()
    free_init = fixed_log_g[observed_idxs].clone().requires_grad_(True)

    def build_log_g(free_vals: Tensor) -> Tensor:
        return fixed_log_g.index_copy(0, observed_idxs, free_vals)

    optimizer = torch.optim.LBFGS(
        [free_init],
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=1e-12,
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        log_g = build_log_g(free_init)
        nll = free_codon_nll(log_g, inputs)
        nll.backward()
        return nll

    optimizer.step(closure)

    with torch.no_grad():
        log_g_final = build_log_g(free_init).detach()
        final_nll = float(free_codon_nll(log_g_final, inputs))

    return FitResult(
        betas=torch.empty(0),
        n_positions=len(inputs),
        nll=final_nll,
        log_g=log_g_final,
        observed_mask=observed_mask.detach().clone(),
    )


_MULTIHIT_MODEL_UNSET = object()


def collect_fit_inputs(
    trainer: "Trainer",
    bundle: Any,
    head_type: "HeadType",
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    multihit_model: Any = _MULTIHIT_MODEL_UNSET,
) -> FitInputs:
    """Collect pooled valid-position inputs from a bundle's in-frame data.

    Runs the model forward once (no grad) for log selection factors and
    recomputes the base (table-free) neutral codon probs at the bundle's
    current branch lengths. Mirrors the in-memory path of the branch-length
    optimizer.

    Shared by :class:`CodonStrengthOptimizer` (the in-training β refit), the
    frozen Rung-1 fit (issue #930), and issue #501's DASM-conditional
    hit-class fit: all three must read the *same* pooled tensors, or the
    ``g≡0``/``θ≡0`` baseline identities fail for reasons unrelated to the
    parameter being fit.

    Args:
        trainer: Trainer whose (possibly frozen) model supplies log selection
            factors via a forward pass.
        bundle: A ``BranchLengthDatasetBundle`` (or duck-typed equivalent)
            exposing ``get_branch_length_optimization_inputs()``, ``dataset``,
            and optionally ``collator``.
        head_type: Which head's output to read as the log selection factors.
        inference_batch_size: Batch size for the forward pass.
        multihit_model: Override for the multihit correction baked into
            ``log_base``. Defaults to the bundle's live
            ``multihit_model`` (matching every caller before issue #501 —
            the production behavior). Pass ``None`` explicitly to get the
            *uncorrected* table-free base: fitting
            ``HitClassModel.log_hit_class_factors`` on top of a ``log_base``
            that already has the multihit correction applied double-applies
            it and makes the ``θ≡0`` identity check pass against the
            pretrained-corrected baseline instead of the truly uncorrected
            one.

    Returns:
        A :class:`FitInputs` pooling every valid in-frame position, with
        ``pcp_idx`` set to each position's source-sample index within
        ``bundle``'s dataset order (0-based), and ``normalization`` resolved
        off the trainer's own DASM loss for ``head_type``.
    """
    model = trainer.unwrapped_model
    device = str(model.device)
    model.eval()

    # Resolve the endpoint kernel once here rather than at each of the five
    # downstream call sites (issue #1238 §4.6). These fits maximize the DASM
    # NLL; running them under a different kernel than training would make
    # the fitted nuisance parameters inconsistent with the likelihood.
    normalization = resolve_dasm_normalization(model, head_type)

    opt_inputs = bundle.get_branch_length_optimization_inputs()
    samples = opt_inputs["samples"]
    if multihit_model is _MULTIHIT_MODEL_UNSET:
        multihit_model = opt_inputs["multihit_model"]
    # The bundle's tandem term (issue #1683), so log_base is the loss's neutral.
    tandem = getattr(getattr(bundle, "neutral_model", None), "tandem", None)

    # Fill per-sample log selection factors via a batched forward pass,
    # matching dataset order (same pattern as the BL optimizer).
    collate_fn = getattr(bundle, "collator", None)
    dl_kwargs: dict = {"batch_size": inference_batch_size, "shuffle": False}
    if collate_fn is not None:
        dl_kwargs["collate_fn"] = collate_fn
    dataloader = DataLoader(bundle.dataset, **dl_kwargs)

    log_sf_per_sample: List[Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model.forward(batch, head_names=[head_type])
            log_sf = outputs[head_type].cpu()
            for i in range(log_sf.shape[0]):
                log_sf_per_sample.append(log_sf[i])

    if len(log_sf_per_sample) != len(samples):
        raise RuntimeError(
            f"Selection-factor count ({len(log_sf_per_sample)}) does not "
            f"match sample count ({len(samples)})."
        )

    log_base_rows: List[Tensor] = []
    log_sf_rows: List[Tensor] = []
    parent_rows: List[Tensor] = []
    child_rows: List[Tensor] = []
    pcp_idx_rows: List[Tensor] = []
    with torch.no_grad():
        for sample_idx, (sample, log_sf_full) in enumerate(
            zip(samples, log_sf_per_sample)
        ):
            actual_len = sample.mask.shape[0]
            # Base (table-free) neutral codon probs at the current BL.
            base_probs = neutral_codon_probs_of_seq(
                sample.codon_parents_idxs,
                sample.mask,
                sample.nt_rates,
                sample.nt_csps,
                sample.branch_length,
                multihit_model=multihit_model,
                codon_pref_log=None,
                tandem=tandem,
            )  # [actual_len, 64], NaN at masked positions
            child = sample.codon_children_idxs[:actual_len]
            valid = sample.mask[:actual_len] & (child < 64)
            if not bool(valid.any()):
                continue
            log_base_rows.append(torch.log(base_probs[valid]))
            log_sf_rows.append(log_sf_full[:actual_len][valid])
            parent_rows.append(
                torch.clamp(sample.codon_parents_idxs[:actual_len][valid], 0, 63)
            )
            child_rows.append(torch.clamp(child[valid], 0, 63))
            pcp_idx_rows.append(
                torch.full((int(valid.sum()),), sample_idx, dtype=torch.long)
            )

    if not log_base_rows:
        return FitInputs(
            log_base=torch.empty(0, 64),
            log_selection_factors=torch.empty(0, 20),
            parent_codon_idxs=torch.empty(0, dtype=torch.long),
            child_codon_idxs=torch.empty(0, dtype=torch.long),
            pcp_idx=torch.empty(0, dtype=torch.long),
            normalization=normalization,
        )

    inputs = FitInputs(
        log_base=torch.cat(log_base_rows, dim=0),
        log_selection_factors=torch.cat(log_sf_rows, dim=0),
        parent_codon_idxs=torch.cat(parent_rows, dim=0).long(),
        child_codon_idxs=torch.cat(child_rows, dim=0).long(),
        pcp_idx=torch.cat(pcp_idx_rows, dim=0),
        normalization=normalization,
    )
    if trainer.is_main_process and len(inputs) > _LARGE_POSITION_WARN:
        logger.warning(
            "Codon strength fit pooled %d positions in memory "
            "(~%.1f GB for the [P, 64] + [P, 20] design); the large-scale "
            "joint fit is issue #752.",
            len(inputs),
            len(inputs) * (64 + 20) * 4 / 1e9,
        )
    return inputs


class CodonStrengthOptimizer(TrainingHook):
    """Post-cycle hook that fits the codon-optimality strengths ``β_k``.

    Between cycles, the hook:
      1. Runs the frozen model over the in-frame training data to get the
         current log selection factors, and recomputes the base (table-free)
         neutral codon probs at the current branch lengths.
      2. Fits ``β_k`` by maximizing the pooled in-frame likelihood.
      3. Rewrites ``codon_pref_log = Σ_k β_k·ξ̃ᵏ`` into the cache of both the
         train and val bundles, keeping the loss and BL paths consistent:
         precomputed bundles go through the shared ``neutral_codon_probs``
         primitive (``update_branch_lengths`` at the current BLs); streaming
         and Parquet-backed bundles get an in-place update of the dataset's
         shared-memory ``codon_pref_log`` tensor
         (``TransformDataset.set_codon_pref_log``, DESIGN.md D6, the same
         live-dataset-state pattern ``BranchLengthOptimizer`` already uses
         for branch lengths — verified by probe, issue #1039, to propagate
         to DataLoader workers exactly at the next cycle's iterator with no
         staleness).
      4. Reports the regression conditioning (CSC–tAI collinearity and the
         design condition number, restricted to single-nt-reachable synonymous
         codons).

    The fitted ``β_k`` and each table's provenance are recorded in checkpoint
    ``metadata.yml`` via :meth:`to_config` (serialized into
    ``training_config.post_hooks``).

    Ordered BEFORE the branch-length optimizer in the post-cycle hook list so
    BL re-optimization sees the freshly fitted table.
    """

    def __init__(
        self,
        tables: List[CodonPreferenceTable],
        init_betas: Optional[List[float]] = None,
        max_iter: int = 50,
        inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    ) -> None:
        """Initialize the codon strength optimizer.

        Args:
            tables: Fixed codon-optimality axes (the ``ξ̃ᵏ`` references), in
                strength-vector column order.
            init_betas: Initial/warm-start strengths. Defaults to 1.0 per axis.
            max_iter: Maximum L-BFGS iterations per refit.
            inference_batch_size: Batch size for model inference.
        """
        if not tables:
            raise ValueError("CodonStrengthOptimizer requires at least one table")
        if init_betas is not None and len(init_betas) != len(tables):
            raise ValueError(
                f"Got {len(init_betas)} init_betas for {len(tables)} tables"
            )
        self.tables = tables
        self.betas: List[float] = (
            list(init_betas) if init_betas is not None else [1.0] * len(tables)
        )
        self.max_iter = max_iter
        self.inference_batch_size = inference_batch_size
        self._within64 = within_aa_64_design(tables)

    def current_codon_pref_log(self) -> Tensor:
        """Return ``Σ_k β_k·ξ̃ᵏ`` (64,) for the current strengths."""
        return build_codon_pref_log(self.tables, self.betas)

    def pre_cycle(self, trainer: "Trainer", **context: Any) -> None:
        """No pre-cycle action; the fit happens post-cycle."""

    def post_cycle(self, trainer: "Trainer", **context: Any) -> None:
        """Re-fit ``β_k`` and rewrite the codon-preference cache.

        Args:
            trainer: Trainer instance with the current model.
            **context: Must include ``train_data``, ``val_data``,
                ``cycle_count`` and ``head_type``.
        """
        required = ("train_data", "val_data", "cycle_count", "head_type")
        missing = [k for k in required if k not in context]
        if missing:
            raise ValueError(f"Hook context missing required keys: {missing}")

        train_bl = context["train_data"]
        val_bl = context["val_data"]
        cycle_count = context["cycle_count"]
        head_type = context["head_type"]

        start = time.perf_counter()

        inputs = collect_fit_inputs(
            trainer, train_bl, head_type, inference_batch_size=self.inference_batch_size
        )
        if len(inputs) == 0:
            if trainer.is_main_process:
                logger.warning(
                    "Codon strength optimizer: no valid in-frame positions; "
                    "leaving betas unchanged."
                )
            return

        result = fit_strengths(
            inputs,
            self._within64,
            torch.tensor(self.betas, dtype=torch.float32),
            max_iter=self.max_iter,
        )
        self.betas = [float(b) for b in result.betas]

        # Rewrite the cache for both bundles via the shared primitive.
        new_pref = self.current_codon_pref_log()
        for bundle in (train_bl, val_bl):
            self._rewrite_cache(bundle, new_pref)

        duration = time.perf_counter() - start
        self._log_diagnostics(result, cycle_count, duration, trainer)

    def _rewrite_cache(self, bundle: Any, codon_pref_log: Tensor) -> None:
        """Set the bundle's codon preference and recompute its cached probs.

        Precomputed mode: uses the bundle's ``update_branch_lengths`` (at the
        current BLs), which funnels through ``neutral_codon_probs`` — the
        same primitive the dataset precompute and the BL hook use — so all
        writers stay consistent.

        Streaming/Parquet mode: mutates the dataset's shared-memory
        ``codon_pref_log`` tensor in place via
        ``TransformDataset.set_codon_pref_log`` — the same live-dataset-
        state pattern ``BranchLengthOptimizer`` already uses for branch
        lengths (DESIGN.md D6). Verified by probe (issue #1039): a
        between-cycle mutation is immediately and exactly visible starting
        with the next cycle's DataLoader iterator, which is how
        ``post_cycle`` actually invokes this — unlike ``MultihitOptimizer``
        (issue #501 trap 4), a codon preference is not baked into the
        transform chain at construction, so there is no equivalent
        unreachable state to raise about here.

        Raises only if the bundle's dataset has no codon_pref_log channel at
        all (streaming/Parquet constructed with ``codon_pref_log=None``) —
        not the real training path, since ``fit_codon_strengths`` always
        implies a configured initial table (see ``train_command.py``).

        ``bundle.codon_pref_log`` is a read-through property to
        ``bundle.dataset.codon_pref_log`` (issue #1039) — there is one
        underlying tensor, not two independently-writable copies, so this
        call is the single point of update for both the loss's live view and
        the branch-length objective's duck-typed lookup.
        """
        bundle.dataset.set_codon_pref_log(codon_pref_log)
        if bundle.dataset.is_precomputed:
            bundle.update_branch_lengths(bundle.branch_lengths)

    def _log_diagnostics(
        self,
        result: FitResult,
        cycle_count: int,
        duration: float,
        trainer: "Trainer",
    ) -> None:
        """Log the fitted strengths and the regression conditioning."""
        if not trainer.is_main_process:
            return
        cond = design_conditioning(self.tables)
        beta_str = ", ".join(
            f"{name}={b:.4f}" for name, b in zip(cond.axis_names, self.betas)
        )
        corr_str = ", ".join(
            f"{a}~{c}={r:.3f}" for (a, c), r in cond.correlations.items()
        )
        logger.info(
            "Codon strength fit (cycle %d): %.2fs | positions=%d | β: %s | "
            "NLL=%.4f | design: n=%d cond=%.3g%s",
            cycle_count,
            duration,
            result.n_positions,
            beta_str,
            result.nll,
            cond.n_codons,
            cond.condition_number,
            f" | corr: {corr_str}" if corr_str else "",
        )

    def to_config(self) -> "HookConfig":
        """Serialize fitted strengths + table provenance for ``metadata.yml``.

        The fitted ``β_k`` and each axis's name/source land in
        ``training_config.post_hooks`` of the checkpoint metadata (DESIGN D9).
        """
        from dasm2.config import HookConfig

        return HookConfig(  # type: ignore[call-arg]
            type="CodonStrengthOptimizer",
            betas=[float(b) for b in self.betas],
            axes=[{"name": t.name, "source": t.source} for t in self.tables],
            max_iter=self.max_iter,
            inference_batch_size=self.inference_batch_size,
        )
