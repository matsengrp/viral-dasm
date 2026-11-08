"""Multihit correction factor fit and between-cycle hook (issue #501).

Fits ``HitClassModel``'s 3 ``log_hit_class_factors`` (hit classes 1-3) by
L-BFGS on the exact pooled in-frame DASM likelihood, holding the trained
selection factors frozen. This replaces the pretrained antibody-fit values
(``dasm2/pretrained.py:34-39``) that are otherwise loaded once and never
updated, with a fit conditioned on the current dataset and the current model's
selection factors.

Sibling to :class:`~dasm2.training.hooks.codon_strength.CodonStrengthOptimizer`
(same between-cycle-refit shape) but a separate hook class per DESIGN D5: a
new per-task behavior is a new hook, not a generalization of an existing one.
The two hooks fit different quantities (a multiplicative codon-optimality
table vs. hit-class factors) via different designs and reuse only the shared
:func:`~dasm2.training.hooks.codon_strength.collect_fit_inputs` collection
primitive.

Estimator (per valid in-frame position *j* with parent codon ``x_j``)::

    log_base_j = log neutral_codon_probs_of_seq(..., multihit_model=None)[j]  # [64]
    log_corr_j = log_base_j + theta_pad[hc_j]      # theta_pad = cat([0], theta)
    log_P_j    = compute_log_adjusted_codon_probs(parent_flat_j, log_corr_j, s_j)
    NLL        = -sum_j log_P_j[child_flat_j]

``collect_fit_inputs`` must be called with ``multihit_model=None`` (trap 1):
its default behavior bakes the bundle's *live* multihit correction into
``log_base``, and fitting ``theta`` on top of an already-corrected base
double-applies the correction.

Dtype: the fit runs in float64 (``theta`` and the pooled ``log_base``); 3-hit
probabilities live at 1e-9..1e-14 and interact with the ``SMALL_PROBABILITY``
floor (see :mod:`dasm2.constants`), where float32 tolerances (this module
copies ``fit_strengths``'s L-BFGS block, whose ``tolerance_change=1e-12`` is
below float32 eps) would make convergence reporting meaningless. The
per-position selection factors ``s_j`` stay float32: they multiply the
module-global ``CODON_AA_INDICATOR_MATRIX``/``STOP_CODON_ZAPPER`` (fixed
float32, in :mod:`dasm2.molevol.mutsel`, which issue #501 leaves unchanged)
inside :func:`compute_log_adjusted_codon_probs`, and PyTorch's matmul refuses
mixed float32/float64 operands. Elementwise addition promotes automatically,
so a float64 ``log_neutral_codon_probs`` plus a float32 selection-lift term
still yields a float64 result — this is why only the neutral/hit-class side
needs the explicit upcast.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from dasm2.constants import SMALL_PROBABILITY
from dasm2.molevol.mutsel import compute_log_adjusted_codon_probs
from dasm2.molevol.tensor_ops import (
    flatten_codons,
    set_parent_codon_prob,
    unflatten_codons,
)
from dasm2.neutral.hit_class import (
    HitClassModel,
    apply_multihit_correction,
    parent_specific_hit_classes,
)
from dasm2.sequence.conversion import unflatten_codon_idxs
from dasm2.training.hooks.base import TrainingHook
from dasm2.training.hooks.codon_strength import (
    DEFAULT_INFERENCE_BATCH_SIZE,
    FitInputs,
    collect_fit_inputs,
)
from dasm2.utils.tensor import clamp_probability, clamp_probability_above_only

if TYPE_CHECKING:
    from dasm2.config import HookConfig
    from dasm2.training.trainer import Trainer

logger = logging.getLogger(__name__)


# =============================================================================
# Differentiable estimator
# =============================================================================


def compute_hit_class_grid(parent_codon_idxs: Tensor) -> Tensor:
    """Per-position hit class (0-3) of every possible child codon [P, 64].

    Args:
        parent_codon_idxs: Flat (0-63) parent codon indices [P].

    Returns:
        Long tensor [P, 64]: ``hc_flat[j, c]`` is the hit class of flat child
        codon ``c`` relative to position ``j``'s parent codon.
    """
    parent_nt = unflatten_codon_idxs(parent_codon_idxs)  # [P, 3]
    hc_grid = parent_specific_hit_classes(parent_nt)  # [P, 4, 4, 4]
    return flatten_codons(hc_grid).long()  # [P, 64]


def hit_class_child_log_probs(
    theta: Tensor, inputs: FitInputs, hc_flat: Tensor
) -> Tensor:
    """Per-position ``log P(child)`` as a function of the 3 hit-class factors.

    Differentiable in ``theta``. ``inputs.log_base`` must have been collected
    with ``multihit_model=None`` (trap 1) — this function applies the hit-class
    correction itself, additively in log space, so it must start from the
    truly uncorrected base or the correction is double-applied.

    Args:
        theta: Log hit-class factors for hit classes 1-3, shape [3].
        inputs: Pooled valid-position inputs (``log_base`` uncorrected).
        hc_flat: Per-position hit-class grid [P, 64] from
            :func:`compute_hit_class_grid`, precomputed once per fit since it
            depends only on ``parent_codon_idxs``, not on ``theta``.

    Returns:
        Per-position ``log P(child)``, shape [P], float64.
    """
    zero = theta.new_zeros(1)
    theta_pad = torch.cat([zero, theta])  # [4], hit class 0 has no free factor
    log_corr = inputs.log_base.double() + theta_pad[hc_flat]  # [P, 64], float64
    log_adjusted = compute_log_adjusted_codon_probs(
        inputs.parent_codon_idxs,
        log_corr,
        inputs.log_selection_factors,
        normalization=inputs.normalization,
    )  # [P, 64], float64 (promoted; see module docstring on dtype)
    return log_adjusted.gather(-1, inputs.child_codon_idxs.unsqueeze(-1)).squeeze(-1)


def hit_class_nll(theta: Tensor, inputs: FitInputs, hc_flat: Tensor) -> Tensor:
    """Pooled in-frame DASM NLL as a function of the 3 hit-class factors.

    See :func:`hit_class_child_log_probs`; applies ``inputs.weights`` (if any),
    sums, and negates.
    """
    child_log_probs = hit_class_child_log_probs(theta, inputs, hc_flat)
    if inputs.weights is not None:
        child_log_probs = child_log_probs * inputs.weights.double()
    return -child_log_probs.sum()


@dataclass(frozen=True)
class HitClassFitResult:
    """Result of the L-BFGS hit-class factor fit.

    Attributes:
        theta: Fitted log hit-class factors [3] (float64).
        n_positions: Number of pooled positions the fit ran over.
        nll: Final NLL at ``theta``.
        grad_norm: Gradient norm at the last L-BFGS closure evaluation.
    """

    theta: Tensor
    n_positions: int
    nll: float
    grad_norm: float


def fit_hit_class_factors_by_lbfgs(
    inputs: FitInputs,
    init_theta: Tensor,
    max_iter: int = 50,
) -> HitClassFitResult:
    """Fit the 3 hit-class log factors by maximizing the pooled in-frame likelihood.

    Uses L-BFGS with a strong-Wolfe line search, copying the optimizer block
    from ``fit_strengths`` (``dasm2/training/hooks/codon_strength.py``), but in
    float64 (see module docstring).

    Args:
        inputs: Pooled valid-position inputs, with ``log_base`` collected via
            ``collect_fit_inputs(..., multihit_model=None)`` (trap 1).
        init_theta: Warm-start factors [3].
        max_iter: Maximum L-BFGS iterations.

    Returns:
        A :class:`HitClassFitResult`.
    """
    hc_flat = compute_hit_class_grid(inputs.parent_codon_idxs)
    theta = init_theta.detach().clone().double().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [theta],
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=1e-12,
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        nll = hit_class_nll(theta, inputs, hc_flat)
        nll.backward()
        return nll

    optimizer.step(closure)

    grad_norm = float(theta.grad.norm()) if theta.grad is not None else float("nan")
    with torch.no_grad():
        final_nll = float(hit_class_nll(theta, inputs, hc_flat))

    return HitClassFitResult(
        theta=theta.detach().clone(),
        n_positions=len(inputs),
        nll=final_nll,
        grad_norm=grad_norm,
    )


# =============================================================================
# Probability-floor diagnostic
# =============================================================================
#
# Three arms at the same theta, all funneled through compute_log_adjusted_codon_probs
# plus a gather so they are at matched per-position granularity (a per-PCP
# float, e.g. compute_dasm_log_pcp_probability, cannot be compared this way):
#
#   (a) production      — HitClassModel.forward() (both SMALL_PROBABILITY
#                          floor clamps: hit_class.py's _normalize_corrected_probs
#                          and neutral_evolution.py's neutral_codon_probs_of_seq).
#   (b) clamp-free       — the same multihit renormalization, with both floor
#                          clamps skipped (the ceiling-only
#                          clamp_probability_above_only is kept: it guards the
#                          parent-absorb's numerical validity and is not the
#                          floor under study).
#   (c) fit form         — hit_class_child_log_probs itself: theta applied
#                          additively in log space, no intermediate
#                          renormalize-then-clamp step at all. Has at most the
#                          *outer* floor clamp baked into log_base (multihit=None),
#                          which is inert in practice: it clamps the
#                          *uncorrected* base, and pre-correction 3-hit
#                          probabilities (products of per-nt mutation
#                          probabilities) are not usually near 1e-10 — only the
#                          multihit-*corrected* ones are. This is why (c) is
#                          the "single-normalization" form.
#
# All three read inputs.normalization (issue #1238), so the comparison stays
# a comparison of floors at one theta. Threading the endpoint kernel into
# only the fit arm would leave a conditional (c) against a native (a) and
# (b) and make the diagnostic meaningless.


def _uncorrected_probs_unflattened(inputs: FitInputs) -> Tensor:
    """Recover ``[P, 4, 4, 4]`` linear uncorrected codon probs from ``log_base``.

    ``inputs.log_base`` must come from ``collect_fit_inputs(..., multihit_model=None)``.
    """
    return unflatten_codons(inputs.log_base.exp())


def _diagnostic_context(inputs: FitInputs) -> Tuple[Tensor, Tensor]:
    """``(parent_nt, uncorrected)`` shared setup for the three floor-diagnostic
    arms below -- each needs both the unflattened parent codon and the
    uncorrected probability grid before diverging into its own correction
    path."""
    return unflatten_codon_idxs(
        inputs.parent_codon_idxs
    ), _uncorrected_probs_unflattened(inputs)


def production_child_log_probs(
    theta_values: Tuple[float, float, float], inputs: FitInputs
) -> Tensor:
    """Floor-diagnostic arm (a): the real production path at a fixed theta.

    Builds a (non-differentiable) ``HitClassModel`` from ``theta_values`` and
    runs the exact ``HitClassModel.forward`` -> outer-clamp pipeline that
    ``neutral_codon_probs_of_seq``/``collect_fit_inputs`` use in training.

    Args:
        theta_values: Fixed hit-class factors to evaluate at.
        inputs: Pooled valid-position inputs (``log_base`` uncorrected).

    Returns:
        Per-position ``log P(child)``, shape [P].
    """
    model = HitClassModel.from_weights(theta_values)
    parent_nt, uncorrected = _diagnostic_context(inputs)
    with torch.no_grad():
        corrected = model.forward(parent_nt, uncorrected)  # clamp #1 inside
        normalized = flatten_codons(corrected)
        log_neutral = torch.log(clamp_probability(normalized))  # clamp #2
        log_adjusted = compute_log_adjusted_codon_probs(
            inputs.parent_codon_idxs,
            log_neutral,
            inputs.log_selection_factors,
            normalization=inputs.normalization,
        )
        return log_adjusted.gather(-1, inputs.child_codon_idxs.unsqueeze(-1)).squeeze(
            -1
        )


def clamp_free_child_log_probs(
    theta_values: Tuple[float, float, float], inputs: FitInputs
) -> Tensor:
    """Floor-diagnostic arm (b): production path with both floor clamps removed.

    Mirrors :func:`production_child_log_probs`'s renormalization exactly, but
    skips both ``clamp_probability`` (floor+ceiling) calls the production path
    makes — only the numerical-safety ceiling clamp
    (``clamp_probability_above_only``) is kept.

    Args:
        theta_values: Fixed hit-class factors to evaluate at.
        inputs: Pooled valid-position inputs (``log_base`` uncorrected).

    Returns:
        Per-position ``log P(child)``, shape [P].
    """
    theta = torch.tensor(theta_values, dtype=torch.float32)
    parent_nt, uncorrected = _diagnostic_context(inputs)
    with torch.no_grad():
        corrected = apply_multihit_correction(parent_nt, uncorrected, theta)
        clamped_above = clamp_probability_above_only(corrected)
        normalized = set_parent_codon_prob(
            flatten_codons(clamped_above), inputs.parent_codon_idxs
        )
        log_neutral = torch.log(normalized.clamp(min=1e-300))
        log_adjusted = compute_log_adjusted_codon_probs(
            inputs.parent_codon_idxs,
            log_neutral,
            inputs.log_selection_factors,
            normalization=inputs.normalization,
        )
        return log_adjusted.gather(-1, inputs.child_codon_idxs.unsqueeze(-1)).squeeze(
            -1
        )


def floor_occupancy_fraction(
    theta_values: Tuple[float, float, float], inputs: FitInputs
) -> float:
    """Fraction of the [P, 64] production multihit-corrected grid sitting at the floor.

    "Scored (position, child) entries" is read as every entry of the [P, 64]
    grid the fit's NLL sums over (not just the observed child per position):
    the corrected, pre-selection probability at the ``SMALL_PROBABILITY``
    floor (see :mod:`dasm2.constants`), before the DASM selection lift.

    Args:
        theta_values: Fixed hit-class factors to evaluate at.
        inputs: Pooled valid-position inputs (``log_base`` uncorrected).

    Returns:
        Fraction in [0, 1].
    """
    model = HitClassModel.from_weights(theta_values)
    parent_nt, uncorrected = _diagnostic_context(inputs)
    with torch.no_grad():
        corrected = model.forward(parent_nt, uncorrected)
        normalized = flatten_codons(corrected)
        at_floor = torch.isclose(
            normalized,
            torch.full_like(normalized, SMALL_PROBABILITY),
            atol=SMALL_PROBABILITY * 1e-3,
            rtol=0.0,
        )
        return float(at_floor.float().mean())


@dataclass(frozen=True)
class FloorDiagnostic:
    """The three-arm floor diagnostic report (issue #501 success criterion 4).

    Attributes:
        theta: The theta the diagnostic was run at.
        n_positions: Number of pooled positions scored.
        nll_production: Pooled NLL, arm (a).
        nll_clamp_free: Pooled NLL, arm (b).
        nll_fit_form: Pooled NLL, arm (c) — matches ``hit_class_nll``.
        floor_occupancy_fraction: Fraction of the [P, 64] production
            multihit-corrected grid sitting at the floor.
    """

    theta: Tuple[float, float, float]
    n_positions: int
    nll_production: float
    nll_clamp_free: float
    nll_fit_form: float
    floor_occupancy_fraction: float

    @property
    def production_vs_clamp_free_gap_nats_per_position(self) -> float:
        """(NLL_production - NLL_clamp_free) / n_positions; large => floor binds."""
        return (self.nll_production - self.nll_clamp_free) / max(1, self.n_positions)


def compute_floor_diagnostic(
    theta_values: Tuple[float, float, float], inputs: FitInputs
) -> FloorDiagnostic:
    """Run the three-arm floor diagnostic at a fixed theta.

    Args:
        theta_values: The hit-class factors to evaluate at (e.g. the fitted
            theta, the antibody-pretrained triple, or a synthetic extreme).
        inputs: Pooled valid-position inputs (``log_base`` uncorrected).

    Returns:
        A :class:`FloorDiagnostic`.
    """
    n_positions = len(inputs)
    nll_production = -float(production_child_log_probs(theta_values, inputs).sum())
    nll_clamp_free = -float(clamp_free_child_log_probs(theta_values, inputs).sum())
    theta_tensor = torch.tensor(theta_values, dtype=torch.float64)
    hc_flat = compute_hit_class_grid(inputs.parent_codon_idxs)
    with torch.no_grad():
        nll_fit_form = float(hit_class_nll(theta_tensor, inputs, hc_flat))
    occupancy = floor_occupancy_fraction(theta_values, inputs)
    return FloorDiagnostic(
        theta=theta_values,
        n_positions=n_positions,
        nll_production=nll_production,
        nll_clamp_free=nll_clamp_free,
        nll_fit_form=nll_fit_form,
        floor_occupancy_fraction=occupancy,
    )


def compute_floor_diagnostic_by_hit_class(
    theta_values: Tuple[float, float, float], inputs: FitInputs
) -> Dict[int, FloorDiagnostic]:
    """:func:`compute_floor_diagnostic`, stratified by hit class (0-3).

    The pooled diagnostic has ~no power to detect a floor effect concentrated
    at hit class 3, where theta_3 acts and where a binding floor would
    attenuate exactly the gradient this issue's estimator needs there (issue
    #501 follow-up, surprising-conclusion-skeptic review of PR #1025) — the
    pooled aggregate is dominated by hit-class-0/1 positions, where a 1e-10
    floor essentially never binds.

    The two halves of :class:`FloorDiagnostic` are stratified along
    *different* axes of the ``[P, 64]`` grid, deliberately: ``nll_*`` sums
    the observed child's log-prob per position, so it is grouped by the
    *observed* substitution's hit class (matching the convention of
    ``hit_class_nll_by_codon_step`` in
    ``experiments/2026-07-30-multihit-refit/lib.py``, an experiment that keeps
    its pre-#1644 names -- "how well does
    the fit do on hit-class-k substitutions"). ``floor_occupancy_fraction``
    and the clamp-free gap are properties of individual *candidate* grid
    entries, not of which position they happen to sit in, so they are
    grouped by each entry's own hit class instead -- the mechanistically
    relevant question is "does the floor bind on hit-class-3 *candidates*",
    since those are exactly the entries theta_3 corrects, regardless of what
    was actually observed at that position. Averaging entry-level floor
    occupancy over a whole row (grouped by the row's observed outcome
    instead) would dilute a real hit-class-3 floor effect across the other
    63 candidates' hit classes in that row.

    Args:
        theta_values: The hit-class factors to evaluate at.
        inputs: Pooled valid-position inputs (``log_base`` uncorrected).

    Returns:
        One :class:`FloorDiagnostic` per hit class present in ``inputs``
        (0-3). ``n_positions`` is the observed-outcome count that scales
        the NLL fields; ``floor_occupancy_fraction`` is computed over a
        differently-sized (candidate-entry) population -- see the
        docstring above.
    """
    hc_flat = compute_hit_class_grid(inputs.parent_codon_idxs)
    hit_class_of_observed = hc_flat.gather(
        -1, inputs.child_codon_idxs.unsqueeze(-1)
    ).squeeze(-1)

    theta_tensor = torch.tensor(theta_values, dtype=torch.float64)
    model = HitClassModel.from_weights(theta_values)
    parent_nt, uncorrected = _diagnostic_context(inputs)
    with torch.no_grad():
        per_position_production = production_child_log_probs(theta_values, inputs)
        per_position_clamp_free = clamp_free_child_log_probs(theta_values, inputs)
        per_position_fit_form = hit_class_child_log_probs(theta_tensor, inputs, hc_flat)
        normalized = flatten_codons(model.forward(parent_nt, uncorrected))
        at_floor = torch.isclose(
            normalized,
            torch.full_like(normalized, SMALL_PROBABILITY),
            atol=SMALL_PROBABILITY * 1e-3,
            rtol=0.0,
        )

    result: Dict[int, FloorDiagnostic] = {}
    for hc in sorted(hit_class_of_observed.unique().tolist()):
        observed_mask = hit_class_of_observed == hc
        entry_mask = hc_flat == hc  # [P, 64] -- this hit class's own candidates
        result[hc] = FloorDiagnostic(
            theta=theta_values,
            n_positions=int(observed_mask.sum()),
            nll_production=-float(per_position_production[observed_mask].sum()),
            nll_clamp_free=-float(per_position_clamp_free[observed_mask].sum()),
            nll_fit_form=-float(per_position_fit_form[observed_mask].sum()),
            floor_occupancy_fraction=float(at_floor[entry_mask].float().mean()),
        )
    return result


# =============================================================================
# Phase 2: MultihitOptimizer between-cycle hook
# =============================================================================


def _write_hit_class_factors(
    model: HitClassModel, theta: Tuple[float, float, float]
) -> None:
    """Write ``theta`` into ``model``'s existing hit-class parameter in place.

    ``copy_``, never ``reinitialize_weights``/``from_weights``: those rebind
    the ``nn.Parameter``, which would break the shared-memory channel that
    lets a refit reach streaming/Parquet DataLoader workers (issue #1041; see
    ``dasm2.data.evolutionary_dataset._prepare_transform_multihit``).

    Via ``.values`` for the same reason that function uses it: it is the
    parameter name netam's ``HitClassModel`` uses and the read-only alias
    dasm2's exposes, so one in-place write covers both.
    """
    # no_grad is load-bearing: the trainer's neutral model comes from
    # HitClassModel.from_weights, which never clears requires_grad (unlike the
    # bundles' copies, frozen by _prepare_transform_multihit).
    with torch.no_grad():
        model.values.copy_(torch.tensor(theta, dtype=model.values.dtype))


class MultihitOptimizer(TrainingHook):
    """Post-cycle hook that refits the multihit hit-class factors ``theta``.

    Between cycles, the hook:
      1. Collects pooled in-frame positions via ``collect_fit_inputs`` with
         ``multihit_model=None`` (trap 1), so ``log_base`` is the truly
         uncorrected table.
      2. Fits ``theta`` by :func:`fit_hit_class_factors_by_lbfgs`, warm-started
         from the current ``theta``.
      3. Writes ``theta`` *in place* into each bundle's existing
         ``HitClassModel``, and into the trainer's neutral model so the fitted
         factors reach checkpoint metadata (trap 5). Precomputed bundles then
         also get ``update_branch_lengths``, funneling the new factors through
         the same ``neutral_codon_probs`` primitive both cache-writers use.

    Reaches every loader/mode (CSV, pickle, Parquet; precomputed or
    streaming). A bundle's ``multihit_model`` *is* the object baked into its
    per-sample transform pipeline, and its ``log_hit_class_factors`` live in
    shared memory (``_prepare_transform_multihit`` in
    ``dasm2/data/evolutionary_dataset.py``), so an in-place ``copy_`` here is
    picked up by streaming/Parquet DataLoader workers on their next
    ``__getitem__`` — the same live-dataset-state pattern
    ``BranchLengthOptimizer`` uses for branch lengths and
    ``CodonStrengthOptimizer`` uses for ``codon_pref_log`` (DESIGN.md D6,
    issue #1041, superseding #501 trap 4's precomputed-only restriction).

    This is also why the refit must never install a *fresh* ``HitClassModel``
    (as it did under #501 trap 3): ``reinitialize_weights``, and so
    ``HitClassModel.from_weights``, rebinds the ``nn.Parameter``, replacing the
    shared tensor with a private one and silently stranding every worker on the
    construction-time factors. Mutate, never rebind.

    Ordered BEFORE ``CodonStrengthOptimizer`` and ``BranchLengthOptimizer`` in
    the post-cycle hook list: ``CodonStrengthOptimizer.post_cycle`` calls
    ``collect_fit_inputs`` *without* a multihit override, so it bakes in
    whatever ``multihit_model`` is live on the bundle at that point — for its
    fit to see this cycle's fresh ``theta``, this hook must run first. Branch-
    length re-optimization reads ``inputs.multihit_model`` directly out of the
    optimization-inputs container (``dasm2/training/hooks/branch_lengths.py``),
    so it too needs the fresh factors installed before it runs.

    Opt-in wiring: issue #501 originally specified opt-in via a generic
    ``{type: MultihitOptimizer, ...}`` entry in a task's hook-config list.
    That generic mechanism (``HOOK_REGISTRY``/``create_hook``, this hook *is*
    registered there for config round-trip parity with
    ``DASMFitAlphaOptimizer``) is not actually wired into live training runs
    today — ``dasm2/train_command.py`` builds every real ``Task``'s
    ``pre_cycle_hooks``/``post_cycle_hooks`` by constructing hook instances
    directly and appending them based on dedicated config flags (this is how
    ``CodonStrengthOptimizer`` itself is wired, via ``fit_codon_strengths``,
    contrary to the issue's characterization of that as a superseded pattern).
    This hook follows the same, actually-live convention via
    ``fit_multihit_factors``.
    """

    def __init__(
        self,
        max_iter: int = 50,
        inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
        init_theta: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Initialize the multihit optimizer.

        Args:
            max_iter: Maximum L-BFGS iterations per refit.
            inference_batch_size: Batch size for model inference.
            init_theta: Initial hit-class factors, used to warm-start the first
                refit. Defaults to ``(0.0, 0.0, 0.0)`` (no correction) if the
                caller passes nothing. In practice ``train_command.py`` passes
                the task's neutral model's live ``multihit_model.to_weights()``
                here, so the first refit actually warm-starts from whatever
                factors the bundle was constructed with — this class itself
                has no knowledge of the bundle at construction time.
        """
        self.max_iter = max_iter
        self.inference_batch_size = inference_batch_size
        # `is not None` (not truthiness): after a create_hook() round-trip,
        # init_theta arrives as a plain list (HookConfig is a free-form model,
        # so Pydantic doesn't coerce it back to a tuple) -- normalize to a
        # tuple explicitly rather than silently letting self.theta hold
        # whatever container type the caller passed.
        if init_theta is None:
            self.theta: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        else:
            t0, t1, t2 = init_theta
            self.theta = (t0, t1, t2)

    def pre_cycle(self, trainer: "Trainer", **context: Any) -> None:
        """No pre-cycle action; the fit happens post-cycle."""

    def post_cycle(self, trainer: "Trainer", **context: Any) -> None:
        """Re-fit ``theta`` and rewrite the multihit cache.

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
            trainer,
            train_bl,
            head_type,
            inference_batch_size=self.inference_batch_size,
            multihit_model=None,  # trap 1: fit against the uncorrected base
        )
        if len(inputs) == 0:
            if trainer.is_main_process:
                logger.warning(
                    "Multihit optimizer: no valid in-frame positions; "
                    "leaving theta unchanged."
                )
            return

        result = fit_hit_class_factors_by_lbfgs(
            inputs,
            torch.tensor(self.theta, dtype=torch.float64),
            max_iter=self.max_iter,
        )
        theta_values = result.theta.tolist()
        self.theta = (theta_values[0], theta_values[1], theta_values[2])

        for bundle in (train_bl, val_bl):
            self._rewrite_cache(bundle)
        self._update_neutral_model(trainer)

        duration = time.perf_counter() - start
        self._log_diagnostics(result, cycle_count, duration, trainer)

    def _rewrite_cache(self, bundle: Any) -> None:
        """Write ``self.theta`` into the bundle's live multihit model.

        The bundle's ``multihit_model`` is the same object its per-sample
        transform pipeline holds, with shared-memory
        ``log_hit_class_factors`` — so for streaming and Parquet-backed
        bundles this in-place write *is* the update: workers read the new
        factors on their next ``__getitem__`` (DESIGN.md D6). Precomputed
        bundles cache ``log_neutral_codon_probs`` instead of recomputing it
        per sample, so they additionally need ``update_branch_lengths`` (at
        the current BLs) to rebake the cache through the shared
        ``neutral_codon_probs`` primitive.

        Raises:
            RuntimeError: If the bundle has no multihit model at all — there
                is no shared parameter for the refit to reach. Not the real
                training path, since ``fit_multihit_factors`` requires a
                configured ``neutral_model.multihit_model_name`` (see
                ``train_command.py``).
        """
        if bundle.multihit_model is None:
            raise RuntimeError(
                "MultihitOptimizer cannot refit a bundle with no multihit "
                "model: the dataset was constructed with multihit_model=None, "
                "so there is no shared-memory hit-class parameter to update "
                "in place. Configure neutral_model.multihit_model_name."
            )
        _write_hit_class_factors(bundle.multihit_model, self.theta)
        if bundle.dataset.is_precomputed:
            bundle.update_branch_lengths(bundle.branch_lengths)

    def _update_neutral_model(self, trainer: "Trainer") -> None:
        """Write ``self.theta`` into the trainer's neutral model (trap 5).

        ``Trainer.export_metadata`` serializes ``neutral_model.to_config()``,
        and ``MultihitWrapper.to_config`` reads its live
        ``multihit_model.to_weights()`` into ``multihit_log_factors`` — the
        field that takes precedence when the checkpoint is reloaded. The
        bundles hold their own deep copies, so without this write a refit run
        would save (and reload) the warm-start triple rather than the fitted
        one.
        """
        neutral_multihit = getattr(trainer.neutral_model, "multihit_model", None)
        if neutral_multihit is None:
            if trainer.is_main_process:
                logger.warning(
                    "Multihit optimizer: trainer's neutral model has no "
                    "multihit_model, so the fitted factors will not reach "
                    "checkpoint metadata's multihit_log_factors."
                )
            return
        _write_hit_class_factors(neutral_multihit, self.theta)

    def _log_diagnostics(
        self,
        result: HitClassFitResult,
        cycle_count: int,
        duration: float,
        trainer: "Trainer",
    ) -> None:
        """Log the fitted theta and convergence diagnostics."""
        if not trainer.is_main_process:
            return
        theta_str = ", ".join(f"{t:.4f}" for t in self.theta)
        logger.info(
            "Multihit factor fit (cycle %d): %.2fs | positions=%d | "
            "theta=[%s] | NLL=%.4f | grad_norm=%.3e",
            cycle_count,
            duration,
            result.n_positions,
            theta_str,
            result.nll,
            result.grad_norm,
        )

    def to_config(self) -> "HookConfig":
        """Serialize fitted theta for ``metadata.yml`` (DESIGN D9)."""
        from dasm2.config import HookConfig

        return HookConfig(  # type: ignore[call-arg]
            type="MultihitOptimizer",
            init_theta=list(self.theta),
            max_iter=self.max_iter,
            inference_batch_size=self.inference_batch_size,
        )
