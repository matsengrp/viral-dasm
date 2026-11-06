"""Branch length optimization utilities for molecular evolution.

Functions for warm-started bounded optimization of branch length parameters.
Uses closed-form estimates as starting points with fixed bounds, replacing the
previous bracketing heuristic approach.

Also provides a neutral-GY94 branch-length profile-likelihood scorer (issue
#943) for diagnosing `--reestimate-bl` branch-length overshoot independently
of DASM's selection-weighted objective: given externally-supplied GY94
kappa/omega/codon-frequencies (e.g. phyz's own fitted model), scores LL(t) on
a grid of candidate branch lengths for a fixed (ASR-reconstructed) parent/child
codon sequence pair, and classifies the edge as optimizer-failure /
likelihood-ridge / data-supported.

See Also:
    Issue #499 for the warm-start analysis and benchmarks.
    Issue #943 for the profile-likelihood scorer and classification thresholds.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional
from warnings import warn

import numpy as np
import torch
from scipy import optimize  # type: ignore[import-untyped]

from dasm2.codon_table import (
    NUM_SENSE_CODONS,
    SENSE_CODON_DIFF_I,
    SENSE_CODON_DIFF_J,
    SENSE_CODON_NT_PAIRS,
    SENSE_CODON_TO_AA,
)
from dasm2.molevol.tensor_ops import normalize_rate_matrix

# GY94 transitions: A<->G (indices 0,2) and C<->T (indices 1,3), matching the
# ACGT nucleotide index convention shared by dasm2.sequence.constants.BASES
# and phyz's `codon.zig` (verified identical: both use A=0, C=1, G=2, T=3).
_TRANSITION_NT_PAIRS = {(0, 2), (2, 0), (1, 3), (3, 1)}

# Classification thresholds pinned by issue #943 (do not tune per-edge).
GY94_OPTIMIZER_FAILURE_TAU0 = 0.01
GY94_RIDGE_DELTA_LL = 2.0
GY94_DATA_SUPPORTED_T_MIN = 0.1

EdgeProfileLabel = Literal["optimizer_failure", "ridge", "data_supported", "ambiguous"]

# Default factor for bounds around the warm estimate in log-space.
# bounds = [estimate / BOUND_FACTOR, estimate * BOUND_FACTOR]
DEFAULT_BOUND_FACTOR = 3.0

# Absolute log-space distance from a bound below which we widen and retry.
LOG_BOUNDARY_TOL = 0.01

# Absolute floor/ceiling for log branch lengths.
LOG_BL_LOWER = -14.0  # exp(-14) ~ 8e-7
LOG_BL_UPPER = 4.0  # exp(4) ~ 55


def optimize_branch_length(
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    warm_estimate: float,
    max_optimization_steps: int = 1000,
    optimization_tol: float = 1e-4,
    bound_factor: float = DEFAULT_BOUND_FACTOR,
) -> tuple[float, bool]:
    """Optimize branch length with warm start from a closed-form estimate.

    Uses ``scipy.optimize.minimize_scalar`` with ``method="bounded"`` centered
    on the warm estimate. If the optimum is within ``LOG_BOUNDARY_TOL`` of either
    bound, the bounds are widened by another factor and optimization re-runs.

    Args:
        log_prob_fn: Function taking log(branch_length) tensor and returning
            log probability (to be maximized).
        warm_estimate: Closed-form branch length estimate to center bounds around.
            Must be positive.
        max_optimization_steps: Maximum iterations for scipy.
        optimization_tol: Convergence tolerance (xatol for bounded method).
        bound_factor: Multiplier for bounds in linear space. Default 3.0 means
            bounds span [estimate/3, estimate*3].

    Returns:
        Tuple of (optimized_branch_length, maxiter_exceeded).
    """
    assert warm_estimate > 0, f"warm_estimate must be positive, got {warm_estimate}"

    def loss_func(x: float) -> float:
        return float(-log_prob_fn(torch.tensor(x)).item())

    log_estimate = np.log(max(warm_estimate, np.exp(LOG_BL_LOWER)))
    log_bound_width = np.log(bound_factor)
    lower = max(log_estimate - log_bound_width, LOG_BL_LOWER)
    upper = min(log_estimate + log_bound_width, LOG_BL_UPPER)
    # Ensure valid range
    upper = max(upper, lower + 0.1)

    result = optimize.minimize_scalar(
        loss_func,
        bounds=(lower, upper),
        options={"xatol": optimization_tol, "maxiter": max_optimization_steps},
        method="bounded",
    )

    # If optimum is near a bound, widen and retry once.
    near_bound = (result.x - lower < LOG_BOUNDARY_TOL) or (
        upper - result.x < LOG_BOUNDARY_TOL
    )
    if near_bound:
        new_lower = max(lower - log_bound_width, LOG_BL_LOWER)
        new_upper = min(upper + log_bound_width, LOG_BL_UPPER)
        new_upper = max(new_upper, new_lower + 0.1)
        if new_lower < lower or new_upper > upper:
            lower, upper = new_lower, new_upper
            result = optimize.minimize_scalar(
                loss_func,
                bounds=(lower, upper),
                options={
                    "xatol": optimization_tol,
                    "maxiter": max_optimization_steps,
                },
                method="bounded",
            )

    maxiter_exceeded = not result.success

    # Warn if result is near either absolute bound (even after widening).
    if result.x < LOG_BL_LOWER + 0.5:
        warn(
            "Optimization result is near lower threshold. This may indicate a problem."
        )
    if result.x > LOG_BL_UPPER - 0.5:
        warn(
            "Optimization result is near upper threshold. This may indicate a problem."
        )
    return np.exp(result.x), maxiter_exceeded


# =============================================================================
# Neutral-GY94 branch-length profile likelihood (issue #943)
# =============================================================================


def build_gy94_rate_matrix(
    kappa: float,
    omega: float,
    codon_frequencies: torch.Tensor,
) -> torch.Tensor:
    """Build a normalized 61x61 GY94 codon rate matrix from kappa/omega/pi.

    Implements the standard M0 GY94 model (Goldman & Yang 1994) exactly as
    parameterized by phyz's `--codon-model GY94`: for sense codon pairs
    differing at exactly one nucleotide position,

        Q[i, j] = pi[j] * kappa^(is_transition) * omega^(is_nonsynonymous)

    and Q[i, j] = 0 for codons differing at two or more positions. This
    differs from `dasm2.neutral.gtr_codon.GTRCodonNeutralModel`, which builds
    a codon-GTR mutation-only matrix with no omega term (DASM supplies
    selection separately) and takes 4 nucleotide frequencies rather than 61
    codon frequencies — it cannot represent phyz's fitted GY94 model, which
    carries a real dN/dS omega and F3x4-derived codon frequencies directly.

    Args:
        kappa: Transition/transversion rate ratio (must be positive).
        omega: Nonsynonymous/synonymous rate ratio, i.e. dN/dS (must be positive).
        codon_frequencies: Shape (61,) equilibrium codon frequencies (sense
            codons only), must be non-negative and sum to 1.

    Returns:
        Normalized 61x61 rate matrix (rows sum to 0, mean substitution rate
        = 1, so branch length t is in expected-substitutions-per-site units —
        matching phyz's `unit_rate` Q-scaling convention used whenever
        `--reestimate-bl` is set).

    Raises:
        ValueError: If kappa/omega are not positive, or codon_frequencies has
            the wrong shape or does not sum to 1.
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if omega <= 0:
        raise ValueError(f"omega must be positive, got {omega}")
    if codon_frequencies.shape != (NUM_SENSE_CODONS,):
        raise ValueError(
            f"codon_frequencies must have shape ({NUM_SENSE_CODONS},), "
            f"got {tuple(codon_frequencies.shape)}"
        )
    if (codon_frequencies < 0).any():
        raise ValueError(
            f"codon_frequencies must be non-negative, got min="
            f"{codon_frequencies.min().item()}"
        )
    freq_sum = codon_frequencies.sum()
    if not torch.isclose(freq_sum, torch.ones_like(freq_sum), atol=1e-6):
        raise ValueError(f"codon_frequencies must sum to 1.0, got {freq_sum.item()}")

    codon_frequencies = codon_frequencies.double()

    is_transition = torch.tensor(
        [
            (int(a), int(b)) in _TRANSITION_NT_PAIRS
            for a, b in SENSE_CODON_NT_PAIRS.tolist()
        ],
        dtype=torch.bool,
    )
    aa_i = SENSE_CODON_TO_AA[SENSE_CODON_DIFF_I]
    aa_j = SENSE_CODON_TO_AA[SENSE_CODON_DIFF_J]
    is_nonsynonymous = aa_i != aa_j

    target_freqs = codon_frequencies[SENSE_CODON_DIFF_J]
    rates = target_freqs.clone()
    rates = torch.where(is_transition, rates * kappa, rates)
    rates = torch.where(is_nonsynonymous, rates * omega, rates)

    Q = torch.zeros((NUM_SENSE_CODONS, NUM_SENSE_CODONS), dtype=torch.float64)
    Q[SENSE_CODON_DIFF_I, SENSE_CODON_DIFF_J] = rates
    Q.diagonal().copy_(-Q.sum(dim=1))

    return normalize_rate_matrix(Q, codon_frequencies)


def gy94_log_likelihood_grid(
    Q: torch.Tensor,
    parent_codon_idxs: torch.Tensor,
    child_codon_idxs: torch.Tensor,
    mask: torch.Tensor,
    branch_lengths: torch.Tensor,
) -> torch.Tensor:
    """Score LL(t) on a grid of branch lengths for a fixed codon sequence pair.

    Computes the profile log-likelihood of a single edge under the neutral
    GY94 CTMC, given ASR-reconstructed (fixed, observed) parent and child
    codon states at each site — no marginalization over ancestral state, since
    joint ASR has already resolved it. This is the likelihood phyz's
    `--reestimate-bl` optimizes; scoring it independently (via scipy-free,
    vectorized batched `matrix_exp`) is what lets Phase 1 classify whether
    phyz's reported optimum is real.

    Args:
        Q: Normalized 61x61 GY94 rate matrix, e.g. from `build_gy94_rate_matrix`.
        parent_codon_idxs: Shape (num_sites,) sense-codon indices (0-60) of the
            observed parent state at each site.
        child_codon_idxs: Shape (num_sites,) sense-codon indices of the
            observed child state at each site.
        mask: Shape (num_sites,) boolean mask selecting valid (non-ambiguous)
            codon sites to include in the likelihood.
        branch_lengths: Shape (num_grid,) grid of candidate branch lengths,
            must be non-negative.

    Returns:
        Shape (num_grid,) tensor of summed log-likelihoods, one per grid point.

    Raises:
        ValueError: If branch_lengths contains negative values.
    """
    if (branch_lengths < 0).any():
        raise ValueError("branch_lengths must be non-negative")

    Q64 = Q.double()
    ts = branch_lengths.double()

    # Batched matrix exponential: (num_grid, 61, 61).
    P = torch.matrix_exp(Q64.unsqueeze(0) * ts.view(-1, 1, 1))

    # Transition probs for the observed (parent, child) pair at each site,
    # for every grid point: (num_grid, num_sites).
    site_probs = P[:, parent_codon_idxs, child_codon_idxs]
    log_probs = torch.log(site_probs.clamp_min(torch.finfo(torch.float64).tiny))
    log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))

    return log_probs.sum(dim=1)


@dataclass
class EdgeProfileClassification:
    """Classification of one edge's branch-length profile likelihood.

    Attributes:
        t_grid: Grid of branch lengths the profile was evaluated on.
        log_likelihoods: LL(t) at each grid point.
        t_star: argmax_t LL(t) over the grid.
        ll_at_t_star: LL(t_star).
        ll_near_zero: LL(GY94_OPTIMIZER_FAILURE_TAU0), the "near-zero" reference.
        phyz_refit_length: The externally-reported (phyz) refit branch length
            for this edge, used only by the optimizer-failure rule.
        label: See `classify_branch_length_profile` for the meaning of each
            `EdgeProfileLabel` value.
    """

    t_grid: torch.Tensor
    log_likelihoods: torch.Tensor
    t_star: float
    ll_at_t_star: float
    ll_near_zero: float
    phyz_refit_length: float
    label: EdgeProfileLabel


def default_gy94_profile_grid(
    t_min: float = 1e-6, t_max: float = 3.0, num_points: int = 400
) -> torch.Tensor:
    """Default log-spaced branch-length grid for profile-likelihood scoring.

    Args:
        t_min: Smallest branch length in the grid (must be positive).
        t_max: Largest branch length in the grid.
        num_points: Number of grid points.

    Returns:
        Shape (num_points,) log-spaced grid from t_min to t_max.
    """
    return torch.from_numpy(np.geomspace(t_min, t_max, num_points)).double()


def classify_branch_length_profile(
    Q: torch.Tensor,
    parent_codon_idxs: torch.Tensor,
    child_codon_idxs: torch.Tensor,
    mask: torch.Tensor,
    phyz_refit_length: float,
    t_grid: Optional[torch.Tensor] = None,
) -> EdgeProfileClassification:
    """Classify one edge as optimizer-failure / ridge / data-supported.

    Applies the pinned thresholds from issue #943:
        - optimizer-failure: argmax_t LL(t) <= tau0 (0.01) while phyz's
          reported refit is > 0.1 (i.e. phyz overshot a genuinely near-zero
          optimum).
        - ridge: LL(t*) - LL(t~=0) < Delta (2 log-likelihood units) — the
          peak is not meaningfully better than near-zero.
        - data-supported: peaked, LL(t*) - LL(t~=0) >= Delta and t* > 0.1.
        - ambiguous: none of the above pinned rules apply (e.g. t* in
          (tau0, 0.1] with LL(t*) - LL(t~=0) >= Delta — a gap the pinned
          thresholds do not cover).

    "LL(t~=0)" is evaluated at t = GY94_OPTIMIZER_FAILURE_TAU0 exactly (not at
    t = 0, which is degenerate: exp(Q*0) = I, so any edge with an actual
    substitution gets LL(0) = -inf). Using tau0 as the near-zero anchor keeps
    the "near zero" reference consistent with the optimizer-failure cutoff.

    Args:
        Q: Normalized 61x61 GY94 rate matrix.
        parent_codon_idxs: Shape (num_sites,) observed parent codon indices.
        child_codon_idxs: Shape (num_sites,) observed child codon indices.
        mask: Shape (num_sites,) boolean mask of valid codon sites.
        phyz_refit_length: phyz's reported `--reestimate-bl` branch length for
            this edge.
        t_grid: Optional branch-length grid; defaults to
            `default_gy94_profile_grid()`.

    Returns:
        EdgeProfileClassification with the full profile and the assigned label.
    """
    if t_grid is None:
        t_grid = default_gy94_profile_grid()

    log_likelihoods = gy94_log_likelihood_grid(
        Q, parent_codon_idxs, child_codon_idxs, mask, t_grid
    )
    argmax_idx = int(torch.argmax(log_likelihoods).item())
    t_star = float(t_grid[argmax_idx].item())
    ll_at_t_star = float(log_likelihoods[argmax_idx].item())

    tau0_tensor = torch.tensor([GY94_OPTIMIZER_FAILURE_TAU0], dtype=torch.float64)
    ll_near_zero = float(
        gy94_log_likelihood_grid(
            Q, parent_codon_idxs, child_codon_idxs, mask, tau0_tensor
        )[0].item()
    )

    ll_gain = ll_at_t_star - ll_near_zero
    label: EdgeProfileLabel
    if t_star <= GY94_OPTIMIZER_FAILURE_TAU0 and phyz_refit_length > 0.1:
        label = "optimizer_failure"
    elif ll_gain < GY94_RIDGE_DELTA_LL:
        label = "ridge"
    elif t_star > GY94_DATA_SUPPORTED_T_MIN:
        label = "data_supported"
    else:
        label = "ambiguous"

    return EdgeProfileClassification(
        t_grid=t_grid,
        log_likelihoods=log_likelihoods,
        t_star=t_star,
        ll_at_t_star=ll_at_t_star,
        ll_near_zero=ll_near_zero,
        phyz_refit_length=phyz_refit_length,
        label=label,
    )
