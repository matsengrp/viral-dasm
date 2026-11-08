"""Uniformization (randomization / Jensen's method) parent-row exp(tQ) kernel.

Issue #823 landed the kernel standalone, evaluated against a pre-registered
benchmark. Issue #838 wires it into ``_batched_matrix_exp_to_log_probs`` /
``QDASMLoss`` as ``matexp_method="uniformization"`` and adds per-matrix λ
(``per_matrix_lambda``) plus the two parent-row normalization conventions
(``_normalize_parent_row``) needed for a valid low-``N`` likelihood.

Pick a rate ``λ >= max_i |Q_ii|`` — either the FIXED ``UNIFORMIZATION_LAMBDA``
(a safety margin above the observed production max exit rate; legacy/scalar
path, kept for benchmarking) or the tighter per-matrix ``λ_i = max_k|Q_{i,kk}|``
(``per_matrix_lambda``, the #838 default: maximizes captured Poisson mass per
matrix at low ``n_terms``). Because ``Q``'s off-diagonals are nonnegative and
each row sums to zero, ``P̃ = I + Q/λ`` is nonnegative and row-stochastic for
any Q satisfying the λ bound, and

    exp(tQ) = sum_{n=0}^∞ Poisson(n; λt) · P̃^n

is a Poisson-weighted average of nonnegative row-stochastic matrices — every
partial sum (any truncation) is itself a genuine sub-distribution, so there is
no clamp and no negativity footgun. Truncating at a fixed ``n_terms`` leaves a
truncation error exactly equal to the Poisson right tail, computable in
advance via the regularized incomplete gamma function (the exact modern
equivalent of Fox & Glynn 1988's truncation-point calculation — see
``poisson_tail_prob``).

With the FIXED λ, ``n_terms`` can be chosen once for a whole batch (indeed
once for the whole deployment, from an assumed λt ceiling) rather than per
matrix, so the recurrence below has no per-matrix branching/masking — unlike
``_row_expmv``, which needs a data-dependent per-matrix sub-step count. With
per-matrix λ, the Poisson weights (not the recurrence structure) vary per
matrix — the shared ``n_terms`` still runs identically for every matrix in the
chunk; it's the caller's responsibility to size ``n_terms`` for the worst-case
``λ_i t_i`` in the chunk.

References:
    van Dijk, van Brummelen & Boucherie, "Uniformization: Basics, extensions
    and applications", Performance Evaluation 118 (2018).
    Fox & Glynn, "Computing Poisson probabilities", CACM 1988.
"""

import itertools
from typing import Literal, Optional, Union

import torch
from torch import Tensor

from dasm2.codon_table import (
    CODON64_TO_SENSE61_TENSOR,
    NUM_SENSE_CODONS,
    SENSE_CODON_LIST_TENSOR,
    STOP_CODON_ZAPPER,
    raise_if_stop_codon_parent,
)

# Fixed uniformization rate: a safety margin above the observed max CTMC exit
# rate on real QDASM Q. The issue's original estimate (λ≈32, from the
# 2026-05-06 negativity audit's ||Q||_∞ = 58.94 -> max|Q_ii| ≈ 29.47 on ONE
# captured batch) turned out not to be conservative enough: running the
# actual benchmark batch (production hit scales + the real global_13mer
# table, issue #823's own benchmark harness) found max exit rates from 27.5
# to 50.1 across just 10 random seeds. λ=64 keeps a >25% margin above that
# observed tail. ``_uniformization_parent_row`` raises if any matrix violates
# the bound rather than silently going negative — see the docstring above for
# why nonnegativity depends on it.
UNIFORMIZATION_LAMBDA = 64.0

# Assumed ceiling on the branch length t, used only to size the fixed,
# batch-independent truncation order below. This is a property of the DATA
# (production BL mix is ~88% <=0.1, ~12% in 0.1-0.5 — 2026-07-07 speed
# profile), independent of λ — unlike a synthetic generator swept to a target
# ||tQ||_∞ (whose own max exit rate scales with λ by construction), a real
# branch length does not shrink just because λ went up. An earlier version of
# this constant conflated the two and silently under-covered the real
# production BL range after λ was recalibrated from 32 to 64 (see below);
# expressing the ceiling directly in t avoids that mistake recurring.
ASSUMED_MAX_BRANCH_LENGTH = 0.6

# Assumed ceiling on λt used to size the fixed, batch-independent truncation
# order (see ``choose_truncation_n``): λ · ``ASSUMED_MAX_BRANCH_LENGTH``. This
# MUST be recomputed whenever λ changes (it no longer was, in an earlier
# version of this module, after λ was recalibrated 32 -> 64 below — that bug
# under-truncated at n_terms=50 for real BLs up to 0.5, producing a max abs
# log-prob error of ~9e-4 on the issue #823 benchmark batch instead of the
# targeted <=1e-4).
DEFAULT_LAM_T_MAX = UNIFORMIZATION_LAMBDA * ASSUMED_MAX_BRANCH_LENGTH

# Fox & Glynn (1988) target right-tail tolerance.
DEFAULT_TAIL_TOLERANCE = 1e-8


def poisson_tail_prob(n: int, lam_t: float) -> float:
    """``P(Poisson(lam_t) > n)``, exact via the regularized incomplete gamma function.

    ``torch.special.gammainc(a, x)`` is the regularized LOWER incomplete gamma
    function ``P(a, x)``; for integer ``n``,
    ``P(Poisson(lam_t) > n) == P(n + 1, lam_t)`` exactly (verified against
    ``scipy.stats.poisson.sf`` to double-precision agreement). This is the
    exact modern equivalent of the truncation-point calculation in Fox & Glynn
    (1988), which approximated the same Poisson tail via asymptotic,
    numerically-stabilized recursions because exact incomplete-gamma routines
    were not readily available at the time — double-precision ``lgamma`` makes
    the direct computation stable today.
    """
    return float(
        torch.special.gammainc(
            torch.tensor(float(n + 1), dtype=torch.float64),
            torch.tensor(float(lam_t), dtype=torch.float64),
        ).item()
    )


def choose_truncation_n(
    lam_t_max: float,
    tol: float = DEFAULT_TAIL_TOLERANCE,
    max_n: int = 2000,
) -> int:
    """Smallest ``N`` such that the ``Poisson(lam_t_max)`` right tail is ``<= tol``.

    A single ``N`` computed from the largest ``λt`` a batch (or deployment)
    can produce is safe for every matrix with smaller ``λt``: for fixed ``N``,
    the Poisson right tail ``P(X > N; λt)`` is monotonically increasing in
    ``λt`` (more mass pushed to large ``n``), so bounding the tail at the
    worst-case ``λt`` bounds it everywhere below that.
    """
    n = 0
    while n < max_n:
        if poisson_tail_prob(n, lam_t_max) <= tol:
            return n
        n += 1
    raise RuntimeError(
        f"Poisson tail did not fall below tol={tol} within max_n={max_n} terms "
        f"at lam_t_max={lam_t_max}; increase max_n or lower lam_t_max."
    )


# Default truncation order, fixed at import time from DEFAULT_LAM_T_MAX — a
# genuine constant (no data dependency), analogous to the fixed λ itself.
DEFAULT_N_TERMS = choose_truncation_n(DEFAULT_LAM_T_MAX, DEFAULT_TAIL_TOLERANCE)


def per_matrix_lambda(Q_chunk: Tensor) -> Tensor:
    """Per-matrix uniformization rate ``λ_i = max_k |Q_{i,kk}|`` — matrix ``i``'s
    own exit rate, the tightest λ keeping ``P̃_i`` nonnegative.

    A CTMC generator has ``Q_ii = -sum_{j!=i} Q_ij``, so ``|Q_ii| == exit_rate_i``;
    computed here from the off-diagonal sum (matching
    ``_uniformization_parent_row``'s convention) rather than trusting
    ``Q_chunk``'s own diagonal, for the same floating-point reason documented
    there. Using this as ``lam`` (vs. the fixed global ``UNIFORMIZATION_LAMBDA``)
    maximizes captured Poisson mass per matrix at low ``n_terms`` (issue #838).

    Args:
        Q_chunk: Rate matrices ``[N, 61, 61]`` (unscaled).

    Returns:
        Per-matrix λ ``[N]``.
    """
    diag_mask = torch.eye(NUM_SENSE_CODONS, device=Q_chunk.device, dtype=torch.bool)
    exit_rate = Q_chunk.masked_fill(diag_mask, 0.0).sum(dim=-1)  # [N, 61]
    return exit_rate.amax(dim=-1)  # [N]


def _normalize_parent_row(
    row: Tensor,
    parent_row_idxs: Tensor,
    normalization: Optional[Literal["parent_absorbing", "proportional"]],
) -> Tensor:
    """Renormalize a truncated uniformization parent row to a valid distribution.

    Both conventions are exact no-ops at ``n_terms -> ∞`` (the untruncated row
    already sums to 1), so both converge to the same ``exp(tQ)`` reference —
    they differ only in where they put the truncation error at finite ``N``.

    * ``"parent_absorbing"`` (classical DASM's convention): leave every child
      entry as computed; overwrite the parent (stay) entry with
      ``1 - sum(children)``. Puts all truncation error on the stay state — at
      ``n_terms=1`` this floors the stay probability at ``~0.63`` regardless of
      ``t`` (see ``TestParentAbsorbingStayFloor``).
    * ``"proportional"``: divide the whole row by its (sub-unity, truncated)
      sum. Spreads the truncation error proportionally across every entry;
      no stay-floor.

    Args:
        row: Raw truncated parent rows ``[N, 61]`` (nonnegative, sums to
            ``<= 1``).
        parent_row_idxs: Parent codon indices in 61-space ``[N]``.
        normalization: Which convention to apply, or ``None`` to leave the
            raw truncated row untouched (the legacy/default behavior — an
            explicit opt-in is required to normalize, per issue #838).

    Returns:
        Normalized parent rows ``[N, 61]`` summing to 1 (to float precision),
        or the untouched input if ``normalization is None``.
    """
    if normalization is None:
        return row
    arange_n = torch.arange(row.shape[0], device=row.device)
    if normalization == "proportional":
        row_sum = row.sum(dim=-1, keepdim=True)
        # Guard the degenerate case where every Poisson weight up to n_terms
        # has underflowed toward zero — a very hot Q (large λ) on a long
        # branch (large t) pushes essentially all Poisson(λt) mass past a low
        # n_terms, e.g. e^{-λt} itself underflows float32 for λt gtrsim 90.
        #
        # The threshold is a FIXED absolute epsilon (1e-20), not
        # ``torch.finfo(dtype).tiny`` (~1.18e-38 for float32): a row_sum
        # anywhere below ~1e-20 makes ``1/row_sum`` a >=1e20-scale local
        # gradient, and a real batched call mixes many (b,l) positions with
        # wildly different λt in one op — plenty of headroom below float32's
        # ~3.4e38 max for that local gradient, times a same-order-tiny
        # upstream gradient, times accumulation over 60 other entries in the
        # same row, to overflow to inf and produce NaN a few ops later. Using
        # ``tiny`` as the threshold (an earlier version of this fix) still hit
        # this on a real production batch — the failure mode is precision
        # *and* dynamic-range headroom, not just "exactly zero". Fall back to
        # "stayed put" (e_p) below the threshold — the same degenerate answer
        # ``parent_absorbing`` gives here (children_sum=0 -> stay=1): an
        # honest signal that n_terms was far too low for this branch length,
        # not a silently wrong (and non-row-stochastic) distribution.
        degenerate_row_sum_threshold = 1e-20
        degenerate = row_sum.squeeze(-1) < degenerate_row_sum_threshold
        if degenerate.any():
            # Route degenerate rows' denominator to a benign constant (1.0)
            # BEFORE dividing, rather than dividing by a tiny value and
            # masking the result afterward: never ask autograd to form an
            # extreme-magnitude local jacobian at all, removing the failure
            # mode regardless of the exact backward-accumulation mechanism.
            safe_row_sum = torch.where(
                degenerate[:, None], torch.ones_like(row_sum), row_sum
            )
            fallback = torch.zeros_like(row)
            fallback[arange_n, parent_row_idxs] = 1.0
            return torch.where(degenerate[:, None], fallback, row / safe_row_sum)
        return row / row_sum
    if normalization == "parent_absorbing":
        children_sum = row.sum(dim=-1) - row[arange_n, parent_row_idxs]
        row = row.clone()
        row[arange_n, parent_row_idxs] = 1.0 - children_sum
        return row
    raise ValueError(f"Unknown normalization convention: {normalization!r}")


def _uniformization_weights(lam_t: Tensor, n_terms: int) -> Tensor:
    """Poisson(n; λt) weights for ``n = 0..n_terms``, vectorized over the batch.

    Returns ``[*lam_t.shape, n_terms + 1]``. Computed in log-space (stable for
    any λt magnitude since ``lgamma`` has no dynamic-range limit):
    ``log w_n = n·log(λt) - λt - lgamma(n+1)``.

    ``λt = 0`` (BL = 0) is handled by clamping away from exactly zero before
    the log: the ``n=0`` term's coefficient is then ``0 * log(eps)``, a
    finite ``0 * (large negative)`` product that evaluates to ``0`` rather
    than the ``0 * -inf = nan`` that occurs at literal ``λt = 0``, and the
    weight still comes out as ``exp(0 - eps - 0) ≈ 1`` (correct: ``P(Poisson(0)
    = 0) = 1``).
    """
    dtype = lam_t.dtype
    device = lam_t.device
    n = torch.arange(n_terms + 1, device=device, dtype=dtype)  # [n_terms+1]
    log_lam_t = torch.log(lam_t.clamp(min=1e-30))  # [*]
    lgamma_np1 = torch.lgamma(n + 1)  # [n_terms+1]
    log_w = n * log_lam_t[..., None] - lam_t[..., None] - lgamma_np1
    return torch.exp(log_w)  # [*, n_terms+1]


def _uniformization_parent_row(
    Q_chunk: Tensor,
    t_chunk: Tensor,
    parent_row_idxs: Tensor,
    n_terms: int = DEFAULT_N_TERMS,
    lam: Union[float, Tensor] = UNIFORMIZATION_LAMBDA,
    normalization: Optional[Literal["parent_absorbing", "proportional"]] = None,
) -> Tensor:
    """Parent row of ``exp(tQ)`` via uniformization, without forming ``exp(tQ)``.

    Computes ``e_p^T sum_{n=0}^{n_terms} Poisson(n; λt) P̃^n`` for each matrix
    in the chunk, where ``P̃ = I + Q/λ``. Unlike ``_row_expmv`` (whose Taylor
    sub-step count is data-dependent), ``n_terms`` is a fixed hyperparameter,
    so every matrix in the chunk runs the identical ``n_terms``-step
    vector-matrix recurrence; only the Poisson weights vary per matrix.

    Args:
        Q_chunk: Rate matrices ``[N, 61, 61]`` (UNSCALED — ``t`` is separate,
            unlike the ``tQ_chunk`` convention in ``matrix_exp.py``, since
            uniformization needs ``Q``'s diagonal directly to validate the λ
            bound and to form ``P̃``).
        t_chunk: Branch lengths ``[N]``.
        parent_row_idxs: Parent codon indices in 61-space ``[N]``.
        n_terms: Number of uniformization terms (fixed truncation order).
        lam: Uniformization rate; either a scalar (shared across the chunk,
            the legacy fixed-``UNIFORMIZATION_LAMBDA`` path) or a per-matrix
            tensor ``[N]`` (e.g. from ``per_matrix_lambda``). Must satisfy
            ``lam_i >= max_k |Q_{i,kk}|`` for every matrix in the chunk
            (checked below).
        normalization: Renormalization convention for the truncated row, or
            ``None`` (default) to leave it as raw truncation — see
            ``_normalize_parent_row``.

    Returns:
        Parent rows ``[N, 61]`` of ``exp(tQ)``.
    """
    n_mat = Q_chunk.shape[0]
    device = Q_chunk.device
    dtype = Q_chunk.dtype

    # Recompute the diagonal from the off-diagonal entries (Q_ii := -sum_{j!=i}
    # Q_ij) rather than trusting Q_chunk's own diagonal. A CTMC generator
    # satisfies this by definition, but the Q-builders only hold it to
    # floating precision (~1e-7 in float32; see _parent_row_q.py's row-sum
    # comment): applying P̃ to a row vector n_terms (~50) times compounds that
    # per-matrix residual roughly linearly in n_terms, which otherwise shows
    # up as a real drift away from row-stochasticity (observed ~1e-7 total
    # rowsum error at n_terms=50 in testing). Deriving the diagonal here makes
    # P̃ exactly row-stochastic up to a single division's rounding, regardless
    # of Q_chunk's residual.
    diag_mask = torch.eye(NUM_SENSE_CODONS, device=device, dtype=torch.bool)
    Q_offdiag = Q_chunk.masked_fill(diag_mask, 0.0)  # [N, 61, 61]
    exit_rate = Q_offdiag.sum(dim=-1)  # [N, 61]
    # Same formula as per_matrix_lambda(Q_chunk), inlined rather than called:
    # Q_offdiag/exit_rate are needed again below for P_tilde, so computing
    # them a second time inside a per_matrix_lambda call would be wasted work.
    max_exit_rate_per_matrix = exit_rate.amax(dim=-1)  # [N]

    lam_per_matrix = (
        lam
        if isinstance(lam, Tensor)
        else torch.full((n_mat,), float(lam), device=device, dtype=dtype)
    )
    # Floor away from exactly 0: a matrix with zero exit rate everywhere
    # (Q identically 0 — e.g. a masked/absorbing position) has per-matrix
    # λ_i = 0 by construction, and P̃ = I + Q/λ would divide 0-valued
    # Q_offdiag by that same 0 (NaN). Q_offdiag is 0 there regardless of the
    # floor's exact value, so this only rescues the 0/0 case and does not
    # change P̃ for any matrix with a genuine positive exit rate.
    lam_per_matrix = lam_per_matrix.clamp(min=1e-12)

    # The nonnegativity/row-stochasticity guarantee is contingent on this
    # bound (see module docstring); a violation is a real configuration bug
    # (λ set too low, or a Q-matrix construction producing an unexpectedly
    # hot exit rate), not float noise — raise loudly rather than silently
    # emitting a negative P̃ diagonal entry.
    violation = max_exit_rate_per_matrix > lam_per_matrix
    if violation.any():
        num_violations = int(violation.sum().item())
        raise RuntimeError(
            f"Uniformization requires lam >= max exit rate per matrix; "
            f"{num_violations}/{n_mat} matrices violate this. Worst case: exit rate = "
            f"{max_exit_rate_per_matrix[violation].amax().item():.3f} > "
            f"lam = {lam_per_matrix[violation][max_exit_rate_per_matrix[violation].argmax()].item():.3f}. "
            f"Increase lam (or use per_matrix_lambda) or investigate the "
            f"Q-matrix construction."
        )

    eye = torch.eye(NUM_SENSE_CODONS, device=device, dtype=dtype)
    lam_bcast = lam_per_matrix[:, None, None]
    P_tilde = eye + Q_offdiag / lam_bcast - torch.diag_embed(exit_rate) / lam_bcast

    lam_t = lam_per_matrix * t_chunk  # [N]
    weights = _uniformization_weights(lam_t, n_terms)  # [N, n_terms+1]

    arange_n = torch.arange(n_mat, device=device)
    w = torch.zeros(n_mat, 1, NUM_SENSE_CODONS, device=device, dtype=dtype)
    w[arange_n, 0, parent_row_idxs] = 1.0  # e_p as a row vector

    row_accum = weights[:, 0, None] * w.squeeze(1)  # n=0 term: weight * e_p
    for step in range(1, n_terms + 1):
        w = torch.bmm(w, P_tilde)  # advance to P̃^step applied to e_p
        row_accum = row_accum + weights[:, step, None] * w.squeeze(1)
    return _normalize_parent_row(row_accum, parent_row_idxs, normalization)


def _flatten_uniformization_inputs(
    Q_all: Tensor, parent_codon_idxs: Tensor, branch_lengths: Tensor
) -> tuple[Tensor, Tensor, Tensor, int, int]:
    """Shared ``[B, L, ...]`` -> ``[B*L, ...]`` flattening for both
    ``uniformization_log_probs`` and ``bucketed_uniformization_log_probs``:
    scale-independent Q/t flattening plus the 64-space -> 61-space parent
    index conversion.

    Returns:
        ``(Q_flat, t_flat, parent_61_flat, B, L)``.
    """
    device = Q_all.device
    B, L = Q_all.shape[:2]
    total = B * L

    Q_flat = Q_all.reshape(total, NUM_SENSE_CODONS, NUM_SENSE_CODONS)
    t_flat = branch_lengths[:, None].expand(B, L).reshape(total)

    parent_61_idxs = CODON64_TO_SENSE61_TENSOR.to(device)[
        parent_codon_idxs.clamp(max=63)
    ].clamp(min=0, max=NUM_SENSE_CODONS - 1)  # [B, L]
    parent_61_flat = parent_61_idxs.reshape(total)

    return Q_flat, t_flat, parent_61_flat, B, L


def _chunked_uniformization_parent_rows(
    Q_flat: Tensor,
    t_flat: Tensor,
    parent_61_flat: Tensor,
    idx: Tensor,
    n_terms: int,
    lam: Union[float, Tensor],
    normalization: Optional[Literal["parent_absorbing", "proportional"]],
    max_matrices_per_chunk: int,
) -> Tensor:
    """Runs ``_uniformization_parent_row`` over ``idx`` (indices into the
    flattened ``B*L`` matrices — the full range for ``uniformization_log_probs``,
    one bucket's subset for ``bucketed_uniformization_log_probs``), chunked by
    ``max_matrices_per_chunk`` to bound peak memory.

    ``idx`` being ``torch.arange(total)`` reproduces
    ``uniformization_log_probs``'s original chunking exactly (same order, same
    chunk boundaries), which is what makes
    ``bucketed_uniformization_log_probs(n_buckets=1)`` match it bit-for-bit.

    Returns:
        Parent rows ``[idx.numel(), 61]``, in ``idx``'s order.
    """
    parts = []
    for start in range(0, idx.numel(), max_matrices_per_chunk):
        sub = idx[start : start + max_matrices_per_chunk]
        lam_chunk = lam[sub] if isinstance(lam, Tensor) else lam
        parts.append(
            _uniformization_parent_row(
                Q_flat[sub],
                t_flat[sub],
                parent_61_flat[sub],
                n_terms,
                lam_chunk,
                normalization,
            )
        )
    return torch.cat(parts, dim=0)


def uniformization_log_probs(
    Q_all: Tensor,
    parent_codon_idxs: Tensor,
    branch_lengths: Tensor,
    L_output: int,
    n_terms: int = DEFAULT_N_TERMS,
    lam: Union[float, Tensor] = UNIFORMIZATION_LAMBDA,
    normalization: Optional[Literal["parent_absorbing", "proportional"]] = None,
    max_matrices_per_chunk: int = 10000,
    apply_stop_zapper: bool = True,
) -> Tensor:
    """Batched uniformization driver mirroring ``_batched_matrix_exp_to_log_probs``.

    Same I/O contract (``Q_all``/``parent_codon_idxs``/``branch_lengths`` in,
    ``[B, L_output, 64]`` log-probs out, same NaN/negativity checks, clamp,
    stop-zapper) as the shipped ``adaptive=True`` path, so it is a drop-in
    comparison point for benchmarking/validation. Wired into
    ``_batched_matrix_exp_to_log_probs`` / ``QDASMLoss`` as
    ``matexp_method="uniformization"`` (issue #838), with a single global
    ``n_terms`` for the whole batch — see ``bucketed_uniformization_log_probs``
    below for the λt-adaptive-per-bucket driver (issue #844).

    Args:
        Q_all: Rate matrices ``[B, L, 61, 61]``.
        parent_codon_idxs: Parent codon indices in 64-space ``[B, L]``.
        branch_lengths: Per-sequence branch lengths ``[B]``.
        L_output: Output sequence length (may be > L for padded AA positions).
        n_terms: Uniformization truncation order (fixed; see ``DEFAULT_N_TERMS``).
        lam: Uniformization rate — a scalar (legacy fixed-``UNIFORMIZATION_LAMBDA``
            path) or a per-matrix tensor ``[B*L]`` (e.g. from
            ``per_matrix_lambda(Q_flat)``).
        normalization: Renormalization convention for the truncated parent
            row, or ``None`` (default) to leave it as raw truncation — see
            ``_normalize_parent_row``. ``matrix_exp.py``'s
            ``matexp_method="uniformization"`` dispatch always passes an
            explicit convention (never ``None``); ``None`` is for direct/
            benchmark callers of this function only.
        max_matrices_per_chunk: Chunk size bounding peak memory.
        apply_stop_zapper: Whether to apply ``STOP_CODON_ZAPPER`` to the output.

    Returns:
        Log codon transition probabilities ``[B, L_output, 64]``.
    """
    raise_if_stop_codon_parent(parent_codon_idxs, "uniformization_log_probs")

    Q_flat, t_flat, parent_61_flat, B, L = _flatten_uniformization_inputs(
        Q_all, parent_codon_idxs, branch_lengths
    )
    total = B * L
    idx = torch.arange(total, device=Q_all.device)

    parent_rows = _chunked_uniformization_parent_rows(
        Q_flat,
        t_flat,
        parent_61_flat,
        idx,
        n_terms,
        lam,
        normalization,
        max_matrices_per_chunk,
    ).reshape(B, L, NUM_SENSE_CODONS)
    return _finalize_uniformization_parent_rows(
        parent_rows, Q_all, branch_lengths, L_output, apply_stop_zapper
    )


def _finalize_uniformization_parent_rows(
    parent_rows: Tensor,
    Q_all: Tensor,
    branch_lengths: Tensor,
    L_output: int,
    apply_stop_zapper: bool,
) -> Tensor:
    """Shared NaN/negativity checks, clamp, 61->64 embed, log, stop-zapper.

    Factored out of ``uniformization_log_probs`` so
    ``bucketed_uniformization_log_probs`` shares the identical postprocessing
    (and, at ``n_buckets=1``, identical floating-point output) rather than a
    parallel copy that could drift.

    Args:
        parent_rows: Raw truncated/normalized parent rows ``[B, L, 61]``.
        Q_all: Rate matrices ``[B, L, 61, 61]`` (only used for error messages).
        branch_lengths: Per-sequence branch lengths ``[B]`` (only used for
            error messages).
        L_output: Output sequence length (may be > L for padded AA positions).
        apply_stop_zapper: Whether to apply ``STOP_CODON_ZAPPER`` to the output.

    Returns:
        Log codon transition probabilities ``[B, L_output, 64]``.
    """
    device = parent_rows.device
    B, L = parent_rows.shape[:2]

    if torch.isnan(parent_rows).any():
        nan_count = int(torch.isnan(parent_rows).any(dim=-1).sum().item())
        raise RuntimeError(
            f"Parent-row uniformization exp(tQ) contains NaN entries: "
            f"{nan_count} affected (B,L) sites, "
            f"max |Q| = {float(Q_all.abs().max().item()):.3e}, "
            f"max BL = {float(branch_lengths.max().item()):.3e}"
        )
    # Tighter than the -1e-3 threshold in matrix_exp.py: uniformization is
    # provably nonnegative given the lam bound checked in
    # _uniformization_parent_row, so any entry below fp32 noise is a red flag.
    if not (parent_rows >= -1e-6).all():
        raise RuntimeError(
            f"Parent-row uniformization exp(tQ) has negative entries: "
            f"min = {float(parent_rows.min().item()):.2e}; "
            f"max |Q| = {float(Q_all.abs().max().item()):.3e}, "
            f"max BL = {float(branch_lengths.max().item()):.3e}. "
            f"This indicates a numerical bug, not float32 cancellation."
        )
    parent_rows = parent_rows.clamp(min=1e-10)

    sense_codon_list = SENSE_CODON_LIST_TENSOR.to(device)
    result_64 = torch.zeros(B, L_output, 64, device=device, dtype=parent_rows.dtype)
    result_64[:, :L, sense_codon_list] = parent_rows

    result_64 = torch.log(result_64.clamp(min=1e-30))
    if apply_stop_zapper:
        result_64 = result_64 + STOP_CODON_ZAPPER.to(device)[None, None, :]

    return result_64


def bucketed_uniformization_log_probs(
    Q_all: Tensor,
    parent_codon_idxs: Tensor,
    branch_lengths: Tensor,
    L_output: int,
    n_buckets: int = 1,
    tol: float = DEFAULT_TAIL_TOLERANCE,
    lam: Optional[Union[float, Tensor]] = None,
    normalization: Optional[
        Literal["parent_absorbing", "proportional"]
    ] = "proportional",
    max_matrices_per_chunk: int = 10000,
    apply_stop_zapper: bool = True,
    max_n: int = 2000,
) -> Tensor:
    """λt-bucketing driver: per-bucket adaptive ``N`` over the ``B*L`` matrices.

    Issue #844's compute-scheduling wrapper over the chunked
    ``_uniformization_parent_row`` recurrence. ``uniformization_log_probs``
    takes one ``n_terms`` for the whole batch (every matrix runs the
    longest-branch matrix's term count); this instead computes per-matrix
    ``λt`` (default: ``per_matrix_lambda(Q).detach() * t``), partitions the
    ``B*L`` matrices into ``n_buckets`` equal-count groups by ``λt`` (quantile
    edges), and runs each group through the recurrence at its own
    ``n_terms = choose_truncation_n(bucket_max λt, tol)`` — so a heterogeneous
    batch (e.g. Compara's long-branch mix) doesn't force every short-branch
    matrix to pay for the longest branch's term count.

    Correctness never depends on ``n_buckets`` or the bucket assignment: each
    matrix's own Poisson weights (not the recurrence step count) do the
    attenuating, so running any bucket to ``N >= that matrix's own need``
    truncates it correctly regardless of what else shares the bucket (see
    module docstring and ``uniformization_log_probs``). ``n_buckets=1`` is the
    global-``N_max`` baseline — literally ``uniformization_log_probs`` with
    ``n_terms`` computed fresh from this batch's own worst-case ``λt`` instead
    of a caller-supplied constant — and, because it walks the ``B*L`` matrices
    in the same order with the same chunk size, its output matches
    ``uniformization_log_probs`` bit-for-bit when passed that same computed
    ``n_terms``.

    Args:
        Q_all: Rate matrices ``[B, L, 61, 61]``.
        parent_codon_idxs: Parent codon indices in 64-space ``[B, L]``.
        branch_lengths: Per-sequence branch lengths ``[B]``.
        L_output: Output sequence length (may be > L for padded AA positions).
        n_buckets: Number of λt quantile buckets (``K`` in the issue). ``1``
            is the per-batch global-``N_max`` baseline.
        tol: Poisson right-tail tolerance passed to ``choose_truncation_n``.
        lam: Uniformization rate — a scalar, a per-matrix tensor ``[B*L]``, or
            ``None`` (default) to compute ``per_matrix_lambda(Q_all).detach()``
            internally (the issue's proposed default source of λt
            heterogeneity).
        normalization: Renormalization convention for the truncated parent
            row — ``"proportional"`` (default; the #841 winner) or
            ``"parent_absorbing"``, or ``None`` for raw truncation.
        max_matrices_per_chunk: Chunk size within each bucket, bounding peak
            memory.
        apply_stop_zapper: Whether to apply ``STOP_CODON_ZAPPER`` to the output.
        max_n: Passed to ``choose_truncation_n``'s search ceiling.

    Returns:
        Log codon transition probabilities ``[B, L_output, 64]``.
    """
    raise_if_stop_codon_parent(parent_codon_idxs, "bucketed_uniformization_log_probs")
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")

    device = Q_all.device
    dtype = Q_all.dtype
    Q_flat, t_flat, parent_61_flat, B, L = _flatten_uniformization_inputs(
        Q_all, parent_codon_idxs, branch_lengths
    )
    total = B * L

    if lam is None:
        lam_flat = per_matrix_lambda(Q_flat).detach()
    elif isinstance(lam, Tensor):
        lam_flat = lam.reshape(total)
    else:
        lam_flat = torch.full((total,), float(lam), device=device, dtype=dtype)
    lam_t = lam_flat * t_flat  # [total]

    if n_buckets == 1:
        # Original index order (not argsort) so this exactly replays
        # uniformization_log_probs's own chunking loop -- the bit-for-bit
        # regression guard depends on this.
        bucket_order = torch.arange(total, device=device)
        bucket_sizes = [total]
    else:
        bucket_order = torch.argsort(lam_t)
        base, remainder = divmod(total, n_buckets)
        bucket_sizes = [base + 1 if i < remainder else base for i in range(n_buckets)]
    bucket_boundaries = list(itertools.accumulate(bucket_sizes))

    parent_rows = torch.zeros(total, NUM_SENSE_CODONS, device=device, dtype=dtype)
    start = 0
    for end in bucket_boundaries:
        idx = bucket_order[start:end]
        start = end
        if idx.numel() == 0:
            continue
        bucket_max_lam_t = float(lam_t[idx].max().item())
        n_terms_b = choose_truncation_n(bucket_max_lam_t, tol, max_n=max_n)
        bucket_rows = _chunked_uniformization_parent_rows(
            Q_flat,
            t_flat,
            parent_61_flat,
            idx,
            n_terms_b,
            lam_flat,
            normalization,
            max_matrices_per_chunk,
        )
        parent_rows = parent_rows.index_copy(0, idx, bucket_rows)

    parent_rows = parent_rows.reshape(B, L, NUM_SENSE_CODONS)
    return _finalize_uniformization_parent_rows(
        parent_rows, Q_all, branch_lengths, L_output, apply_stop_zapper
    )
