"""Matrix exponential utilities for QDASM/Hybrid.

Exact-with-autograd, scaling-and-squaring quintic, cubic-Taylor, and a
parent-row ``expmv`` variant, plus the batched chunked
``exp(tQ) → log-probability`` driver used by both heads.

The batched driver's ``adaptive`` path computes only the *parent row* of
``exp(tQ)`` — the sole quantity the QDASM likelihood, ``predict_codons``
argmax, and branch-length optimization consume — via the action of the
exponential (``_row_expmv``) for the ``||tQ||_∞`` bulk and full
scaling-and-squaring for the hot tail, never materializing the full 61×61 P.
See issue #793 (WI 1) and ``experiments/2026-07-07-qdasm-speed-profile/``.

Extracted from ``mutsel.py``.
"""

import logging
import warnings
from typing import Literal, cast

import torch
from torch import Tensor

from dasm2.codon_table import (
    CODON64_TO_SENSE61_TENSOR,
    NUM_SENSE_CODONS,
    SENSE_CODON_LIST_TENSOR,
    STOP_CODON_ZAPPER,
    raise_if_stop_codon_parent,
)
from dasm2.molevol.uniformization import (
    DEFAULT_N_TERMS as _UNIFORMIZATION_DEFAULT_N_TERMS,
)
from dasm2.molevol.uniformization import (
    per_matrix_lambda,
    uniformization_log_probs,
)

logger = logging.getLogger(__name__)

# Parent-row expmv (issue #793 WI 1). The training/BL-opt likelihood consumes
# only the parent row of exp(tQ), so the adaptive path computes that row
# directly via the action of the exponential (Al-Mohy & Higham 2011) instead of
# forming the full 61×61 P. A degree-6 Taylor of exp(tQ/s) applied to the row
# vector, with s sized so ``||tQ/s||_∞ <= 0.35``, keeps the per-sub-step
# truncation error (~``||tQ/s||^7 / 7!``) at the fp32 noise floor. The
# 2026-07-07 sweep measured max abs row error ~1e-9 (f64) / ~1e-6 (f32) and
# gradient error ~5e-7 vs the Fréchet derivative for ``||tQ||_∞`` up to ~2.
_ROW_EXPMV_TAYLOR_DEGREE = 6
_ROW_EXPMV_TARGET_NORM = 0.35


def _cubic_taylor_chunk(tQ_chunk: Tensor, eye: Tensor) -> Tensor:
    """Cubic Taylor approximation P ≈ I + tQ + (tQ)²/2 + (tQ)³/6."""
    tQ2 = tQ_chunk @ tQ_chunk
    return eye + tQ_chunk + 0.5 * tQ2 + (1.0 / 6.0) * (tQ2 @ tQ_chunk)


def _exact_matrix_exp_chunk(tQ_chunk: Tensor) -> Tensor:
    """Exact matrix exponential with full autograd (Fréchet derivative backward).

    Runs in the input dtype (f32 in training): max abs err vs f64 is <1e-6
    on QDASM-shaped 61×61 Q matrices, well inside the 1e-3 sanity threshold,
    and ~6x faster fwd+bwd than the f64-cast path.
    """
    return cast(Tensor, torch.linalg.matrix_exp(tQ_chunk))


def _scaling_squaring_quintic_chunk(
    tQ_chunk: Tensor, target_norm: float = 0.5
) -> Tensor:
    """Hand-rolled scaling-and-squaring of degree-5 Taylor.

    Picks per-matrix ``k`` such that ``||tQ / 2^k||_∞ < target_norm``, evaluates
    a quintic Taylor series on the scaled matrix, then squares ``k_max`` times
    using ``torch.where`` to gate which matrices still need squaring.

    Backward is plain matmul chain-rule (no Fréchet-derivative intermediate),
    which trims peak fwd+bwd memory ~32 % vs ``torch.linalg.matrix_exp`` while
    matching its accuracy on QDASM-shaped 61×61 generator-form Q matrices
    (max abs err ~1e-6 vs the f64 reference; benchmark in
    experiments/2026-05-05-matrix-exp-bench/results/bench_results.json).

    One CPU sync per call (``int(k.max())``); the squaring loop dispatches GPU
    kernels but does no further syncs.
    """
    # Row-sum (∞-) norm per matrix: [N]
    norm = tQ_chunk.abs().sum(dim=-1).max(dim=-1).values
    # k = ceil(log2(norm / target_norm)), floored at 0.
    k = torch.clamp(torch.ceil(torch.log2(norm / target_norm + 1e-12)), min=0).long()
    k_max = int(k.max().item())

    scale = (2.0 ** k.to(tQ_chunk.dtype))[:, None, None]
    tQ_s = tQ_chunk / scale

    # Quintic Taylor: I + tQ + (tQ)^2/2 + ... + (tQ)^5/120
    eye = torch.eye(tQ_s.shape[-1], device=tQ_s.device, dtype=tQ_s.dtype)
    tQ2 = tQ_s @ tQ_s
    tQ3 = tQ2 @ tQ_s
    tQ4 = tQ3 @ tQ_s
    tQ5 = tQ4 @ tQ_s
    P: torch.Tensor = (
        eye
        + tQ_s
        + 0.5 * tQ2
        + (1.0 / 6.0) * tQ3
        + (1.0 / 24.0) * tQ4
        + (1.0 / 120.0) * tQ5
    )

    if k_max == 0:
        return P

    for i in range(k_max):
        needs_more = (k > i)[:, None, None]
        P = torch.where(needs_more, P @ P, P)
    return P


def _row_expmv(
    tQ_chunk: Tensor,
    parent_row_idxs: Tensor,
    taylor_degree: int = _ROW_EXPMV_TAYLOR_DEGREE,
    target_norm: float = _ROW_EXPMV_TARGET_NORM,
) -> Tensor:
    """Parent row of ``exp(tQ)`` via the action of the exponential (no full P).

    Computes ``e_p^T exp(tQ)`` for each matrix in the chunk without forming
    ``exp(tQ)``. Uses ``exp(tQ) = exp(tQ/s)^s``: applies a degree-
    ``taylor_degree`` Taylor of ``exp(tQ/s)`` to the unit row vector ``e_p``,
    ``s`` times, with ``s = ceil(||tQ||_∞ / target_norm)`` chosen per matrix so
    every sub-step stays inside the Taylor's accurate envelope. Each sub-step
    multiplies by the full ``tQ``, so all multi-step CTMC paths are summed — the
    result equals the parent row of the full ``exp(tQ)`` to fp32 noise over the
    ``||tQ||_∞`` range the adaptive router sends here (at/below ``tq_threshold``,
    default 1.0, where ``s <= 3``).

    The backward is the vector-matrix product's VJP — a rank-1 outer product per
    sub-step, not the 61×61 Fréchet solve the full ``matrix_exp`` backward needs
    — so it is both correct and cheaper (verified to ~5e-7 vs the Fréchet
    derivative in the 2026-07-07 sweep).

    Args:
        tQ_chunk: Scaled rate matrices ``[N, 61, 61]`` (already ``t * Q``).
        parent_row_idxs: Parent codon indices in 61-space ``[N]`` — the row of
            ``exp(tQ)`` to return for each matrix.
        taylor_degree: Taylor degree per sub-step (default 6).
        target_norm: Per-sub-step ``||tQ/s||_∞`` ceiling (default 0.35).

    Returns:
        Parent rows ``[N, 61]`` of ``exp(tQ)``.
    """
    n = tQ_chunk.shape[0]
    arange_n = torch.arange(n, device=tQ_chunk.device)
    norm = tQ_chunk.abs().sum(dim=-1).amax(dim=-1)  # [N] row-sum inf-norm
    s = torch.clamp(torch.ceil(norm / target_norm), min=1.0).long()  # [N] sub-steps
    s_max = int(s.max().item())
    # ``.long()`` detaches s from autograd, so exp(tQ/s) treats the integer
    # step-count as a constant scale — the scaling-and-squaring convention.
    A = tQ_chunk / s.to(tQ_chunk.dtype)[:, None, None]  # ||A||_∞ <= target_norm
    still_active = s[:, None, None]  # [N,1,1]; gate matrices needing more sub-steps
    # Unit row vector e_p as [N,1,61] so ``bmm(w, A)`` is a row·matrix product.
    w = torch.zeros(n, 1, 61, device=tQ_chunk.device, dtype=tQ_chunk.dtype)
    w[arange_n, 0, parent_row_idxs] = 1.0
    for step in range(s_max):
        active = step < still_active  # [N,1,1] bool
        term = w
        nxt = w
        for k in range(1, taylor_degree + 1):  # nxt = Taylor(exp(A)) applied to w
            term = torch.bmm(term, A) / k
            nxt = nxt + term
        w = torch.where(active, nxt, w)
    return w.squeeze(1)  # [N, 61]


def _ss_quintic_parent_row(tQ_chunk: Tensor, parent_row_idxs: Tensor) -> Tensor:
    """Parent rows of ``exp(tQ)`` via full scaling-squaring quintic, then extract.

    The hot-tail companion to ``_row_expmv``: for ``||tQ||_∞ > tq_threshold``
    the row-only expmv would need many sequential sub-steps and lose to parallel
    squaring, so this forms the full ``[N, 61, 61]`` P for the (small) tail
    subset and reads off the parent row.
    """
    P = _scaling_squaring_quintic_chunk(tQ_chunk)  # [N, 61, 61]
    n = tQ_chunk.shape[0]
    return P[torch.arange(n, device=tQ_chunk.device), parent_row_idxs]  # [N, 61]


def _matrix_exp_parent_row(tQ_chunk: Tensor, parent_row_idxs: Tensor) -> Tensor:
    """Parent rows via ``torch.linalg.matrix_exp`` (Fréchet backward), then extract.

    The ``long_path="matrix_exp"`` variant of ``_ss_quintic_parent_row``.
    """
    P = _exact_matrix_exp_chunk(tQ_chunk)  # [N, 61, 61]
    n = tQ_chunk.shape[0]
    return P[torch.arange(n, device=tQ_chunk.device), parent_row_idxs]  # [N, 61]


def _adaptive_parent_row_exp(
    tQ_flat: Tensor,
    parent_row_idxs: Tensor,
    total: int,
    device: torch.device,
    tq_threshold: float,
    max_matrices_per_chunk: int,
    long_path: Literal["matrix_exp", "ss_quintic"] = "ss_quintic",
) -> Tensor:
    """Parent-row hybrid matrix exponential (issue #793 WI 1).

    Returns only the parent row of ``exp(tQ)`` for each of the ``total`` scaled
    rate matrices — the sole quantity the QDASM likelihood, ``predict_codons``
    argmax, and BL optimization consume — without ever forming the full
    ``[total, 61, 61]`` P.

    Routes per position by ``||tQ||_∞`` (row-sum inf-norm):

    * ``||tQ||_∞ <= tq_threshold`` → ``_row_expmv`` (action of the exponential
      on the parent row; memory-light, row-only, and — capped at
      ``tq_threshold`` (default 1.0) — needs only ``s <= 3`` sub-steps).
    * ``||tQ||_∞ > tq_threshold`` → the full ``long_path`` kernel
      (``ss_quintic`` by default) on that subset, then extract the parent row.
      Past the crossover (``||tQ||_∞ ~ 1.3`` on L40S) parallel squaring beats
      the row's sequential sub-steps, so the hot tail keeps the full kernel.

    Both branches match the parent row of ``torch.linalg.matrix_exp`` to fp32
    noise (~1e-6). Unlike the superseded full-matrix adaptive router this
    replaces, it never routes to cubic Taylor (whose negativity footgun forced
    the ``||tQ||_∞ <= 0.1`` gate — see the 2026-05-06 negativity audit); the
    ``_row_expmv`` Taylor is applied to the *vector* with per-matrix inner
    scaling, so it stays accurate across this ``||tQ||`` range.

    Routing is always per-position (the split is a cheap vector index), so the
    batched driver's ``adaptive_mode`` no longer changes results here.

    Args:
        tQ_flat: Scaled rate matrices ``[total, 61, 61]`` (already ``t * Q``).
        parent_row_idxs: Parent codon indices in 61-space ``[total]``.
        total: Number of matrices (``B * L``).
        device: Computation device.
        tq_threshold: ``||tQ||_∞`` cutoff between ``_row_expmv`` (at/below) and
            the full ``long_path`` kernel (above). Default 1.0 — just under the
            measured ~1.3 crossover, keeping ``_row_expmv`` at ``s <= 3``.
        max_matrices_per_chunk: Chunk size bounding peak memory of the
            long-kernel subset.
        long_path: Hot-tail kernel — ``"ss_quintic"`` (default) or
            ``"matrix_exp"``.

    Returns:
        Parent rows ``[total, 61]`` of ``exp(tQ)``.
    """
    dtype = tQ_flat.dtype
    tq_norms = tQ_flat.abs().sum(dim=-1).amax(dim=-1)  # [total]
    is_long = tq_norms > tq_threshold  # [total]

    parent_rows = torch.zeros(total, 61, device=device, dtype=dtype)

    row_idx = (~is_long).nonzero(as_tuple=True)[0]
    if row_idx.numel() > 0:
        row_parts = []
        for start in range(0, row_idx.numel(), max_matrices_per_chunk):
            sub = row_idx[start : start + max_matrices_per_chunk]
            row_parts.append(_row_expmv(tQ_flat[sub], parent_row_idxs[sub]))
        parent_rows = parent_rows.index_copy(0, row_idx, torch.cat(row_parts, dim=0))

    long_idx = is_long.nonzero(as_tuple=True)[0]
    if long_idx.numel() > 0:
        long_kernel = (
            _ss_quintic_parent_row
            if long_path == "ss_quintic"
            else _matrix_exp_parent_row
        )
        long_parts = []
        for start in range(0, long_idx.numel(), max_matrices_per_chunk):
            sub = long_idx[start : start + max_matrices_per_chunk]
            long_parts.append(long_kernel(tQ_flat[sub], parent_row_idxs[sub]))
        parent_rows = parent_rows.index_copy(0, long_idx, torch.cat(long_parts, dim=0))

    return parent_rows


def _batched_matrix_exp_to_log_probs(
    Q_all: Tensor,
    parent_codon_idxs: Tensor,
    branch_lengths: Tensor,
    L_output: int,
    max_matrices_per_chunk: int = 10000,
    quadratic_ste: bool = False,
    use_cubic: bool = False,
    apply_stop_zapper: bool = True,
    adaptive: bool = False,
    tq_threshold: float = 1.0,
    adaptive_mode: Literal["per_chunk", "per_sequence"] = "per_sequence",
    long_path: Literal["matrix_exp", "ss_quintic"] = "ss_quintic",
    matexp_method: Literal["adaptive", "uniformization"] = "adaptive",
    n_terms: int = _UNIFORMIZATION_DEFAULT_N_TERMS,
    normalization: Literal["parent_absorbing", "proportional"] = "parent_absorbing",
) -> Tensor:
    """Vectorized matrix exponential → log transition probabilities.

    Computes exp(t_b * Q_{b,l}) for all sequences and sites, processing in
    chunks to avoid GPU OOM. Much faster than a per-sequence loop because
    each chunk processes thousands of 61×61 matrix exponentials in parallel.

    Shared by compute_qdasm_log_codon_probs and compute_hybrid_log_codon_probs.

    ``matexp_method="uniformization"`` (issue #838) short-circuits everything
    below to the nonnegative-by-construction uniformization kernel
    (``uniformization_log_probs``) on the UNSCALED ``Q_all`` — before the
    ``tQ`` scaling the other four modes share — using per-matrix
    ``λ_i = max_k|Q_{i,kk}|`` (``per_matrix_lambda``) so a low ``n_terms``
    captures the most Poisson mass per matrix. Otherwise, four computational
    modes on ``tQ``:

    1. ``adaptive=True`` (production QDASM): parent-row hybrid — computes only
       the parent row of exp(tQ), routing per-position by ``||tQ||_∞``. Sites
       at/below ``tq_threshold`` use ``_row_expmv`` (action of the exponential
       on the row; no full matrix); the hot tail uses the full ``long_path``
       kernel (default ``ss_quintic``). Never materializes the full P; both
       branches carry consistent forward/backward. See issue #793 WI 1.
    2. ``use_cubic=True``: Cubic Taylor for forward+backward on the full P.
       Fast, but only stays non-negative for ``||tQ||_∞ ≲ 0.5`` (guarded by the
       entry-level negativity check below).
    3. ``quadratic_ste=True``: Exact forward, quadratic backward (STE). Can
       cause training divergence due to forward/backward mismatch.
    4. Default: Exact matrix_exp with Fréchet derivative backward on the full P.
       Correct but memory-intensive; used by the Hybrid head
       (``adaptive=False``).

    Args:
        Q_all: Rate matrices [B, L, 61, 61]
        parent_codon_idxs: Parent codon indices in 64-space [B, L]
        branch_lengths: Per-sequence branch lengths [B]
        L_output: Output sequence length (may be > L for padded AA positions)
        max_matrices_per_chunk: Max number of 61×61 matrices per matrix_exp call.
            Controls GPU memory usage vs speed tradeoff. Default 10000 is the
            sweet spot: same speed as larger chunks but ~10GB peak vs ~24GB at 50K.
            Benchmarked on A100 with B=1024, L=136 antibody data.
        quadratic_ste: Use quadratic straight-through estimator for gradients.
            Ignored when use_cubic=True, adaptive=True, or
            matexp_method="uniformization".
        use_cubic: Use cubic Taylor approximation for both forward and backward.
            Ignored when adaptive=True or matexp_method="uniformization".
        apply_stop_zapper: Whether to apply STOP_CODON_ZAPPER to the output.
            Set to False when the caller will apply it (e.g., Hybrid path passes
            output to compute_log_adjusted_codon_probs which applies it).
        adaptive: Use the parent-row hybrid mode (mode 1). Overrides use_cubic
            and quadratic_ste when True; this is the production QDASM path.
            Ignored when matexp_method="uniformization".
        tq_threshold: ``||tQ||_∞`` cutoff for the adaptive mode. Positions at or
            below it use ``_row_expmv`` (row-only); the hotter tail uses the
            full ``long_path`` kernel. Default 1.0 — just under the ~1.3
            row-vs-full crossover measured on L40S.
        adaptive_mode: Retained for checkpoint/config compat. The parent-row
            adaptive path always routes per-position by ``||tQ||_∞``, so this no
            longer changes results (``"per_chunk"`` and ``"per_sequence"`` give
            identical output).
        matexp_method: ``"adaptive"`` (default) dispatches to the four modes
            above based on ``adaptive``/``use_cubic``/``quadratic_ste``.
            ``"uniformization"`` (issue #838) instead calls
            ``uniformization_log_probs`` on the unscaled ``Q_all`` with
            per-matrix λ — a fast, nonnegative-by-construction low-order
            training surrogate under evaluation; see
            ``experiments/2026-07-13-qdasm-low-order-uniformization/``.
        n_terms: Uniformization truncation order. Only used when
            ``matexp_method="uniformization"``.
        normalization: Parent-row renormalization convention for the
            truncated uniformization row — ``"parent_absorbing"`` (default,
            classical DASM's convention: puts truncation error on the stay
            state) or ``"proportional"`` (spreads it across the whole row).
            Only used when ``matexp_method="uniformization"``.

    Returns:
        Log codon transition probabilities [B, L_output, 64]

    Notes:
        Stop-codon parents (``CODON64_TO_SENSE61[idx] == -1``) must be filtered
        upstream by ``ComputeMasks`` (see ``dasm2/data/transforms.py``). The
        molevol layer asserts this contract at dispatch entry.
    """
    raise_if_stop_codon_parent(parent_codon_idxs, "compute_qdasm_log_codon_probs")

    if matexp_method == "uniformization":
        # Unscaled Q_all — uniformization needs Q's own diagonal (for
        # per_matrix_lambda and P̃), unlike the tQ-scaled path below.
        # ``.detach()``: λ is a numerical-stability knob picked FROM the
        # current Q (the tightest bound keeping P̃ nonnegative), not a
        # modeled quantity — same convention as ``_row_expmv``'s ``.long()``
        # sub-step count. Differentiating through it would route gradient
        # through an ``amax`` argmax that can flip discretely between
        # near-tied rows, injecting subgradient noise into real training
        # gradients for no modeling benefit.
        lam = per_matrix_lambda(
            Q_all.reshape(-1, NUM_SENSE_CODONS, NUM_SENSE_CODONS)
        ).detach()
        return uniformization_log_probs(
            Q_all,
            parent_codon_idxs,
            branch_lengths,
            L_output,
            n_terms=n_terms,
            lam=lam,
            normalization=normalization,
            max_matrices_per_chunk=max_matrices_per_chunk,
            apply_stop_zapper=apply_stop_zapper,
        )

    device = Q_all.device
    B, L = Q_all.shape[:2]

    # Scale Q by per-sequence branch lengths: [B,1,1,1] * [B,L,61,61]
    tQ = branch_lengths[:, None, None, None] * Q_all
    tQ_flat = tQ.reshape(B * L, 61, 61)

    # QDASM/Hybrid likelihood, predict_codons argmax, and BL optimization all
    # consume only the PARENT ROW of exp(tQ). Convert parent indices to 61-space
    # up front so the adaptive path can compute that row directly (stop-codon
    # parents are guarded above).
    total = B * L
    parent_61_idxs = CODON64_TO_SENSE61_TENSOR.to(device)[
        parent_codon_idxs.clamp(max=63)
    ].clamp(min=0, max=NUM_SENSE_CODONS - 1)  # [B, L]

    if adaptive:
        # Parent-row hybrid (issue #793 WI 1): row_expmv for the ||tQ||_∞ bulk,
        # full ss_quintic/matrix_exp for the hot tail. Never materializes the
        # full [B*L, 61, 61] P; ``adaptive_mode`` is inert here (routing is
        # always per-position).
        parent_rows = _adaptive_parent_row_exp(
            tQ_flat,
            parent_61_idxs.reshape(total),
            total,
            device,
            tq_threshold,
            max_matrices_per_chunk,
            long_path=long_path,
        ).reshape(B, L, 61)
        # NaN + non-negativity checks cover ONLY the consumed parent row
        # [B, L, 61] — the sole entries that reach the loss. This is a
        # deliberately narrower surface than the pre-WI-1 full-P path, which
        # checked all 61 rows of every [B, L, 61, 61] P (~18 ms of scans over the
        # 60 rows/site that were then discarded). The ``_row_expmv`` branch is
        # row-space throughout, so there is no "other 60 rows" to have gone
        # wrong. The hot-tail branch (``_ss_quintic_parent_row`` /
        # ``_matrix_exp_parent_row``) DOES still form a full [N, 61, 61]
        # internally, and its non-parent rows now go unchecked before being
        # dropped — an accepted trade-off: every known blow-up mode here (NaN,
        # cubic negativity — the latter no longer even routed to) is uniform
        # across a matrix's rows, so a corruption that spared the parent row
        # while hitting only the unused rows is not a failure mode any of them
        # produce. Same -1e-3 bug threshold as the full-P paths;
        # ``TestHotQRoutingRegression`` still guards the case that motivated the
        # original full-matrix check.
        if torch.isnan(parent_rows).any():
            nan_count = int(torch.isnan(parent_rows).any(dim=-1).sum().item())
            raise RuntimeError(
                f"Parent-row exp(tQ) contains NaN entries (numerical blow-up): "
                f"{nan_count} affected (B,L) sites, "
                f"max |Q| = {float(Q_all.abs().max().item()):.3e}, "
                f"max BL = {float(branch_lengths.max().item()):.3e}"
            )
        if not (parent_rows >= -1e-3).all():
            raise RuntimeError(
                f"Parent-row exp(tQ) has negative entries: "
                f"min = {float(parent_rows.min().item()):.2e}; "
                f"max |Q| = {float(Q_all.abs().max().item()):.3e}, "
                f"max BL = {float(branch_lengths.max().item()):.3e}. "
                f"This indicates a numerical bug, not float32 cancellation."
            )
        parent_rows = parent_rows.clamp(min=1e-10)
    else:
        # Full-matrix paths (use_cubic / quadratic_ste / exact matrix_exp): form
        # the whole P, sanity-check it, then extract the parent row. Only the
        # default ``adaptive=True`` QDASM path uses the parent-row kernel above.
        if use_cubic:
            # Warn if branch lengths are large (cubic error grows as O(t^4))
            max_t = branch_lengths.max().item()
            if max_t > 0.5:
                warnings.warn(
                    f"use_cubic=True with max branch length {max_t:.3f} > 0.5; "
                    f"cubic Taylor error may be significant (O(t^4))",
                    stacklevel=2,
                )
            eye = torch.eye(61, device=device, dtype=tQ_flat.dtype)
            P_flat = torch.empty(total, 61, 61, device=device, dtype=tQ_flat.dtype)
            for start in range(0, total, max_matrices_per_chunk):
                end = min(start + max_matrices_per_chunk, total)
                P_flat[start:end] = _cubic_taylor_chunk(tQ_flat[start:end], eye)
        else:
            use_ste = quadratic_ste and torch.is_grad_enabled()
            if use_ste:
                # STE path: exact forward (no_grad), quadratic surrogate backward
                eye_ste = torch.eye(61, device=device, dtype=tQ_flat.dtype)
                P_flat = torch.empty(total, 61, 61, device=device, dtype=tQ_flat.dtype)
                for start in range(0, total, max_matrices_per_chunk):
                    end = min(start + max_matrices_per_chunk, total)
                    tQ_chunk = tQ_flat[start:end]
                    with torch.no_grad():
                        P_exact = torch.linalg.matrix_exp(tQ_chunk)
                    P_quad = eye_ste + tQ_chunk + 0.5 * (tQ_chunk @ tQ_chunk)
                    # Straight-through: exact forward value, quadratic backward
                    P_flat[start:end] = P_exact + (P_quad - P_quad.detach())
            else:
                # Exact path: full Fréchet derivative backward through matrix_exp
                P_flat = torch.empty(total, 61, 61, device=device, dtype=tQ_flat.dtype)
                for start in range(0, total, max_matrices_per_chunk):
                    end = min(start + max_matrices_per_chunk, total)
                    tQ_chunk = tQ_flat[start:end]
                    P_flat[start:end] = torch.linalg.matrix_exp(tQ_chunk)

        P = P_flat.reshape(B, L, 61, 61)
        if torch.isnan(P).any():
            nan_count = int(torch.isnan(P).any(dim=(-2, -1)).sum().item())
            max_abs_q = float(Q_all.abs().max().item())
            max_bl = float(branch_lengths.max().item())
            raise RuntimeError(
                f"P matrix contains NaN entries (numerical blow-up in matrix_exp): "
                f"{nan_count} affected (B,L) sites, max |Q| = {max_abs_q:.3e}, "
                f"max BL = {max_bl:.3e}"
            )
        # Entry-level non-negativity check. ``ss_quintic`` and ``matrix_exp``
        # produce zero negatives below -1e-6; cubic Taylor only stays positive
        # when ||tQ||_∞ ≲ 0.5. An entry below -1e-3 is a real numerical bug —
        # not float32 noise. See
        # experiments/2026-05-06-matexp-negativity-audit/FINDINGS.md for the
        # silent-corruption mechanism this used to hide.
        if not (P >= -1e-3).all():
            min_val = P.min().item()
            max_abs_q = float(Q_all.abs().max().item())
            max_bl = float(branch_lengths.max().item())
            raise RuntimeError(
                f"P matrix has negative entries: min = {min_val:.2e}; "
                f"max |Q| = {max_abs_q:.3e}, max BL = {max_bl:.3e}. "
                f"This indicates a numerical bug, not float32 cancellation "
                f"(see experiments/2026-05-06-matexp-negativity-audit)."
            )
        P = P.clamp(min=1e-10)

        # Extract parent rows: P[b, l, parent_61[b,l], :] → [B, L, 61]
        b_idx = torch.arange(B, device=device)[:, None].expand(B, L)
        l_idx = torch.arange(L, device=device)[None, :].expand(B, L)
        parent_rows = P[b_idx, l_idx, parent_61_idxs]  # [B, L, 61]

    # Embed 61 → 64: insert zeros at stop codon positions
    sense_codon_list = SENSE_CODON_LIST_TENSOR.to(device)
    result_64 = torch.zeros(B, L_output, 64, device=device)
    result_64[:, :L, sense_codon_list] = parent_rows

    # Log probs, optionally with stop codon zapping
    result_64 = torch.log(result_64.clamp(min=1e-30))
    if apply_stop_zapper:
        result_64 = result_64 + STOP_CODON_ZAPPER.to(device)[None, None, :]

    return result_64


def mutsel_codon_transition_probs(
    q_matrices: Tensor,
    parent_codon_idxs: Tensor,
    branch_length: float,
    use_cubic: bool = False,
    apply_stop_zapper: bool = True,
    adaptive: bool = False,
    tq_threshold: float = 0.1,
    long_path: Literal["matrix_exp", "ss_quintic"] = "ss_quintic",
) -> Tensor:
    """Compute codon transition probabilities via matrix exponential P = exp(tQ).

    Args:
        q_matrices: Rate matrices [L, 61, 61]
        parent_codon_idxs: Parent codon indices in 64-space [L]
        branch_length: Evolutionary time (scalar)
        use_cubic: Use cubic Taylor approximation I + tQ + (tQ)^2/2 + (tQ)^3/6
            instead of exact matrix exponential. ~12x faster.
        apply_stop_zapper: Whether to apply STOP_CODON_ZAPPER to the output.
            Set to False when the caller will apply it (e.g., Hybrid path passes
            output to compute_log_adjusted_codon_probs which applies it).
        adaptive: Use hybrid cubic/long-kernel mode. The single-sample path
            takes the **maximum** ``||tQ||_∞`` across all positions in the
            sequence and routes the entire sequence to one kernel: cubic if
            ``max ||tQ||_∞ ≤ tq_threshold``, otherwise the long kernel
            (``ss_quintic`` by default). This is one-shot routing, not
            per-row — analogous to the batched path's ``per_chunk`` mode but
            with a chunk of size L. Conservative-correct: any single hot row
            sends the whole sequence to the long kernel.
        tq_threshold: ``||tQ||_∞`` threshold for adaptive mode (default 0.1,
            empirical crossover where cubic abs error rises above ss_quintic's).
        long_path: Long-||tQ|| kernel. ``"ss_quintic"`` (default) matches the
            batched path; ``"matrix_exp"`` uses ``torch.linalg.matrix_exp``.

    Returns:
        Log transition probabilities in 64-space [L, 64] with stop codons zapped

    Notes:
        Stop-codon parents (``CODON64_TO_SENSE61[idx] == -1``) must be filtered
        upstream by ``ComputeMasks`` (see ``dasm2/data/transforms.py``). The
        molevol layer asserts this contract at dispatch entry.
    """
    raise_if_stop_codon_parent(parent_codon_idxs, "mutsel_codon_transition_probs")

    L = q_matrices.shape[0]
    device = q_matrices.device
    tQ = branch_length * q_matrices  # [L, 61, 61]

    # Adaptive mode: one-shot routing on the worst-case ``||tQ||_∞`` across
    # the sequence. Cubic if every row stays inside the safe envelope; the
    # long kernel otherwise. (Per-row routing within a single sample is not
    # implemented here — callers that want it should use the batched path.)
    use_cubic_effective = use_cubic
    if adaptive:
        max_tq_norm = float(tQ.abs().sum(dim=-1).max().item())
        use_cubic_effective = max_tq_norm <= tq_threshold

    if use_cubic_effective:
        # Cubic Taylor: P ≈ I + tQ + (tQ)²/2 + (tQ)³/6.
        # See experiments/2026-05-06-matexp-negativity-audit/FINDINGS.md for
        # why cubic must be gated by ||tQ||_∞, not by branch length alone.
        tQ2 = torch.bmm(tQ, tQ)
        eye = torch.eye(NUM_SENSE_CODONS, device=device, dtype=tQ.dtype).unsqueeze(0)
        P = eye + tQ + 0.5 * tQ2 + (1.0 / 6.0) * torch.bmm(tQ2, tQ)
    elif long_path == "ss_quintic":
        # Scaling-and-squaring quintic: ~1e-6 max abs err vs fp64 across
        # ||tQ||_∞ up to 300 on real Q (audit experiment).
        P = _scaling_squaring_quintic_chunk(tQ)
    else:
        # f32 matrix_exp: max abs err ~1e-6 (vs f64), but Fréchet-derivative
        # backward is ~3× more expensive than ss_quintic's plain matmul chain.
        P = torch.linalg.matrix_exp(tQ)

    # Sanity checks. Generator-form Q has ``Q·1 = 0``, so ``exp(tQ)·1 = 1``.
    # Cubic Taylor preserves this analytically (since ``(tQ)^k·1 = 0`` for k ≥ 1),
    # so the row-sum check alone misses cubic's entry-level mistakes — the
    # entry-level check below is the one that catches them.
    row_sums = P.sum(dim=-1)
    max_deviation = (row_sums - 1).abs().max().item()
    if max_deviation > 1e-3:
        raise ValueError(
            f"P matrix rows don't sum to 1: max deviation = {max_deviation}"
        )
    if torch.isnan(P).any():
        nan_count = int(torch.isnan(P).any(dim=(-2, -1)).sum().item())
        max_abs_q = float(q_matrices.abs().max().item())
        raise RuntimeError(
            f"P matrix contains NaN entries (numerical blow-up in matrix_exp): "
            f"{nan_count} affected sites, max |Q| = {max_abs_q:.3e}, "
            f"BL = {branch_length:.3e}"
        )
    if not (P >= -1e-3).all():
        min_val = P.min().item()
        max_abs_q = float(q_matrices.abs().max().item())
        raise RuntimeError(
            f"P matrix has negative entries: min = {min_val:.2e}; "
            f"max |Q| = {max_abs_q:.3e}, BL = {branch_length:.3e}. "
            f"This indicates a numerical bug, not float32 cancellation "
            f"(see experiments/2026-05-06-matexp-negativity-audit)."
        )

    # Clamp small negative values to a small positive floor.
    # Zero would propagate through log() as -inf, risking NaN in training.
    P = P.clamp(min=1e-10)

    # Convert parent indices from 64-space to 61-space using the cached
    # module-level lookup. Stop-codon parents are guarded above; ambiguous
    # codons (index >= 64) clamp to 63 — the mask prevents their contribution
    # to the final likelihood.
    parent_61_idxs = CODON64_TO_SENSE61_TENSOR.to(device)[
        parent_codon_idxs.clamp(max=63)
    ].clamp(min=0, max=NUM_SENSE_CODONS - 1)

    # Extract parent rows: P[l, parent_61[l], :] → [L, 61]
    parent_rows = P[torch.arange(L, device=device), parent_61_idxs]  # [L, 61]

    # Embed 61→64: insert zeros at stop codon positions
    sense_codon_list = SENSE_CODON_LIST_TENSOR.to(device)
    result_64 = torch.zeros((L, 64), device=device, dtype=parent_rows.dtype)
    result_64[:, sense_codon_list] = parent_rows

    # Return log probs, optionally with stop codon zapping
    log_probs = torch.log(result_64.clamp(min=1e-30))
    if apply_stop_zapper:
        log_probs = log_probs + STOP_CODON_ZAPPER.to(device)

    return log_probs
