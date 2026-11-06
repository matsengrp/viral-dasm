"""Matrix-exp GY94 neutral codon probabilities for the precomputed training path.

This is a pilot-only alternative to the product-of-three Poisson neutral
applied at ``_batched_compute_neutral_codon_probs``. It reconstructs the
per-family GTR Q_nt 4×4 from each PCP's ``nt_rates`` and ``nt_csps``,
recovers the implied exchangeabilities and stationary nucleotide
distribution, then delegates the GY94 ``Q_codon`` construction and
``P = exp(Q · branch_length)`` to ``GTRCodonNeutralModel`` so the two
paths cannot drift.

The neutral path is gated by the env var ``DASM2_NEUTRAL_MODE``:

- ``DASM2_NEUTRAL_MODE=matrix_exp`` → use this module.
- unset / ``product_of_three`` → use the existing product-of-three path.

The matrix-exp path is meant for diagnosis (issue #654); making it the
default would require: (a) batched torch.matrix_exp at training step
time when branch-length hooks update branch lengths, and (b) confirming
hit-scale defaults don't need to be refit. Both are deliberately out of
scope for the pilot.
"""

from __future__ import annotations

import math
import os
from typing import Literal

import torch

from dasm2.codon_table import (
    CODON64_TO_SENSE61_TENSOR,
    SENSE_CODON_LIST_TENSOR,
)
from dasm2.constants import SMALL_PROBABILITY
from dasm2.neutral.gtr_codon import EXCHANGEABILITY_PAIRS, GTRCodonNeutralModel
from dasm2.utils.tensor import clamp_probability

NeutralMode = Literal["matrix_exp", "product_of_three"]

_NEUTRAL_MODE_ENV = "DASM2_NEUTRAL_MODE"
_MATRIX_EXP_MODE: NeutralMode = "matrix_exp"
_PRODUCT_OF_THREE_MODE: NeutralMode = "product_of_three"


def is_matrix_exp_mode_enabled() -> bool:
    """Return True iff ``DASM2_NEUTRAL_MODE=matrix_exp``."""
    return os.environ.get(_NEUTRAL_MODE_ENV, _PRODUCT_OF_THREE_MODE) == _MATRIX_EXP_MODE


def check_multihit_compatible(multihit_model: object | None) -> None:
    """Raise if a multihit correction is passed while matrix-exp mode is on.

    The matrix-exp neutral already accounts for multi-step transitions
    through ``expm(Q · t)``; layering a separate multihit correction on
    top would double-count. Both call sites that gate on the env var
    use this same guard so the error message stays in lockstep.
    """
    if multihit_model is not None:
        raise ValueError(
            f"{_NEUTRAL_MODE_ENV}={_MATRIX_EXP_MODE} is incompatible with "
            "multihit_model; the matrix-exp neutral already accounts for "
            "multi-step transitions."
        )


def check_codon_pref_compatible(codon_pref_log: "torch.Tensor | None") -> None:
    """Raise if a codon preference table is passed while matrix-exp mode is on.

    Mirrors :func:`check_multihit_compatible`: both the batched precomputed
    path (``_batched_compute_neutral_codon_probs``) and the per-sample
    streaming path (``ComputeNeutralCodonProbs``) gate on the env var and
    use this same guard so the error message stays in lockstep (issue
    #1039).
    """
    if codon_pref_log is not None:
        raise ValueError(
            "Codon preference table (codon_pref_log) is incompatible with "
            f"{_NEUTRAL_MODE_ENV}={_MATRIX_EXP_MODE}. Use the default "
            "product-of-three neutral mode when applying codon-optimality "
            "tables (issue #747)."
        )


def q_nt_from_rates_csps(
    parent_nt_idxs_flat: torch.Tensor,
    nt_rates: torch.Tensor,
    nt_csps: torch.Tensor,
) -> torch.Tensor:
    """Recover a 4×4 GTR nucleotide rate matrix from per-position rates/CSPs.

    Each position with parent nt ``i`` carries ``rate = -Q_nt[i, i]`` and
    ``csps[i, j] = Q_nt[i, j] / rate`` for ``j != i``. The 4 rows of
    Q_nt are therefore recoverable from any 4 positions whose parent
    nucleotides span {A,C,G,T}. Compara parent sequences are >800 nt, so
    every base type is essentially always present.

    Args:
        parent_nt_idxs_flat: shape (n_nt,) parent nucleotide indices (0..3)
        nt_rates: shape (n_nt,) per-position departure rates
        nt_csps: shape (n_nt, 4) per-position conditional substitution probs

    Returns:
        Q_nt of shape (4, 4) as float64. Raises ValueError if any base is
        missing from the parent.
    """
    Q_nt = torch.zeros((4, 4), dtype=torch.float64)
    found = [False, False, False, False]
    # Pick the position with the largest rate for each base. Positions
    # whose original parent nt was 'N' get rate=0 / uniform csps from
    # ``compute_rates_and_csps``, so the first-position heuristic fails
    # in N-rich tails. Maximizing the rate avoids that.
    nt_rates_f64 = nt_rates.to(torch.float64)
    for base in range(4):
        mask = parent_nt_idxs_flat == base
        if not mask.any():
            continue
        masked_rates = torch.where(
            mask, nt_rates_f64, torch.full_like(nt_rates_f64, -1.0)
        )
        pos = int(masked_rates.argmax().item())
        rate = float(nt_rates_f64[pos].item())
        if rate <= 0.0:
            continue
        csps_row = nt_csps[pos].to(torch.float64)
        Q_nt[base, :] = csps_row * rate
        Q_nt[base, base] = -rate
        found[base] = True
    if not all(found):
        missing = [i for i, ok in enumerate(found) if not ok]
        raise ValueError(
            f"Cannot recover Q_nt: missing base indices {missing} — each is "
            "either absent from the parent or has all-nonpositive rates "
            "(e.g. an N-padding tail), so its Q_nt row cannot be reconstructed."
        )
    return Q_nt


def stationary_distribution_of_q(Q_nt: torch.Tensor) -> torch.Tensor:
    """Compute π_nt: stationary distribution of the 4x4 GTR Q_nt.

    π satisfies π^T Q = 0 with Σ π = 1. We solve as a linear system by
    replacing the last equation in Q^T x = 0 with Σ x = 1.
    """
    A = Q_nt.T.clone()
    b = torch.zeros(4, dtype=torch.float64)
    A[-1, :] = 1.0
    b[-1] = 1.0
    pi: torch.Tensor = torch.linalg.solve(A, b)
    return pi


def build_reference_model(
    Q_nt: torch.Tensor, pi_nt: torch.Tensor
) -> GTRCodonNeutralModel:
    """Recover the (exchangeabilities, frequencies) GTR parameterization
    from a recovered Q_nt and stationary π_nt, then instantiate the
    canonical ``GTRCodonNeutralModel``.

    For a reversible GTR rate matrix Q[i, j] = r_ij × π_j with i ≠ j,
    so r_ij = Q[i, j] / π_j. The 6-vector matches the order used by
    ``EXCHANGEABILITY_PAIRS`` in ``dasm2.neutral.gtr_codon``.

    Delegating the codon Q construction and matrix exponentiation to
    ``GTRCodonNeutralModel`` ensures the pilot path can never drift
    from the validated implementation.
    """
    exchangeabilities = torch.tensor(
        [float(Q_nt[i, j].item() / pi_nt[j].item()) for i, j in EXCHANGEABILITY_PAIRS],
        dtype=torch.float32,
    )
    return GTRCodonNeutralModel(exchangeabilities, pi_nt.to(torch.float32))


def compute_neutral_codon_probs_matrix_exp_per_pcp(
    parent_nt_idxs_flat: torch.Tensor,
    parent_sense_idxs_per_codon: torch.Tensor,
    nt_rates: torch.Tensor,
    nt_csps: torch.Tensor,
    branch_length: float,
) -> torch.Tensor:
    """Compute per-position neutral codon probs via GY94 matrix-exp.

    Args:
        parent_nt_idxs_flat: shape (3*num_codons,) flat parent nt indices.
        parent_sense_idxs_per_codon: shape (num_codons,) sense-61 indices
            for each parent codon; -1 for stop/ambig (handled by caller).
        nt_rates: shape (3*num_codons,) parent-base rates.
        nt_csps: shape (3*num_codons, 4) parent-base CSPs.
        branch_length: float, evolutionary distance.

    Returns:
        codon_probs: shape (num_codons, 64) probabilities of each child
            codon given parent codon at every position. Stop child codons
            stay at 0 (probabilities only defined over sense targets).
    """
    Q_nt = q_nt_from_rates_csps(parent_nt_idxs_flat, nt_rates, nt_csps)
    pi_nt = stationary_distribution_of_q(Q_nt)
    model = build_reference_model(Q_nt, pi_nt)
    P_codon = model.compute_transition_matrix(float(branch_length))  # (61, 61)

    num_codons = parent_sense_idxs_per_codon.shape[0]
    out = torch.zeros((num_codons, 64), dtype=torch.float32)

    valid_mask = parent_sense_idxs_per_codon >= 0
    if not valid_mask.any():
        return out

    # P_codon[parent, :] for each valid position, then scatter to 64-space.
    valid_idx = parent_sense_idxs_per_codon[valid_mask].to(torch.long)
    rows = P_codon[valid_idx, :]  # (n_valid, 61)
    scratch = torch.zeros((rows.shape[0], 64), dtype=torch.float32)
    scratch[:, SENSE_CODON_LIST_TENSOR] = rows
    out[valid_mask] = scratch
    return out


def finalize_matrix_exp_log_probs(
    probs: torch.Tensor,
    truncated_mask: torch.Tensor,
    target_len: int,
    *,
    error_context: str = "",
) -> torch.Tensor:
    """Validate-clamp-pad-nan_to_num-log finalization for matrix-exp probs.

    Single source of truth for the per-sample bookkeeping shared by the
    streaming transform and the batched precomputed path; keeps the two
    matrix-exp code paths from drifting (see the parity test in
    ``tests/data/test_neutral_matrix_exp.py``).

    Args:
        probs: (aa_length, 64) per-PCP matrix-exp codon probabilities.
        truncated_mask: (aa_length,) loss mask truncated to aa_length.
        target_len: original ``codon_parents_idxs`` length (for pad / truncate).
        error_context: prefix for the ValueError message, e.g. ``"Sample 3: "``.

    Returns:
        ``log_neutral_codon_probs`` of shape ``(target_len, 64)``.
    """
    aa_length = probs.shape[0]
    if aa_length == 0:
        return torch.full((target_len, 64), math.log(SMALL_PROBABILITY))

    if truncated_mask.any() and not torch.isfinite(probs[truncated_mask]).all():
        raise ValueError(
            f"{error_context}matrix-exp neutral_probs has non-finite values"
        )
    probs_out = probs.clone()
    probs_out[truncated_mask] = clamp_probability(probs[truncated_mask])
    probs_out[~truncated_mask] = float("nan")

    if aa_length < target_len:
        probs_out = torch.nn.functional.pad(
            probs_out, (0, 0, 0, target_len - aa_length), value=SMALL_PROBABILITY
        )
    elif aa_length > target_len:
        probs_out = probs_out[:target_len]

    probs_out = torch.nan_to_num(probs_out, nan=SMALL_PROBABILITY)
    return torch.log(probs_out)


def parent_sense_idxs_of_codon64_idxs(codon64_idxs: torch.Tensor) -> torch.Tensor:
    """Map (num_codons,) codon-64 indices to (num_codons,) sense-61 indices.

    Returns -1 for stop codons or out-of-range entries; the caller is
    expected to mask these out via the per-PCP loss_mask.
    """
    return CODON64_TO_SENSE61_TENSOR[codon64_idxs.clamp(0, 63)]
