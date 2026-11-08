"""QDASM Q-matrix construction and dispatch.

The CTMC rate matrix Q over 61 sense codons, built from per-position parent NT
rates, a global k-mer lookup, or a context-free codon generator. The dispatchers
``_build_q_matrix`` (unbatched) and ``_build_q_matrix_batched`` choose which
builder to call based on ``q_variant``.

Extracted from ``mutsel.py`` to mirror the rest of the ``molevol`` package
factorization (mutation_matrices, tensor_ops, codon_neutral_rates, ...).
"""

import logging
from typing import Literal, Optional, cast

from torch import Tensor

QVariant = Literal["parent_row_degenerate", "global_13mer", "global_5mer", "shm1_codon"]
GlobalKmerQVariant = Literal["global_13mer", "global_5mer"]

from dasm2.molevol._parent_row_q import (
    _build_mutsel_q_matrices_batched,
    build_mutsel_q_from_codon_q,
    build_mutsel_q_matrices,
)
from dasm2.molevol.kmer_q import build_mutsel_q_matrices_from_kmer_lookup
from dasm2.molevol.matrix_exp import mutsel_codon_transition_probs

# The mapping is the single source of truth for variants backed by a global
# k-mer lookup. Derive the public set from it so adding another bounded-context
# neutral model cannot update one dispatcher while silently missing another.
_GLOBAL_KMER_LENGTHS: dict[GlobalKmerQVariant, int] = {
    "global_13mer": 13,
    "global_5mer": 5,
}
GLOBAL_KMER_Q_VARIANTS: frozenset[QVariant] = frozenset(_GLOBAL_KMER_LENGTHS)

__all__ = [
    "QVariant",
    "GLOBAL_KMER_Q_VARIANTS",
    "shared_codon_length",
    "build_mutsel_q_matrices",
    "_build_mutsel_q_matrices_batched",
    "build_mutsel_q_from_codon_q",
    "mutsel_codon_transition_probs",
    "_build_q_matrix",
    "_build_q_matrix_batched",
]

logger = logging.getLogger(__name__)


def shared_codon_length(l_aa: int, codon_rates: Tensor) -> int:
    """Codon positions that have both an AA slot and NT rates.

    For multi-segment data ``L_codon`` can differ from ``L_aa`` because
    separators occupy AA positions but carry no NT rates, so Q only ever
    covers the shared prefix. Defined once here because three call sites must
    agree on it: the QDASM and Hybrid entry points that build Q inline, and
    the BL-opt hoist in ``dasm2.training.hooks.branch_lengths`` that builds Q
    ahead of the golden-section loop and hands it back as ``precomputed_q``
    (QDASM) / ``precomputed_neutral_q`` (Hybrid). Drift between them would trip
    the shape check at the injection point.

    Args:
        l_aa: AA-space sequence length (``parent_codon_idxs.shape[1]``).
        codon_rates: ``[B, L_codon, 3]`` per-position NT rates.
    """
    return min(l_aa, codon_rates.shape[1])


def _validate_global_kmer_inputs(
    q_variant: GlobalKmerQVariant,
    codon_parent_nt_window: Optional[Tensor],
    codon_parent_boundary_mask: Optional[Tensor],
    table: Optional[Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate the three tensors required by a global k-mer Q variant.

    Returns the same tensors with ``Optional`` stripped so callers can pass
    the result directly to the lookup builders without further None checks.
    """
    missing = [
        name
        for name, value in (
            ("kmer_table", table),
            ("codon_parent_nt_window", codon_parent_nt_window),
            ("codon_parent_boundary_mask", codon_parent_boundary_mask),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"q_variant='{q_variant}' requires {missing} to be non-None "
            "(all of kmer_table, codon_parent_nt_window, and "
            "codon_parent_boundary_mask must be provided)."
        )
    assert codon_parent_nt_window is not None  # for mypy
    assert codon_parent_boundary_mask is not None
    assert table is not None
    return codon_parent_nt_window, codon_parent_boundary_mask, table


def _global_kmer_length(q_variant: GlobalKmerQVariant) -> int:
    """Return the lookup width for a registered global-kmer variant."""
    try:
        return _GLOBAL_KMER_LENGTHS[q_variant]
    except KeyError as exc:
        raise ValueError(
            f"q_variant='{q_variant}' is marked as a global k-mer variant but "
            "has no registered k-mer length"
        ) from exc


def _validate_shm1_codon_inputs(neutral_codon_q: Optional[Tensor]) -> Tensor:
    """Validate the neutral codon generator required by ``q_variant='shm1_codon'``.

    Returns the tensor with ``Optional`` stripped so callers can pass it
    directly to ``build_mutsel_q_from_codon_q``.
    """
    if neutral_codon_q is None:
        raise ValueError(
            "q_variant='shm1_codon' requires neutral_codon_q to be non-None "
            "(the [61, 61] SHM-1 codon generator; QDASMLoss loads it from "
            "neutral_codon_q_path into a buffer)."
        )
    return neutral_codon_q


def _build_q_matrix(
    codon_rates: Tensor,
    codon_csps: Tensor,
    aa_selection_factors: Tensor,
    single_hit_scale: float = 1.0,
    double_hit_scale: float = 0.02,
    triple_hit_scale: float = 0.001,
    q_variant: QVariant = "parent_row_degenerate",
    codon_parent_nt_window: Optional[Tensor] = None,
    codon_parent_boundary_mask: Optional[Tensor] = None,
    kmer_table: Optional[Tensor] = None,
    neutral_codon_q: Optional[Tensor] = None,
    codon_boundary_features: Optional[Tensor] = None,
    codon_rate_adjustments: Optional[Tensor] = None,
) -> Tensor:
    """Dispatcher for the unbatched Q-matrix construction variants.

    Args:
        codon_rates: [L, 3] per-position parent rates (used for the
            ``parent_row_degenerate`` variant).
        codon_csps: [L, 3, 4] per-position parent CSPs (same).
        aa_selection_factors: [L, 20] (NOT log).
        single_hit_scale: 1-nt attenuator (γ).
        double_hit_scale, triple_hit_scale: Multi-hit attenuators.
        q_variant: One of ``"parent_row_degenerate"``, ``"global_13mer"``,
            ``"global_5mer"``, or ``"shm1_codon"``.
        codon_parent_nt_window: [L, 15] — required when
            ``q_variant`` is ``"global_13mer"`` or ``"global_5mer"``.
        codon_parent_boundary_mask: [L] bool — required when
            ``q_variant`` is ``"global_13mer"`` or ``"global_5mer"``.
        codon_boundary_features: Sparse cached full-segment source features
            ``[..., N, 916]``; required at valid boundary/ambiguous-window sites.
        codon_rate_adjustments: Required per-site multipliers for global k-mer
            variants; rejected for other variants. See
            :func:`~dasm2.molevol.kmer_q.build_mutsel_q_matrices_from_kmer_lookup`
            and ``docs/concepts/neutral-models.md#heavylight-rate-adjustment``.
        kmer_table: ``[4^k, 5]`` float lookup matching the selected global
            k-mer variant.
        neutral_codon_q: [61, 61] float — required when
            ``q_variant="shm1_codon"``.

    Returns:
        Q matrices ``[L, 61, 61]``.
    """
    if q_variant in GLOBAL_KMER_Q_VARIANTS:
        global_variant = cast(GlobalKmerQVariant, q_variant)
        nt_window, boundary, table = _validate_global_kmer_inputs(
            global_variant,
            codon_parent_nt_window,
            codon_parent_boundary_mask,
            kmer_table,
        )
        return build_mutsel_q_matrices_from_kmer_lookup(
            nt_window,
            boundary,
            aa_selection_factors,
            table,
            _global_kmer_length(global_variant),
            single_hit_scale=single_hit_scale,
            double_hit_scale=double_hit_scale,
            triple_hit_scale=triple_hit_scale,
            codon_rates=codon_rates,
            codon_csps=codon_csps,
            codon_boundary_features=codon_boundary_features,
            codon_rate_adjustments=codon_rate_adjustments,
        )

    if codon_rate_adjustments is not None:
        raise ValueError(
            f"q_variant='{q_variant}' does not accept codon_rate_adjustments; "
            "provide already-adjusted neutral rates for non-global variants."
        )

    if q_variant == "shm1_codon":
        neutral_q = _validate_shm1_codon_inputs(neutral_codon_q)
        return build_mutsel_q_from_codon_q(
            neutral_q,
            aa_selection_factors,
            single_hit_scale=single_hit_scale,
            double_hit_scale=double_hit_scale,
            triple_hit_scale=triple_hit_scale,
        )

    # Default: parent_row_degenerate. Pydantic config validation rejects
    # unknown variants before reaching this dispatcher.
    return build_mutsel_q_matrices(
        codon_rates,
        codon_csps,
        aa_selection_factors,
        single_hit_scale=single_hit_scale,
        double_hit_scale=double_hit_scale,
        triple_hit_scale=triple_hit_scale,
    )


def _build_q_matrix_batched(
    codon_rates: Tensor,
    codon_csps: Tensor,
    aa_selection_factors: Tensor,
    single_hit_scale: float = 1.0,
    double_hit_scale: float = 0.02,
    triple_hit_scale: float = 0.001,
    q_variant: QVariant = "parent_row_degenerate",
    codon_parent_nt_window: Optional[Tensor] = None,
    codon_parent_boundary_mask: Optional[Tensor] = None,
    kmer_table: Optional[Tensor] = None,
    neutral_codon_q: Optional[Tensor] = None,
    codon_boundary_features: Optional[Tensor] = None,
    codon_rate_adjustments: Optional[Tensor] = None,
) -> Tensor:
    """Batched twin of ``_build_q_matrix``.

    Same dispatch logic, but every per-sequence tensor gains a leading batch
    dimension. Used by ``compute_qdasm_log_codon_probs`` and
    ``compute_hybrid_log_codon_probs`` so they don't have to inline the
    if/elif themselves.

    Args:
        codon_rates: [B, L, 3]
        codon_csps: [B, L, 3, 4]
        aa_selection_factors: [B, L, 20] (NOT log). Pass ``ones_like`` for
            Hybrid's neutral Q.
        single_hit_scale: 1-nt attenuator (γ).
        double_hit_scale, triple_hit_scale: Multi-hit attenuators.
        q_variant: One of ``"parent_row_degenerate"``, ``"global_13mer"``,
            ``"global_5mer"``, or ``"shm1_codon"``.
        codon_parent_nt_window: [B, L, 15] — required when
            ``q_variant`` is ``"global_13mer"`` or ``"global_5mer"``.
        codon_parent_boundary_mask: [B, L] bool — required when
            ``q_variant`` is ``"global_13mer"`` or ``"global_5mer"``.
        codon_boundary_features: Sparse cached full-segment source features
            ``[..., N, 916]``; required at valid boundary/ambiguous-window sites.
        codon_rate_adjustments: Required per-site multipliers for global k-mer
            variants; rejected for other variants. See
            :func:`~dasm2.molevol.kmer_q.build_mutsel_q_matrices_from_kmer_lookup`
            and ``docs/concepts/neutral-models.md#heavylight-rate-adjustment``.
        kmer_table: ``[4^k, 5]`` lookup matching the selected global-k-mer
            variant.
        neutral_codon_q: [61, 61] — required when
            ``q_variant="shm1_codon"``.

    Returns:
        Q matrices ``[B, L, 61, 61]``.
    """
    if q_variant in GLOBAL_KMER_Q_VARIANTS:
        global_variant = cast(GlobalKmerQVariant, q_variant)
        nt_window, boundary, table = _validate_global_kmer_inputs(
            global_variant,
            codon_parent_nt_window,
            codon_parent_boundary_mask,
            kmer_table,
        )
        return build_mutsel_q_matrices_from_kmer_lookup(
            nt_window,
            boundary,
            aa_selection_factors,
            table,
            _global_kmer_length(global_variant),
            single_hit_scale=single_hit_scale,
            double_hit_scale=double_hit_scale,
            triple_hit_scale=triple_hit_scale,
            codon_rates=codon_rates,
            codon_csps=codon_csps,
            codon_boundary_features=codon_boundary_features,
            codon_rate_adjustments=codon_rate_adjustments,
        )

    if codon_rate_adjustments is not None:
        raise ValueError(
            f"q_variant='{q_variant}' does not accept codon_rate_adjustments; "
            "provide already-adjusted neutral rates for non-global variants."
        )

    if q_variant == "shm1_codon":
        neutral_q = _validate_shm1_codon_inputs(neutral_codon_q)
        return build_mutsel_q_from_codon_q(
            neutral_q,
            aa_selection_factors,
            single_hit_scale=single_hit_scale,
            double_hit_scale=double_hit_scale,
            triple_hit_scale=triple_hit_scale,
        )

    # Default: parent_row_degenerate. Pydantic config validation rejects
    # unknown variants before reaching this dispatcher.
    return _build_mutsel_q_matrices_batched(
        codon_rates,
        codon_csps,
        aa_selection_factors,
        single_hit_scale=single_hit_scale,
        double_hit_scale=double_hit_scale,
        triple_hit_scale=triple_hit_scale,
    )
