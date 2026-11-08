"""Selection-factor distribution metrics for the DASM-family heads.

Issue #678 §7 introduced this instrumentation to detect calibration
breakdown early (the focal-cycle-0 regime from #652) rather than at the
final checkpoint. The accumulator mirrors :class:`CodonPredictionMetrics`:
on-device tensor counts, ``counts() / metrics_from_counts()`` for
multi-rank gather, ``compute()`` as the single-rank shortcut.

Six metrics are emitted, all derived from the per-batch ``(B, L, 20)``
post-parent-AA-zap selection-factor tensor:

- ``sf_child_median_nonsyn_1nt`` — median push on the observed child AA
  at single-NT non-syn positions; the central placement signal from #652.
- ``sf_child_median_nonsyn_multihit`` — same for 2-/3-NT non-syn positions.
- ``sf_nonparent_mean_conserved`` — mean SF over the 19 non-parent AAs at
  conserved positions; the calibration signal.
- ``sf_frac_positive`` — fraction of all non-parent SF entries that are
  > 0; the calibration-breakdown canary.
- ``sf_l2_per_pos_mean`` — mean ‖SF‖₂ per position (over 20 entries);
  total push magnitude.
- ``sf_nonparent_p95_abs`` — 95th percentile of ``|SF|`` on non-parent
  entries; tail strength the median hides.

Quantiles are computed from a fixed-range histogram (``[-_SF_HIST_RANGE,
+_SF_HIST_RANGE]`` for child-AA medians, ``[0, _SF_HIST_RANGE]`` for
``|SF|`` p95). The range covers the observed regime (the 3.09-nat flip
threshold in #652 plus headroom) at sub-bin precision (~0.04 nat at the
default bin count). Out-of-range mass is clamped to the boundary bins so
the percentile remains computable even on focal-failure runs that
overshoot.
"""

from __future__ import annotations

from typing import Dict, Protocol, runtime_checkable

import torch
from torch import Tensor

from dasm2.codon_table import get_codon_to_aa_lookup_with_sentinel
from dasm2.sequence.constants import AMBIGUOUS_CODON_IDX, CODONS

# Histogram bounds: |SF| values much above this are pathological. Picked to
# cover the 3.09-nat flip threshold from #652 §4 with headroom.
_SF_HIST_RANGE = 10.0
_SF_HIST_BINS = 256

_FLOAT_FIELDS = (
    "sf_l2_sum",
    "sf_l2_n",
    "sf_nonparent_sum_conserved",
    "sf_nonparent_n_conserved",
    "sf_nonparent_n_positive",
    "sf_nonparent_n_total",
)


def _build_codon_pair_nt_distance() -> torch.Tensor:
    """``(65, 65)`` int8 Hamming distance over codon strings.

    The trailing row/column (index ``AMBIGUOUS_CODON_IDX = 64``) is a
    sentinel that no real position ever lands in: the loss mask filters
    padded positions out upstream, but the lookup still needs a
    representable cell at index 64 so a stray ambiguous codon doesn't
    raise an ``IndexError``. The sentinel cell holds the maximum
    distance (3) — purely a placeholder; downstream bucketing only
    reads it for positions the valid mask excluded.
    """
    n = AMBIGUOUS_CODON_IDX + 1
    table = torch.full((n, n), 3, dtype=torch.int8)
    for i, ci in enumerate(CODONS):
        for j, cj in enumerate(CODONS):
            table[i, j] = sum(1 for a, b in zip(ci, cj) if a != b)
    return table


_CODON_PAIR_NT_DISTANCE_CPU = _build_codon_pair_nt_distance()


def _codon_pair_nt_distance(device: torch.device) -> Tensor:
    if _CODON_PAIR_NT_DISTANCE_CPU.device == device:
        return _CODON_PAIR_NT_DISTANCE_CPU
    return _CODON_PAIR_NT_DISTANCE_CPU.to(device)


@runtime_checkable
class SelectionFactorPredictingLoss(Protocol):
    """Losses that expose a ``(B, L, 20)`` selection-factor tensor.

    The SF :class:`~dasm2.training.metric_providers.MetricProvider` is
    gated on the head's ``SELECTION_FACTOR`` capability, not on the loss
    (issue #738): every SF-producing head emits the tensor, so the head
    output is the SF tensor and an identity head-level fallback supplies
    the extractor when a loss does not override it. A loss may still
    implement this method to customize extraction; ``DASMLoss`` does (an
    identity return documenting the contract). The method should return
    the post-parent-AA-zap tensor — the same tensor the loss itself
    consumes after the head's diagonal zap.
    """

    def selection_factors(
        self, predictions: Tensor, batch: Dict[str, Tensor]
    ) -> Tensor:
        """Return the per-position selection factors, ``(B, L, 20)``."""
        ...


def _zero_count(device: torch.device, dtype: torch.dtype = torch.float64) -> Tensor:
    return torch.zeros((), dtype=dtype, device=device)


class SelectionFactorMetrics:
    """Accumulates selection-factor distribution stats across a val epoch.

    The accumulator is purely on-device: per-batch updates only touch the
    GPU memory holding the histograms and running sums. The trainer can
    therefore call :meth:`update` inside the validation autocast/inference
    block without forcing host syncs.
    """

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self._codon_to_aa = get_codon_to_aa_lookup_with_sentinel(self.device)
        assert self._codon_to_aa.shape[0] == AMBIGUOUS_CODON_IDX + 1
        self._nt_distance = _codon_pair_nt_distance(self.device)

        # Histograms. ``sf_child_*`` are signed; ``sf_nonparent_abs`` is |SF|.
        self._hist_child_nonsyn_1nt = torch.zeros(
            _SF_HIST_BINS, dtype=torch.float64, device=self.device
        )
        self._hist_child_nonsyn_multihit = torch.zeros(
            _SF_HIST_BINS, dtype=torch.float64, device=self.device
        )
        self._hist_nonparent_abs = torch.zeros(
            _SF_HIST_BINS, dtype=torch.float64, device=self.device
        )

        # Scalar running sums — declared explicitly so static checkers can
        # see them; ``_FLOAT_FIELDS`` is the iteration source for ``counts()``.
        self.sf_l2_sum: Tensor = _zero_count(self.device)
        self.sf_l2_n: Tensor = _zero_count(self.device)
        self.sf_nonparent_sum_conserved: Tensor = _zero_count(self.device)
        self.sf_nonparent_n_conserved: Tensor = _zero_count(self.device)
        self.sf_nonparent_n_positive: Tensor = _zero_count(self.device)
        self.sf_nonparent_n_total: Tensor = _zero_count(self.device)

    def update(
        self,
        sf: Tensor,
        parent_codon: Tensor,
        child_codon: Tensor,
        valid_mask: Tensor,
    ) -> None:
        """Accumulate one validation batch.

        Args:
            sf: Post-zap selection factors, ``(B, L, 20)``.
            parent_codon: ``(B, L)`` long.
            child_codon: ``(B, L)`` long.
            valid_mask: ``(B, L)`` bool — the loss mask for that batch.
        """
        if sf.dim() != 3 or sf.shape[-1] != 20:
            raise ValueError(f"sf must be (B, L, 20); got shape {tuple(sf.shape)}")
        if sf.shape[:2] != parent_codon.shape:
            raise ValueError(
                "sf and parent_codon must share (B, L); "
                f"got {tuple(sf.shape)} vs {tuple(parent_codon.shape)}"
            )
        if parent_codon.shape != child_codon.shape:
            raise ValueError(
                "parent_codon and child_codon must share shape; "
                f"got {tuple(parent_codon.shape)} vs {tuple(child_codon.shape)}"
            )
        if parent_codon.shape != valid_mask.shape:
            raise ValueError(
                "parent_codon and valid_mask must share shape; "
                f"got {tuple(parent_codon.shape)} vs {tuple(valid_mask.shape)}"
            )

        sf = sf.to(self.device, dtype=torch.float32)
        parent = parent_codon.to(self.device, dtype=torch.long)
        child = child_codon.to(self.device, dtype=torch.long)
        valid = valid_mask.to(self.device).bool()

        if not valid.any():
            return

        parent_aa = self._codon_to_aa[parent]
        child_aa = self._codon_to_aa[child]
        nt_dist = self._nt_distance[parent, child].to(torch.long)

        truly_conserved = parent == child
        truly_mutated = ~truly_conserved
        is_nonsyn = parent_aa != child_aa
        # Position-bucket masks, all over (B, L).
        m_conserved = valid & truly_conserved
        m_nonsyn_1nt = valid & truly_mutated & is_nonsyn & (nt_dist == 1)
        m_nonsyn_multi = valid & truly_mutated & is_nonsyn & (nt_dist >= 2)

        # === sf_child_*: SF at the child AA, per-bucket histogram ===
        # Gather requires non-negative indices everywhere, so substitute the
        # invalid sentinel (which we never actually read) with 0; the bucket
        # masks ensure we only consume entries where child_aa is a real AA.
        gather_idx = torch.where(child_aa < 20, child_aa, child_aa.new_zeros(()))
        sf_at_child = sf.gather(2, gather_idx.unsqueeze(-1)).squeeze(-1)
        self._add_signed_hist(self._hist_child_nonsyn_1nt, sf_at_child[m_nonsyn_1nt])
        self._add_signed_hist(
            self._hist_child_nonsyn_multihit, sf_at_child[m_nonsyn_multi]
        )

        # === sf_nonparent_*: SF over non-parent AAs at every valid pos ===
        # Mask off the parent-AA column (its post-zap value is constant 0,
        # not a learned push) before reducing.
        aa_idx_3d = torch.arange(20, device=self.device).view(1, 1, 20)
        parent_safe = torch.where(parent_aa < 20, parent_aa, parent_aa.new_zeros(()))
        is_parent_aa = aa_idx_3d == parent_safe.unsqueeze(-1)
        nonparent_mask_3d = valid.unsqueeze(-1) & ~is_parent_aa
        nonparent_sf = sf[nonparent_mask_3d]
        if nonparent_sf.numel() > 0:
            self.sf_nonparent_n_total += nonparent_sf.numel()
            self.sf_nonparent_n_positive += (nonparent_sf > 0).sum().to(torch.float64)
            self._add_unsigned_hist(self._hist_nonparent_abs, nonparent_sf.abs())

        # === sf_nonparent_mean_conserved ===
        if m_conserved.any():
            cons_nonparent_3d = m_conserved.unsqueeze(-1) & ~is_parent_aa
            cons_vals = sf[cons_nonparent_3d]
            self.sf_nonparent_sum_conserved += cons_vals.sum().to(torch.float64)
            self.sf_nonparent_n_conserved += cons_vals.numel()

        # === sf_l2_per_pos_mean ===
        if valid.any():
            l2 = sf.norm(dim=-1)  # (B, L)
            self.sf_l2_sum += l2[valid].sum().to(torch.float64)
            self.sf_l2_n += valid.sum().to(torch.float64)

    @staticmethod
    def _add_signed_hist(hist: Tensor, values: Tensor) -> None:
        """Histogram ``values`` into a signed [-R, +R] bucketing."""
        if values.numel() == 0:
            return
        bin_idx = SelectionFactorMetrics._signed_bin_indices(values)
        contrib = torch.bincount(bin_idx, minlength=_SF_HIST_BINS).to(hist.dtype)
        hist += contrib

    @staticmethod
    def _add_unsigned_hist(hist: Tensor, values: Tensor) -> None:
        """Histogram non-negative ``values`` into [0, +R] bucketing."""
        if values.numel() == 0:
            return
        bin_idx = SelectionFactorMetrics._unsigned_bin_indices(values)
        contrib = torch.bincount(bin_idx, minlength=_SF_HIST_BINS).to(hist.dtype)
        hist += contrib

    @staticmethod
    def _signed_bin_indices(values: Tensor) -> Tensor:
        clamped = values.clamp(min=-_SF_HIST_RANGE, max=_SF_HIST_RANGE)
        # Map [-R, +R] → [0, BINS); clamp into the last bin only for the
        # exact-boundary case so floats at +R don't escape the array.
        scaled = (clamped + _SF_HIST_RANGE) / (2.0 * _SF_HIST_RANGE)
        idx = (scaled * _SF_HIST_BINS).floor().to(torch.long)
        return idx.clamp(min=0, max=_SF_HIST_BINS - 1)

    @staticmethod
    def _unsigned_bin_indices(values: Tensor) -> Tensor:
        clamped = values.clamp(min=0.0, max=_SF_HIST_RANGE)
        scaled = clamped / _SF_HIST_RANGE
        idx = (scaled * _SF_HIST_BINS).floor().to(torch.long)
        return idx.clamp(min=0, max=_SF_HIST_BINS - 1)

    @staticmethod
    def _hist_quantile(hist: Tensor, q: float, lo: float, hi: float) -> float:
        """Linear quantile from a uniform-bin histogram on ``[lo, hi]``.

        Used for both signed (``lo = -R``, ``hi = +R``) and unsigned
        (``lo = 0``, ``hi = +R``) histograms — the only difference is
        the value-range mapping. Returns 0.0 on empty histograms so
        the metric dict stays complete on cold-start eval windows.
        """
        total = float(hist.sum().item())
        if total <= 0.0:
            return 0.0
        cumsum = torch.cumsum(hist, dim=0)
        target = q * total
        idx = int(
            torch.searchsorted(cumsum, torch.tensor([target], device=hist.device))[
                0
            ].item()
        )
        idx = max(0, min(_SF_HIST_BINS - 1, idx))
        bin_width = (hi - lo) / _SF_HIST_BINS
        return lo + (idx + 0.5) * bin_width

    def counts(self) -> Dict[str, float]:
        """Return raw accumulated values + histograms, flattened.

        Histogram bins are returned with stable keys so the trainer's DDP
        gather can sum bins across ranks before reducing to quantiles.
        """
        out: Dict[str, float] = {
            field: float(getattr(self, field).item()) for field in _FLOAT_FIELDS
        }
        for i, count in enumerate(self._hist_child_nonsyn_1nt.tolist()):
            out[f"hist_child_1nt_{i}"] = float(count)
        for i, count in enumerate(self._hist_child_nonsyn_multihit.tolist()):
            out[f"hist_child_multi_{i}"] = float(count)
        for i, count in enumerate(self._hist_nonparent_abs.tolist()):
            out[f"hist_nonparent_abs_{i}"] = float(count)
        return out

    @staticmethod
    def metrics_from_counts(counts: Dict[str, float]) -> Dict[str, float]:
        """Pure function: reduce raw counts/histograms to the final metric dict.

        Symmetric with :meth:`CodonPredictionMetrics.metrics_from_counts`;
        the trainer can call it on local counts (single-rank shortcut) or
        on globally-summed counts (DDP path).
        """

        def _get_hist(prefix: str) -> Tensor:
            bins = [counts.get(f"{prefix}{i}", 0.0) for i in range(_SF_HIST_BINS)]
            return torch.tensor(bins, dtype=torch.float64)

        hist_1nt = _get_hist("hist_child_1nt_")
        hist_multi = _get_hist("hist_child_multi_")
        hist_abs = _get_hist("hist_nonparent_abs_")

        n_total = counts.get("sf_nonparent_n_total", 0.0)
        n_positive = counts.get("sf_nonparent_n_positive", 0.0)
        sf_l2_n = counts.get("sf_l2_n", 0.0)
        sf_l2_sum = counts.get("sf_l2_sum", 0.0)
        sf_cons_n = counts.get("sf_nonparent_n_conserved", 0.0)
        sf_cons_sum = counts.get("sf_nonparent_sum_conserved", 0.0)

        signed = (-_SF_HIST_RANGE, _SF_HIST_RANGE)
        unsigned = (0.0, _SF_HIST_RANGE)
        return {
            "sf_child_median_nonsyn_1nt": SelectionFactorMetrics._hist_quantile(
                hist_1nt, 0.5, *signed
            ),
            "sf_child_median_nonsyn_multihit": SelectionFactorMetrics._hist_quantile(
                hist_multi, 0.5, *signed
            ),
            "sf_nonparent_mean_conserved": (
                sf_cons_sum / sf_cons_n if sf_cons_n > 0 else 0.0
            ),
            "sf_frac_positive": n_positive / n_total if n_total > 0 else 0.0,
            "sf_l2_per_pos_mean": sf_l2_sum / sf_l2_n if sf_l2_n > 0 else 0.0,
            "sf_nonparent_p95_abs": SelectionFactorMetrics._hist_quantile(
                hist_abs, 0.95, *unsigned
            ),
        }

    def compute(self) -> Dict[str, float]:
        """Single-rank shortcut: ``metrics_from_counts(counts())``."""
        return self.metrics_from_counts(self.counts())
