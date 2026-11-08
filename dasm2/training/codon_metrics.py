"""Codon-prediction validation metrics.

Computes per-position child-codon prediction metrics on the validation set,
partitioned by whether the true child codon equals the parent codon
(conserved) or differs from it (mutated). Intended to be accumulated across
the validation epoch and reduced once at the end.

Metrics produced (see issue #620):
- ``codon_acc_overall``: top-1 codon accuracy across all valid positions
- ``codon_acc_conserved``: top-1 accuracy on positions with child == parent
- ``codon_acc_mutated``: top-1 accuracy on positions with child != parent
- ``aa_acc_mutated``: top-1 amino-acid accuracy on the mutated partition
- ``mutation_rate``: fraction of valid positions with child != parent
- ``mut_precision``, ``mut_recall``, ``mut_f1``: binary-detection metrics
  for "model predicted any mutation here" vs. "position truly mutated"

The accumulator only handles per-rank reduction. Multi-rank gathering is the
trainer's responsibility (see ``Trainer._gather_val_codon_metrics``), keeping
this module free of compute-backend coupling. The accumulator exposes raw
integer counts via :meth:`CodonPredictionMetrics.counts` so the trainer can
sum them across ranks and compute ratios from the global totals — averaging
per-rank ratios would bias unequal-partition shards.
"""

from __future__ import annotations

from typing import Dict, Protocol, runtime_checkable

import torch
from torch import Tensor

from dasm2.codon_table import get_codon_to_aa_lookup_with_sentinel
from dasm2.sequence.constants import AMBIGUOUS_CODON_IDX

# Sentinel slot appended after the 64 sense+stop codons so that padded
# positions, which carry AMBIGUOUS_CODON_IDX in batch tensors, never produce
# an out-of-bounds index into the AA lookup. The sentinel maps to an AA index
# that no real codon emits, so even if a clamped site somehow slipped past
# the loss_mask its AA comparison would always be False.
_INVALID_AA_SENTINEL = 20

# Names of the integer counts the accumulator carries. Exposed as a constant
# so the trainer's DDP-gather loop can iterate them without re-listing fields.
_COUNT_FIELDS = (
    "n_total",
    "n_correct_total",
    "n_conserved",
    "n_correct_conserved",
    "n_mutated",
    "n_correct_mutated",
    "n_correct_aa_mutated",
    "n_pred_mut",
    "n_pred_mut_and_true_mut",
)


@runtime_checkable
class CodonPredictingLoss(Protocol):
    """Losses that opt into per-position codon-prediction validation metrics.

    The codon :class:`~dasm2.training.metric_providers.MetricProvider`
    resolves its extractor by looking up ``predict_codons`` on the task's
    loss (issue #738); a loss exposing it has its per-batch top-1 codon
    predictions fed into a :class:`CodonPredictionMetrics` accumulator
    during validation. Codon eligibility is loss-intrinsic (not a head
    capability): ``DASMLoss``, ``QDASMLoss``, and ``HybridLoss`` implement
    ``predict_codons``; ``AAOWhichmutLoss`` is AA-level and cannot. This
    protocol documents the extractor contract.
    """

    def predict_codons(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Return the top-1 codon prediction per position, ``(B, L)`` long."""
        ...


def _safe_div(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator`` or 0.0 if the denominator is zero.

    Returning 0.0 for empty partitions keeps the JSON log row complete; the
    paired count denominator on the same row (e.g. ``mutation_rate`` for the
    ``mutated`` partition) tells callers when the ratio is undefined.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _zero_count(device: torch.device) -> Tensor:
    """Construct a 0-d int64 tensor used as a running count."""
    return torch.zeros((), dtype=torch.long, device=device)


class CodonPredictionMetrics:
    """Accumulates per-position codon-prediction counts across a val epoch.

    Counts are stored as int64 tensors on the model's device to avoid overflow
    on large validation sets and to keep the per-batch update on-device. The
    accumulator only requires the model's top-1 codon prediction; it does not
    touch the loss path.

    Two reduction surfaces are exposed:

    - :meth:`counts` returns raw integer counts. Multi-rank gathering must use
      this surface so summing happens before ratios are taken.
    - :meth:`compute` is a single-rank shortcut equivalent to
      ``metrics_from_counts(counts())``.

    Usage::

        metrics = CodonPredictionMetrics(device=trainer.device)
        for batch in val_loader:
            top1 = head.predict_codons(predictions, batch)  # (B, L)
            metrics.update(
                top1_codon=top1,
                parent_codon=batch["codon_parents_idxs"],
                child_codon=batch["codon_children_idxs"],
                valid_mask=batch["loss_mask"],
            )
        metric_values = metrics.compute()  # Dict[str, float]
    """

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self._codon_to_aa = get_codon_to_aa_lookup_with_sentinel(self.device)
        assert self._codon_to_aa.shape[0] == AMBIGUOUS_CODON_IDX + 1

        # Counts over all valid positions
        self.n_total = _zero_count(self.device)
        self.n_correct_total = _zero_count(self.device)
        # Conserved partition: child == parent
        self.n_conserved = _zero_count(self.device)
        self.n_correct_conserved = _zero_count(self.device)
        # Mutated partition: child != parent
        self.n_mutated = _zero_count(self.device)
        self.n_correct_mutated = _zero_count(self.device)
        self.n_correct_aa_mutated = _zero_count(self.device)
        # Binary mutation-detection counts
        self.n_pred_mut = _zero_count(self.device)
        self.n_pred_mut_and_true_mut = _zero_count(self.device)

    def update(
        self,
        top1_codon: Tensor,
        parent_codon: Tensor,
        child_codon: Tensor,
        valid_mask: Tensor,
    ) -> None:
        """Accumulate counts from a single validation batch.

        Args:
            top1_codon: Model's top-1 codon prediction per position, ``(B, L)``
                long tensor.
            parent_codon: Parent codon indices, ``(B, L)`` long tensor.
            child_codon: True child codon indices, ``(B, L)`` long tensor.
            valid_mask: Boolean mask of positions to count, ``(B, L)``.
        """
        if top1_codon.shape != parent_codon.shape:
            raise ValueError(
                "top1_codon and parent_codon must share shape; "
                f"got {top1_codon.shape} vs {parent_codon.shape}"
            )
        if top1_codon.shape != child_codon.shape:
            raise ValueError(
                "top1_codon and child_codon must share shape; "
                f"got {top1_codon.shape} vs {child_codon.shape}"
            )
        if top1_codon.shape != valid_mask.shape:
            raise ValueError(
                "top1_codon and valid_mask must share shape; "
                f"got {top1_codon.shape} vs {valid_mask.shape}"
            )

        # Move predictions / targets onto the accumulator's device once,
        # which avoids host->device round-trips on every comparison and keeps
        # the per-batch update purely on-device.
        top1 = top1_codon.to(device=self.device, dtype=torch.long)
        parent = parent_codon.to(device=self.device, dtype=torch.long)
        child = child_codon.to(device=self.device, dtype=torch.long)
        valid = valid_mask.to(device=self.device).bool()

        codon_correct = top1 == child
        truly_mutated = child != parent
        truly_conserved = ~truly_mutated
        pred_mutated = top1 != parent

        valid_mutated = valid & truly_mutated
        valid_conserved = valid & truly_conserved

        pred_aa = self._codon_to_aa[top1]
        true_aa = self._codon_to_aa[child]
        aa_correct = pred_aa == true_aa

        self.n_total += valid.sum()
        self.n_correct_total += (codon_correct & valid).sum()

        self.n_conserved += valid_conserved.sum()
        self.n_correct_conserved += (codon_correct & valid_conserved).sum()

        self.n_mutated += valid_mutated.sum()
        self.n_correct_mutated += (codon_correct & valid_mutated).sum()
        self.n_correct_aa_mutated += (aa_correct & valid_mutated).sum()

        valid_pred_mut = pred_mutated & valid
        self.n_pred_mut += valid_pred_mut.sum()
        self.n_pred_mut_and_true_mut += (valid_pred_mut & truly_mutated).sum()

    def counts(self) -> Dict[str, int]:
        """Return raw accumulated counts as a flat dict of ints.

        Use this surface for DDP gathering: sum each count across ranks (so
        the global totals are exact), then call :meth:`metrics_from_counts`.
        """
        return {field: int(getattr(self, field).item()) for field in _COUNT_FIELDS}

    @staticmethod
    def metrics_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
        """Compute the final ratio dict from a counts dict.

        Pure function so the trainer can call it on either local counts (no
        DDP) or globally-summed counts (after gathering).
        """
        n_total = counts["n_total"]
        n_correct_total = counts["n_correct_total"]
        n_conserved = counts["n_conserved"]
        n_correct_conserved = counts["n_correct_conserved"]
        n_mutated = counts["n_mutated"]
        n_correct_mutated = counts["n_correct_mutated"]
        n_correct_aa_mutated = counts["n_correct_aa_mutated"]
        n_pred_mut = counts["n_pred_mut"]
        n_pred_mut_and_true_mut = counts["n_pred_mut_and_true_mut"]

        precision = _safe_div(n_pred_mut_and_true_mut, n_pred_mut)
        recall = _safe_div(n_pred_mut_and_true_mut, n_mutated)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "codon_acc_overall": _safe_div(n_correct_total, n_total),
            "codon_acc_conserved": _safe_div(n_correct_conserved, n_conserved),
            "codon_acc_mutated": _safe_div(n_correct_mutated, n_mutated),
            "aa_acc_mutated": _safe_div(n_correct_aa_mutated, n_mutated),
            "mutation_rate": _safe_div(n_mutated, n_total),
            "mut_precision": precision,
            "mut_recall": recall,
            "mut_f1": f1,
        }

    def compute(self) -> Dict[str, float]:
        """Single-rank shortcut: ``metrics_from_counts(counts())``."""
        return self.metrics_from_counts(self.counts())


# The two accuracy fields that survive when the loss_mask is narrowed to
# mutated positions (whichmut). The conserved / overall / detection fields
# CodonPredictionMetrics also emits are degenerate in that regime (no
# conserved positions; precision/recall = 1 by construction), so the whichmut
# provider drops them. Reusing these exact field names keeps a whichmut run
# consistent with DASM in the same dashboard panels (issue #738).
_MUTATED_ACCURACY_KEYS = ("codon_acc_mutated", "aa_acc_mutated")


def mutated_codon_accuracy_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    """Conditional-on-mutation codon/AA accuracy from raw codon counts.

    A thin filter over :meth:`CodonPredictionMetrics.metrics_from_counts`
    selecting only the mutated-partition accuracies. Used by the whichmut
    codon-accuracy provider, whose ``loss_mask`` already restricts every
    counted position to a mutation — so ``codon_acc_mutated`` /
    ``aa_acc_mutated`` are the conditional accuracies and the remaining fields
    carry no information. Shares the accumulator and field names with the DASM
    codon group; the two providers are mutually exclusive per task (gated on
    ``predict_codons`` vs ``predict_child_codons``), so the reused names never
    collide in the trainer's flat per-task merge.

    Note: ``aa_acc_mutated`` (AA of the predicted codon) tends to *equal*
    ``codon_acc_mutated`` for whichmut. The two diverge only on a synonymous
    near-miss -- the argmax landing on a different codon of the observed child's
    AA -- which is rare here: a synonymous sibling carries no selection boost
    (the parent-AA SF column is zapped), and the observed mutation, being
    high-rate, usually already dominates its synonymous group. The two are kept
    separate anyway for schema parity with DASM and to let the gap widen as a
    model learns AA-level selection.
    """
    full = CodonPredictionMetrics.metrics_from_counts(counts)
    return {key: full[key] for key in _MUTATED_ACCURACY_KEYS}
