"""Capability-driven validation-metric providers (issue #738).

Validation metric groups (child-codon prediction, selection-factor
distribution) used to be gated by ``isinstance`` checks against the
concrete loss class, threaded through the trainer and logger as a
separate optional kwarg per group. That conflated *metric eligibility*
(a property of the tensor the head produces) with *the loss that happens
to consume the tensor*, so a Whichmut or QDASM run silently dropped
metrics it could perfectly well emit.

This module replaces the per-group gate/accumulate plumbing with one
iterable registry. Each :class:`MetricProvider` declares how to gate,
build, extract, and reduce its group; the trainer iterates the registry
once instead of special-casing each group in five places.

Two gating shapes coexist behind one uniform loop:

* **Head-capability-gated** (selection factor): eligibility is a genuine
  property of the head's output space, declared via
  ``head.OUTPUT_CAPABILITIES`` (see :class:`dasm2.heads.base.Capability`).
  The loss supplies the extractor when it has one; otherwise the head
  output *is* the tensor, so an identity head-level fallback applies. This
  is what closes the QDASM / Hybrid / Whichmut / AAO-Whichmut SF gaps.
* **Loss-extractor-gated** (codon distribution): eligibility is
  loss-intrinsic — ``AAOWhichmutLoss`` is AA-level and cannot produce a
  codon distribution even though it shares a head with codon-capable
  losses — so the provider gates on the loss exposing ``predict_codons``.
  Behavior is identical to the pre-#738 ``isinstance(loss, ...)`` gate.

Either shape can be narrowed by loss class. ``excluded_loss_classes`` removes
a provider for losses whose batches lack the keys its ``update`` reads (the
``sf`` group, which reads codon keys, is excluded for
``DMSMatrixCalibrationLoss``). It is a narrowing on top of the capability or
extractor gate, never a replacement for it.

See ``DESIGN.md`` ("Eval-metric gating") for the taxonomy and the
cross-loss non-comparability caveat on ``val_sf_*``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import torch
from torch import Tensor

from dasm2.heads.base import Capability
from dasm2.heads.factory import HEAD_REGISTRY
from dasm2.losses.dms_matrix import DMSMatrixCalibrationLoss
from dasm2.training.codon_metrics import (
    CodonPredictionMetrics,
    mutated_codon_accuracy_from_counts,
)
from dasm2.training.sf_metrics import SelectionFactorMetrics

if TYPE_CHECKING:
    from dasm2.training.task import Task

# (predictions, batch) -> metric tensor consumed by the accumulator.
Extractor = Callable[[Tensor, Dict[str, Tensor]], Tensor]
# The trainer's ``self.device`` is a string or ``torch.device``; accumulator
# constructors accept either.
DeviceLike = Union[str, torch.device]


@dataclass(frozen=True)
class MetricProvider:
    """One validation metric group: how to gate, build, extract, and reduce it.

    Attributes:
        name: Group namespace, e.g. ``"codon"`` / ``"sf"``. Used only for
            internal bookkeeping; the flattened JSONL keys come from the
            accumulator's metric names, not this prefix.
        loss_method: Name of the loss method that supplies the extractor
            (``"predict_codons"`` / ``"selection_factors"``).
        requires_capability: When set, the provider is eligible only for
            tasks whose head declares this capability (head-driven gate).
            When ``None``, eligibility falls back to the loss exposing
            ``loss_method`` (loss-driven gate).
        head_fallback: When the loss does not expose ``loss_method``, use
            an identity extractor (the head output *is* the tensor). Valid
            only for capability-gated providers whose head guarantees the
            output space.
        make_accumulator: Builds a fresh per-task accumulator on a device.
        update: Feeds one batch's extracted tensor into the accumulator.
        metrics_from_counts: Pure reduction from summed counts to the
            final metric dict (shared with the DDP gather path).
        counts_are_int: Whether ``accumulator.counts()`` are integer
            counters (codon) or carry fractional histogram mass (sf); read
            by the trainer's DDP gather to choose the round-trip dtype.
        excluded_loss_classes: Loss classes whose batches lack the keys this
            group's ``update`` reads, so the group must not run for them even
            when the head declares the capability.
    """

    name: str
    loss_method: str
    requires_capability: Optional[Capability]
    head_fallback: bool
    make_accumulator: Callable[[DeviceLike], Any]
    update: Callable[[Any, Tensor, Dict[str, Tensor]], None]
    metrics_from_counts: Callable[[Dict[str, Any]], Dict[str, float]]
    counts_are_int: bool
    excluded_loss_classes: Tuple[type, ...] = ()

    def __post_init__(self) -> None:
        """Reject head-level fallback on a loss-gated provider.

        ``head_fallback`` returns the raw head output as the metric tensor
        when the loss has no extractor; that is only sound when the head's
        declared capability guarantees the output space. A loss-gated
        provider (``requires_capability is None``) has no such guarantee, so
        the combination would silently feed arbitrary predictions into the
        accumulator. Fail loud at registry-build time instead.
        """
        if self.head_fallback and self.requires_capability is None:
            raise ValueError(
                f"MetricProvider {self.name!r}: head_fallback=True is only valid "
                "for capability-gated providers (requires_capability must be set)."
            )

    def is_eligible(self, task: "Task") -> bool:
        """Whether this group should be emitted for ``task``.

        Capability-gated providers read the head's declared output spaces;
        loss-gated providers read whether the loss exposes the extractor.
        ``excluded_loss_classes`` narrows either shape.
        """
        if isinstance(task.loss, self.excluded_loss_classes):
            return False
        if self.requires_capability is not None:
            return (
                self.requires_capability
                in HEAD_REGISTRY[task.head_type].OUTPUT_CAPABILITIES
            )
        return callable(getattr(task.loss, self.loss_method, None))

    def resolve_extractor(self, task: "Task") -> Optional[Extractor]:
        """Resolve the extractor for ``task``, or ``None`` if unavailable.

        Prefers the loss method (it knows the forward model that turns raw
        head output into the metric tensor); falls back to the identity
        head output when the provider permits it. ``None`` means an
        eligible capability has no way to produce the tensor — the trainer
        turns that into a fail-loud error at task-build time.
        """
        method = getattr(task.loss, self.loss_method, None)
        if callable(method):
            return cast(Extractor, method)
        if self.head_fallback:
            return _identity_extractor
        return None


def _identity_extractor(predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
    """Head output *is* the selection-factor tensor (already diagonal-zapped).

    Defined as a named module-level function rather than a lambda so that
    ``resolve_extractor`` returns a stable, identity-comparable object. The
    test ``test_dasm_sf_extractor_prefers_loss_method`` relies on this:
    it asserts the resolved extractor *is* the bound loss method (not this
    fallback), which only works if the fallback is a fixed identity.
    """
    return predictions


def _update_codon(
    accumulator: CodonPredictionMetrics, tensor: Tensor, batch: Dict[str, Tensor]
) -> None:
    accumulator.update(
        top1_codon=tensor,
        parent_codon=batch["codon_parents_idxs"],
        child_codon=batch["codon_children_idxs"],
        valid_mask=batch["loss_mask"],
    )


def _update_sf(
    accumulator: SelectionFactorMetrics, tensor: Tensor, batch: Dict[str, Tensor]
) -> None:
    accumulator.update(
        sf=tensor,
        parent_codon=batch["codon_parents_idxs"],
        child_codon=batch["codon_children_idxs"],
        valid_mask=batch["loss_mask"],
    )


# Order is the emission order. Within any single task at most one of the two
# codon-accuracy providers is eligible — ``codon`` gates on the loss exposing
# ``predict_codons`` (DASM/QDASM/Hybrid), ``whichmut_codon`` on it exposing
# ``predict_child_codons`` (whichmut) — so although they deliberately share the
# ``*_mutated`` field names, the trainer's flat per-task merge never sees both.
# ``sf`` keys (``sf_*``) are disjoint from both, so it can co-emit with either.
METRIC_PROVIDERS: List[MetricProvider] = [
    MetricProvider(
        name="codon",
        loss_method="predict_codons",
        requires_capability=None,
        head_fallback=False,
        make_accumulator=lambda device: CodonPredictionMetrics(device=device),
        update=_update_codon,
        metrics_from_counts=CodonPredictionMetrics.metrics_from_counts,
        counts_are_int=True,
    ),
    MetricProvider(
        name="whichmut_codon",
        loss_method="predict_child_codons",
        requires_capability=None,
        head_fallback=False,
        make_accumulator=lambda device: CodonPredictionMetrics(device=device),
        update=_update_codon,
        metrics_from_counts=mutated_codon_accuracy_from_counts,
        counts_are_int=True,
    ),
    MetricProvider(
        name="sf",
        loss_method="selection_factors",
        requires_capability=Capability.SELECTION_FACTOR,
        head_fallback=True,
        make_accumulator=lambda device: SelectionFactorMetrics(device=device),
        update=_update_sf,
        metrics_from_counts=SelectionFactorMetrics.metrics_from_counts,
        counts_are_int=False,
        # The sf update reads codon keys a dense DMS-matrix batch does not have.
        excluded_loss_classes=(DMSMatrixCalibrationLoss,),
    ),
]


@dataclass
class ActiveMetric:
    """A provider bound to a freshly built accumulator and resolved extractor."""

    provider: MetricProvider
    accumulator: Any
    extractor: Extractor

    def observe(self, predictions: Tensor, batch: Dict[str, Tensor]) -> None:
        """Extract this group's tensor from a batch and fold it into the accumulator."""
        tensor = self.extractor(predictions, batch)
        self.provider.update(self.accumulator, tensor, batch)


def build_active_metrics(
    task: "Task",
    device: DeviceLike,
    providers: Optional[List[MetricProvider]] = None,
) -> List[ActiveMetric]:
    """Build the validation metric accumulators eligible for ``task``.

    Iterates the provider registry (``providers`` defaults to the module
    :data:`METRIC_PROVIDERS`; the override exists for tests), skipping
    ineligible groups. For each eligible group an extractor must resolve
    (from the loss or the head fallback); a declared capability with no
    resolvable extractor is a wiring bug and raises here — at task-build
    time — rather than silently omitting the metric at eval time (the
    failure mode #738 fixes).
    """
    active: List[ActiveMetric] = []
    for provider in providers if providers is not None else METRIC_PROVIDERS:
        if not provider.is_eligible(task):
            continue
        extractor = provider.resolve_extractor(task)
        if extractor is None:
            raise ValueError(
                f"Head {task.head_type.value!r} declares capability "
                f"{provider.requires_capability} (metric group {provider.name!r}), "
                f"but its loss {type(task.loss).__name__!r} does not expose "
                f"{provider.loss_method!r} and the provider has no head-level "
                f"fallback. Add the extractor method to the loss or remove the "
                f"capability from the head."
            )
        active.append(
            ActiveMetric(provider, provider.make_accumulator(device), extractor)
        )
    return active
