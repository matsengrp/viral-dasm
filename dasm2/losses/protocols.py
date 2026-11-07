"""Cross-cutting loss protocols consumed by the trainer.

These are structural (duck-typed) contracts: a loss opts in simply by
defining the method, with no base class to inherit. The trainer detects
participation via ``isinstance(task.loss, <Protocol>)`` thanks to
``@runtime_checkable``.
"""

from typing import Dict, Protocol, Tuple, runtime_checkable

from torch import Tensor


@runtime_checkable
class SelfNormalizingLoss(Protocol):
    """A loss whose normalization label count differs from ``loss_mask.sum()``.

    Joint training normalizes each task by its own window-total label count
    (issue #688). The default count is ``batch['loss_mask'].sum()`` (resolved
    by ``Trainer._get_n_loss_positions``). A loss whose "events" are a strict
    subset of ``loss_mask`` — e.g. CSP, defined only at substituted
    sites while sharing DASM's broad evolutionary mask — implements this to
    report its own label count so that ``loss_weight`` stays a clean relative
    weight rather than silently carrying a label-count ratio.

    The returned count is the denominator the trainer divides this task's
    sum-reduced loss by; it must use the same unit as the loss's own
    sum-reduction (one count per term the loss sums over).

    Called during the trainer's mask-only counting pass, *before*
    ``move_batch_to_device`` — so implementations must read only label/mask
    counts and must not assume tensors are on the training device.

    .. note::

       ``@runtime_checkable`` only verifies that ``n_loss_labels`` *exists* on
       the loss; it does not check its signature. A misnamed argument or wrong
       arity will pass ``isinstance`` and fail at the first call site instead.

    Example:
        A loss whose events are the strict subset of ``loss_mask`` sites at
        which the child codon actually substitutes::

            class MyCSPLoss:
                def forward(self, predictions, batch):
                    ...  # sum-reduce over substituted sites

                def n_loss_labels(self, batch) -> int:
                    return int(
                        (batch["loss_mask"] & batch["is_substituted"]).sum()
                    )

        ``isinstance(MyCSPLoss(), SelfNormalizingLoss)`` is now ``True`` and
        the trainer will divide its sum-loss by this count.
    """

    def n_loss_labels(self, batch: Dict[str, Tensor]) -> int:
        """Return this loss's label (event) count for ``batch`` (CPU tensors)."""
        ...


@runtime_checkable
class CodonSamplingLoss(Protocol):
    """A loss that can produce, and therefore sample from, its own codon kernel.

    ``log_codon_probs`` returns per-site log P(child codon | parent codon, t)
    for all 64 codons, ``[B, L, 64]``, each site's row normalized over that
    site alone and carrying zero stop-codon mass. Sites are conditionally
    independent given the parent — the contract a per-site multinomial draw
    requires.

    Three clauses make up the contract, and all three are load-bearing:

    1. **Per-site normalization.** ``exp(log_codon_probs)[b, l].sum() == 1``
       at every position the batch's ``loss_mask`` selects. This is what
       makes "any conforming loss can be sampled" a true statement rather
       than a hopeful one.

       One caveat for the mutsel construction under its default
       ``normalization="parent_absorbing"`` (``DASMLoss``, and
       ``HybridLoss``' selection half): it normalizes by letting the
       *parent* codon absorb ``1 - sum(non-parent)``
       (``dasm2/molevol/tensor_ops.py::set_parent_codon_log_prob``), so the
       clause holds only while the non-parent mass stays below 1. Real
       trained log selection factors are strongly negative (empirically
       ≈ -4.6 ± 3.4 across amino acids), so it holds comfortably in
       practice, but a caller feeding in large *positive* factors can leave
       a row summing above 1. QDASM's ``exp(tQ)`` rows are normalized by
       construction and have no such condition.
       ``DASMLoss(normalization="sense_conditional")`` is the
       unconditional mutsel form: one ``log_softmax`` over the
       sense-restricted weight row, so the clause holds for any finite
       selection factors.
    2. **Zero stop-codon mass.** The 3 stop-codon columns carry probability
       zero, so a sampled child never contains a stop codon. Under
       ``parent_absorbing`` this is zero to within the ``exp(-80)`` clamp
       floor; under ``sense_conditional`` it is exactly 0.0.
    3. **Conditional independence across sites.** Sequence-level
       competing-risks models cannot satisfy this: ``WhichmutLoss``
       normalizes a per-mutation score by a per-sequence partition function
       summed over every structurally valid position, so there is no
       per-site normalized ``[B, L, 64]`` to draw from. ``predict_codons``
       escapes that because an argmax is normalization-free; a multinomial
       draw does not. Whichmut needs a *sibling* contract (sample event
       count and locations, then the substitution), not a stretched version
       of this one — see #192. ``AAOWhichmutLoss`` is further out still,
       being AA-level.

    Positions the batch masks out (padding, separators, ambiguous parent
    codons) are *not* covered by clause 1: implementations may emit
    anything there. The simulator normalizes those to NaN itself, via the
    shared masking adaptor
    (:func:`dasm2.molevol.simulation.nan_fill_invalid_codon_probs`), so
    that :func:`dasm2.molevol.simulation.sample_sequence_from_codon_probs`
    emits ``NNN``.

    ``PREDICT_REQUIRED_KEYS`` names every batch key ``log_codon_probs``
    reads. It is the loss's own declaration of what a caller must assemble,
    and the simulator's batch builder drives its transform chain off it
    (``dasm2/simulation.py::_create_sampling_batch``). A loss whose
    key set depends on its configuration (e.g. ``QDASMLoss`` under
    a global-kmer ``q_variant``) must compute the property-aware union in
    ``__init__`` rather than declaring a static class-level tuple.

    Conformance is structural: a loss opts in by defining the method and
    the attribute, with no base class to inherit. Every implementer is
    covered automatically by the parameterized contract tests in
    ``tests/losses/test_codon_sampling_protocol.py``.
    """

    PREDICT_REQUIRED_KEYS: Tuple[str, ...]

    def log_codon_probs(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Return ``[B, L, 64]`` log P(child codon | parent codon, t)."""
        ...
