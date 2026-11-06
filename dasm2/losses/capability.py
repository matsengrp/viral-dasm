"""Resolve a checkpoint's generative codon kernel by loss-level capability.

The simulator, the O/E scorer and the branch-length optimizer all need the
same three answers from a loaded checkpoint: *which head do I run*, *which
loss owns the kernel*, and *how do I turn selection factors into codon
probabilities*. Before issue #1092 each caller answered them with its own
head-type frozenset, which meant adding a simulable head required editing
every caller and forgetting one was a silent capability gap.

Here the answers are derived from
:class:`~dasm2.losses.protocols.CodonSamplingLoss` conformance instead. A
head is simulable exactly when the task loss registered for it implements
``log_codon_probs``; forgetting to implement the method is a named error at
resolution time rather than a wrong kernel at run time.

This module deliberately imports ``CoreModel`` only under ``TYPE_CHECKING``:
``dasm2.models.core`` reaches ``dasm2.losses`` through
``heads.structural_guidance``, so a runtime import here would close an
import cycle. Nothing at run time needs more than ``model.prediction_heads``
and ``model.task_losses``.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypeGuard,
    Union,
)

import torch
from torch import Tensor, nn

from dasm2.heads.types import DASM_LOSS_HEAD_TYPES, HeadType
from dasm2.losses.dasm import DASMLoss
from dasm2.losses.hybrid import HybridLoss
from dasm2.losses.protocols import CodonSamplingLoss
from dasm2.losses.qdasm import QDASMLoss
from dasm2.molevol.mutsel import DASMNormalization
from dasm2.utils.batch_operations import move_batch_to_device

if TYPE_CHECKING:
    from dasm2.models.core import CoreModel

# ``(batch, log_selection_factors) -> [B, L, 64]`` log codon probabilities.
CodonProbsFn = Callable[[Dict[str, Tensor], Tensor], Tensor]


class MissingTaskLossError(ValueError):
    """No task loss is registered for the requested head.

    A ``ValueError`` subclass so callers written against the older,
    undifferentiated behaviour keep working unchanged. It exists as its own
    type so :func:`is_simulable_head` can swallow "this head simply has no
    loss" without also swallowing :class:`AmbiguousTaskLossError`.
    """


class AmbiguousTaskLossError(ValueError):
    """Several task losses are registered for one head; the lookup cannot choose.

    Never swallowed on a scoring or head-resolution path: a head with two
    candidate losses is a misconfiguration to surface by name, not a head to
    silently skip. Only :func:`dasm_normalization_label` and the
    error-message helper ``_loss_class_names`` catch it, since neither scores
    anything. Callers whose signature accepts an explicit ``loss`` argument
    should pass it to disambiguate.
    """


def lookup_task_loss_for_head(model: "CoreModel", head_type: HeadType) -> nn.Module:
    """Locate the single task loss registered on ``model`` for ``head_type``.

    Shared by every caller that needs to resolve a loss object off
    ``model.task_losses`` when it wasn't passed one explicitly —
    :func:`resolve_sampling_loss` here, and
    ``optimize_branch_lengths_for_dataset_qdasm`` in
    ``dasm2/training/hooks/branch_lengths.py``.

    Raises:
        MissingTaskLossError: If the model carries no ``task_losses`` at
            all, or none registered for ``head_type``.
        AmbiguousTaskLossError: If more than one loss is registered for
            ``head_type`` and this lookup cannot pick between them.
    """
    task_losses = getattr(model, "task_losses", None)
    if not task_losses:
        raise MissingTaskLossError(
            f"No task_losses registered on the model; cannot locate the "
            f"{head_type.value} loss needed for branch-length optimization, "
            f"simulation, or OE evaluation. The trainer should have registered "
            f"task losses before this call — if the caller you're using accepts "
            f"an explicit ``loss`` argument, pass it directly instead of "
            f"relying on this lookup."
        )
    # ``task_loss_state_dict_key`` uses ``head_type.value`` when only one
    # task uses this head; for the multi-loss-per-head case (currently AAH
    # only) callers whose signature accepts an explicit ``loss`` argument
    # (e.g. optimize_branch_lengths_for_dataset_qdasm) must pass it directly
    # to disambiguate — this lookup cannot.
    candidate_keys = [
        k
        for k in task_losses
        if k == head_type.value or k.startswith(f"{head_type.value}__")
    ]
    if not candidate_keys:
        raise MissingTaskLossError(
            f"No task loss registered for head {head_type.value}; "
            f"task_losses keys: {list(task_losses.keys())}"
        )
    if len(candidate_keys) > 1:
        raise AmbiguousTaskLossError(
            f"Multiple task losses registered for head {head_type.value}: "
            f"{candidate_keys}. This lookup cannot disambiguate between "
            f"them; if the caller accepts an explicit ``loss`` argument, "
            f"pass it directly."
        )
    resolved_loss: nn.Module = task_losses[candidate_keys[0]]
    return resolved_loss


def resolve_dasm_normalization(
    model: "CoreModel",
    head_type: HeadType,
    loss: Optional[nn.Module] = None,
) -> DASMNormalization:
    """Which DASM endpoint kernel a checkpoint was trained under (issue #1238).

    The single resolution point for every consumer that runs
    ``compute_log_adjusted_codon_probs`` on a *checkpoint's* selection
    factors — branch-length refitting, the between-cycle nuisance-parameter
    hooks, ``dasm2 likelihood``, ``dasm2 surprise``. Leaving any of them on
    the native default would silently score a conditionally-trained
    checkpoint under the wrong likelihood, which is the #1092 failure mode.

    Three cases have to be got right, and getting them wrong in three
    places independently is why this is one function:

    1. **An explicit ``loss`` wins.** :func:`lookup_task_loss_for_head`
       cannot disambiguate a head carrying several losses, so a caller that
       already holds the right object must be able to say so.
    2. **A model with no registered task losses resolves to the native
       default, not an error.** Existing callers legitimately pass models
       that carry no ``task_losses``, so an unconditional lookup would
       widen this path's failure surface.
       :class:`AmbiguousTaskLossError` still propagates: two candidate
       losses is a misconfiguration to name, not to paper over.
    3. **Narrow by class before reading the attribute.**
       ``DASM_LOSS_HEAD_TYPES`` spans four heads and the loss registered for
       one of them need not be a :class:`DASMLoss`. ``QDASMLoss`` carries an
       attribute of the *same name* with a disjoint value vocabulary
       (``"proportional"``), so a ``getattr(loss, "normalization", ...)`` on
       an unnarrowed loss would hand a DASM path a value that is not a valid
       DASM kernel — and the ``Literal`` annotation would not catch it.

    Args:
        model: Loaded ``CoreModel``, consulted only when ``loss`` is None.
        head_type: The head whose loss owns the kernel.
        loss: The task loss, when the caller already has it.

    Returns:
        The resolved normalization; ``"parent_absorbing"`` whenever no
        :class:`DASMLoss` can be resolved.
    """
    if loss is None:
        try:
            loss = lookup_task_loss_for_head(model, head_type)
        except MissingTaskLossError:
            return "parent_absorbing"
    if isinstance(loss, DASMLoss):
        return loss.normalization
    return "parent_absorbing"


def dasm_normalization_label(
    model: "CoreModel",
    head_type: HeadType,
    loss: Optional[nn.Module] = None,
) -> Union[DASMNormalization, Literal["n/a"]]:
    """The DASM kernel to *record* next to a checkpoint's outputs (issue #1252).

    :func:`resolve_dasm_normalization` feeds a scoring path, which needs some
    DASM kernel, so it falls back to ``"parent_absorbing"``. A label must not:
    a QDASM or Hybrid checkpoint, or a head with no task loss, has no DASM
    kernel, and recording the fallback would claim one.

    Args:
        model: Loaded ``CoreModel``, consulted only when ``loss`` is None.
        head_type: The head whose loss owns the kernel.
        loss: The task loss, when the caller already has it.

    Unlike the scoring resolver, this swallows :class:`AmbiguousTaskLossError`
    too: a head with several task losses (the AAH multi-loss case) has no
    single DASM kernel to record, and a command that only wants the label
    (``functional-eval``) must not abort over it.

    Returns:
        The :class:`DASMLoss`'s normalization, or ``"n/a"`` when the head's
        loss is not a :class:`DASMLoss`, is not registered, or is one of
        several registered for the head.
    """
    if loss is None:
        try:
            loss = lookup_task_loss_for_head(model, head_type)
        except (MissingTaskLossError, AmbiguousTaskLossError):
            return "n/a"
    if isinstance(loss, DASMLoss):
        return loss.normalization
    return "n/a"


def is_simulable_head(model: "CoreModel", head_type: HeadType) -> bool:
    """Whether ``head_type``'s task loss satisfies ``CodonSamplingLoss``.

    Returns False rather than raising when no loss is registered for the
    head — a head with no loss simply isn't a candidate, which is what
    :func:`resolve_simulable_head` needs when scanning a joint checkpoint.

    :class:`AmbiguousTaskLossError` deliberately propagates: a head with
    several candidate losses is a misconfiguration, and reporting it as
    "not a candidate" would turn a named resolution-time error into a
    silent capability gap — the very failure mode this module exists to
    remove.
    """
    try:
        loss = lookup_task_loss_for_head(model, head_type)
    except MissingTaskLossError:
        return False
    return isinstance(loss, CodonSamplingLoss)


def resolve_simulable_head(model: "CoreModel", *, command: str) -> HeadType:
    """Return the single loaded head whose task loss can produce codon probs.

    Args:
        model: The loaded ``CoreModel``.
        command: Name of the calling command (``"oe-plot"``,
            ``"simulate"``, …), interpolated into the error messages so the
            wording stays specific to what the user actually ran.

    Raises:
        ValueError: If no loaded head has a conforming loss, or if more than
            one does (a joint checkpoint this caller cannot disambiguate).
        AmbiguousTaskLossError: Propagated from :func:`is_simulable_head`
            when a loaded head has several registered losses. Deliberately
            not caught here: a misconfigured head must surface by name
            rather than be scanned past as "not a candidate".
    """
    loaded_heads = {HeadType(key) for key in model.prediction_heads.keys()}
    candidates = sorted(
        (h for h in loaded_heads if is_simulable_head(model, h)), key=lambda h: h.value
    )
    if not candidates:
        raise ValueError(
            f"{command} requires a checkpoint whose head is backed by a loss "
            f"implementing the CodonSamplingLoss protocol (log_codon_probs) — "
            f"DASM, QDASM and Hybrid do. None of the loaded heads "
            f"{sorted(h.value for h in loaded_heads)} qualify: their losses are "
            f"{sorted(_loss_class_names(model, loaded_heads))}. The checkpoint is "
            f"likely from a non-DASM/QDASM/Hybrid training path."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"{command} does not yet support joint training across multiple "
            f"codon-sampling heads: the checkpoint has candidate heads "
            f"{[h.value for h in candidates]}. Extend {command} to disambiguate "
            f"(e.g. via a --head flag) before running this checkpoint."
        )
    return candidates[0]


def resolve_dasm_loss_head(model: "CoreModel") -> HeadType:
    """Return the single loaded head in ``DASM_LOSS_HEAD_TYPES``.

    For consumers that score a checkpoint's L×20 selection factors with the
    DASM kernel directly (``dasm2 likelihood``, ``dasm2 surprise``,
    ``dasm2 neutral-subs``), so a ``NeutralHead``, ``M0Head`` or
    ``SingleMatrixHead`` checkpoint goes through them like a
    ``SelectionFactorHead`` one.

    Raises an actionable error (rather than silently guessing) when the
    checkpoint has no DASM-loss head — likely a non-DASM training path — or
    more than one, which needs explicit disambiguation.

    Args:
        model: The loaded ``CoreModel``.

    Raises:
        ValueError: If no loaded head, or more than one, is a DASM-loss head.
    """
    loaded_heads = {HeadType(key) for key in model.prediction_heads.keys()}
    candidate_heads = loaded_heads & DASM_LOSS_HEAD_TYPES
    if not candidate_heads:
        raise ValueError(
            f"A DASM-loss head is required in the checkpoint, but none of the loaded "
            f"heads {sorted(h.value for h in loaded_heads)} are in DASM_LOSS_HEAD_TYPES "
            f"({sorted(h.value for h in DASM_LOSS_HEAD_TYPES)}). The checkpoint is "
            f"likely from a non-DASM training path."
        )
    if len(candidate_heads) > 1:
        raise ValueError(
            f"Joint DASM-loss training is not yet supported: the checkpoint has "
            f"multiple DASM-loss heads {sorted(h.value for h in candidate_heads)}. "
            f"Disambiguate (e.g. via a --head flag) before running this checkpoint."
        )
    (dasm_head,) = candidate_heads
    return dasm_head


def _loss_class_names(model: "CoreModel", heads: set[HeadType]) -> List[str]:
    """Loss class names for ``heads``, for the zero-candidate error message."""
    names: List[str] = []
    for head in heads:
        try:
            names.append(type(lookup_task_loss_for_head(model, head)).__name__)
        except (MissingTaskLossError, AmbiguousTaskLossError) as exc:
            # The two named types rather than a bare ``ValueError``: this
            # only builds an error message and must never be the thing that
            # explodes, but naming them keeps the reachability claim
            # checkable. An ambiguous head has already raised out of
            # ``is_simulable_head`` before we get here, so the
            # ``AmbiguousTaskLossError`` branch is unreachable today.
            names.append(f"<{type(exc).__name__} for {head.value}>")
    return names


def resolve_sampling_loss(model: "CoreModel", head_type: HeadType) -> CodonSamplingLoss:
    """Return ``head_type``'s task loss, asserting ``CodonSamplingLoss``.

    Raises:
        ValueError: If the registered loss cannot produce codon
            probabilities. The message names both the head and the loss
            class so the failure points at the checkpoint, not the caller.
    """
    loss = lookup_task_loss_for_head(model, head_type)
    if not isinstance(loss, CodonSamplingLoss):
        raise ValueError(
            f"Head {head_type.value} is backed by {type(loss).__name__}, which "
            f"does not implement the CodonSamplingLoss protocol "
            f"(log_codon_probs + PREDICT_REQUIRED_KEYS) and therefore has no "
            f"per-site codon kernel to sample from. Sequence-level "
            f"competing-risks losses (Whichmut) are structurally excluded — "
            f"see dasm2/losses/protocols.py and issue #192."
        )
    return loss


def is_qdasm_family_loss(loss: object) -> TypeGuard[Union[QDASMLoss, HybridLoss]]:
    """Whether ``loss`` is one of the codon-CTMC (``P = exp(tQ)``) objectives.

    The single place the Q-family membership test lives. Two behaviours key
    on it and must not drift apart: :func:`force_exact_evaluation` (which
    losses carry approximation knobs to force off) and
    ``needs_qdasm_branch_length_path`` in
    ``dasm2/training/hooks/branch_lengths.py`` (which branch-length
    optimizer a checkpoint needs). Adding a third Q-family loss is then one
    edit here rather than one edit per caller — the same
    forgetting-a-caller gap this module was written to close (issue #1092).

    Unlike :func:`is_simulable_head`, this cannot be replaced by a Protocol
    check: both callers need "runs a codon CTMC", which is a statement about
    *how* the kernel is computed, not about the ``log_codon_probs`` method
    every simulable loss already provides.
    """
    return isinstance(loss, (QDASMLoss, HybridLoss))


def force_exact_evaluation(loss: CodonSamplingLoss) -> None:
    """Force exact matrix-exponential evaluation on a loss, in place.

    A no-op for losses with no approximation knobs (DASM's kernel is exact
    by construction).

    ``DASMLoss.normalization`` is deliberately **not** forced, even though
    DASM now has a kernel knob (issue #1238). This function forces
    *approximations* back to exact; ``parent_absorbing`` vs
    ``sense_conditional`` is not an approximation of anything, it is part
    of the model definition the checkpoint was trained under. Forcing it
    would silently score a conditionally-trained checkpoint under the
    native likelihood — the failure mode this module exists to remove.

    For QDASM this means ``use_cubic=False``,
    ``adaptive=False``, ``matexp_method="adaptive"``, and — defensively —
    ``long_path="matrix_exp"``.

    Three of those four are load-bearing. ``adaptive``/``use_cubic`` select
    the approximation kernels directly. ``matexp_method="uniformization"``
    (if left set, e.g. from a checkpoint trained under issue #838's
    low-order uniformization surrogate) short-circuits
    ``_batched_matrix_exp_to_log_probs`` before ``use_cubic``/``adaptive``
    are read at all, routing through a truncated, low-order approximation
    regardless of what they say — so it must be forced back independently.

    ``long_path`` is forced for defence in depth only, and is **inert here
    today**: ``_batched_matrix_exp_to_log_probs`` reads it solely on the
    ``adaptive=True`` branch, so once ``adaptive=False`` it selects nothing
    and output is bit-identical either way (pinned by
    ``test_long_path_is_inert_when_adaptive_false``). It is set anyway to
    match the precedent script and to keep the loss's exactness flags
    mutually consistent. Do not infer from its presence that
    ``adaptive=False`` alone leaves an approximation live — it does not.

    ``HybridLoss`` has none of ``adaptive``/``long_path``/``matexp_method``
    — only ``use_cubic`` — so only that flag is forced for Hybrid;
    ``compute_hybrid_log_codon_probs`` leaves ``adaptive`` at its ``False``
    default, landing on exact ``torch.linalg.matrix_exp``.

    Under forced-exact settings ``normalization`` and ``n_terms`` are dead
    parameters: ``torch.linalg.matrix_exp`` reads neither.

    This mutates the loss object in place and does not restore the original
    values afterward — matching the precedent in ``qdasm_oe_eval.py``
    (issue #784 / PR #788). Callers that need to resume training from the
    same in-memory trainer must save and restore these values themselves.
    Note ``long_path``, inert for eval, *is* live during training (which
    runs ``adaptive=True``), so a resume that skips the restore would
    silently train under a different hot-tail kernel than it started with.

    Keying on the loss classes that *have* approximation knobs (rather than
    on a head-type set) means a future exact-by-construction kernel needs
    no entry here.
    """
    if is_qdasm_family_loss(loss):
        loss.use_cubic = False
    if isinstance(loss, QDASMLoss):
        loss.adaptive = False
        loss.long_path = "matrix_exp"
        loss.matexp_method = "adaptive"


def resolve_codon_probs_fn(
    model: "CoreModel",
    head_type: HeadType,
    device: Union[str, torch.device],
) -> CodonProbsFn:
    """Build the ``(batch, log_selection_factors) -> log codon probs`` closure.

    The one place the generative kernel is chosen. Because the returned
    closure captures the *checkpoint's own* loss object, the sampler, the
    O/E scorer and the branch-length optimizer all evaluate the same forced,
    reproducible model — they share one loss instance, not three
    reconstructions of one.

    :func:`force_exact_evaluation` runs **immediately**, as a side effect of
    this call, not lazily inside the closure: callers that also optimize
    branch lengths must do so under the same forced settings, and forcing
    only inside the closure would let branch lengths be fit under whichever
    approximation was active at training time while scoring used the exact
    model — an inconsistent pair of numbers.

    Args:
        model: Loaded ``CoreModel`` carrying the task losses.
        head_type: The head whose loss owns the kernel. Resolve it with
            :func:`resolve_simulable_head` rather than hardcoding it.
        device: Device to run the kernel on. Batches are moved there inside
            the closure; the caller keeps its CPU copy.

    Returns:
        A callable taking ``(batch, log_selection_factors)`` and returning
        the ``[B, L, 64]`` log-codon-probability tensor on ``device``.
    """
    loss = resolve_sampling_loss(model, head_type)
    force_exact_evaluation(loss)

    def codon_probs_fn(
        batch: Dict[str, Tensor], log_selection_factors: Tensor
    ) -> Tensor:
        device_batch = move_batch_to_device(batch, device)
        device_sf = log_selection_factors.to(device)
        return loss.log_codon_probs(device_sf, device_batch)

    return codon_probs_fn
