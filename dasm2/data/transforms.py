"""Per-sample transforms for TransformDataset.

Each transform is a frozen dataclass with a __call__ method that takes
a sample dict and returns it with new keys added. Transforms are composed
into a list and run sequentially in __getitem__.

Phase 1 transforms (DeltaAffinity):
    TokenizeParent, ComputeLayout, ComputeMasks

Phase 3 transforms (evolutionary heads):
    TranslateAndTokenize, CodonIndex, ComputeMutIndicator,
    NarrowLossMask, ComputeNtRatesAndCsps,
    ComputeCodonMutationRates, ComputeNeutralCodonProbs,
    LookupAAONeutralRates, MarginalizeCodonsToAAO
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from dasm2.codon_table import NUM_AA
from dasm2.constants import SMALL_PROBABILITY
from dasm2.data._neutral_matrix_exp import (
    check_codon_pref_compatible,
    check_multihit_compatible,
    compute_neutral_codon_probs_matrix_exp_per_pcp,
    finalize_matrix_exp_log_probs,
    is_matrix_exp_mode_enabled,
    parent_sense_idxs_of_codon64_idxs,
)
from dasm2.molevol.aao_neutral_rates import marginalize_codon_to_aao_rates
from dasm2.molevol.codon_neutral_rates import (
    NUM_CODON_INDICES,
    censored_codon_entries,
    combine_nt_rates_and_csps,
    compute_codon_mutation_rates,
    parent_base_at_offset,
)
from dasm2.molevol.neutral_evolution import neutral_codon_probs_with_padding
from dasm2.molevol.tandem import TandemTerm, check_tandem_multihit_compatible
from dasm2.neutral.base import SourceCodonFeatureLookup
from dasm2.neutral.codon_boundary import (
    SourceCodonContextCache,
    codon_segment_rate_adjustments,
    source_codon_boundary_features,
)
from dasm2.sequence.constants import (
    AA_AMBIG_IDX,
    AMBIGUOUS_CODON_IDX,
    BASES,
    NT_SEPARATOR,
)
from dasm2.sequence.conversion import (
    build_codon_nt_windows,
    codon_idx_tensor_of_str_ambig,
    count_codon_nt_differences,
    fused_translate_tokenize_codon_index,
    translate_sequence,
    unflatten_codon_idxs,
)
from dasm2.sequence.layout import layout_of_single_aa_idxs, padding_sentinel_for

# CensorMode/CENSOR_MODES live beside CensorSpec so that `TrainingConfig` can
# share the alias without `dasm2.config` importing this module; re-exported here
# because the transform and the dataset factories are what callers reach for.
from dasm2.sequence.masking import CENSOR_MODES, CensorMode, CensorSpec  # noqa: F401
from dasm2.tensors_of_seqs import tokenize_single_aa_sequence

if TYPE_CHECKING:
    from dasm2.data.segment_handlers import SegmentHandler
    from dasm2.neutral.base import RatesCSPNeutralModel
    from dasm2.neutral.hit_class import HitClassModel


# ---------------------------------------------------------------------------
# Phase 1 transforms (preserved for backward compatibility)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenizeParent:
    """Tokenize raw_parent AA sequence into aa_parents_idxs tensor.

    For AA-only datasets (e.g., DeltaAffinity) where raw_parent is already
    an amino acid string. For NT datasets, use TranslateAndTokenize instead.
    """

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["aa_parents_idxs"] = tokenize_single_aa_sequence(sample["raw_parent"])
        return sample


@dataclass(frozen=True)
class ComputeLayout:
    """Compute layout tensor from aa_parents_idxs.

    Args:
        num_segments: Number of protein segments (1 for single-chain,
            2 for antibody heavy+light, etc.).
    """

    num_segments: int = 1

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["layout"] = layout_of_single_aa_idxs(
            sample["aa_parents_idxs"], self.num_segments
        )
        return sample


@dataclass(frozen=True)
class ComputeMasks:
    """Compute mask and loss_mask from layout and aa_parents_idxs.

    Produces two boolean masks:
        mask: True for valid AA positions (not padding, not ambiguous).
            Includes separator positions (layout == 0) because
            AA_SEPARATOR_IDX != AA_AMBIG_IDX.
        loss_mask: Same as mask but also excludes separators (layout != 0).
            Optionally excludes positions with ambiguous parent/child codons
            when the corresponding flags are set.

    Args:
        num_segments: Number of protein segments (must match ComputeLayout).
        exclude_parent_codon_ambiguity: If True, exclude positions where
            codon_parents_idxs == AMBIGUOUS_CODON_IDX from loss_mask.
            Requires codon_parents_idxs in sample. Set True for PCP datasets.
        exclude_child_codon_ambiguity: If True, exclude positions where
            codon_children_idxs == AMBIGUOUS_CODON_IDX from loss_mask.
            Requires codon_children_idxs in sample. Set True for PCP datasets.
    """

    num_segments: int = 1
    exclude_parent_codon_ambiguity: bool = False
    exclude_child_codon_ambiguity: bool = False
    _padding_sentinel: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Precompute the padding sentinel from num_segments."""
        # frozen=True requires object.__setattr__ for derived fields
        object.__setattr__(
            self,
            "_padding_sentinel",
            padding_sentinel_for(self.num_segments),
        )

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        layout = sample["layout"]
        aa_idxs = sample["aa_parents_idxs"]
        base_mask = (layout != self._padding_sentinel) & (aa_idxs != AA_AMBIG_IDX)
        sample["mask"] = base_mask
        loss_mask = base_mask & (layout != 0)

        if self.exclude_parent_codon_ambiguity:
            loss_mask = loss_mask & (
                sample["codon_parents_idxs"] != AMBIGUOUS_CODON_IDX
            )
        if self.exclude_child_codon_ambiguity:
            loss_mask = loss_mask & (
                sample["codon_children_idxs"] != AMBIGUOUS_CODON_IDX
            )

        sample["loss_mask"] = loss_mask
        return sample


@dataclass(frozen=True)
class RenameKey:
    """Rename a key in the sample dict (move, not copy).

    Useful when a pre-built transform writes under a generic name (e.g.
    ``loss_mask``) but the head expects a head-specific name. Pop/insert so
    the original key is removed.
    """

    src: str
    dst: str

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[self.dst] = sample.pop(self.src)
        return sample


# ---------------------------------------------------------------------------
# Phase 3 transforms (evolutionary heads)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranslateAndTokenize:
    """Translate raw sequence (NT or AA) and tokenize into AA index tensor.

    Parameterized replacement for TokenizeParent/TokenizeChild. Handles both
    NT evolutionary datasets (translate=True) and AA-only datasets
    (translate=False) with one class.

    Args:
        source_key: Key in sample dict containing the raw sequence string.
        target_key: Key to store the resulting AA index tensor.
        translate: If True, translate NT -> AA before tokenizing. If False,
            tokenize the string directly as AA (for AA-only datasets).
    """

    source_key: str
    target_key: str
    translate: bool = True

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raw = sample[self.source_key]
        aa_seq = translate_sequence(raw) if self.translate else raw
        sample[self.target_key] = tokenize_single_aa_sequence(aa_seq)
        return sample


@dataclass(frozen=True)
class CodonIndex:
    """Compute codon index tensor from a raw NT sequence.

    Args:
        source_key: Key containing the raw NT string (e.g. "raw_parent").
        target_key: Key to store the codon index tensor
            (e.g. "codon_parents_idxs").
    """

    source_key: str
    target_key: str

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[self.target_key] = codon_idx_tensor_of_str_ambig(sample[self.source_key])
        return sample


@dataclass(frozen=True)
class FusedTranslateTokenizeCodonIndex:
    """Translate NT sequence and produce AA tokens + codon indices in one pass.

    Replaces ``TranslateAndTokenize`` + ``CodonIndex`` with a single loop
    over codons using pre-built lookup tables. Produces bit-for-bit identical
    results but avoids separate iteration passes.

    Used by both streaming (per-sample in __getitem__) and precomputed
    (bulk at init via the underlying ``fused_translate_tokenize_codon_index``
    function) paths.

    Note: The underlying function also returns the AA string, but this
    transform does not store it. The AA string is only needed in the bulk
    precomputed path (``_precompute_sequence_tensors``), not in the
    per-sample streaming path.

    Args:
        source_key: Key containing the raw NT string (e.g. "raw_parent").
        aa_target_key: Key to store the AA token tensor (e.g. "aa_parents_idxs").
        codon_target_key: Key to store the codon index tensor
            (e.g. "codon_parents_idxs"). If None, codon indices are not stored.
    """

    source_key: str
    aa_target_key: str
    codon_target_key: Optional[str] = None

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        _, aa_tokens, codon_idxs = fused_translate_tokenize_codon_index(
            sample[self.source_key]
        )
        sample[self.aa_target_key] = aa_tokens
        if self.codon_target_key is not None:
            sample[self.codon_target_key] = codon_idxs
        return sample


@dataclass(frozen=True)
class ComputeMutIndicator:
    """Compute mutation indicator from parent/child index tensors.

    Produces True where indices differ, excluding positions where both
    are the ambiguous sentinel value.

    Works for both codon-level (Whichmut) and AA-level (AAO Whichmut)
    mutation indicators.

    Args:
        parent_key: Sample key for parent indices (e.g. "codon_parents_idxs").
        child_key: Sample key for child indices (e.g. "codon_children_idxs").
        target_key: Sample key to store the boolean indicator tensor.
        ambiguous_idx: Sentinel value for ambiguous positions to exclude.
        single_nt_only: If True, exclude multi-nucleotide codon mutations
            (2+ nt changes). The Whichmut neutral rate tensor only has entries
            for single-nt transitions; multi-hit mutations produce zero-rate
            numerators with no useful gradient. See issue #403.
    """

    parent_key: str
    child_key: str
    target_key: str
    ambiguous_idx: int
    single_nt_only: bool = False

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        parent = sample[self.parent_key]
        child = sample[self.child_key]
        mut = parent != child
        both_ambig = (parent == self.ambiguous_idx) & (child == self.ambiguous_idx)
        indicator = mut & ~both_ambig
        if self.single_nt_only:
            indicator = indicator & (count_codon_nt_differences(parent, child) == 1)
        sample[self.target_key] = indicator
        return sample


@dataclass(frozen=True)
class NarrowLossMask:
    """Narrow loss_mask to positions where a mutation indicator is True.

    Whichmut-style losses are only meaningful at positions where a mutation
    actually occurred. This intersects the structural ``loss_mask`` (from
    ``ComputeMasks``) with a mutation indicator (from ``ComputeMutIndicator``)
    so that only mutation positions contribute to loss.

    Must run *after* both ``ComputeMasks`` (which produces ``loss_mask``)
    and ``ComputeMutIndicator`` (which produces the indicator tensor).

    Note:
        Unlike other transforms that add new keys, this *overwrites*
        ``loss_mask`` in place to narrow it.

    Args:
        indicator_key: Sample key for the boolean mutation indicator
            (e.g. "codon_mut_indicator" or "nonsyn_mut_indicator").
    """

    indicator_key: str

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        assert "loss_mask" in sample, (
            "NarrowLossMask requires 'loss_mask' in sample; "
            "ensure ComputeMasks runs before NarrowLossMask in the chain"
        )
        assert self.indicator_key in sample, (
            f"NarrowLossMask requires '{self.indicator_key}' in sample; "
            f"ensure ComputeMutIndicator runs before NarrowLossMask in the chain"
        )
        sample["loss_mask"] = sample["loss_mask"] & sample[self.indicator_key]
        return sample


# Named constants for separator fill values in multi-segment rate/CSP tensors.
# Separators are masked out during loss computation; these values ensure
# the tensors have the correct shape without affecting results.
SEPARATOR_NT_RATE_FILL = 1.0
SEPARATOR_NT_CSP_FILL = 0.0
NT_SEPARATOR_LEN = 3  # "^^^" = 3 nucleotides


@dataclass(frozen=True)
class ComputeNtRatesAndCsps:
    """Run neutral model to produce nt_rates and nt_csps per sample.

    Handles multi-segment sequences: splits by NT_SEPARATOR, runs model per
    segment, applies per-segment rate adjustments, recombines with separator
    fill values.

    All forward passes run under torch.no_grad() since these are data
    transforms, not training operations. The neutral model is expected to
    already be in eval mode (concrete implementations like ThriftySHMModel
    set this during construction).

    A model with a non-empty ``required_context`` (e.g. a per-family GTR
    model keyed on ``sample_id``) gets those keys from the sample as
    ``context``, so the seed sample must carry them. The Parquet factories add
    them to the row converter's scalar columns.

    Args:
        neutral_model: Neutral mutation model (pickle-safe for multiprocessing).
        segment_handler: Handler defining segment suffixes and rate adjustments.
    """

    neutral_model: "RatesCSPNeutralModel"
    segment_handler: "SegmentHandler"

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raw_parent = sample["raw_parent"]
        segments = raw_parent.split(NT_SEPARATOR)
        rate_adjustments = self.segment_handler.get_rate_adjustments()
        context = {key: sample[key] for key in self.neutral_model.required_context}

        all_rates: list[torch.Tensor] = []
        all_csps: list[torch.Tensor] = []
        for seg_idx, segment in enumerate(segments):
            if seg_idx > 0:
                all_rates.append(
                    torch.full((NT_SEPARATOR_LEN,), SEPARATOR_NT_RATE_FILL)
                )
                all_csps.append(
                    torch.full((NT_SEPARATOR_LEN, 4), SEPARATOR_NT_CSP_FILL)
                )

            with torch.no_grad():
                rates_list, csps_list = self.neutral_model.compute_rates_and_csps(
                    segment, context or None
                )
            all_rates.append(rates_list[0] * rate_adjustments[seg_idx])
            all_csps.append(csps_list[0])

        sample["nt_rates"] = torch.cat(all_rates)
        sample["nt_csps"] = torch.cat(all_csps)
        return sample


@dataclass(frozen=True)
class ComputeNeutralCodonProbs:
    """Compute log_neutral_codon_probs per sample from rates/CSPs + branch length.

    Uses the existing neutral_codon_probs_with_padding() function, which handles
    masked positions, multihit correction, and stop codon normalization.

    Must run after ComputeMasks (needs loss_mask) and ComputeNtRatesAndCsps
    (needs nt_rates, nt_csps). Also reads codon_parents_idxs and branch_length.

    Note:
        Not mutable despite holding a model reference. The multihit model
        should be put in eval mode and have gradients disabled at factory
        time before constructing this transform.

    Args:
        multihit_model: Optional multihit correction model. Deep-copied and
            prepared at factory time; pickle-safe for multiprocessing.
        codon_pref_log: Optional log-space codon preference vector, shape
            (64,) (issue #747). Fallback default used only if the sample
            dict has no ``"codon_pref_log"`` key. In production,
            ``TransformDataset`` injects the live value from its
            shared-memory ``_codon_pref_log`` tensor into the sample dict
            before running the transform chain (mirroring how
            ``branch_length`` is injected) — this field exists so the
            transform still works when constructed and called directly,
            outside a ``TransformDataset`` (e.g. in isolation in tests).
            A between-cycle refit (``CodonStrengthOptimizer``) mutates the
            shared tensor in place, which streaming/Parquet workers pick up
            on their next ``__getitem__`` call (DESIGN.md D6; issue #1039).
        tandem: Optional fixed tandem term, the neutral model's ``tandem``
            (issue #1683). The branch-length objective reads the same object
            off the bundle's neutral model, so both see one neutral.
            Construction raises if it is active alongside ``multihit_model``
            (issue #1753).
    """

    multihit_model: Optional["HitClassModel"] = None
    codon_pref_log: Optional[torch.Tensor] = None
    tandem: Optional[TandemTerm] = None

    def __post_init__(self) -> None:
        # Fail at construction, not inside a DataLoader worker (issue #1753).
        check_tandem_multihit_compatible(self.multihit_model, self.tandem)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        codon_pref_log = sample.get("codon_pref_log", self.codon_pref_log)
        if is_matrix_exp_mode_enabled():
            # Pilot-only escape hatch for issue #654: route per-sample
            # neutral through GY94 matrix-exp instead of product-of-three.
            check_multihit_compatible(self.multihit_model)
            check_codon_pref_compatible(codon_pref_log)
            if self.tandem is not None:
                raise ValueError(
                    "The tandem term (issue #1683) is applied only on the "
                    "product-of-three neutral, not in matrix-exp mode."
                )
            sample["log_neutral_codon_probs"] = (
                _per_sample_log_neutral_codon_probs_matrix_exp(sample)
            )
            return sample
        codon_probs = neutral_codon_probs_with_padding(
            sample["codon_parents_idxs"],
            sample["loss_mask"],
            sample["nt_rates"],
            sample["nt_csps"],
            float(sample["branch_length"].item()),
            multihit_model=self.multihit_model,
            target_length=sample["codon_parents_idxs"].shape[0],
            codon_pref_log=codon_pref_log,
            tandem=self.tandem,
        )
        codon_probs = torch.nan_to_num(codon_probs, nan=SMALL_PROBABILITY)
        sample["log_neutral_codon_probs"] = torch.log(codon_probs)
        return sample


def _per_sample_log_neutral_codon_probs_matrix_exp(
    sample: Dict[str, Any],
) -> "torch.Tensor":
    """Compute log_neutral_codon_probs for a single sample via GY94 matrix-exp.

    Mirrors the per-sample bookkeeping (truncation, padding, masking,
    nan_to_num, log) of ``neutral_codon_probs_with_padding`` but
    routes through ``exp(Q_codon · branch_length)`` per PCP.
    """
    codon_idxs = sample["codon_parents_idxs"]
    target_len = codon_idxs.shape[0]
    nt_rates = sample["nt_rates"]
    nt_csps = sample["nt_csps"]
    bl = float(sample["branch_length"].item())
    mask = sample["loss_mask"]

    aa_length = nt_rates.shape[0] // 3
    if aa_length == 0:
        return torch.full((target_len, 64), math.log(SMALL_PROBABILITY))

    truncated_codon_idxs = codon_idxs[:aa_length]
    truncated_mask = mask[:aa_length]
    clamped = torch.clamp(truncated_codon_idxs, 0, 63)
    parent_nt_idxs = unflatten_codon_idxs(clamped).reshape(-1)

    sense_idxs = parent_sense_idxs_of_codon64_idxs(clamped)
    sense_idxs = torch.where(
        truncated_mask, sense_idxs, torch.full_like(sense_idxs, -1)
    )

    probs = compute_neutral_codon_probs_matrix_exp_per_pcp(
        parent_nt_idxs_flat=parent_nt_idxs,
        parent_sense_idxs_per_codon=sense_idxs,
        nt_rates=nt_rates,
        nt_csps=nt_csps,
        branch_length=bl,
    )

    return finalize_matrix_exp_log_probs(probs, truncated_mask, target_len)


@dataclass(frozen=True)
class ReshapeNtToCodonRatesAndCsps:
    """Reshape nt_rates [L_nt] and nt_csps [L_nt, 4] to codon-space tensors.

    Produces codon_rates [L_codon, 3] and codon_csps [L_codon, 3, 4] needed
    by QDASM and Hybrid heads for Q-matrix construction.

    Must run after ComputeNtRatesAndCsps (needs nt_rates, nt_csps) and
    FusedTranslateTokenizeCodonIndex (needs codon_parents_idxs for length).
    """

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        L_codon = sample["codon_parents_idxs"].shape[0]
        nt_rates = sample["nt_rates"]
        nt_csps = sample["nt_csps"]
        L_nt = min(L_codon * 3, nt_rates.shape[0])
        L = L_nt // 3
        sample["codon_rates"] = nt_rates[: L * 3].reshape(L, 3)
        # Flatten [L, 3, 4] → [L, 12] for collator compatibility (2D max).
        # Loss mixins reshape back to [B, L, 3, 4].
        sample["codon_csps"] = nt_csps[: L * 3].reshape(L, 12)
        return sample


@dataclass(frozen=True)
class ComputeParentNtWindows:
    """Compute per-codon-site 15-nt parent windows for QDASM Option 1a.

    For each codon site ``ℓ``, extracts a 15-nt window spanning nt positions
    ``[3ℓ - 6, 3ℓ + 8]``. The window is wide enough to contain the three
    13-mers centered on each of the codon's three nt positions:

        position 3ℓ     → 13-mer = window[0:13]
        position 3ℓ + 1 → 13-mer = window[1:14]
        position 3ℓ + 2 → 13-mer = window[2:15]

    These 13-mers are the exact receptive field of ``ThriftyHumV0.2-59`` at
    each nt position within codon ``ℓ``, so substituting any intermediate
    codon at the 3 center positions ``(window[6], window[7], window[8])``
    lets us look up that intermediate's rate and CSP at each of the three
    codon positions from a precomputed 13-mer table.

    Positions outside ``[0, len(raw_parent))``, at segment separators ("^"),
    or at ambiguous bases are encoded as index 4 and any codon site whose
    window contains such a value is flagged via
    ``codon_parent_boundary_mask`` so the Q-builder can route those sites
    to source-specific boundary lookups.

    Must run after ``FusedTranslateTokenizeCodonIndex`` (needs
    ``codon_parents_idxs`` for the codon count). Reads ``raw_parent``.

    Also emits ``codon_rate_adjustments [L_codon]``: the segment handler's
    per-nucleotide multiplier at each codon site (unit at separators, padding,
    and without a handler). Lookup tables contain unadjusted rates, including
    when parent rates came from Parquet, so the Q-builder scales every lookup
    source row by it; boundary features below already carry the same factor.

    With a neutral model, also emits sparse ``codon_boundary_features`` records
    [N, 916]: codon site index followed by [61 sources, 3 bases, rate + 4 CSPs].
    Dataset factories validate complete Thrifty coverage before constructing
    this transform and attach a frozen table. Lookup never evaluates missing
    entries. The table stores compact source-specific nucleotide transitions
    and keeps a bounded decoded-entry cache. Models without frozen tables use
    the full-segment oracle. Thrifty simulation/audit callers must explicitly
    select ``use_source_oracle=True``; without it a missing table raises. Segment adjustments
    apply after lookup, so heavy/light chains can share unadjusted entries.
    Without a model this transform only builds windows, useful for inspecting
    contexts, not scoring boundary sites.
    """

    neutral_model: Optional["RatesCSPNeutralModel"] = None
    segment_handler: Optional["SegmentHandler"] = None
    use_source_oracle: bool = False
    _context_cache: Optional[SourceCodonFeatureLookup] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Bind the prevalidated table or explicit oracle cache to the neutral model."""
        if self.neutral_model is not None:
            lookup = (
                SourceCodonContextCache(self.neutral_model)
                if self.use_source_oracle
                else self.neutral_model.source_codon_lookup()
            )
            object.__setattr__(self, "_context_cache", lookup)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raw_parent = sample["raw_parent"]
        L_codon = sample["codon_parents_idxs"].shape[0]
        window, boundary_mask = build_codon_nt_windows(raw_parent, L_codon)
        sample["codon_parent_nt_window"] = window
        sample["codon_parent_boundary_mask"] = boundary_mask
        adjustments = (
            self.segment_handler.get_rate_adjustments()
            if self.segment_handler is not None
            else None
        )
        sample["codon_rate_adjustments"] = codon_segment_rate_adjustments(
            raw_parent, L_codon, adjustments
        )
        if self.neutral_model is not None:
            sample["codon_boundary_features"] = source_codon_boundary_features(
                raw_parent,
                boundary_mask,
                self.neutral_model,
                adjustments,
                context_cache=self._context_cache,
            )
        return sample


@dataclass(frozen=True)
class ComputeCodonMutationRates:
    """Compute codon_neutral_rates per sample from nt_rates x nt_csps.

    Reads codon_parents_idxs (already computed by CodonIndex) to determine
    which codons are at each position. Does not access raw_parent.

    Must run after CodonIndex("raw_parent", "codon_parents_idxs") and
    ComputeNtRatesAndCsps.
    """

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        nt_rates_per_base = combine_nt_rates_and_csps(
            sample["nt_rates"], sample["nt_csps"]
        )
        codon_idxs = sample["codon_parents_idxs"]
        num_codons = codon_idxs.shape[0]
        sample["codon_neutral_rates"] = compute_codon_mutation_rates(
            num_codons, nt_rates_per_base, codon_idxs
        )
        return sample


def _require_single_segment(transform: str, num_segments: int) -> None:
    """Reject multi-segment corpora for the censoring transforms.

    Shared by :class:`CensorMaskedMutations` and
    :class:`CensorReferenceReversions` because both index nucleotides as
    ``nt_index // 3`` and both are wrong in the same way for more than one
    segment: multi-segment ``raw_parent`` carries ``^`` separators occupying real
    codon slots, so a per-segment index is displaced by 3 per preceding
    separator. Viral corpora are single-segment; raising beats computing
    something plausible.
    """
    if num_segments != 1:
        raise NotImplementedError(
            f"{transform} supports num_segments == 1, got {num_segments}. "
            f"Nucleotide positions are resolved as nt_index // 3, which is the "
            f"codon only for a single segment: multi-segment raw_parent carries "
            f"'^' separators occupying real codon slots, so the index is "
            f"displaced by 3 per preceding separator. Censoring multi-segment "
            f"corpora needs a per-segment index."
        )


def _require_censor_mode(transform: str, mode: str) -> None:
    """Validate a censoring transform's ``mode``, shared for one error string."""
    if mode not in CENSOR_MODES:
        raise ValueError(
            f"{transform} mode must be one of {sorted(CENSOR_MODES)}, got {mode!r}."
        )


def _apply_censored_entries(
    sample: Dict[str, Any], censored: torch.Tensor, mode: CensorMode
) -> Dict[str, Any]:
    """Apply one ``(L, 64)`` censored-entry mask to both halves of the loss.

    The single place either censoring transform edits a sample, which is what
    structurally prevents the numerator and denominator drifting apart: the
    positions dropped from ``loss_mask`` are by construction exactly those
    removed from the partition function, because both are derived from the same
    tensor. Two transforms with two copies of this would be two chances to fix
    one half and not the other.

    Composition across transforms is exact for the same reason, which is what
    makes :class:`CensorReferenceReversions` a sibling of
    :class:`CensorMaskedMutations` rather than a branch inside it: rate zeroing
    is idempotent, ``loss_mask`` narrowing is monotone, and a second application
    reads the ``loss_mask`` the first already narrowed. Order does not matter and
    an entry censored by both is counted once.

    Args:
        sample: Whichmut sample, edited in place except for the two tensors
            below, which are replaced rather than mutated.
        censored: Boolean ``(L, NUM_CODON_INDICES)`` marking every rate entry to
            remove, indexed by destination codon.
        mode: ``"both"`` in production. The single-half modes are the ablation
            that demonstrates the two halves do different things; they are not
            valid objectives.
    """
    if mode in ("both", "loss_mask_only"):
        loss_mask = sample["loss_mask"]
        children = sample["codon_children_idxs"]
        positions = torch.nonzero(loss_mask, as_tuple=True)[0]
        # loss_mask already excludes ambiguous child codons in the whichmut
        # chain, but filtering rather than trusting that keeps this correct in a
        # chain that does not set exclude_child_codon_ambiguity.
        positions = positions[children[positions] < NUM_CODON_INDICES]
        if positions.numel():
            drop = censored[positions, children[positions]]
            if bool(drop.any()):
                loss_mask = loss_mask.clone()
                loss_mask[positions[drop]] = False
                sample["loss_mask"] = loss_mask

    if mode in ("both", "rates_only"):
        sample["codon_neutral_rates"] = sample["codon_neutral_rates"].masked_fill(
            censored, 0.0
        )

    return sample


# `eq=False` for the reason spelled out on `CensorSpec`: the generated `__eq__`
# would compare `censor_table` dicts, whose values hold tensors, and raise
# RuntimeError out of the `bool()`.
@dataclass(frozen=True, eq=False)
class CensorMaskedMutations:
    """Censor UShER-masked mutations, per clade, in both halves of the loss (#1079).

    UShER masking *deletes* mutations from the MAT rather than marking sites
    ambiguous, so the child inherits the ancestral state: a masked site is
    **censored, not ambiguous** -- the state is known, the *event* was destroyed.
    An uncensored whichmut loss still offers every candidate at those positions,
    so the model is asked to explain "why didn't this happen?" for events removed
    by fiat, which drives selection factors at masked sites down with nothing
    opposing it.

    Two edits, both required:

    * **Denominator** -- zeroing a ``codon_neutral_rates`` entry removes that
      candidate from the partition function, which masks on
      ``neutral_rates_gathered > 0``. Because the tensor is
      ``(L_codon, 64)`` indexed by *destination* codon, this is nt-granular,
      direction-granular and per-sample (hence per-clade) for free.
    * **Numerator** -- ``loss_mask`` selects a separate gather. Left alone, an
      observed censored transition contributes ``log(SMALL_PROBABILITY)``
      *exactly* -- a constant, with a gradient of exactly zero, because ``probs``
      no longer depends on the selection factor. Silent, uninformative, and it
      makes reported loss incomparable across runs.

    Land only one and the likelihood is not normalized over the set it sums.
    Both single-edit variants nevertheless train and produce plausible curves,
    which is why ``mode`` exists and why the arms are compared on a
    model-external metric rather than on validation loss.

    Must run after **both** ``ComputeCodonMutationRates`` and ``NarrowLossMask``.
    Appending to the whichmut chain satisfies that for the streaming and Parquet
    paths; the precomputed path builds ``codon_neutral_rates`` in a later batched
    pass, so there it is applied as a separate pass instead (see
    ``_build_precomputed_dataset``).

    Args:
        censor_table: ``{clade: CensorSpec}`` from
            :func:`~dasm2.sequence.masking.load_censor_table`, covering every
            clade of the corpus including those censoring nothing.
        num_segments: Segment count of the corpus. Only 1 is supported; see
            below.
        mode: ``"both"`` in production. ``"rates_only"`` and
            ``"loss_mask_only"`` apply a single half and exist for the ablation
            that demonstrates the two halves do different things -- they are not
            valid objectives.

    Raises:
        NotImplementedError: If ``num_segments > 1``. ``nt_index // 3`` is the
            codon only for a single segment: multi-segment ``raw_parent`` carries
            ``^`` separators that occupy real codon slots, so a per-segment index
            is displaced by 3 per preceding separator. Viral corpora are
            single-segment; raising is better than computing something plausible.
    """

    censor_table: Dict[str, "CensorSpec"]
    num_segments: int = 1
    mode: CensorMode = "both"
    _max_nt_index: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _require_single_segment("CensorMaskedMutations", self.num_segments)
        _require_censor_mode("CensorMaskedMutations", self.mode)
        if not self.censor_table:
            raise ValueError(
                "censor_table is empty. A table covering no clades censors "
                "nothing for every sample, which is indistinguishable from the "
                "transform not being installed; pass censor_table=None to the "
                "factory instead."
            )
        # Precomputed so the per-sample bounds check is an int comparison rather
        # than a max() over the table. Bounds are checked per sample and not only
        # at construction because pairing a gene's table with another gene's
        # corpus otherwise censors arbitrary in-range positions and still yields
        # a plausible loss curve.
        #
        # Deliberately table-wide rather than per-clade: every PCP of a viral gene
        # has the same length, so the strictest bound is also the right one, and
        # one int beats a per-clade dict lookup on every sample. A corpus with
        # clade-dependent sequence lengths would need the bound scoped per clade
        # -- it would otherwise reject a correctly-paired short sample because
        # some *other* clade reaches further.
        object.__setattr__(
            self,
            "_max_nt_index",
            max(
                (
                    max(
                        int(spec.site_nt_indices.max())
                        if spec.site_nt_indices.numel()
                        else -1,
                        int(spec.directional[:, 0].max())
                        if spec.directional.numel()
                        else -1,
                    )
                    for spec in self.censor_table.values()
                ),
                default=-1,
            ),
        )

    def _censored_entries_mask(
        self, spec: "CensorSpec", codon_parents: torch.Tensor
    ) -> torch.Tensor:
        """Boolean ``(L, 64)`` marking every rate entry this clade censors."""
        censored = torch.zeros(
            (codon_parents.shape[0], NUM_CODON_INDICES), dtype=torch.bool
        )

        for nt_indices, to_bases in (
            (spec.site_nt_indices, None),
            (spec.directional[:, 0], spec.directional[:, 2]),
        ):
            if nt_indices.numel() == 0:
                continue
            positions = torch.div(nt_indices, 3, rounding_mode="floor")
            offsets = nt_indices % 3

            # Separator and ambiguous parent codons (>= 64) index nothing in the
            # (64, 9) tables. Their rates are all zero anyway, so dropping them
            # changes no result -- but indexing with 64 would raise.
            live = codon_parents[positions] < NUM_CODON_INDICES
            if to_bases is not None:
                # The from_nt check is load-bearing, not defensive: a directional
                # mask records one transition, so censoring G->A at a position
                # whose parent carries C must leave C->A alone.
                live = live & (
                    parent_base_at_offset(codon_parents[positions], offsets)
                    == spec.directional[:, 1]
                )
            positions, offsets = positions[live], offsets[live]
            if positions.numel() == 0:
                continue

            rows, children = censored_codon_entries(
                codon_parents,
                positions,
                offsets,
                to_base_idxs=None if to_bases is None else to_bases[live],
            )
            censored[rows, children] = True

        return censored

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Censor in place, reading the clade from the sample.

        The transform-chain entry point, used by the streaming and Parquet paths
        where ``clade`` rides in the sample dict.
        """
        if "clade" not in sample:
            raise KeyError(
                "CensorMaskedMutations requires 'clade' in the sample. Parquet "
                "corpora carry it as a scalar column; CSV/streaming corpora need "
                "it passed via extra_string_fields."
            )
        return self.apply(sample, str(sample["clade"]))

    def apply(self, sample: Dict[str, Any], clade: str) -> Dict[str, Any]:
        """Censor in place with the clade supplied out of band.

        The precomputed path's entry point. There the sample dicts hold tensors
        only -- ``clade`` was stripped two passes earlier -- so the clade arrives
        as an argument rather than being written into the dict and deleted again.
        """
        if clade not in self.censor_table:
            raise ValueError(
                f"clade {clade!r} is not in the censor table, so whether it "
                f"censors anything is unknown. A clade censoring nothing is "
                f"present with an empty spec, so this is a corpus/table "
                f"mismatch. Known clades: {sorted(self.censor_table)}."
            )

        codon_parents = sample["codon_parents_idxs"]
        num_codons = codon_parents.shape[0]
        if self._max_nt_index >= 3 * num_codons:
            raise ValueError(
                f"censor table reaches nt_index {self._max_nt_index}, but this "
                f"sample has {num_codons} codons ({3 * num_codons} nt). The "
                f"table almost certainly belongs to a different gene -- nothing "
                f"else ties a censored_mutations.csv to a corpus."
            )

        spec = self.censor_table[clade]
        if spec.is_empty:
            return sample

        return _apply_censored_entries(
            sample, self._censored_entries_mask(spec, codon_parents), self.mode
        )


# `eq=False` because the generated `__eq__` compares `reference_base_idxs` with
# `==`, which on a tensor returns a tensor and then raises RuntimeError in the
# `bool()` the tuple comparison does. Identity comparison is what a transform
# wants anyway.
@dataclass(frozen=True, eq=False)
class CensorReferenceReversions:
    """Censor parent -> reference reversions in both halves of the loss (#1099).

    Reference bias inflates the observed rate of **every** reversion to the
    reference, not only the ones the UShER team curated: read mapping, gap
    filling and consensus calling all fail toward the reference allele, so any
    parent -> reference transition is over-represented relative to its true rate.
    That is the same defect #1079 fixes for UShER-masked mutations, from a
    different cause, so it gets the same treatment -- an observation whose rate is
    artifactually amplified should not have its *absence* charged to selection.

    The rule is clade-independent and per-sample: at every nucleotide where the
    parent differs from the reference, censor the single candidate codon that
    restores the reference base. Nucleotide- and direction-granular, never
    whole-site, so of a parent codon's nine single-nucleotide neighbours at most
    three are removed and the forward direction away from the reference stays
    live. At S 484 in BA.1 (``GCA``/A against Wuhan-Hu-1 ``GAA``/E) it censors
    ``GCA -> GAA`` alone.

    **A sibling of :class:`CensorMaskedMutations`, not a branch inside it.**
    That class requires a non-empty table, requires ``clade`` in the sample, and
    returns at ``if spec.is_empty`` *before* the mask is built -- so reversion
    censoring bolted in would be silently skipped for every clade whose spec is
    empty, which is 54 of 56 for E. The two compose exactly instead, because
    :func:`_apply_censored_entries` makes rate zeroing idempotent and
    ``loss_mask`` narrowing monotone; each keeps its own ``mode``.

    **This rule's profile is the opposite of both existing masks**, and it is
    what makes the single-half modes matter more here. On v11 S it removes 0.212%
    of offered candidates -- the smallest of the three -- but 7.22% of observed
    substitutions, roughly 45x the largest. The loss shift is therefore driven by
    lost numerator terms rather than by Z shrinking, so ``rates_only`` and
    ``loss_mask_only`` are correspondingly more informative and are meant to be
    run rather than skipped.

    Two consequences worth stating rather than discovering:

    * **Where the parent and reference differ at 2+ positions in one codon**,
      each differing position still yields a single-nucleotide candidate, and
      restoring one reference base can give a *third* residue -- neither the
      parent's nor the reference's. Over S's 1,859 differing (clade, codon,
      offset) triples, 1,555 restore the reference residue, 198 give a third one
      and 106 are synonymous. Censoring them follows from the reference-bias
      argument, but it is the one place where "reversion" stops meaning "back to
      the reference residue".
    * **The selection factor is indexed by destination amino acid**, so where
      another single-nucleotide child of the same parent codon encodes the same
      residue (``GCA -> GCG``, synonymous) the surviving route still supplies
      gradient to ``SF[pos, ref_aa]``. The claim that the model is not fit to
      reversions holds only for the nonsynonymous-unique case.

    Must run after **both** ``ComputeCodonMutationRates`` and ``NarrowLossMask``,
    like its sibling, and for the same reason.

    Args:
        reference_base_idxs: ``(3 * num_codons,)`` int64 indices into
            :data:`~dasm2.sequence.constants.BASES`, from
            :func:`~dasm2.sequence.masking.load_reference_base_idxs`.
        num_segments: Segment count of the corpus. Only 1 is supported.
        mode: ``"both"`` in production; the single-half modes are ablations and
            are not valid objectives.

    Raises:
        NotImplementedError: If ``num_segments > 1``.
        ValueError: If ``mode`` is unknown, if ``reference_base_idxs`` is not a
            1-D int64 tensor of ``BASES`` indices whose length is a multiple of
            3, or -- per sample -- if its length is not exactly ``3 *
            num_codons``.

    Note:
        The per-sample length check is exact equality, which makes it *stronger*
        than the table's in-range bound on ``nt_index``. That bound cannot catch
        a short gene's table paired with a long gene's corpus, because every
        index is legitimately in range; a reference of the wrong length is
        detectable, so it is detected.
    """

    reference_base_idxs: torch.Tensor
    num_segments: int = 1
    mode: CensorMode = "both"

    def __post_init__(self) -> None:
        _require_single_segment("CensorReferenceReversions", self.num_segments)
        _require_censor_mode("CensorReferenceReversions", self.mode)

        reference = self.reference_base_idxs
        if reference.ndim != 1:
            raise ValueError(
                f"reference_base_idxs must be 1-D, got shape "
                f"{tuple(reference.shape)}. It is a flat nucleotide sequence, "
                f"indexed as nt_index // 3 and nt_index % 3."
            )
        if reference.dtype != torch.int64:
            raise ValueError(
                f"reference_base_idxs must be int64, got {reference.dtype}. It "
                f"indexes the (64, 9) candidate tables and is compared against "
                f"base indices, so a float dtype would compare unequal "
                f"everywhere and censor the whole gene."
            )
        if reference.numel() == 0:
            raise ValueError(
                "reference_base_idxs is empty. A reference covering no positions "
                "censors nothing for every sample, which is indistinguishable "
                "from the transform not being installed; pass "
                "reference_base_idxs=None to the factory instead."
            )
        if reference.numel() % 3 != 0:
            raise ValueError(
                f"reference_base_idxs is {reference.numel()} long, which is not "
                f"a multiple of 3. It is a coding sequence, read as "
                f"{reference.numel() // 3} codons of 3 nucleotides."
            )
        # Ambiguity is rejected rather than tolerated: base index 4 compares
        # unequal to every parent base and then matches nothing in the (64, 9)
        # tables, so an ambiguous reference position censors nothing and reports
        # nothing. `load_reference_base_idxs` catches this at the FASTA, but the
        # transform is constructible without it.
        low, high = int(reference.min()), int(reference.max())
        if low < 0 or high >= len(BASES):
            raise ValueError(
                f"reference_base_idxs holds values in [{low}, {high}], outside "
                f"[0, {len(BASES) - 1}] for BASES={list(BASES)}. An out-of-range "
                f"base matches no candidate, so it would censor nothing at that "
                f"position and raise nothing."
            )

    def _censored_entries_mask(self, codon_parents: torch.Tensor) -> torch.Tensor:
        """Boolean ``(L, 64)`` marking the reference-restoring entry per difference."""
        num_codons = codon_parents.shape[0]
        censored = torch.zeros((num_codons, NUM_CODON_INDICES), dtype=torch.bool)

        positions = torch.arange(num_codons).repeat_interleave(3)
        offsets = torch.tensor([0, 1, 2]).repeat(num_codons)

        # Mandatory rather than defensive, and for two different reasons.
        # `censored_codon_entries` raises IndexError on a parent index >= 64,
        # and `parent_base_at_offset` does *not* raise there -- it decodes 64 as
        # base 4 at offset 0 but as base 0 at offsets 1 and 2, so an unfiltered
        # comparison would read a plausible base off a separator codon and
        # censor a position that does not exist.
        live = codon_parents[positions] < NUM_CODON_INDICES
        # The `!=` is load-bearing for a non-obvious reason: a self-transition
        # yields ZERO entries from `censored_codon_entries` silently rather than
        # raising, so dropping this filter would censor strictly less and never
        # fail a smoke test.
        hit = live & (
            parent_base_at_offset(codon_parents[positions], offsets)
            != self.reference_base_idxs
        )
        if not bool(hit.any()):
            return censored

        rows, children = censored_codon_entries(
            codon_parents,
            positions[hit],
            offsets[hit],
            to_base_idxs=self.reference_base_idxs[hit],
        )
        censored[rows, children] = True
        return censored

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Censor in place. One entry point, because the rule needs no clade.

        Unlike :meth:`CensorMaskedMutations.apply`, there is nothing to supply
        out of band -- which is why this transform installs identically on the
        streaming, Parquet and precomputed paths.
        """
        codon_parents = sample["codon_parents_idxs"]
        num_codons = codon_parents.shape[0]
        if self.reference_base_idxs.numel() != 3 * num_codons:
            raise ValueError(
                f"reference is {self.reference_base_idxs.numel()} nt "
                f"({self.reference_base_idxs.numel() // 3} codons), but this "
                f"sample has {num_codons} codons ({3 * num_codons} nt). The "
                f"reference almost certainly belongs to a different gene -- "
                f"nothing else ties a reference FASTA to a corpus."
            )
        return _apply_censored_entries(
            sample, self._censored_entries_mask(codon_parents), self.mode
        )


# `eq=False` for the same reason `CensorReferenceReversions` needs it: the
# generated `__eq__` compares the tensor field with `==`, which returns a tensor
# and raises RuntimeError in the `bool()` the tuple comparison does.
@dataclass(frozen=True, eq=False)
class LookupAAONeutralRates:
    """Look up AAO neutral rates from a context-free (20, 20) rate matrix.

    For each position, indexes the rate matrix by the parent AA to produce
    a (L, 20) tensor of neutral rates to each target AA.

    Args:
        neutral_aa_rates: Context-free rate matrix of shape (20, 20).
            Entry [a, a'] is the neutral rate of AA a -> a'.
    """

    neutral_aa_rates: torch.Tensor

    def __post_init__(self) -> None:
        """Validate rate matrix shape at construction time."""
        if self.neutral_aa_rates.shape != (NUM_AA, NUM_AA):
            raise ValueError(
                f"neutral_aa_rates must be ({NUM_AA}, {NUM_AA}), "
                f"got {tuple(self.neutral_aa_rates.shape)}"
            )

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        aa_idxs = sample["aa_parents_idxs"]
        valid_mask = aa_idxs < NUM_AA
        safe_indices = aa_idxs.clamp(0, NUM_AA - 1)
        rates = self.neutral_aa_rates[safe_indices].clone()  # (L, 20)
        rates[~valid_mask] = 0.0
        sample["aao_neutral_rates"] = rates
        return sample


@dataclass(frozen=True)
class MarginalizeCodonsToAAO:
    """Marginalize codon_neutral_rates to amino-acid-level aao_neutral_rates.

    Reads ``codon_neutral_rates`` (L, 64) from the sample and produces
    ``aao_neutral_rates`` (L, 20) by summing rates over child codons that
    translate to each amino acid.

    Must run after ComputeCodonMutationRates (needs ``codon_neutral_rates``).

    Note:
        In precomputed mode, ``codon_parents_idxs`` (produced by CodonIndex)
        must be retained through batched CMR via ``extended_keys``. It is not
        part of the final output and is removed after marginalization.
    """

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # marginalize_codon_to_aao_rates expects (N, L, 64), add batch dim
        codon_rates = sample["codon_neutral_rates"].unsqueeze(0)
        aao_rates = marginalize_codon_to_aao_rates(codon_rates).squeeze(0)
        sample["aao_neutral_rates"] = aao_rates
        return sample
