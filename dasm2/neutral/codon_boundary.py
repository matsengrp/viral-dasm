"""Segment scaling and full-segment source-codon features for global k-mer Q."""

from collections import OrderedDict
from typing import Hashable, Iterator, Optional, Tuple

import torch
from torch import Tensor

from dasm2.codon_table import CODONS, SENSE_CODON_LIST
from dasm2.neutral.base import RatesCSPNeutralModel, SourceCodonFeatureLookup
from dasm2.sequence.constants import NT_SEPARATOR

# One sparse record: codon site index, then [61 source codons, 3 nt, 5 fields].
# Packing keeps samples two-dimensional for the collator and avoids allocating
# 915 floats at every interior site. Padding records have site index -1.
BOUNDARY_FEATURE_WIDTH = 1 + 61 * 3 * 5
DEFAULT_MAX_CONTEXTS = 128
SENSE_CODONS = tuple(CODONS[i] for i in SENSE_CODON_LIST)
assert len(NT_SEPARATOR) % 3 == 0, "NT_SEPARATOR must span whole codons"


def _segment_codon_starts(
    raw_parent: str, rate_adjustments: Optional[Tensor] = None
) -> Iterator[Tuple[int, str]]:
    """Yield ``(first codon site, segment)`` for each segment of ``raw_parent``.

    The single segment walk behind :func:`codon_segment_rate_adjustments` and
    :func:`source_codon_boundary_features`, so lookup rows and boundary
    rescans always agree on which segment a codon site belongs to. Each
    separator occupies one codon site.

    Raises:
        ValueError: If fewer adjustments than segments are supplied, or a
            segment's length is not a multiple of 3. Every later
            segment's codon sites would otherwise be silently shifted.
            Dataset loading already rejects such segments
            (:func:`dasm2.data.validation.has_scored_substitution`), so only
            ad hoc callers can hit this.
    """
    segments = raw_parent.split(NT_SEPARATOR)
    if rate_adjustments is not None and len(segments) > len(rate_adjustments):
        raise ValueError(
            f"raw_parent has {len(segments)} segments but only "
            f"{len(rate_adjustments)} segment rate adjustments"
        )
    site = 0
    for segment in segments:
        if len(segment) % 3:
            raise ValueError(
                f"Segment length ({len(segment)}) is not a multiple of 3; "
                "codon sites would be shifted"
            )
        yield site, segment
        site += (len(segment) + len(NT_SEPARATOR)) // 3


def codon_segment_rate_adjustments(
    raw_parent: str,
    length: int,
    rate_adjustments: Optional[Tensor] = None,
) -> Tensor:
    """Per-codon-site segment nucleotide multipliers, ``[length]``.

    Shares :func:`_segment_codon_starts` with
    :func:`source_codon_boundary_features`, so lookup rows and boundary
    rescans scale each site by the same factor. Separator codons, padding past
    ``raw_parent``, and every site when ``rate_adjustments`` is None get a
    unit multiplier.
    """
    adjustments = torch.ones(length)
    if rate_adjustments is None:
        return adjustments
    segments = list(_segment_codon_starts(raw_parent, rate_adjustments))
    for (start, segment), adjustment in zip(segments, rate_adjustments.tolist()):
        adjustments[start : start + len(segment) // 3] = adjustment
    return adjustments


class SourceCodonContextCache:
    """Bounded cache of observed contexts for one frozen neutral model.

    Each value holds unadjusted [61 sources, 3 nt, rate + 4 CSPs] float32
    features. Capacity is bounded by DEFAULT_MAX_CONTEXTS.
    Misses use the original full-segment oracle;
    eviction affects only preprocessing work, never prediction accuracy.
    Segment rate adjustments are applied by the caller after retrieval.
    """

    def __init__(
        self,
        neutral_model: RatesCSPNeutralModel,
        max_contexts: int = DEFAULT_MAX_CONTEXTS,
    ) -> None:
        """Bind a cache to a model, with a positive context-count memory bound."""
        if max_contexts < 1:
            raise ValueError(f"max_contexts must be positive, got {max_contexts}")
        self.neutral_model = neutral_model
        self.max_contexts = max_contexts
        self._features: OrderedDict[Hashable, Tensor] = OrderedDict()

    def __len__(self) -> int:
        """Number of contexts currently retained."""
        return len(self._features)

    def get(self, segment: str, start: int) -> Tensor:
        """Return read-only source features, evaluating a new context once.

        Args:
            segment: Original full nucleotide segment.
            start: Nucleotide offset of a complete source codon.

        Returns:
            Cached CPU float32 tensor with shape [61, 3, 5]. Callers must not
            mutate it; packing site records copies it into per-sample storage.

        Raises:
            ValueError: The model cannot return the complete source codon.
        """
        key = self.neutral_model.source_codon_context_key(segment, start)
        if key in self._features:
            self._features.move_to_end(key)
            return self._features[key]
        sources = [
            segment[:start] + codon + segment[start + 3 :] for codon in SENSE_CODONS
        ]
        with torch.no_grad():
            rates, csps = self.neutral_model.compute_rates_and_csps_vectorized(sources)
        r = torch.stack([v[start : start + 3] for v in rates]).cpu()
        c = torch.stack([v[start : start + 3] for v in csps]).cpu()
        if r.shape != (61, 3) or c.shape != (61, 3, 4):
            raise ValueError(
                f"Neutral model did not return all three bases at boundary site {start // 3}"
            )
        features = torch.cat((r.unsqueeze(-1), c), dim=-1).float()
        if len(self._features) >= self.max_contexts:
            self._features.popitem(last=False)
        self._features[key] = features
        return features


def source_codon_boundary_features(
    raw_parent: str,
    boundary_mask: Tensor,
    neutral_model: RatesCSPNeutralModel,
    rate_adjustments: Optional[Tensor] = None,
    context_cache: Optional[SourceCodonFeatureLookup] = None,
) -> Tensor:
    """Look up each boundary site's 61 source features, rescanning cache misses.

    Returns ``[N, 916]`` sparse records for valid sense codons flagged by
    ``boundary_mask``. Ambiguous/stop codons, separators, and padding have no
    scored parent state and need no record. Segments are evaluated separately,
    exactly as in parent-rate preprocessing; no synthetic N padding is added
    and the model's encoder length is left intact. Segment rate adjustments
    apply to every source before products for multi-hit edges are formed.

    An optional context_cache shares unadjusted features across parents/sites
    with identical model inputs. It must belong to neutral_model. Without one,
    a temporary cache still shares contexts across this parent's segments.
    """
    if context_cache is None:
        context_cache = SourceCodonContextCache(neutral_model)
    elif context_cache.neutral_model is not neutral_model:
        raise ValueError("context_cache must belong to the supplied neutral_model")
    records = []
    for segment_index, (first_site, segment) in enumerate(
        _segment_codon_starts(raw_parent, rate_adjustments)
    ):
        adjustment = (
            float(rate_adjustments[segment_index])
            if rate_adjustments is not None
            else 1.0
        )
        for start in range(0, len(segment) - 2, 3):
            site = first_site + start // 3
            if site >= boundary_mask.numel():
                break
            if (
                not boundary_mask[site]
                or segment[start : start + 3] not in SENSE_CODONS
            ):
                continue
            cached = context_cache.get(segment, start)
            features = torch.cat(
                (cached[..., :1] * adjustment, cached[..., 1:]), dim=-1
            )
            records.append(torch.cat((torch.tensor([float(site)]), features.flatten())))
    if not records:
        return torch.empty((0, BOUNDARY_FEATURE_WIDTH), dtype=torch.float32)
    return torch.stack(records).float()
