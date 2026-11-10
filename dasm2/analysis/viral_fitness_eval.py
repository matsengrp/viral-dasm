"""Model-external evaluation of viral DASMs against mat-pcp's fitness effects.

A fork of the upstream module of the same name, carrying only what scores a
checkpoint: the forward pass, the filter chain, the clip and the clade
adequacy floor. The run axis, the manifest resolution and the shift and
effect-error families are not here -- `config.yaml` names every path this
pipeline reads, and the per-figure readouts live with the figures.

What this module owns:

* the forward pass that turns a checkpoint and a clade founder into
  per-mutation log selection factors (:class:`SelectionFactorSource`),
* which fitness column counts as "the fitness effect"
  (:data:`FITNESS_METRICS`),
* how fitness rows join to log selection factors, which of them survive the
  count filters, and the common lower bound both axes are clamped to before
  anything is correlated (:func:`merge_and_filter`, :data:`CLIP_LOWER`),
* when a clade holds too few mutations to correlate at all
  (:func:`adequate_clades`).

Fitness effects are read, never recomputed. mat-pcp publishes
``delta_fitness_ratio`` (the raw log count ratio,
``log((actual + 0.5) / (expected + 0.5))``) and ``delta_fitness_bayesian``
(the same ratio shrunk toward zero by a fitted prior). Both are taken as
written; there is no pseudocount control anywhere, because a second
pseudocount would let a caller report a number the source data does not
contain. Count thresholds select *which mutations enter a correlation* and
never change a mutation's fitness value.

The clip is the one control that does change a plotted value, and it is not a
fitness estimate: it clamps both axes to a common lower bound so the saturated
left tail stops dominating a least-squares fit. Every correlation computed
through :func:`merge_and_filter` uses the clipped values -- move the bound and
the numbers move with it -- and :func:`clip_axes` keeps the pre-clip values in
``*_raw`` columns so the effect is inspectable rather than baked in.

GISAID-derived: everything read through here descends from the cluster-only
mat-pcp mount, and nothing read through it may be redistributed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, cast

import pandas as pd
import torch

from dasm2.analysis.viral import (
    build_sf_dataframe_from_tensor,
    filter_merged_fitness,
    merge_fitness_and_sf,
)
from dasm2.heads.types import HeadType
from dasm2.models.core.base import CoreModel
from dasm2.serialization.checkpoints import load_model_only

# Both are mat-pcp's own published estimates. `delta_fitness_ratio` first because
# it is the raw log count ratio and carries no estimator choice into the
# correlation; `delta_fitness_bayesian` is the better estimate of any single
# mutation's effect, but its shrinkage is count-dependent.
FITNESS_METRICS: list[str] = ["delta_fitness_ratio", "delta_fitness_bayesian"]

# Count-filter defaults.
# `or` rather than `and`: the two counts fail in opposite directions. A strongly
# deleterious mutation can be well observed at a site the neutral model expected
# little traffic through, while a lethal mutation at a well-sampled site is
# legitimately near zero actual. An OR keeps a mutation whose estimate rests on
# real observations *or* on a meaningful denominator. The cost is that the
# surviving set is not uniform in precision, which is why `uncertainty` is
# carried alongside rather than dropped.
MIN_ACTUAL_COUNT = 10
MIN_PREDICTED_COUNT = 10
COUNT_FILTER_MODE: Literal["and", "or"] = "or"

# The threshold ladder for the count-filter sweep. Both thresholds move
# together, so a rung is a single number and the `or` above still decides how
# the two counts combine at it -- see :func:`apply_count_threshold`.
#
# 10 is the default filter, so the first rung reproduces every other number in
# a readout rather than starting somewhere new. Above it the spacing is
# geometric because that is how count precision improves: the standard error on
# a log count ratio falls as 1/sqrt(n), so 10 -> 20 buys the same factor as
# 50 -> 100, and equal *additive* steps would spend all four rungs in the
# regime where nothing changes. 100 is the ceiling because the surviving
# population is what pays for each rung: raising the floor sharpens the fitness
# estimates that remain while shrinking their number, and past some point the
# clade adequacy floor starts removing whole clades rather than trimming
# mutations. Where that point falls is a property of the run, so the sweep
# reports the surviving counts per rung instead of asserting one here.
COUNT_THRESHOLD_LADDER: list[int] = [10, 20, 50, 100]

# Below this many surviving mutations a clade's Pearson r is not a measurement.
# 20 is a floor, not a sufficiency threshold: it exists because the alternative
# is worse, not because 20 mutations make a solid correlation.
#
# What it is for: mat-pcp emits fitness for every clade, but the early ones have
# almost no descendants, so their counts sit in the single digits on *both*
# measures and nearly every row falls to the count filter. A clade left with no
# rows contributes a NaN dot to a within-clade plot and an all-NaN pair to
# every clade pair it appears in, so the floor has to be applied before a
# correlation is taken rather than after.
DEFAULT_MIN_CLADE_MUTATIONS = 20

# Common lower bound applied to BOTH axes before anything is correlated or
# plotted.
#
# Both quantities have a long left tail, where "lethal" and "very lethal" are
# not meaningfully distinguishable but do dominate a least-squares fit and
# squash the bulk of the cloud into a corner. Clamping both to one bound keeps
# them on a single scale and keeps the correlation about the region where the
# two measures actually carry information. -20 is beneath every observed value
# and is therefore equivalent to no clipping, which is how a caller turns this
# off.
#
# Consequence worth knowing, and the reason `clip_axes` keeps `*_raw` columns:
# every r moves with this bound, and cross-clade shifts are differences of
# *clipped* values, so a mutation pinned to the floor in both clades has a shift
# of exactly 0 -- the model did predict, both predictions were just below the
# bound. Read a high "no shift" fraction as "pinned to the floor in both clades"
# and expect it to fall as the bound is lowered.
CLIP_LOWER = -4.0


class SelectionFactorSource:
    """Model log selection factors per (gene, run, clade), cached.

    Inference is the expensive step and the cache is keyed on
    ``(gene, run_id, clade)``, so re-reading a frame never re-runs it.

    The cache key carries no sequence, so a caller that scores more than one
    parent haplotype per clade must key its own storage on the haplotype. Asking
    this class for two haplotypes under one clade returns the first one twice,
    with no error and a plausible-looking frame.

    ``load_model_only`` rather than ``load_checkpoint``: only
    :meth:`selection_factors_of_aa_str` is called here, and a checkpoint's
    recorded neutral ``model_path`` need not resolve off the training host.
    """

    def __init__(
        self,
        checkpoints: dict[tuple[str, str], Path],
        device: str | torch.device,
    ) -> None:
        self.checkpoints = checkpoints
        self.device = torch.device(device)
        self._models: dict[tuple[str, str], tuple[CoreModel, HeadType]] = {}
        self._frames: dict[tuple[str, str, str], pd.DataFrame] = {}
        self._factors: dict[tuple[str, str, str], torch.Tensor] = {}

    def _model(self, gene: str, run_id: str) -> tuple[CoreModel, HeadType]:
        key = (gene, run_id)
        if key not in self._models:
            # `load_model_only` is annotated as returning the torch base class,
            # but every checkpoint this reads holds a `CoreModel` -- which is
            # what carries `prediction_heads` and `selection_factors_of_aa_str`.
            #
            # Load straight onto the target device rather than onto CPU and
            # moving after. The dense-attention fallback is belt and braces
            # here: every config in this repository sets
            # `flash_attention: false`, so a checkpoint should already load
            # without it. `flash_attention_available` tests
            # `device.startswith("cuda")`, so the device has to arrive as a
            # string.
            module = cast(
                CoreModel,
                load_model_only(
                    self.checkpoints[key],
                    device=str(self.device),
                    allow_dense_attention_fallback=True,
                ),
            )
            module.eval()
            module.to(self.device)
            head = HeadType(list(module.prediction_heads.keys())[0])
            self._models[key] = (module, head)
        return self._models[key]

    def log_sf_frame(
        self, gene: str, run_id: str, clade: str, founder_aa: str
    ) -> pd.DataFrame:
        """Per-mutation log selection factors on one clade's founder.

        ``selection_factors_of_aa_str`` returns gauge-zapped factors, so the
        wildtype entry at each site is exactly 1.0 and its log is exactly 0.
        Synonymous rows are therefore structural zeros rather than predictions
        and are dropped by :func:`merge_and_filter`.

        Returns:
            The frame from :func:`build_sf_dataframe_from_tensor`: one row per
            (site, amino acid), with ``aa_mutation``, ``aa_site``,
            ``mutant_aa``, ``log_selection_factor`` and ``is_synonymous``.
        """
        key = (gene, run_id, clade)
        if key not in self._frames:
            log_sf = torch.log(
                self.selection_factors(gene, run_id, clade, founder_aa)
            ).numpy()
            self._frames[key] = build_sf_dataframe_from_tensor(log_sf, founder_aa)
        return self._frames[key]

    def selection_factors(
        self, gene: str, run_id: str, clade: str, founder_aa: str
    ) -> torch.Tensor:
        """``[L, 20]`` linear selection factors on one clade's founder.

        The whole matrix, on the CPU, cached beside the frame
        :meth:`log_sf_frame` derives from it. Whichmut's partition function
        sums over every site and every amino acid, including the ones no
        per-mutation frame keeps: the synonymous entries, which the gauge
        fixes at exactly 1, and the amino acids no single nucleotide change
        reaches, which the neutral rates zero out rather than the factors.

        Linear, not log -- what ``selection_factors_of_aa_str`` returns, and
        what the loss's normalizer expects.
        """
        key = (gene, run_id, clade)
        if key not in self._factors:
            module, head = self._model(gene, run_id)
            with torch.no_grad():
                factors = module.selection_factors_of_aa_str(founder_aa, head_type=head)
            self._factors[key] = factors[0].cpu()
        return self._factors[key]

    def release(self) -> None:
        """Drop loaded modules, keeping the computed SF frames.

        A run's worth of resident weights is the one thing here worth freeing on
        a shared GPU; the frames themselves are small.
        """
        self._models.clear()


def merge_and_filter(
    fitness_df: pd.DataFrame,
    sf_df: pd.DataFrame,
    *,
    exclude_sites: Optional[set[int]] = None,
    restrict_to_sites: Optional[set[int]] = None,
    exclude_synonymous: bool = True,
    min_actual_count: float = MIN_ACTUAL_COUNT,
    min_predicted_count: float = MIN_PREDICTED_COUNT,
    count_filter_mode: Literal["and", "or"] = COUNT_FILTER_MODE,
    clip_lower: Optional[float] = CLIP_LOWER,
) -> pd.DataFrame:
    """Join fitness to log SF for one (gene, run, clade), filter it and clip it.

    This is an **inner** join, because :func:`merge_fitness_and_sf` is one, so
    the output holds only mutations mat-pcp estimated. The scored table this
    repository writes is a left join from the model grid and does not go
    through here; this function exists so that table can be filtered down to
    the upstream row set and compared against it.

    The join is on ``(aa_mutation, aa_site, mutant_aa)``, and ``aa_mutation``
    embeds the wildtype residue -- so a fitness row whose wildtype disagrees with
    the residue the model conditioned on cannot enter the output at all,
    whatever happens downstream. ``slice_fitness_for_gene`` has already checked
    that agreement and raised on any violation, so this is belt and braces
    rather than the primary guard.

    Filtering only decides which of mat-pcp's rows to keep. Nothing here
    computes or alters a fitness effect.

    Args:
        fitness_df: One (gene, clade) slice of mat-pcp's per-clade effects.
        sf_df: The matching frame from :meth:`SelectionFactorSource.log_sf_frame`.
        exclude_sites: Codon sites to drop, normally the globally masked ones.
            Not redundant with the pipeline: mat-pcp's ``exclude`` flag is per
            *nucleotide* mutation while the mask file is per *codon*, so masked
            codons still carry fitness rows reachable through their unexcluded
            nucleotide positions -- while DASM training masked the whole codon
            and so never saw them.
        restrict_to_sites: Keep only these codon sites, applied *before* the
            other filters. This is the site-class split: pass
            :func:`~dasm2.sequence.masking.censored_codon_sites` for the class
            censoring acts on, and its complement for the rest. Composes with
            ``exclude_sites`` rather than replacing it, so a censored *and*
            problematic site is still dropped: the comparison is about censoring,
            and a site DASM never trained on cannot speak to it.
        exclude_synonymous: Drop wildtype-to-wildtype rows, whose log SF is a
            structural 0 rather than a prediction.
        min_actual_count: Minimum ``actual_count`` to keep a mutation.
        min_predicted_count: Minimum ``predicted_count`` to keep a mutation.
        count_filter_mode: ``"or"`` or ``"and"``; see :data:`COUNT_FILTER_MODE`.
        clip_lower: Common lower bound for both axes, or None for no clipping.
            See :data:`CLIP_LOWER` -- every correlation downstream is computed on
            the clipped values, and ``*_raw`` columns keep the originals.

    Returns:
        The filtered merged frame, one row per surviving mutation, with
        ``log_selection_factor`` and both fitness estimates clipped and their
        pre-clip values preserved as ``log_selection_factor_raw`` and
        ``{metric}_raw``.
    """
    merged = merge_fitness_and_sf(fitness_df, sf_df)
    if restrict_to_sites is not None:
        merged = merged[merged["aa_site"].isin(restrict_to_sites)]
    merged = filter_merged_fitness(
        merged,
        exclude_synonymous=exclude_synonymous,
        exclude_sites=exclude_sites or None,
        min_actual_count=min_actual_count,
        min_predicted_count=min_predicted_count,
        count_filter_mode=count_filter_mode,
    )
    return clip_axes(merged, clip_lower)


def clip_axes(merged: pd.DataFrame, clip_lower: Optional[float]) -> pd.DataFrame:
    """Clamp log SF and both fitness estimates to a common lower bound.

    Both fitness columns are clipped, not just the selected one, so switching the
    fitness selector cannot switch between a clipped and an unclipped y-axis.

    Pre-clip values are kept as ``log_selection_factor_raw`` and ``{metric}_raw``.
    They are what a per-mutation readout reads, so a point sitting on the bound
    still reveals what it actually was -- a per-mutation readout of a clamped
    value would be misleading.

    Args:
        merged: A merged fitness+SF frame.
        clip_lower: The bound, or None to leave values untouched. The ``*_raw``
            columns are written either way, so downstream code can read them
            without knowing whether clipping was on.

    Returns:
        A copy with the clipped columns in place and the raw ones added.
    """
    columns = ["log_selection_factor"] + [
        metric for metric in FITNESS_METRICS if metric in merged.columns
    ]
    out = merged.copy()
    for column in columns:
        out[f"{column}_raw"] = out[column]
        if clip_lower is not None:
            out[column] = out[column].clip(lower=clip_lower)
    return out


def adequate_clades(
    merged_by_clade: dict[str, pd.DataFrame],
    min_mutations: int = DEFAULT_MIN_CLADE_MUTATIONS,
) -> tuple[list[str], pd.DataFrame]:
    """Split clades into those with enough surviving mutations and those without.

    Args:
        merged_by_clade: ``{clade: filtered merged frame}``.
        min_mutations: Keep a clade only if it holds at least this many rows.
            0 keeps everything.

    Returns:
        ``(kept, dropped)``. ``kept`` is the sorted list of clades meeting the
        threshold; ``dropped`` is a frame of ``clade``/``n_mutations`` for the
        rest, ascending, so a caller can name what it excluded rather than
        silently showing fewer dots.
    """
    counts = {clade: len(frame) for clade, frame in merged_by_clade.items()}
    kept = sorted(c for c, n in counts.items() if n >= min_mutations)
    dropped = sorted(
        ((c, n) for c, n in counts.items() if n < min_mutations), key=lambda cn: cn[1]
    )
    return kept, pd.DataFrame(dropped, columns=["clade", "n_mutations"])


def apply_count_threshold(
    merged: pd.DataFrame,
    threshold: float,
    *,
    count_filter_mode: Literal["and", "or"] = COUNT_FILTER_MODE,
) -> pd.DataFrame:
    """Re-apply the count filter at one rung of :data:`COUNT_THRESHOLD_LADDER`.

    **Both thresholds move together.** A rung is one number applied to
    ``actual_count`` and to ``predicted_count`` alike -- mat-pcp's
    ``expected_count``, renamed by
    :func:`~dasm2.analysis.viral.normalize_fitness_columns`. Sweeping them
    independently would be a two-dimensional grid whose off-diagonal cells mix
    two different questions, and with ``count_filter_mode="or"`` the diagonal is
    the one that reads as "this mutation rests on at least ``threshold``
    observations *or* at least ``threshold`` expected traffic".

    The filter is monotone in the threshold, so this composes: applying rung 50
    to a frame already filtered at rung 20 gives the same rows as applying 50 to
    the unfiltered frame. It is still the caller's job to start from a frame
    whose count filter is *off* -- pass ``min_actual_count=0`` and
    ``min_predicted_count=0`` to :func:`merge_and_filter` -- because starting
    from a frame filtered above a rung would silently report that frame's
    threshold for every rung below it.

    Args:
        merged: A merged fitness+SF frame, normally from
            :func:`merge_and_filter` with both count thresholds at 0.
        threshold: The minimum count, applied to both count columns.
        count_filter_mode: ``"or"`` or ``"and"``; see :data:`COUNT_FILTER_MODE`.

    Returns:
        The subset of ``merged`` meeting the threshold. Nothing else is
        filtered and no value is altered, so the frame keeps the clipping and
        the site restrictions it arrived with.
    """
    return filter_merged_fitness(
        merged,
        min_actual_count=threshold,
        min_predicted_count=threshold,
        count_filter_mode=count_filter_mode,
    )
