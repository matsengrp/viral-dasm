"""Model-external evaluation of viral DASMs against mat-pcp's fitness effects.

A fork of the upstream module of the same name, carrying what scores a
checkpoint -- the forward pass, the filter chain, the clip and the clade
adequacy floor -- and the agreement statistics `fig:agreement` plots. The run
axis, the manifest resolution and the effect-error family are not here:
`config.yaml` names every path this pipeline reads, and the readouts for the
other figures arrive with those figures.

What this module owns:

* the forward pass that turns a checkpoint and a clade founder into
  per-mutation log selection factors (:class:`SelectionFactorSource`),
* which fitness column counts as "the fitness effect"
  (:data:`FITNESS_METRICS`),
* how fitness rows join to log selection factors, which of them survive the
  count filters, and the common lower bound both axes are clamped to before
  anything is correlated (:func:`merge_and_filter`, :data:`CLIP_LOWER`),
* when a clade holds too few mutations to correlate at all
  (:func:`adequate_clades`),
* the agreement statistics themselves: per-clade correlation
  (:func:`within_clade_metrics`), per-pair cross-clade shift correlation
  (:func:`shift_pair_metrics`, :func:`shift_frame`), the count-ladder sweep of
  both (:func:`within_clade_count_ladder`, :func:`shift_pair_count_ladder`,
  :func:`count_ladder_adequacy`), which clade pairs no model can answer
  (:func:`identical_founder_pairs`), and the band a count supports under
  Poisson noise (:func:`poisson_effect_interval`,
  :func:`poisson_effect_p_value`).

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

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, cast

import numpy as np
import pandas as pd
import torch
from scipy import stats

from dasm2.analysis.viral import (
    build_sf_dataframe_from_tensor,
    compute_concordance,
    compute_cross_clade_shift,
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

# A shift smaller than this counts as "no movement" rather than as a direction.
# One constant for both the per-mutation `concordant` label and the aggregate
# concordance, since a dashboard shows them side by side and two epsilons would
# let the point colours contradict the number above them.
SHIFT_EPS = 1e-10

# A cross-clade shift correlation answers "does the model track how a
# mutation's fitness *changed*". On a pair whose shared mutations all moved a
# little, both axes are mostly counting noise and r measures the agreement of
# two noise processes -- it comes out near zero and says nothing about the
# model. This is the magnitude at which a pair's *counts-based* shift counts as
# a real change rather than noise, and `shift_pair_metrics` reports how many
# such mutations each pair holds.
#
# The threshold is on the counts-based shift, not the model's, because the
# question is whether the *measurement* resolved a real change; a model can
# predict a large shift that the counts never saw.
#
# 3 in log-ratio units is a ~20x change in relative fitness -- far outside what
# count noise produces at these depths. Reported, never filtered on: no panel
# here gates a pair on it, so the threshold can move without any correlation
# being recomputed.
LARGE_SHIFT_THRESHOLD = 3.0


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


def identical_founder_pairs(
    founders: dict[str, str], clades: Sequence[str]
) -> list[tuple[str, str]]:
    """Unordered clade pairs whose founder is identical in this gene.

    Such a pair is the same input to the model twice, so **no** model can show a
    shift between them -- the transformer included, since its only clade input is
    the founder sequence. Meanwhile mat-pcp's Δfitness is nonzero, because that
    comes from each clade's own tree counts. The pair is therefore not evidence
    about any model and belongs out of the shift statistics rather than in them at
    r = 0.

    This is common rather than exceptional for the short genes, and it is not a
    defect: two clades differ elsewhere in the genome without differing here.
    How many pairs it removes depends on the clade set, so it is computed from
    the founders rather than tabulated: Spike is long enough that identical
    founders are rare, and the short genes are where the case bites.

    Args:
        founders: ``{clade: founder amino acid sequence}`` for one gene.
        clades: Clades to consider, in any order.

    Returns:
        ``(a, b)`` pairs with ``a`` before ``b`` in ``clades``.
    """
    clades = list(clades)
    return [
        (a, b)
        for index, a in enumerate(clades)
        for b in clades[index + 1 :]
        if founders[a] == founders[b]
    ]


def shift_frame(
    merged_a: pd.DataFrame,
    merged_b: pd.DataFrame,
    clade_a: str,
    clade_b: str,
    fitness_col: str,
) -> pd.DataFrame:
    """Per-mutation cross-clade shift for one clade pair, labelled for plotting.

    Wraps :func:`dasm2.analysis.viral.compute_cross_clade_shift`, which pairs the
    two clades on ``(aa_site, mutant_aa)`` and then keeps only mutations whose
    wildtype residue agrees, so D614G in one clade is never compared with E614G
    in the other.

    Adds what a scatter needs beyond the shifts themselves: the pair label, and a
    three-way ``concordant`` category. ``no shift`` is its own value rather than
    folded into ``disagree`` because those mutations are excluded from the
    concordance denominator -- colouring them as disagreements would contradict
    the number reported beside them.

    Returns:
        The shift frame with ``pair``, ``concordant`` and ``model``-free columns;
        rows with a NaN shift on either axis are dropped.
    """
    shift = compute_cross_clade_shift(merged_a, merged_b, fitness_col=fitness_col)
    shift = shift.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["sf_shift", "fitness_shift"]
    )
    shift = shift.assign(pair=f"{clade_a}→{clade_b}")
    moved = (shift["sf_shift"].abs() > SHIFT_EPS) & (
        shift["fitness_shift"].abs() > SHIFT_EPS
    )
    agree = np.sign(shift["sf_shift"]) == np.sign(shift["fitness_shift"])
    shift["concordant"] = np.where(
        ~moved, "no shift", np.where(agree, "agree", "disagree")
    )
    return shift


def shift_pair_metrics(
    merged_by_clade: dict[str, pd.DataFrame],
    clades: Sequence[str],
    fitness_col: str,
    min_mutations: int = 3,
    founders: Optional[dict[str, str]] = None,
    large_shift_threshold: float = LARGE_SHIFT_THRESHOLD,
) -> pd.DataFrame:
    """Shift correlation for every unordered pair drawn from ``clades``.

    All pairs rather than a fixed reference clade: with a reference every dot
    shares an endpoint, so the spread partly reflects which clade was picked.
    Pair order is ``(a, b)`` with ``a`` before ``b`` in ``clades``, and both
    shifts are ``b - a``, so flipping a pair would flip both signs and leave
    Pearson r unchanged.

    **The one interpretive step.** ``PerSiteParentChildHead`` over an
    ``IdentityEncoder`` makes the context-independent baseline's selection factor
    a function of (site, parent residue, child residue) and nothing else. This
    comparison keeps only mutations whose parent residue agrees between the two
    clades, so that model returns the identical number on both sides and its
    ΔSF is exactly 0 for every mutation -- one distinct value, ``max |ΔSF| =
    0``. Pearson r on a constant predictor is
    undefined (0/0), and this reports **0** for that case, flagged in
    ``sf_shift_is_constant``. A predictor whose shift carries no information
    about the target has expected r = 0, and a constant predictor is the limit of
    those, so 0 says what is true -- the model explains none of the cross-clade
    variation -- while NaN reads as missing data and blocks the margin
    arithmetic. Read such a column as a floor, not a fit: cross-clade shift is a
    question only a context-dependent model can answer at all.

    Pairs that are thin rather than degenerate keep their NaN, because that
    genuinely is missing data.

    :func:`~dasm2.analysis.viral.compute_shift_r_all_pairs` is the same
    computation reporting NaN for the constant-predictor case rather than 0.
    The two are not interchangeable at exactly that case -- which is the
    context-independent baseline, the contrast this comparison rests on --
    and everywhere else they agree.

    **The substitution is about the model, so pairs no model can answer are
    dropped instead.** Pass ``founders`` and any pair whose founder is identical
    in this gene is omitted from the output entirely -- see
    :func:`identical_founder_pairs`. Without that, such a pair enters the
    transformer's column at r = 0 and drags its median toward the baseline's
    floor for a reason that has nothing to do with the transformer. How much it
    drags depends on how many of the pairs are degenerate, so the fix is to drop
    them rather than to bound the damage.

    Args:
        merged_by_clade: ``{clade: filtered merged frame}``. Must contain every
            entry of ``clades``.
        clades: Clades to draw pairs from. Fewer than two gives an empty frame
            with the full column set.
        fitness_col: Which mat-pcp column carries the fitness effect; one of
            :data:`FITNESS_METRICS`.
        min_mutations: Below this many shared mutations a pair reports NaN
            without attempting a correlation. 3 is the floor for Pearson r.
        founders: ``{clade: founder sequence}`` for this gene. When given, pairs
            with identical founders are omitted. When omitted, every pair is
            scored and the degenerate ones show up as flagged zeros.
        large_shift_threshold: Magnitude at which a counts-based shift counts as
            large, for the ``n_large_shifts`` column. See
            :data:`LARGE_SHIFT_THRESHOLD`. Reported, never filtered on, so the
            threshold can move without the correlations being recomputed.

    Returns:
        One row per pair: ``clade_a``, ``clade_b``, ``pair_label``,
        ``pearson_r``, ``spearman_rho``, ``directional_concordance``,
        ``n_mutations``, ``n_large_shifts``, ``sf_shift_is_constant``.

    Raises:
        ValueError: If ``clades`` repeats a clade (a self-pair would report a
            perfect correlation) or names one absent from ``merged_by_clade``.
    """
    clades = list(clades)
    if len(set(clades)) != len(clades):
        duplicated = sorted({c for c in clades if clades.count(c) > 1})
        raise ValueError(
            f"clades contains duplicates {duplicated}; a repeated clade would "
            "pair with itself and report a perfect correlation."
        )
    absent = [c for c in clades if c not in merged_by_clade]
    if absent:
        raise ValueError(f"clades not in merged_by_clade: {absent}")

    columns = [
        "clade_a",
        "clade_b",
        "pair_label",
        "pearson_r",
        "spearman_rho",
        "directional_concordance",
        "n_mutations",
        "n_large_shifts",
        "sf_shift_is_constant",
    ]

    skipped = (
        set(identical_founder_pairs(founders, clades))
        if founders is not None
        else set()
    )

    rows = []
    for index, clade_a in enumerate(clades):
        for clade_b in clades[index + 1 :]:
            if (clade_a, clade_b) in skipped:
                continue
            shift = shift_frame(
                merged_by_clade[clade_a],
                merged_by_clade[clade_b],
                clade_a,
                clade_b,
                fitness_col,
            )
            row = {
                "clade_a": clade_a,
                "clade_b": clade_b,
                "pair_label": f"{clade_a}→{clade_b}",
                "n_mutations": len(shift),
                # Counted on every pair, including ones too small to correlate,
                # so the gate can tell "no large shifts" from "not measured".
                "n_large_shifts": int(
                    (shift["fitness_shift"].abs() >= large_shift_threshold).sum()
                )
                if len(shift)
                else 0,
                "sf_shift_is_constant": False,
                "pearson_r": float("nan"),
                "spearman_rho": float("nan"),
                "directional_concordance": float("nan"),
            }
            if len(shift) < min_mutations:
                rows.append(row)
                continue

            sf_constant = bool(np.ptp(shift["sf_shift"].to_numpy()) == 0)
            fitness_constant = bool(np.ptp(shift["fitness_shift"].to_numpy()) == 0)
            row["sf_shift_is_constant"] = sf_constant
            if sf_constant and not fitness_constant:
                # Undefined, reported as 0. See this function's docstring.
                row["pearson_r"] = 0.0
                row["spearman_rho"] = 0.0
            elif not (sf_constant or fitness_constant):
                row["pearson_r"] = float(
                    stats.pearsonr(shift["sf_shift"], shift["fitness_shift"])[0]
                )
                row["spearman_rho"] = float(
                    stats.spearmanr(shift["sf_shift"], shift["fitness_shift"])[0]
                )
            concordance = compute_concordance(shift["sf_shift"], shift["fitness_shift"])
            row["directional_concordance"] = (
                float("nan") if np.isnan(concordance) else float(concordance)
            )
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def within_clade_metrics(
    merged_by_clade: dict[str, pd.DataFrame],
    fitness_col: str,
    min_mutations: int = 3,
) -> pd.DataFrame:
    """Per-clade agreement between log SF and the mat-pcp fitness effect.

    Filtering must already have happened; this correlates what it is given.

    ``regression_slope`` is the ordinary-least-squares slope of log SF **on**
    the fitness effect, matching mat-pcp's `subsample_correlations.py` so the
    two sides of a counts-vs-DASM comparison are the same statistic. It is
    reported beside r because r cannot see compression: a model whose estimates
    are a flattened copy of the target scores a high r and a slope far below 1,
    and the slope is the only one of the two that says so.

    **The slope is attenuated, and deliberately so.** OLS with error in the
    regressor is biased toward zero by ``var(true) / (var(true) + var(error))``
    -- textbook regression dilution -- and the fitness target carries plenty of
    count noise. It is still the right statistic here for two reasons: every
    arm is regressed on the *same* target, so the attenuation factor is shared
    and the arms remain comparable; and an errors-in-variables estimator
    (Deming, orthogonal) needs the error-variance ratio, which nothing here
    measures. Read the slope as a comparand between arms, not as an absolute
    calibration -- a slope of 1.0 is not the no-bias reference it would be with
    a noiseless x.

    Args:
        merged_by_clade: ``{clade: filtered merged frame}``.
        fitness_col: One of :data:`FITNESS_METRICS`.
        min_mutations: Below this many rows the clade reports NaN.

    Returns:
        One row per clade: ``clade``, ``pearson_r``, ``spearman_rho``,
        ``regression_slope``, ``n_mutations``, sorted by clade.

    Raises:
        ValueError: If a frame lacks ``log_selection_factor`` or ``fitness_col``.
    """
    rows = []
    for clade in sorted(merged_by_clade):
        frame = merged_by_clade[clade]
        for column in ("log_selection_factor", fitness_col):
            if column not in frame.columns:
                raise ValueError(
                    f"clade {clade!r} frame has no {column!r} column; columns are "
                    f"{sorted(frame.columns)}"
                )
        valid = frame.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["log_selection_factor", fitness_col]
        )
        row = {
            "clade": clade,
            "n_mutations": len(valid),
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "regression_slope": float("nan"),
        }
        if len(valid) >= min_mutations and valid[fitness_col].nunique() > 1:
            row["pearson_r"] = float(
                stats.pearsonr(valid["log_selection_factor"], valid[fitness_col])[0]
            )
            row["spearman_rho"] = float(
                stats.spearmanr(valid["log_selection_factor"], valid[fitness_col])[0]
            )
            # Spelled out rather than via `np.polyfit`, to match mat-pcp's
            # `subsample_correlations.py` term for term. The `nunique() > 1`
            # guard above is what makes the denominator non-zero.
            x = valid[fitness_col].to_numpy(dtype=float)
            y = valid["log_selection_factor"].to_numpy(dtype=float)
            row["regression_slope"] = float(
                ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
            )
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "clade",
            "pearson_r",
            "spearman_rho",
            "regression_slope",
            "n_mutations",
        ],
    )


def count_ladder_adequacy(
    base_by_clade: dict[str, pd.DataFrame],
    *,
    thresholds: Sequence[float] = COUNT_THRESHOLD_LADDER,
    count_filter_mode: Literal["and", "or"] = COUNT_FILTER_MODE,
    min_mutations: int = DEFAULT_MIN_CLADE_MUTATIONS,
) -> tuple[list[str], pd.DataFrame]:
    """Clades usable at *every* rung, and the per-rung adequacy behind that.

    **This is the function that keeps the sweep honest.** Raising the count
    floor removes mutations, which removes clades through
    :func:`adequate_clades` -- and the clades it removes first are the thin,
    poorly-measured ones whose r is lowest. A sweep that let each rung keep its
    own clade set would therefore show r rising with the threshold even if no
    surviving clade's r moved at all: the trend would be survivorship, not
    precision. Restricting every rung to the clades that clear the floor at all
    of them makes the rungs paired, so a line that moves moved because the
    mutations changed.

    The cost is that the strictest rung sets the clade set for the whole sweep,
    which is why ``adequacy`` is returned rather than just the list: it names
    what each rung would have kept on its own, so the size of what pairing gave
    up is visible instead of assumed.

    Args:
        base_by_clade: ``{clade: merged frame with the count filter off}``.
        thresholds: The ladder. An empty ladder gives no paired clades.
        count_filter_mode: ``"or"`` or ``"and"``; see :data:`COUNT_FILTER_MODE`.
        min_mutations: The per-clade floor, as in :func:`adequate_clades`.

    Returns:
        ``(paired, adequacy)``. ``paired`` is the sorted list of clades meeting
        the floor at every rung. ``adequacy`` has one row per (rung, clade) with
        ``min_count``, ``clade``, ``n_mutations`` and ``adequate``, so a caller
        can report both the pairing and what each rung dropped.
    """
    rows = []
    adequate_sets = []
    for threshold in thresholds:
        filtered = {
            clade: apply_count_threshold(
                frame, threshold, count_filter_mode=count_filter_mode
            )
            for clade, frame in base_by_clade.items()
        }
        kept, _ = adequate_clades(filtered, min_mutations)
        adequate_sets.append(set(kept))
        for clade, frame in sorted(filtered.items()):
            rows.append(
                {
                    "min_count": threshold,
                    "clade": clade,
                    "n_mutations": len(frame),
                    "adequate": clade in set(kept),
                }
            )
    paired = sorted(set.intersection(*adequate_sets)) if adequate_sets else []
    return paired, pd.DataFrame(
        rows, columns=["min_count", "clade", "n_mutations", "adequate"]
    )


def within_clade_count_ladder(
    base_by_clade: dict[str, pd.DataFrame],
    fitness_col: str,
    clades: Sequence[str],
    *,
    thresholds: Sequence[float] = COUNT_THRESHOLD_LADDER,
    count_filter_mode: Literal["and", "or"] = COUNT_FILTER_MODE,
) -> pd.DataFrame:
    """Within-clade metrics at each rung of the count ladder.

    :func:`within_clade_metrics` applied once per rung to the same ``clades``,
    with the rung recorded in a ``min_count`` column. Pass the paired clade set
    from :func:`count_ladder_adequacy` so a moving line is not survivorship.

    Args:
        base_by_clade: ``{clade: merged frame with the count filter off}``. Must
            contain every entry of ``clades``.
        fitness_col: One of :data:`FITNESS_METRICS`.
        clades: The clades to score, the same at every rung.
        thresholds: The ladder.
        count_filter_mode: ``"or"`` or ``"and"``; see :data:`COUNT_FILTER_MODE`.

    Returns:
        :func:`within_clade_metrics`' columns plus ``min_count``, one row per
        (rung, clade). Empty with the full column set if ``clades`` is empty.

    Raises:
        KeyError: If a clade in ``clades`` is absent from ``base_by_clade``.
    """
    frames = []
    for threshold in thresholds:
        filtered = {
            clade: apply_count_threshold(
                base_by_clade[clade], threshold, count_filter_mode=count_filter_mode
            )
            for clade in clades
        }
        metrics = within_clade_metrics(filtered, fitness_col)
        frames.append(metrics.assign(min_count=threshold))
    if not frames:
        return pd.DataFrame(
            columns=[
                "clade",
                "pearson_r",
                "spearman_rho",
                "regression_slope",
                "n_mutations",
                "min_count",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def shift_pair_count_ladder(
    base_by_clade: dict[str, pd.DataFrame],
    fitness_col: str,
    clades: Sequence[str],
    *,
    thresholds: Sequence[float] = COUNT_THRESHOLD_LADDER,
    count_filter_mode: Literal["and", "or"] = COUNT_FILTER_MODE,
    founders: Optional[dict[str, str]] = None,
    large_shift_threshold: float = LARGE_SHIFT_THRESHOLD,
) -> pd.DataFrame:
    """Shift-pair metrics at each rung of the count ladder.

    :func:`shift_pair_metrics` applied once per rung, with the rung recorded in
    a ``min_count`` column.

    A pair's shift is computed over the mutations the two clades *share*, so
    raising the floor shrinks a pair's population faster than it shrinks either
    clade's own -- a mutation is lost from the pair as soon as *either* side
    falls below the threshold. Read ``n_mutations`` per rung rather than
    assuming the pairs thin at the same rate the clades do.

    Args:
        base_by_clade: ``{clade: merged frame with the count filter off}``. Must
            contain every entry of ``clades``.
        fitness_col: One of :data:`FITNESS_METRICS`.
        clades: The clades to draw pairs from, the same at every rung. Fewer
            than two gives an empty frame.
        thresholds: The ladder.
        count_filter_mode: ``"or"`` or ``"and"``; see :data:`COUNT_FILTER_MODE`.
        founders: ``{clade: founder sequence}``; pairs with identical founders
            are dropped, as in :func:`shift_pair_metrics`. Founders do not
            depend on the count filter, so the same pairs are dropped at every
            rung.
        large_shift_threshold: Passed to :func:`shift_pair_metrics` for its
            ``n_large_shifts`` column, which is what
            reports.

    Returns:
        :func:`shift_pair_metrics`' columns plus ``min_count``, one row per
        (rung, pair).

    Raises:
        KeyError: If a clade in ``clades`` is absent from ``base_by_clade``.
    """
    frames = []
    for threshold in thresholds:
        filtered = {
            clade: apply_count_threshold(
                base_by_clade[clade], threshold, count_filter_mode=count_filter_mode
            )
            for clade in clades
        }
        pairs = shift_pair_metrics(
            filtered,
            clades,
            fitness_col,
            founders=founders,
            large_shift_threshold=large_shift_threshold,
        )
        frames.append(pairs.assign(min_count=threshold))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# Coverage of the count-noise interval below. 0.95 because the question it
# answers -- "is the model's effect inside what this count could support" -- is
# read as a yes/no, and a yes/no needs a conventional level rather than a
# tunable one.
COUNT_NOISE_COVERAGE = 0.95

# mat-pcp's `delta_fitness_ratio` is `log((a + 0.5) / (e + 0.5))`, and the two
# functions below take that pseudocount as a parameter rather than assuming it.
# It is not a detail: an interval computed on `log(lambda / e)` while the
# plotted point uses the pseudocount puts the point *outside its own band*
# wherever the expected count is small -- the shift is `log((e + 0.5) / e)`,
# which is 0.26 at e = 1.7 against a half-width of 0.23. Pass the pseudocount
# the plotted metric actually uses and the point is inside by construction.
RATIO_PSEUDOCOUNT = 0.5


def poisson_effect_interval(
    actual: Sequence[float] | np.ndarray | pd.Series,
    expected: Sequence[float] | np.ndarray | pd.Series,
    *,
    coverage: float = COUNT_NOISE_COVERAGE,
    pseudocount: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Log effects a count is consistent with, under noise in the count alone.

    The counts-based effect is a log ratio of an observed count to a neutral
    model's expected count. Treating the expected count as exact and the
    observed count as ``Poisson(expected * exp(f))``, this inverts the exact
    (Garwood) Poisson test to give every ``f`` the observation does not reject:

    .. math::

        \\lambda_{lo} = \\tfrac{1}{2}\\chi^2_{\\alpha/2}(2a), \\quad
        \\lambda_{hi} = \\tfrac{1}{2}\\chi^2_{1-\\alpha/2}(2a+2), \\quad
        f \\in \\left[\\log\\frac{\\lambda_{lo}+c}{e+c},
                      \\log\\frac{\\lambda_{hi}+c}{e+c}\\right]

    for pseudocount :math:`c`.

    **Why the expected count is treated as exact.** This interval exists to ask
    whether a *model's* effect estimate sits inside what the counts can
    support, and the model and the counts are anchored to the same neutral
    expectation. An error in that expectation moves both sides together, so it
    is common mode for this comparison and belongs outside the interval. That
    makes this narrower than mat-pcp's published ``uncertainty``, which is the
    posterior SD of ``delta_fitness_bayesian`` and deliberately includes the
    expected-count term (the GLM's residual variance) and a prior. The two are
    answering different questions; see the module docstring's note on metrics.

    **Match the pseudocount to the estimate being bounded.** ``pseudocount=0``
    bounds :math:`\\log(\\lambda/e)`, the quantity a model of selection
    estimates, and at ``a = 0`` its lower bound is ``-inf``: a count of zero
    cannot bound an effect from below. ``pseudocount=0.5`` bounds
    :data:`RATIO_PSEUDOCOUNT`-shifted ``delta_fitness_ratio`` instead, which is
    what to use when the band is drawn under that column -- it keeps the
    plotted point inside the band at every expected count, and at ``a = 0`` the
    point estimate sits exactly on the lower edge rather than off the bottom of
    the panel. The two differ by ``log((e + c) / e)``, which is negligible at
    large expected counts and is not at small ones.

    Args:
        actual: Observed counts. Non-integer values are used as given; the
            chi-square form is continuous in ``a``.
        expected: Neutral-model expected counts, treated as exact.
        coverage: Two-sided coverage, e.g. 0.95.
        pseudocount: Added to both counts before the log, matching the point
            estimator the band is drawn under. 0 bounds ``log(lambda / e)``.

    Returns:
        ``(f_lo, f_hi)``, both float arrays shaped like the inputs. With
        ``pseudocount=0``, ``f_lo`` is ``-inf`` where ``actual == 0``. Both are
        ``nan`` where a count is missing or the expected count is not positive,
        since there is no ratio to bound.

    Raises:
        ValueError: If ``coverage`` is not in ``(0, 1)``, ``pseudocount`` is
            negative, or the two inputs have different lengths.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must lie in (0, 1), got {coverage!r}.")
    if pseudocount < 0.0:
        raise ValueError(f"pseudocount must be >= 0, got {pseudocount!r}.")
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    if a.shape != e.shape:
        raise ValueError(
            f"actual and expected must have the same shape, got {a.shape} and "
            f"{e.shape}."
        )
    alpha = 1.0 - coverage
    usable = np.isfinite(a) & np.isfinite(e) & (e > 0) & (a >= 0)
    a_safe = np.where(usable, a, 0.0)
    # `gamma.ppf(q, a)` is `0.5 * chi2.ppf(q, 2a)`; at a = 0 the lower bound is
    # exactly 0, which scipy returns as nan for shape 0, so it is set directly.
    lam_lo = np.where(
        a_safe > 0, stats.gamma.ppf(alpha / 2.0, np.maximum(a_safe, 1e-300)), 0.0
    )
    lam_hi = stats.gamma.ppf(1.0 - alpha / 2.0, a_safe + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_lo = np.log((lam_lo + pseudocount) / (e + pseudocount))
        f_hi = np.log((lam_hi + pseudocount) / (e + pseudocount))
    return (
        np.where(usable, f_lo, np.nan),
        np.where(usable, f_hi, np.nan),
    )


def poisson_effect_p_value(
    actual: Sequence[float] | np.ndarray | pd.Series,
    expected: Sequence[float] | np.ndarray | pd.Series,
    log_effect: Sequence[float] | np.ndarray | pd.Series,
    *,
    pseudocount: float = 0.0,
) -> np.ndarray:
    """Two-sided exact Poisson p-value for one log effect being the truth.

    Under the null that ``log_effect`` is correct, the count should be a draw
    from ``Poisson(lambda)`` for the rate that log effect implies,
    ``lambda = (expected + c) * exp(log_effect) - c``. This returns
    ``2 * min(P(A <= a), P(A >= a))``, capped at 1.

    ``pseudocount`` must match the one :func:`poisson_effect_interval` was
    given, and defaults the same way, because it is the map between an effect
    and a rate. A null whose implied rate is not positive is rejected outright
    (``p = 0``): that can only happen for effects at or below the interval's
    lower edge, so the two stay consistent there too.

    This is the same statement :func:`poisson_effect_interval` draws: an effect
    is inside the interval at coverage ``1 - alpha`` exactly when its p-value
    here is at least ``alpha``. Both are provided because a picture wants the
    interval and a tooltip wants the number, and computing them from one model
    in one place is what stops the shaded band from disagreeing with the number
    printed on top of it.

    Args:
        actual: Observed counts.
        expected: Neutral-model expected counts, treated as exact.
        log_effect: The effect under test, on the same log scale as
            ``poisson_effect_interval`` returns.
        pseudocount: As in :func:`poisson_effect_interval`; must match it.

    Returns:
        Float array shaped like the inputs, ``nan`` where any input is missing
        or the implied rate is not positive.

    Raises:
        ValueError: If ``pseudocount`` is negative or the inputs do not all
            have the same shape.
    """
    if pseudocount < 0.0:
        raise ValueError(f"pseudocount must be >= 0, got {pseudocount!r}.")
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    f = np.asarray(log_effect, dtype=float)
    if not (a.shape == e.shape == f.shape):
        raise ValueError(
            "actual, expected and log_effect must have the same shape, got "
            f"{a.shape}, {e.shape} and {f.shape}."
        )
    usable = np.isfinite(a) & np.isfinite(e) & np.isfinite(f) & (e > 0) & (a >= 0)
    with np.errstate(over="ignore", invalid="ignore"):
        lam = np.where(usable, (e + pseudocount) * np.exp(f) - pseudocount, 1.0)
    # A null predicting a non-positive rate is outside the model, and it sits
    # at or below the interval's lower edge by construction, so 0 is both the
    # limit of the p-value as the rate goes to zero and the answer that agrees
    # with the band.
    rejected = usable & np.isfinite(lam) & (lam <= 0)
    usable = usable & np.isfinite(lam) & (lam > 0)
    lam_safe = np.where(usable, lam, 1.0)
    a_safe = np.where(usable, a, 0.0)
    lower = stats.poisson.cdf(a_safe, lam_safe)
    upper = stats.poisson.sf(a_safe - 1.0, lam_safe)
    p = np.minimum(1.0, 2.0 * np.minimum(lower, upper))
    return np.where(usable, p, np.where(rejected, 0.0, np.nan))
