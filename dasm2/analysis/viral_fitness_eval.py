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
  :func:`poisson_effect_p_value`),
* how far a predicted shift sits from the measured one, per mutation and per
  pair, and that error summarised in fixed windows of the measured shift
  (:func:`shift_error_frame`, :func:`shift_error_quantiles`,
  :func:`shift_error_trend`), which is `si.tex`'s shift-error figure.

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
# The spacing is geometric because that is how count precision improves: the
# standard error on a log count ratio falls as 1/sqrt(n), so 10 -> 20 buys the
# same factor as 50 -> 100, and equal *additive* steps would spend every rung in
# the regime where nothing changes. Raising the floor sharpens the fitness
# estimates that remain while shrinking their number, and past some point it
# removes whole clades through the adequacy floor rather than trimming
# mutations; where that point falls is a property of the run, so the sweep
# reports the surviving counts per rung instead of asserting one here.
COUNT_THRESHOLD_LADDER: list[int] = [10, 20, 50, 100]

# Below this many surviving mutations a clade's Pearson r is not a measurement.
# 20 is a floor, not a sufficiency threshold: it exists because the alternative
# is worse, not because 20 mutations make a solid correlation.
#
# What it is for: mat-pcp emits fitness for every clade, including clades the
# count filter empties. A clade left with no rows contributes a NaN dot to a
# within-clade panel and an all-NaN pair to every clade pair it appears in, so
# the floor has to be applied before a correlation is taken rather than after.
DEFAULT_MIN_CLADE_MUTATIONS = 20

# Common lower bound applied to BOTH axes before anything is correlated or
# plotted.
#
# Both quantities have a long left tail, where "lethal" and "very lethal" are
# not meaningfully distinguishable but do dominate a least-squares fit and
# squash the bulk of the cloud into a corner. Clamping both to one bound keeps
# them on a single scale and keeps the correlation about the region where the
# two measures actually carry information. A bound beneath every observed value
# is equivalent to no clipping, which is how a caller turns this off.
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
# concordance, since a panel shows them side by side and two epsilons would let
# the point colours contradict the number above them.
SHIFT_EPS = 1e-10

# The magnitude at which a clade pair's *counts-based* shift counts as a real
# change rather than noise: 3 in log-ratio units, which is an e^3 ~ 20x change
# in relative fitness.
#
# **No panel in this repository reads it.** Upstream gated which pairs entered a
# shift panel on how many large shifts they held; that gate is not ported, so
# `shift_pair_metrics` computes an `n_large_shifts` column nothing plots. The
# constant stays because the column is produced inside a function vendored
# code-identical, and it is reported rather than filtered on, so moving it
# recomputes no correlation.
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
        exclude_sites: Codon sites to drop. **No caller here passes any, and
            the agreement panels deliberately pass none.** The argument's
            original justification -- that mat-pcp's ``exclude`` flag is per
            nucleotide while training masked the whole codon, so masked codons
            keep fitness rows the model never saw -- stopped being true of this
            repository when training moved to the nucleotide-granular
            ``masked_nt_sites_file``. For ``S`` that mask is 18 nucleotides,
            which widen to 51 at codon granularity, and a mutation reachable
            through the other 33 is one both sides estimated. Dropping its codon
            would narrow the comparison rather than make it fair; see #36 and
            ``scripts/panels.py``.
        restrict_to_sites: Keep only these codon sites, applied *before* the
            other filters. This is the site-class split: pass
            :func:`~dasm2.sequence.masking.censored_codon_sites` for the class
            censoring acts on, and its complement for the rest. Composes with
            ``exclude_sites`` rather than replacing it, so a censored *and*
            problematic site is still dropped -- which costs nothing here, since
            the caller passes no ``exclude_sites``.
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
            ``n_large_shifts`` column, which is reported and never filtered
            on here.

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


# ---------------------------------------------------------------------------
# Error of the estimated shift, per mutation and per clade pair
# ---------------------------------------------------------------------------

# The five levels the box and whisker are drawn from: the median, the quartiles
# that make the box, and the 5th and 95th that make the whisker.
#
# Tail quantiles rather than the full range, because sample extremes grow with
# n and this figure's windows differ in n by orders of magnitude: the window
# holding the smallest shifts holds a quarter of a million mutations where the
# window holding the largest holds a few hundred, and a full-range whisker
# would make the first look wider than the second for that reason alone.
SHIFT_ERROR_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)

# The whisker's two ends, as a (low, high) pair out of
# `SHIFT_ERROR_QUANTILES`. Named apart from the five so that the drawn whisker
# and anything that asks which rows fall outside it cannot disagree about where
# the tail starts.
SHIFT_ERROR_WHISKER_QUANTILES = (0.05, 0.95)

# The window grid: fixed-width windows stated in the windowed axis's own units,
# rather than equal-count bins. Equal-count edges move with the data, so two
# runs are not comparable and no edge names anything a reader can state, where
# a fixed window is "the mutations whose counts-based shift was between 2 and 3
# nats", which is a sentence.
#
# The -2.5 anchor is upstream's, for a figure that windows along the effect
# axis, where both axes are clipped at -4 and the anchor keeps the windows
# clear of the floor. **This figure overrides it with 0.0**, because it windows
# along `abs_fitness_shift`, which is non-negative by construction, so the
# anchor is never reached. It is carried because it is the signature default of
# the function below, and dropping it would make that signature a second
# version of upstream's.
SHIFT_ERROR_WINDOW_START = -2.5
SHIFT_ERROR_WINDOW_WIDTH = 1.0

# The default below which a window is omitted, as sampling noise rather than a
# distribution. **This figure overrides it with 1**, so its windows are the
# whole range of the data and every one is drawn alike however thin, with its
# row count in the panel's table. That puts the weight on a number a reader can
# read rather than on an opacity they would have to decode -- but it does mean
# a window at the far end can rest on a handful of mutations, and this constant
# is the line below which that is true.
SHIFT_ERROR_MIN_WINDOW_ROWS = 100


def shift_error_frame(
    merged_by_clade: dict[str, pd.DataFrame],
    clades: Sequence[str],
    fitness_col: str,
    *,
    founders: Optional[dict[str, str]] = None,
    min_shift_magnitude: float = 0.0,
) -> pd.DataFrame:
    """Per-mutation, per-pair error of the model's shift against the counts'.

    :func:`shift_pair_metrics` reduces each pair to one correlation. This keeps
    the rows behind it, so the error can be read against a property of the
    mutation rather than of the pair.

    ``shift_error`` is ``sf_shift - fitness_shift``: the model's predicted
    change in effect between the two clades minus the change mat-pcp's counts
    measured, in nats. Both are differences of log selection factors -- the
    model's from its head, mat-pcp's from a log count ratio -- so the
    subtraction is between commensurable quantities. It is **not** a residual
    from a fit: no slope or intercept is estimated, so a model that ranked
    every shift perfectly but at twice the amplitude would still show a large
    error here. That is the intended reading (the shift is meant to be an
    effect size, not a score), but it means the level of this error is not
    comparable to a regression residual.

    ``max_clade_effect`` is ``max(delta_fitness_a, delta_fitness_b)``: the more
    positive of the two counts-based per-clade effects whose difference *is* the
    measured shift. A mutation that is beneficial in at least one of the two
    clades scores high here however the shift itself came out.

    ``abs_fitness_shift`` is ``|fitness_shift|``, how far the counts say the
    mutation moved regardless of direction. It is the second axis the figures
    window along, and it is one of the error's own two inputs: a model
    predicting no shift at all scores ``abs_shift_error == abs_fitness_shift``
    exactly, so a rise against this axis is only a finding above that line.

    Args:
        merged_by_clade: ``{clade: filtered merged frame}``. The count filter
            must already be applied; this reads what it is given.
        clades: Clades to draw unordered pairs from, ``(a, b)`` with ``a``
            first in this sequence. Fewer than two gives an empty frame with
            the full column set.
        fitness_col: One of :data:`FITNESS_METRICS`.
        founders: ``{clade: founder sequence}``; identical-founder pairs are
            dropped exactly as :func:`shift_pair_metrics` drops them, since no
            model can show a shift across them and their error is therefore a
            property of the clade set rather than of the model.
        min_shift_magnitude: Keep only mutations that moved at least this far
            under **either** approach -- ``|fitness_shift| >= m`` or
            ``|sf_shift| >= m``. 0 (the default) keeps everything, and is
            what this repository's figure passes. The bar is an ``or`` so
            that the model's false alarms -- a shift it predicted that never
            happened -- stay in the same frame as its misses. Note that
            conditioning on either quantity removes the rows where both are
            near zero, which are the easiest rows there are, so the level of
            the error rises under any nonzero bar and only the comparison
            across windows survives it.

    Returns:
        One row per (pair, shared mutation): ``clade_a``, ``clade_b``,
        ``pair_label``, ``mutation_label``, ``aa_site``, ``mutant_aa``,
        ``sf_shift``, ``fitness_shift``, ``shift_error``, ``abs_shift_error``,
        ``abs_fitness_shift``, ``max_clade_effect``, ``min_clade_effect``,
        and -- when the inputs
        carried counts -- ``min_actual_count`` and ``min_expected_count``, the
        weaker of the two clades' counts, which is what limits the pair.

    Raises:
        ValueError: If ``clades`` repeats a clade, names one absent from
            ``merged_by_clade``, or if ``min_shift_magnitude`` is negative.
    """
    clades = list(clades)
    if min_shift_magnitude < 0.0:
        raise ValueError(
            f"min_shift_magnitude must be >= 0, got {min_shift_magnitude!r}."
        )
    if len(set(clades)) != len(clades):
        duplicated = sorted({c for c in clades if clades.count(c) > 1})
        raise ValueError(
            f"clades contains duplicates {duplicated}; a repeated clade would "
            "pair with itself and report zero error."
        )
    absent = [c for c in clades if c not in merged_by_clade]
    if absent:
        raise ValueError(f"clades not in merged_by_clade: {absent}")

    columns = [
        "clade_a",
        "clade_b",
        "pair_label",
        "mutation_label",
        "aa_site",
        "mutant_aa",
        "sf_shift",
        "fitness_shift",
        "shift_error",
        "abs_shift_error",
        "abs_fitness_shift",
        "max_clade_effect",
        "min_clade_effect",
    ]
    count_columns = ["min_actual_count", "min_expected_count"]

    skipped = (
        set(identical_founder_pairs(founders, clades))
        if founders is not None
        else set()
    )

    frames = []
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
            if min_shift_magnitude > 0.0 and len(shift):
                # `or`, not `and`, so a shift the model predicted that never
                # happened is kept. Applied before the derived columns so the
                # kept rows are decided by the two shifts alone.
                shift = shift[
                    (shift["fitness_shift"].abs() >= min_shift_magnitude)
                    | (shift["sf_shift"].abs() >= min_shift_magnitude)
                ]
            if not len(shift):
                continue
            effects = shift[["delta_fitness_a", "delta_fitness_b"]].to_numpy(float)
            row = shift.assign(
                clade_a=clade_a,
                clade_b=clade_b,
                pair_label=f"{clade_a}→{clade_b}",
                shift_error=shift["sf_shift"] - shift["fitness_shift"],
                abs_shift_error=(shift["sf_shift"] - shift["fitness_shift"]).abs(),
                abs_fitness_shift=shift["fitness_shift"].abs(),
                max_clade_effect=effects.max(axis=1),
                min_clade_effect=effects.min(axis=1),
            )
            # `predicted_count` is the canonical spelling
            # `normalize_fitness_columns` writes; `expected_count` is mat-pcp's
            # own, carried through unchanged. Either is read, the canonical one
            # first, so a frame that has only the original still gets the
            # column.
            for name, prefixes in zip(
                count_columns,
                (("actual_count",), ("predicted_count", "expected_count")),
            ):
                for prefix in prefixes:
                    sides = [f"{prefix}_a", f"{prefix}_b"]
                    if all(side in shift.columns for side in sides):
                        row[name] = shift[sides].to_numpy(float).min(axis=1)
                        break
            frames.append(row[[c for c in columns + count_columns if c in row.columns]])

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def window_label(lo: float, hi: float) -> str:
    """A window's range, as the string a discrete axis labels it with.

    The figure's x axis is discrete -- one band per window -- so each band has
    to name the interval it summarises. A quantitative axis with a point at
    the window centre invites reading between windows and reading the centre
    as a measurement; neither is what a window is.

    Rendered as ``"0-1"`` with an en dash, and with spaces around the dash
    when either edge is negative, since ``"-2.5--1.5"`` is unreadable.

    Args:
        lo: Left edge, inclusive.
        hi: Right edge, exclusive.
    """
    dash = " \u2013 " if lo < 0 or hi < 0 else "\u2013"
    return f"{lo:g}{dash}{hi:g}"


def shift_error_quantiles(
    error_df: pd.DataFrame,
    *,
    value_col: str = "max_clade_effect",
    error_col: str = "abs_shift_error",
    start: float = SHIFT_ERROR_WINDOW_START,
    width: float = SHIFT_ERROR_WINDOW_WIDTH,
    step: Optional[float] = None,
    stop: Optional[float] = None,
    min_rows: int = SHIFT_ERROR_MIN_WINDOW_ROWS,
) -> pd.DataFrame:
    """Summarise the error inside fixed windows along the effect axis.

    The distribution per window rather than its mean: the error is a
    heavy-tailed positive quantity, so a mean moves with a handful of rows and a
    difference of means between windows says less than the quantiles do.

    Windows are **fixed-width and stated in the effect's own units** --
    ``[start, start + width)``, then the same window advanced by ``step``,
    until the data runs out. Every window therefore means something a reader
    can name ("mutations whose more positive clade effect was between -0.5 and
    +0.5"), the edges do not move with the sample, and the window a mutation
    falls in does not depend on the other mutations. Intervals are half-open on
    the right, so with ``step == width`` (the default) each row lands in
    exactly one.

    ``step`` below ``width`` gives overlapping windows -- a genuine sliding
    window, smoother but with neighbouring points sharing rows, which is worth
    knowing before reading a trend off them.

    Rows below ``start`` are **not** binned. See
    :data:`SHIFT_ERROR_WINDOW_START`: the counts-based effect and the model's
    log SF are clipped at the same bound, so near it both axes are compressed
    and the error narrows for a reason that is not the model. The caller has
    the frame and can count what the floor left out; this function does not
    hide it, it declines to draw it.

    Args:
        error_df: Rows from :func:`shift_error_frame`. An empty frame, or one
            with no usable rows at or above ``start``, gives an empty result
            rather than raising, so a notebook can draw nothing without a
            guard.
        value_col: The column windowed along x.
        error_col: The column summarised in each window.
        start: Left edge of the first window.
        width: Window width, in the units of ``value_col``.
        step: Distance between consecutive left edges. Defaults to ``width``
            (adjacent, non-overlapping windows).
        stop: Stop once a window's left edge reaches this; defaults to the
            largest usable ``value_col``, so the grid covers the data.
        min_rows: Windows holding fewer rows than this are omitted.

    Returns:
        One row per drawn window, ascending: ``bin`` (the window's index in the
        grid, so a gap is visible as a skipped index), ``n``, ``value_lo``,
        ``value_hi``, ``value_label`` (the range as
        :func:`window_label` renders it, for a discrete axis),
        ``value_center``, ``value_median``, ``error_min``,
        ``error_max``, and one ``error_p{q}`` column per entry of
        :data:`SHIFT_ERROR_QUANTILES` (``error_p5``, ``error_p25``,
        ``error_p50``, ``error_p75``, ``error_p95``). ``error_min`` and
        ``error_max`` are the window's extremes, kept for reference; the
        drawn whisker is ``error_p5`` to ``error_p95``. The figure draws one
        band per window, labelled ``value_label``. ``value_center`` is the
        window's midpoint, which with a fixed grid *is* the window's
        definition, and ``value_median`` is kept beside it so a window whose
        mass sits off-centre can be spotted.

    Raises:
        KeyError: If either column is absent, naming both.
        ValueError: If ``width`` or ``step`` is not positive, or if ``stop`` is
            below ``start`` -- an empty grid is a mistake, not an empty figure.
    """
    quantile_columns = [f"error_p{int(q * 100)}" for q in SHIFT_ERROR_QUANTILES]
    columns = [
        "bin",
        "n",
        "value_lo",
        "value_hi",
        "value_label",
        "value_center",
        "value_median",
        "error_min",
        "error_max",
    ] + quantile_columns
    usable, windows = _shift_error_windows(
        error_df,
        value_col=value_col,
        error_col=error_col,
        start=start,
        width=width,
        step=step,
        stop=stop,
        min_rows=min_rows,
    )
    if not windows:
        return pd.DataFrame(columns=columns)

    values = usable[value_col].to_numpy(float)
    errors = usable[error_col].to_numpy(float)
    rows = []
    for index, lo, hi, inside in windows:
        window_errors = errors[inside]
        row = {
            "bin": index,
            "n": int(inside.sum()),
            "value_lo": lo,
            "value_hi": hi,
            "value_label": window_label(lo, hi),
            "value_center": lo + width / 2.0,
            "value_median": float(np.median(values[inside])),
            # The extremes are kept for the tooltip, not for the whisker:
            # the whisker is `error_p5` to `error_p95` and the rows outside
            # it are drawn one by one. See `SHIFT_ERROR_QUANTILES`.
            "error_min": float(window_errors.min()),
            "error_max": float(window_errors.max()),
        }
        for name, q in zip(quantile_columns, SHIFT_ERROR_QUANTILES):
            row[name] = float(np.quantile(window_errors, q))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _shift_error_windows(
    error_df: pd.DataFrame,
    *,
    value_col: str,
    error_col: str,
    start: float,
    width: float,
    step: Optional[float],
    stop: Optional[float],
    min_rows: int,
) -> tuple[pd.DataFrame, list[tuple[int, float, float, np.ndarray]]]:
    """The window grid shared by the quantiles and the outlier points.

    Returns the usable rows -- finite in both columns and at or above
    ``start`` -- and one ``(index, lo, hi, mask)`` per window holding at least
    ``min_rows`` of them, where ``mask`` indexes those rows positionally.
    Shared rather than duplicated so the drawn box and the drawn points are
    guaranteed to describe the same windows.
    """
    step = width if step is None else step
    if width <= 0:
        raise ValueError(f"width must be positive, got {width!r}.")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step!r}.")
    if stop is not None and stop < start:
        raise ValueError(
            f"stop {stop!r} is below start {start!r}; the window grid would be empty."
        )
    if not len(error_df):
        return error_df, []
    for column in (value_col, error_col):
        if column not in error_df.columns:
            raise KeyError(
                f"{column!r} is not a column of error_df; columns are "
                f"{sorted(error_df.columns)}."
            )

    finite = (
        error_df[[value_col, error_col]]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    usable = error_df[finite & (error_df[value_col] >= start)]
    if not len(usable):
        return usable, []

    values = usable[value_col].to_numpy(float)
    last = float(values.max()) if stop is None else float(stop)
    windows, index, lo = [], 0, float(start)
    while lo <= last:
        hi = lo + width
        # Half-open on the right. The largest effect in the data is never lost
        # by that: it either falls inside a window, or sits exactly on an edge
        # and opens the next one, since the loop runs while `lo <= last`.
        inside = (values >= lo) & (values < hi)
        if int(inside.sum()) >= min_rows:
            windows.append((index, lo, hi, inside))
        index += 1
        lo += step
    return usable, windows


def shift_error_trend(
    error_df: pd.DataFrame,
    *,
    value_col: str = "max_clade_effect",
    error_col: str = "abs_shift_error",
) -> dict[str, float]:
    """Spearman correlation between the windowed axis and the error itself.

    The number the figure is a picture of, computed on the unbinned rows so it
    does not depend on the window grid. Spearman rather than Pearson because
    the error is heavy-tailed and the hypothesis is monotone, not linear.

    Every row given is used, including any below
    :data:`SHIFT_ERROR_WINDOW_START`. Restricting to the drawn range is the
    caller's choice and the caller's to report -- both numbers are worth
    seeing, since they differ by exactly the clip artifact the window floor
    exists to exclude.

    Returns:
        ``{"spearman_rho", "p_value", "n"}``. ``rho`` and ``p_value`` are NaN
        with fewer than three usable rows or when either column is constant,
        which is the same convention :func:`shift_pair_metrics` uses.
    """
    if not len(error_df):
        return {"spearman_rho": float("nan"), "p_value": float("nan"), "n": 0.0}
    for column in (value_col, error_col):
        if column not in error_df.columns:
            raise KeyError(
                f"{column!r} is not a column of error_df; columns are "
                f"{sorted(error_df.columns)}."
            )
    usable = (
        error_df[[value_col, error_col]].replace([np.inf, -np.inf], np.nan).dropna()
    )
    constant = len(usable) and (
        np.ptp(usable[value_col].to_numpy()) == 0
        or np.ptp(usable[error_col].to_numpy()) == 0
    )
    if len(usable) < 3 or constant:
        return {
            "spearman_rho": float("nan"),
            "p_value": float("nan"),
            "n": float(len(usable)),
        }
    result = stats.spearmanr(usable[value_col], usable[error_col])
    return {
        "spearman_rho": float(result[0]),
        "p_value": float(result[1]),
        "n": float(len(usable)),
    }


# Coverage of the count-noise interval below. 0.95 because the question it
# answers -- "is the model's effect inside what this count could support" -- is
# read as a yes/no, and a yes/no needs a conventional level rather than a
# tunable one.
COUNT_NOISE_COVERAGE = 0.95

# mat-pcp's `delta_fitness_ratio` is `log((a + 0.5) / (e + 0.5))`, and the two
# functions below take that pseudocount as a parameter rather than assuming it.
# The two choices differ by `log((e + 0.5) / e)`, which is negligible at large
# expected counts and is not at small ones: an interval on `log(lambda / e)`
# drawn under a point that carries the pseudocount puts the point outside its
# own band wherever the expected count is near 1.
#
# **Panel A passes 0, not this.** See `with_count_noise_interval` in
# `scripts/panels.py` for why: the shifted estimand has a floor, so a zero count
# that can rule nothing out would "reject" a strongly deleterious prediction.
# This constant is the alternative, and nothing here uses it.
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
    answering different questions; see this module's docstring on why fitness
    effects are read and never recomputed.

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
