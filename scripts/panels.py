"""The agreement analysis, read out of the scored table.

``fig:agreement`` asks one question in five panels: does a DASM's log selection
factor agree with mat-pcp's counts-based fitness effect -- within a clade, and
on how an effect *shifts* between clades. Every number behind those panels is a
correlation over a filtered subset of ``selection_factors.parquet``, so this
module is the filter and the statistics, and the notebooks beside it are the
layout.

**No inference and no GPU.** ``rule score`` already did the forward passes. What
is left is to read the table, drop the cells mat-pcp never estimated, exclude
the three site classes the models did not train on normally, clip both axes to a
common bound, and correlate. That replaces upstream's ``score_runs``, which did
all of that *and* the inference, in one function per ``(model, clade)``.

**The three exclusions, and why the third is not in the table.** A comparison is
only meaningful where the model trained normally, so masked, censored and
reverting sites all come out. The table carries ``globally_masked`` and
``censored`` as columns. It does not carry the reverting sites, and
``censored_directional`` is not them: that column is the censor table's
directional entries, which are a subset of ``censored`` by construction, so
substituting it would run a two-layer exclusion while believing it ran three.
The reverting sites come from ``audit/{gene}.json``, which ``rule audit_censor``
derives from the corpus the models trained on -- not re-derived here, because a
second derivation could disagree with what the model saw.

GISAID-derived: the table this reads descends from the cluster-only mat-pcp
mount. The per-panel CSVs are counts, fitness effects and selection factors,
which are publishable; no panel carries a sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import GENE, assert_dasm2_is_this_repo

# Then `dasm2`, before pandas or pyarrow: `dasm2/arrow_pool.py` moves Arrow off
# the bundled mimalloc, and can only do so if pyarrow has not been imported yet.
import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from founders import founder_aa_sequences

# Every name below is the figure's shape rather than its arithmetic, and
# `workflow/figures.smk` needs the same names to declare its outputs. They
# live in a module it can import without pulling in `dasm2`, pandas and
# pyarrow to learn five file names.
from panel_spec import (  # noqa: F401  (re-exported for the notebooks)
    ARM,
    AUDIT_JSON,
    COUNT_LADDER,
    DEFAULT_SELECTION,
    FIDELITY_MODELS,
    FIGURE_DIR,
    FIGURE_MODEL,
    LADDER_PANELS,
    LOOKUP_MUTATION,
    MUTATION_PANELS,
    PANEL_CSV_DIR,
    PANEL_MIN_COUNT,
    SCATTER_CLADE,
    SELECTIONS,
    SHIFT_CLADES,
    TABLE_COLUMNS,
    figure_dir,
    panel_csv_path,
    selection_models,
)

from dasm2.analysis.viral import filter_merged_fitness

# The statistics are re-exported rather than imported again in each notebook:
# the panels then take their filter chain and their correlations from one
# module, and a notebook cannot reach past it to a function this layer does not
# intend them to use.
from dasm2.analysis.viral_fitness_eval import (  # noqa: F401  (re-exported)
    CLIP_LOWER,
    COUNT_FILTER_MODE,
    COUNT_NOISE_COVERAGE,
    COUNT_THRESHOLD_LADDER,
    DEFAULT_MIN_CLADE_MUTATIONS,
    FITNESS_METRICS,
    apply_count_threshold,
    clip_axes,
    count_ladder_adequacy,
    identical_founder_pairs,
    poisson_effect_interval,
    poisson_effect_p_value,
    shift_frame,
    shift_pair_count_ladder,
    shift_pair_metrics,
    within_clade_count_ladder,
    within_clade_metrics,
)


@dataclass(frozen=True)
class FilterSpec:
    """Every filter choice that moves a plotted number, in one object.

    One object rather than six loose arguments because the point is that the
    panels share the choices: a spec is comparable, printable and testable for
    equality, where six arguments are six chances for one notebook to default
    differently from another.

    **The count filter is one number, not two.** A cutoff is a single minimum
    applied to ``actual_count`` and ``predicted_count`` alike, combined with the
    module-fixed ``or`` of
    :data:`~dasm2.analysis.viral_fitness_eval.COUNT_FILTER_MODE` -- the two
    counts fail in opposite directions, so an ``or`` keeps a mutation whose
    estimate rests on real observations *or* on a meaningful denominator.
    Collapsing it to one number is also what makes the ladder coherent: panels C
    and E sweep this field, so a rung and a cutoff are the same quantity.
    """

    fitness_metric: str = "delta_fitness_ratio"
    min_count: float = 0.0
    clip_lower: float = CLIP_LOWER
    min_clade_mutations: int = DEFAULT_MIN_CLADE_MUTATIONS
    exclude_masked: bool = True
    exclude_synonymous: bool = True

    def __post_init__(self) -> None:
        if self.fitness_metric not in FITNESS_METRICS:
            raise ValueError(
                f"fitness_metric must be one of {FITNESS_METRICS}, got "
                f"{self.fitness_metric!r}."
            )

    def as_row(self) -> dict[str, object]:
        """The spec as CSV columns, so a table says what produced it.

        Every name is prefixed ``spec_``. Without it ``min_count`` collides with
        the column panels C and E carry their **rung** in, and the spec's own
        value -- 0, because the ladder owns the count axis -- would silently
        replace the x axis of the two panels that sweep it.
        """
        return {
            "spec_fitness_metric": self.fitness_metric,
            "spec_min_count": self.min_count,
            "spec_clip_lower": self.clip_lower,
            "spec_min_clade_mutations": self.min_clade_mutations,
            "spec_exclude_masked": self.exclude_masked,
            "spec_exclude_synonymous": self.exclude_synonymous,
            "spec_count_filter_mode": COUNT_FILTER_MODE,
        }


#: The published spec, at cutoff **zero**: the ladder owns the count axis, so
#: the base frames carry no count filter and every rung applies its own. The
#: per-mutation panels re-apply :data:`PANEL_MIN_COUNT` to the same frames,
#: which is what makes a dot in panel C and the cloud in panel B the same rows.
BASE_SPEC = FilterSpec(min_count=0.0)


def reverting_sites_by_clade(audit_json: Path = AUDIT_JSON) -> dict[str, set[int]]:
    """``{clade: reverting codon sites}``, read from the censor audit.

    ``scripts/audit_censor.py`` derives these from the corpus the models trained
    on and writes them to ``audit/{gene}.json``. Not re-derived here, for the
    reason the censor table is not either: a second derivation could disagree
    with what the model saw.

    Raises:
        FileNotFoundError: If the audit has not been written. The sites come
            from the corpus, not from a manifest, so there is nothing to fall
            back to -- run ``rule audit_censor``.
        ValueError: If the audit carries no ``reverting_codon_sites_by_clade``.
            A stale audit would silently exclude nothing and report every site
            as scored, which is the one failure mode that looks like success.
    """
    if not audit_json.exists():
        raise FileNotFoundError(
            f"{audit_json} does not exist, so the reverting exclusion cannot be "
            f"applied. Run `rule audit_censor` -- the sites are derived "
            f"from the corpus, not from a manifest."
        )
    payload = json.loads(audit_json.read_text())
    if "reverting_codon_sites_by_clade" not in payload:
        raise ValueError(
            f"{audit_json} carries no reverting_codon_sites_by_clade. Delete it "
            f"and re-run `rule audit_censor`; a stale audit would silently "
            f"exclude nothing and report every site as scored."
        )
    return {
        clade: {int(site) for site in sites}
        for clade, sites in payload["reverting_codon_sites_by_clade"].items()
    }


def reverting_sites(by_clade: dict[str, set[int]], clade: str) -> set[int]:
    """One clade's reverting sites.

    Raises:
        KeyError: If the clade contributed no PCPs to the audit. Omitted rather
            than empty there deliberately: an empty set reads as "this clade
            reverts nowhere" when nothing was measured.
    """
    if clade not in by_clade:
        raise KeyError(
            f"clade {clade!r} contributed no PCPs to {GENE}'s censor audit, so "
            f"whether it reverts anywhere is unmeasured rather than empty. "
            f"Clades present: {sorted(by_clade)}."
        )
    return by_clade[clade]


def excluded_sites(
    frame: pd.DataFrame,
    *,
    clade: str,
    reverting_by_clade: dict[str, set[int]],
    exclude_masked: bool = True,
) -> pd.Series:
    """The site exclusion for one clade's rows, as a boolean mask.

    All three layers at once: the global artifact mask and the per-clade censor
    table, both carried as columns of the scored table, and the per-clade
    reversion rule, read from the audit. mat-pcp's ``aamut_fitness`` at a
    censored site is still built from counts that contain the censored
    observations, so it is not an independent yardstick there -- a model
    deliberately trained not to fit those observations would be marked down for
    succeeding.

    **It takes the clade, not that clade's reverting set.** Of the three layers
    only the reversion rule needs a lookup -- the other two are columns -- and a
    caller handed the set could hoist the lookup out of its clade loop and score
    every clade against the first clade's reversions, with the right number of
    columns and no error. Looking it up here makes that unavailable.

    ``exclude_masked=False`` leaves two layers rather than three, which is the
    one caller-visible way to get the exclusion this module's docstring warns
    about. Nothing in the panels passes it.
    """
    dropped = frame["censored"] | frame["aa_site"].isin(
        reverting_sites(reverting_by_clade, clade)
    )
    if exclude_masked:
        dropped = dropped | frame["globally_masked"]
    return dropped


def read_scored_table(
    table_path: str | Path,
    *,
    models: Sequence[str],
    arm: str = ARM,
    clades: Iterable[str] | None = None,
    columns: Sequence[str] = TABLE_COLUMNS,
) -> pd.DataFrame:
    """The slice of ``selection_factors.parquet`` the panels read.

    Pushed into the Parquet reader rather than filtered afterwards: the table is
    1,425,760 rows per run across seven runs, and a panel reads one model's
    full-depth rows for a handful of clades.

    Args:
        table_path: ``selection_factors.parquet``, passed in by the rule. One
            spelling, in ``workflow/score.smk``; a second one in a script is a
            path that can drift.
        models: Which models to read. A run that has not been scored yet is
            absent from the table rather than an error here.
        arm: The corpus arm, ``full_depth`` for every panel of this figure.
        clades: Which clades, or None for all of them.
        columns: Which columns; see :data:`TABLE_COLUMNS`.

    Returns:
        One frame, with ``parent_aa`` also carried as ``clade_founder_aa``.
        Upstream's shift join keys its wildtype-agreement guard on that name and
        silently skips the check when it is absent, so the rename is what makes
        D614G in one clade never compare against E614G in another.
    """
    filters = [("arm", "=", arm), ("model", "in", list(models))]
    if clades is not None:
        filters.append(("clade", "in", list(clades)))
    frame = pq.read_table(
        str(table_path), columns=list(columns), filters=filters
    ).to_pandas()
    return frame.assign(clade_founder_aa=frame["parent_aa"])


def base_frames(
    table: pd.DataFrame,
    *,
    spec: FilterSpec = BASE_SPEC,
    reverting_by_clade: dict[str, set[int]],
    models: Sequence[str],
    clades: Sequence[str],
) -> dict[tuple[str, str], pd.DataFrame]:
    """One filtered, clipped frame per ``(model, clade)``. The shared core.

    The replacement for upstream's ``score_runs``: the inference and the join
    already happened in ``rule score``, so what is left is the null drop, the
    three site exclusions, the count filter and the clip -- all of it here once,
    so a dot in panel C and the cloud behind it in panel B are literally the
    same rows.

    **The null drop is the inner join.** ``rule score`` writes a left join from
    the model grid, so every cell mat-pcp never estimated carries a null fitness
    column. Those rows are not zero-effect mutations, and a correlation over
    them is undefined; dropping them recovers the row set upstream's inner join
    produced.

    Args:
        table: :func:`read_scored_table`'s output.
        spec: The filter choices. Normally :data:`BASE_SPEC`, at cutoff zero, so
            each rung can apply its own.
        reverting_by_clade: :func:`reverting_sites_by_clade`'s output.
        models: Models to score. One absent from ``table`` is skipped, since a
            campaign in flight is a normal state.
        clades: Clades to score.

    Returns:
        ``{(model, clade): frame}``, keyed by model rather than by run id
        because the arm is fixed and a caller plotting the model axis should not
        have to split a compound key.
    """
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for model in models:
        rows = table[table["model"] == model]
        if rows.empty:
            continue
        for clade in clades:
            frame = rows[rows["clade"] == clade]
            if frame.empty:
                continue
            frame = frame.dropna(subset=[spec.fitness_metric])
            frame = frame[
                ~excluded_sites(
                    frame,
                    clade=clade,
                    reverting_by_clade=reverting_by_clade,
                    exclude_masked=spec.exclude_masked,
                )
            ]
            frame = filter_merged_fitness(
                frame,
                exclude_synonymous=spec.exclude_synonymous,
                min_actual_count=spec.min_count,
                min_predicted_count=spec.min_count,
                count_filter_mode=COUNT_FILTER_MODE,
            )
            frames[(model, clade)] = clip_axes(frame, spec.clip_lower)
    return frames


def by_clade(
    frames: dict[tuple[str, str], pd.DataFrame], model: str
) -> dict[str, pd.DataFrame]:
    """One model's frames, keyed by clade, as the statistics functions want."""
    return {clade: frame for (m, clade), frame in frames.items() if m == model}


#: Panel A's columns, which are not the table's: the two approaches arrive as
#: two series of one quantity rather than as two of the table's columns.
LOOKUP_COLUMNS = (
    "clade",
    "delta_fitness",
    "log_sf",
    "expected_count",
    "actual_count",
    "censored_here",
)


def lookup_frame(
    table: pd.DataFrame,
    *,
    mutation: str,
    model: str,
    min_count: float = PANEL_MIN_COUNT,
    spec: FilterSpec = BASE_SPEC,
) -> pd.DataFrame:
    """One mutation across every clade that carries an estimate of it.

    Panel A's own row set, and deliberately not a slice of
    :func:`base_frames`': the point of the panel is to show *where* the two
    approaches disagree, which is not confined to the clades a correlation
    panel happens to score.

    **The per-clade exclusions are not applied here, and that is the panel.**
    A censored clade is shown and flagged in ``censored_here`` rather than
    dropped: a reader asking "what does this mutation do across the tree" is
    owed the clades where the answer is suspect, labelled. Excluding them
    instead would leave the flag constantly ``False`` -- a column that reads as
    informative and cannot be.

    The global mask is different, and refuses the whole panel. It is a
    sequencing artifact rather than a clade-specific judgment, so a mutation at
    a masked site has no reading anywhere and a per-clade flag would imply it
    had one somewhere.

    **The count cutoff does apply.** The panel is captioned as the clades that
    carry at least ``min_count`` actual or expected counts of this mutation, so
    a clade whose estimate rests on nothing is not drawn at all -- that is a
    statement about evidence rather than about censoring, and it is the same
    cutoff the correlation panels use.

    Values are the table's own, unclipped: this is a per-mutation readout, and
    a clamped value would misreport a clade pinned to the bound. The clipped
    axes belong to the correlation panels.

    Returns:
        One row per clade, with :data:`LOOKUP_COLUMNS`, in the clade order of
        ``table``.

    Raises:
        ValueError: If the mutation is synonymous, or its site is on the global
            mask. Both are states where no clade carries a reading, so an empty
            frame would be indistinguishable from a misspelled label.
    """
    rows = table[(table["model"] == model) & (table["aa_mutation"] == mutation)]
    if len(rows) and rows["is_synonymous"].all():
        raise ValueError(
            f"{mutation} is synonymous: its log selection factor is a "
            f"structural 0 rather than a prediction, so there is nothing to "
            f"compare against the counts."
        )
    if spec.exclude_masked and len(rows) and rows["globally_masked"].all():
        raise ValueError(
            f"{mutation} sits on mat-pcp's global mask for {GENE} -- a "
            f"sequencing artifact rather than a clade-specific censoring -- so "
            f"it is excluded in every clade, not just some."
        )
    rows = rows.dropna(subset=[spec.fitness_metric])
    rows = filter_merged_fitness(
        rows,
        min_actual_count=min_count,
        min_predicted_count=min_count,
        count_filter_mode=COUNT_FILTER_MODE,
    )
    if rows.empty:
        return pd.DataFrame(columns=list(LOOKUP_COLUMNS))
    return pd.DataFrame(
        {
            "clade": rows["clade"],
            "delta_fitness": rows[spec.fitness_metric].astype(float),
            "log_sf": rows["log_selection_factor"].astype(float),
            "expected_count": rows["predicted_count"].astype(float),
            "actual_count": rows["actual_count"].astype(float),
            "censored_here": rows["censored"],
        }
    ).reset_index(drop=True)


def with_count_noise_interval(
    frame: pd.DataFrame, *, spec: FilterSpec = BASE_SPEC
) -> pd.DataFrame:
    """Add the band the counts support, and the model's p-value against it.

    The band is the set of effects the observed count is consistent with under
    Poisson noise in that count alone, holding the expected count fixed. Fixed,
    and not given an error term of its own, because **both estimates are
    anchored to the same neutral model**: an error in the expectation moves the
    counts-based effect and the DASM's together, so for "is the model inside
    what the counts can support" it is common mode. That is what makes this band
    narrower than mat-pcp's published ``uncertainty``, which is the posterior SD
    of ``delta_fitness_bayesian`` and carries the expected-count variance and a
    prior as well -- so the band exists only for ``delta_fitness_ratio``.

    ``count_noise_p`` is the same statement as a number: the two-sided exact
    Poisson p-value for the model's effect being the truth.
    """
    if spec.fitness_metric != "delta_fitness_ratio":
        raise ValueError(
            f"the count-noise band exists only for delta_fitness_ratio, not "
            f"{spec.fitness_metric!r}: a Poisson interval around "
            f"delta_fitness_bayesian would be wrong twice over, since that "
            f"column is shrunk toward a prior and carries its own "
            f"`uncertainty`. Draw panel A without a band instead."
        )
    if not len(frame):
        return frame
    # **No pseudocount**, though the plotted point has one. The band is on
    # `log(lambda / e)`, the quantity both approaches estimate. Matching
    # mat-pcp's 0.5 would keep the point visually inside the band at every
    # expected count, but the shifted estimand has a floor at
    # `log(0.5 / (e + 0.5))` -- so a strongly deleterious DASM prediction at a
    # site with almost no expected traffic would be "rejected" by a zero count
    # that cannot rule anything out. A cosmetic offset at small expected counts
    # is the cheaper error; `RATIO_PSEUDOCOUNT` is there for anyone who prefers
    # the other trade.
    lo, hi = poisson_effect_interval(frame["actual_count"], frame["expected_count"])
    # An open-below bound is `-inf`, which cannot be drawn: Vega-Lite would
    # extend the panel's y scale to negative infinity and take both series with
    # it. It is redrawn at the floor of what the panel already shows, with a
    # margin, so the band runs off the bottom of the panel -- which is what
    # "this count cannot bound the effect from below" should look like.
    drawn = frame[["delta_fitness", "log_sf"]].to_numpy(float).ravel()
    floor = float(
        np.nanmin(np.concatenate([drawn, lo[np.isfinite(lo)], hi[np.isfinite(hi)]]))
    )
    span = float(np.nanmax(drawn) - floor)
    lo = np.where(np.isfinite(lo), lo, floor - 0.08 * (span or 1.0))
    return frame.assign(
        count_noise_lo=lo,
        count_noise_hi=hi,
        count_noise_p=poisson_effect_p_value(
            frame["actual_count"], frame["expected_count"], frame["log_sf"]
        ),
    )


def write_panel_csv(
    frame: pd.DataFrame, stem: str, *, selection: str, spec: FilterSpec
) -> Path:
    """Write one panel's table to tracked ``results/panels/<selection>``.

    The filter spec rides along as columns rather than sitting in a sidecar:
    every number in the file moves with the cutoff and the clip, and a reader
    who opens the CSV alone should not have to find out which they were.
    """
    columns = spec.as_row()
    collisions = sorted(set(columns) & set(frame.columns))
    if collisions:
        raise ValueError(
            f"{stem}'s frame already carries {collisions}, which the filter "
            f"spec would overwrite. A panel's own column wins: rename the "
            f"spec's field rather than letting `assign` replace the data."
        )
    path = panel_csv_path(selection, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.assign(**columns).to_csv(path, index=False)
    return path


def shift_pair_founders(clades: Sequence[str] = SHIFT_CLADES) -> dict[str, str]:
    """``{clade: founder}`` for the shift clades.

    Needed because a pair whose founder is identical in this gene is the same
    input to the model twice, so no model can show a shift between them while
    mat-pcp's Δfitness still moves with each clade's own tree counts. Such a
    pair is not evidence about any model and is dropped rather than scored at
    r = 0.
    """
    return founder_aa_sequences(clades, GENE)
