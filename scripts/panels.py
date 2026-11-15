"""The agreement analysis, read out of the scored table.

``fig:agreement`` asks one question in five panels: does a DASM's log selection
factor agree with mat-pcp's counts-based fitness effect -- within a clade, and
on how an effect *shifts* between clades. Every number behind those panels is a
correlation over a filtered subset of ``selection_factors.parquet``, so this
module is the filter and the statistics, and the notebooks beside it are the
layout.

**No inference and no GPU.** ``rule score`` already did the forward passes. What
is left is to read the table, drop the cells mat-pcp never estimated, clip both
axes to a common bound, and correlate. That replaces upstream's ``score_runs``,
which did all of that *and* the inference, in one function per
``(model, clade)``.

**The exclusion is the null drop, and it is per mutation.** mat-pcp excludes a
mutation one nucleotide route at a time -- the global masks, the per-clade site
and mutation masks, and founder->reference reversions all set its ``exclude``
flag, and ``OUTPUT_SCHEMA.md`` records that excluded routes are dropped before
any fitness effect is computed. A mutation it excluded entirely therefore has no
row in ``aamut_fitness``, arrives in the scored table's left join with null
counts, and leaves at :func:`base_frames`' ``dropna``.

Nothing here widens that to the codon. An earlier version did, dropping every
mutation at a site carrying any flag, which cost 1,067 of the 33,782 mutations
the figure can compare over its 22 clades -- among them 21K's R346K at 753
observed against 22.2 expected, whose counts are computed on the clade founder's
background and are untouched by the censoring that flagged its site. See #36.

**What makes that fair is that training censors the same routes.** mat-pcp's
``exclude`` and this pipeline's censoring are two applications of one set:
``censor_table.py`` partitions the five reasons of ``masking.compute_exclusions``
between ``pcps/{gene}/censored_mutations.csv`` and
``pcps/{gene}/masked_nt_sites.csv``, with founder->reference reversions in
neither because a training run derives them, and ``masked_nt_sites.py``
reconciles its file against ``counts_all.csv``'s flag in both directions. So a
mutation the model was trained on is one mat-pcp estimated, and the comparison
holds no route one side saw and the other did not.
:func:`tests.test_panels.test_training_censors_exactly_what_mat_pcp_excludes`
re-derives that set equality from the two files and the counts table.

**One class of row survives the null drop and is still not a measurement.**
Where a clade's founder codon contains a nucleotide masked in the tree, mat-pcp
can attribute no observation to that founder background and the codon arrives
with an actual count of zero against a full expectation -- a lethal-looking
estimate with nothing behind it. :func:`unobservable_founder_codons` rebuilds
that predicate and :func:`drop_unobservable_codons` applies it. It is 14
mutations of 33,782, and they were the figure's largest outliers. See #39 and
matsengrp/mat-pcp#161.

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
from lib import (  # noqa: F401  (`corpus` re-exported for the notebooks)
    CORPUS_NAMES,
    FULL_DEPTH,
    GENE,
    assert_dasm2_is_this_repo,
    clade_roster,
    corpus,
    mat_pcp_path,
)

# Then `dasm2`, before pandas or pyarrow: `dasm2/arrow_pool.py` moves Arrow off
# the bundled mimalloc, and can only do so if pyarrow has not been imported yet.
import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from functools import cache

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
    BACKGROUND_PANELS,
    CANDIDATE_CLADES,
    CANDIDATE_CSV,
    CANDIDATE_LABEL_SIDES,
    COUNT_LADDER,
    DEFAULT_SELECTION,
    DEPTH_PANELS,
    ECDF_CLADES,
    ECDF_MAX_COUNT,
    ECDF_POINTS,
    EMERGENCE_CSV,
    FIDELITY_MODELS,
    FIGURE_DIR,
    FIGURE_MODEL,
    LADDER_ARM_PANELS,
    LADDER_PANELS,
    LOOKUP_MUTATION,
    MUTATION_PANELS,
    OUTLIER_MIN_COUNT,
    OUTLIER_MIN_DIFFERENCE,
    OUTLIER_PANELS,
    OUTLIER_TABLE_STEM,
    PANEL_CSV_DIR,
    PANEL_MIN_COUNT,
    PCP_CHUNK_ROWS,
    SCATTER_CLADE,
    SELECTIONS,
    SHIFT_CLADES,
    SUBSAMPLING_CLIP_LOWER,
    SUBSAMPLING_LADDER_CLADES,
    SUBSAMPLING_MODEL,
    SUBSAMPLING_SCATTER_CLADE,
    SUBSAMPLING_SELECTION,
    TABLE_COLUMNS,
    aamut_grid_path,
    figure_candidates,
    figure_dir,
    labelled_candidates,
    panel_csv_path,
    parse_mutation,
    selection_models,
    subsampling_caps,
)

from dasm2.analysis.viral import filter_merged_fitness, normalize_fitness_columns

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
from dasm2.analysis.within_clade_pools import (
    DEFAULT_PERMUTATIONS,
    ZERO_COUNT_PSEUDOCOUNT,
)
from dasm2.io import read_fasta_first_sequence
from dasm2.neutral.scov2_rates import GENE_NT_LENGTH
from dasm2.sequence.conversion import translate_sequence


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

    **There is no mask field.** An earlier spec carried ``exclude_masked``, and
    after #36 it filtered nothing: the exclusion comes from mat-pcp through the
    null drop in :func:`base_frames`, and :func:`lookup_frame` now refuses a
    mutation on the evidence -- no clade carries an estimate -- rather than on
    the ``globally_masked`` flag, which is the nucleotide mask widened to the
    codon and so refuses mutations mat-pcp did estimate. Keeping the field would
    have written ``spec_exclude_masked,True`` into panels that carry masked
    sites.
    """

    fitness_metric: str = "delta_fitness_ratio"
    min_count: float = 0.0
    clip_lower: float = CLIP_LOWER
    min_clade_mutations: int = DEFAULT_MIN_CLADE_MUTATIONS
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
            "spec_exclude_synonymous": self.exclude_synonymous,
            "spec_count_filter_mode": COUNT_FILTER_MODE,
        }


#: The published spec, at cutoff **zero**: the ladder owns the count axis, so
#: the base frames carry no count filter and every rung applies its own. The
#: per-mutation panels re-apply :data:`PANEL_MIN_COUNT` to the same frames,
#: which is what makes a dot in panel C and the cloud in panel B the same rows.
BASE_SPEC = FilterSpec(min_count=0.0)


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


#: The mask reason that makes a nucleotide unobservable rather than merely
#: dropped. ``masked_nt_sites.csv`` carries two, and the difference decides
#: whether a founder codon can be counted at all: a ``vcf_masked_sites``
#: position is masked in the tree, so every sequence reads the reference base
#: there, while a ``sites_to_exclude`` position is mat-pcp's own config
#: exclusion and leaves the base observable.
TREE_MASK_REASON = "vcf_masked_sites"


def _founder_nt_sequence(clade: str) -> str:
    """One clade's founder nucleotides, from the FASTA ``founders.py`` translates.

    Read here rather than through that module, which exists so ``rule score``
    and the panel rules do not invalidate each other: a reader added there for
    the panels alone would mark seven GPU scoring jobs out of date every time
    it was edited. The length check is the same one, against the same constant.
    """
    path = mat_pcp_path("clade_founders_dir") / f"{clade}.fa"
    sequence = read_fasta_first_sequence(path)
    if len(sequence) != GENE_NT_LENGTH[GENE]:
        raise ValueError(
            f"{GENE}/{clade}: founder is {len(sequence)} nucleotides but "
            f"GENE_NT_LENGTH says {GENE_NT_LENGTH[GENE]}"
        )
    return sequence


@cache
def unobservable_founder_codons(clade: str) -> frozenset[int]:
    """The codon sites where this clade's founder background cannot be counted.

    mat-pcp counts an observation only on the clade founder's codon background,
    comparing the codon matUtils read against the founder grid's
    (``build_count_table.py:331``). Where one of that codon's nucleotides is
    masked in the tree, every sequence reads the reference base there, the two
    codons never match, and the codon's observations are all discarded as a
    mutated background. **The expected counts are not discarded with them**:
    they come from the fitted rates per branch, which know nothing about that
    filter. So the codon arrives with ``actual_count`` zero against a full
    expectation and reads as lethal.

    21J's codon 142 is the case. Delta carries G142D, so the founder codon is
    ``GAT`` where the reference is ``GGT``, differing at the masked nt 21987;
    mat-pcp holds 226 observed counts there, every one recorded as ``GGT>...``,
    and keeps none of them. The synonymous route is what proves this is not
    selection: ``D142D`` has 64.3 expected and zero observed, while the
    neighbouring nucleotides in the same clade carry hundreds of counts.

    **Nothing in the scored table marks these rows.** ``exclude`` is ``False``
    for the routes, correctly -- the routes are not masked, the codon's *other*
    nucleotide is -- and ``globally_masked`` is ``True`` for 337 mutations at
    rung 10 that are good comparisons, which is what #36 established. So the
    predicate is rebuilt here from the mask file, the founder and the
    reference. See #39, and matsengrp/mat-pcp#161 for the fix upstream that
    will make this function return nothing.

    Returns:
        1-indexed codon sites. Empty for most clades.
    """
    masked = pd.read_csv(mat_pcp_path("masked_nt_sites_file"))
    tree_masked = {
        int(index)
        for index, reason in zip(masked["nt_index"], masked["reason"])
        if reason == TREE_MASK_REASON
    }
    reference = read_fasta_first_sequence(mat_pcp_path("reference_sequence_file"))
    founder = _founder_nt_sequence(clade)
    return frozenset(
        index // 3 + 1 for index in tree_masked if founder[index] != reference[index]
    )


def drop_unobservable_codons(frame: pd.DataFrame, clade: str) -> pd.DataFrame:
    """Rows at the codons :func:`unobservable_founder_codons` names, removed.

    A codon-level drop, and deliberately so where #36's was wrong to be: the
    unit mat-pcp cannot observe here *is* the codon, so no mutation at it
    carries an interpretable count for this clade. #36 removed a codon-level
    drop of a *route*-level exclusion, which discarded mutations mat-pcp had
    estimated normally. The two rules disagree on 1,411 mutations at rung 10,
    and agree on 14.
    """
    codons = unobservable_founder_codons(clade)
    if not codons:
        return frame
    return frame[~frame["aa_site"].isin(codons)]


def founder_background_fraction(
    wanted: dict[str, list[tuple[int, str]]], pcp_path: str | Path
) -> dict[tuple[str, int, str], float]:
    """The share of a clade's branch mass still carrying the founder residue.

    **What the counts approach assumes, and cannot check.** A mutation's
    expected count is the clade's fitted rate for that nucleotide change
    scaled to the whole clade (mat-pcp ``rates.py``), while its actual count
    is taken only on branches whose parent carries the founder codon
    (``build_count_table.py``). For a mutation that has risen to high
    frequency the two cover different branch sets: by the time most of the
    clade already carries it, the opportunity left to acquire it again is a
    fraction of what the expectation assumes. Dividing the expectation by
    that fraction is what puts the two on the same footing.

    Reads only the codon at each site asked for, not the whole translation:
    the question is per site, and translating 1.5M spike sequences to answer
    it for a handful would cost minutes for nothing.

    Args:
        wanted: ``{clade: [(1-indexed aa site, founder residue), ...]}``.
        pcp_path: The corpus the clade's branches come from.

    Returns:
        ``{(clade, site, residue): fraction}``, weighted by branch length.
    """
    founder_mass = {
        (clade, site, residue): 0.0
        for clade, sites in wanted.items()
        for site, residue in sites
    }
    total_mass = dict.fromkeys(wanted, 0.0)
    for chunk in pd.read_csv(
        pcp_path, usecols=["clade", "parent", "branch_length"], chunksize=PCP_CHUNK_ROWS
    ):
        chunk = chunk[chunk["clade"].isin(wanted)]
        if chunk.empty:
            continue
        for clade, part in chunk.groupby("clade"):
            lengths = part["branch_length"].to_numpy(float)
            total_mass[clade] += lengths.sum()
            parents = part["parent"].tolist()
            for site, residue in wanted[clade]:
                codons = (parent[3 * (site - 1) : 3 * site] for parent in parents)
                carried = np.fromiter(
                    (translate_sequence(codon) == residue for codon in codons),
                    dtype=bool,
                    count=len(parents),
                )
                founder_mass[(clade, site, residue)] += lengths[carried].sum()
    return {
        key: mass / total_mass[key[0]] if total_mass[key[0]] else float("nan")
        for key, mass in founder_mass.items()
    }


def base_frames(
    table: pd.DataFrame,
    *,
    spec: FilterSpec = BASE_SPEC,
    models: Sequence[str],
    clades: Sequence[str],
) -> dict[tuple[str, str], pd.DataFrame]:
    """One filtered, clipped frame per ``(model, clade)``. The shared core.

    The replacement for upstream's ``score_runs``: the inference and the join
    already happened in ``rule score``, so what is left is the null drop, the
    count filter and the clip -- all of it here once, so a dot in panel C and
    the cloud behind it in panel B are literally the same rows.

    **The null drop is the inner join, and it is all but one of the
    exclusion.**
    ``rule score`` writes a left join from the model grid, so every cell mat-pcp
    never estimated carries a null fitness column -- whether it was never
    reachable, never observed with a usable denominator, or excluded by mat-pcp
    as masked or reverting. Those rows are not zero-effect mutations, and a
    correlation over them is undefined; dropping them recovers the row set
    upstream's inner join produced, at mat-pcp's own per-mutation granularity.

    **The one exclusion it does not cover** is
    :func:`drop_unobservable_codons`, where mat-pcp emits a row whose actual
    count is structurally zero because the clade's founder codon is one the
    tree cannot represent. Those rows are not nulls and cannot be recognised
    from the table, so they are the one thing dropped here on a predicate
    rebuilt from mat-pcp's inputs.

    Args:
        table: :func:`read_scored_table`'s output.
        spec: The filter choices. Normally :data:`BASE_SPEC`, at cutoff zero, so
            each rung can apply its own.
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
            frame = drop_unobservable_codons(frame, clade)
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


def candidate_rows(
    frame: pd.DataFrame, clade: str, candidates: Iterable[tuple[str, str, str]]
) -> pd.DataFrame:
    """One clade's labelled candidates, as rows of its own comparison frame.

    Drawn from ``frame`` rather than joined from the candidate file, so a
    label sits on the mark it names and carries the values the panel plots.
    The candidate's status rides along as ``candidate_group``: panel A draws
    its two groups in different ink, and the group is a property of the
    candidate list rather than of anything the comparison frame holds.

    **A missing candidate raises.** These are the mutations
    `fig:within_clade_shifts` is about, named in `data/shift_candidates.csv`
    and discussed by name in `main.tex`; one that falls out of the comparison
    -- below the cutoff after a recount, or dropped by a filter -- must stop
    the panel rather than quietly draw six points where the caption says
    seven. That is the failure this function exists to make loud, and it is
    not hypothetical: #37, #39 and #40 each moved which mutations the
    comparison holds.

    Raises:
        ValueError: If a candidate for ``clade`` is absent from ``frame``.
    """
    wanted = {
        mutation: status
        for candidate, mutation, status in candidates
        if candidate == clade
    }
    rows = frame[frame["aa_mutation"].isin(wanted)]
    missing = sorted(set(wanted) - set(rows["aa_mutation"]))
    if missing:
        raise ValueError(
            f"{clade} has no comparison row for {', '.join(missing)}, which "
            f"{CANDIDATE_CSV.name} names as a figure candidate. The panel "
            f"would draw {len(rows)} of {len(wanted)} labelled mutations."
        )
    return rows.assign(candidate_group=rows["aa_mutation"].map(wanted))


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

    **A censored clade is shown and flagged rather than dropped.** ``censored_
    here`` carries the flag: a reader asking "what does this mutation do across
    the tree" is owed the clades where the answer is suspect, labelled.
    Excluding them instead would leave the flag constantly ``False`` -- a column
    that reads as informative and cannot be.

    **The mask flags refuse nothing.** ``globally_masked`` is mat-pcp's
    nucleotide mask widened to the codon -- 18 masked nucleotides become 51 at
    codon granularity for ``S`` -- so a flagged codon routinely carries
    mutations mat-pcp estimated normally, reached through its unmasked
    nucleotides. Refusing on the flag would refuse rows the correlation panels
    plot. What refuses is the evidence: a mutation no clade carries an estimate
    of would draw an empty panel.

    **A clade whose founder codon the tree cannot represent is dropped**, as it
    is from the correlation panels, and for the same reason: its counts are
    structurally zero rather than small. That is per clade, so a mutation can
    be shown in one clade and dropped in another at the same codon.

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
        ValueError: If the mutation is synonymous, or mat-pcp estimated it in
            no clade. Both are states where no clade carries a reading, so an
            empty frame would be indistinguishable from a misspelled label --
            which is what a mutation absent from the table returns.
    """
    rows = table[(table["model"] == model) & (table["aa_mutation"] == mutation)]
    if len(rows) and rows["is_synonymous"].all():
        raise ValueError(
            f"{mutation} is synonymous: its log selection factor is a "
            f"structural 0 rather than a prediction, so there is nothing to "
            f"compare against the counts."
        )
    estimated = rows.dropna(subset=[spec.fitness_metric])
    # Per clade, because the predicate is per clade: the same codon is
    # countable in a clade whose founder matches the reference at the masked
    # nucleotide and not in one whose founder does not.
    estimated = estimated[
        [
            site not in unobservable_founder_codons(clade)
            for clade, site in zip(estimated["clade"], estimated["aa_site"])
        ]
    ]
    if len(rows) and estimated.empty:
        raise ValueError(
            f"mat-pcp estimated {mutation} in no clade: every nucleotide route "
            f"to it is excluded in every clade, so there is nothing to plot "
            f"the DASM against."
        )
    rows = estimated
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
    frame: pd.DataFrame, stem: str, *, selection: str, spec: FilterSpec | None
) -> Path:
    """Write one panel's table to tracked ``results/panels/<selection>``.

    The filter spec rides along as columns rather than sitting in a sidecar:
    every number in the file moves with the cutoff and the clip, and a reader
    who opens the CSV alone should not have to find out which they were.

    ``spec=None`` writes the frame unchanged, for a panel that has no filter to
    declare. ``fig:subsampling``'s panels A and B are that case: they measure
    the tree rather than a model, so there is no cutoff, no clip and no site
    exclusion, and spec columns there would state six choices that were never
    made. It is required rather than defaulted, so a panel that *does* filter
    cannot omit it by forgetting.
    """
    if spec is None:
        path = panel_csv_path(selection, stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return path
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


def write_pool_csv(frame: pd.DataFrame, stem: str, *, selection: str) -> Path:
    """Write one background panel's table, with its own provenance columns.

    **Not** :func:`write_panel_csv`. That stamps a :class:`FilterSpec`, which
    describes the agreement figure's filter chain -- a fitness metric, a count
    cutoff, a clip at -4, a clade-mutation floor. None of them is true of a
    pooled count: the estimate is ``log(actual / expected)`` over one
    background's branches, unclipped, with no count cutoff and no clade floor.
    Stamping that spec here would tell a reader the numbers were clipped when
    they were not, which is exactly why `FilterSpec` dropped its
    ``exclude_masked`` field rather than write ``True`` into panels carrying
    masked sites.

    What these tables do have is a provenance worth recording, and it is a
    different list: the pseudocount convention, the randomization test's draw
    count and sidedness, and where the two backgrounds came from.
    """
    columns = {
        # Withheld except where a mutation is never observed on a background.
        # In a sub-pool the expected counts run 0.01 to 1, so a two-sided
        # pseudocount collapses the estimate onto a function of the event
        # count with the expected count edited out.
        "pool_zero_count_pseudocount": ZERO_COUNT_PSEUDOCOUNT,
        "pool_permutation_draws": DEFAULT_PERMUTATIONS,
        # Two-sided: the test asks whether a counts shift this large arises by
        # chance, and nothing fixes a direction in advance.
        "pool_permutation_sided": "two",
        # No model has a say in which backgrounds are contrasted, which is
        # what makes the comparison a test rather than a restatement.
        "pool_background_source": "branch frequency",
    }
    collisions = sorted(set(columns) & set(frame.columns))
    if collisions:
        raise ValueError(
            f"{stem}'s frame already carries {collisions}, which the pool "
            f"provenance would overwrite."
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


# ---------------------------------------------------------------------------
# `fig:subsampling` panels A and B: the tree's sampling depth
# ---------------------------------------------------------------------------
#
# Neither reads the scored table. These two panels are a measurement of the
# corpus rather than of a model: panel A is how unevenly the clades are
# sampled, and panel B is what that costs a clade's mutation counts. They take
# no checkpoint, no GPU and no `selection_factors.parquet`, which is what lets
# them be built while the models are still training.

#: Decimals a share is written with. The denominator is a clade's spike grid,
#: about 11,371 mutations, so one mutation moves a share by 8.8e-5 and six
#: decimals is already finer than the smallest real step. The float's own 16
#: digits record how a pandas version chose to print it, not what the tree
#: says, and they are what makes a tracked table a diff on noise.
SHARE_DECIMALS = 6

#: The amino-acid-grid columns panel B reads. The grid is genome-wide, so
#: `gene` restricts it to spike; the founder and mutant residues are read to
#: drop the synonymous rows, and the two counts are the statistic.
AA_GRID_COLUMNS = (
    "actual_count",
    "expected_count",
    "gene",
    "clade_founder_aa",
    "mutant_aa",
)


def clade_size_frame(counts_csv: Path, emergence_csv: Path) -> pd.DataFrame:
    """Sequences per clade and when each emerged, for the clades scored.

    ``clade,n_sequences,year``, descending by count. ``year`` is the
    clade's emergence as a decimal year, which panel A colours its dots by;
    `rule clade_emergence` measures it and
    `scripts/clade_emergence.py` says what it means. The count is mat-pcp's own
    ``mat/clade_counts.csv``, whose ``count`` is the line count of
    ``mat/clades/{clade}_samples.txt`` -- the file the source panel walks --
    so this is the cheaper route to the same heights and not a second
    derivation of them.

    **The table carries more clades than this pipeline scores**, and the extras
    are dropped against :func:`lib.clade_roster` rather than passed through.
    ``21A`` and ``25I`` have a count here, no founder FASTA, and no row in any
    fitness table, so a panel drawn from the file as it stands would put two
    bars beside 56 that every other table in the pipeline omits.

    Raises:
        ValueError: If a clade in the roster has no row in the counts table.
            That direction is a real inconsistency rather than a known extra:
            the roster is what gets scored, so a scored clade with no sequence
            count means the two files describe different trees.
    """
    path = counts_csv
    frame = pd.read_csv(path)
    counts = dict(zip(frame["Nextstrain_clade_usher"], frame["count"], strict=True))

    roster = clade_roster()
    missing = sorted(clade for clade in roster if clade not in counts)
    if missing:
        raise ValueError(
            f"{path}: no sequence count for {missing}, which this pipeline "
            f"scores. The counts table and the founder roster describe "
            f"different trees."
        )

    out = pd.DataFrame(
        {"clade": roster, "n_sequences": [int(counts[c]) for c in roster]}
    ).merge(_emergence_years(emergence_csv, roster), on="clade")
    return out.sort_values("n_sequences", ascending=False, ignore_index=True)


#: The origin the emergence date is measured from. Panel A colours on a
#: *quantitative* scale rather than a temporal one -- see
#: `panel_charts.clade_size_panel` on why -- so the date has to arrive as a
#: number, and this is the conversion.
EMERGENCE_EPOCH = date(2020, 1, 1)
DAYS_PER_YEAR = 365.25


def _emergence_years(emergence_csv: Path, roster: Sequence[str]) -> pd.DataFrame:
    """``clade,year`` for ``roster``, as decimal years.

    Raises:
        ValueError: If a clade in the roster has no emergence row. That table
            is written from the same founder roster this reads, so a gap means
            the two were measured against different trees.
    """
    frame = pd.read_csv(emergence_csv)
    missing = sorted(set(roster) - set(frame["clade"]))
    if missing:
        raise ValueError(
            f"{emergence_csv}: no emergence date for {missing}, which this "
            f"pipeline scores. Rerun `snakemake clade_emergence`."
        )
    epoch = EMERGENCE_EPOCH.toordinal()
    years = [
        None
        if pd.isna(value)
        else (date.fromisoformat(str(value)).toordinal() - epoch) / DAYS_PER_YEAR
        + EMERGENCE_EPOCH.year
        for value in frame["emergence"]
    ]
    return pd.DataFrame({"clade": frame["clade"], "year": years})


def _aa_grid(
    aamut_dir: Path, clade: str, arm: str, *, gene: str = GENE
) -> pd.DataFrame:
    """One clade's amino-acid grid at one sampling depth, restricted to a gene.

    The layout is :func:`panel_spec.aamut_grid_path`'s, which
    ``workflow/figures.smk`` declares this rule's inputs from, so the file a
    rule promises and the file this opens are one spelling.

    **The synonymous rows come out, and the stops stay in.** The paper counts
    "amino-acid mutations to the founder spike sequence", which is every row
    whose mutant residue differs from the founder's: 7,901 for 21J, against
    9,140 rows in the file. A row where the two agree is a synonymous codon
    change counted at the amino-acid level and is not a mutation this figure is
    about; a stop is.

    ``gene`` is a semicolon-joined list where a site falls in more than one
    reading frame, and is null on the intergenic rows, so the membership test
    runs on ``fillna("")`` -- a bare ``split`` raises on those.
    """
    frame = pd.read_csv(
        aamut_grid_path(aamut_dir, clade, arm), usecols=list(AA_GRID_COLUMNS)
    )
    in_gene = frame["gene"].fillna("").str.split(";").apply(lambda g: gene in g)
    return frame.loc[in_gene & (frame["mutant_aa"] != frame["clade_founder_aa"])]


def count_ecdf_frame(
    aamut_dir: Path,
    clades: Iterable[str] = tuple(ECDF_CLADES),
    *,
    gene: str = GENE,
    arms: Sequence[str] | None = None,
) -> pd.DataFrame:
    """The share of a clade's mutations carrying at least *x* counts.

    ``clade,arm,x,share``, one curve per ``(clade, arm)``. The statistic is
    ``max(actual_count, expected_count)``, the paper's "actual or expected
    counts", evaluated on :data:`ECDF_POINTS` points from 0 to
    :data:`ECDF_MAX_COUNT`.

    **The population is the whole amino-acid grid**, every non-synonymous
    amino-acid mutation away from the clade founder, and not the subset that
    clears the accuracy threshold. The subset is *defined* by a threshold on
    these counts,
    so its own curve would pass through 1.0 at exactly that threshold and be
    truncated by construction -- it could not show what share of a clade's
    mutations the threshold leaves out, which is the panel's question.

    Raises:
        ValueError: If ``arms`` does not carry full depth, which is the
            reference every other arm's grid is checked against.
        ValueError: If a thinned arm's grid has a different number of rows from
            the full-depth one. The grid is enumerated from the clade founder
            and cannot depend on sampling depth, so a changed count means the
            founder moved -- and the two curves would then be drawn over
            different mutations with nothing saying so.
    """
    arms = tuple(CORPUS_NAMES) if arms is None else tuple(arms)
    if FULL_DEPTH.name not in arms:
        raise ValueError(
            f"arms={arms} carries no {FULL_DEPTH.name!r}, which every other "
            f"arm's grid is sized against and which panel B draws as its "
            f"reference curve."
        )
    grid = np.linspace(0, ECDF_MAX_COUNT, ECDF_POINTS)
    rows = []
    for clade in clades:
        # Every arm read before the comparison, rather than full depth being
        # whichever arm the loop happens to reach first: a caller passing
        # `arms` in another order would otherwise skip the size check on every
        # arm ahead of full depth, silently.
        grids = {arm: _aa_grid(aamut_dir, clade, arm, gene=gene) for arm in arms}
        n_full = len(grids[FULL_DEPTH.name])
        for arm, frame in grids.items():
            if len(frame) != n_full:
                raise ValueError(
                    f"{arm}/{clade}: the {gene} amino-acid grid has "
                    f"{len(frame)} rows against {n_full} at full depth. The "
                    f"grid is enumerated from the clade founder and cannot "
                    f"depend on sampling depth, so the founder moved."
                )
            counts = frame[["actual_count", "expected_count"]].max(axis=1).to_numpy()
            rows.append(
                pd.DataFrame(
                    {
                        "clade": clade,
                        "arm": arm,
                        "x": grid,
                        "share": [
                            round(float((counts >= x).mean()), SHARE_DECIMALS)
                            for x in grid
                        ],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# `fig:subsampling` panels C and D: what thinning costs each approach
# ---------------------------------------------------------------------------

#: The two approaches panels C and D compare, in the order they are drawn.
#: Spelled here rather than at each call site because the string is a column
#: value in the panel CSVs and a facet title in the charts.
LADDER_METHODS = ("counts", "DASM")

#: Panel C's and D's filter choices. The clip is this figure's, not the
#: library's; everything else is the published spec.
SUBSAMPLING_SPEC = FilterSpec(min_count=0.0, clip_lower=SUBSAMPLING_CLIP_LOWER)


def arm_counts_estimate(
    aamut_dir: Path, clade: str, arm: str, *, gene: str = GENE
) -> pd.DataFrame:
    """One arm's own counts-based estimate, as the predictor it is here.

    This is the only per-arm read the figure adds. ``rule score`` joined the
    **full-depth** counts estimate onto every arm's rows -- that is the target
    a thinned model is measured against -- so the thinned refits under
    ``aamut_fitness/cap_*_0/`` are a different comparison and nothing upstream
    reads them.

    Returned under the name ``log_selection_factor`` because that is the column
    :func:`within_clade_metrics` takes its predictor from: the counts approach
    and the DASM are scored by one function, so neither can be scored by a
    statistic the other is not.
    """
    frame = normalize_fitness_columns(
        pd.read_csv(aamut_grid_path(aamut_dir, clade, arm))
    )
    in_gene = frame["gene"].fillna("").str.split(";").apply(lambda g: gene in g)
    return (
        frame.loc[in_gene, ["aa_mutation", "delta_fitness_ratio"]]
        .rename(columns={"delta_fitness_ratio": "log_selection_factor"})
        .dropna()
    )


def ladder_frames(
    table_path: str | Path,
    aamut_dir: Path,
    *,
    clades: Sequence[str] = SUBSAMPLING_LADDER_CLADES,
    model: str = SUBSAMPLING_MODEL,
    spec: FilterSpec = SUBSAMPLING_SPEC,
    arms: Sequence[str] | None = None,
) -> dict[tuple[str, str, str], pd.DataFrame]:
    """One frame per ``(arm, clade, method)``, all on one fixed population.

    **The mutation set is fixed on the full-depth arm**, and that is the point
    of the design. Filtering each arm on its own counts would score a thinned
    arm only over the mutations it still happened to observe -- the subset it
    is best at -- and every curve in panels C and D would be drawn over a
    different population. So the count filter runs once, on full depth, and
    every arm is scored over exactly those mutations.

    **The target is the same column in every frame**: the full-depth counts
    estimate, which ``rule score`` joined onto every arm's rows. Only the
    predictor changes -- the arm's DASM for one method, the arm's own counts
    refit for the other.

    Returns:
        ``{(arm, clade, method): frame}``, each frame carrying
        ``aa_mutation``, ``delta_fitness_ratio`` (the target) and
        ``log_selection_factor`` (the predictor), clipped by ``spec``.

    Raises:
        ValueError: If ``model`` has no rows at some arm. A half-scored
            campaign would otherwise draw a panel with a column missing and
            nothing saying which.
    """
    arms = tuple(CORPUS_NAMES) if arms is None else tuple(arms)
    target = spec.fitness_metric

    population: dict[str, pd.DataFrame] = {}
    full = read_scored_table(
        table_path, models=[model], arm=FULL_DEPTH.name, clades=clades
    )
    for clade, frame in by_clade(
        base_frames(full, spec=spec, models=[model], clades=clades),
        model,
    ).items():
        population[clade] = apply_count_threshold(frame, PANEL_MIN_COUNT)[
            ["aa_mutation", f"{target}_raw"]
        ].rename(columns={f"{target}_raw": target})

    frames: dict[tuple[str, str, str], pd.DataFrame] = {}
    for arm in arms:
        table = read_scored_table(table_path, models=[model], arm=arm, clades=clades)
        if table.empty:
            raise ValueError(
                f"{model!r} has no rows at arm {arm!r} in {table_path}. Panels "
                f"C and D compare the arms, so a missing one is a gap in the "
                f"comparison rather than a column to leave out."
            )
        dasm = by_clade(
            base_frames(table, spec=spec, models=[model], clades=clades),
            model,
        )
        for clade, kept in population.items():
            arm_dasm = dasm[clade].merge(kept[["aa_mutation"]], on="aa_mutation")
            frames[(arm, clade, "DASM")] = arm_dasm[
                ["aa_mutation", target, "log_selection_factor"]
            ]
            counts = kept.merge(
                arm_counts_estimate(aamut_dir, clade, arm), on="aa_mutation"
            )
            frames[(arm, clade, "counts")] = clip_axes(counts, spec.clip_lower)[
                ["aa_mutation", target, "log_selection_factor"]
            ]
    return frames


def ladder_metrics(
    frames: dict[tuple[str, str, str], pd.DataFrame],
    *,
    spec: FilterSpec = SUBSAMPLING_SPEC,
) -> pd.DataFrame:
    """Panel D's table: ``arm,clade,method,pearson_r,regression_slope,n_mutations``.

    ``spearman_rho`` is dropped on write, as the source experiment's
    ``sweep_count_cutoffs.py`` does: the figure reports r and the slope, and a
    third statistic in the CSV is one a reader would look for in the panel.
    """
    rows = []
    for (arm, clade, method), frame in frames.items():
        metrics = within_clade_metrics(
            {clade: frame},
            fitness_col=spec.fitness_metric,
            min_mutations=spec.min_clade_mutations,
        )
        for row in metrics.to_dict("records"):
            rows.append(
                {
                    "arm": arm,
                    "clade": row["clade"],
                    "method": method,
                    "pearson_r": row["pearson_r"],
                    "regression_slope": row["regression_slope"],
                    "n_mutations": row["n_mutations"],
                }
            )
    return pd.DataFrame(rows).sort_values(["arm", "clade", "method"], ignore_index=True)
