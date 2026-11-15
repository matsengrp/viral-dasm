"""The deep mutational scans `fig:dms` compares the model against.

Three published full-spike cell-entry assays, each matched to the clade whose
founder the model is scored on, read into one 20 x L matrix apiece and split
into the strata the figure reports.

Everything here is a function of its arguments and the files ``config.yaml``
names -- no model, no checkpoint, no GPU -- so the rules, the notebooks and
the tests share one implementation. Nothing here reads
``selection_factors.parquet``: a DMS matrix and the strata are properties of
the assay and the founder, so they are built before a model exists and do not
move when one is rescored.

**The site mask here is not the figure's exclusion.** It is the narrower thing
``from_multidms`` needs: that loader raises on a wildtype/reference
disagreement, and the disagreements all sit at masked sites, so the mask has
to be applied before it runs. What a *panel* excludes is
``panels.excluded_sites`` on the scored table's own columns, which is three
layers and is applied when the frames are built. The two are kept apart
because one is about which rows a loader can parse and the other is about
which mutations the models were trained on.

Published datasets, not GISAID-derived: their measurements, and anything
derived from them here, may be committed.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import (
    GENE,
    assert_dasm2_is_this_repo,
    clade_roster,
    dms_dataset,
    file_md5,
    mat_pcp_path,
)

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import pandas as pd
from dms_published import normalise_published
from founders import founder_aa_sequence, founder_nt_sequence
from panel_spec import FIGURE_MODEL
from panel_spec import TABLE_COLUMNS as BASE_TABLE_COLUMNS
from panels import BASE_SPEC, FilterSpec, base_frames
from scipy import stats

from dasm2.codon_table import CODON_NEIGHBOR_MATRIX
from dasm2.data.functional import to_dms_matrix
from dasm2.data.functional_formats import (
    NON_SUBSTITUTION_MUTANT_CODES,
    from_multidms,
)
from dasm2.sequence.constants import AA_STR_SORTED, CODONS
from dasm2.sequence.conversion import iter_codons
from dasm2.sequence.masking import (
    censored_codon_sites,
    load_censor_table,
    parse_masked_nt_sites_file,
)

#: The mutations one nucleotide change to the founder's codon reaches.
SINGLE_NT = "single_nt"

#: The rest: two or three changes away. The model is trained on none of them,
#: and mat-pcp estimates none of them, so both are making what `main.tex` calls
#: an "educated guess" here.
MULTI_NT = "multi_nt"

#: Every measured substitution, both strata together. Named rather than left
#: implicit because a readout that reports it has to say so: the three are not
#: nested the way a cutoff ladder's rungs are, they partition.
ALL = "all"

#: The strata every readout of this figure reports, in the order they are read.
STRATA: tuple[str, ...] = (SINGLE_NT, MULTI_NT, ALL)

#: The DMS CSV's own column names. mat-pcp and this pipeline both say
#: ``aa_site``/``mutant_aa``; these four are the published files' spelling, and
#: the rename happens once, on the way out of :func:`load_dms`.
#:
#: They are passed into ``from_multidms`` explicitly rather than left to its
#: defaults, which happen to be the same strings today: a file whose columns
#: were renamed would otherwise be pre-filtered on one set of names and parsed
#: on another.
DMS_SITE_COL = "site"
DMS_WT_COL = "wildtype"
DMS_MUT_COL = "mutant"
DMS_EFFECT_COL = "functional_effect"


@dataclass(frozen=True)
class DmsAccounting:
    """What each filter removed on the way to one homolog's matrix.

    Every drop is counted rather than left to a row-count difference, because
    the three have different meanings: a stop codon was never a substitution, a
    masked site is one the model never trained on, and a founder mismatch is a
    disagreement between the assay's construct and the tree's founder. A panel
    that reported only the total would read all three as coverage.

    ``mismatch_sites`` is the list rather than the count: there are few, they
    are the interesting ones, and a new one appearing is what a repointed
    dataset looks like.
    """

    raw_rows: int
    dropped_stop_or_deletion: int
    dropped_masked_site: int
    dropped_founder_mismatch: int
    mismatch_sites: tuple[int, ...]
    retained_substitutions: int

    def as_row(self) -> dict[str, object]:
        """The accounting as CSV columns, so a panel's table carries it."""
        row = asdict(self)
        row["mismatch_sites"] = ";".join(str(site) for site in self.mismatch_sites)
        return row


def codon_sites_of(nt_indices: Iterable[int]) -> set[int]:
    """The 1-indexed codon sites a set of 0-indexed nucleotide indices touches.

    The same widening ``scripts/score_run.py`` applies to build the scored
    table's ``globally_masked`` column, and deliberately a second spelling of
    it -- but not to keep torch out of the panel path, which already has it
    (this module imports ``dasm2.codon_table``, and ``scripts/panels.py``
    imports ``dasm2.analysis.viral_fitness_eval``). The reason is snakemake:
    moving these into a module ``score_run.py`` imports would change a declared
    input of ``rule score`` and mark every scored table out of date, and the
    training campaign that produces them is mid-flight. ``tests/test_dms.py``
    asserts the two agree on this run, which is what keeps the duplication
    honest until a shared module is safe to make.
    """
    return {int(index) // 3 + 1 for index in nt_indices}


def globally_masked_sites() -> set[int]:
    """Codon sites the global nucleotide mask covers."""
    return codon_sites_of(
        parse_masked_nt_sites_file(mat_pcp_path("masked_nt_sites_file"))
    )


def censored_sites(clade: str) -> set[int]:
    """Codon sites this clade's censor table touches at least one candidate in."""
    table = load_censor_table(
        mat_pcp_path("censored_mutations_file"),
        mat_pcp_path("censored_clades_file"),
    )
    return censored_codon_sites(table[clade])


def unreadable_sites(clade: str) -> set[int]:
    """Sites where the assay's wildtype may legitimately differ from the founder.

    The union of the global codon mask and this clade's censor table. It is
    what :func:`load_dms` drops before ``from_multidms`` runs, because that
    loader raises on a wildtype/reference disagreement and this is where the
    disagreements are.

    It is **not** the figure's exclusion, which adds the reversion layer and
    comes from ``panels.excluded_sites`` on the scored table's own columns.
    The reversion rule censors a change back toward the reference, so it is
    per mutation rather than per site; flattening it to a site set here would
    drop about eighteen trained mutations for every censored one, on rows this
    loader has no reason to touch.
    """
    return globally_masked_sites() | censored_sites(clade)


def single_nt_indicator(clade: str, gene: str = GENE) -> pd.DataFrame:
    """Which amino-acid substitutions one nucleotide change can reach.

    ``True`` at ``(aa, position)`` when one substitution in the founder's
    *actual* codon at that position yields ``aa``. Codon-aware on purpose: an
    amino-acid-only answer would take the best case over every synonymous
    realization of the parent residue, and here the founder's nucleotides are
    known, so the actual codon gives the honest and stricter split.

    Read from the founder rather than from mat-pcp's coverage. A mutation
    mat-pcp emitted no count for is still single-nucleotide if its codon is one
    change away, and a stratification taken from which mutations the counts
    table happens to carry would make the stratum a statement about sampling
    depth instead of about the codon table.

    Returns:
        A boolean frame indexed by :data:`AA_STR_SORTED`, one column per
        0-indexed position, laid out like a DMS matrix.
    """
    nt_founder = founder_nt_sequence(clade, gene)
    columns: dict[int, list[bool]] = {}
    for position, codon in enumerate(iter_codons(nt_founder)):
        if codon not in CODONS:
            # Refused rather than filled with False. A partially ambiguous
            # codon such as `ACN` still translates (to `T`), so the DMS
            # wildtype would match the founder and the site would survive the
            # mismatch drop -- and every mutation at it would then be labelled
            # multi-nucleotide, which is a wrong stratum rather than a missing
            # one. No founder in this run's roster has a non-ACGT base, so this
            # fires only if that changes.
            raise ValueError(
                f"{gene}/{clade}: codon {position + 1} is {codon!r}, which is "
                f"not one of the 64. A partially ambiguous codon would be "
                f"silently labelled multi-nucleotide at every mutation, so it "
                f"is refused rather than guessed at."
            )
        codon_index = CODONS.index(codon)
        columns[position] = [
            bool(CODON_NEIGHBOR_MATRIX[codon_index, aa_index].item())
            for aa_index in range(len(AA_STR_SORTED))
        ]
    return pd.DataFrame(columns, index=list(AA_STR_SORTED))


def verify_dms_md5(homolog: str) -> str:
    """Check one homolog's CSV against the digest ``config.yaml`` pins.

    A gate rather than a label. These files are copies of what three studies
    released, and a study can re-release one at the same path, so without this
    every number the figure reports would move while the pipeline kept
    running. `data/dms/PROVENANCE.md` records which release each copy is.

    Raises:
        ValueError: If the file's md5 is not the pinned one.
    """
    dataset = dms_dataset(homolog)
    digest = file_md5(dataset.path)
    if digest != dataset.md5:
        raise ValueError(
            f"{homolog}: {dataset.path} has md5 {digest}, but config.yaml pins "
            f"{dataset.md5}. These files are copies of what each study "
            f"released rather than anything built here, so a changed digest "
            f"means the copy changed: re-measure what this figure reports "
            f"rather than repinning."
        )
    return digest


def load_dms(homolog: str, gene: str = GENE) -> tuple[pd.DataFrame, DmsAccounting]:
    """One homolog's cell-entry DMS as a matrix, with its accounting.

    The filters run in a fixed order, and the order is load-bearing:

    1. Drop ``mutant`` in ``{"*", "-"}``. A stop codon is not an amino-acid
       substitution, and a surviving deletion row has no residue to place.
    2. Drop sites in :func:`unreadable_sites`. This **has** to precede
       ``from_multidms``, which raises on a wildtype/founder disagreement, and
       the masked set is exactly where those disagreements live.
    3. Drop whatever wildtype disagreement is left, counted rather than
       asserted away. KP.3.1.1's site 1253 is one: the construct's stop codon,
       outside every mask.

    Reversing 2 and 3 does not raise -- step 3 would clear the frame either way
    -- but it moves every masked disagreement into
    ``dropped_founder_mismatch``, so the accounting would report a set of
    construct-versus-founder differences that is mostly masking.

    Returns:
        The 20 x L matrix and a :class:`DmsAccounting` of what was removed.
    """
    dataset = dms_dataset(homolog)
    if dataset.clade not in clade_roster():
        raise ValueError(
            f"{homolog} names clade {dataset.clade}, which has no founder in "
            f"this mat-pcp run. Its measurements would join to nothing, and "
            f"the panel would be drawn on the other homologs alone."
        )
    verify_dms_md5(homolog)
    founder = founder_aa_sequence(dataset.clade, gene)
    mask_sites = unreadable_sites(dataset.clade)

    # The study's published file, converted to this pipeline's four columns
    # and filtered on that study's own terms. `raw_rows` is what the
    # conversion emits rather than what the file holds, so the accounting
    # below measures this loader's masking rather than the study's filtering,
    # which `data/dms/PROVENANCE.md` records instead.
    raw = normalise_published(pd.read_csv(dataset.path), dataset)
    raw_rows = len(raw)

    frame = raw[~raw[DMS_MUT_COL].isin(NON_SUBSTITUTION_MUTANT_CODES)]
    after_stop = len(frame)

    frame = frame[~frame[DMS_SITE_COL].isin(mask_sites)]
    after_mask = len(frame)

    founder_aa = frame[DMS_SITE_COL].astype(int).map(lambda site: founder[site - 1])
    mismatch = frame[DMS_WT_COL] != founder_aa
    mismatch_sites = tuple(sorted({int(s) for s in frame.loc[mismatch, DMS_SITE_COL]}))
    frame = frame[~mismatch]

    # `from_multidms` builds full mutant sequences and needs a file, and it is
    # the one loader for this format, so the filtered frame goes back through a
    # temporary CSV rather than being pivoted by hand here. Keeping its path
    # is what lets the retained-substitution counts below be compared against
    # the numbers the original comparison published.
    with tempfile.TemporaryDirectory() as scratch:
        filtered = Path(scratch) / "filtered_multidms.csv"
        frame.to_csv(filtered, index=False)
        tidy = from_multidms(
            filtered,
            wt_sequence=founder,
            wt_column=DMS_WT_COL,
            site_column=DMS_SITE_COL,
            mutant_column=DMS_MUT_COL,
            func_score_column=DMS_EFFECT_COL,
        )
    tidy = tidy[~tidy["is_reference"]]

    matrix = to_dms_matrix(
        tidy, y_col="y_functional_effect", reference_sequence=founder
    )
    accounting = DmsAccounting(
        raw_rows=raw_rows,
        dropped_stop_or_deletion=raw_rows - after_stop,
        dropped_masked_site=after_stop - after_mask,
        dropped_founder_mismatch=after_mask - len(frame),
        mismatch_sites=mismatch_sites,
        retained_substitutions=len(tidy),
    )
    # Every row is either counted as dropped or retained. `from_multidms`
    # warns rather than raises on a row with no score or a mutant equal to its
    # wildtype, and `to_dms_matrix` overwrites a duplicate `(site, mutant)`
    # silently, so without this a repointed file could lose rows with nothing
    # saying so. All three hold on today's files.
    counted = (
        accounting.dropped_stop_or_deletion
        + accounting.dropped_masked_site
        + accounting.dropped_founder_mismatch
        + accounting.retained_substitutions
    )
    if counted != accounting.raw_rows:
        raise ValueError(
            f"{homolog}: {accounting.raw_rows} rows in, {counted} accounted "
            f"for. {accounting.raw_rows - counted} were dropped by a filter "
            f"this accounting does not name -- a missing score, a mutation "
            f"equal to its wildtype, or a duplicate `(site, mutant)`."
        )
    measured = int(matrix.notna().to_numpy().sum())
    if measured != accounting.retained_substitutions:
        raise ValueError(
            f"{homolog}: the matrix holds {measured} measured cells against "
            f"{accounting.retained_substitutions} retained rows, so a "
            f"duplicate `(site, mutant)` overwrote a cell."
        )
    return matrix, accounting


def melt_dms(matrix: pd.DataFrame, clade: str, gene: str = GENE) -> pd.DataFrame:
    """One homolog's matrix as long rows, keyed the way this pipeline joins.

    Columns are ``aa_site`` (1-indexed), ``mutant_aa``, ``dms_cell_entry`` and
    ``stratum`` -- the first two being the names ``selection_factors.parquet``
    carries, so the join needs no renaming on either side.

    Unmeasured cells are dropped. The matrix is dense over 20 x L and an assay
    covers neither every site nor every substitution, so a frame that kept them
    would report its own padding as coverage.
    """
    indicator = single_nt_indicator(clade, gene)
    long = (
        matrix.rename_axis(index="mutant_aa", columns="position")
        .stack()
        .rename("dms_cell_entry")
        .reset_index()
    )
    long = long[long["dms_cell_entry"].notna()]
    reachable = (
        indicator.rename_axis(index="mutant_aa", columns="position")
        .stack()
        .rename("single_nt")
        .reset_index()
    )
    merged = long.merge(reachable, on=["mutant_aa", "position"], how="left")
    if merged["single_nt"].isna().any():
        missing = int(merged["single_nt"].isna().sum())
        raise ValueError(
            f"{clade}: {missing} measured cells have no entry in the "
            f"single-nucleotide indicator, so the DMS matrix and the founder "
            f"disagree on their site range."
        )
    merged["aa_site"] = merged["position"].astype(int) + 1
    merged["stratum"] = merged["single_nt"].map({True: SINGLE_NT, False: MULTI_NT})
    return merged[["aa_site", "mutant_aa", "dms_cell_entry", "stratum"]]


# ---------------------------------------------------------------------------
# The figure's frames: the assay joined to what `rule score` already computed
# ---------------------------------------------------------------------------

#: The pooled counts estimate panel A's third column and panel B's third bar
#: are drawn from. `panel_spec.TABLE_COLUMNS` leaves every `*_all` column out,
#: because no panel of `fig:agreement` reads one; this figure's "full-tree
#: counts" predictor is exactly that column, so it is named here rather than
#: by widening a tuple every other panel pays for.
POOLED_METRIC = "delta_fitness_ratio_all"

#: What this figure reads out of `selection_factors.parquet`.
TABLE_COLUMNS: tuple[str, ...] = (*BASE_TABLE_COLUMNS, POOLED_METRIC)

#: **This figure does not clip.** The source experiment drew it at
#: `dms_comparison.py`'s `DEFAULT_CLIP_MIN` / `DEFAULT_CLIP_MAX` of -8 and +6,
#: and that is a deliberate divergence here: the other two figures clip a
#: counts estimate against a counts estimate, where the floor is set by how
#: deeply the clade was sampled, but here the y axis is an assay with a floor
#: of its own and every predictor is plotted and correlated as the model
#: reports it.
#:
#: The bound bit on one point in 5,415 on this run -- KP.3.1.1's R346T, pinned
#: at +6 -- so what it bought was not a different picture but a hidden one:
#: a correlation cannot see the difference, which is the reason to show the
#: value rather than the bound.
#:
#: :func:`dms_spec` passes ``clip_lower=None``, which :func:`clip_axes`
#: accepts; it writes the ``*_raw`` columns either way, so nothing downstream
#: needs to know clipping is off.
CLIP_LOWER = None

#: The two populations the figure draws, top row then bottom.
#:
#: 0 is not the no-op it looks like. A mutation mat-pcp never estimated
#: carries no count at all and no fitness effect, so the top row is already
#: restricted to what the tree could estimate -- which, as the source
#: experiment establishes, is the single-nucleotide-reachable set. 10 is the
#: well-observed subset of it.
MIN_COUNTS: tuple[int, ...] = (0, 10)

#: The three predictors, in the order both panels draw them: the column each
#: reads, and the name the figure gives it. Ported from `dms_comparison.py`'s
#: `_specs` and `_methods`, which are the same three in the same order.
METHODS: tuple[tuple[str, str], ...] = (
    ("log_selection_factor", "DASM"),
    ("delta_fitness_ratio", "per-clade counts"),
    (POOLED_METRIC, "full-tree counts"),
)

#: The column every predictor is correlated against.
TARGET = "dms_cell_entry"


def dms_spec(min_count: int) -> FilterSpec:
    """The filter one row of `fig:dms` is drawn under.

    ``clip_lower=None`` is the whole of :data:`CLIP_LOWER`'s content: this
    figure plots what the model reports. `write_panel_csv` stamps the field
    either way, so the tracked table says so -- an empty ``spec_clip_lower``
    is "no bound", beside `fig:agreement`'s -4 and `fig:subsampling`'s -6.
    """
    return replace(BASE_SPEC, min_count=float(min_count), clip_lower=CLIP_LOWER)


@cache
def _measured_effects(homolog: str) -> pd.DataFrame:
    """One homolog's assay, melted and tagged, read once per process.

    :func:`dms_frames` draws every homolog at two count thresholds, and the
    assay does not move between them: the floor is on the scored table's
    counts, not on the measurement. Without this, each homolog's CSV would be
    read, digested, parsed by ``from_multidms``, masked against its censor
    table and melted twice over.

    A copy is returned, so a caller that adds a column to its frame cannot
    write into the cached one.
    """
    dataset = dms_dataset(homolog)
    matrix, _accounting = load_dms(homolog)
    return melt_dms(matrix, dataset.clade)


def dms_frame(
    table: pd.DataFrame,
    homolog: str,
    *,
    model: str = FIGURE_MODEL,
    min_count: int = 0,
) -> pd.DataFrame:
    """One homolog's measured effects beside the three predictors of them.

    The tidy table `build_tidy.py` assembles in the source experiment, minus
    the scoring pass: `rule score` already evaluated the checkpoint on every
    clade founder and joined mat-pcp's per-clade and pooled estimates onto it,
    so what is left here is the assay.

    **The population is this pipeline's, not the source's.** There the mask
    was read from `counts_all.csv`'s `exclude` and `masked_in_usher` flags and
    widened to whole codon sites; here the same exclusion arrives already
    applied, per nucleotide route, as the nulls `rule score` left behind --
    which :func:`base_frames` drops. That is the one reason the retained
    counts differ, and :class:`DmsAccounting` reports what the loader dropped
    on top of it.

    Args:
        table: :func:`read_scored_table` over :data:`TABLE_COLUMNS`.
        homolog: One of ``config.yaml``'s, which fixes both the clade and the
            assay.
        model: The checkpoint's model name; the paper's is `dasm-1077k`.
        min_count: The actual-or-expected floor; see :data:`MIN_COUNTS`.

    Returns:
        One row per measured, estimated mutation, carrying ``dms_cell_entry``,
        the three predictors as the model and the tree report them, the
        ``*_raw`` columns :func:`clip_axes` writes whether or not clipping is
        on, and the ``mutation`` and ``site`` the scatter's tooltip reads.
    """
    dataset = dms_dataset(homolog)
    measured = _measured_effects(homolog).copy()
    frames = base_frames(
        table, spec=dms_spec(min_count), models=[model], clades=[dataset.clade]
    )
    frame = frames.get((model, dataset.clade))
    if frame is None:
        raise ValueError(
            f"{homolog}: the scored table carries no {model} rows for clade "
            f"{dataset.clade}, so there is nothing to set the assay against."
        )
    joined = frame.merge(measured, on=["aa_site", "mutant_aa"], how="inner")
    # `clip_axes` wrote `*_raw` for `log_selection_factor` and for
    # `FITNESS_METRICS`; the pooled estimate is in neither, so it gets its own
    # here. With the clip off these equal their columns, and they are kept so
    # that every predictor spells the pair the same way whoever reads it.
    for column, _label in METHODS:
        if f"{column}_raw" not in joined:
            joined[f"{column}_raw"] = joined[column]
    return joined.assign(
        homolog=homolog,
        mutation=joined["aa_mutation"],
        site=joined["aa_site"],
    )


def dms_frames(
    table: pd.DataFrame,
    homologs: Sequence[str],
    *,
    model: str = FIGURE_MODEL,
    min_counts: Sequence[int] = MIN_COUNTS,
) -> dict[tuple[str, int], pd.DataFrame]:
    """``{(homolog, min_count): frame}`` over every cell the figure draws."""
    return {
        (homolog, min_count): dms_frame(
            table, homolog, model=model, min_count=min_count
        )
        for homolog in homologs
        for min_count in min_counts
    }


def dms_correlations(frames: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    """Panel B's and panel D's table: one r per homolog, threshold and method.

    Ported from `dms_comparison.py`'s `corr_vs_target` and the cell that calls
    it: the correlation against the measurement with the ``n`` it was computed
    over, dropping the rows where either side is missing.

    Spearman rides along, as `corr_vs_target` returns it. The bars draw
    Pearson -- the source figure's own choice, because these subsets are the
    regime where the predictors are close -- and the CSV carries both, so the
    rank correlation the source README reports needs no second pass.
    """
    rows = []
    for (homolog, min_count), frame in frames.items():
        for column, method in METHODS:
            pair = frame.dropna(subset=[column, TARGET])
            if len(pair) > 1:
                pearson = float(stats.pearsonr(pair[column], pair[TARGET])[0])
                spearman = float(stats.spearmanr(pair[column], pair[TARGET]).statistic)
            else:
                pearson = spearman = float("nan")
            rows.append(
                {
                    "homolog": homolog,
                    "min_count": min_count,
                    "method": method,
                    "column": column,
                    "pearson": pearson,
                    "spearman": spearman,
                    "n": len(pair),
                }
            )
    return pd.DataFrame(rows)
