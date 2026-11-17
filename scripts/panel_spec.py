"""What `fig:agreement` and `fig:within_clade_shifts` are made of: their
panels, their selections, their paths.

Separate from `scripts/panels.py`, and stdlib plus `lib` only. `workflow/
figures.smk` needs these names to declare a rule's outputs, and a `.smk` is
parsed on every snakemake invocation -- a dry run and a `--lint` included. With
the panel code imported there instead, each of those would pull in `dasm2`,
pandas and pyarrow, and pay the fork's import-time assertions, to learn a
handful of file names.

So this is the one spelling of each of these facts. `scripts/panels.py` and
`workflow/figures.smk` both read them from here, which is what makes a rule's
declared output the path the notebook actually writes.
"""

from __future__ import annotations

import csv
import re
import shlex
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import (
    ARTIFACT_ROOT,
    CORPUS_NAMES,
    FITNESS_SUFFIX,
    FULL_DEPTH,
    REPO_ROOT,
    SAMPLE_PATHS_SUFFIX,
    TEX_FIGURE_DIR,
    cap_of,
)

#: The committed era partition: which clades each era's GLM was fitted over.
#: `workflow/corpus.smk` spells this too, for the rule that renders the map.
#:
#: **Here rather than in `scripts/lib.py`, deliberately.** Every training rule
#: declares `lib.py` as an input, so a helper added there puts the seven runs
#: out of date and `snakemake -n figures` schedules `train` -- which is the one
#: thing this pipeline must never do for a figure. This module is declared by
#: the figure rules and by nothing upstream of them.
#:
#: The partition is viral-dasm's own choice, and reading it here rather than
#: importing mat-pcp's `era.py` is the seam `dasm2/PROVENANCE.md` records.
ERA_TEMPLATE = REPO_ROOT / "data" / "eras-pre-omicron-omicron.yaml"


def era_of_clade(clade: str) -> str:
    """Which era's fitted rate table a clade's expected counts are built from.

    ``yaml`` is imported in the body rather than at module scope: this module
    is parsed on every snakemake invocation, a dry run included, and its job
    is to stay cheap enough that a `.smk` can read a path out of it.

    Raises:
        KeyError: If the clade is in no era. Defaulting would silently price a
            clade's opportunity off the wrong fit, which reads downstream as
            the model getting that clade wrong.
    """
    import yaml

    spec = yaml.safe_load(ERA_TEMPLATE.read_text())
    for era, body in spec["eras"].items():
        if clade in body["clades"]:
            return era
    raise KeyError(
        f"{clade} is in no era of {ERA_TEMPLATE}, so there is no fitted rate "
        f"table for its expected counts."
    )


#: Where the panel images go: under the artifact root, not in the tree. A panel
#: is a rebuildable picture of the CSV committed beside it, and `README.md`'s
#: layout gives tracked `results/` to the tables.
FIGURE_DIR = ARTIFACT_ROOT / "figures"

#: Where the tables go: tracked, one CSV per panel, holding exactly what that
#: panel plots.
PANEL_CSV_DIR = REPO_ROOT / "results" / "panels"

#: Peak RSS for one panel job, in MB. One model's full-depth rows across every
#: clade is about 1.4M rows over 15 columns; the `fidelity` selection reads two
#: models' worth, and `shift_frame` pairs clades into frames of their own.
PANEL_MEM_MB = 8_000

#: The model every panel of the published figure is drawn from: the largest
#: transformer. The subsampling figure is the one exception to that policy, and
#: it is not this figure.
FIGURE_MODEL = "dasm-1077k"

#: The two models the published CSVs were drawn from, for the port-fidelity
#: gate. `dasm-293k` is `size_large`'s architecture field for field, and
#: `dasm-85k` is the ladder's `transformer`; the published panels C and E pool
#: the two under one median.
FIDELITY_MODELS: tuple[str, ...] = ("dasm-293k", "dasm-85k")

#: The two model selections the panels are drawn for, and the one name each is
#: built under. Two, because they answer different questions: `figure` is what
#: `fig:agreement` publishes, and `fidelity` is the pair the published CSVs were
#: drawn from. Both come out of one table and one `groupby`, so the second costs
#: no rescoring.
#:
#: **`fidelity` no longer reproduces upstream, and is not meant to.** It showed
#: the port computing what `score_runs` computed, and did, at the port (#21).
#: #36 then narrowed the exclusion from the codon site to the mutation, which
#: upstream does not do, so the two now differ by exactly that: 67,564 mutations
#: at rung 10 against upstream's 65,430, over the same 22 clades, at a pooled
#: median r of 0.9662 against 0.9664. What the selection is still good for is a
#: second model pair through the same code, and a diff a reviewer can read when
#: this filter changes again. Reproducing upstream is now the port's history
#: rather than a live gate, and it lives in #21.
#:
#: Panels C and E pool the models of their selection under one median, which is
#: what the published numbers do -- so pooling is a property of the selection
#: rather than a fixed groupby, or the fidelity gate could not be run at all.
SELECTIONS: dict[str, tuple[str, ...]] = {
    "figure": (FIGURE_MODEL,),
    "fidelity": FIDELITY_MODELS,
}

#: The selection a rule builds unless it is asked for the other one.
DEFAULT_SELECTION = "figure"

#: The corpus arm every panel of this figure reads. The thinned arms are the
#: subsampling figure's axis, and mixing one in here would confound the model
#: with the corpus.
ARM = "full_depth"

#: The rungs panels C and E sweep: a prefix of the library's
#: `COUNT_THRESHOLD_LADDER`, which this module cannot import -- that is what it
#: is for. `tests/test_panels.py` pins the two together, since a literal here
#: that drifted from the library's rungs would sweep a different axis from the
#: one the filter applies.
#:
#: The top rung is dropped: 100 is the rung that would set the paired clade set
#: for every other rung, so the whole sweep -- the lowest rung included -- would
#: be drawn on a clade set chosen by the rung nobody reads.
COUNT_LADDER: list[int] = [10, 20, 50]

#: The single cutoff the per-mutation panels are drawn at, which is the ladder's
#: first rung rather than a fourth spelling of 10. Tied, not repeated: panel B's
#: cloud is the data behind panel C's leftmost dot, and a cutoff that drifted
#: from the rung would make them different mutations with nothing saying so.
PANEL_MIN_COUNT = COUNT_LADDER[0]

#: The clades the shift panels draw pairs from. Quadratic in the number of
#: clades -- four is six pairs, the full roster would be 1,540 -- so this is a
#: selection rather than the roster.
SHIFT_CLADES: tuple[str, ...] = ("21J", "21L", "23A", "24A")

#: The clade panel B scatters. One clade, because the panel's job is to show
#: the cloud behind one of panel C's dots.
SCATTER_CLADE = "21J"

#: The mutation panel A traces across the clades.
LOOKUP_MUTATION = "F456L"

# ---------------------------------------------------------------------------
# `fig:within_clade_shifts`
# ---------------------------------------------------------------------------

#: Rows per chunk when streaming the PCP corpus. The file is 112 MB gzipped
#: and every row carries a 3,819-nucleotide parent, so it is read in pieces
#: and reduced as it goes rather than held.
#:
#: Here rather than in `scripts/panels.py`, which is where it used to live:
#: four modules now stream the corpus, and this one is stdlib-only, so a
#: `.smk` can read it without pulling pandas into a dry run. `panels`
#: re-exports it, so `from panels import PCP_CHUNK_ROWS` still resolves.
PCP_CHUNK_ROWS = 200_000

#: The clades that figure's rows cover, in the order the figure reads: the
#: three at the head of the outlier ranking. Separate from
#: :data:`SHIFT_CLADES`, which is `fig:agreement`'s cross-clade pairing and
#: whose consumers would change meaning if these were added to it.
CANDIDATE_CLADES: tuple[str, ...] = ("22D", "20B", "21K")

#: The candidate list panels A and B share, as tracked data rather than two
#: hand-maintained lists. Moving a candidate off the figures is one edit that
#: reaches both panels.
CANDIDATE_CSV = REPO_ROOT / "data" / "shift_candidates.csv"

#: What a candidate's `status` may be. `figure` reaches the panels as a
#: *subject*: a mutation whose effect panels B, D and F measure on two
#: backgrounds. `null` is the designed null, measured and deliberately not
#: drawn; `held_back` is measured and off the figure for some other reason the
#: file records. The two non-figure reasons stay distinct rather than
#: collapsing to "excluded".
#:
#: **The mutations that define the backgrounds are not among these.** They are
#: derived from branch frequencies by `scripts/backgrounds.py` and written to
#: :data:`BACKGROUND_CSV`, so this file holds only what a person chose.
CANDIDATE_ON_FIGURE = "figure"
CANDIDATE_STATUSES = (CANDIDATE_ON_FIGURE, "null", "held_back")

#: The columns :func:`_candidate_rows` requires, found by name.
CANDIDATE_COLUMNS = ("clade", "mutation", "status")

#: Where the derived background definition goes: one row per substitution that
#: defines a clade's two backgrounds, with the branch-mass fractions it was
#: chosen on. In `results/` rather than `data/`, because the pipeline writes
#: it; model-free, so not under a selection.
BACKGROUND_CSV = REPO_ROOT / "results" / "backgrounds" / "clade_backgrounds.csv"

#: Where the clade emergence dates go: one row per clade, with the sample
#: count the date was measured over. `fig:subsampling`'s panel A colours its
#: dots by it. In `results/` and not under a selection, for the reasons
#: :data:`BACKGROUND_CSV` gives -- the pipeline writes it, and no model
#: touches it.
EMERGENCE_CSV = REPO_ROOT / "results" / "backgrounds" / "clade_emergence.csv"

#: The role a background row may carry. `marker` is the clade's most abundant
#: substitution, `cotraveller` one carried by most of the marker's branches,
#: and `fixed` one carried by nearly the whole clade -- which belongs to both
#: backgrounds and so defines neither.
BACKGROUND_SPLIT_ROLES = ("marker", "cotraveller")

#: Which clades name their background mutations on panel A. Not all of them:
#: the panel's subject is where the two approaches disagree, and a label on a
#: mark sitting on the diagonal points at nothing for the reader to see. Only
#: 22D's background mutations have swept far enough for the counts approach to
#: understate them, which is what puts them off the diagonal and what the
#: labels are there to explain. 20B's and 21K's are on it.
#:
#: The split is still derived for every clade -- it is what panels B, D and F
#: contrast -- and this governs only which of them panel A draws.
BACKGROUND_LABEL_CLADES: tuple[str, ...] = ("22D",)

#: Where the scored tables live. `workflow/score.smk` spells this too, for
#: `rule score`'s own output; this module's job is to be the one spelling for
#: everything the figures read, so a later haplotype rule takes it from here.
SCORED_DIR = ARTIFACT_ROOT / "scored"

#: Where the per-haplotype scores go: its own table, beside the founder one.
#: **Not** rows in `selection_factors.parquet`. That table's `background` is
#: empty for every row and a test pins it so, and `panels.base_frames` filters
#: on `(model, clade)` alone -- a haplotype row sharing it would be pooled into
#: every agreement panel with nothing saying so.
HAPLOTYPE_TABLE = SCORED_DIR / "haplotype_selection_factors.parquet"

#: Where the figure's clades get their distinct amino-acid parents enumerated.
#: Under the artifact root: a row is a Spike haplotype, so it is
#: GISAID-derived. `scripts/clade_haplotypes.py` writes it and
#: `rule score_haplotypes` scores it.
CLADE_HAPLOTYPE_TABLE = ARTIFACT_ROOT / "backgrounds" / "clade_haplotypes.parquet"

#: Where the pooled counts land: one row per (clade, candidate, pool, cell)
#: and one per (clade, candidate) shift, plus the randomization test. Under
#: the artifact root for the same reason -- a pool is named by the parents it
#: holds -- with the per-panel CSVs under `results/` carrying only labels.
POOL_DIR = ARTIFACT_ROOT / "pools"
POOL_CELLS = POOL_DIR / "pool_cells.parquet"
POOL_SHIFTS = POOL_DIR / "pool_shifts.parquet"
POOL_SUMMARY = POOL_DIR / "pool_summary.json"


#: ``<founder><site><mutant>``, e.g. ``D574V``. A candidate is named against
#: the clade founder, which is why a consumer wants the pieces: on a haplotype
#: that already differs from the founder at that site, the mutation's own
#: string is spelled differently and a string match would miss it.
MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_mutation(text: str) -> tuple[str, int, str]:
    """``"D574V"`` -> ``("D", 574, "V")``, the founder residue, 1-based site
    and mutant residue.

    One spelling, because two consumers need it -- the scorer, to pick the
    cells of a haplotype's grid, and the counts side, to build the at-risk
    requirement -- and a site read one-based in one place and zero-based in
    the other is the error this exists to make impossible.
    """
    match = MUTATION_RE.match(text)
    if match is None:
        raise ValueError(
            f"{text!r} is not a mutation of the form D574V (founder residue, "
            f"1-based amino acid site, mutant residue)."
        )
    return match.group(1), int(match.group(2)), match.group(3)


def candidate_cells() -> dict[str, tuple[tuple[int, str], ...]]:
    """``{clade: ((aa_site, mutant_aa), ...)}`` for every candidate.

    Every candidate, not only the ones drawn: the designed null and the
    held-back nomination are measured through the same pipeline, which is
    what makes them a null and a comparison rather than two rows nobody ran.

    Keyed on ``(site, residue)`` rather than on the mutation string, which is
    what a haplotype's grid can actually be indexed by.
    """
    cells: dict[str, list[tuple[int, str]]] = {}
    for clade, mutation in [
        *figure_candidates(),
        *((clade, mutation) for clade, mutation, _ in off_figure_candidates()),
    ]:
        _, site, mutant = parse_mutation(mutation)
        cells.setdefault(clade, []).append((site, mutant))
    return {clade: tuple(sorted(set(v))) for clade, v in cells.items()}


def figure_candidates() -> tuple[tuple[str, str], ...]:
    """The `(clade, mutation)` pairs the figures draw, in file order.

    The off-figure rows are dropped here rather than by each caller, so panel A
    and panel B cannot disagree about their subjects -- which is the whole
    reason the list is one file.
    """
    return tuple(
        (clade, mutation)
        for clade, mutation, status in _candidate_rows()
        if status == CANDIDATE_ON_FIGURE
    )


def labelled_candidates() -> tuple[tuple[str, str, str], ...]:
    """The `(clade, mutation, group)` rows panel A names, in drawing order.

    Two sources, one list: the subjects a person chose, from
    :data:`CANDIDATE_CSV`, and the substitutions that define each clade's two
    backgrounds, derived into :data:`BACKGROUND_CSV`. Returned together rather
    than as two calls because the panel stacks its label column over every
    labelled row at once, and a caller that fetched the groups separately would
    stack each blind to the other's.

    The `fixed` rows are left out: they belong to both backgrounds, so they
    distinguish nothing and naming them on the panel would say they do. So are
    the clades outside :data:`BACKGROUND_LABEL_CLADES`, whose background
    mutations sit on the diagonal.
    """
    rows = [
        (clade, mutation, CANDIDATE_ON_FIGURE)
        for clade, mutation in figure_candidates()
    ]
    # `csv`, not pandas: this module is imported by the Snakefile to build the
    # DAG, and it stays free of the heavy imports for that reason.
    with BACKGROUND_CSV.open(newline="") as handle:
        rows.extend(
            (row["clade"], row["mutation"], "background")
            for row in csv.DictReader(handle)
            if row["role"] in BACKGROUND_SPLIT_ROLES
            and row["clade"] in BACKGROUND_LABEL_CLADES
        )
    return tuple(rows)


def off_figure_candidates() -> tuple[tuple[str, str, str], ...]:
    """The `(clade, mutation, status)` rows measured but kept off the figures.

    Reported rather than silently dropped: a designed null and a held-back
    nomination are off the figure for different reasons, and a reader checking
    the panel against the source needs to see both.

    """
    return tuple(
        (clade, mutation, status)
        for clade, mutation, status in _candidate_rows()
        if status != CANDIDATE_ON_FIGURE
    )


def _candidate_rows() -> list[tuple[str, str, str]]:
    """Every candidate row, validated.

    Columns are found by name, not position, so adding one cannot silently
    shift what is read. An unknown `status` raises here rather than quietly
    landing a candidate on or off the figure.
    """
    with CANDIDATE_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            name for name in CANDIDATE_COLUMNS if name not in (reader.fieldnames or ())
        ]
        if missing:
            raise ValueError(
                f"{CANDIDATE_CSV}: missing column(s) {missing}; it must carry "
                f"{list(CANDIDATE_COLUMNS)}."
            )
        rows = list(reader)
    out = []
    seen = set()
    for row in rows:
        status = row["status"]
        if status not in CANDIDATE_STATUSES:
            raise ValueError(
                f"{CANDIDATE_CSV}: {row['clade']}/{row['mutation']} has status "
                f"{status!r}, which is not one of {list(CANDIDATE_STATUSES)}."
            )
        pair = (row["clade"], row["mutation"])
        if pair in seen:
            raise ValueError(
                f"{CANDIDATE_CSV}: {pair[0]}/{pair[1]} appears twice, which "
                f"would build a DAG whose two jobs write the same output."
            )
        seen.add(pair)
        out.append((row["clade"], row["mutation"], status))
    return out


# ---------------------------------------------------------------------------
# `fig:subsampling`
# ---------------------------------------------------------------------------

#: The directory `fig:subsampling`'s panels are written under -- a **directory
#: name, not a model selection**, and deliberately absent from
#: :data:`SELECTIONS`.
#:
#: `SELECTIONS` is `rule panels`' wildcard domain, so a third entry there would
#: let the agreement rules resolve `selection=subsampling` and draw agreement
#: panels into this figure's directory. `figure_dir` and `panel_csv_path` are
#: plain joins and validate nothing, so the two new rules pass this string with
#: no `{selection}` wildcard and `tests/test_panels.py` pins it out of
#: `SELECTIONS`.
SUBSAMPLING_SELECTION = "subsampling"

#: The directory `fig:dms`'s panels are written under -- a **directory name,
#: not a model selection**, and absent from :data:`SELECTIONS` for the same
#: reason :data:`SUBSAMPLING_SELECTION` is. `tests/test_panels.py` pins it out.
DMS_SELECTION = "dms"

#: Which panel directories hold a composition the manuscript takes. See
#: :func:`tex_figure_dir`, which is where the distinction bites: `fidelity`
#: is the one directory that composes a figure the paper does not publish.
PUBLISHED_SELECTIONS: frozenset[str] = frozenset(
    {DEFAULT_SELECTION, SUBSAMPLING_SELECTION, DMS_SELECTION}
)

#: The two clades panel B draws, under the names the paper uses beside the
#: labels this pipeline keys on. `main.tex` names clades by their Pango or WHO
#: alias and every table here is keyed by the bare Nextstrain clade: Delta is
#: `21J` and XFG is `25C`. The tree's own metadata spells the pair -- its
#: `Nextstrain_clade_usher` field reads `25C (XFG)` over `XFG.*` lineages --
#: and the panel's facet titles use that form, so a reader of the paper and a
#: reader of `results/panels/` can follow the same figure.
ECDF_CLADES: dict[str, str] = {"21J": "21J (Delta)", "25C": "25C (XFG)"}

#: The counts panel B's eCDF is evaluated at. Linear and stopping at 20: the
#: accuracy threshold is 10, so this is the range either side of the cutoff the
#: rest of the paper reads everything at, and a log axis would spend most of
#: its width past the point where the curves have flattened.
ECDF_MAX_COUNT = 20
ECDF_POINTS = 201


def subsampling_caps() -> tuple[int, ...]:
    """Each thinned arm's per-clade cap, deepest first.

    Read off the corpus names rather than restated -- `cap_50000_0` is a cap of
    50,000 -- so panel A's rules cannot drift from the arms that were built.
    Panel A draws one rule per cap, and every clade above a rule is cut back
    *to* it, which is the whole difference between a cap arm and a fraction
    arm.
    """
    return tuple(cap_of(name) for name in CORPUS_NAMES if name.startswith("cap_"))


def aamut_grid_path(root: Path, clade: str, arm: str) -> Path:
    """One clade's amino-acid grid at one sampling depth.

    Full depth is the **top level** of `aamut_fitness_dir` and each thinned arm
    is a subdirectory named for its corpus. Here rather than in the two callers
    because `workflow/figures.smk` declares these files as `rule
    subsampling_depth`'s inputs and `scripts/panels.py` opens them: a rule that
    declared one layout while the notebook read another would be a
    `MissingInputException` at best and a silently undeclared input at worst.
    """
    directory = root if arm == FULL_DEPTH.name else root / arm
    return directory / f"{clade}{FITNESS_SUFFIX}"


# ---------------------------------------------------------------------------
# `fig:dms`
# ---------------------------------------------------------------------------

#: The homolog whose scatter is panel A. The paper draws one homolog's cloud
#: and leaves the other two to panel B's summary, so the choice is the
#: figure's rather than this pipeline's: `main.tex`'s caption names KP.3.1.1.
#: `tests/test_dms.py` requires it to be one of `config.yaml`'s homologs, since
#: this module stays stdlib-and-`lib` only and does no config lookup of its own.
SCATTER_HOMOLOG = "KP.3.1.1"

#: The two populations `fig:dms` draws, top row then bottom: every mutation
#: the tree could estimate at all, and the subset with at least ten actual or
#: expected counts. The figure's own spelling of `dms.MIN_COUNTS`, which is
#: where the two are defined; named here because they are in the panel stems,
#: and a rule has to know which files it writes without importing the loader.
DMS_MIN_COUNTS: tuple[int, ...] = (0, 10)

#: `fig:dms`'s four stems. One scatter row and one bar row per threshold,
#: each saved on its own rather than as a slice of a stacked pair: a single
#: image of both rows would fix the split between them at whatever Vega-Lite
#: produced, and the composition has no way to move it. Four images let the
#: layout put the break in the same place on both sides, which is what makes
#: the scatters and the bars line up.
DMS_SCATTER_PANELS: tuple[str, ...] = tuple(
    f"panel_{letter}_scatters_min{min_count}"
    for letter, min_count in zip("ac", DMS_MIN_COUNTS)
)
DMS_BAR_PANELS: tuple[str, ...] = tuple(
    f"panel_{letter}_bars_min{min_count}"
    for letter, min_count in zip("bd", DMS_MIN_COUNTS)
)
DMS_PANELS: tuple[str, ...] = (*DMS_SCATTER_PANELS, *DMS_BAR_PANELS)

#: The composed figure's stem, which is its LaTeX label: `main.tex` includes
#: `figures/dms.pdf` and refers to `\\autoref{fig:dms}`.
DMS_FIGURE_STEM = "dms"

#: Which stem each letter is, reading across and then down: one row per
#: threshold, scatters then bars. All four are lettered, including the two
#: that :data:`DMS_FIGURE_ROWS` no longer lays out: the letter is in the
#: stem, so renumbering them would rename files and the CSVs beside them for
#: a change the page does not see.
DMS_PANEL_LETTERS: dict[str, str] = {
    chr(ord("A") + index): stem
    for index, stem in enumerate(
        stem
        for scatter, bars in zip(DMS_SCATTER_PANELS, DMS_BAR_PANELS)
        for stem in (scatter, bars)
    )
}

#: The threshold the composed figure draws. `rule dms_correlations` writes a
#: panel at each of :data:`DMS_MIN_COUNTS`, but only this one is laid out, so
#: only this one's headings are the figure's: see :data:`DMS_FIGURE_ROWS`.
DMS_FIGURE_MIN_COUNT = 0

#: Which letter sits where: one row, the scatters then the bars, both drawn
#: over every mutation carrying a counts-based estimate. Reading across
#: answers "does this hold for the other homologs". Rows are top aligned and
#: left justified, so the bar panel starts level with the scatters beside it.
#:
#: **C and D are still drawn, and are not laid out.** The thresholded rung is
#: a check on the figure rather than part of it --- at ten counts KP.3.1.1 is
#: down to 407 mutations --- and `main.tex` reports it in prose instead. Its
#: panels stay in :data:`DMS_PANELS`, so the rule still writes them and their
#: CSVs; dropping them from this table is the whole of what takes them off
#: the page.
DMS_FIGURE_ROWS: tuple[tuple[str, ...], ...] = (("A", "B"),)

#: `fig:agreement`'s box rather than `fig:within_clade_shifts`'s taller one:
#: this figure is two plots on one row, so it has no reason to overhang the
#: text block. At scale 1.0 the width binds and the height follows from it,
#: so the second number is a ceiling that is not reached rather than a target.
DMS_BOX_IN: tuple[float, float] = (7.5, 8.0)
DMS_FIGURE_SCALE = 1.0

#: The type both panels were drawn at, which is `panel_charts.DMS_FONT_SIZE`
#: -- the source notebook's `PAPER_FONT_SIZE`, and its `BAR_FONT_SIZE` is set
#: equal to it. Named here as well so the panel letters follow the panels when
#: the type changes.
DMS_TITLE_PX = 15


#: `fig:dms`'s reference predictors, in the order the comparison groups them.
#: Spelled here rather than imported from `scripts/dms_models.py`, which pulls
#: pandas, Biopython and `dasm2`: this module is stdlib-plus-`lib` so that
#: `workflow/figures.smk` stays cheap to parse. `tests/test_dms_models.py` pins
#: the two together.
#:
#: The two reference floors first, then the published model, then this paper's,
#: which is the order a reader needs: what a substitution matrix and a per-site
#: conservation statistic already explain, then the model the paper claims to
#: beat, and only last the one making the claim.
#:
#: The last two are that same model read a different way, which is why they come
#: after it rather than among the other arms: `scripts/dms_prob.py` scores the
#: DASM's mutation *probability*, and the neutral rate by which that differs
#: from the selection factor. Both are defined on the single-nucleotide stratum
#: alone, so their `multi_nt` rows carry `n = 0` by declaration rather than by
#: accident -- see `dms_models.Manifest`.
DMS_PREDICTORS: tuple[str, ...] = (
    "blosum62",
    "leaf_logfreq",
    "antigen",
    "dasm",
    "dasm_prob",
    "neutral_rate",
)

#: Every predictor's correlation with the DMS, per homolog and per stratum.
#:
#: **A table first, and now also a figure.** The four stems in
#: :data:`DMS_PANELS` are the published figure's, reproduced by #27; this table
#: had no drawn home until the DASM's own column landed, which needed
#: `selection_factors.parquet`. #46 draws it as `fig:dms_strata` below, which
#: reads its `r` and `n` from here rather than recomputing them. The stem is
#: deliberately not spelled `panel_c_*`, which on `fig:dms` already means
#: something else.
DMS_PREDICTOR_CORRELATIONS = "predictor_correlations"


# ---------------------------------------------------------------------------
# `fig:dms_strata`
# ---------------------------------------------------------------------------
#
# The drawn comparison #44 deferred: two models that are defined on every
# mutation, cut by whether one nucleotide change to the founder's codon reaches
# it. A separate figure from `fig:dms` rather than more panels on it -- the two
# are over different populations, which the captions say, and sharing
# `DMS_PANELS` would move `rule dms_correlations`' outputs and
# `notebooks/dms.py`'s completeness check.

#: The two models `fig:dms_strata` draws, top row then bottom, each as
#: ``(predictor id, y axis title)``. The ids are
#: :data:`DMS_PREDICTORS` entries, so the figure cannot name a predictor the
#: table does not carry; `tests/test_dms_strata.py` pins that.
#:
#: **"fitness effect", not "log selection factor"**, which is `main.tex`'s own
#: term for this quantity and `DMS_AXIS_TITLES`' spelling of it on `fig:dms`.
#: The two are the same column; the paper's reader meets it as a fitness
#: effect, and a figure that renamed it would read as a second quantity.
#:
#: **The units still differ and the titles still say so.**
#: `scripts/dms_antigen.py` records that antiGen's column is ``log p`` while
#: the DASM's is ``log f``, and that nothing converts between them "because no
#: conversion exists that does not assume a mutation model" -- which is why the
#: rows never share a y domain. Two lines each so both titles take the same
#: width and the cells line up.
#: **antiGen first, the DASM under it**, which is the order the argument is
#: made in: the section shows the published model's probability carrying a
#: nucleotide-accessibility bias, then shows the DASM's fitness effect not
#: carrying it. A reader meets the claim and then its comparison, rather than
#: the other way round.
DMS_STRATA_MODELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("antigen", ("antiGen", "log probability")),
    ("dasm", ("DASM", "fitness effect")),  # `panel_charts.DMS_AXIS_TITLES`' DASM entry
)

#: The columns, left to right. **Not `dms.STRATA`**, which orders the two
#: halves before their union because that is how a table is read -- the rows
#: that partition, then the total. A figure is read the other way: the whole
#: population first, then what it splits into, so a reader meets the cloud
#: before being shown the cut through it. Spelled as literals for the reason
#: `DMS_PREDICTORS` is, and `tests/test_dms_strata.py` pins that it is a
#: permutation of `dms.STRATA` rather than a different set.
DMS_STRATA_COLUMNS: tuple[str, ...] = ("all", "single_nt", "multi_nt")

#: One figure per assay, each keeping the two-row shape above. The three are
#: separate figures rather than six rows of one, so the cells stay the size #46
#: drew them at and a column still reads down the page.
#:
#: **Each figure's x domain is its own assay's**, which
#: `notebooks/dms_strata.py` already gives it by computing the domain from the
#: homolog's own frame. So a column may be read down a figure and **not**
#: across figures, and the captions have to say so.


def homolog_slug(homolog: str) -> str:
    """One homolog's id as a figure stem spells it: ``KP.3.1.1`` -> ``kp311``.

    **One reason, and it is that a composed figure's stem leaves the
    pipeline**: it becomes ``figures/dms_strata_*.pdf`` in the manuscript and
    the ``\\label`` beside it. `main.tex` always writes the extension
    explicitly, which is the case `graphicx` handles best, so this is a path
    not worth having to reason about rather than a demonstrated failure.

    **Dots in a pipeline filename are fine, and nothing here says otherwise.**
    Score Parquets carry them (``antigen_KP.3.1.1.parquet``), five rules in
    `workflow/figures.smk` already take a dotted ``{homolog}`` wildcard,
    `scripts/lib.py` records that homolog names "are written into panel
    filenames and read back as join keys", and :data:`PROBABILITIES_PANELS`
    ships a dotted panel stem whose PNG, SVG and tracked CSV all exist. The
    panel stems here are slugged only so that one wildcard serves a rule whose
    other outputs are the composed figure's.

    **Not injective**, which is why :func:`homolog_for_slug` checks rather than
    assumes: ``KP.3.1.1`` and ``KP.3.11`` both come out ``kp311``, as do
    ``XBB.1.5`` and ``XBB.15``. Two such homologs would write one filename and
    the second job would overwrite the first, with the wildcard constraint
    holding the same token twice and nothing raising.
    """
    return homolog.lower().replace(".", "")


def check_slugs_are_distinct(homologs: Iterable[str]) -> dict[str, str]:
    """Map every homolog's slug back to it, refusing a roster that collides.

    Called once where `workflow/figures.smk` builds its wildcard constraint, so
    a roster whose slugs collide stops the workflow at parse time rather than
    silently drawing one assay's figure over another's.

    Returns:
        ``{slug: homolog}``, which is also what :func:`homolog_for_slug` looks
        a wildcard up in.

    Raises:
        ValueError: if two homologs share a slug.
    """
    roster = list(homologs)
    mapping: dict[str, str] = {}
    for homolog in roster:
        slug = homolog_slug(homolog)
        if slug in mapping:
            raise ValueError(
                f"{mapping[slug]!r} and {homolog!r} both slug to {slug!r}, so "
                f"they would share a figure stem and one would overwrite the "
                f"other. `homolog_slug` strips dots and case, which is not "
                f"injective; this roster needs a different rule."
            )
        mapping[slug] = homolog
    return mapping


def homolog_for_slug(slug: str, homologs: Iterable[str]) -> str:
    """The homolog a slug came from, for the rule that holds only the wildcard.

    **The roster is passed in, not looked up.** This module reads no config of
    its own -- :data:`SCATTER_HOMOLOG`'s comment records that discipline, and
    `tests/test_dms.py` is what checks that constant against `config.yaml`
    instead. Every caller already holds the roster: `workflow/figures.smk`
    imports `dms_homologs` for its other wildcard constraints.

    Raises:
        ValueError: if two homologs share this slug (via
            :func:`check_slugs_are_distinct`), or if none has it. Under the
            rules' `wildcard_constraints` the second cannot happen, so it
            guards a direct caller rather than a job.
    """
    mapping = check_slugs_are_distinct(homologs)
    if slug in mapping:
        return mapping[slug]
    known = ", ".join(f"{homolog} -> {s}" for s, homolog in mapping.items())
    raise ValueError(f"{slug!r} is no homolog's slug. The roster is {known}.")


def dms_strata_panels(slug: str) -> tuple[str, ...]:
    """One assay's two panel stems. One image per model row, three cells in it.

    `DMS_PANELS`' own precedent: each of those stems is one image of three
    scatters taking one letter. A cell per image would move the shared y domain
    and the single y axis out of the chart and into the layout, which cannot
    express either.

    Takes the **slug**, not the homolog, so `workflow/figures.smk` can call it
    with the wildcard token itself.
    """
    return tuple(
        f"panel_{letter}_{model}_strata_{slug}"
        for letter, (model, _) in zip("ab", DMS_STRATA_MODELS)
    )


def dms_strata_figure_stem(slug: str) -> str:
    """One assay's composed stem, which is its LaTeX label.

    The manuscript would include ``figures/dms_strata_kp311.pdf`` and refer to
    ``\\autoref{fig:dms_strata_kp311}``. `main.tex` references none of the three
    today, and the three captions are a manuscript change this pipeline does
    not make.
    """
    return f"dms_strata_{slug}"


def dms_strata_panel_letters(slug: str) -> dict[str, str]:
    """Which stem each letter is, reading down: one row per model."""
    return {
        chr(ord("A") + index): stem
        for index, stem in enumerate(dms_strata_panels(slug))
    }


def dms_strata_figure_rows(slug: str) -> tuple[tuple[str, ...], ...]:
    """One panel a row, stacked.

    Reading down a column answers the figure's question -- the same mutations
    under two models -- which is why the measurement is on x and shared across
    both rows of one figure.
    """
    return tuple((letter,) for letter in dms_strata_panel_letters(slug))


#: `fig:dms`'s box. Two rows of three cells each, so the width binds and there
#: is no reason to overhang the text block.
DMS_STRATA_BOX_IN: tuple[float, float] = (7.5, 8.0)

#: **Shrunk so this figure's type matches its neighbours'.** Every DMS panel is
#: drawn at :data:`panel_charts.DMS_FONT_SIZE` in chart units, so what sets the
#: size of a label on the page is how many page inches one rendered pixel gets,
#: and that follows from how many cells a row has to fit across the text block.
#: `fig:dms` packs seven cells into one row and this figure three, so at scale
#: 1.0 its labels came out at 10.9 pt against 6.4 pt there and 7.7 pt in
#: `fig:probabilities` -- the same figure, printed half again as large.
#:
#: 0.7 puts it at 7.6 pt, which is `fig:probabilities`, the figure it is read
#: beside in this section. Matching `fig:dms` exactly would mean 0.585, and a
#: 4.4-inch figure is thin for six scatters. The composition is what shrinks,
#: so `main.tex` includes the PDF at its natural size and scales nothing.
DMS_STRATA_FIGURE_SCALE = 0.7

#: The type the panels are drawn at. The same size `fig:dms` is drawn at, and
#: taken from it rather than respelled: both figures are Vega-Lite at
#: `panel_charts.DMS_FONT_SIZE`, so a change there has to move both.
DMS_STRATA_TITLE_PX = DMS_TITLE_PX


# ---------------------------------------------------------------------------
# `fig:probabilities`
# ---------------------------------------------------------------------------
#
# What a mutation probability is, and what scoring one costs. Panel A draws the
# identity `log p = log f + log lambda - log Z` as a displacement; panel B draws
# what that displacement does to agreement with an assay, beside the published
# model that also reports a probability.
#
# Both panels read what `scripts/dms_prob.py` writes, so neither needs a
# producer of its own -- which is why this figure is a separate change from the
# one that added those predictors.

#: The founder panel A's scatter is drawn on. One founder, because a scatter
#: can only show one; this one, because it is the dataset the paper leads with
#: and `fig:dms_strata` already does -- so it is :data:`SCATTER_HOMOLOG`
#: rather than a second spelling of the same choice. Upstream drew BA.1, on the
#: different criterion of where the probability's penalty is largest; the
#: identity panel A draws is exact on every founder, so leading with the
#: paper's own dataset costs the panel nothing.
PROBABILITIES_SCATTER_HOMOLOG = SCATTER_HOMOLOG

#: The three arms of panel B, in bar order: the model's own two readings, then
#: the published model's. `dasm` first because it is the reading the rest of the
#: paper uses, and `antigen` last because it is the comparison rather than the
#: subject.
PROBABILITIES_BAR_ARMS: tuple[str, ...] = ("dasm", "dasm_prob", "antigen")

#: One image per panel. Panel A is a single scatter and panel B a single row of
#: grouped bars, so unlike `dms_strata_panels` there is nothing concatenated
#: inside either.
PROBABILITIES_PANELS: tuple[str, ...] = (
    f"panel_a_probability_vs_fitness_{PROBABILITIES_SCATTER_HOMOLOG}",
    "panel_b_dms_correlation_bars",
)

#: The composed figure's stem, which is its LaTeX label: `main.tex` would
#: include `figures/probabilities.pdf` and refer to `\\autoref{fig:probabilities}`.
PROBABILITIES_FIGURE_STEM = "probabilities"

#: Which stem each letter is, reading left to right.
PROBABILITIES_PANEL_LETTERS: dict[str, str] = {
    chr(ord("A") + index): stem for index, stem in enumerate(PROBABILITIES_PANELS)
}

#: One row of two panels, side by side. The figure is a single row, so the two
#: images must be level in **height** -- which is the mirror of the width
#: levelling `dms_strata_figure_rows` needs, and is why
#: `notebooks/probabilities_figure.py` checks heights rather than widths.
PROBABILITIES_FIGURE_ROWS: tuple[tuple[str, ...], ...] = (
    tuple(PROBABILITIES_PANEL_LETTERS),
)

#: Wider than tall: one square scatter beside a short bar row.
PROBABILITIES_BOX_IN: tuple[float, float] = (7.5, 3.4)
PROBABILITIES_FIGURE_SCALE = 1.0

#: The type the panels are drawn at, taken from `fig:dms` for
#: `DMS_STRATA_TITLE_PX`'s reason: all three are Vega-Lite at
#: `panel_charts.DMS_FONT_SIZE`, so a change there has to move all of them.
PROBABILITIES_TITLE_PX = DMS_TITLE_PX

#: Where `fig:dms`'s predictor intermediates go: under the artifact root, not in
#: the tree. One narrow Parquet per `(predictor, homolog)`.
DMS_SCORE_DIR = ARTIFACT_ROOT / "dms" / "scores"

#: Where each clade's leaf residue tally goes. **Under the artifact root, not in
#: tracked ``results/``.** A 20 x 1,273 tally over a clade's GISAID leaves is a
#: derived per-site slice of the tree, which `README.md` says this repository
#: does not hold; what is committed is the correlation table built from it, which
#: carries no site and no sample.
DMS_LEAF_COUNT_DIR = ARTIFACT_ROOT / "dms" / "leaf_counts"


def dms_score_path(model: str, homolog: str) -> Path:
    """One predictor's scores for one homolog."""
    return DMS_SCORE_DIR / f"{model}_{homolog}.parquet"


def dms_leaf_count_path(clade: str) -> Path:
    """One clade's leaf residue tally."""
    return DMS_LEAF_COUNT_DIR / f"{clade}.csv"


def sample_paths_path(root: Path, clade: str) -> Path:
    """One clade's per-leaf mutation paths, inside the directory `config.yaml` names.

    Here for the reason the other path helpers are here: `workflow/figures.smk`
    declares this file as a rule's input and `scripts/dms_leaf_freq.py` opens it,
    so a rule that declared one spelling while the script read another would be a
    `MissingInputException` at best and an undeclared input at worst.
    """
    return root / f"{clade}{SAMPLE_PATHS_SUFFIX}"


#: Panel stems, as the figure's composition reads them. Named here so a rule
#: knows which notebook produces which panel.
MUTATION_PANELS = (
    "panel_a_mutation_lookup",
    "panel_b_within_clade_scatter",
    "panel_d_shift_grid",
)
LADDER_PANELS = (
    "panel_c_within_clade_ladder",
    "panel_e_shift_ladder",
)

#: `fig:subsampling`'s depth pair, panels A and B, which measure the tree and
#: take no model.
DEPTH_PANELS = (
    "panel_a_sequences_per_clade",
    "panel_b_count_ecdf",
)

#: `fig:subsampling`'s recovery pair, panels C and D, which do take a model.
LADDER_ARM_PANELS = (
    "panel_c_arm_scatter_grid",
    "panel_d_arm_ladder",
)

#: The model panels C and D draw **at each sampling depth**: the largest one
#: trained on that tree, since a dramatically thinned tree is fitted with a
#: smaller model.
#:
#: **The comparison these panels make is unaffected.** Each column sets the
#: DASM approach against the counts approach on one tree, and both sides of
#: that comparison see the same tree; the model's size varies down the
#: sampling ladder but never within a column, which is where the claim lives.
#:
#: Full depth is :data:`FIGURE_MODEL`, so panel C's baseline column is the
#: model the rest of the paper publishes and `fig:agreement` panel B draws --
#: the same clade and the same mutations, which is a comparison a reader will
#: make whether or not the caption invites it.
#:
#: The thinned arms are `dasm-85k` because that is the only architecture
#: `scripts/runs.py`'s `RUNS` trains on the cap corpora; every other one runs
#: at full depth alone. So this mapping is not free -- adding an arm here
#: means training it there.
SUBSAMPLING_FULL_DEPTH_MODEL = FIGURE_MODEL
SUBSAMPLING_THINNED_MODEL = "dasm-85k"


def subsampling_model(arm: str) -> str:
    """Which model panels C and D draw at ``arm``.

    A function rather than a dict literal so a corpus added to `config.yaml`
    is covered the moment it exists, and the full-depth name is spelled once.
    """
    return (
        SUBSAMPLING_FULL_DEPTH_MODEL
        if arm == FULL_DEPTH.name
        else SUBSAMPLING_THINNED_MODEL
    )


#: Every model the two panels load, deduplicated and in arm order. The rules
#: declare their scored inputs from this, so a model named above and never
#: trained fails at the DAG rather than inside the notebook.
SUBSAMPLING_MODELS: tuple[str, ...] = tuple(
    dict.fromkeys(subsampling_model(arm) for arm in CORPUS_NAMES)
)

#: The clade panel C scatters, at every arm. Delta, the deepest clade: the
#: panel's job is to show the clouds behind one row of panel D.
SUBSAMPLING_SCATTER_CLADE = "21J"

#: The clades panel D draws a line each for, deepest first. Ranked on sequence
#: count in `mat/clade_counts.csv` -- 4,344,573 / 2,061,093 / 1,962,904 /
#: 1,942,882 / 1,327,053 -- which is the ranking `main.tex` names. The figure
#: re-ranks on its own `n_mutations` and asserts the same five in the same
#: order, rather than trusting the two rankings to agree.
#:
#: These are also :data:`SCAN_CLADES`, which `fig:within_clade_shifts` selects
#: by a third rule. Spelled separately rather than aliased: one name meaning
#: three different selections is a name that cannot be changed for one of them.
SUBSAMPLING_LADDER_CLADES: tuple[str, ...] = ("21J", "22B", "21L", "21K", "20I")

#: The lower bound panels C and D clamp both axes to.
#:
#: **Not `CLIP_LOWER`.** The library's bound is -4.0, which is what
#: `fig:agreement` is drawn at; this figure is drawn at -6.0, and a sibling
#: constant of the library's name would be a trap for whoever read one and
#: meant the other. `tests/test_panels.py` pins the two apart.
#:
#: The clip is a control rather than a filter: it moves where a point is drawn
#: and leaves the row set alone, so `n_mutations` is the same with it and
#: without.
SUBSAMPLING_CLIP_LOWER = -6.0

#: `fig:subsampling`'s stem, which is its LaTeX label: `main.tex` includes
#: `figures/subsampling.pdf`. One spelling, so the file the manuscript
#: includes and the file this pipeline writes cannot drift.
SUBSAMPLING_FIGURE_STEM = "subsampling"

#: Which panel stem each letter is, in the caption's order. Derived nowhere:
#: a letter is a caption fact and a stem is a filename, and nothing in either
#: implies the other.
SUBSAMPLING_PANEL_LETTERS: dict[str, str] = {
    "A": DEPTH_PANELS[0],
    "B": DEPTH_PANELS[1],
    "C": LADDER_ARM_PANELS[0],
    "D": LADDER_ARM_PANELS[1],
}

#: Which panels sit on which row, left to right; a tuple inside a row is a
#: column of panels stacked within it.
#:
#: **This is the published figure's arrangement, carried across unchanged.** A
#: spans the top, because 56 clades across need the width. Below it, B sits
#: directly over C in one column, with D -- the tall one -- beside them and its
#: top level with B's. It is `notebooks/subsampling.py`'s ``ROWS`` in
#: `dasm2-experiments`, `experiments/2026-09-09-viral-dasm-figure-drafts`, the
#: notebook `main.tex`'s provenance comment names.
#:
#: The arrangement only works while the panels keep the shapes it was chosen
#: for: D tall and narrow (the five clades down the page, the two statistics
#: across) and C short and wide (the four arms across, the two approaches
#: down). `scripts/panel_charts.py` draws them that way deliberately, and
#: `tests/test_panels.py` pins the aspect each one has to keep.
SUBSAMPLING_FIGURE_ROWS: tuple[tuple[str | tuple[str, ...], ...], ...] = (
    ("A",),
    (("B", "C"), "D"),
)

#: The box this figure has to fit inside, in inches, and the fraction of it to
#: use -- `fig:agreement`'s box and the published figure's own scale, so the
#: two sit at one size in the manuscript. Nine tenths of the box: the panels'
#: type is set larger on the page than the other figures' (see
#: :data:`SUBSAMPLING_TITLE_PX`), and the figure gives back some width so the
#: type-to-figure ratio lands where the manuscript wants it.
SUBSAMPLING_BOX_IN: tuple[float, float] = (7.5, 8.0)
SUBSAMPLING_FIGURE_SCALE = 0.9

#: What this figure's panel titles and axis labels measure in the Vega-Lite
#: spec. The panel letters are sized against the title; see `figure_layout`'s
#: header for the conversion.
#:
#: **This is the one figure not drawn at `PANEL_FONT_SIZE`**, and it is drawn
#: larger for a reason that belongs to its layout rather than to its taste.
#: `SUBSAMPLING_FIGURE_ROWS` puts two panels across the lower row, so about
#: 1,580 Vega-Lite units of panel have to fit the page width, against roughly
#: 1,350 for `fig:within_clade_shifts`' widest row and one panel for each row
#: of a stacked figure. Type is sized in those same units, so the same spec
#: size lands smaller on the page here than anywhere else: at
#: `PANEL_FONT_SIZE` these titles compose to about 5 pt against the paper's
#: 6.6 to 7.2 pt. The published figure's two sources both drew it at 24 px for
#: this reason, and these are their sizes.
SUBSAMPLING_TITLE_PX = 24
SUBSAMPLING_LABEL_PX = 22

#: `fig:within_clade_shifts`'s stems, one tuple per producing notebook.
#: One stem per clade, not one image of both. The composed figure puts each
#: clade's panel A beside that clade's row of panel B, and a single two-panel
#: image cannot be split across two rows of the page.
OUTLIER_PANELS = tuple(f"panel_a_outliers_{clade}" for clade in CANDIDATE_CLADES)

#: Which side of each clade's panel its labels are drawn on. The candidates
#: are outliers below the diagonal, so they sit to the lower right of the
#: cloud: in 22D that is the cloud's right-hand end, where the gutter past the
#: right edge is the nearest empty space, but in 20B they sit high above the
#: middle of the diagonal, where reaching the right edge means a leader drawn
#: across the whole cloud and the space immediately to their left is empty.
#: 21K is labelled left for the same reason as 20B, and for one more: its
#: subject M177V sits mid-cloud horizontally while its background mutation
#: R346K sits at the cloud's right-hand end, so exactly one of the two gets a
#: leader drawn across the panel whichever gutter is used. Left gives the short
#: one to the mutation the row is about.
CANDIDATE_LABEL_SIDES = {"22D": "right", "20B": "left", "21K": "left"}

#: Table S1's cutoff and its cut. The ranking is the agreement comparison at a
#: higher count floor: `main.tex` describes it as the mutations whose actual or
#: expected count in the clade reached 50, ordered by the absolute difference
#: between the two approaches, and printed down to a difference of 2.
#:
#: **Deliberately not `PANEL_MIN_COUNT`.** The panels' cutoff is 10 and this is
#: 50, and the two answer different questions: a panel asks what the comparison
#: looks like, a ranking of disagreements asks which rows are well enough
#: measured to be worth explaining.
OUTLIER_MIN_COUNT = 50.0
OUTLIER_MIN_DIFFERENCE = 2.0

#: Where the ranking lands. Not a panel -- it is a table in the supplement --
#: but it is written through the same `write_panel_csv` choke point, so it
#: carries the filter spec in its columns like every other tracked table here.
OUTLIER_TABLE_STEM = "table_s1_outlier_ranking"
#: `fig:within_clade_shifts`' right-hand column: one stem per clade, paired
#: with that clade's `OUTLIER_PANELS` stem. One image per clade rather than
#: one of all three, for the reason `OUTLIER_PANELS` gives -- the composed
#: figure puts each clade's pair on its own row, and a single image cannot be
#: split across rows.
BACKGROUND_PANELS = tuple(f"panel_b_backgrounds_{clade}" for clade in CANDIDATE_CLADES)

#: `fig:within_clade_shifts`' stem, which is its LaTeX label: `main.tex`
#: includes `figures/within_clade_shifts.pdf`. One spelling, so the file the
#: manuscript includes and the file this pipeline writes cannot drift.
SHIFT_FIGURE_STEM = "within_clade_shifts"

#: Which panels sit on which row, left to right. Three rows of two: each
#: clade's scatter beside that clade's background row. The keys are the
#: letters drawn on the figure, which is also the order the caption reads.
#:
#: This is why both halves are written one image per clade rather than one
#: image of all three -- a single stacked image cannot be split across rows.
SHIFT_FIGURE_ROWS: tuple[tuple[str, ...], ...] = (("A", "B"), ("C", "D"), ("E", "F"))

#: Which panel stem each letter is. Derived nowhere: a letter is a caption
#: fact and a stem is a filename, and nothing in either implies the other.
#: Built from :data:`CANDIDATE_CLADES` so the row order is the figure's one
#: spelling of it, and a clade added there cannot be given a scatter and no
#: background row.
SHIFT_PANEL_LETTERS: dict[str, str] = {
    letter: stem
    for clade_index, _clade in enumerate(CANDIDATE_CLADES)
    for letter, stem in (
        ("ACE"[clade_index], OUTLIER_PANELS[clade_index]),
        ("BDF"[clade_index], BACKGROUND_PANELS[clade_index]),
    )
}

#: The box this figure has to fit inside, in inches. Taller than
#: `fig:agreement`'s: three rows rather than two, and each row carries a
#: scatter whose square aspect sets the row height.
SHIFT_BOX_IN: tuple[float, float] = (7.5, 9.0)
SHIFT_FIGURE_SCALE = 1.0

#: The columns the panels read. Named rather than taken whole: the table is
#: 1,425,760 rows per run over 24 columns, and the pooled `*_all` half,
#: `uncertainty` and `censored_directional` are read by no panel here.
#: `delta_fitness_bayesian` is, barely -- it is the other member of
#: :data:`FITNESS_METRICS`, so a spec that names it must find it.
TABLE_COLUMNS: tuple[str, ...] = (
    "clade",
    "model",
    "arm",
    "aa_mutation",
    "aa_site",
    "parent_aa",
    "mutant_aa",
    "is_synonymous",
    "log_selection_factor",
    "actual_count",
    "predicted_count",
    "delta_fitness_ratio",
    "delta_fitness_bayesian",
    "globally_masked",
    "censored",
)


#: The composed figure's stem, which is its LaTeX label: `main.tex` includes
#: `figures/agreement.pdf` and refers to `\\autoref{fig:agreement}`. One
#: spelling, so the file the manuscript includes and the file this pipeline
#: writes cannot drift apart.
FIGURE_STEM = "agreement"

#: What the composition writes beside the panels, in `save_figure`'s order.
#: The PDF is what the manuscript includes; the PNG is what the HTML export
#: embeds; the SVG is what a journal asks for.
COMPOSED_FORMATS: tuple[str, ...] = ("png", "pdf", "svg")

#: Which panels sit on which row of the composed figure, left to right. Two
#: rows: the lookup and the within-clade pair on top, the shift grid and its
#: ladder beneath. Rows are left justified and top aligned, so row 2 being
#: narrower leaves whitespace on the right rather than being stretched.
#:
#: The keys are the letters drawn on the figure, which is also the order the
#: caption reads them in.
FIGURE_ROWS: tuple[tuple[str, ...], ...] = (("A", "B", "C"), ("D", "E"))

#: Which panel stem each letter is. Derived nowhere: a letter is a caption
#: fact and a stem is a filename, and nothing in either implies the other.
PANEL_LETTERS: dict[str, str] = {
    "A": "panel_a_mutation_lookup",
    "B": "panel_b_within_clade_scatter",
    "C": "panel_c_within_clade_ladder",
    "D": "panel_d_shift_grid",
    "E": "panel_e_shift_ladder",
}

#: The box the composed figure has to fit inside, in inches, and the fraction
#: of it to use. `FIGURE_SCALE` shrinks the box and every margin and gap
#: alike, so the composition stays in proportion; the figure fits the box
#: exactly in one dimension, so this is the only way to make it smaller on the
#: page.
BOX_IN: tuple[float, float] = (7.5, 8.0)
FIGURE_SCALE = 0.9

#: What one panel's title measures in the Vega-Lite spec, which is what the
#: panel letters are sized against. It is `scripts/panel_charts.py`'s
#: `PANEL_FONT_SIZE`, and `tests/test_panels.py` pins the two together: a
#: letter sized against a title the panels no longer draw is a letter that
#: drifts every time the type does.
PANEL_TITLE_PX = 18


def composed_path(selection: str, extension: str, stem: str = FIGURE_STEM) -> Path:
    """Where one selection's composed figure goes, in one format.

    ``stem`` defaults to `fig:agreement`'s, which is what every existing
    caller means; `fig:within_clade_shifts` passes its own.
    """
    return figure_dir(selection) / f"{stem}.{extension}"


def composed_paths(selection: str, stem: str = FIGURE_STEM) -> list[Path]:
    """One selection's composed figure, in every format it is written in."""
    return [composed_path(selection, ext, stem) for ext in COMPOSED_FORMATS]


def tex_figure_dir(selection: str) -> Path | None:
    """Where the composed PDF is copied for the manuscript, if anywhere.

    Re-exported from `lib` so a notebook and `workflow/figures.smk` read this
    module alone for every path in this figure.

    **Only a published selection is copied.** The two agreement selections
    compose to the same `FIGURE_STEM`, so a `fidelity` composition would land
    on the manuscript's `agreement.pdf` and replace the paper's figure with a
    panel set drawn from two other models -- same size, same layout, different
    numbers, and nothing downstream to notice. `snakemake figures` builds both,
    which is how that became reachable.

    `subsampling` is not a model selection at all but this figure's own
    directory, with one panel set and a stem of its own, so there is no second
    composition for it to be confused with and it is copied.
    """
    if selection not in PUBLISHED_SELECTIONS:
        return None
    return TEX_FIGURE_DIR


def selection_models(selection: str) -> tuple[str, ...]:
    """The models one selection draws, named so a typo fails here."""
    if selection not in SELECTIONS:
        raise KeyError(
            f"{selection!r} is not a panel selection: {sorted(SELECTIONS)}. "
            f"`figure` is what the paper publishes and `fidelity` is the pair "
            f"the published CSVs were drawn from."
        )
    return SELECTIONS[selection]


def figure_dir(selection: str) -> Path:
    """Where one selection's panel images go."""
    return FIGURE_DIR / selection


def panel_csv_path(selection: str, stem: str) -> Path:
    """Where one selection's panel table goes, in the tracked tree."""
    return PANEL_CSV_DIR / selection / f"{stem}.csv"


def scores_flags(paths: Iterable[str | Path]) -> str:
    """``--scores`` once per path, which is the only form marimo reads.

    ``mo.cli_args().get_all("scores")`` returns one entry per *occurrence of
    the flag*, so a notebook given ``--scores a b c`` sees a single entry,
    ``"a b c"``, and opens a file named after all three at once. Snakemake
    expands ``{input.scores}`` to exactly that, space-separated. It is a
    parsing bug rather than a quoting one, so no amount of escaping rescues
    the single-flag form; the flag has to be repeated. Verified against
    marimo 0.23.15, where ``--scores p1 p2 p3`` yields ``['p1 p2 p3']`` and
    ``--scores p1 --scores p2 --scores p3`` yields ``['p1', 'p2', 'p3']``.

    Each path is quoted with `shlex.quote`, because Snakemake splices
    ``params`` into the shell string verbatim. That is a complete guard where
    refusing paths that hold a space would be a partial one: a path can also
    hold ``;``, ``$`` or a glob character, and quoting handles all of them
    while leaving an ordinary path untouched.

    Here rather than in `workflow/figures.smk` so that it can be tested: a
    `.smk` is not importable, and `tests/test_rule_inputs.py` cannot see this
    because the inputs are *declared* correctly and only their spelling on the
    command line was wrong.

    Raises:
        ValueError: On an empty path, or on no paths at all. Both callers pass
            every predictor's scores for every homolog, so neither can happen
            without a bug upstream -- and a notebook handed no scores fails
            later, further from the cause.
    """
    flags = []
    for path in paths:
        text = str(path)
        if not text:
            raise ValueError("a score path is the empty string, which names no file")
        flags.append(f"--scores {shlex.quote(text)}")
    if not flags:
        raise ValueError(
            "no score paths to pass, so the notebook would be handed none and "
            "fail on an empty table instead of here"
        )
    return " ".join(flags)
