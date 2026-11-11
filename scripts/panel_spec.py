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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import (
    ARTIFACT_ROOT,
    CORPUS_NAMES,
    FITNESS_SUFFIX,
    FULL_DEPTH,
    REPO_ROOT,
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

#: Which panel directories hold a composition the manuscript takes. See
#: :func:`tex_figure_dir`, which is where the distinction bites: `fidelity`
#: is the one directory that composes a figure the paper does not publish.
PUBLISHED_SELECTIONS: frozenset[str] = frozenset(
    {DEFAULT_SELECTION, SUBSAMPLING_SELECTION}
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

#: The directory `fig:dms`'s panels are written under -- a **directory name,
#: not a model selection**, and absent from :data:`SELECTIONS` for the same
#: reason :data:`SUBSAMPLING_SELECTION` is. `tests/test_panels.py` pins it out.
DMS_SELECTION = "dms"

#: The homolog whose scatter is panel A. The paper draws one homolog's cloud
#: and leaves the other two to panel B's summary, so the choice is the
#: figure's rather than this pipeline's: `main.tex`'s caption names KP.3.1.1.
#: `tests/test_dms.py` requires it to be one of `config.yaml`'s homologs, since
#: this module stays stdlib-and-`lib` only and does no config lookup of its own.
SCATTER_HOMOLOG = "KP.3.1.1"

#: `fig:dms`'s stems, one tuple per producing notebook. Panel C -- the
#: comparison against published deep-learning models -- is not here: it needs
#: antiGen and CoVFit, and it is added when the rule that draws it exists.
DMS_CORRELATION_PANELS = (
    "panel_a_homolog_scatter",
    "panel_b_homolog_correlations",
)
DMS_PROBABILITY_PANELS = ("panel_d_fitness_vs_probability",)


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
