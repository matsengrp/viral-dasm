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

from lib import ARTIFACT_ROOT, REPO_ROOT, TEX_FIGURE_DIR

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


def composed_path(selection: str, extension: str) -> Path:
    """Where one selection's composed figure goes, in one format."""
    return figure_dir(selection) / f"{FIGURE_STEM}.{extension}"


def composed_paths(selection: str) -> list[Path]:
    """One selection's composed figure, in every format it is written in."""
    return [composed_path(selection, ext) for ext in COMPOSED_FORMATS]


def tex_figure_dir(selection: str) -> Path | None:
    """Where the composed PDF is copied for the manuscript, if anywhere.

    Re-exported from `lib` so a notebook and `workflow/figures.smk` read this
    module alone for every path in this figure.

    **Only the published selection is copied.** Every selection composes to
    `FIGURE_STEM`, so a `fidelity` composition would land on the manuscript's
    `agreement.pdf` and replace the paper's figure with a panel set drawn from
    two other models -- same size, same layout, different numbers, and nothing
    downstream to notice. `snakemake figures` builds both, which is how that
    became reachable.
    """
    if selection != DEFAULT_SELECTION:
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
