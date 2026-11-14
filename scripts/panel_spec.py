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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import ARTIFACT_ROOT, REPO_ROOT, TEX_FIGURE_DIR

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

#: The clades that figure's panels A and B cover: the two the candidate list
#: names. Separate from :data:`SHIFT_CLADES`, which is `fig:agreement`'s
#: cross-clade pairing and whose consumers would change meaning if these were
#: added to it.
#:
#: In neither clade do the two heaviest haplotypes differ at a single residue,
#: so a shift in either is attributable to the haplotype rather than to one
#: substitution. That is why the scan's row excludes these two and the
#: candidate panels cover them instead.
CANDIDATE_CLADES: tuple[str, ...] = ("22D", "20B")

#: The clades the scan's panel row draws, in the order the row reads. The scan
#: axis is ten clades and the readout seven; these five are the seven minus
#: :data:`CANDIDATE_CLADES`, which panels A and B already cover.
SCAN_CLADES: tuple[str, ...] = ("21J", "22B", "21L", "21K", "20I")

#: The rung the scan's panel is drawn at. One rung, not a sweep: the saved
#: panel is written only at the lowest of the scan's three cutoffs, because
#: letting another setting overwrite it is how a composed figure comes to
#: disagree with its own caption.
#:
#: **Deliberately not tied to `PANEL_MIN_COUNT` or `COUNT_LADDER`, which hold
#: the same number.** This counts a cell's pooled *events* across two
#: backgrounds of one clade; those count a mutation's actual-or-expected
#: occurrences in a clade. Different numerators over different denominators,
#: equal at 10 today, with no reason to move together. It is also not the
#: corroboration gate's own 20-event floor.
SCAN_MIN_EVIDENCE = 10

#: The candidate list panels A and B share, as tracked data rather than two
#: hand-maintained lists. Moving a candidate off the figures is one edit that
#: reaches both panels.
CANDIDATE_CSV = REPO_ROOT / "data" / "shift_candidates.csv"

#: What a candidate's `status` may be. `figure` reaches the panels; `null` is
#: the designed null, measured and deliberately not drawn; `held_back` is
#: measured and off the figure for some other reason the file records. The two
#: non-figure reasons stay distinct rather than collapsing to "excluded".
CANDIDATE_ON_FIGURE = "figure"
CANDIDATE_STATUSES = (CANDIDATE_ON_FIGURE, "null", "held_back")

#: The columns :func:`_candidate_rows` requires, found by name.
CANDIDATE_COLUMNS = ("clade", "mutation", "status")

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
CANDIDATE_LABEL_SIDES = {"22D": "right", "20B": "left"}

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
BACKGROUND_PANELS = ("panel_b_background_effects",)
SCAN_PANELS = ("panel_c_clade_row",)

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
