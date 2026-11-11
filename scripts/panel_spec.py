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

from lib import (
    ARTIFACT_ROOT,
    CORPUS_NAMES,
    FULL_DEPTH,
    GENE,
    NTMUT_SUFFIX,
    REPO_ROOT,
    cap_of,
)

#: Where the panel images go: under the artifact root, not in the tree. A panel
#: is a rebuildable picture of the CSV committed beside it, and `README.md`'s
#: layout gives tracked `results/` to the tables.
FIGURE_DIR = ARTIFACT_ROOT / "figures"

#: Where the tables go: tracked, one CSV per panel, holding exactly what that
#: panel plots.
PANEL_CSV_DIR = REPO_ROOT / "results" / "panels"

#: The censor audit, which is the only source of the reverting exclusion. The
#: table carries `censored` and `globally_masked` as columns and does not carry
#: this layer, and `censored_directional` is not it.
AUDIT_JSON = ARTIFACT_ROOT / "audit" / f"{GENE}.json"

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
#: drawn from, which is how the port is shown to compute what the old code
#: computed. Both come out of one table and one `groupby`, so the second costs
#: no rescoring.
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


def ntmut_grid_path(root: Path, clade: str, arm: str) -> Path:
    """One clade's nucleotide grid at one sampling depth.

    Full depth is the **top level** of `ntmut_fitness_dir` and each thinned arm
    is a subdirectory named for its corpus. Here rather than in the two callers
    because `workflow/figures.smk` declares these files as `rule
    subsampling_depth`'s inputs and `scripts/panels.py` opens them: a rule that
    declared one layout while the notebook read another would be a
    `MissingInputException` at best and a silently undeclared input at worst.
    """
    directory = root if arm == FULL_DEPTH.name else root / arm
    return directory / f"{clade}{NTMUT_SUFFIX}"


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
#: take no model. Panels C and D are added here when the rule that draws them
#: exists.
DEPTH_PANELS = (
    "panel_a_sequences_per_clade",
    "panel_b_count_ecdf",
)

#: `fig:within_clade_shifts`'s stems, one tuple per producing notebook.
OUTLIER_PANELS = ("panel_a_candidate_scatter",)
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
