"""`fig:agreement`'s five panels, drawn from the one scored table.

Two notebooks, five panels, five tables. Nothing here infers and nothing here
takes a GPU: ``rule score`` did the forward passes, so a panel rule reads
``selection_factors.parquet``, filters it, correlates it and draws.

The DAG:

    scored ---> agreement_mutations  (panels A, B, D)
            \\-> agreement_ladders    (panels C, E)

**The two selections.** Each notebook is run per ``{selection}``, which names
which models it draws: ``figure`` is ``dasm-1077k``, what the paper publishes,
and ``fidelity`` is ``dasm-293k`` and ``dasm-85k``, the pair the published CSVs
were drawn from. Both come out of one table and one ``groupby``, so the second
costs no rescoring. ``scripts/panels.py`` owns the mapping.

**The panels are declared outputs, not side effects.** ``marimo export html``
embeds a failed cell in the HTML and exits 0, so a rule whose only output was
the export would succeed with no panel written. Naming every PNG and CSV is
what makes a raised cell fail the rule.

The images go under the artifact root and the tables into tracked
``results/panels/``: a panel is a rebuildable picture of the CSV beside it, and
`README.md`'s layout gives tracked ``results/`` to the tables. Counts, fitness
effects and selection factors are publishable; no panel carries a sequence.
"""

# `re` is imported here rather than taken from the Snakefile's namespace: this
# file is included, so relying on an includer's import makes it depend on include
# order for something it uses in a wildcard constraint.
import re

# Every path below comes from `scripts/panel_spec.py`, which is what the
# notebooks compute them from too: one spelling per fact, so a rule cannot
# declare an output the notebook writes somewhere else. The spec module is
# stdlib plus `lib`, deliberately: this file is parsed on every snakemake
# invocation, and importing the panel code here would pull `dasm2`, pandas and
# pyarrow into a dry run to learn five file names.
from panel_spec import (
    AUDIT_JSON,
    DEFAULT_SELECTION,
    DEPTH_PANELS,
    DMS_PREDICTOR_CORRELATIONS,
    DMS_PREDICTOR_PANELS,
    DMS_PREDICTORS,
    DMS_SELECTION,
    ECDF_CLADES,
    LADDER_PANELS,
    MUTATION_PANELS,
    PANEL_MEM_MB,
    SELECTIONS,
    SHIFT_CLADES,
    SUBSAMPLING_SELECTION,
    dms_leaf_count_path,
    dms_score_path,
    figure_dir,
    ntmut_grid_path,
    panel_csv_path,
    sample_paths_path,
)

NOTEBOOK_DIR = f"{ARTIFACT_ROOT}/notebooks"


def shift_clade_fastas(*_):
    """The founder FASTAs of the clades the shift panels pair."""
    directory = mat_pcp_path("clade_founders_dir")
    return [str(directory / f"{clade}.fa") for clade in SHIFT_CLADES]


def panel_outputs(stems, selection):
    """One selection's panel images and tables, as a rule's outputs.

    Both image formats are declared, not just the PNG. `save_panel` writes a
    vector copy of every panel, and an undeclared output is one snakemake will
    not clean up after a failed job -- so a half-written SVG would survive a
    rerun that replaced its PNG.
    """
    directory = figure_dir(selection)
    return [
        str(directory / f"{stem}.{extension}")
        for stem in stems
        for extension in ("png", "svg")
    ] + [str(panel_csv_path(selection, stem)) for stem in stems]


def ecdf_grid_files(*_):
    """The nucleotide grids panel B's curves are drawn from.

    Eight files, not the directory: two clades at four sampling depths, where
    full depth is the directory's own level and each cap arm is a subdirectory
    named for its corpus. Declaring the contents is how a rule declares a
    directory `config.yaml` names, as `rule score`'s `founder_fastas` does.
    """
    root = mat_pcp_path("ntmut_fitness_dir")
    return [
        str(ntmut_grid_path(root, clade, arm))
        for clade in ECDF_CLADES
        for arm in CORPUS_NAMES
    ]


rule panels:
    """Every panel of `fig:agreement`, for the model the paper publishes."""
    input:
        panel_outputs(MUTATION_PANELS, DEFAULT_SELECTION),
        panel_outputs(LADDER_PANELS, DEFAULT_SELECTION),


rule subsampling:
    """`fig:subsampling`'s panels, as far as this stage of the port builds them.

    Panels A and B only. C and D need `rule score`'s table for `dasm-85k` on
    all four corpora, so they are a separate rule once those runs land -- and a
    target that demanded them now would be red for the whole gap.
    """
    input:
        panel_outputs(DEPTH_PANELS, SUBSAMPLING_SELECTION),


rule subsampling_depth:
    """Panels A and B: the sampling imbalance, and what it costs the counts."""
    input:
        panels="scripts/panels.py",
        spec="scripts/panel_spec.py",
        charts="scripts/panel_charts.py",
        lib="scripts/lib.py",
        config="config.yaml",
        # What the notebook reaches through those modules. Spelled out per rule
        # rather than shared through a dict, for the reason the agreement rules
        # spell theirs out: `tests/test_rule_inputs.py` reads these blocks as
        # source, and an input expanded from a variable is one it cannot see.
        analysis_charts="dasm2/analysis/charts.py",
        analysis_viral="dasm2/analysis/viral.py",
        analysis_eval="dasm2/analysis/viral_fitness_eval.py",
        io="dasm2/io.py",
        conversion="dasm2/sequence/conversion.py",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        dasm2_init="dasm2/__init__.py",
        notebook="notebooks/subsampling_depth.py",
        # The two mat-pcp tables these panels measure. No scored table and no
        # audit: neither panel reads a model.
        clade_counts=mat_pcp_path("clade_counts_file"),
        nt_grids=ecdf_grid_files,
        # The roster panel A plots, which `clade_size_frame` drops the counts
        # table's two extra clades against.
        clade_founders=founder_fastas,
        founder_reader="scripts/founders.py",
    output:
        html=f"{NOTEBOOK_DIR}/subsampling_depth.html",
        panels=panel_outputs(DEPTH_PANELS, SUBSAMPLING_SELECTION),
    log:
        f"{ARTIFACT_ROOT}/logs/subsampling_depth.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " > {log} 2>&1"


rule agreement_mutations:
    """Panels A, B and D: one mutation, one clade's cloud, and the shift grid."""
    wildcard_constraints:
        selection="|".join(SELECTIONS),
    input:
        panels="scripts/panels.py",
        spec="scripts/panel_spec.py",
        charts="scripts/panel_charts.py",
        founder_reader="scripts/founders.py",
        lib="scripts/lib.py",
        config="config.yaml",
        # What the notebook reaches through those two modules. Spelled out, and
        # spelled out on both rules rather than shared through a dict, because
        # `tests/test_rule_inputs.py` reads these blocks as source: an input
        # expanded from a variable is an input the check cannot see.
        analysis_charts="dasm2/analysis/charts.py",
        analysis_viral="dasm2/analysis/viral.py",
        analysis_eval="dasm2/analysis/viral_fitness_eval.py",
        io="dasm2/io.py",
        conversion="dasm2/sequence/conversion.py",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        dasm2_init="dasm2/__init__.py",
        notebook="notebooks/agreement_mutations.py",
        table=SCORED_TABLE,
        audit=AUDIT_JSON,
        # The founders the shift panel drops identical pairs on -- those four
        # clades', not the roster's 56. `rule score` declares the whole glob
        # because it scores every clade; this reads `SHIFT_CLADES` alone.
        clade_founders=shift_clade_fastas,
    output:
        html=f"{NOTEBOOK_DIR}/agreement_mutations_{{selection}}.html",
        panels=panel_outputs(MUTATION_PANELS, "{selection}"),
    log:
        f"{ARTIFACT_ROOT}/logs/agreement_mutations_{{selection}}.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " -- --table {input.table} --audit {input.audit}"
        " --selection {wildcards.selection}"
        " > {log} 2>&1"


rule agreement_ladders:
    """Panels C and E: both correlations against the count cutoff."""
    wildcard_constraints:
        selection="|".join(SELECTIONS),
    input:
        panels="scripts/panels.py",
        spec="scripts/panel_spec.py",
        charts="scripts/panel_charts.py",
        founder_reader="scripts/founders.py",
        lib="scripts/lib.py",
        config="config.yaml",
        # What the notebook reaches through those two modules. Spelled out, and
        # spelled out on both rules rather than shared through a dict, because
        # `tests/test_rule_inputs.py` reads these blocks as source: an input
        # expanded from a variable is an input the check cannot see.
        analysis_charts="dasm2/analysis/charts.py",
        analysis_viral="dasm2/analysis/viral.py",
        analysis_eval="dasm2/analysis/viral_fitness_eval.py",
        io="dasm2/io.py",
        conversion="dasm2/sequence/conversion.py",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        dasm2_init="dasm2/__init__.py",
        notebook="notebooks/agreement_ladders.py",
        table=SCORED_TABLE,
        audit=AUDIT_JSON,
        clade_founders=shift_clade_fastas,
    output:
        html=f"{NOTEBOOK_DIR}/agreement_ladders_{{selection}}.html",
        panels=panel_outputs(LADDER_PANELS, "{selection}"),
    log:
        f"{ARTIFACT_ROOT}/logs/agreement_ladders_{{selection}}.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " -- --table {input.table} --audit {input.audit}"
        " --selection {wildcards.selection}"
        " > {log} 2>&1"


# ---------------------------------------------------------------------------
# `fig:dms` panel C: the predictors that are not the DASM
# ---------------------------------------------------------------------------
#
# Three rules rather than two. The leaf tally is the expensive one -- clade 21K
# is 1.9M leaves and 1.7 GB of paths -- and it does not depend on BLOSUM62, so
# folding both predictors into one rule would rebuild it on every edit to the
# substitution matrix.
#
# Scores and tallies go under the artifact root; only the correlation table is
# tracked. `panel_spec.DMS_LEAF_COUNT_DIR` records why.

#: The clades the three homologs are matched to, each read from `config.yaml`
#: rather than listed here, so a repointed homolog moves the clade with it.
DMS_CLADES = [dms_dataset(homolog).clade for homolog in dms_homologs()]


def dms_homolog_counts(wildcards):
    """The leaf tally one homolog's scores are read out of."""
    return str(dms_leaf_count_path(dms_dataset(wildcards.homolog).clade))


def dms_all_scores(*_):
    """Every predictor's scores for every homolog."""
    return [
        str(dms_score_path(model, homolog))
        for model in DMS_PREDICTORS
        for homolog in dms_homologs()
    ]


def dms_all_counts(*_):
    """Every matched clade's leaf tally."""
    return [str(dms_leaf_count_path(clade)) for clade in DMS_CLADES]


rule dms:
    """`fig:dms`'s predictor table, as far as this stage of the port builds it.

    Panel C's two reference predictors and the join they land through. The DASM's
    own column is a filter on `selection_factors.parquet`, so it arrives with
    `rule score`'s table; the chart waits on it, and a target that demanded it now
    would be red for the whole gap.
    """
    input:
        [
            str(figure_dir(DMS_SELECTION) / f"{stem}.{extension}")
            for stem in DMS_PREDICTOR_PANELS
            for extension in ("png", "svg")
        ],


rule dms_leaf_counts:
    """One clade's per-site residue tally over its own tree leaves.

    Every leaf's spike is reconstructed from its own root-to-tip mutation path
    applied to 19A's coding sequence, then translated and tallied. Clade 21K is
    1.9M leaves; the three together read about 2.6 GB.
    """
    wildcard_constraints:
        clade="|".join(DMS_CLADES),
    input:
        script="scripts/dms_leaf_freq.py",
        dms="scripts/dms.py",
        spec="scripts/panel_spec.py",
        founder_reader="scripts/founders.py",
        lib="scripts/lib.py",
        config="config.yaml",
        # What the entry point imports. Spelled out per rule rather than shared
        # through a dict, for the reason the other figure rules spell theirs out:
        # `tests/test_rule_inputs.py` reads these blocks as source, and an input
        # expanded from a variable is one it cannot see.
        scov2_rates="dasm2/neutral/scov2_rates.py",
        constants="dasm2/sequence/constants.py",
        conversion="dasm2/sequence/conversion.py",
        dasm2_init="dasm2/__init__.py",
        # The per-leaf paths this reconstructs from, and the founders it reads:
        # 19A for the starting state, the clade's own for the readback. The whole
        # glob, as `rule score` declares it -- it already covers both, and a
        # two-file list beside it would be a second way of naming founders.
        paths=lambda w: str(
            sample_paths_path(mat_pcp_path("sample_paths_dir"), w.clade)
        ),
        clade_founders=founder_fastas,
        # Read through `dms.globally_masked_sites()` for the founder-agreement
        # line this rule prints. Declared although it changes no count, because
        # an undeclared read is the defect `tests/test_rule_inputs.py` exists for.
        masked=mat_pcp_path("masked_nt_sites_file"),
    output:
        counts=str(dms_leaf_count_path("{clade}")),
    log:
        f"{ARTIFACT_ROOT}/logs/dms_leaf_counts_{{clade}}.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "python {input.script}"
        " --clade {wildcards.clade}"
        " --output {output.counts}"
        " > {log} 2>&1"


rule dms_blosum_scores:
    """BLOSUM62's score for every cell one homolog's assay measured.

    Declares no leaf tally: it reads none, and a rule that declared one would
    make every BLOSUM62 score wait on 2.6 GB of path files it never opens.
    """
    wildcard_constraints:
        homolog="|".join(re.escape(homolog) for homolog in dms_homologs()),
    input:
        script="scripts/dms_blosum.py",
        models="scripts/dms_models.py",
        dms="scripts/dms.py",
        published="scripts/dms_published.py",
        founder_reader="scripts/founders.py",
        lib="scripts/lib.py",
        config="config.yaml",
        constants="dasm2/sequence/constants.py",
        parquet_backend="dasm2/data/parquet_backend.py",
        dasm2_init="dasm2/__init__.py",
        # The assay itself, and the clade founder every score is taken away from.
        # `scripts/dms.py` gates the CSV on the md5 `config.yaml` pins, so a
        # repointed homolog fails there rather than moving a correlation quietly.
        dms_csv=lambda w: str(dms_dataset(w.homolog).path),
        clade_founders=founder_fastas,
        censored=mat_pcp_path("censored_mutations_file"),
        clades=mat_pcp_path("censored_clades_file"),
        masked=mat_pcp_path("masked_nt_sites_file"),
    output:
        scores=str(dms_score_path("blosum62", "{homolog}")),
    log:
        f"{ARTIFACT_ROOT}/logs/dms_scores_blosum62_{{homolog}}.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "python {input.script}"
        " --homolog {wildcards.homolog}"
        " --output {output.scores}"
        " > {log} 2>&1"


rule dms_leaf_scores:
    """The leaf log frequency ratio for every cell one homolog's assay measured."""
    wildcard_constraints:
        homolog="|".join(re.escape(homolog) for homolog in dms_homologs()),
    input:
        script="scripts/dms_leaf_scores.py",
        models="scripts/dms_models.py",
        leaf_freq="scripts/dms_leaf_freq.py",
        dms="scripts/dms.py",
        published="scripts/dms_published.py",
        founder_reader="scripts/founders.py",
        lib="scripts/lib.py",
        config="config.yaml",
        constants="dasm2/sequence/constants.py",
        parquet_backend="dasm2/data/parquet_backend.py",
        dasm2_init="dasm2/__init__.py",
        dms_csv=lambda w: str(dms_dataset(w.homolog).path),
        clade_founders=founder_fastas,
        censored=mat_pcp_path("censored_mutations_file"),
        clades=mat_pcp_path("censored_clades_file"),
        masked=mat_pcp_path("masked_nt_sites_file"),
        # The tally this reads out, for the clade the homolog is matched to.
        counts=dms_homolog_counts,
    output:
        scores=str(dms_score_path("leaf_logfreq", "{homolog}")),
    log:
        f"{ARTIFACT_ROOT}/logs/dms_scores_leaf_logfreq_{{homolog}}.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "python {input.script}"
        " --homolog {wildcards.homolog}"
        " --counts {input.counts}"
        " --output {output.scores}"
        " > {log} 2>&1"


rule dms_predictor_summary:
    """Panel C's table: each predictor against the DMS, per homolog and stratum."""
    input:
        script="scripts/dms_summary.py",
        models="scripts/dms_models.py",
        leaf_freq="scripts/dms_leaf_freq.py",
        dms="scripts/dms.py",
        published="scripts/dms_published.py",
        spec="scripts/panel_spec.py",
        founder_reader="scripts/founders.py",
        lib="scripts/lib.py",
        config="config.yaml",
        analysis_eval="dasm2/analysis/viral_fitness_eval.py",
        constants="dasm2/sequence/constants.py",
        parquet_backend="dasm2/data/parquet_backend.py",
        dasm2_init="dasm2/__init__.py",
        scores=dms_all_scores,
        counts=dms_all_counts,
    output:
        table=str(panel_csv_path(DMS_SELECTION, DMS_PREDICTOR_CORRELATIONS)),
    log:
        f"{ARTIFACT_ROOT}/logs/dms_predictor_summary.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "python {input.script}"
        " --scores {input.scores}"
        " --counts {input.counts}"
        " --output {output.table}"
        " > {log} 2>&1"


rule dms_predictors:
    """Panel C: every predictor's correlation with the DMS, as grouped bars.

    Reads the table `dms_predictor_summary` wrote and draws it. No CSV of its
    own: a panel is a rebuildable picture of the table beside it, and that table
    already exists at this stem.
    """
    input:
        panels="scripts/panels.py",
        spec="scripts/panel_spec.py",
        charts="scripts/panel_charts.py",
        dms="scripts/dms.py",
        lib="scripts/lib.py",
        config="config.yaml",
        # What the notebook reaches through those modules. Spelled out per rule
        # rather than shared through a dict, for the reason the other figure
        # rules spell theirs out: `tests/test_rule_inputs.py` reads these blocks
        # as source, and an input expanded from a variable is one it cannot see.
        analysis_charts="dasm2/analysis/charts.py",
        analysis_viral="dasm2/analysis/viral.py",
        analysis_eval="dasm2/analysis/viral_fitness_eval.py",
        io="dasm2/io.py",
        conversion="dasm2/sequence/conversion.py",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        dasm2_init="dasm2/__init__.py",
        notebook="notebooks/dms_predictors.py",
        table=str(panel_csv_path(DMS_SELECTION, DMS_PREDICTOR_CORRELATIONS)),
    output:
        html=f"{NOTEBOOK_DIR}/dms_predictors.html",
        panels=[
            str(figure_dir(DMS_SELECTION) / f"{stem}.{extension}")
            for stem in DMS_PREDICTOR_PANELS
            for extension in ("png", "svg")
        ],
    log:
        f"{ARTIFACT_ROOT}/logs/dms_predictors.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " > {log} 2>&1"
