"""`fig:agreement`'s five panels, drawn from the one scored table.

Two notebooks, five panels, five tables. Nothing here infers and nothing here
takes a GPU: ``rule score`` did the forward passes, so a panel rule reads
``selection_factors.parquet``, filters it, correlates it and draws.

The DAG:

    scored ---> agreement_mutations  (panels A, B, D) --\\
            \\-> agreement_ladders    (panels C, E)    ----> compose_agreement

**The two selections.** Each notebook is run per ``{selection}``, which names
which models it draws: ``figure`` is ``dasm-1077k``, what the paper publishes,
and ``fidelity`` is ``dasm-293k`` and ``dasm-85k``, the pair the published CSVs
were drawn from. Both come out of one table and one ``groupby``, so the second
costs no rescoring. ``scripts/panels.py`` owns the mapping.

**The panels are declared outputs, not side effects.** ``marimo export html``
embeds a failed cell in the HTML and exits 0, so a rule whose only output was
the export would succeed with no panel written. Naming every PNG and CSV is
what makes a raised cell fail the rule.

**Composition is a third rule, not a cell in either notebook.** The panels
read a 1.4M-row Parquet per model; the layout reads five PNGs. Keeping them
apart is what lets the spacing be nudged and re-run in a second instead of
rescoring four clades to move a panel by a tenth of an inch.

The images go under the artifact root and the tables into tracked
``results/panels/``: a panel is a rebuildable picture of the CSV beside it, and
`README.md`'s layout gives tracked ``results/`` to the tables. Counts, fitness
effects and selection factors are publishable; no panel carries a sequence.
"""

# Imported here, not taken from the Snakefile's namespace. This file is
# included, so a name it reaches through the includer is one that binds at
# include time; that is fine for the helpers below, which the Snakefile defines
# before including this, but a module used while the wildcard constraints are
# being evaluated is worth having outright.
import re

# Every path below comes from `scripts/panel_spec.py`, which is what the
# notebooks compute them from too: one spelling per fact, so a rule cannot
# declare an output the notebook writes somewhere else. The spec module is
# stdlib plus `lib`, deliberately: this file is parsed on every snakemake
# invocation, and importing the panel code here would pull `dasm2`, pandas and
# pyarrow into a dry run to learn five file names.
from panel_spec import (
    ARM,
    DMS_BOX_IN,
    DMS_FIGURE_STEM,
    DMS_PANELS,
    DMS_PANEL_LETTERS,
    DMS_PREDICTOR_CORRELATIONS,
    DMS_PREDICTORS,
    DMS_SELECTION,
    DMS_STRATA_FIGURE_STEM,
    DMS_STRATA_PANEL_LETTERS,
    DMS_STRATA_PANELS,
    BACKGROUND_CSV,
    BACKGROUND_PANELS,
    CANDIDATE_CSV,
    DEFAULT_SELECTION,
    DEPTH_PANELS,
    EMERGENCE_CSV,
    ECDF_CLADES,
    LADDER_ARM_PANELS,
    LADDER_PANELS,
    MUTATION_PANELS,
    OUTLIER_PANELS,
    OUTLIER_TABLE_STEM,
    PANEL_LETTERS,
    PANEL_MEM_MB,
    SHIFT_BOX_IN,
    SHIFT_FIGURE_STEM,
    SHIFT_PANEL_LETTERS,
    POOL_CELLS,
    POOL_SHIFTS,
    POOL_SUMMARY,
    SELECTIONS,
    SHIFT_CLADES,
    SUBSAMPLING_FIGURE_STEM,
    SUBSAMPLING_LADDER_CLADES,
    SUBSAMPLING_PANEL_LETTERS,
    SUBSAMPLING_SELECTION,
    composed_paths,
    era_of_clade,
    figure_dir,
    aamut_grid_path,
    dms_leaf_count_path,
    dms_score_path,
    panel_csv_path,
    sample_paths_path,
)

NOTEBOOK_DIR = f"{ARTIFACT_ROOT}/notebooks"


def clade_founder_fastas(*_):
    """Every clade founder FASTA under `clade_founders_dir`.

    The whole directory rather than `SHIFT_CLADES`, which is what these rules
    declared until #39. The shift panels still read only the four they pair,
    but `unobservable_founder_codons` asks every scored clade whether its
    founder differs from the reference at a tree-masked nucleotide, so a
    founder that changed would change a panel while the rule reported nothing
    to do.
    """
    return sorted(str(path) for path in mat_pcp_path("clade_founders_dir").glob("*.fa"))


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
    """The amino-acid grids panel B's curves are drawn from.

    Eight files, not the directory: two clades at four sampling depths, where
    full depth is the directory's own level and each cap arm is a subdirectory
    named for its corpus. Declaring the contents is how a rule declares a
    directory `config.yaml` names, as `rule score`'s `founder_fastas` does.
    """
    root = mat_pcp_path("aamut_fitness_dir")
    return [
        str(aamut_grid_path(root, clade, arm))
        for clade in ECDF_CLADES
        for arm in CORPUS_NAMES
    ]


def ladder_arm_fitness_files(*_):
    """The per-arm counts refits panels C and D read.

    Twenty files, not the directory: five clades at four sampling depths,
    where full depth is the directory's own level and each cap arm is a
    subdirectory named for its corpus.
    """
    root = mat_pcp_path("aamut_fitness_dir")
    return [
        str(aamut_grid_path(root, clade, arm))
        for clade in SUBSAMPLING_LADDER_CLADES
        for arm in CORPUS_NAMES
    ]


rule panels:
    """Every panel of `fig:agreement`, for the model the paper publishes."""
    input:
        panel_outputs(MUTATION_PANELS, DEFAULT_SELECTION),
        panel_outputs(LADDER_PANELS, DEFAULT_SELECTION),


rule subsampling:
    """`fig:subsampling`, and so its four panels beneath it.

    Three rules rather than one: A and B measure the tree and take no model,
    C and D read the scored table for `dasm-85k` on all four corpora, and the
    composition reads the four PNGs.
    """
    input:
        composed_paths(SUBSAMPLING_SELECTION, SUBSAMPLING_FIGURE_STEM),


rule compose_subsampling:
    """`fig:subsampling` itself, laid out from the four panels.

    No measuring, no scoring and no table: it reads PNGs, the same split the
    other two compositions are built on. The panels cost a pass over twenty
    amino-acid grids and over the scored table; the layout costs a second, so
    the spacing can be nudged without re-measuring anything.

    No `{selection}` wildcard. `SUBSAMPLING_SELECTION` is this figure's
    directory rather than a choice of models, so unlike `rule
    compose_agreement` there is nothing here to expand over.

    The manuscript copy is not a declared output, for the reason
    `rule compose_agreement` gives: `tex_figure_dir` points outside both this
    repository and the artifact root, and snakemake deletes a job's declared
    outputs when it fails -- so naming it would hand a failed layout the
    power to delete the figure currently in the paper. The notebook copies it
    and says whether it did.
    """
    input:
        layout="scripts/figure_layout.py",
        spec="scripts/panel_spec.py",
        lib="scripts/lib.py",
        config="config.yaml",
        notebook="notebooks/subsampling.py",
        # The PNGs only: `compose` reads raster, and declaring the SVGs too
        # would make this rule rerun on a vector copy it never opens.
        panels=[
            str(figure_dir(SUBSAMPLING_SELECTION) / f"{stem}.png")
            for stem in SUBSAMPLING_PANEL_LETTERS.values()
        ],
    output:
        html=f"{NOTEBOOK_DIR}/subsampling.html",
        figure=[
            str(path)
            for path in composed_paths(SUBSAMPLING_SELECTION, SUBSAMPLING_FIGURE_STEM)
        ],
    log:
        f"{ARTIFACT_ROOT}/logs/compose_subsampling.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " > {log} 2>&1"


def dms_files(*_):
    """The three published DMS files, by the paths `config.yaml` names.

    They are copies of what each study released, carried in `data/dms/` and
    converted by `scripts/dms_published.py` when they are read, so this figure
    rebuilds from the checkout alone.

    Declared as files rather than as the directory holding them, so that
    `data/dms/PROVENANCE.md` -- which these rules do not read -- cannot
    invalidate a panel by being edited. The md5 in `config.yaml` is the other
    half: the paths catch a file that moved, the digest catches one replaced
    in place by a later release.
    """
    return [str(dms_dataset(homolog).path) for homolog in dms_homologs()]


rule dms:
    """`fig:dms`, and so its four panels beneath it.

    Two rules rather than one, as every figure here is split: the panels read
    the scored table, the three assays and twenty founders; the composition
    reads four PNGs.
    """
    input:
        composed_paths(DMS_SELECTION, DMS_FIGURE_STEM),


rule dms_correlations:
    """All four panels: two count thresholds, scatters and bars at each.

    One rule for the four, not two: the bars are the coefficient the scatters
    label, computed over the same frames in the same pass, so splitting them
    would read the scored table and all three assays twice to draw a number
    that is already in hand.
    """
    input:
        dms="scripts/dms.py",
        published="scripts/dms_published.py",
        panels="scripts/panels.py",
        spec="scripts/panel_spec.py",
        charts="scripts/panel_charts.py",
        lib="scripts/lib.py",
        config="config.yaml",
        # What the notebook reaches through those modules, spelled out per
        # rule for the reason `rule subsampling_depth` spells its own out.
        analysis_charts="dasm2/analysis/charts.py",
        analysis_viral="dasm2/analysis/viral.py",
        analysis_eval="dasm2/analysis/viral_fitness_eval.py",
        codon_table="dasm2/codon_table.py",
        functional="dasm2/data/functional/__init__.py",
        dms_matrix="dasm2/data/functional/dms_matrix.py",
        functional_formats="dasm2/data/functional_formats.py",
        io="dasm2/io.py",
        constants="dasm2/sequence/constants.py",
        conversion="dasm2/sequence/conversion.py",
        masking="dasm2/sequence/masking.py",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        dasm2_init="dasm2/__init__.py",
        notebook="notebooks/dms_correlations.py",
        table=SCORED_TABLE,
        # The assays themselves, as `data/dms/` carries them, and the founders
        # every one of them is read against: `load_dms` drops the measurements
        # whose wildtype disagrees with the clade founder, and the strata are
        # the founder's codons.
        dms_data=dms_files,
        clade_founders=founder_fastas,
        founder_reader="scripts/founders.py",
        # The two layers of the loader's mask, in the three files they are
        # read from. Not the figure's exclusion -- that arrives as the nulls
        # `rule score` left behind -- but what `from_multidms` needs before it
        # will parse a masked site. `censored_sites` calls `load_censor_table`
        # on the mutations *and* the clades, so both are declared here: with
        # only the first, a re-emitted clades table would leave these panels
        # stale and snakemake would report nothing to do.
        masked_nt_sites=str(mat_pcp_path("masked_nt_sites_file")),
        censored_mutations=str(mat_pcp_path("censored_mutations_file")),
        censored_clades=str(mat_pcp_path("censored_clades_file")),
        reference=str(mat_pcp_path("reference_sequence_file")),
    output:
        html=f"{NOTEBOOK_DIR}/dms_correlations.html",
        panels=panel_outputs(DMS_PANELS, DMS_SELECTION),
    log:
        f"{ARTIFACT_ROOT}/logs/dms_correlations.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " -- --table {input.table}"
        " > {log} 2>&1"


rule compose_dms:
    """`fig:dms` itself, laid out from its four panels.

    No scoring, no assay and no table: it reads PNGs, the split every
    composition here is built on. The manuscript copy is not a declared
    output, for the reason `rule compose_agreement` gives.
    """
    input:
        layout="scripts/figure_layout.py",
        spec="scripts/panel_spec.py",
        lib="scripts/lib.py",
        config="config.yaml",
        notebook="notebooks/dms.py",
        panels=[
            str(figure_dir(DMS_SELECTION) / f"{stem}.png")
            for stem in DMS_PANEL_LETTERS.values()
        ],
    output:
        html=f"{NOTEBOOK_DIR}/dms.html",
        figure=[str(path) for path in composed_paths(DMS_SELECTION, DMS_FIGURE_STEM)],
    log:
        f"{ARTIFACT_ROOT}/logs/compose_dms.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " > {log} 2>&1"


# ---------------------------------------------------------------------------
# `fig:dms`'s reference predictors: the floors that are not the DASM
# ---------------------------------------------------------------------------
#
# One rule per predictor, plus the leaf tally on its own. The tally is the
# expensive one -- clade 21K is 1.9M leaves and 1.6 GB of paths -- and neither
# BLOSUM62 nor antiGen reads it, so folding the predictors together would
# rebuild it on every edit to the substitution matrix.
#
# Most of these need no trained model, which is why they could land before #16.
# `dms_dasm_scores` and `dms_prob_scores` are the exceptions and read
# `selection_factors.parquet`: the first is a filter and a rename over a table
# `rule score` already built, the second exponentiates the same table's grid
# and weights it by the neutral rates. Neither is a forward pass.
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


rule dms_predictors:
    """The DMS predictor comparison: every predictor's correlation, per stratum.

    The two reference predictors, antiGen, the DASM, and the DASM read as a
    probability (`dasm_prob`) beside the neutral rate it differs from the
    selection factor by (`neutral_rate`) -- all through one join onto one
    population. `fig:dms_strata` is what draws the first four; `rule dms_strata`
    builds that, and this target is the table alone. The last two are drawn by
    nothing: a probability is defined only on the single-nucleotide stratum, so
    it is a row of the table rather than a column of the figure.

    **This target is no longer model-free, and that changed when the DASM
    joined `DMS_PREDICTORS`.** It was: the three reference predictors needed no
    checkpoint, which is why the section header above says they could land
    before #16. `dms_dasm_scores` and `dms_prob_scores` both read
    `selection_factors.parquet`, so this target now sits downstream of
    `rule score` and therefore of `rule train`.
    Building it alone puts the training rules in the plan -- which is a
    statement about the DAG rather than about this rule, since `rule all`
    already reaches `train` through `models` and `scored`. Dry-run before
    running it, as with anything that can reach a checkpoint.
    """
    input:
        str(panel_csv_path(DMS_SELECTION, DMS_PREDICTOR_CORRELATIONS)),


rule dms_leaf_counts:
    """One clade's per-site residue tally over its own tree leaves.

    Every leaf's spike is reconstructed from its own root-to-tip mutation path
    applied to 19A's coding sequence, then translated and tallied. Clade 21K is
    1.9M leaves; the three together read about 2.4 GB in roughly 3.5 minutes.

    `PANEL_MEM_MB` covers it with an order of magnitude to spare, and that is a
    measurement rather than an assumption: the tally is a fixed 20 x 1,273 and
    the paths are streamed a leaf at a time, so peak RSS is flat in the leaf
    count -- 0.80 GB for 24E's 124,359 leaves and 0.82 GB for 21K's 1,942,882.
    """
    wildcard_constraints:
        clade="|".join(re.escape(clade) for clade in DMS_CLADES),
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
        # glob, as the other figure rules declare it -- it already covers both,
        # and a two-file list beside it would be a second way of naming founders.
        paths=lambda w: str(
            sample_paths_path(mat_pcp_path("sample_paths_dir"), w.clade)
        ),
        clade_founders=clade_founder_fastas,
        # Read through `dms.globally_masked_sites()` for the founder-agreement
        # line this rule prints. Declared although it changes no count, because
        # an undeclared read is the defect `tests/test_rule_inputs.py` exists for.
        masked=str(mat_pcp_path("masked_nt_sites_file")),
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


rule dms_dasm_scores:
    """The DASM's log selection factor for every cell one homolog's assay measured.

    **The one score rule that reads `selection_factors.parquet`.** `rule score`
    did the forward passes; this reads one model's rows for one clade and
    renames a column, so it claims no GPU and loads no checkpoint.

    Declares no leaf tally, for `dms_blosum_scores`' reason.
    """
    wildcard_constraints:
        homolog="|".join(re.escape(homolog) for homolog in dms_homologs()),
    input:
        script="scripts/dms_dasm.py",
        models="scripts/dms_models.py",
        dms="scripts/dms.py",
        published="scripts/dms_published.py",
        censor="scripts/dms_censor.py",
        panels="scripts/panels.py",
        spec="scripts/panel_spec.py",
        founder_reader="scripts/founders.py",
        lib="scripts/lib.py",
        config="config.yaml",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        constants="dasm2/sequence/constants.py",
        conversion="dasm2/sequence/conversion.py",
        masking="dasm2/sequence/masking.py",
        codon_table="dasm2/codon_table.py",
        parquet_backend="dasm2/data/parquet_backend.py",
        dasm2_init="dasm2/__init__.py",
        dms_csv=lambda w: str(dms_dataset(w.homolog).path),
        clade_founders=clade_founder_fastas,
        censored=str(mat_pcp_path("censored_mutations_file")),
        clades=str(mat_pcp_path("censored_clades_file")),
        masked=str(mat_pcp_path("masked_nt_sites_file")),
        reference=str(mat_pcp_path("reference_sequence_file")),
        table=SCORED_TABLE,
    output:
        scores=str(dms_score_path("dasm", "{homolog}")),
    log:
        f"{ARTIFACT_ROOT}/logs/dms_scores_dasm_{{homolog}}.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "python {input.script}"
        " --homolog {wildcards.homolog}"
        " --table {input.table}"
        " --output {output.scores}"
        " > {log} 2>&1"


rule dms_prob_scores:
    """The DASM's mutation probability, and the neutral rate behind it.

    **Two outputs, one pass.** `scripts/dms_prob.py` writes one score file per
    predictor because the contract is one `Manifest` per model, but the archive
    load and the partition function are shared, so splitting this into two rules
    would do the expensive half twice for the same founder.

    Reads `selection_factors.parquet` through `dms_dasm.selection_factor_grid`,
    which is why `scripts/dms_dasm.py` is declared here: the linear selection
    factors a probability is defined against are that table's own column,
    exponentiated. Like `dms_dasm_scores` it claims no GPU and loads no
    checkpoint.

    **The archive is resolved here rather than in the script**, so the path
    `workflow/corpus.smk` owns has one spelling and so the rule can declare it.
    `NEUTRAL_ZIP` is that file's mapping for exactly this; importing
    `build_era_neutral_models.archive_name` instead would pull `dasm2` and torch
    into every dry run.

    Declares no leaf tally, for `dms_blosum_scores`' reason.
    """
    wildcard_constraints:
        homolog="|".join(re.escape(homolog) for homolog in dms_homologs()),
    input:
        script="scripts/dms_prob.py",
        dasm="scripts/dms_dasm.py",
        models="scripts/dms_models.py",
        dms="scripts/dms.py",
        published="scripts/dms_published.py",
        censor="scripts/dms_censor.py",
        panels="scripts/panels.py",
        spec="scripts/panel_spec.py",
        founder_reader="scripts/founders.py",
        lib="scripts/lib.py",
        config="config.yaml",
        whichmut_probs="dasm2/analysis/whichmut_probs.py",
        whichmut_loss="dasm2/losses/whichmut_loss.py",
        codon_neutral_rates="dasm2/molevol/codon_neutral_rates.py",
        neutral_base="dasm2/neutral/base.py",
        tabular_rates="dasm2/neutral/tabular_rates.py",
        zip_io="dasm2/serialization/zip_io.py",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        constants="dasm2/sequence/constants.py",
        conversion="dasm2/sequence/conversion.py",
        masking="dasm2/sequence/masking.py",
        codon_table="dasm2/codon_table.py",
        parquet_backend="dasm2/data/parquet_backend.py",
        dasm2_init="dasm2/__init__.py",
        dms_csv=lambda w: str(dms_dataset(w.homolog).path),
        clade_founders=clade_founder_fastas,
        censored=str(mat_pcp_path("censored_mutations_file")),
        clades=str(mat_pcp_path("censored_clades_file")),
        masked=str(mat_pcp_path("masked_nt_sites_file")),
        reference=str(mat_pcp_path("reference_sequence_file")),
        archive=lambda w: NEUTRAL_ZIP[era_of_clade(dms_dataset(w.homolog).clade)],
        table=SCORED_TABLE,
    output:
        scores=str(dms_score_path("dasm_prob", "{homolog}")),
        neutral=str(dms_score_path("neutral_rate", "{homolog}")),
    log:
        f"{ARTIFACT_ROOT}/logs/dms_scores_dasm_prob_{{homolog}}.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "python {input.script}"
        " --homolog {wildcards.homolog}"
        " --table {input.table}"
        " --neutral-archive {input.archive}"
        " --output {output.scores}"
        " --neutral-output {output.neutral}"
        " > {log} 2>&1"


rule dms_blosum_scores:
    """BLOSUM62's score for every cell one homolog's assay measured.

    Declares no leaf tally: it reads none, and a rule that declared one would
    make every BLOSUM62 score wait on 2.4 GB of path files it never opens.
    """
    wildcard_constraints:
        homolog="|".join(re.escape(homolog) for homolog in dms_homologs()),
    input:
        script="scripts/dms_blosum.py",
        models="scripts/dms_models.py",
        censor="scripts/dms_censor.py",
        panels="scripts/panels.py",
        dms="scripts/dms.py",
        published="scripts/dms_published.py",
        founder_reader="scripts/founders.py",
        lib="scripts/lib.py",
        config="config.yaml",
        # `scov2_rates` and `conversion` are reached through `dms_models` and
        # `scripts/founders.py` rather than imported here, and are declared for
        # the same reason `parquet_backend` is: the three score rules below each
        # run code that reaches all four, so they declare the same four.
        scov2_rates="dasm2/neutral/scov2_rates.py",
        constants="dasm2/sequence/constants.py",
        conversion="dasm2/sequence/conversion.py",
        parquet_backend="dasm2/data/parquet_backend.py",
        dasm2_init="dasm2/__init__.py",
        # The assay itself, and the clade founder every score is taken away from.
        # `scripts/dms.py` gates the CSV on the md5 `config.yaml` pins, so a
        # repointed homolog fails there rather than moving a correlation quietly.
        dms_csv=lambda w: str(dms_dataset(w.homolog).path),
        clade_founders=clade_founder_fastas,
        censored=str(mat_pcp_path("censored_mutations_file")),
        clades=str(mat_pcp_path("censored_clades_file")),
        masked=str(mat_pcp_path("masked_nt_sites_file")),
        reference=str(mat_pcp_path("reference_sequence_file")),
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


rule dms_antigen_scores:
    """antiGen's log probability for every cell one homolog's assay measured.

    **Claims no GPU.** `antiGen/infer.py` takes CUDA whenever a card is visible
    and the figure rules hold no slot from `scripts/gpu_slot.py`, so the script
    pins the device rather than defaulting it; the pass is a few seconds of CPU
    on one 1,273-residue query.

    Declares no leaf tally, for `dms_blosum_scores`' reason, and declares two
    files no import walk can see: the checkpoint and the query FASTA, which the
    script reaches through `config.yaml`'s `antigen` block. It does **not**
    declare the antiGen source tree -- `antigen.checkout` names a tree outside
    this repository, not content-tracked here and pinned instead by the commit
    `data/antigen/PROVENANCE.md` records.
    """
    wildcard_constraints:
        homolog="|".join(re.escape(homolog) for homolog in dms_homologs()),
    input:
        script="scripts/dms_antigen.py",
        models="scripts/dms_models.py",
        censor="scripts/dms_censor.py",
        panels="scripts/panels.py",
        dms="scripts/dms.py",
        published="scripts/dms_published.py",
        founder_reader="scripts/founders.py",
        lib="scripts/lib.py",
        config="config.yaml",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        constants="dasm2/sequence/constants.py",
        conversion="dasm2/sequence/conversion.py",
        parquet_backend="dasm2/data/parquet_backend.py",
        dasm2_init="dasm2/__init__.py",
        dms_csv=lambda w: str(dms_dataset(w.homolog).path),
        clade_founders=clade_founder_fastas,
        censored=str(mat_pcp_path("censored_mutations_file")),
        clades=str(mat_pcp_path("censored_clades_file")),
        masked=str(mat_pcp_path("masked_nt_sites_file")),
        reference=str(mat_pcp_path("reference_sequence_file")),
        # The model, and the construct projection it is fed. Both gated on the
        # md5 `config.yaml` pins, as the DMS CSVs are, so a replaced checkpoint
        # fails rather than moving the antiGen row quietly.
        checkpoint=str(antigen_model().checkpoint),
        parents=str(antigen_model().parents),
    output:
        scores=str(dms_score_path("antigen", "{homolog}")),
    log:
        f"{ARTIFACT_ROOT}/logs/dms_scores_antigen_{{homolog}}.log",
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
        censor="scripts/dms_censor.py",
        panels="scripts/panels.py",
        leaf_freq="scripts/dms_leaf_freq.py",
        dms="scripts/dms.py",
        published="scripts/dms_published.py",
        spec="scripts/panel_spec.py",
        founder_reader="scripts/founders.py",
        lib="scripts/lib.py",
        config="config.yaml",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        constants="dasm2/sequence/constants.py",
        conversion="dasm2/sequence/conversion.py",
        parquet_backend="dasm2/data/parquet_backend.py",
        dasm2_init="dasm2/__init__.py",
        dms_csv=lambda w: str(dms_dataset(w.homolog).path),
        clade_founders=clade_founder_fastas,
        censored=str(mat_pcp_path("censored_mutations_file")),
        clades=str(mat_pcp_path("censored_clades_file")),
        masked=str(mat_pcp_path("masked_nt_sites_file")),
        reference=str(mat_pcp_path("reference_sequence_file")),
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
    """The table: each predictor against the DMS, per homolog and per stratum."""
    input:
        script="scripts/dms_summary.py",
        models="scripts/dms_models.py",
        censor="scripts/dms_censor.py",
        panels="scripts/panels.py",
        dms="scripts/dms.py",
        published="scripts/dms_published.py",
        spec="scripts/panel_spec.py",
        founder_reader="scripts/founders.py",
        lib="scripts/lib.py",
        config="config.yaml",
        analysis_eval="dasm2/analysis/viral_fitness_eval.py",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        constants="dasm2/sequence/constants.py",
        conversion="dasm2/sequence/conversion.py",
        parquet_backend="dasm2/data/parquet_backend.py",
        dasm2_init="dasm2/__init__.py",
        dms_csv=dms_files,
        # No leaf tally: this reads the scores, and the correlation it computes
        # is over the cells they carry. The tallies reach it only through
        # `dms_leaf_scores`, which is upstream of the scores it does read.
        clade_founders=clade_founder_fastas,
        censored=str(mat_pcp_path("censored_mutations_file")),
        clades=str(mat_pcp_path("censored_clades_file")),
        masked=str(mat_pcp_path("masked_nt_sites_file")),
        reference=str(mat_pcp_path("reference_sequence_file")),
        scores=dms_all_scores,
    output:
        table=str(panel_csv_path(DMS_SELECTION, DMS_PREDICTOR_CORRELATIONS)),
    log:
        f"{ARTIFACT_ROOT}/logs/dms_predictor_summary.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "python {input.script}"
        " --scores {input.scores}"
        " --output {output.table}"
        " > {log} 2>&1"


rule dms_strata:
    """`fig:dms_strata`, and so its two panels beneath it."""
    input:
        composed_paths(DMS_SELECTION, DMS_STRATA_FIGURE_STEM),


rule dms_strata_panels:
    """The two model rows: DASM over antiGen, each cut by codon accessibility.

    One rule for both, not two: they read the same join of the same scores and
    differ only in which column becomes y, so splitting them would build the
    merged frame twice to draw the second picture.

    **Declares the correlation table.** `r` and `n` are read from it rather
    than recomputed in the notebook, which is what keeps the picture and the
    CSV beside it from drifting -- so this rule waits on
    `rule dms_predictor_summary` rather than on the scores alone.
    """
    input:
        notebook="notebooks/dms_strata.py",
        summary="scripts/dms_summary.py",
        models="scripts/dms_models.py",
        censor="scripts/dms_censor.py",
        dms="scripts/dms.py",
        published="scripts/dms_published.py",
        panels="scripts/panels.py",
        spec="scripts/panel_spec.py",
        charts="scripts/panel_charts.py",
        founder_reader="scripts/founders.py",
        lib="scripts/lib.py",
        config="config.yaml",
        analysis_eval="dasm2/analysis/viral_fitness_eval.py",
        codon_table="dasm2/codon_table.py",
        constants="dasm2/sequence/constants.py",
        conversion="dasm2/sequence/conversion.py",
        masking="dasm2/sequence/masking.py",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        parquet_backend="dasm2/data/parquet_backend.py",
        dasm2_init="dasm2/__init__.py",
        dms_data=dms_files,
        clade_founders=clade_founder_fastas,
        censored=str(mat_pcp_path("censored_mutations_file")),
        clades=str(mat_pcp_path("censored_clades_file")),
        masked=str(mat_pcp_path("masked_nt_sites_file")),
        reference=str(mat_pcp_path("reference_sequence_file")),
        scores=dms_all_scores,
        table=str(panel_csv_path(DMS_SELECTION, DMS_PREDICTOR_CORRELATIONS)),
    output:
        html=f"{NOTEBOOK_DIR}/dms_strata.html",
        panels=panel_outputs(DMS_STRATA_PANELS, DMS_SELECTION),
    log:
        f"{ARTIFACT_ROOT}/logs/dms_strata.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " -- --scores {input.scores}"
        " > {log} 2>&1"


rule compose_dms_strata:
    """`fig:dms_strata` itself, laid out from its two panels.

    No scoring, no assay and no table: it reads PNGs, the split every
    composition here is built on. The manuscript copy is not a declared
    output, for the reason `rule compose_agreement` gives.
    """
    input:
        layout="scripts/figure_layout.py",
        spec="scripts/panel_spec.py",
        lib="scripts/lib.py",
        config="config.yaml",
        charts="scripts/panel_charts.py",
        notebook="notebooks/dms_strata_figure.py",
        panels=[
            str(figure_dir(DMS_SELECTION) / f"{stem}.png")
            for stem in DMS_STRATA_PANEL_LETTERS.values()
        ],
    output:
        html=f"{NOTEBOOK_DIR}/dms_strata_figure.html",
        figure=[
            str(path)
            for path in composed_paths(DMS_SELECTION, DMS_STRATA_FIGURE_STEM)
        ],
    log:
        f"{ARTIFACT_ROOT}/logs/compose_dms_strata.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " > {log} 2>&1"


def clade_sample_files(*_):
    """Every clade's sample list under `clades_dir`.

    The contents rather than the directory, the way `clade_founder_fastas`
    above declares its own, and keyed off those same founders: the roster is
    what this pipeline scores, and `clade_emergence.py` raises for a clade
    with a founder and no sample list.
    """
    root = mat_pcp_path("clades_dir")
    founders = mat_pcp_path("clade_founders_dir").glob("*.fa")
    return sorted(str(root / f"{fasta.stem}_samples.txt") for fasta in founders)


rule clade_emergence:
    """When each clade emerged, which panel A colours its dots by.

    Its own rule rather than a cell of `rule subsampling_depth`, in the style
    of `rule clade_backgrounds`: this walks 2.7 GB of sample names across 56
    files, the panel beside it is redrawn often, and the answer does not
    change unless the tree does.

    No model and no wildcard. The date is a property of mat-pcp's tree, so
    every figure that ever reads it reads the same one.
    """
    input:
        script="scripts/clade_emergence.py",
        lib="scripts/lib.py",
        spec="scripts/panel_spec.py",
        config="config.yaml",
        dasm2_init="dasm2/__init__.py",
        # The sample lists, and the founders the roster is read from.
        samples=clade_sample_files,
        clade_founders=clade_founder_fastas,
    output:
        table=str(EMERGENCE_CSV),
    log:
        f"{ARTIFACT_ROOT}/logs/clade_emergence.log",
    params:
        clades_dir=lambda *_: str(mat_pcp_path("clades_dir")),
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "python {input.script} --clades-dir {params.clades_dir}"
        " --out {output.table} > {log} 2>&1"


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
        # The dates panel A colours its dots by. `rule clade_emergence`'s
        # table rather than the sample lists themselves: this rule is redrawn
        # often and that walk is 2.7 GB.
        emergence=str(EMERGENCE_CSV),
        nt_grids=ecdf_grid_files,
        # The roster panel A plots, which `clade_size_frame` drops the counts
        # table's two extra clades against.
        clade_founders=founder_fastas,
        founder_reader="scripts/founders.py",
        # `_aa_grid` drops each clade's unobservable founder codons, which
        # `unobservable_founder_codons` rebuilds from these two and the
        # founder above.
        masked_nt_sites=str(mat_pcp_path("masked_nt_sites_file")),
        reference=str(mat_pcp_path("reference_sequence_file")),
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


rule subsampling_ladder:
    """Panels C and D: what thinning the tree costs each approach.

    The per-arm amino-acid tables are declared as files rather than as the
    directory, in the style of `rule subsampling_depth`'s grids: twenty of
    them, five clades at four sampling depths. These are the arms' **own**
    counts refits, which are the panel's second predictor -- the target is the
    full-depth estimate `rule score` already joined onto every arm's rows.
    """
    input:
        panels="scripts/panels.py",
        spec="scripts/panel_spec.py",
        charts="scripts/panel_charts.py",
        lib="scripts/lib.py",
        config="config.yaml",
        analysis_charts="dasm2/analysis/charts.py",
        analysis_viral="dasm2/analysis/viral.py",
        analysis_eval="dasm2/analysis/viral_fitness_eval.py",
        io="dasm2/io.py",
        conversion="dasm2/sequence/conversion.py",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        dasm2_init="dasm2/__init__.py",
        notebook="notebooks/subsampling_ladder.py",
        table=SCORED_TABLE,
        arm_fitness=ladder_arm_fitness_files,
        founder_reader="scripts/founders.py",
    output:
        html=f"{NOTEBOOK_DIR}/subsampling_ladder.html",
        panels=panel_outputs(LADDER_ARM_PANELS, SUBSAMPLING_SELECTION),
    log:
        f"{ARTIFACT_ROOT}/logs/subsampling_ladder.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " -- --table {input.table}"
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
        # Every clade's founder: the shift panel drops identical pairs on
        # four of them, and `unobservable_founder_codons` reads one per clade
        # scored. Plus the two files that predicate is built from.
        clade_founders=clade_founder_fastas,
        masked_nt_sites=str(mat_pcp_path("masked_nt_sites_file")),
        reference=str(mat_pcp_path("reference_sequence_file")),
    output:
        html=f"{NOTEBOOK_DIR}/agreement_mutations_{{selection}}.html",
        panels=panel_outputs(MUTATION_PANELS, "{selection}"),
    log:
        f"{ARTIFACT_ROOT}/logs/agreement_mutations_{{selection}}.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " -- --table {input.table}"
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
        # As above: every founder, and the two files the unobservable-codon
        # predicate is rebuilt from.
        clade_founders=clade_founder_fastas,
        masked_nt_sites=str(mat_pcp_path("masked_nt_sites_file")),
        reference=str(mat_pcp_path("reference_sequence_file")),
    output:
        html=f"{NOTEBOOK_DIR}/agreement_ladders_{{selection}}.html",
        panels=panel_outputs(LADDER_PANELS, "{selection}"),
    log:
        f"{ARTIFACT_ROOT}/logs/agreement_ladders_{{selection}}.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " -- --table {input.table}"
        " --selection {wildcards.selection}"
        " > {log} 2>&1"


rule compose_agreement:
    """`fig:agreement` itself, laid out from the five panels the exports wrote.

    No scoring and no table: it reads PNGs. `scripts/figure_layout.py` solves
    the dpi rather than taking one, so the figure is as large as it can be
    inside the page box with nothing resampled and no panel moved relative to
    another.

    The manuscript copy is not a declared output. `config.yaml`'s
    `tex_figure_dir` points outside both this repository and the artifact
    root, and snakemake deletes a job's declared outputs when it fails -- so
    naming it would hand a failed layout the power to delete the figure
    currently in the paper. The notebook copies it and says whether it did.
    """
    wildcard_constraints:
        selection="|".join(SELECTIONS),
    input:
        layout="scripts/figure_layout.py",
        spec="scripts/panel_spec.py",
        lib="scripts/lib.py",
        config="config.yaml",
        notebook="notebooks/agreement.py",
        # The panels themselves, by the letters the figure draws. The PNGs
        # only: `compose` reads raster, and declaring the SVGs too would make
        # this rule rerun on a vector copy it never opens.
        panels=[
            str(figure_dir("{selection}") / f"{stem}.png")
            for stem in PANEL_LETTERS.values()
        ],
    output:
        html=f"{NOTEBOOK_DIR}/agreement_{{selection}}.html",
        figure=[str(path) for path in composed_paths("{selection}")],
    log:
        f"{ARTIFACT_ROOT}/logs/compose_agreement_{{selection}}.log",
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " -- --selection {wildcards.selection}"
        " > {log} 2>&1"


rule within_clade_outliers:
    """`fig:within_clade_shifts` panel A: the candidate mutations, two clades.

    Its own rule rather than a third panel of `rule agreement_mutations`,
    which draws a different figure: these two are built and read together, and
    a rule that wrote panels of two figures would rebuild one whenever the
    other's notebook changed.
    """
    wildcard_constraints:
        selection="|".join(SELECTIONS),
    input:
        panels="scripts/panels.py",
        spec="scripts/panel_spec.py",
        charts="scripts/panel_charts.py",
        founder_reader="scripts/founders.py",
        lib="scripts/lib.py",
        config="config.yaml",
        # As the agreement rules: spelled out rather than shared through a
        # dict, because `tests/test_rule_inputs.py` reads these blocks as
        # source and cannot see an input expanded from a variable.
        analysis_charts="dasm2/analysis/charts.py",
        analysis_viral="dasm2/analysis/viral.py",
        analysis_eval="dasm2/analysis/viral_fitness_eval.py",
        io="dasm2/io.py",
        conversion="dasm2/sequence/conversion.py",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        dasm2_init="dasm2/__init__.py",
        notebook="notebooks/within_clade_outliers.py",
        table=SCORED_TABLE,
        # The candidate list the panel labels. A rule input, not just a module
        # constant, so moving a mutation off the figure redraws the panel.
        candidates=str(CANDIDATE_CSV),
        # ...and the derived half of that list: the substitutions that define
        # each clade's two backgrounds, which the panel names in its own ink.
        backgrounds=str(BACKGROUND_CSV),
        clade_founders=clade_founder_fastas,
        masked_nt_sites=str(mat_pcp_path("masked_nt_sites_file")),
        reference=str(mat_pcp_path("reference_sequence_file")),
    output:
        html=f"{NOTEBOOK_DIR}/within_clade_outliers_{{selection}}.html",
        panels=panel_outputs(OUTLIER_PANELS, "{selection}"),
    log:
        f"{ARTIFACT_ROOT}/logs/within_clade_outliers_{{selection}}.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " -- --table {input.table}"
        " --selection {wildcards.selection}"
        " > {log} 2>&1"


rule outlier_ranking:
    """Table S1: the mutations the two approaches most disagree about.

    A table rather than a panel, and its own rule rather than a cell of a
    panel notebook: it is the agreement comparison at a higher count floor,
    read by the supplement, and nothing in a figure depends on it.
    """
    wildcard_constraints:
        selection="|".join(SELECTIONS),
    input:
        panels="scripts/panels.py",
        spec="scripts/panel_spec.py",
        lib="scripts/lib.py",
        founder_reader="scripts/founders.py",
        config="config.yaml",
        analysis_viral="dasm2/analysis/viral.py",
        analysis_eval="dasm2/analysis/viral_fitness_eval.py",
        io="dasm2/io.py",
        conversion="dasm2/sequence/conversion.py",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        dasm2_init="dasm2/__init__.py",
        notebook="notebooks/outlier_ranking.py",
        table=SCORED_TABLE,
        # The clade roster, taken from the agreement figure's own tracked
        # table so the two cannot disagree about which clades are compared.
        roster=str(panel_csv_path(DEFAULT_SELECTION, "panel_c_within_clade_ladder")),
        # The branches each clade's opportunity is measured over. mat-pcp's
        # PCPs rather than this pipeline's Parquet: the Parquet drops the
        # branch length the mass is weighted by.
        pcps=str(corpus(ARM).source_pcp),
        clade_founders=clade_founder_fastas,
        masked_nt_sites=str(mat_pcp_path("masked_nt_sites_file")),
        reference=str(mat_pcp_path("reference_sequence_file")),
    output:
        html=f"{NOTEBOOK_DIR}/outlier_ranking_{{selection}}.html",
        table_csv=str(panel_csv_path("{selection}", OUTLIER_TABLE_STEM)),
    log:
        f"{ARTIFACT_ROOT}/logs/outlier_ranking_{{selection}}.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " -- --table {input.table}"
        " --selection {wildcards.selection}"
        " > {log} 2>&1"


rule clade_backgrounds:
    """The two genetic backgrounds each clade of `fig:within_clade_shifts` splits into.

    From branch frequencies alone: no model, no selection, no wildcard. That
    is the point of the rule. The earlier pipeline chose the split by scoring
    every parent haplotype under the DASM, which made the contrast move with
    the checkpoint; defined this way it is a property of the tree, so every
    model is measured against the same two backgrounds.

    Two passes over a 112 MB corpus, so it is its own rule rather than a cell
    of the panel notebook: the panels are redrawn often and this answer does
    not change unless the corpus does.
    """
    input:
        backgrounds="scripts/backgrounds.py",
        panels="scripts/panels.py",
        spec="scripts/panel_spec.py",
        lib="scripts/lib.py",
        config="config.yaml",
        conversion="dasm2/sequence/conversion.py",
        dasm2_init="dasm2/__init__.py",
        # The branches the frequencies are measured over, and the founders
        # each parent is compared against. mat-pcp's PCPs rather than this
        # pipeline's Parquet, which drops the branch length the mass is
        # weighted by.
        pcps=str(corpus(ARM).source_pcp),
        clade_founders=clade_founder_fastas,
    output:
        table=str(BACKGROUND_CSV),
    log:
        f"{ARTIFACT_ROOT}/logs/clade_backgrounds.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "python {input.backgrounds} --pcp {input.pcps} --out {output.table}"
        " > {log} 2>&1"


rule pool_counts:
    """The counts half of `fig:within_clade_shifts`' panels B, D and F.

    For every candidate mutation, its effect in each of its clade's two
    backgrounds, the shift between them, and a randomization test of that
    shift. Two streaming passes per clade rather than two per candidate:
    `tally_pools` serves every pool in one and `tally_pcps` every candidate
    in the other.

    **No checkpoint and no card.** The split comes from branch frequencies
    and the counts from the corpus, so nothing the model produces reaches the
    pools or the p-value -- which is what makes the panel a test of the DASM
    rather than a restatement of it. The scored haplotypes are an input
    because the panel draws both estimates from one table, not because the
    counts depend on them.
    """
    input:
        script="scripts/pool_counts.py",
        haplotype_reader="scripts/clade_haplotypes.py",
        founder_reader="scripts/founders.py",
        spec="scripts/panel_spec.py",
        lib="scripts/lib.py",
        config="config.yaml",
        pools="dasm2/analysis/within_clade_pools.py",
        conversion="dasm2/sequence/conversion.py",
        io="dasm2/io.py",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        dasm2_init="dasm2/__init__.py",
        candidates=str(CANDIDATE_CSV),
        backgrounds=str(BACKGROUND_CSV),
        scores=str(HAPLOTYPE_TABLE),
        # The era partition this pipeline owns, and both fitted rate tables
        # it picks between. Declared by hand: `test_rule_inputs.py` finds
        # only literal `mat_pcp_path(...)` calls, and these are reached
        # through `lib.master_table_path(era)`, so a re-emitted rate table
        # would otherwise leave the panels stale with nothing reporting it.
        era_template="data/eras-pre-omicron-omicron.yaml",
        rates_pre_omicron=str(mat_pcp_path("master_table_pre_omicron_file")),
        rates_omicron=str(mat_pcp_path("master_table_omicron_file")),
        annotations=str(mat_pcp_path("site_annotations_file")),
        pcps=str(corpus(ARM).source_pcp),
        clade_founders=clade_founder_fastas,
    output:
        cells=str(POOL_CELLS),
        shifts=str(POOL_SHIFTS),
        summary=str(POOL_SUMMARY),
    log:
        f"{ARTIFACT_ROOT}/logs/pool_counts.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "python {input.script}"
        " --pcp {input.pcps}"
        " --backgrounds {input.backgrounds}"
        " --haplotype-scores {input.scores}"
        " --cells {output.cells}"
        " --shifts {output.shifts}"
        " --summary {output.summary}"
        " > {log} 2>&1"


rule compose_within_clade_shifts:
    """`fig:within_clade_shifts` itself, laid out from the six panels.

    No scoring, no tallying and no table: it reads PNGs, the same split that
    makes `fig:agreement` cheap to re-lay-out. The panels cost two corpus
    passes per clade and a scoring run; the layout costs a second, so the
    spacing can be nudged without re-measuring anything.

    The manuscript copy is not a declared output, for the reason
    `rule compose_agreement` gives: `tex_figure_dir` points outside both this
    repository and the artifact root, and snakemake deletes a job's declared
    outputs when it fails -- so naming it would hand a failed layout the
    power to delete the figure currently in the paper. The notebook copies it
    and says whether it did.
    """
    wildcard_constraints:
        selection="|".join(SELECTIONS),
    input:
        layout="scripts/figure_layout.py",
        spec="scripts/panel_spec.py",
        lib="scripts/lib.py",
        config="config.yaml",
        notebook="notebooks/within_clade_shifts.py",
        # The PNGs only: `compose` reads raster, and declaring the SVGs too
        # would make this rule rerun on a vector copy it never opens.
        panels=[
            str(figure_dir("{selection}") / f"{stem}.png")
            for stem in SHIFT_PANEL_LETTERS.values()
        ],
    output:
        html=f"{NOTEBOOK_DIR}/within_clade_shifts_{{selection}}.html",
        figure=[
            str(path) for path in composed_paths("{selection}", SHIFT_FIGURE_STEM)
        ],
    log:
        f"{ARTIFACT_ROOT}/logs/compose_within_clade_shifts_{{selection}}.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " -- --selection {wildcards.selection}"
        " > {log} 2>&1"


rule figures:
    """Every composed figure of the paper, and so every panel beneath them.

    Both selections, not just the published one. `fidelity`'s tables are
    tracked beside `figure`'s, and until #36 nothing in the DAG built them: a
    filter change rewrote five CSVs and left the other five as they were, with
    `--dry-run all` reporting nothing to do over tracked tables that no longer
    matched the code beside them. They cost one more pass over the same scored
    table, which is how they stay a diff rather than a fossil.
    """
    input:
        [
            str(path)
            for selection in SELECTIONS
            for path in composed_paths(selection)
        ],
        # `fig:within_clade_shifts`, and so its six panels beneath it. The
        # published selection only: its expensive half is the haplotype
        # scoring, which `fidelity` would triple for a comparison no panel
        # reads -- unlike `fig:agreement`, where a second selection is one
        # more pass over a table already in hand.
        [
            str(path)
            for path in composed_paths(DEFAULT_SELECTION, SHIFT_FIGURE_STEM)
        ],
        # `fig:subsampling`, and so its four panels. One panel set and no
        # selection to expand over: `SUBSAMPLING_SELECTION` is its directory.
        [
            str(path)
            for path in composed_paths(SUBSAMPLING_SELECTION, SUBSAMPLING_FIGURE_STEM)
        ],
        # `fig:dms`, on the same footing: one panel set, one directory.
        [str(path) for path in composed_paths(DMS_SELECTION, DMS_FIGURE_STEM)],
        # `fig:dms_strata`, which shares that directory: it is the same
        # selection drawn from the same model, two panels rather than four.
        [
            str(path)
            for path in composed_paths(DMS_SELECTION, DMS_STRATA_FIGURE_STEM)
        ],


rule background_effects:
    """`fig:within_clade_shifts` panels B, D and F: one row per clade.

    Its own rule rather than a cell of `rule pool_counts`, which streams the
    corpus twice: this reads two small tables and draws, so the layout can be
    nudged and re-run in a second instead of re-tallying every pool to move a
    panel.
    """
    wildcard_constraints:
        selection="|".join(SELECTIONS),
    input:
        panels="scripts/panels.py",
        spec="scripts/panel_spec.py",
        charts="scripts/panel_charts.py",
        founder_reader="scripts/founders.py",
        lib="scripts/lib.py",
        config="config.yaml",
        analysis_charts="dasm2/analysis/charts.py",
        analysis_viral="dasm2/analysis/viral.py",
        analysis_eval="dasm2/analysis/viral_fitness_eval.py",
        io="dasm2/io.py",
        conversion="dasm2/sequence/conversion.py",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        dasm2_init="dasm2/__init__.py",
        notebook="notebooks/background_effects.py",
        candidates=str(CANDIDATE_CSV),
        cells=str(POOL_CELLS),
        shifts=str(POOL_SHIFTS),
        clade_founders=clade_founder_fastas,
        masked_nt_sites=str(mat_pcp_path("masked_nt_sites_file")),
        reference=str(mat_pcp_path("reference_sequence_file")),
    output:
        html=f"{NOTEBOOK_DIR}/background_effects_{{selection}}.html",
        panels=panel_outputs(BACKGROUND_PANELS, "{selection}"),
    log:
        f"{ARTIFACT_ROOT}/logs/background_effects_{{selection}}.log",
    resources:
        mem_mb=PANEL_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " -- --cells {input.cells}"
        " --shifts {input.shifts}"
        " --selection {wildcards.selection}"
        " > {log} 2>&1"
