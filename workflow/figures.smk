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

# `re` is imported here rather than taken from the Snakefile's namespace: this
# file is included, so relying on an includer's import makes it depend on
# include order for something it uses in a wildcard constraint.
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
# Four rules rather than two. The leaf tally is the expensive one -- clade 21K
# is 1.9M leaves and 1.7 GB of paths -- and it does not depend on BLOSUM62, so
# folding both predictors into one rule would rebuild it on every edit to the
# substitution matrix.
#
# Nothing here reads `selection_factors.parquet`: these are the part of the DMS
# comparison that needs no trained model, which is why they can land before #16.
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


rule dms_predictors:
    """The DMS predictor comparison, as far as it is built without a model.

    The two reference predictors and the join they land through. **A table and
    no panel:** `fig:dms`'s four drawn panels are the published figure's, which
    `rule dms` builds, and this comparison has no drawn home until the DASM's
    own column lands -- a filter on `selection_factors.parquet`, so it arrives
    with `rule score`'s table. A target that demanded the chart now would be red
    for the whole gap.
    """
    input:
        str(panel_csv_path(DMS_SELECTION, DMS_PREDICTOR_CORRELATIONS)),


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
        clade_founders=clade_founder_fastas,
        censored=str(mat_pcp_path("censored_mutations_file")),
        clades=str(mat_pcp_path("censored_clades_file")),
        masked=str(mat_pcp_path("masked_nt_sites_file")),
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
        leaf_freq="scripts/dms_leaf_freq.py",
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
        clade_founders=clade_founder_fastas,
        censored=str(mat_pcp_path("censored_mutations_file")),
        clades=str(mat_pcp_path("censored_clades_file")),
        masked=str(mat_pcp_path("masked_nt_sites_file")),
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
