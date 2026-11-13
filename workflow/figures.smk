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

# Every path below comes from `scripts/panel_spec.py`, which is what the
# notebooks compute them from too: one spelling per fact, so a rule cannot
# declare an output the notebook writes somewhere else. The spec module is
# stdlib plus `lib`, deliberately: this file is parsed on every snakemake
# invocation, and importing the panel code here would pull `dasm2`, pandas and
# pyarrow into a dry run to learn five file names.
from panel_spec import (
    DEFAULT_SELECTION,
    LADDER_PANELS,
    MUTATION_PANELS,
    PANEL_LETTERS,
    PANEL_MEM_MB,
    SELECTIONS,
    SHIFT_CLADES,
    composed_paths,
    figure_dir,
    panel_csv_path,
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


rule panels:
    """Every panel of `fig:agreement`, for the model the paper publishes."""
    input:
        panel_outputs(MUTATION_PANELS, DEFAULT_SELECTION),
        panel_outputs(LADDER_PANELS, DEFAULT_SELECTION),


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


rule figures:
    """Every composed figure of the paper, and so every panel beneath them."""
    input:
        [str(path) for path in composed_paths(DEFAULT_SELECTION)],
