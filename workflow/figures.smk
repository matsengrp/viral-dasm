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

from panels import (
    LADDER_PANELS,
    MUTATION_PANELS,
    DEFAULT_SELECTION,
    SELECTIONS,
)

FIGURE_DIR = f"{ARTIFACT_ROOT}/figures"
NOTEBOOK_DIR = f"{ARTIFACT_ROOT}/notebooks"
PANEL_CSV_DIR = "results/panels"

#: The censor audit, which is where the reverting exclusion comes from. The
#: table carries `censored` and `globally_masked` as columns and does not carry
#: this -- `censored_directional` is the censor table's directional entries and
#: a subset of `censored` by construction, so a panel that substituted it would
#: run a two-layer exclusion while believing it ran three.
AUDIT_JSON = f"{ARTIFACT_ROOT}/audit/{GENE}.json"


def panel_outputs(stems, selection):
    """One selection's panel images and tables, as a rule's outputs."""
    return [f"{FIGURE_DIR}/{selection}/{stem}.png" for stem in stems] + [
        f"{PANEL_CSV_DIR}/{selection}/{stem}.csv" for stem in stems
    ]


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
        # The founders the shift panel drops identical pairs on. Globbed, as
        # `rule score` globs them: the roster is whatever mat-pcp wrote.
        clade_founders=founder_fastas,
    output:
        html=f"{NOTEBOOK_DIR}/agreement_mutations_{{selection}}.html",
        panels=panel_outputs(MUTATION_PANELS, "{selection}"),
    log:
        f"{ARTIFACT_ROOT}/logs/agreement_mutations_{{selection}}.log",
    resources:
        mem_mb=SCORE_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " -- --table {input.table} --selection {wildcards.selection}"
        " > {log} 2>&1"


rule agreement_ladders:
    """Panels C and E: both correlations against the count cutoff."""
    wildcard_constraints:
        selection="|".join(SELECTIONS),
    input:
        panels="scripts/panels.py",
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
        clade_founders=founder_fastas,
    output:
        html=f"{NOTEBOOK_DIR}/agreement_ladders_{{selection}}.html",
        panels=panel_outputs(LADDER_PANELS, "{selection}"),
    log:
        f"{ARTIFACT_ROOT}/logs/agreement_ladders_{{selection}}.log",
    resources:
        mem_mb=SCORE_MEM_MB,
    shell:
        "marimo export html {input.notebook} -o {output.html} --force"
        " -- --table {input.table} --selection {wildcards.selection}"
        " > {log} 2>&1"
