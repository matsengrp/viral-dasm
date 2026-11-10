"""Turn the seven finished runs into one tidy selection-factor table.

Each run's best checkpoint is scored against mat-pcp's amino-acid fitness
effects on all 56 clade founders, and the seven results are concatenated. The
table is written raw -- unclipped, no count gate, no clade roster applied --
because all three are swept axes in the figures, and applying one here would
force a rescore to move it.

Downstream of ``train`` and independent of ``read_run`` and ``train_eval``:
those two report where a run *ended up*, this one reports what it *predicts*.
A run is scored from its ``training_complete.done`` marker rather than from a
checkpoint path, because ``lib.select_checkpoint`` resolves the checkpoint by
glob and no rule declares one as an output.

GISAID-derived: everything here is a per-site slice of GISAID counts joined to
model predictions, so it stays under the artifact root. Nothing lands in
tracked ``data/``.
"""

SCORED_DIR = f"{ARTIFACT_ROOT}/scored"

#: The one table the figures read.
SCORED_TABLE = f"{SCORED_DIR}/selection_factors.parquet"

#: The clade guard's report. Phase-gates every scoring job: the roster is read
#: out of mat-pcp's directories, so a run scored before the three readings are
#: known to agree could silently cover fewer clades than the run has.
CLADE_GUARD = f"{ARTIFACT_ROOT}/clade_guard/clades.json"


rule scored:
    """The tidy table, with the clade roster checked first."""
    input:
        guard=CLADE_GUARD,
        table=SCORED_TABLE,


rule check_clades:
    """Require mat-pcp's three clade rosters to name the same clades.

    A mismatch does not error downstream -- a clade with a founder and no
    fitness file scores to an all-null join and reads as a clade the model got
    wrong, while a clade with fitness and no founder never enters the grid.
    Both give a full-looking table with fewer clades in it than the run has.
    """
    input:
        script="scripts/check_clades.py",
        lib="scripts/lib.py",
        config="config.yaml",
        clades=mat_pcp_path("censored_clades_file"),
        pooled=mat_pcp_path("aamut_fitness_pooled_file"),
    output:
        report=CLADE_GUARD,
    log:
        f"{ARTIFACT_ROOT}/logs/check_clades.log",
    resources:
        mem_mb=1000,
    shell:
        "python {input.script} --report {output.report} > {log} 2>&1"


rule score:
    """Score one run's best checkpoint on all 56 clade founders.

    One forward pass per clade, not per mutation: a selection factor is a
    property of the founder sequence, so a clade's whole ``[L, 20]`` table
    comes out of a single pass. 56 passes per run, 392 across the seven.

    Takes a GPU, and the same lock directory ``train`` uses, so a scoring pass
    and a live training run can never both hold one card.
    """
    wildcard_constraints:
        corpus="|".join(CORPUS_NAMES),
    input:
        script="scripts/score_run.py",
        gpu_slot="scripts/gpu_slot.py",
        lib="scripts/lib.py",
        config="config.yaml",
        guard=CLADE_GUARD,
        analysis_viral="dasm2/analysis/viral.py",
        analysis_eval="dasm2/analysis/viral_fitness_eval.py",
        checkpoints="dasm2/serialization/checkpoints.py",
        masking="dasm2/sequence/masking.py",
        conversion="dasm2/sequence/conversion.py",
        io="dasm2/io.py",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        dasm2_init="dasm2/__init__.py",
        # The marker, not the checkpoint: `lib.select_checkpoint` resolves the
        # file by glob and no rule declares one as an output.
        done=f"{RUN_DIR}/{{model}}_{{corpus}}/training_complete.done",
        fitness=mat_pcp_path("aamut_fitness_pooled_file"),
        censored=mat_pcp_path("censored_mutations_file"),
        masked=mat_pcp_path("masked_nt_sites_file"),
    output:
        table=f"{SCORED_DIR}/{{model}}_{{corpus}}.parquet",
    params:
        outdir=f"{RUN_DIR}/{{model}}_{{corpus}}",
    log:
        f"{ARTIFACT_ROOT}/logs/score_{{model}}_{{corpus}}.log",
    resources:
        mem_mb=SCORE_MEM_MB,
        gpu=1,
    shell:
        "{TRAIN_ENV} python {input.script}"
        " --model {wildcards.model}"
        " --corpus {wildcards.corpus}"
        " --run-dir {params.outdir}"
        " --output {output.table}"
        " --device cuda"
        " > {log} 2>&1"


rule scored_table:
    """Concatenate the seven per-run tables into the one the figures read."""
    input:
        script="scripts/scored_table.py",
        runs="scripts/runs.py",
        lib="scripts/lib.py",
        config="config.yaml",
        dasm2_init="dasm2/__init__.py",
        tables=expand(
            f"{SCORED_DIR}/{{run}}.parquet", run=[run_id(m, c) for m, c in RUNS]
        ),
    output:
        table=SCORED_TABLE,
    log:
        f"{ARTIFACT_ROOT}/logs/scored_table.log",
    resources:
        mem_mb=SCORE_MEM_MB,
    shell:
        "python {input.script}"
        " --inputs {input.tables}"
        " --output {output.table}"
        " > {log} 2>&1"
