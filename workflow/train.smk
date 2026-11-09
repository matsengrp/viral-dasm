"""The seven models the paper reports, and each run's stopping point.

**Seven ``(model, corpus)`` pairs, four architectures, one stopping rule.**
Three transformer sizes and the context-independent baseline at full depth,
where the size ladder and the pooling contrast are read; ``dasm-85k`` alone at
each of the three thinned depths, because that is the architecture the
sampling-depth ladder trained at every rung. The list is in
``scripts/runs.py``; it is seven named pairs rather than the cross product, so
``models`` enumerates them and an input function raises on anything else.

**Every run trains to the learning-rate floor.** ``max_train_steps`` is a
budget guard, and a run that reaches it instead leaves *no* stop event --
``trainer.py`` breaks on the budget before ``_should_stop_early`` is called.
``read_run`` writes the absence down and ``check_run`` fails on it, so a
capped arm stops its own branch of the DAG with its curve in hand while the
other six finish.

**One run per card, claimed by the training process.** Not for memory:
attention runs through ``scaled_dot_product_attention`` with
``need_weights=False``, so no ``L x L`` matrix is materialised. For
throughput, and for leaving cards visibly free. ``scripts/train.py`` takes
the lock itself, so its lifetime is the run's lifetime.

**``train`` does not declare ``tests_passed.done``**, and that is deliberate
rather than an oversight against #12's policy: an edit to any test file would
otherwise invalidate a finished multi-day run, and ``validate_tests`` is the
slow gate now that it imports torch. The suite still runs ahead of the
corpora, which is where a regression in the corpus builder would do its
damage. Every other rule here declares the code it runs, as #12 requires.

GISAID: every run directory, rendered config and log under ``ARTIFACT_ROOT``
is GISAID-derived and stays on the cluster. ``data/runs.csv`` is aggregate
per-run summary statistics and is tracked.
"""

from runs import (  # noqa: E402
    EVAL_EVERY_STEPS,
    MAX_TRAIN_STEPS,
    MODEL_NAMES,
    RUNS,
    SMOKE_EVAL_EVERY_STEPS,
    SMOKE_MEM_MB,
    SMOKE_STEPS,
    TRAIN_MEM_MB,
    run_id,
)

CONFIG_DIR = f"{ARTIFACT_ROOT}/configs"
RUN_DIR = f"{ARTIFACT_ROOT}/runs"
SMOKE_DIR = f"{ARTIFACT_ROOT}/smoke"
READOUT_DIR = f"{ARTIFACT_ROOT}/readout"

#: The committed table every run lands one row in.
RUNS_CSV = "data/runs.csv"

#: The env every training rule needs.
#:
#: ``DASM2_PARQUET_CACHE_BUDGET_GB`` because the library's 2 GB default caps
#: the row-group cache below the shuffle buffer's working set on these
#: corpora, and the failure is silent: the run trains ~34x too slowly to
#: finish. It is per corpus -- computed from each file's own row groups, which
#: is why ``cap_5000_0`` lands lower than the rest -- so it is built in
#: ``params`` rather than pinned here.
#:
#: The thread caps because without them each dataloader worker sizes its BLAS
#: pools from the host core count. ``PYTHONUNBUFFERED`` because these rules
#: redirect stdout to a log, which otherwise gets 8 KB block buffering and
#: shows nothing for the first hour of a multi-day run; the trainer's own
#: JSONL is written per evaluation either way.
TRAIN_ENV = (
    "PYTHONUNBUFFERED=1"
    " OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
    " NUMEXPR_NUM_THREADS=1"
)

#: The head each architecture's config declares. ``train.py`` keys the data
#: dict by it, so a mismatch loads a dataset for a head the model does not
#: have.
HEAD_OF = {
    "dasm-1077k": "SelectionFactorHead",
    "dasm-293k": "SelectionFactorHead",
    "dasm-85k": "SelectionFactorHead",
    "ci-484k": "PerSiteParentChildHead",
}

#: The one template that asks for the corpus's amino-acid length, because its
#: head sizes a parameter table from it. Supplying it elsewhere is a
#: ``KeyError`` from ``render_config.render``, by design.
NEEDS_MAX_AA_SEQ_LEN = {"ci-484k"}


def max_aa_seq_len_flag(model, parquet):
    """``--max-aa-seq-len`` for the one model that needs it, else nothing."""
    if model not in NEEDS_MAX_AA_SEQ_LEN:
        return ""
    return f" --max-aa-seq-len {corpus_max_aa_seq_len(parquet)}"


def require_run(model, corpus_name):
    """Refuse a ``(model, corpus)`` pair that is not one of the seven.

    The wildcard constraints below are per wildcard, so they still admit
    `ci-484k` x `cap_5000_0` -- a pair nothing in the paper reports and which
    would cost days to discover was never wanted.
    """
    if (model, corpus_name) not in RUNS:
        raise ValueError(
            f"({model!r}, {corpus_name!r}) is not one of the seven runs. The "
            f"run list in scripts/runs.py is seven named pairs, not the cross "
            f"product of four models and four corpora: the three larger "
            f"architectures run at full depth only."
        )
    return run_id(model, corpus_name)


wildcard_constraints:
    # Required, not optional: corpus names contain underscores, so
    # `{model}_{corpus}` is ambiguous without pinning the model alternatives.
    model="|".join(MODEL_NAMES),


rule models:
    """The seven runs, each checked and each a row in ``data/runs.csv``.

    **The smokes are siblings here, not a precondition of ``train``.** Every
    input of ``smoke`` but two is already a direct input of ``train``, so the
    edge caught no code drift ``train`` misses; what it did do was put a
    ``touch()`` output upstream of a multi-day run, so re-pinning the smoke
    corpus would restart one. That is the hazard this file's docstring
    refuses for ``tests_passed.done``, and the same argument applies. Asking
    for them here keeps them in the default DAG -- ``models`` does not
    succeed until all four pass -- while letting them run on CPU alongside
    the GPU jobs rather than ahead of them.
    """
    input:
        expand(
            f"{ARTIFACT_ROOT}/run_guard/{{run}}.json",
            run=[run_id(m, c) for m, c in RUNS],
        ),
        expand(f"{SMOKE_DIR}/{{model}}/smoke.done", model=MODEL_NAMES),
        RUNS_CSV,


rule render_config:
    """Fill one model's template for one corpus, from ``config.yaml``.

    Per ``(model, corpus)`` rather than per model: ``ci-484k``'s
    ``max_aa_seq_len`` is read from the corpus's own Parquet footer, so the
    rendered file depends on both.
    """
    input:
        script="scripts/render_config.py",
        template="configs/{model}.template.yaml",
        config="config.yaml",
        lib="scripts/lib.py",
        # The budget and the cadence this config is rendered with live there.
        runs="scripts/runs.py",
        corpus_guard=f"{ARTIFACT_ROOT}/corpus_guard/{{corpus}}.json",
    output:
        config=f"{CONFIG_DIR}/{{model}}_{{corpus}}.yaml",
    params:
        # Reads one integer out of the corpus's footer; touches no rows.
        max_aa=lambda wc: max_aa_seq_len_flag(wc.model, corpus(wc.corpus).path),
        run=lambda wc: require_run(wc.model, wc.corpus),
        steps=MAX_TRAIN_STEPS,
        cadence=EVAL_EVERY_STEPS,
    resources:
        mem_mb=1000,
    shell:
        # Written as one plain string, with every value through `params`:
        # `tests/test_rule_inputs.py` reads the command as a literal to find
        # the script it starts, and an f-string here makes the whole command
        # invisible to it.
        "python {input.script}"
        " --template {input.template}"
        " --output {output.config}"
        " --max-train-steps {params.steps}"
        " --eval-every-steps {params.cadence}"
        "{params.max_aa}"


rule check_params:
    """Build the model and check its effective count against its own name.

    CPU, seconds. This is what catches a template transcribed from the wrong
    arm -- the realistic way to train the wrong model for five days -- and it
    also makes the censor-path assertions the deleted
    ``check_rendered_config.py`` made, which ``render_config.py`` does not:
    that script checks the five paths exist, not that they are the ones
    ``config.yaml`` names.
    """
    input:
        script="scripts/check_params.py",
        config_yaml=f"{CONFIG_DIR}/{{model}}_{{corpus}}.yaml",
        config="config.yaml",
        lib="scripts/lib.py",
        runs="scripts/runs.py",
        dasm2_config="dasm2/config.py",
        factory="dasm2/models/factory.py",
        constants="dasm2/sequence/constants.py",
        dasm2_init="dasm2/__init__.py",
    output:
        report=f"{ARTIFACT_ROOT}/param_guard/{{model}}_{{corpus}}.json",
    log:
        f"{ARTIFACT_ROOT}/logs/check_params_{{model}}_{{corpus}}.log",
    resources:
        mem_mb=4000,
    shell:
        "python {input.script}"
        " --model {wildcards.model}"
        " --corpus {wildcards.corpus}"
        " --config {input.config_yaml}"
        " --report {output.report}"
        " > {log} 2>&1"


rule smoke:
    """Four steps of one architecture on 400 PCPs, on CPU.

    A wiring test, not a science run: config to dataset to tokenisation at
    the gene's real length to censor load to forward to backward to
    evaluation. Four architectures now need covering rather than one, which
    is what this rule is for; it renders its own config at a cadence low
    enough that a four-step cycle contains an evaluation.

    The smoke corpus is pinned rather than built, and sets no
    ``DASM2_PARQUET_CACHE_BUDGET_GB``: at 866 KB the library's own default
    covers it many times over.
    """
    input:
        script="scripts/train.py",
        config=f"{CONFIG_DIR}/smoke_{{model}}.yaml",
        gpu_slot="scripts/gpu_slot.py",
        lib="scripts/lib.py",
        config_yaml="config.yaml",
        dasm2_config="dasm2/config.py",
        train_command="dasm2/train_command.py",
        dasm2_init="dasm2/__init__.py",
        # Not `ancient()`, unlike the training corpora: at 866 KB this file
        # is under snakemake's 1 MB checksum threshold, so its content is
        # tracked and a changed smoke corpus reruns the smoke rather than
        # passing on a stale marker. `config.yaml` records its md5.
        parquet=str(SMOKE_CORPUS),
    output:
        done=touch(f"{SMOKE_DIR}/{{model}}/smoke.done"),
    params:
        outdir=f"{SMOKE_DIR}/{{model}}",
        head=lambda wc: HEAD_OF[wc.model],
    log:
        f"{ARTIFACT_ROOT}/logs/smoke_{{model}}.log",
    resources:
        mem_mb=SMOKE_MEM_MB,
    shell:
        "{TRAIN_ENV} python {input.script}"
        " --config {input.config}"
        " --parquet {input.parquet}"
        " --output-dir {params.outdir}"
        " --head {params.head}"
        " --device cpu --no-resume"
        " > {log} 2>&1"


rule render_smoke_config:
    """The smoke's own config: four steps, and a cadence that fits in them.

    ``TrainingConfig`` rejects ``max_train_steps < eval_every_steps`` -- "a
    cycle must contain at least one evaluation" -- which is the whole reason
    the cadence is a template placeholder rather than a literal 2,000.
    """
    input:
        script="scripts/render_config.py",
        template="configs/{model}.template.yaml",
        config="config.yaml",
        lib="scripts/lib.py",
        runs="scripts/runs.py",
    output:
        config=f"{CONFIG_DIR}/smoke_{{model}}.yaml",
    params:
        max_aa=lambda wc: max_aa_seq_len_flag(wc.model, SMOKE_CORPUS),
        steps=SMOKE_STEPS,
        cadence=SMOKE_EVAL_EVERY_STEPS,
    resources:
        mem_mb=1000,
    shell:
        "python {input.script}"
        " --template {input.template}"
        " --output {output.config}"
        " --max-train-steps {params.steps}"
        " --eval-every-steps {params.cadence}"
        "{params.max_aa}"


rule train:
    """Train one model on one corpus, to the learning-rate floor.

    **``ancient()`` on the data, not on the guard.** The corpus and the three
    censor tables are all far above snakemake's 1 MB checksum threshold, so
    only their mtime is compared, and mat-pcp re-emits its tree in place: a
    byte-identical re-emit must not restart a multi-day run. What catches a
    genuinely changed corpus is ``corpus_guard/{corpus}.json``, declared
    un-ancient and small enough to be content-tracked. The rendered config is
    small and content-tracked too, and a change to it *should* retrain.

    ``tests_passed.done`` is deliberately absent; see this file's docstring.
    """
    input:
        script="scripts/train.py",
        gpu_slot="scripts/gpu_slot.py",
        lib="scripts/lib.py",
        config="config.yaml",
        dasm2_config="dasm2/config.py",
        train_command="dasm2/train_command.py",
        dasm2_init="dasm2/__init__.py",
        rendered=f"{CONFIG_DIR}/{{model}}_{{corpus}}.yaml",
        params_checked=f"{ARTIFACT_ROOT}/param_guard/{{model}}_{{corpus}}.json",
        corpus_guard=f"{ARTIFACT_ROOT}/corpus_guard/{{corpus}}.json",
        parquet=lambda wc: ancient(str(corpus(wc.corpus).path)),
        censored_mutations=ancient(str(mat_pcp_path("censored_mutations_file"))),
        censored_clades=ancient(str(mat_pcp_path("censored_clades_file"))),
        masked_nt_sites=ancient(str(mat_pcp_path("masked_nt_sites_file"))),
        reference_sequence=ancient(str(mat_pcp_path("reference_sequence_file"))),
    output:
        done=touch(f"{RUN_DIR}/{{model}}_{{corpus}}/training_complete.done"),
    params:
        outdir=f"{RUN_DIR}/{{model}}_{{corpus}}",
        head=lambda wc: HEAD_OF[wc.model],
        cache=lambda wc: (
            "DASM2_PARQUET_CACHE_BUDGET_GB="
            f"{corpus(wc.corpus).parquet_cache_budget_gb}"
        ),
        run=lambda wc: require_run(wc.model, wc.corpus),
    log:
        f"{ARTIFACT_ROOT}/logs/train_{{model}}_{{corpus}}.log",
    resources:
        mem_mb=TRAIN_MEM_MB,
        gpu=1,
    shell:
        "{TRAIN_ENV} {params.cache} python {input.script}"
        " --config {input.rendered}"
        " --parquet {input.parquet}"
        " --output-dir {params.outdir}"
        " --head {params.head}"
        " --device cuda"
        # Appended, not truncated: `train.py` resumes from its latest
        # checkpoint, and `>` would discard everything the killed attempt
        # printed. The trainer's own JSONL is per attempt and unaffected.
        " >> {log} 2>&1"


rule read_run:
    """One finished run, as one row, entirely from its own log."""
    wildcard_constraints:
        corpus="|".join(CORPUS_NAMES),
    input:
        script="scripts/read_run.py",
        lib="scripts/lib.py",
        runs="scripts/runs.py",
        config="config.yaml",
        log_reader="dasm2/training/log_reader.py",
        dasm2_init="dasm2/__init__.py",
        done=f"{RUN_DIR}/{{model}}_{{corpus}}/training_complete.done",
    output:
        row=f"{READOUT_DIR}/{{model}}_{{corpus}}.json",
    params:
        outdir=f"{RUN_DIR}/{{model}}_{{corpus}}",
    log:
        f"{ARTIFACT_ROOT}/logs/read_run_{{model}}_{{corpus}}.log",
    resources:
        mem_mb=8000,
    shell:
        "python {input.script}"
        " --model {wildcards.model}"
        " --corpus {wildcards.corpus}"
        " --run-dir {params.outdir}"
        " --output {output.row}"
        " > {log} 2>&1"


rule check_run:
    """Fail a run that stopped anywhere but the learning-rate floor.

    Downstream of ``read_run`` rather than merged into it, so the row survives
    a failure: a capped or patience-stopped arm is then readable while the
    DAG refuses to carry it into a figure.
    """
    input:
        script="scripts/check_run.py",
        runs="scripts/runs.py",
        row=f"{READOUT_DIR}/{{model}}_{{corpus}}.json",
    output:
        report=f"{ARTIFACT_ROOT}/run_guard/{{model}}_{{corpus}}.json",
    log:
        f"{ARTIFACT_ROOT}/logs/check_run_{{model}}_{{corpus}}.log",
    resources:
        mem_mb=1000,
    shell:
        "python {input.script}"
        " --row {input.row}"
        " --report {output.report}"
        " > {log} 2>&1"


rule train_eval:
    """Score one run's best checkpoint on its own training split, dropout off.

    The trainer's ``avg_train_loss`` is a running mean taken with dropout
    active and with the weights moving across the averaging window, so the
    gap between it and ``val_loss`` mixes a generalization gap with a dropout
    penalty. This recomputes the training side the way the validation side is
    already computed -- ``module.eval()``, a full deterministic pass, the
    trainer's own denominator -- which is what lets a larger model fitting
    its *training* rows worse be read as an optimization failure rather than
    a capacity limit.

    Downstream of ``train`` and declared separately from ``read_run``, whose
    contract is that every number in it came out of the run's own log. This
    one is a second pass over the corpus, so it gets its own file and
    ``runs_table`` joins the two.

    Takes a GPU: a full training pass is several times a validation pass.
    """
    wildcard_constraints:
        corpus="|".join(CORPUS_NAMES),
    input:
        script="scripts/train_eval.py",
        eval_common="scripts/eval_common.py",
        gpu_slot="scripts/gpu_slot.py",
        lib="scripts/lib.py",
        config="config.yaml",
        dasm2_config="dasm2/config.py",
        checkpoints="dasm2/serialization/checkpoints.py",
        log_reader="dasm2/training/log_reader.py",
        dasm2_init="dasm2/__init__.py",
        # What `eval_common` runs. Not reached by the entry-point check,
        # which reads one file's imports, but this rule runs them all the
        # same and a change to any of them changes the number it writes.
        evolutionary_dataset="dasm2/data/evolutionary_dataset.py",
        head_types="dasm2/heads/types.py",
        whichmut_losses="dasm2/losses/whichmut_losses.py",
        train_command="dasm2/train_command.py",
        rendered=f"{CONFIG_DIR}/{{model}}_{{corpus}}.yaml",
        done=f"{RUN_DIR}/{{model}}_{{corpus}}/training_complete.done",
        parquet=lambda wc: ancient(str(corpus(wc.corpus).path)),
    output:
        row=f"{READOUT_DIR}/{{model}}_{{corpus}}_train_eval.json",
    params:
        outdir=f"{RUN_DIR}/{{model}}_{{corpus}}",
        cache=lambda wc: (
            "DASM2_PARQUET_CACHE_BUDGET_GB="
            f"{corpus(wc.corpus).parquet_cache_budget_gb}"
        ),
    log:
        f"{ARTIFACT_ROOT}/logs/train_eval_{{model}}_{{corpus}}.log",
    resources:
        mem_mb=TRAIN_MEM_MB,
        gpu=1,
    shell:
        "{TRAIN_ENV} {params.cache} python {input.script}"
        " --model {wildcards.model}"
        " --corpus {wildcards.corpus}"
        " --run-dir {params.outdir}"
        " --config {input.rendered}"
        " --parquet {input.parquet}"
        " --output {output.row}"
        " --device cuda"
        " > {log} 2>&1"


rule runs_table:
    """Gather the seven rows into the one table this repository commits."""
    input:
        script="scripts/runs_table.py",
        runs="scripts/runs.py",
        rows=expand(
            f"{READOUT_DIR}/{{run}}.json", run=[run_id(m, c) for m, c in RUNS]
        ),
        train_eval=expand(
            f"{READOUT_DIR}/{{run}}_train_eval.json",
            run=[run_id(m, c) for m, c in RUNS],
        ),
        # The gate, not just the row. Without this a run that stopped on the
        # step budget still reaches `data/runs.csv`, which is tracked, and
        # only `models` would notice afterwards.
        guards=expand(
            f"{ARTIFACT_ROOT}/run_guard/{{run}}.json",
            run=[run_id(m, c) for m, c in RUNS],
        ),
    output:
        csv=RUNS_CSV,
    params:
        rows_dir=READOUT_DIR,
    resources:
        mem_mb=1000,
    shell:
        "python {input.script}"
        " --rows-dir {params.rows_dir}"
        " --output {output.csv}"
