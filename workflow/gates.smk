"""M1's gates: prove the vendored code still trains the reference model.

Four of them, cheapest first, which is also most-informative-per-hour first.

- **Gate 0**, under a minute: the whole training path on 400 PCPs.
- **check_config**, seconds: the rendered config, and the three library
  defaults no config file states that together decide where a run stops.
- **Gate 1**, minutes: score the reference checkpoint. The strongest check
  here and it costs no GPU-hours -- one fixed-weight pass exercises the
  Parquet backend, the split, all three censoring sources, the transform
  chain, the encoder, the SF projection, the head, the loss and the
  denominator.
- **Gate 2**, hours: the 20,000-step canary against the reference's first ten
  evals.

Gate 3, the run to the LR floor, is roughly 40 hours at flash-off rates and
lands separately.

**GPU selection is the caller's.** These rules take whatever
``CUDA_VISIBLE_DEVICES`` the environment gives them; none of them picks a card
or takes a lock. That is adequate while one gate runs at a time, which is the
case here -- Gate 1 is a single pass of a few minutes. M3 schedules nine
training runs in parallel and needs real dispatch; it should not inherit this
as though it were a decision.
"""

GATE_DIR = f"{ARTIFACT_ROOT}/gates"
CONFIG_DIR = f"{ARTIFACT_ROOT}/configs"

#: Every vendored source file, as an input to the rules that execute it.
#:
#: Without this the DAG has no edge from the code to the gates: deleting a
#: module leaves every output newer than every declared input, so ``snakemake
#: gates`` answers "nothing to be done" and a prune looks safe because nothing
#: ran. Issue #4 deletes 91 files, and this is what makes each batch checkable.
DASM2_SOURCES = sorted(
    str(path.relative_to(workflow.basedir))
    for path in (Path(workflow.basedir) / "dasm2").rglob("*.py")
)

#: The env every training and scoring rule needs.
#:
#: ``DASM2_PARQUET_CACHE_BUDGET_GB`` because the library's 2 GB default caps
#: the row-group cache below the shuffle buffer's working set on this corpus,
#: and the failure is silent: the run trains ~34x too slowly to finish. The
#: thread caps because without them each dataloader worker sizes its BLAS
#: pools from the host core count.
#: ``PYTHONUNBUFFERED`` because a rule that redirects stdout to a file gets
#: 8 KB block buffering, and a run of several hours then shows nothing for
#: its first hour. The trainer's own JSONL log is written per evaluation
#: either way, but the human-readable log should not lag it.
TRAIN_ENV = (
    "PYTHONUNBUFFERED=1"
    " OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
    " NUMEXPR_NUM_THREADS=1"
    f" DASM2_PARQUET_CACHE_BUDGET_GB={FULL_DEPTH.parquet_cache_budget_gb}"
)


rule gates:
    """Every gate M1 lands with."""
    input:
        f"{GATE_DIR}/fast_test.done",
        f"{GATE_DIR}/check_config_canary.json",
        f"{GATE_DIR}/gate1_{CANARY_MODEL}.json",
        f"{ARTIFACT_ROOT}/corpus_guard/full_depth.json",


rule render_config:
    """Fill a training template's paths from ``config.yaml``."""
    input:
        template="configs/{name}.template.yaml",
    output:
        config=f"{CONFIG_DIR}/{{name}}_{{budget}}.yaml",
    params:
        steps=lambda w: {"canary": CANARY_STEPS, "fast": 5, "full": 500000}[w.budget],
    shell:
        "python scripts/render_config.py"
        " --template {input.template}"
        " --output {output.config}"
        " --max-train-steps {params.steps}"


rule check_config:
    """The rendered config, and the three defaults it does not state."""
    input:
        config=f"{CONFIG_DIR}/{CANARY_MODEL}_canary.yaml",
    output:
        report=f"{GATE_DIR}/check_config_canary.json",
    shell:
        "python scripts/check_rendered_config.py"
        " --config {input.config}"
        " --report {output.report}"
        f" --expect-max-train-steps {CANARY_STEPS}"


rule fast_test:
    """Gate 0: five steps on 400 PCPs, under a minute.

    A wiring test, not a science run. It is the first thing to get green, and
    the only gate that does not need the full-depth corpus.
    """
    input:
        config=f"{CONFIG_DIR}/fast_test_fast.yaml",
        sources=DASM2_SOURCES,
    output:
        done=touch(f"{GATE_DIR}/fast_test.done"),
    params:
        outdir=f"{ARTIFACT_ROOT}/fast_test",
        parquet=str(FAST_TEST_PARQUET),
    log:
        f"{ARTIFACT_ROOT}/logs/fast_test.log",
    shell:
        "{TRAIN_ENV} python scripts/train.py"
        " --config {input.config}"
        " --parquet {params.parquet}"
        " --output-dir {params.outdir}"
        " --device cpu --no-resume"
        " > {log} 2>&1"


rule gate1_score_reference:
    """Gate 1: score the reference checkpoint in this tree.

    Six assertions, in the order the script learns them: split sizes, the
    censor banner, the state-dict key set and shapes against a model built
    from this repository's own config, the effective parameter count, the
    validation loss to 1e-4, and the loss-mask position count.
    """
    input:
        config=f"{CONFIG_DIR}/{CANARY_MODEL}_canary.yaml",
        corpus_guard=f"{ARTIFACT_ROOT}/corpus_guard/full_depth.json",
        sources=DASM2_SOURCES,
    output:
        report=f"{GATE_DIR}/gate1_{{model}}.json",
    log:
        f"{ARTIFACT_ROOT}/logs/gate1_{{model}}.log",
    shell:
        "{TRAIN_ENV} python scripts/gate1_score_reference.py"
        " --model {wildcards.model}"
        " --corpus full_depth"
        " --config {input.config}"
        " --report {output.report}"
        " --device cuda"
        " > {log} 2>&1"


rule canary:
    """Gate 2: train dasm-85k for 20,000 steps on the full-depth corpus.

    Ten evaluations at the reference's 2,000-step cadence, compared point by
    point against the curve `data/reference/` carries. A dropped censoring
    source, a lost mask, the wrong reference sequence, a fold off by one or a
    wrong `min_delta` all move the curve from the first evaluation.

    Roughly 2-2.5 hours at flash-off rates, against the reference's 50.6
    minutes with flash on. The real number is measured by `gate2` below and
    is itself a deliverable: it sets the nine-run budget in matsengrp/
    viral-dasm#1, which currently costs M3 at flash-on rates.

    `--seed` is a wildcard so the two-seed tolerance measurement is the same
    rule run twice, and each seed gets its own `output_dir` -- the resume
    pattern protects a run against dying, not against two attempts sharing a
    directory.
    """
    input:
        config=f"{CONFIG_DIR}/{CANARY_MODEL}_canary.yaml",
        corpus_guard=f"{ARTIFACT_ROOT}/corpus_guard/full_depth.json",
        config_check=f"{GATE_DIR}/check_config_canary.json",
    output:
        done=touch(f"{ARTIFACT_ROOT}/canary/seed{{seed}}/training_complete.done"),
    params:
        outdir=f"{ARTIFACT_ROOT}/canary/seed{{seed}}",
        parquet=str(FULL_DEPTH.path),
    log:
        f"{ARTIFACT_ROOT}/logs/canary_seed{{seed}}.log",
    shell:
        "{TRAIN_ENV} python scripts/train.py"
        " --config {input.config}"
        " --parquet {params.parquet}"
        " --output-dir {params.outdir}"
        " --seed {wildcards.seed}"
        " --device cuda"
        " > {log} 2>&1"


rule gate2:
    """Compare a canary's ten evaluations against the reference curve.

    Without `--tolerance` this reports the deltas and checks only what does
    not depend on one: ten rows at the right steps, and a learning rate still
    at 1e-3. The tolerance comes from running `canary` at two seeds and
    taking the per-row spread, which has to be written down before the full
    run is launched rather than fitted to its result.
    """
    input:
        done=f"{ARTIFACT_ROOT}/canary/seed{{seed}}/training_complete.done",
        reference=f"data/reference/{CANARY_MODEL}_rep0_eval_through_20000.csv",
    output:
        report=f"{GATE_DIR}/gate2_seed{{seed}}.json",
    params:
        canary_dir=f"{ARTIFACT_ROOT}/canary/seed{{seed}}",
        tolerance=lambda w: (
            f"--tolerance {config['tolerance']}" if "tolerance" in config else ""
        ),
    log:
        f"{ARTIFACT_ROOT}/logs/gate2_seed{{seed}}.log",
    shell:
        "python scripts/gate2_compare_curve.py"
        " --canary-dir {params.canary_dir}"
        " --reference-csv {input.reference}"
        " --report {output.report}"
        " {params.tolerance}"
        " > {log} 2>&1"


rule full_run:
    """Gate 3: train dasm-85k to the learning-rate floor.

    The reference stopped at step 436,000 with its best validation loss
    7.766710 at step 412,000. It stopped on the LR floor, not on patience:
    1e-3 x 0.5^10 = 9.77e-7 < 1e-6, and patience 30 never fires.

    Roughly 33 hours at the rate the first canary measured -- 20,000 steps in
    1.51 h, of which 31% was evaluation. `scripts/train.py` resumes from its
    own latest checkpoint, so a preemption or a full disk costs the time since
    the last save rather than the whole run.
    """
    input:
        config=f"{CONFIG_DIR}/{CANARY_MODEL}_full.yaml",
        corpus_guard=f"{ARTIFACT_ROOT}/corpus_guard/full_depth.json",
        config_check=f"{GATE_DIR}/check_config_canary.json",
    output:
        done=touch(f"{ARTIFACT_ROOT}/full_run/seed{{seed}}/training_complete.done"),
    params:
        outdir=f"{ARTIFACT_ROOT}/full_run/seed{{seed}}",
        parquet=str(FULL_DEPTH.path),
    log:
        f"{ARTIFACT_ROOT}/logs/full_run_seed{{seed}}.log",
    shell:
        "{TRAIN_ENV} python scripts/train.py"
        " --config {input.config}"
        " --parquet {params.parquet}"
        " --output-dir {params.outdir}"
        " --seed {wildcards.seed}"
        " --device cuda"
        " > {log} 2>&1"


rule gate3:
    """Compare the full run's endpoint against the reference's.

    Gates the stop *reason* -- reaching the LR floor is a structural fact
    about the schedule -- and reports the losses for judgement rather than
    thresholding them. The two canaries tracked the reference to within 0.012
    nats with a wider seed-to-seed spread than that, so a loss threshold here
    would be measuring the seed rather than the port.
    """
    input:
        done=f"{ARTIFACT_ROOT}/full_run/seed{{seed}}/training_complete.done",
    output:
        report=f"{GATE_DIR}/gate3_seed{{seed}}.json",
    params:
        run_dir=f"{ARTIFACT_ROOT}/full_run/seed{{seed}}",
    log:
        f"{ARTIFACT_ROOT}/logs/gate3_seed{{seed}}.log",
    shell:
        "python scripts/gate3_compare_endpoint.py"
        " --run-dir {params.run_dir}"
        f" --model {CANARY_MODEL}"
        " --report {output.report}"
        " > {log} 2>&1"
