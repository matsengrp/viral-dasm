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

#: The env every training and scoring rule needs.
#:
#: ``DASM2_PARQUET_CACHE_BUDGET_GB`` because the library's 2 GB default caps
#: the row-group cache below the shuffle buffer's working set on this corpus,
#: and the failure is silent: the run trains ~34x too slowly to finish. The
#: thread caps because without them each dataloader worker sizes its BLAS
#: pools from the host core count.
TRAIN_ENV = (
    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
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
