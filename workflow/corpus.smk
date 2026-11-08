"""The corpora the pipeline trains on: built from mat-pcp, then identified.

``full_depth`` is pinned rather than built. The paper holds out fold 0 of a
10-fold ``cv_seed`` 42 split, and that split is a row-positional draw: the
index set comes from ``(seed, num_folds, n_rows)`` and which PCP sits at each
index comes from the file's row order. Rebuilding the file would reshuffle the
partition even if every row survived. See ``scripts/check_corpus.py``.

The three thinned arms are built here. Each is a mat-pcp aggregate in its own
right -- the thinning happens there, and nothing below filters, subsamples or
reshapes rows -- so a rebuild reproduces the source file's row order and with
it that corpus's fold 0. ``check_corpus`` is what says whether it did.

**One neutral model, every corpus.** mat-pcp does not refit its GLM per
subsampling arm, so ``build_neutral`` and ``render_era_map`` take no corpus
wildcard. That is what pins lambda across the ladder: a per-corpus rebuild
would let the neutral model move with the corpus, which is the one thing a
depth contrast cannot afford.

**Every rule that runs a script declares that script's code**: the
script itself, its direct imports of repository code, and ``config.yaml``
wherever ``scripts/lib.py`` is required, since ``lib`` reads and validates it
at import rather than importing it. Direct imports only -- a glob over
``dasm2`` would also make a comment edit anywhere in it invalidate the
corpora, and with them the training runs below. What covers the residual
transitive risk is ``check_corpus``, which gates each corpus's row-order
digest. ``tests/test_rule_inputs.py`` holds every rule to this.

Two consequences worth knowing. ``validate_tests`` is the exception: it runs
pytest over all of ``dasm2`` and declares none of it, because declaring that
*is* the glob above; it declares every test file instead and leans on the
suite's own assertions. And ``config.yaml`` is declared whole, Snakemake
having no way to depend on a subset of its keys, so editing an unrelated key
rebuilds every corpus.

GISAID: every file under ``PARQUET_DIR``, ``NEUTRAL_DIR`` and ``ERA_MAP_DIR``
is GISAID-derived and stays on the cluster.
"""

NEUTRAL_DIR = f"{ARTIFACT_ROOT}/neutral_models"
ERA_MAP_DIR = f"{ARTIFACT_ROOT}/era_maps"
PARQUET_DIR = f"{ARTIFACT_ROOT}/parquet/{GENE}"
VALIDATION_DIR = f"{ARTIFACT_ROOT}/validation"

#: The committed era template, and the map rendered from it.
ERA_TEMPLATE = f"{workflow.basedir}/data/eras-pre-omicron-omicron.yaml"
ERA_MAP = f"{ERA_MAP_DIR}/{GENE}.yaml"

#: The three archives ``build_neutral`` writes, named exactly as
#: ``build_era_neutral_models.archive_name`` spells them. Declared as concrete
#: paths rather than a pattern so a rename there breaks the DAG rather than
#: leaving the rule writing files nothing consumes.
NEUTRAL_ZIPS = [
    f"{NEUTRAL_DIR}/TabularRates-SCoV2-gisaid-public-{GENE}-{CUTOFF}-{VERSION}-{v}.zip"
    for v in ("pre_omicron", "omicron", "pooled")
]


wildcard_constraints:
    # Only the corpora config.yaml names. Without this a typo reaches
    # `corpus()` as a wildcard and fails at run time with a bare KeyError,
    # after Snakemake has already committed to a DAG built around it.
    corpus="|".join(CORPUS_NAMES),


rule corpora:
    """Every corpus this pipeline trains on, each one checked, plus the audit.

    `CORPUS_NAMES`, not `BUILT_CORPUS_NAMES`: `full_depth` is received rather
    than built, but it is the corpus the paper's models were trained on and it
    is the one whose fold-0 draw is row-positional, so it is the last corpus
    that should go unchecked. It once reached the DAG only because a rule that
    no longer exists consumed its report, which made the guard on the paper's
    own data an accident of scaffolding rather than something this target
    asks for.
    """
    input:
        expand(f"{ARTIFACT_ROOT}/corpus_guard/{{corpus}}.json", corpus=CORPUS_NAMES),
        f"{ARTIFACT_ROOT}/audit/{GENE}.json",


rule build_neutral:
    """Build the gene's three TabularRates archives: both eras plus pooled.

    The gene-aware rate selection in ``scov2_rates.py`` is the
    correctness-critical part, and it is declared below so that a fix there
    invalidates these archives.
    """
    input:
        script="scripts/build_era_neutral_models.py",
        rates_lib="dasm2/neutral/scov2_rates.py",
        tabular_lib="dasm2/neutral/tabular_rates.py",
        config="config.yaml",
        lib="scripts/lib.py",
    output:
        zips=NEUTRAL_ZIPS,
    resources:
        mem_mb=PREP_MEM_MB,
    shell:
        "python {input.script}"
        " --gene {GENE}"
        " --output-dir {NEUTRAL_DIR}"
        " --force"


rule verify_neutral:
    """Gate every corpus build on the four archive checks.

    Not optional and not advisory. A neutral model holding the wrong rates
    trains without error and yields wrong selection factors, so this runs
    against the built archives rather than trusting the builder. It is a script
    rather than a test because its inputs are GISAID-derived and live on the
    cluster.
    """
    input:
        script="scripts/verify_era_neutral_models.py",
        # It rebuilds the expected rates through the builder's own
        # `BUILD_VARIANTS`, `archive_name` and `build_inputs`, so a change to
        # how the archives are named or assembled changes what this checks.
        builder="scripts/build_era_neutral_models.py",
        rates_lib="dasm2/neutral/scov2_rates.py",
        zip_lib="dasm2/serialization/zip_io.py",
        dasm2_init="dasm2/__init__.py",
        config="config.yaml",
        lib="scripts/lib.py",
        zips=NEUTRAL_ZIPS,
    output:
        done=f"{VALIDATION_DIR}/neutral_verified_{GENE}.done",
    resources:
        # Loads and diffs TabularRates archives. Declared because a rule that
        # declares nothing reserves ZERO under `--resources mem_mb=`, so it
        # would co-schedule beside a 64 GB training job with no accounting.
        mem_mb=PREP_MEM_MB,
    shell:
        "python {input.script}"
        " --archive-dir {NEUTRAL_DIR}"
        " --genes {GENE}"
        " && mkdir -p $(dirname {output.done})"
        " && touch {output.done}"


rule validate_tests:
    """Run the test suite before anything writes a corpus.

    ``prepare_viral_parquet.py`` is what every corpus goes through, so a
    regression there would reach every arm at once. Declaring the test files as
    inputs means an edited or newly added test reruns pytest rather than
    finding the marker fresh.

    Not gated on ``-x``: the full suite runs so one failure does not hide
    others.
    """
    input:
        tests=sorted(
            str(Path(p).relative_to(workflow.basedir))
            for p in (Path(workflow.basedir) / "tests").glob("test_*.py")
        ),
        prep_lib="scripts/prepare_viral_parquet.py",
        lib="scripts/lib.py",
        config="config.yaml",
    output:
        done=f"{VALIDATION_DIR}/tests_passed.done",
    shell:
        "python -m pytest tests -q"
        " && mkdir -p $(dirname {output.done})"
        " && touch {output.done}"


rule render_era_map:
    """Render the gene's per-era map from the committed template.

    One map, shared by every corpus. ``build_era_map`` is used rather than
    bypassed because it is what fills ``{scheme}`` -- the value that lands in
    the Parquet's ``neutral_model_name`` metadata and is the only record in a
    finished file of which scheme produced it.

    Written under the artifact root rather than committed: it is derived, and
    it names absolute archive paths that are installation-specific.
    """
    input:
        template=ERA_TEMPLATE,
        era_lib="scripts/write_era_config.py",
        # `write_era_config` takes `archive_name` from the builder, so the
        # paths this map names move when that function does.
        builder="scripts/build_era_neutral_models.py",
        config="config.yaml",
        lib="scripts/lib.py",
        zips=NEUTRAL_ZIPS,
        verified=f"{VALIDATION_DIR}/neutral_verified_{GENE}.done",
    output:
        era_map=ERA_MAP,
    run:
        import sys

        sys.path.insert(0, str(Path(workflow.basedir) / "scripts"))
        import yaml
        from write_era_config import build_era_map, render_era_config

        rendered = render_era_config(
            build_era_map(Path(input.template).read_text()), GENE, ARTIFACT_ROOT
        )
        # Fail here rather than one stage downstream: a surviving placeholder
        # becomes part of an archive path, and would only surface when the
        # archive was opened.
        for token in ("{gene}", "{artifact_root}", "{era}", "{scheme}"):
            if token in rendered:
                raise ValueError(
                    f"{output.era_map}: {token} survived rendering, so an "
                    f"archive path is not a real path."
                )
        spec = yaml.safe_load(rendered)
        for era, body in spec["eras"].items():
            if not Path(body["archive"]).exists():
                raise ValueError(
                    f"{output.era_map}: era {era!r} names a missing archive "
                    f"{body['archive']}."
                )
        Path(output.era_map).parent.mkdir(parents=True, exist_ok=True)
        Path(output.era_map).write_text(rendered)


rule prepare_parquet:
    """Write one corpus's training Parquet with per-era precomputed rates.

    **One per corpus**: the corpus *is* the arm. What must be identical across
    arms is the neutral model, and it is -- one era map, one set of archives.

    **No row filtering, subsampling or reshaping happens here.** The source is
    an already-thinned mat-pcp aggregate, read in file order, and the corpus
    axis reaches this rule through ``--pcp-csv`` and nothing else. That is what
    lets ``check_corpus`` reproduce the row-order digest, and so the fold.

    ``--summary`` records the per-era row counts, which is the cheapest
    evidence the clade -> era assignment did something: it must name two eras
    with non-zero counts.

    The script validates its own output by reading it back the way training
    will, so a corrupt or unloadable file fails here rather than at the start
    of a multi-hour run.
    """
    wildcard_constraints:
        # Only the thinned arms are built. Without this the global constraint
        # admits `full_depth`, and Snakemake will happily build an unchecked
        # copy of it at this path from `source_pcp` -- a file nothing consumes,
        # sitting next to the pinned one it is not.
        corpus="|".join(BUILT_CORPUS_NAMES),
    input:
        script="scripts/prepare_viral_parquet.py",
        era_map=ERA_MAP,
        verified=f"{VALIDATION_DIR}/neutral_verified_{GENE}.done",
        tested=f"{VALIDATION_DIR}/tests_passed.done",
        config="config.yaml",
        lib="scripts/lib.py",
        # The data the corpus is made of, declared so the DAG records what
        # this was built from -- and `ancient`, so a new mtime on it does not
        # rebuild. The file is far above snakemake's 1 MB checksum threshold,
        # so only its mtime is compared; mat-pcp re-emits its tree in place,
        # and a byte-identical re-emit would otherwise rebuild every corpus
        # and, under M3, retrain on them.
        #
        # A re-emit that genuinely changed the data must not rebuild here
        # either: a regenerated corpus is a different 90/10 split of the
        # paper's data. `check_corpus` is what catches that, and it declares
        # this same file un-ancient so it reruns and fails on the pins. To
        # rebuild on purpose, update them in `config.yaml`, which is small
        # enough to be content-tracked and does trigger.
        pcp=lambda wc: ancient(str(corpus(wc.corpus).source_pcp)),
        # The code that decides the row-group layout and the `nt_rates` /
        # `nt_csps` columns.
        writer="dasm2/data/parquet_writer.py",
        rates="dasm2/neutral/tabular_rates.py",
        # The rest of the read-convert-write path: the backend and row
        # converter the rows go through, the seam identity and segment handler
        # that decide what a row is, and the sha256 recorded in the metadata.
        backend="dasm2/data/parquet_backend.py",
        registry="dasm2/data/registry.py",
        segment_handlers="dasm2/data/segment_handlers.py",
        local="dasm2/data/local.py",
        # Imported for its side effects: the package `__init__` calls
        # `use_system_arrow_pool()` and `disable_mha_fastpath()` at import.
        # This declares the file holding those two calls, not the modules
        # implementing them -- direct imports only, as above. See
        # `tests/test_import_order.py`.
        dasm2_init="dasm2/__init__.py",
    output:
        parquet=f"{PARQUET_DIR}/{{corpus}}.parquet",
        summary=f"{PARQUET_DIR}/{{corpus}}_era_rows.json",
    resources:
        mem_mb=PREP_MEM_MB,
    shell:
        "python {input.script}"
        " --pcp-csv {input.pcp}"
        " --eras {input.era_map}"
        " --output {output.parquet}"
        " --summary {output.summary}"
        " --force"


rule check_corpus:
    """Row count, fold-0 split sizes, row-order digest and md5.

    Every later milestone inherits this. It is the reason a rebuilt corpus
    cannot silently become a different 90/10 split of the paper's data.

    **This rule watches both files it reads, and neither is ancient.** It is
    the cheap guard, so it is the one that should rerun when anything moves.
    The Parquet was previously declared only for corpora built here, leaving
    the received ``full_depth`` -- the corpus most of the paper trains on --
    unwatched. The source PCP was read and gated by ``check_corpus.py`` and
    declared nowhere at all, which is the gap that matters most: mat-pcp
    re-emits its tree in place, and ``source_pcp_md5`` is the pin that catches
    it. Both files are far above snakemake's 1 MB checksum threshold, so a
    re-emit moves their mtime and this rule reruns, whether or not the bytes
    changed. That is the intended trade -- minutes of hashing against a stale
    pass on the paper's own split.

    A missing received Parquet now raises ``MissingInputException`` at DAG
    build rather than being reported by ``check_corpus.py`` itself. Both
    abort; this one aborts before the corpora are built rather than after.
    """
    input:
        script="scripts/check_corpus.py",
        config="config.yaml",
        lib="scripts/lib.py",
        parquet=lambda wc: str(corpus(wc.corpus).path),
        source_pcp=lambda wc: str(corpus(wc.corpus).source_pcp),
    output:
        report=f"{ARTIFACT_ROOT}/corpus_guard/{{corpus}}.json",
        membership=f"{ARTIFACT_ROOT}/corpus_guard/{{corpus}}_fold0_membership.npz",
    log:
        f"{ARTIFACT_ROOT}/logs/check_corpus_{{corpus}}.log",
    shell:
        "python {input.script}"
        " --corpus {wildcards.corpus}"
        " --report {output.report}"
        " --membership-output {output.membership}"
        " > {log} 2>&1"


rule audit_censor:
    """Measure what the censoring actually removes, as JSON.

    A readout, not a build input: nothing downstream reads it. It is the
    ``O(corpus)`` half of the censoring accounting, and the cheap ``O(table)``
    invariants stay in ``CensorMaskedMutations`` and are paid on every training
    start.

    Full depth only. The censoring is pinned across corpora, so what it removes
    from a thinner one is a smaller version of the same thing, and a per-corpus
    audit would report the corpus size the ladder already reports. What the
    audit answers is "can the censoring separate anything at all", which is a
    question about the gene and the table rather than about the depth.
    """
    input:
        script="scripts/audit_censor.py",
        masking_lib="dasm2/sequence/masking.py",
        constants_lib="dasm2/sequence/constants.py",
        io_lib="dasm2/io.py",
        dasm2_init="dasm2/__init__.py",
        config="config.yaml",
        lib="scripts/lib.py",
    output:
        json=f"{ARTIFACT_ROOT}/audit/{GENE}.json",
    resources:
        mem_mb=PREP_MEM_MB,
    shell:
        "python {input.script} --gene {GENE} --output {output.json}"
