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

The seven-run concatenation stays under the artifact root because of its size
-- 1.4M rows per run, 209 MB in all -- not because of the data rules. Counts,
fitness effects and selection factors are publishable, and so are the clade
founders that a per-clade mutation label reconstructs: `aa_mutation` spells
the founder residue, so any mutation-level table published here gives them up,
the per-panel CSVs included. They are reconstructible from the public UShER
tree without GISAID, so this is not the line the data rules draw.

What is tracked is the single run the manuscript cites -- see
``publish_scored_table`` at the foot of this file.
"""

from lib import FITNESS_SUFFIX, FULL_DEPTH
from panel_spec import (
    ARM,
    CANDIDATE_CLADES,
    CANDIDATE_CSV,
    CLADE_HAPLOTYPE_TABLE,
    FIGURE_MODEL,
    HAPLOTYPE_TABLE,
)
from runs import SCORE_MEM_MB

SCORED_DIR = f"{ARTIFACT_ROOT}/scored"

#: The one table the figures read.
SCORED_TABLE = f"{SCORED_DIR}/selection_factors.parquet"

#: The clade guard's report. Phase-gates every scoring job: the roster is read
#: out of mat-pcp's directories, so a run scored before the three readings are
#: known to agree could silently cover fewer clades than the run has.
CLADE_GUARD = f"{ARTIFACT_ROOT}/clade_guard/clades.json"

#: The one scored run the manuscript cites, and the only one tracked.
#:
#: `FIGURE_MODEL` rather than a second spelling of the model's name: the table
#: the paper publishes is the model its figures are drawn from, so if that ever
#: moves, this moves with it instead of going quietly stale.
PUBLISHED_RUN = run_id(FIGURE_MODEL, FULL_DEPTH.name)

#: Tracked. 23 MiB of parquet, 1,425,760 rows: every amino acid at every spike
#: site on every one of the 56 clade founders, carrying the DASM's selection
#: factor beside both the per-clade and the clade-pooled counts estimate. Every
#: panel of `fig:agreement` is a filter on it, and the three remaining figures
#: read it too, which is why the manuscript names this one file rather than the
#: per-panel CSVs drawn from it.
PUBLISHED_TABLE = f"results/selection_factors/{PUBLISHED_RUN}.parquet"


# The per-clade mat-pcp files the scoring stage reads, declared by globbing the
# directories rather than by naming the clades. `config.yaml` gives the two
# directories and not their contents, on purpose -- the roster is whatever
# mat-pcp wrote -- so a rule that wants the contents as inputs has to look.
#
# Globbed independently of each other, and not derived from one roster: a clade
# present in one directory and absent from the other must reach `check_clades`,
# which says which clade and which directory, rather than surfacing as a
# `MissingInputException` on a path this file invented.
#
# Not `ancient()`, unlike `train`'s declarations of the same mat-pcp run. There
# the asymmetry is brutal -- a byte-identical re-emit would restart a multi-day
# fit -- while a rescore is seven jobs of minutes each, so mtime-tracking a
# re-emit and rescoring is the cheaper mistake to make.
def founder_fastas(*_):
    """Every clade founder FASTA under ``clade_founders_dir``."""
    return sorted(str(p) for p in mat_pcp_path("clade_founders_dir").glob("*.fa"))


def background_clade_founders(*_):
    """The founders `clade_haplotypes` compares its parents against.

    The figure's three clades rather than all 56, because that is what the
    rule reads: naming the rest would re-enumerate the haplotypes whenever an
    unrelated clade's founder was re-released.
    """
    root = mat_pcp_path("clade_founders_dir")
    return [str(root / f"{clade}.fa") for clade in CANDIDATE_CLADES]


def per_clade_fitness_tables(*_):
    """Every per-clade fitness table, less the clade-pooled one.

    Globs the directory's own level, so the ``cap_*_0/`` per-arm refits are not
    swept in. The pooled table is excluded by name and declared separately,
    because it is read through a different code path with a different schema.
    """
    root = mat_pcp_path("aamut_fitness_dir")
    pooled = mat_pcp_path("aamut_fitness_pooled_file").name
    return sorted(
        str(p) for p in root.glob(f"*{FITNESS_SUFFIX}") if p.name != pooled
    )


rule scored:
    """The tidy table, with the clade roster checked first."""
    input:
        # Redundant as a gate -- `table` reaches the guard through `score` --
        # and kept because `rules.scored.input` is what `rule all` carries, so
        # naming it here is what makes the guard visible from the top target.
        guard=CLADE_GUARD,
        table=SCORED_TABLE,
        published=PUBLISHED_TABLE,


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
        # The two directory contents the guard compares. Without them the
        # guard's own trigger case -- a clade added to or dropped from one
        # directory -- leaves the guard up to date and its stale report
        # standing.
        founders=founder_fastas,
        fitness=per_clade_fitness_tables,
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
        # The founder reader, shared with the panel rules so that editing one
        # side does not mark the other's outputs out of date. It is what reads
        # `dasm2/io.py`, `conversion.py` and `scov2_rates.py` below.
        founder_reader="scripts/founders.py",
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
        # What the two vendored analysis modules run. Not reached by the
        # entry-point check, which reads one file's imports, but this rule runs
        # them all the same and a change to any of them changes the numbers it
        # writes. `constants.py` fixes `AA_STR_SORTED`, which is the 20-column
        # layout of every selection-factor row.
        constants="dasm2/sequence/constants.py",
        head_types="dasm2/heads/types.py",
        core_base="dasm2/models/core/base.py",
        factory="dasm2/models/factory.py",
        # The marker, not the checkpoint: `lib.select_checkpoint` resolves the
        # file by glob and no rule declares one as an output.
        done=f"{RUN_DIR}/{{model}}_{{corpus}}/training_complete.done",
        # And the stopping-point guard, not only the marker. `train` writes
        # `training_complete.done` for any run that exits 0, a run that
        # exhausted `max_train_steps` included; `check_run` is what refuses one
        # that stopped anywhere but the learning-rate floor. Without this the
        # capped arm `train.smk` promises "stops its own branch of the DAG"
        # reaches `selection_factors.parquet`, which is the figure input.
        run_guard=f"{ARTIFACT_ROOT}/run_guard/{{model}}_{{corpus}}.json",
        fitness=mat_pcp_path("aamut_fitness_pooled_file"),
        censored=mat_pcp_path("censored_mutations_file"),
        clades=mat_pcp_path("censored_clades_file"),
        masked=mat_pcp_path("masked_nt_sites_file"),
        # The per-clade founders and fitness tables this run is scored on and
        # against. These are the `parent_aa` and `delta_fitness_*` columns; the
        # pooled table above is only the `_all` half.
        founders=founder_fastas,
        per_clade_fitness=per_clade_fitness_tables,
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


rule publish_scored_table:
    """Copy the cited run's scored table into tracked ``results/``.

    A copy rather than a second output of ``rule score``, because ``score``
    claims a GPU: declaring this path there would mean that publishing a file
    which already exists re-staled seven GPU jobs.

    It runs ``cp`` and no repository code, which is why
    ``tests/test_rule_inputs.py`` carries it as a deliberate exemption rather
    than an entry point. The table's *content* is ``score``'s to be right
    about; this rule is only responsible for the copy being of the run
    ``PUBLISHED_RUN`` names.
    """
    input:
        table=f"{SCORED_DIR}/{PUBLISHED_RUN}.parquet",
    output:
        table=PUBLISHED_TABLE,
    shell:
        "cp {input.table} {output.table}"


# ---------------------------------------------------------------------------
# `fig:within_clade_shifts`' model half
# ---------------------------------------------------------------------------

#: The haplotypes the figure's clades hold, and their scores. Both under the
#: artifact root rather than tracked `results/`: a row is a Spike haplotype,
#: so both are GISAID-derived. The tracked per-panel CSVs name a background by
#: its residues and carry no sequence.
CLADE_HAPLOTYPES = str(CLADE_HAPLOTYPE_TABLE)
HAPLOTYPE_SCORES = str(HAPLOTYPE_TABLE)


rule clade_haplotypes:
    """Every distinct amino-acid parent of the three clades the figure draws.

    The DASM side of panels B, D and F is the pool average -- the mean log
    selection factor over the haplotypes a background contains, weighted by
    each one's synonymous events -- so the model has to score every haplotype
    a pool might hold. `pool_average_log_sf` raises on one it was not given,
    rather than averaging over the intersection, and the clade's whole
    distinct-parent set is the only superset of both pools that is known
    before the counts pass runs.

    Its own rule, ahead of the scoring, so the scoring rule does not depend on
    the counts rule: 7,549 of the 8,693 haplotypes sit in a pool, and chasing
    that 13% would buy the dependency this DAG is arranged to avoid.
    """
    input:
        script="scripts/clade_haplotypes.py",
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
        pcps=str(corpus(ARM).source_pcp),
        clade_founders=background_clade_founders,
    output:
        table=CLADE_HAPLOTYPES,
    log:
        f"{ARTIFACT_ROOT}/logs/clade_haplotypes.log",
    resources:
        mem_mb=SCORE_MEM_MB,
    shell:
        "python {input.script} --pcp {input.pcps} --output {output.table}"
        " > {log} 2>&1"


rule score_haplotypes:
    """Score the figure's model on every one of those haplotypes.

    **No GPU, and deliberately so.** Measured on `dasm-1077k_full_depth`, one
    1,273-residue Spike haplotype is 28.5 ms on CPU against 4.7 ms on an idle
    card: 4.1 minutes against 0.7 for the whole set. A 3.4-minute saving does
    not justify holding a card a training run could be using, and it is what
    makes "no rule of this figure takes a card" true outright. So this rule
    carries neither `resources: gpu=1` nor `rule score`'s `{TRAIN_ENV}`
    thread pinning, which exists for jobs that share a node with training.

    One model, not the seven and not both selections. `fidelity` costs no
    rescoring for `fig:agreement`, which is one table and one `groupby`, but
    here it would mean scoring 8,693 haplotypes under two more checkpoints
    for a comparison no panel reads.

    Its own output, **not** rows in `selection_factors.parquet`:
    `tests/test_score_run.py` pins `background` empty across all of that
    table and `panels.base_frames` filters on `(model, clade)` alone, so a
    haplotype row sharing it would be pooled into every agreement panel with
    nothing raising.
    """
    input:
        script="scripts/score_haplotypes.py",
        score_run="scripts/score_run.py",
        gpu_slot="scripts/gpu_slot.py",
        spec="scripts/panel_spec.py",
        lib="scripts/lib.py",
        config="config.yaml",
        analysis_viral="dasm2/analysis/viral.py",
        analysis_eval="dasm2/analysis/viral_fitness_eval.py",
        checkpoints="dasm2/serialization/checkpoints.py",
        masking="dasm2/sequence/masking.py",
        conversion="dasm2/sequence/conversion.py",
        constants="dasm2/sequence/constants.py",
        head_types="dasm2/heads/types.py",
        core_base="dasm2/models/core/base.py",
        factory="dasm2/models/factory.py",
        io="dasm2/io.py",
        scov2_rates="dasm2/neutral/scov2_rates.py",
        dasm2_init="dasm2/__init__.py",
        candidates=str(CANDIDATE_CSV),
        haplotypes=CLADE_HAPLOTYPES,
        censored=mat_pcp_path("censored_mutations_file"),
        clades=mat_pcp_path("censored_clades_file"),
        masked=mat_pcp_path("masked_nt_sites_file"),
        # The marker and its guard, as `rule score` declares them: a run that
        # stopped anywhere but the learning-rate floor must not reach a panel.
        done=f"{RUN_DIR}/{PUBLISHED_RUN}/training_complete.done",
        run_guard=f"{ARTIFACT_ROOT}/run_guard/{PUBLISHED_RUN}.json",
    output:
        table=HAPLOTYPE_SCORES,
    params:
        outdir=f"{RUN_DIR}/{PUBLISHED_RUN}",
    log:
        f"{ARTIFACT_ROOT}/logs/score_haplotypes.log",
    resources:
        mem_mb=SCORE_MEM_MB,
    shell:
        "python {input.script}"
        " --model {FIGURE_MODEL}"
        " --corpus {FULL_DEPTH.name}"
        " --run-dir {params.outdir}"
        " --backgrounds {input.haplotypes}"
        " --output {output.table}"
        " --device cpu"
        " > {log} 2>&1"
