# viral-dasm

This repository trains a DASM from the PCPs the
[`mat-pcp`](https://github.com/matsengrp/mat-pcp) pipeline produces, and
performs every analysis in the manuscript, including the generation of every
figure.

One Snakemake pipeline goes from mat-pcp's outputs to the paper's figures.
Everything the pipeline reads is named in [`config.yaml`](config.yaml): no
manifest lookup, no registry, no artifact-root default. A reader with their own
GISAID access edits that one file to point at their own mat-pcp run.

**Status: milestone 1 of 7.** The training code is vendored and its gates are
green; the corpora, the remaining training runs, the scoring tables, the four
analyses, the figure composition and the release are issues
[#1](https://github.com/matsengrp/viral-dasm/issues/1) tracks.

## Layout

```
config.yaml   every path and pinned number the pipeline uses
Snakefile     terminal targets, named for what they produce
workflow/     the rules, one file per stage
configs/      training-config templates, rendered from config.yaml
scripts/      the pipeline's own code: gates, guards, renderers
dasm2/        a vendored fork of the model-training code (see its PROVENANCE.md)
tests/        what has to hold for the gates to mean anything
data/         the publishable numbers behind each figure
results/      pipeline outputs (not tracked; they live under artifact_root)
```

## Running it

Activate the environment first; the Snakefile calls bare commands.

```bash
uv sync --group dev
source .venv/bin/activate
snakemake --cores 4
```

## What is not here, and why

**Nothing GISAID-derived.** No sequences from the UShER tree, no PCPs, no
censor tables, no founders, no counts tables, no derived per-site or
per-sample slice. That is the condition the data is held under, not a
limitation to work around.

Reproducibility therefore runs through the code, not the data. Both pipelines
are published: a reader with their own GISAID access runs `mat-pcp` on the
UShER tree to produce PCPs and counts, then runs this pipeline on those
outputs. mat-pcp's outputs are plain inputs here — this pipeline never reaches
behind them to a tree, an alignment or a MAT, so a missing input stops the DAG
at the mat-pcp target that produces it.

The trained checkpoints are parameters, not sequence, so they are published.
Every figure depends on a checkpoint *path* that either a training rule or a
fetch rule can supply, and the figures rebuild in minutes for a reader who is
not retraining.

## The vendored training code

`dasm2/` is a one-way fork of the model-training code, taken once from a
private repository and never merged with afterwards — the paper's code should
keep reproducing the paper's numbers, which a dependency that moves does not.
[`dasm2/PROVENANCE.md`](dasm2/PROVENANCE.md) records the commit it came from
and what is still to be pruned.

It keeps the name `dasm2`, so each module reads exactly as it does upstream
and a diff against the source commit stays meaningful. `tests/test_fork.py`
enforces what that name would otherwise hide: `dasm2.__file__` has to resolve
inside this repository, and no dependency may install a different one.

## The gates

The port is checked against runs the private repository already made, on the
same corpus, the same fold and the same seed.

| gate | cost | what it proves |
|---|---|---|
| `fast_test` | under a minute | the whole training path runs: Parquet load, tokenization, censoring, forward, backward, eval |
| `check_config` | seconds | the rendered config, and the three library defaults no config file states that together decide where a run stops |
| `check_corpus` | minutes | the corpus is the file the paper's 90/10 split was drawn from |
| `gate1` | minutes, no training | the reference checkpoint scores 7.766710 here, over exactly 158,137 loss-mask positions |

Two more land separately: a 20,000-step canary against the reference's first
ten evaluations, and a full run to the learning-rate floor.
