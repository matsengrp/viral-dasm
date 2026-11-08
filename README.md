# viral-dasm

This repository trains a DASM from the PCPs the
[`mat-pcp`](https://github.com/matsengrp/mat-pcp) pipeline produces, and
performs every analysis in the manuscript, including the generation of every
figure.

One Snakemake pipeline goes from mat-pcp's outputs to the paper's figures.
Everything the pipeline reads is named in [`config.yaml`](config.yaml): no
manifest lookup, no registry, no artifact-root default. A reader with their own
GISAID access edits that one file to point at their own mat-pcp run.

That is enforced, not just intended. The vendored tree carried a manifest
registry that resolved dataset nicknames by scanning directories named in
`~/.config/dasm2/config.yaml` -- a file outside this repository, whose presence
changed what the code did. It is not vendored. Nothing here consults anything
outside the checkout and the paths `config.yaml` names.

**Status: milestone 2 of 7.** The training code is vendored and shown to still
train the model it was ported from, and the pipeline builds its own corpora and
the neutral model they carry. The training runs, the scoring tables, the four
analyses, the figure composition and the release are issues
[#1](https://github.com/matsengrp/viral-dasm/issues/1) tracks.

## Layout

```
config.yaml   every path and pinned number the pipeline uses
Snakefile     terminal targets, named for what they produce
workflow/     the rules, one file per stage
configs/      training-config templates, rendered from config.yaml
scripts/      the pipeline's own code: builders, guards, renderers
dasm2/        a vendored fork of the model-training code (see its PROVENANCE.md)
tests/        what has to hold for the rest of it to mean anything
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

## Building a corpus

`snakemake corpora` goes from a mat-pcp run to a training Parquet, in four
stages:

| rule | what it produces |
|---|---|
| `build_neutral` | three `TabularRates` archives -- both eras and the pooled fit -- from mat-pcp's GLM tables |
| `verify_neutral` | four checks on those archives, including that they reproduce the ones the paper trained against |
| `render_era_map` | the era -> clade map, from the committed template in `data/`, with real archive paths filled in |
| `prepare_parquet` | one corpus's Parquet, each row carrying the neutral rates of its clade's era |

**One neutral model, every corpus.** mat-pcp does not refit its GLM per
sampling depth, so `build_neutral` and `render_era_map` take no corpus
argument. That is what pins λ across the ladder: a per-corpus rebuild would let
the neutral model move with the corpus, which is the one thing a depth contrast
cannot afford.

**Nothing here filters, subsamples or reshapes rows.** A sampling-depth arm
*is* a mat-pcp aggregate; the thinning happens there. The Parquet inherits its
source file's row order verbatim, which is what lets `check_corpus` prove a
rebuild gives the paper's fold 0 -- the held-out set is a row-positional draw,
so the same row order is the same split.

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

**If you have the private repository installed, read this before deleting
anything from `dasm2/`.** An editable install of it puts a finder on
`sys.meta_path` that maps the *name* `dasm2` to the directory it was installed
from, and resolves submodules against that directory rather than against this
package's `__path__`. A module you delete here therefore still imports, served
from the other checkout, and the pipeline runs green on code this repository
does not contain. Deleting a file and watching the pipeline pass is not
evidence the file stopped running.

Three things make that visible rather than silent. `assert_dasm2_is_this_repo`
in [`scripts/lib.py`](scripts/lib.py) sweeps `sys.modules` after the import and
fails on any `dasm2` module resolving outside this tree; every script that
imports `dasm2` calls it. `tests/test_import_order.py` requires `lib` to be
imported before `dasm2`, since `lib` is what puts the repository root on
`sys.path` and a script run never puts it there by itself. And
`tests/test_fork.py` runs the same sweep in a subprocess started outside the
repository. Checking the package root alone passes while a submodule leaks, so
all three check module by module.

## The corpus guard

`check_corpus` is the one check the pipeline keeps, and it is not about the
port. Fold 0 is a row-positional draw: the held-out index set comes from
`(seed, num_folds, n_rows)`, and which PCP sits at each index comes from the
Parquet's row order. A rebuilt corpus that reorders rows is therefore a
different 90/10 split of the same data, silently. The guard pins the row order
-- a blake2b-128 digest over the per-row `(clade, parent, child)` digests, in
order -- so that cannot happen unnoticed.

It runs for all four corpora, `full_depth` included. That one is received from
mat-pcp rather than built here, which makes it the corpus most easily left
unchecked and the one whose split the paper's models were actually trained on.

The checks that proved the vendored tree still trains the model it came from
were port scaffolding, and
[#8](https://github.com/matsengrp/viral-dasm/issues/8) removed them. What they
measured is recorded in the pull requests that ran them: the reference
checkpoint scored 7.766710 in this tree over 158,137 loss-mask positions, and
a full run from scratch here reached a best validation loss of 7.766302
against the reference's 7.766710, stopping on the same learning-rate floor.
Those were one-time claims about the port, and the pipeline that builds the
paper's figures should not carry the apparatus that made them.
