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

**Status: milestone 5 of 7.** The training code is vendored and shown to still
train the model it was ported from, the pipeline builds its own corpora and the
neutral model they carry, `snakemake models` trains the paper's seven models on
them, and `rule score` reads those models into one table of selection factors.
Of the paper's four analyses the agreement analysis is ported, and
`fig:agreement` composes itself from it -- see [*Drawing the
figures*](#drawing-the-figures). The other three analyses and the release are
issues [#1](https://github.com/matsengrp/viral-dasm/issues/1) tracks.

## Layout

```
config.yaml   every path and pinned number the pipeline uses
Snakefile     terminal targets, named for what they produce
workflow/     the rules, one file per stage
configs/      training-config templates, rendered from config.yaml
scripts/      the pipeline's own code: builders, guards, renderers
dasm2/        a vendored fork of the model-training code (see its PROVENANCE.md)
tests/        what has to hold for the rest of it to mean anything
data/         input the pipeline reads and never writes
results/      the publishable tables it writes, tracked
```

The two data directories split by direction, not by provenance. `data/` holds
input a reader may edit -- today just the era template. `results/` holds what
the pipeline produces and the repository keeps: `runs.csv`, and one CSV per
figure panel. Everything large or intermediate -- corpora,
checkpoints, the scored table, logs -- goes under `artifact_root` on the
cluster, outside this tree.

What may be tracked is a data rule, not a size rule: mutation counts, fitness
effects and selection factors are publishable, raw GISAID sequences are not.
The seven-run `selection_factors.parquet` is kept out because it is 209 MB,
not because of the rules; the single run the manuscript cites is tracked, at
23 MiB, by `rule publish_scored_table`.

**That tracked table gives up the 56 clade founder protein sequences**, and so
does every per-panel CSV beside it: a mutation label spells the founder
residue, so `D574V` in clade 22D says 22D's founder carries D at 574, and
`panel_b_within_clade_scatter.csv` alone pins 1,246 of Delta's 1,273. This is
deliberate. A clade founder is reconstructible from the public UShER tree
without GISAID credentials, so it is not the thing the data rules protect --
what stays out is the per-sample material, the PCPs and the leaf sequences.

## Running it

Activate the environment first; the Snakefile calls bare commands.

```bash
uv sync --group dev
source .venv/bin/activate
snakemake --cores 4
```

## What is not here, and why

**No per-sample GISAID material.** No sequences from the UShER tree's leaves,
no PCPs, no censor tables, no counts tables, no derived per-sample slice. That
is the condition the data is held under, not a limitation to work around.

The clade founders are the one derived sequence that *is* published, inside
the scored table and the panel CSVs — see the layout note above for why they
do not fall under this.

Reproducibility therefore runs through the code, not the data. Both pipelines
are published: a reader with their own GISAID access runs `mat-pcp` on the
UShER tree to produce PCPs and counts, then runs this pipeline on those
outputs. mat-pcp's outputs are plain inputs here — this pipeline never reaches
behind them to a tree, an alignment or a MAT, so a missing input stops the DAG
at the mat-pcp target that produces it.

**One input is not GISAID-derived.** `fig:dms` reads three published deep
mutational scans — full-spike cell-entry measurements on BA.1, XBB.1.5 and
KP.3.1.1 — which have their own papers and their own repositories. A reader
without GISAID access can obtain them. `config.yaml`'s `dms` block names them
by path and pins each one's md5, and this repository does not redistribute
them: they belong to the groups that published them, and a reader who wants to
rebuild that figure fetches them from those repositories.

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

## Training the models

`snakemake models` trains the paper's seven models and writes one row per run
into [`results/runs.csv`](results/runs.csv).

| rule | what it produces |
|---|---|
| `render_config` | one `(model, corpus)` training config, filled from `config.yaml` |
| `check_params` | the model built and counted, against the number in its own name |
| `render_smoke_config` / `smoke` | four CPU steps of each architecture on 400 PCPs: the wiring test |
| `train` | one run, on one claimed GPU, to the learning-rate floor |
| `read_run` | that run's endpoint, read out of its own log |
| `check_run` | the gate: a run that stopped anywhere else fails here |
| `runs_table` | the seven rows, gathered into `results/runs.csv` |

**Seven runs, four architectures.** Three transformer sizes and the
context-independent baseline at full depth, where the size ladder and the
pooling contrast are read; the smallest transformer alone at each of the three
thinned depths, because that is the architecture the sampling-depth ladder
trained at every rung. The list is seven named pairs, not a cross product.

| model | effective parameters | geometry | lr |
|---|---:|---|---:|
| `dasm-1077k` | 1,076,756 | nhead 16, d_model 128, d_pos 64, ff 512 | 5e-4 |
| `dasm-293k` | 292,628 | nhead 8, d_model 64, d_pos 32, ff 256 | 1e-3 |
| `dasm-85k` | 84,884 | nhead 4, d_model 32, d_pos 16, ff 128 | 1e-3 |
| `ci-484k` | 483,740 | `PerSiteParentChildHead`, no encoder | 5e-4 |

**Named by effective parameters**, which excludes the parameters that can
never receive a gradient: a transformer's positional table is allocated at
`pe_max_len` 5,000 rows against a corpus whose longest amino-acid sequence is
1,273, and the baseline's parent->parent diagonal is zapped in every forward.
`check_params` builds each model and reproduces the count rather than reading
it back out of the config, because a template transcribed from the wrong arm
renders, validates and trains perfectly well.

**One stopping rule for all seven.** Each run trains until `ReduceLROnPlateau`
has driven the learning rate below 1e-6. `max_train_steps` is a budget guard,
and a run that reaches it instead writes *no record at all* -- the trainer
breaks on the budget before it asks whether to stop early -- so the signature
is the absence of a stop event, and `check_run` fails on it rather than
letting a capped run reach a figure.

**`early_stopping_min_delta` is 0, for every run.** It is a readout parameter,
not a training one: it reaches best-model tracking and the patience counter,
and nothing else, because the plateau scheduler steps on the raw validation
loss. So two runs at one seed differing only in it descend identically, and
the coarser one is a prefix of the finer one. At 0, the saved best is the
validation curve's own minimum, and the two reference campaigns' thresholds
(1e-4 and 1e-3) are recovered by replaying their rule over the run's recorded
evaluations -- which is what `results/runs.csv`'s `*_at_min_delta_*` columns are.

**One run per card, claimed by the training process.** `scripts/gpu_slot.py`
ranks the cards by use, drops any carrying a process this user does not own --
and any whose occupant cannot be identified at all -- and takes an exclusive
lock under the artifact root. `scripts/train.py` holds it for exactly as long
as it trains. This is for throughput and for leaving other people's cards
alone, not for memory: attention runs through `scaled_dot_product_attention`
with `need_weights=False`, so no L x L matrix is ever materialised.

**`train` does not declare the test marker**, alone among the rules here. Every
other rule declares the code it runs, including this one; what it leaves out is
`tests_passed.done`, because an edit to any test file would otherwise
invalidate a finished multi-day run. The suite still runs ahead of the corpora,
which is where a regression in the corpus builder would do its damage.

## Drawing the figures

`snakemake panels` draws `fig:agreement`'s five panels from the one scored
table. Nothing in this stage infers and nothing takes a GPU: `rule score`
already did the forward passes, so a panel rule reads
`selection_factors.parquet`, filters it, correlates it and draws.

| rule | what it produces |
|---|---|
| `agreement_mutations` | panels A, B and D: one mutation across the clades, one clade's cloud, the shift grid |
| `agreement_ladders` | panels C and E: both correlations against the count cutoff |
| `compose_agreement` | the figure itself, laid out from those five PNGs |

`snakemake dms` draws `fig:dms` the same way, from the same scored table plus
the three published assays:

| rule | what it produces |
|---|---|
| `dms_correlations` | all four panels: the clouds and the bars, at each of the two count thresholds |
| `compose_dms` | the figure itself, laid out from those four PNGs |

`snakemake figures` runs all three. **Composition is its own rule because the
two costs are not alike**: a panel rule reads a 1.4M-row Parquet per model and
takes a minute, and the layout reads five PNGs and takes a second, so the
spacing can be nudged and re-run without rescoring anything. It also means the
figure cannot disagree with its panels -- it *is* their output rather than a
second drawing of them. `scripts/figure_layout.py` solves the dpi rather than
taking one: panels arrive as pixels, so the smallest dpi whose figure still
fits the page box is the largest the figure can legibly be, and nothing is
resampled or moved relative to anything else.

The composed PDF is also copied to `config.yaml`'s `tex_figure_dir`, the
manuscript's `figures/`, because a stale PDF in the paper is the failure this
stage exists to prevent. That copy is deliberately *not* a declared output: it
is outside both the repository and the artifact root, and snakemake deletes a
job's declared outputs when it fails, so naming it would let a failed layout
delete the figure currently in the paper. Set the key to `null` when the
manuscript is not checked out beside this repository; the composition then
writes its own copies and reports that it skipped the copy.

Each runs per `{selection}`, which names the models it draws. `figure` is
`dasm-1077k` -- the largest transformer, which is what every panel of the paper
shows -- and `fidelity` is `dasm-293k` and `dasm-85k`, the pair the published
panels were drawn from. Both come out of one table and one `groupby`, so the
second costs no rescoring, and `snakemake figures` builds both: tracked tables
that nothing in the DAG rebuilds are tables that quietly stop matching the code
beside them.

`fidelity` showed the port computing what the old code computed, and did, at
the port ([#21](https://github.com/matsengrp/viral-dasm/pull/21)).
[#36](https://github.com/matsengrp/viral-dasm/issues/36) then narrowed the
exclusion from the codon site to the mutation, which the old code does not do,
so the two now differ by exactly that -- 67,564 mutations at rung 10 against
65,430, over the same 22 clades, at a pooled median r of 0.9662 against 0.9664.
It is a second model pair through the same code and a diff to read when this
filter changes again, not a live claim about upstream.

**Where the two kinds of output go.** The tables land in tracked
`results/panels/<selection>/`, one CSV per panel holding exactly what that
panel plots, with the filter spec as columns so a reader who opens one alone
knows which cutoff and which clip produced it. The images go under
`artifact_root`: a panel is a rebuildable picture of the CSV beside it.

**What the manuscript cites is neither of those.** It names
`results/selection_factors/dasm-1077k_full_depth.parquet`, the whole scored
table for the model the figures are drawn from: 1,425,760 rows, every amino
acid at every spike site on all 56 founders, carrying the DASM's selection
factor beside both the per-clade and the clade-pooled counts estimate. Every
panel here is a filter on it and the three remaining figures will read it too,
so one file covers the paper where five per-panel CSVs would cover one figure.
`rule publish_scored_table` copies it in.

**The count cutoff is the x axis, not a setting.** Panels C and E sweep it, and
the clade set is pinned across the rungs -- raising the floor removes the thin,
poorly-measured clades first, so a sweep that let each rung keep its own clades
would show r rising with the threshold even if no surviving clade moved.

**The exclusion comes from mat-pcp, one mutation at a time.** A masked site, a
masked mutation or a founder→reference reversion sets mat-pcp's `exclude` flag
on the nucleotide route, and excluded routes are dropped before any fitness
effect is computed — so a mutation it excluded outright has no row to join
against, arrives with null counts, and leaves at `base_frames`' `dropna`.

Nothing here widens that to the codon. An earlier version did, dropping every
mutation at a site carrying any flag, and it cost 1,067 of the 33,782
comparable mutations across the figure's 22 clades — including 21K's R346K at
753 observed against 22.2 expected, whose counts come from the clade founder's
own background and are untouched by the censoring that flagged its site
([#36](https://github.com/matsengrp/viral-dasm/issues/36)).

**One class of row survives that drop and is still not a measurement.** Where a
clade's founder codon contains a nucleotide masked in the tree, every sequence
reads the reference base there, so no observation ever matches the founder
codon mat-pcp counts on, and the codon arrives with an actual count of zero
against a full expectation — a lethal-looking estimate with nothing behind it.
21J's codon 142 is the case: Delta's G142D puts the founder codon one masked
nucleotide away from the reference, mat-pcp holds 226 observed counts there and
keeps none, and even the synonymous route reports 64.3 expected against zero
observed. `unobservable_founder_codons` rebuilds that predicate from the mask
file, the founder and the reference, because nothing in the scored table marks
these rows. It is 14 mutations of 33,782, and they were the figure's largest
outliers ([#39](https://github.com/matsengrp/viral-dasm/issues/39);
[matsengrp/mat-pcp#161](https://github.com/matsengrp/mat-pcp/issues/161) is the
upstream fix that will make it a no-op).

**What makes that fair is that training censors the same routes.** The two
sides are one set applied twice: mat-pcp's `censor_table.py` splits the five
reasons behind `exclude` across the two files a training run reads, and its
`masked_nt_sites.py` reconciles the mask file against the count table's flag in
both directions. So the model was not trained on a route the counts kept, and
the counts do not rest on a route the model was taught to ignore.
`tests/test_panels.py::test_training_censors_exactly_what_mat_pcp_excludes`
re-derives that equality here, from the configured censor files against
mat-pcp's own per-clade count tables, rather than taking upstream's word: it is
the premise the whole comparison stands on, and nothing in the code enforces
it.

## The vendored training code

`dasm2/` is a one-way fork of the model-training code, taken once from a
private repository and never merged with afterwards — the paper's code should
keep reproducing the paper's numbers, which a dependency that moves does not.
[`dasm2/PROVENANCE.md`](dasm2/PROVENANCE.md) records the commit it came from,
what each cut removed, and what the fork still carries without running.

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
measured is recorded in the pull requests that ran them. The strongest of
them was a fixed-weights pass: the reference checkpoint, scored in this tree,
came within 1.1e-08 of the loss its own run logged, over exactly 158,137
loss-mask positions. A full run from scratch here then reached a best
validation loss of 7.766302 against the reference's 7.766710 -- a difference
of 4.1e-4, smaller than the run's own evaluation-to-evaluation jitter.

That 4.1e-4 is the only same-corpus, same-fold, same-seed measurement of this
difference there is, and it is one measurement. The three replicates of that
reference configuration -- 7.766710, 7.776982, 7.768600, sd 0.0055 -- move
*fold* as well as seed, and fold difficulty is likely the larger term, so
their spread bounds something else rather than this. A real tolerance is
still owed; [#1](https://github.com/matsengrp/viral-dasm/issues/1) asks for
one as its own subissue.

Those were one-time claims about the port, and the pipeline that builds the
paper's figures should not carry the apparatus that made them.
