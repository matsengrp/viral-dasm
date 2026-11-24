# viral-dasm

The code behind *Using deep learning to estimate the shifting mutational
fitness landscape of SARS-CoV-2 spike along a phylogeny*.

One Snakemake pipeline runs the whole analysis. It goes from the outputs of the
[`mat-pcp`](https://github.com/matsengrp/mat-pcp) pipeline — parent-child pairs
and counts-based fitness effects read off a SARS-CoV-2 UShER tree — to every
figure in the paper: it builds the training corpora and the neutral mutation
model they carry, trains the deep amino-acid selection models (DASMs), scores
each one against mat-pcp's counts-based fitness effects on every clade
founder, then draws the panels and lays them out.

Every path the pipeline reads is named in [`config.yaml`](config.yaml). There is
no manifest lookup, no registry and no artifact-root default: a reader with
their own GISAID access edits that one file to point at their own mat-pcp run,
and the pipeline follows. Nothing here consults anything outside the checkout
and the paths that file names.

## Key results

Large and intermediate artifacts — corpora, checkpoints, logs, panel images,
composed PDFs — go under `artifact_root`, which `config.yaml` points outside
this tree. What the repository itself keeps is the publishable end of the
analysis, and it is all under `results/`:

| file | what it holds |
|---|---|
| [`results/selection_factors/dasm-1077k_full_depth.parquet`](results/selection_factors) | **The table the paper is built on.** 1,425,760 rows, 23 MiB: every amino acid at every spike site on all 56 clade founders, carrying the DASM's log selection factor beside both the per-clade and the clade-pooled counts-based estimate. Every panel below is a filter on it. |
| [`results/runs.csv`](results/runs.csv) | One row per training run — seven of them — with the geometry, the stopping point read back out of the run's own log, the best validation loss, and the checkpoint's sha256. |
| [`results/panels/<selection>/*.csv`](results/panels) | One CSV per drawn panel, holding exactly what that panel plots, with the filter spec as columns so a reader who opens one alone knows which cutoff and which clip produced it. |
| [`results/backgrounds/`](results/backgrounds) | The genetic backgrounds each clade is split into, and the clades' emergence dates. |

`<selection>` names which models a panel was drawn from. `figure` is
`dasm-1077k`, the largest transformer, which is what the paper's panels show;
`dms` and `subsampling` are the two figures with a model set of their own; and
`fidelity` is a second pass over the same table under `dasm-293k` and
`dasm-85k`, which no published panel reads (see [*Drawing the
figures*](#drawing-the-figures)).

### Where each figure comes from

The composed PDFs land in `artifact_root/figures/<selection>/`. Figure 1 is a
schematic and is not built here; everything else is.

| in the paper | composed as | drawn from |
|---|---|---|
| Figure 2 | `figure/l455f.pdf` | `panels/figure/l455f_jian.csv`, `panels/figure/l455f_backgrounds.csv` |
| Figure 3 | `figure/agreement.pdf` | `panels/figure/panel_{a,b,c,d,e}_*.csv` |
| Figure 4 | `subsampling/subsampling.pdf` | `panels/subsampling/panel_{a,b,c,d}_*.csv` |
| Figure 5 | `dms/dms.pdf` | `panels/dms/panel_a_scatters_min0.csv`, `panels/dms/panel_b_bars_min0.csv` |
| Figure 6 | `dms/probabilities.pdf` | `panels/dms/panel_a_probability_vs_fitness_BA.1.csv`, `panels/dms/panel_b_dms_correlation_bars.csv` |
| Figure 7 | `dms/dms_strata_kp311.pdf` | `panels/dms/panel_{a,b}_*_strata_kp311.csv` |
| Figure S1 | `figure/within_clade_shifts.pdf` | `panels/figure/panel_a_outliers_<clade>.csv`, `panels/figure/panel_b_backgrounds_<clade>.csv` |
| Figure S2 | `figure/shift_error.pdf` (one panel, not composed) | `panels/figure/shift_error.csv` |
| Figure S3 | `dms/dms_strata_supplement.pdf` | `panels/dms/panel_{a,b}_*_strata_{delta,ba1,xbb15}.csv` |
| Table S1 | — | `panels/figure/table_s1_outlier_ranking.csv` |
| Table S2 | — | `panels/dms/predictor_correlations.csv` |

`predictor_correlations.csv` is the wider table behind the DMS comparison: each
predictor against each deep mutational scan, per homolog and per stratum. It
carries more rows than Table S2 prints.

## Running it

**Point [`config.yaml`](config.yaml) at your own mat-pcp run first** —
`artifact_root` for where the pipeline writes, and the `mat_pcp` block for what
it reads. Nothing else needs editing.

The environment is managed by [`uv`](https://docs.astral.sh/uv/). Activate it
before running anything: the Snakefile calls bare commands, not `uv run` or
`.venv/bin/`.

```bash
uv sync --group dev
source .venv/bin/activate
snakemake -n            # plan first: a bare run trains all seven models
snakemake --cores 4
```

Named targets let you build one stage at a time, and each is described in its
own section below:

| target | what it builds |
|---|---|
| `snakemake corpora` | the four training corpora and the neutral model they carry |
| `snakemake models` | the seven training runs, and `results/runs.csv` |
| `snakemake scored` | the models' selection factors against mat-pcp's counts, in one table |
| `snakemake figures` | every composed figure, and so every panel beneath them |
| `snakemake dms_predictors` | the DMS comparison table, `predictor_correlations.csv` |

`figures`, and the narrower `panels`, `dms`, `dms_strata`, `probabilities` and
`subsampling` beneath it, are the cheap end: no rule among them takes a GPU,
because `rule score` already did the forward passes. Training is the expensive
end — one GPU per run, to a learning-rate floor rather than a step budget, so a
run takes days — which is why `snakemake -n <target>` is worth running before
anything you have not built before.

`tex_figure_dir` in `config.yaml` is an optional second destination for the
composed PDFs, a directory outside this repository that the composition copies
into. Set it to `null` and the composition writes its own copies only, and
reports that it skipped the copy.

## What is not here, and why

**No per-sample GISAID material.** No sequences from the UShER tree's leaves,
no PCPs, no censor tables, no counts tables, no derived per-sample slice. That
is the condition the data is held under, not a limitation to work around.

Reproducibility therefore runs through the code, not the data. Both pipelines
are published: a reader with their own GISAID access runs `mat-pcp` on the
UShER tree to produce PCPs and counts, then runs this pipeline on those
outputs. mat-pcp's outputs are plain inputs here — this pipeline never reaches
behind them to a tree, an alignment or a MAT, so a missing input stops the DAG
at the mat-pcp target that produces it.

What may be tracked is a data rule, not a size rule: mutation counts, fitness
effects and selection factors are publishable, raw GISAID sequences are not.
The seven-run `selection_factors.parquet` is kept out because it is 209 MB, not
because of the rules; the single run the paper cites is tracked, at 23 MiB, by
`rule publish_scored_table`.

**That tracked table gives up the 56 clade founder protein sequences**, and so
does every per-panel CSV beside it: a mutation label spells the founder
residue, so `D574V` in clade 22D says 22D's founder carries D at 574, and
`panel_b_within_clade_scatter.csv` alone pins 1,246 of Delta's 1,273. This is
deliberate. A clade founder is reconstructible from the public UShER tree
without GISAID credentials, so it is not the thing the data rules protect —
what stays out is the per-sample material, the PCPs and the leaf sequences.

One case this pipeline produces itself is decided the same way. The
leaf-frequency predictor tallies, per site, which residues a clade's own tree
leaves carry — a 20 × 1,273 table over as many as 1.9 million named GISAID
samples. That table holds no sample name and no sequence, so the clause above
does not forbid it outright; it is kept under `artifact_root` anyway, because
it is a residue-level summary of per-sample material and committing it buys
nothing. What is committed is the correlation table built from it, which
carries a homolog, a predictor, a stratum and a coefficient.

**Two inputs are not GISAID-derived, and both are carried here.** The four
published deep mutational scans — full-spike cell-entry measurements on Delta,
BA.1, XBB.1.5 and KP.3.1.1 — have their own papers and their own repositories.
`data/dms/` holds a copy of each study's released file, pinned by md5 in
`config.yaml`'s `dms` block and documented in `data/dms/PROVENANCE.md`, so a
reader without GISAID access rebuilds those figures from this checkout rather
than reassembling their inputs. The conversion to this pipeline's columns, and
each study's own filtering, are applied in `scripts/dms_published.py` where
they can be read and tested.

`data/antigen/` holds the same arrangement for the published model the
comparison scores against: the antiGen checkpoint and the three sequences it is
fed, pinned by md5 in `config.yaml`'s `antigen` block and documented in
`data/antigen/PROVENANCE.md`. The antiGen source tree is **not** carried — it
is reached through `antigen.checkout`, a path outside this repository — because
its own `pyproject.toml` pins a newer torch than this pipeline runs.

The trained checkpoints are parameters, not sequence, so they are publishable;
they live under `artifact_root` rather than in the repository, because they are
large. Given a checkpoint, the figures rebuild in minutes without retraining.

## Layout

```
config.yaml   every path and pinned number the pipeline uses
Snakefile     terminal targets, named for what they produce
workflow/     the rules, one file per stage
configs/      training-config templates, rendered from config.yaml
scripts/      the pipeline's own code: builders, guards, renderers
notebooks/    marimo notebooks that draw the panels and lay out the figures
dasm2/        a vendored fork of the model-training code (see its PROVENANCE.md)
tests/        what has to hold for the rest of it to mean anything
data/         input the pipeline reads and never writes
results/      the publishable tables it writes, tracked
exploratory/  side analyses that are not part of the pipeline
```

The two data directories split by direction, not by provenance. `data/` holds
input a reader may edit — the era template, the shift candidates, the four
published DMS files. `results/` holds what the pipeline produces and the
repository keeps.

`exploratory/` is the one directory the pipeline neither reads nor writes. Each
subdirectory is a question asked off to the side, with its own README, the
scripts that answer it and the small tables they produced. They import
`scripts/` and the vendored `dasm2`, so they measure what the pipeline
measures, but they never invoke snakemake, never write under `artifact_root`
and never train: scoring reads the published checkpoint. Nothing the paper
says depends on anything here.

## Building a corpus

`snakemake corpora` goes from a mat-pcp run to a training Parquet, in four
stages:

| rule | what it produces |
|---|---|
| `build_neutral` | three `TabularRates` archives — both eras and the pooled fit — from mat-pcp's GLM tables |
| `verify_neutral` | four checks on those archives, including that they reproduce the ones the paper trained against |
| `render_era_map` | the era → clade map, from the committed template in `data/`, with real archive paths filled in |
| `prepare_parquet` | one corpus's Parquet, each row carrying the neutral rates of its clade's era |

**One neutral model, every corpus.** mat-pcp does not refit its GLM per
sampling depth, so `build_neutral` and `render_era_map` take no corpus
argument. That is what pins λ across the ladder: a per-corpus rebuild would let
the neutral model move with the corpus, which is the one thing a depth contrast
cannot afford.

**Nothing here filters, subsamples or reshapes rows.** A sampling-depth arm
*is* a mat-pcp aggregate; the thinning happens there. The Parquet inherits its
source file's row order verbatim, which is what lets `check_corpus` prove a
rebuild gives the paper's fold 0 — the held-out set is a row-positional draw,
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
1,273, and the baseline's parent→parent diagonal is zapped in every forward.
`check_params` builds each model and reproduces the count rather than reading
it back out of the config, because a template transcribed from the wrong arm
renders, validates and trains perfectly well.

**These are the counts the manuscript quotes**, which is why the explanation
lives here rather than in its Methods. The allocated totals are larger --
1,315,284 for `dasm-1077k` against the 1,076,756 above -- and
`<artifact_root>/param_guard/{model}_{corpus}.json` records both for every
run, with the parameter names that never receive a gradient.

**One stopping rule for all seven.** Each run trains until `ReduceLROnPlateau`
has driven the learning rate below 1e-6. `max_train_steps` is a budget guard,
and a run that reaches it instead writes *no record at all* — the trainer
breaks on the budget before it asks whether to stop early — so the signature
is the absence of a stop event, and `check_run` fails on it rather than
letting a capped run reach a figure.

**`early_stopping_min_delta` is 0, for every run.** It is a readout parameter,
not a training one: it reaches best-model tracking and the patience counter,
and nothing else, because the plateau scheduler steps on the raw validation
loss. So two runs at one seed differing only in it descend identically, and
the coarser one is a prefix of the finer one. At 0, the saved best is the
validation curve's own minimum, and the two reference campaigns' thresholds
(1e-4 and 1e-3) are recovered by replaying their rule over the run's recorded
evaluations — which is what `results/runs.csv`'s `*_at_min_delta_*` columns are.

**One run per card, claimed by the training process.** `scripts/gpu_slot.py`
ranks the cards by use, drops any carrying a process this user does not own —
and any whose occupant cannot be identified at all — and takes an exclusive
lock under the artifact root. `scripts/train.py` holds it for exactly as long
as it trains. This is for throughput and for leaving other people's cards
alone, not for memory: attention runs through `scaled_dot_product_attention`
with `need_weights=False`, so no L × L matrix is ever materialised.

**`train` does not declare the test marker**, alone among the rules here. Every
other rule declares the code it runs, including this one; what it leaves out is
`tests_passed.done`, because an edit to any test file would otherwise
invalidate a finished multi-day run. The suite still runs ahead of the corpora,
which is where a regression in the corpus builder would do its damage.

## Scoring

`snakemake scored` reads each run's best checkpoint against mat-pcp's
amino-acid fitness effects on every clade founder, and writes one tidy table:
`selection_factors.parquet` under `artifact_root`, carrying every model. This
is the stage that runs the models forward over the clade founders, and the last
one that needs a GPU.

`rule publish_scored_table` copies out the single run the paper cites, which is
the tracked [`results/selection_factors/dasm-1077k_full_depth.parquet`](results/selection_factors)
described at the top.

**The exclusion comes from mat-pcp, one mutation at a time.** A masked site, a
masked mutation or a founder→reference reversion sets mat-pcp's `exclude` flag
on the nucleotide route, and excluded routes are dropped before any fitness
effect is computed — so a mutation it excluded outright has no row to join
against, arrives with null counts, and leaves at `base_frames`' `dropna`.

Nothing here widens that to the codon. An earlier version did, dropping every
mutation at a site carrying any flag, and it cost 1,067 of the 33,782
comparable mutations across 22 clades — including 21K's R346K at 753 observed
against 22.2 expected, whose counts come from the clade founder's own
background and are untouched by the censoring that flagged its site
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

## Drawing the figures

Every figure is built in two steps, and `snakemake figures` runs them all.
A panel rule reads the scored table, filters it and draws; a composition rule
reads the finished PNGs and lays them out:

| target | panels | composition |
|---|---|---|
| `panels` | `agreement_mutations`, `agreement_ladders`, `l455f_backgrounds`, `l455f_jian` | `compose_agreement`, `compose_l455f` |
| `subsampling` | `subsampling_depth`, `subsampling_ladder` | `compose_subsampling` |
| `dms` | `dms_correlations` | `compose_dms` |
| `dms_strata` | `dms_strata_panels` | `compose_dms_strata`, `compose_dms_strata_supplement` |
| `probabilities` | `probabilities_panels` | `compose_probabilities` |
| `figures` only | `within_clade_outliers`, `background_effects`, `pool_counts` | `compose_within_clade_shifts` |

**Composition is its own rule because the two costs are not alike**: a panel
rule reads a 1.4M-row Parquet per model and takes a minute, and a layout reads
a handful of PNGs and takes a second, so the spacing can be nudged and re-run
without rescoring anything. It also means a figure cannot disagree with its
panels — it *is* their output rather than a second drawing of them.
`scripts/figure_layout.py` solves the dpi rather than taking one: panels arrive
as pixels, so the smallest dpi whose figure still fits the page box is the
largest the figure can legibly be, and nothing is resampled or moved relative
to anything else.

**No rule in this stage takes a GPU.** `rule score` already did the forward
passes over the clade founders, so a panel rule reads
`selection_factors.parquet`, filters it, correlates it and draws. Two of them
do infer — `score_haplotypes` and `l455f_pools` run the published checkpoint
over the haplotypes their pools contain — and both do it on CPU: one
1,273-residue haplotype is 28.5 ms there against 4.7 ms on an idle card, which
does not justify holding a card a training run could be using.

**The count cutoff is the x axis, not a setting.** Figure 3's panels C and E
sweep it, and the clade set is pinned across the rungs — raising the floor
removes the thin, poorly-measured clades first, so a sweep that let each rung
keep its own clades would show r rising with the threshold even if no surviving
clade moved.

### Selections

The agreement figure's rules run per `{selection}`, which names the models they
draw. `figure` is `dasm-1077k`, the largest transformer, which is what every
published panel shows. `fidelity` is `dasm-293k` and `dasm-85k`: a second model
pair through the same code, which no published panel reads. Both come out of
one table and one `groupby`, so the second costs no rescoring, and `snakemake
figures` builds both — a tracked table that nothing in the DAG rebuilds is a
table that quietly stops matching the code beside it.

`fidelity` was originally a port check, and showed the vendored tree computing
what the code it was taken from computed
([#21](https://github.com/matsengrp/viral-dasm/pull/21)).
[#36](https://github.com/matsengrp/viral-dasm/issues/36) then narrowed the
exclusion from the codon site to the mutation, which the older code does not
do, so the two now differ by exactly that — 67,564 mutations at rung 10 against
65,430, over the same 22 clades, at a pooled median r of 0.9662 against 0.9664.
It is a diff to read when that filter changes again, not a live claim about
upstream.

The other figures have one model set each and no selection to expand over:
`dms` and `subsampling` are directory names rather than choices of model.

## The DMS comparison

`snakemake dms_predictors` builds the table behind the deep mutational scan
comparison: two reference predictors, the published model they put in context,
the DASM itself, the DASM read as a mutation probability beside the neutral
rate that reading adds, and the DASM collapsed to one value per site — all
through one join. The last four read `selection_factors.parquet`, so this
target is not model-free; dry-run before building it.

| rule | what it produces |
|---|---|
| `dms_leaf_counts` | one clade's per-site residue tally over its own tree leaves |
| `dms_blosum_scores` | BLOSUM62's score for every cell one assay measured |
| `dms_antigen_scores` | antiGen's log probability for every cell one assay measured |
| `dms_leaf_scores` | the leaf frequency's score for every cell one assay measured |
| `dms_dasm_scores`, `dms_prob_scores`, `dms_site_mean_scores` | the three readings of the DASM |
| `dms_predictor_summary` | the comparison table: each predictor against the DMS, per homolog and stratum |

BLOSUM62 asks how much of a deep mutational scan a general substitution matrix
already explains. The leaf frequency reconstructs every one of a clade's tree
leaves from its own root-to-tip mutation path, tallies residues per site, and
scores a substitution by the log ratio of its residue's leaf count to the clade
founder's at that site. antiGen is a published transformer, scored at the last
rung of its authors' own time-cutoff grid — the only rung whose training split
contains every DMS parent sequence, so every cell is scored by a model that saw
its background. It is fed the WH1 projection of the construct each assay
expressed rather than the clade founder the other two score against; the two
differ only at columns `load_dms` has already dropped, and
`data/antigen/PROVENANCE.md` records why.

**What the leaf predictor measures is closer to the site than to the
substitution**, because a substitution's score is dominated by the parent
residue's own leaf count. A correlation over every measured cell is therefore
substantially a statement about which sites a clade varies at. The table
reports the correlation; reading it as "this ranks which substitution is
tolerated" would be reading more than it says.

**These rules are in the default target.** `rule all` includes them, so a bare
`snakemake` plans the three leaf tallies — about 2.4 GB of path files read, and
3.5 minutes of CPU at a peak of 0.8 GB (measured 2026-11-15: 24E 0:26, 23A
0:47, 21K 2:16, so it is dominated by reading the files) — the first time it is
run in a checkout whose artifact root has none. That is deliberate, for the
reason the selections section gives. The work is CPU-only and never reaches a
GPU, a training run or the scored table.

## The vendored training code

`dasm2/` is a one-way fork of the model-training code, taken once from a
separate repository and never merged with afterwards — the paper's code should
keep reproducing the paper's numbers, which a dependency that moves does not.
[`dasm2/PROVENANCE.md`](dasm2/PROVENANCE.md) records the commit it came from,
what each cut removed, and what the fork still carries without running.

It keeps the name `dasm2`, so each module reads exactly as it does upstream
and a diff against the source commit stays meaningful. `tests/test_fork.py`
enforces what that name would otherwise hide: `dasm2.__file__` has to resolve
inside this repository, and no dependency may install a different one.

**If you have the upstream repository installed, read this before deleting
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

`check_corpus` is the one check the pipeline keeps from the port, and it is not
about the port. Fold 0 is a row-positional draw: the held-out index set comes
from `(seed, num_folds, n_rows)`, and which PCP sits at each index comes from
the Parquet's row order. A rebuilt corpus that reorders rows is therefore a
different 90/10 split of the same data, silently. The guard pins the row order
— a blake2b-128 digest over the per-row `(clade, parent, child)` digests, in
order — so that cannot happen unnoticed.

It runs for all four corpora, `full_depth` included. That one is received from
mat-pcp rather than built here, which makes it the corpus most easily left
unchecked and the one whose split the paper's models were actually trained on.

The checks that proved the vendored tree still trains the model it came from
were port scaffolding, and
[#8](https://github.com/matsengrp/viral-dasm/issues/8) removed them. What they
measured is recorded in the pull requests that ran them. The strongest of them
was a fixed-weights pass: the reference checkpoint, scored in this tree, came
within 1.1e-08 of the loss its own run logged, over exactly 158,137 loss-mask
positions. A full run from scratch here then reached a best validation loss of
7.766302 against the reference's 7.766710 — a difference of 4.1e-4, smaller
than the run's own evaluation-to-evaluation jitter.
