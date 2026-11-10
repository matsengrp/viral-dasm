# Where this package came from

This directory is a **fork** of the `dasm2` package, taken once from
`matsengrp/dasm2-experiments` and never merged with afterwards.

| | |
|---|---|
| source repository | `matsengrp/dasm2-experiments` (private) |
| source commit | `68e41df5c2b66198fb0d992181ede39788fa160e` |
| commit date | 2026-10-30 |
| taken on | 2026-11-06 |
| what was taken | the 207 modules `import dasm2.train_command` loads |
| size at import | 207 files, 79,490 lines |

## Why a fork and not a dependency

The paper's code should keep reproducing the paper's numbers. A dependency
that moves does not, and publishing `dasm2` as a versioned package would
make this repository describe an analysis a reader cannot run without a
second repository they also have to be given.

So: nothing here tracks upstream. A fix landing in `dasm2-experiments` does
not land here unless someone deliberately brings it across and re-establishes
that this tree still trains the same model. Issue #2 did that once, with a set
of comparisons against `dasm2-experiments` runs on the same corpus, fold and
seed; issue #8 then removed them, because a one-time claim about a port
belongs in the record of the port rather than in the pipeline. What they
measured is in the pull requests that ran them. There is no live tie between
the two trees.

## It is still named `dasm2`

The package keeps its name, so every module reads exactly as it does
upstream and a diff against the source commit is meaningful. That means
"no upstream `dasm2`" cannot be enforced by making the import fail.
`tests/test_fork.py` enforces it the other way round: `dasm2.__file__` must
resolve inside this repository, and no entry in `pyproject.toml`'s
dependency list may pull a `dasm2` in.

## What was dropped

Two cuts, for different reasons.

**On the way in:** everything `import dasm2.train_command` does not reach —
118 of the package's 325 modules, including the analysis dashboards, the
other CLI commands, and the test fixtures. That left 218 files and 83,937
lines.

**Once the port could be checked (#4):** what the import closure reaches but never runs.
The closure is inflated by eager registry imports that load every head,
dataset, loss and encoder so their factories populate, so most of it was
reachable without being executed. That cut removed a further 111 files and
left **107 files and 46,088 lines**; a second cut inside
`data/evolutionary_dataset.py`, recorded below, brought the tree to
**107 files and 43,543 lines**; a third, recorded below, brought it to
**100 files and 40,637 lines**. It came deliberately *after* the check that
scores the reference checkpoint in this tree: once that reproduces to seven
significant figures, each deletion is checked rather than argued.

By feature area, it removed the other encoders (ESM2, MPNN light-attention,
site-masked, MLM, identity -- see *The identity encoder came back*, below);
the datasets this pipeline has no input for
(HERMES structures, antibody binding, ProteinGym and Tsuboyama DMS, MLM
corruption, MPNN features); the neutral-model family and the molevol serving
it; branch-length optimization; and the heads and losses the paper does not
train. Three registries now state what this repository can build rather than
what `dasm2` could: two encoders (`TransformerEncoder` for the three models
named by parameter count, `IdentityEncoder` for the one with no encoder), two
heads (`SelectionFactorHead` for the transformers, `PerSiteParentChildHead`
for `ci-484k`), one loss.

It also removed 3Di structural tokens entirely. That one came last and by a
different route: `_load_whichmut_head_data` was typed `neutral_model: None`
once the neutral models were gone, so every non-Parquet arm could only
raise, and `use_tdi` was hardcoded `False` at its only call site. With no
loader left to supply tokens, the encoder's `num_tdi_tokens` embedding, the
`tdi_token_ids` batch key, the two `TrainingConfig` fields and their path
into checkpoint metadata had nothing to do. The embedding is the only part
that could have moved the state dict, and it could not: the reference has
`num_tdi_tokens` unset, so the branch never built a parameter for it, which
is why the checkpoint check's state-dict key-set assertion is what certifies
the cut.

`model_catalog.yaml` went with `neutral/` -- it holds only Thrifty and S5F
models. Two things still named it, and both were inert rather than stale:
`TrainingConfig.tdi_tokens_name` resolved a tokens directory through the
catalog using resolution code that had already been deleted, so the key had
become settable and silent; and `data/segment_handlers.py`'s uncalled
`add_neutral_model_outputs_to_pcp_df` carried it in a docstring example.

### What the measurement got wrong

The cut list came from a `sys.setprofile` trace of three runs, taken by two
scripts that were removed with the rest of the port tooling in #8. A file
counted as having run only if a named `def` in it executed. That rule has a
known failure mode in each direction, and both fired:

* **Dead by the trace, alive in fact.** Nine modules had to be restored
  because live code calls into them on paths the traced runs never took —
  `losses/capability.py`, `molevol/{tensor_ops,neutral_evolution,...}.py`,
  `neutral/base.py` among them. The trace said 128 files were candidates; the
  cut is 107. The gap is the measurement's error bar.
* **Alive by the trace, dead in fact.** `data/functional/codon_neighbor.py`
  executed one function, `_build_preferred_codons()`, at import — reached
  only through a chain that was itself being deleted.

So the trace was evidence, not a work order; what decided each cut was
whether the tree still trained the same model afterwards.

### The identity encoder came back (#14)

`models/core/identity.py`, `IdentityEncoderConfig` and the registry's
`IdentityEncoder` entry went out with the other encoders, and they should not
have: they are not an encoder this repository does not train, they are how it
expresses *having no encoder*.

`ci-484k`, the context-independent baseline the paper reads its transformers
against, is `PerSiteParentChildHead` with nothing in front of it. That head
ignores the encoder output entirely -- its selection factor is a function of
(site, parent amino acid, child amino acid) -- but `CoreModel.forward` calls
`self.encoder(...)` unconditionally before dispatching to heads, so the model
still needs an object there. `IdentityEncoder` is a zero-parameter module
returning a zero tensor nothing reads. Pairing the head with a
`TransformerEncoder` instead would put a transformer's parameters in the
optimizer with no path from them to the loss.

The head, its config, its `HEAD_REGISTRY` entry and the whichmut loss all
survived the prune; only the encoder side went, which left `type:
IdentityEncoder` raising a pydantic `ValidationError`. What came back is the
module, the config class, that class in `ModelConfig.encoder`'s union, and the
registry entry -- and nothing else from the same commit: `ESM2EncoderConfig`
and `MPNNLightAttentionEncoderConfig` were in the same union and stay out.

The restored comments were read rather than carried. The module's docstring
listed five other encoder-free heads this repository does not have, cited
another project's model as prior art, and quoted a dead-parameter count
measured on configs that are not here. The reason is kept; the examples a
reader of this repository cannot look up are not. `ModelConfig.encoder` is a
plain union rather than a discriminated one, as it was upstream, so a config
that omits `type` still validates against the transformer's default.

### Checkpoint migrations are gone

`models/key_migration.py` rewrote state-dict keys for two older `dasm2`
layouts: the `TransformerEncoderLayer` → `TransformerBlock` rename, and the
three shared-SF heads whose `prediction_layer` moved into `sf_projection`.
Both ran only for a key that did not already match the model being
loaded into, and all 64 keys of the `dasm-85k` reference checkpoint match as
written, so neither ever fired here. The rewrites were also not idempotent —
applied to a current key, the block migration yields
`feed_forward.feed_forward.linear1` — so carrying them would have advertised
support for layouts this repository has never read. `load_checkpoint` now
rejects a key that names no parameter of the model.

### Hardcoded cluster paths

Three hardcoded `/fh/fast/...` cluster paths remain, in `config.py` and
`data/local.py`; `data/discovery.py`'s went with four of its five manifest
descriptors. All are group-shared directories rather than anyone's home, and
viral-dasm reaches none of them — `config.yaml` names every path this
pipeline reads, so `dasm2`'s own artifact-root and registry defaults are
never consulted.

## The comment flags

`dasm2`'s comments carry a great deal of detail about work that is not this
paper: results, measurements and design arguments from other projects, and
references to private `dasm2-experiments` issue numbers. Generic explanation
stays. Anything that reported a result or an argument from elsewhere was
collected into a generated inventory and decided before this repository goes
public; the inventory and the script that produced it were removed in #8, once
the count below had reached its final value.

The prune took out 568 of them on its own — 1,235 flags before, **667**
after — which is why it came before the comment review rather than after.
What was left was in files this pipeline actually runs, so each one needed
reading rather than deleting with its module.

### The fourth cut: the comments themselves (#4)

The review is done, and the count is **1,235 → 66**.

**Every reference to another project's issue tracker is gone** — 286 across
53 files, plus two in this document. Where the comment was only a pointer, it
went; where the issue held the reason, the reason is now written in the
comment. `#1234` was a dead link for every reader of this repository.

Three of them were worse than dead links, because they were *runtime* strings
that told a reader to run something this repository does not contain:

| raised by | told the reader to run |
|---|---|
| `_assert_supported_serialization_version` | `pixi run python scripts/migrate_checkpoint_to_v5.py` |
| `_assert_unambiguous_sf_projection` | `experiments/compara-training/patch_checkpoint_wiggle.py` |
| the inert-row guard in `evolutionary_dataset.py` | two corpus-builder scripts, by name |

Each now states the condition and a remedy available here. The second also
carried a thirty-line docstring reconstructing when that script's incidents
happened in another project, down to merge dates and a worked example.

Measurements from other projects went the same way: a Thrifty lookup table's
size and its per-step NCCL cost, a dated observation from a four-dataset
training run, a host filled to ~738 GB, a count of clamped sites on a Compara
corpus, a 7x data-load slowdown, a ~100x speedup, a 340 GB/cycle leak. In each
case the engineering reason is kept and the number is dropped, because the
number is what does not transfer.

A smaller class is more insidious: comments that explained the code through a
class this fork had already deleted. `CoreModel.__init__` and
`Trainer.export_metadata` both justified tracking the freeze switch separately
from `requires_grad` by appealing to `ESM2Encoder`, which is referenced seven
times in the tree and defined nowhere. The reason is sound and is kept; it is
now stated without the example a reader cannot look up.

**What the 66 remaining flags are.** None is a private issue reference. They
divide in two, and neither is a leak:

- Scanner false positives — "naive buffer/row-group ratio", "measured at
  insert", "improved by more than min_delta".
- The *names* of code still in the tree: `ESM2Tokenizer`,
  `AntibodySegmentHandler`, `MPNN_WHICHMUT_HEAD_TYPES`. ESM2 is a published
  model and a handler is named for what it handles; naming them is not a
  disclosure. These resolve by deleting the code, not by renaming it, and that
  is the next cut rather than this one.

Measured rather than asserted: comparing the non-comment, non-string token
stream of every changed file against the previous commit, 52 of 53 were
byte-identical in code. The one that differed was
`serialization/checkpoints.py`, by exactly its dead block and that block's
now-unused `Final` import.

## What is left that the trace calls dead

The trace's final candidate list held 23 files and 6,630 lines, and most of
that list was the measurement's known false positive rather than work
remaining: a module whose contribution is a class, an enum or a `Protocol`
has no named `def` to execute, so it reads as dead however load-bearing it
is. `positional_encoding.py` is the clearest case — deleting it is an
`ImportError`, and it still appears on the list.

Three things on it are real and are deliberately not in this cut:

* **`models/core/attention.py`** (1,356 lines) is the flash-attention
  decision, which belongs with the training-run scheduling in #1 rather than
  here.
Both have since been cut; see "The third cut" below.

## The second cut: the dataset factories (#4)

`data/evolutionary_dataset.py` went from 6,224 lines to 3,687 — 41 top-level
definitions, the DASM and AAO loss families and the DMS, HERMES, MPNN and
structural factories. viral-DASM builds a whichmut dataset from a Parquet
corpus and nothing else.

The cut set is a reachability closure, not the trace's word. The trace seeds
it; a static walk of the call graph adds anything a live definition names,
and anything another module imports *executably*. That last qualifier is why
the round was safe: both `BranchLengthDatasetBundle` importers sit under
`if TYPE_CHECKING:`, and every apparent `create_evolutionary_dataset` importer
was a `>>>` line in a docstring. Neither executes, so neither keeps a
definition alive.

Two things recorded here because they were predicted wrongly first:

* The closure found **2,522 lines**, where a trace-only count of the same file
  had said 3,529. The closure is the stricter instrument — it keeps whatever a
  live definition so much as names, string annotations included — and the
  smaller number is the one to trust.
* The cut **frees no module**. It was expected to orphan
  `molevol/{neutral_evolution,codon_preference,aao_neutral_rates,tandem}.py`
  and `neutral/codon_boundary.py`; all of them are still imported by
  `data/transforms.py` and `neutral/base.py`, which are live, and
  `mutation_matrices.py` is reached by a relative import an earlier grep
  pattern missed. Re-tracing after the cut left the candidate list
  byte-identical, which is the same fact measured a second way.

## The third cut: single-entry modules and the manifest registry (#4)

Three modules were reached from a live module's import line and nothing else,
which is the shape of a file that wants replacing by the one name its caller
uses rather than deleting whole.

* **`training/checkpoint_selection.py`** (567 lines) exported thirteen public
  names, of which exactly one was consumed anywhere: the string `"periodic"`.
  Upstream it shared that marker between the hook that writes step-cadence
  checkpoints and the readers that later pick a final checkpoint out of a
  finished run's directory. This fork has no such reader -- it names the
  checkpoint it wants -- so `CheckpointSaveHook.PERIODIC_TAG` owns the marker.
* **`losses/q_scale_defaults.py`** (21 lines) held six scalars and no code,
  imported by nothing. They are fitted values from BCR somatic-hypermutation
  work: a triple "calibrated by held-out LL maximisation under fixed
  pre-trained Thrifty rates" and an "S5F-HH fit on the pooled 30k Tang+SHMOOF
  out-of-frame subset". Dead and leaking, so deleted rather than edited.
* **`losses/dms_matrix.py`** (283 lines) was reachable only so
  `training/metric_providers.py` could name `DMSMatrixCalibrationLoss` in an
  `excluded_loss_classes` tuple. No factory branch builds that loss here, so
  the exclusion could never fire.

**The manifest registry** -- `data/{yaml_registry,discovery,manifests,
registry_schema}.py` and the parts of `data/registry.py` that consulted them,
about 2,040 lines -- was live but vacuous. `registered_nickname_or_none` sits
on the Parquet load path and feeds the seam check, and returned `None` on every
call, because this repository names every input by path.

It was not inert, though. With a developer's `~/.config/dasm2/config.yaml`
present it scanned that clone's manifests and found 185 entries, so what the
code did depended on a file outside the repository. That is a reproducibility
defect in its own right, and `README.md` promised the opposite.

The seam stamp is what makes removing it provably safe, and it was captured
before any edit rather than asserted after:

| | before | after |
|---|---|---|
| nickname | `path_b2c98048` | `path_b2c98048` |
| `row_preprocessing_inputs` | `4f53cda1...b945` | `4f53cda1...b945` |
| `row_preprocessing_version` | `1` | `1` |

That hash is sha256 over an empty naughty list, and `seam_identity`'s own
docstring already named `None` as the right answer for "the viral PCP corpora,
whose naughty list is vacuously empty". The cut makes it the only answer rather
than one the code reaches by querying a registry that happens to be empty.
Every Parquet file stamped before the cut still verifies.

Naughty clonal families are a BCR concept with no viral counterpart, so
`preprocess_rows_for_split` and `filter_naughty_clonal_families` went with the
registry that fed them. One behaviour changed deliberately: a corpus arriving
without an `in_train` column is now an error, where upstream derived the split
from a registry's `holdout_samples`. A split that lives outside the corpus does
not travel with it to a reader bringing their own GISAID access.


## What came back for the corpus build (#6)

The cuts above were made against a trace of **training**, which reads a
Parquet that already carries `nt_rates` and `nt_csps`. Building that Parquet
is a different code path and was never traced, so three modules came back from
the same source commit:

| module | lines | why it was not in the trace |
|---|---:|---|
| `neutral/tabular_rates.py` | 896 | fits and serves the neutral rates the builder writes into the Parquet; training reads the columns, never the model |
| `neutral/scov2_rates.py` | 584 | assembles one gene's archive from mat-pcp's GLM outputs |
| `data/parquet_writer.py` | 141 | `ChunkedParquetWriter`, which is what writes the file |

Two of the three are forked rather than copied.

**`scov2_rates.py` resolves no manifests.** `resolve_run_root`,
`resolve_build_inputs` and `_region_checked_config` are not vendored, and nor
is the `from dasm2.data.viral_fitness_registry import ...` that fed them.
Vendoring them would have meant restoring `data/viral_fitness_registry.py` and
the `discovery`, `manifests` and `registry_schema` modules it imports — the
~2,200 lines the third cut removed — and reversing `seam_identity`'s new
contract. Callers hand in a `NeutralBuildInputs` built from `config.yaml`
instead, which is what the README promises.

One guard is weaker for it, and that is the cost rather than a detail:
`_region_checked_config` refused to pair one region's founder with another's
rates by comparing two manifests' `region` fields, and there are no manifests
left to compare. What replaces it is structural — `config.yaml` names one
`mat_pcp.gene` under one `mat_pcp.root`, so the founder and the rate tables
come from the same run by construction — plus `validate_gene`, which still
rejects a gene with no coordinate blocks.

**`tabular_rates.py::to_config` refuses instead of returning.** It built a
`TabularRatesNeutralModelConfig`, and the second cut deleted every
neutral-model config class: `TrainingConfig.neutral_model` is typed to admit
nothing but `None` here, because viral-DASM takes neutral rates from
precomputed columns and configures no model. The method could not simply be
dropped — `AbstractNeutralModel` declares it `@abstractmethod`, so removing
the override leaves an abstract class and every corpus build fails at
construction. It now raises `NotImplementedError` naming the archive path,
which is how this fork refers to a neutral model. `tests/test_vendored_corpus_build.py`
holds both halves: that the class is still concrete, and that the method
refuses rather than failing on a missing import.

One check outside this directory got weaker with it, and it belongs in the same
record. `verify_era_neutral_models.py` used to assert that a freshly built
archive reproduced a *separately built* prior campaign's, byte for byte --
equality that was evidence of an independent rebuild. That campaign lives in
the private repository, so the assertion pointed at a path a reader does not
have, and a check a reader cannot run makes no assertion at all. It now
compares against digests recorded in `config.yaml`'s `neutral_archives`. Those
were read out of the archives the paper's models were trained against, so the
check is still a reproduction — but of a recorded answer rather than of a
second independent build. What keeps it from being circular is the round-trip
check beside it: the archive's rate table is rebuilt from the GLM input with
`select_gene_rates` and compared, so a rate-selection change is caught by
derivation rather than by a recorded answer.

The first build in the new venv showed those digests had to be taken over a
*canonical rendering* of `rates_table.csv` and not over its bytes. All three
archives failed the byte comparison while the other three checks passed, and
`metadata.yml` and `secondary_struct.csv` were byte-identical throughout.

The cause was neither the model nor the inputs, and it was not arithmetic.
`select_gene_rates` performs none on rates: it masks rows out of the GLM's
master table and sorts them, so both builds ship the same doubles. The pinned
archives rendered them at about 16 significant digits; this environment writes
Python's shortest round-trip repr, and this build's rate strings are the master
table's own strings. Parsed back, the shorter text lands up to 4.43e-14
relative away — worst case the omicron `TG/GTA` unpaired=0 rate. An earlier
revision of this document blamed NumPy summation order, which is wrong, and the
error is recorded here because a reader who checks `select_gene_rates` for the
sum would find none and lose confidence in the gate.

So the digest now covers the table with rows sorted by key and each rate
rendered to 8 significant digits, beside the round-trip check's own `rtol=1e-9`
and far tighter than a rate-selection change, which takes a different row of
the GLM table and moves rates by whole percent. The closest of these 1,152
rates sits 1.3e-11 relative from an 8-digit rounding boundary, roughly 300x the
rendering spread; six through nine digits all clear it, while at eleven and
twelve the nearest rate is within one part in 1e15 of a boundary, so agreement
there would be luck. That margin is a minimum-gap order statistic over this
particular rate table and would move if the GLM were refit — read it as a
measurement, not a designed tolerance. The
recorded digests are computed from the paper's archives and reproduced by the
rebuild here, which is what the check claims. A byte digest would have failed
again at every `uv lock --upgrade`, and a gate that fires on its own toolchain
teaches a reader to re-pin it — which is how a real failure gets waved through.

`scripts/audit_censor.py` also lost a branch. It could audit the global mask at
either nucleotide or codon granularity; `config.yaml` names only the nucleotide
mask file, and the training templates censor with that same key, so the codon
branch read a file this pipeline does not configure. Auditing a mask no run
uses would over-report the footprint while looking like a passing gate, so the
branch is gone and the audit reads the key the templates name.

## What came back for scoring (#16)

The cuts above were traced against **training**, which never reads a fitness
table. Scoring a finished checkpoint against mat-pcp's effects is a third code
path, and `dasm2/analysis/` was absent entirely. Three files came from the same
source commit:

| module | upstream | here | what was taken |
|---|---:|---:|---|
| `analysis/__init__.py` | 1 | 1 | verbatim |
| `analysis/viral.py` | 1,077 | 1,089 | every definition; comments revised |
| `analysis/viral_fitness_eval.py` | 2,711 | 407 | 12 of its definitions |

`viral.py` arrives whole because it is frame-in, frame-out: its only `dasm2`
import is `sequence.constants`, it reads no path and loads no model, and the
figure readouts need the same functions the scoring rule does. Twelve of its
definitions belong to readouts this repository has not written yet; they are
kept rather than cut so the file still diffs against the source commit.

`viral_fitness_eval.py` is pruned hard, and along the two seams the rest of
this fork already established.

**No manifest resolution.** `MatPcpRun` and `FitnessSource` resolve data-central
*nicknames* — `get_viral_pcp_config`, `list_fitness_datasets`, and a
`pcp_nickname(gene, clade)` that composes `s2SpikeGisaid21J-2026-08-06-v12` — so
vendoring them would restore the `viral_fitness_registry`, `discovery` and
`manifests` modules the third cut removed and reverse `seam_identity`'s
contract. `config.yaml` names the fitness directory, the pooled table and the
founder directory; `scripts/check_clades.py` reads the clade roster out of those
directories rather than restating it.

**No checkpoint discovery.** `discover_checkpoints` assumes a
`{dir}/{gene}/{run_id}/` layout and classifies `dasm2`'s own filenames, treating
an unrecognised name as a legacy final save. `lib.select_checkpoint` already
knows this pipeline's layout and raises instead of guessing.

The run axis, the shift families and the effect-error families are not here.
They compute per-figure readouts, so they belong with the figures.

### The comment pass

Measured the same way as the fourth cut: comparing the two files' definitions
as ASTs with docstrings stripped, **all 34 are code-identical to the source
commit** — 22 in `viral.py`, 12 in `viral_fitness_eval.py`. Every change is a
comment or a docstring.

No private issue reference survived: `viral.py` carried none, and of
`viral_fitness_eval.py`'s nine, two sat in its module docstring and six in
definitions that were not taken. The ninth named an experiment's site-class
split, and is now stated as what the argument does.

Three kinds of comment were rewritten rather than dropped.

**Measurements that replicate here were re-measured here.** `FitnessEstimate`
justified reading mat-pcp's shipped `delta_fitness_ratio` instead of
recomputing it, on the strength of a v9 check: 9,140 clade-21J Spike rows, a
max deviation of 8.9e-16 from the pseudocount formula, and the two estimates
correlating at 0.984. Re-run against this repository's own v12 tables: 9,140
rows, agreement to within one ulp, and 0.9825. The claim is this repository's
now, not a number carried in from a dataset it does not read.

The middle number is stated as a tolerance rather than as a figure, because a
max-over-9,140 rounding residual carries no information beyond "equal to float
precision" — 8.9e-16 and 2.22e-16 are not two measurements of the same thing,
and printing both invites a reader to think a precision moved. What is worth
recording is that the comparison discriminates at all: at pseudocount 0.25,
0.75 or 1.0 the worst disagreement is 0.41 to 0.69, so an ulp is a real result
about the 0.5 and not a vacuous one. The row count is equal across versions for
a structural reason, not because the number was copied: the row set is the
single-nucleotide-accessible (site, mutant_aa) pairs off the 21J Spike founder,
which no count filter touches, while the depth behind it did change (total
expected count 485,015 in v9 against 525,368 in v12).

**Data-vintage history became what v12 carries.** The alias table explained
itself through mat-pcp's v1/v4/v9 schema changes. What matters here is narrower
and sharper: the v12 tables spell the count column `expected_count` while every
function below reads `predicted_count`, and `filter_merged_fitness` gates on a
count column only `if ... in df.columns` — so an un-normalized frame passes the
count filter on `actual_count` alone, with no error and a plausible row count.
The comment now says that.

**One rationale was false here.** `SelectionFactorSource._model` takes the
dense-attention fallback because "these checkpoints were trained with flash
attention". Every config in this repository sets `flash_attention: false`, so
the fallback is belt and braces rather than the load path, and the comment says
so.

Two comments gained a warning the upstream file states elsewhere. The cache in
`SelectionFactorSource` is keyed `(gene, run_id, clade)` and carries no
sequence, so asking it for two parent haplotypes under one clade returns the
first twice — silently. And `merge_and_filter` is an **inner** join, while the
table this repository writes is a left join from the model grid; the docstring
now says which of the two a caller is getting.
