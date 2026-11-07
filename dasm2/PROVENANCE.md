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

So: nothing here tracks upstream. A fix landing in `dasm2-experiments`
does not land here unless someone deliberately brings it across and re-runs
the gates. The only tie left between the two trees is `workflow/gates.smk`,
which checks that this code still trains the model `dasm2-experiments`
trained — see issue #2.

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

**After the gates (#4):** what the import closure reaches but never runs.
The closure is inflated by eager registry imports that load every head,
dataset, loss and encoder so their factories populate, so most of it was
reachable without being executed. That cut removed a further 111 files and
left **107 files and 46,088 lines**; a second cut inside
`data/evolutionary_dataset.py`, recorded below, brought the tree to
**107 files and 43,543 lines**. It came *after* Gate 1 deliberately:
once the reference checkpoint scores to seven significant figures, each
deletion is checked rather than argued.

By feature area, it removed the other encoders (ESM2, MPNN light-attention,
site-masked, MLM, identity); the datasets this pipeline has no input for
(HERMES structures, antibody binding, ProteinGym and Tsuboyama DMS, MLM
corruption, MPNN features); the neutral-model family and the molevol serving
it; branch-length optimization; and the heads and losses the paper does not
train. Three registries now state what this repository can build rather than
what `dasm2` could: one encoder, two heads (`SelectionFactorHead` for
`dasm-85k` and `dasm-293k`, `PerSiteParentChildHead` for `ci-484k`), one loss.

It also removed 3Di structural tokens entirely. That one came last and by a
different route: `_load_whichmut_head_data` was typed `neutral_model: None`
once the neutral models were gone, so every non-Parquet arm could only
raise, and `use_tdi` was hardcoded `False` at its only call site. With no
loader left to supply tokens, the encoder's `num_tdi_tokens` embedding, the
`tdi_token_ids` batch key, the two `TrainingConfig` fields and their path
into checkpoint metadata had nothing to do. The embedding is the only part
that could have moved the state dict, and it could not: the reference has
`num_tdi_tokens` unset, so the branch never built a parameter for it, which
is why Gate 1's key-set assertion is what certifies the cut.

`model_catalog.yaml` went with `neutral/` -- it holds only Thrifty and S5F
models. Two things still named it, and both were inert rather than stale:
`TrainingConfig.tdi_tokens_name` resolved a tokens directory through the
catalog using resolution code that had already been deleted, so the key had
become settable and silent; and `data/segment_handlers.py`'s uncalled
`add_neutral_model_outputs_to_pcp_df` carried it in a docstring example.

### What the measurement got wrong

The cut list came from a `sys.setprofile` trace of three runs
(`scripts/trace_executed_modules.py`, `scripts/prune_candidates.py`). A file
counts as having run only if a named `def` in it executed. That rule has a
known failure mode in each direction, and both fired:

* **Dead by the trace, alive in fact.** Nine modules had to be restored
  because live code calls into them on paths the traced runs never took —
  `losses/capability.py`, `molevol/{tensor_ops,neutral_evolution,...}.py`,
  `neutral/base.py` among them. The trace said 128 files were candidates; the
  cut is 107. The gap is the measurement's error bar.
* **Alive by the trace, dead in fact.** `data/functional/codon_neighbor.py`
  executed one function, `_build_preferred_codons()`, at import — reached
  only through a chain that was itself being deleted.

So the trace is evidence, not a work order, and the gates are what decide.

### Checkpoint migrations are gone

`models/key_migration.py` rewrote state-dict keys for two older `dasm2`
layouts: the `TransformerEncoderLayer` → `TransformerBlock` rename (#379) and
the three shared-SF heads whose `prediction_layer` moved into `sf_projection`
(#513). Both ran only for a key that did not already match the model being
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
stays. Anything that reports a result or an argument from elsewhere is
collected in `dasm2/COMMENT-FLAGS.md` for a decision before this repository
goes public.

The prune took out 568 of them on its own — 1,235 flags before, **667**
after — which is why it came before the comment review rather than after.
What is left is in files this pipeline actually runs, so each one needs
reading rather than deleting with its module.

## What is left that the trace calls dead

`data/prune/delete_candidates.txt` still lists 23 files and 6,630 lines, and
most of that list is the measurement's known false positive rather than work
remaining: a module whose contribution is a class, an enum or a `Protocol`
has no named `def` to execute, so it reads as dead however load-bearing it
is. `positional_encoding.py` is the clearest case — deleting it is an
`ImportError`, and it still appears on the list.

Three things on it are real and are deliberately not in this cut:

* **`models/core/attention.py`** (1,356 lines) is the flash-attention
  decision, which belongs with the training-run scheduling in #1 rather than
  here.
* **`training/checkpoint_selection.py`** (567 lines) and
  `data/{manifests,registry_schema}.py` (956), each reached only from a live
  module's import line. Each is roughly five hundred lines entered through a
  single name, which is the shape of a module that wants replacing by the one
  constant its caller uses rather than deleting whole. That wants reading,
  not grepping.

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
  pattern missed. Re-tracing after the cut leaves
  `data/prune/delete_candidates.txt` byte-identical, which is the same fact
  measured a second way.
