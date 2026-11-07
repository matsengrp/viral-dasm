# Where this package came from

This directory is a **fork** of the `dasm2` package, taken once from
`matsengrp/dasm2-experiments` and never merged with afterwards.

| | |
|---|---|
| source repository | `matsengrp/dasm2-experiments` (private) |
| source commit | `68e41df5c2b66198fb0d992181ede39788fa160e` |
| commit date | 2026-10-30 |
| taken on | 2026-11-06 |
| what was taken | the 207 modules `import dasm2.train_command` loads, plus `model_catalog.yaml` |
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
reachable without being executed. That cut removed a further 107 files and
left 111 files and 48,013 lines. It came *after* Gate 1 deliberately: once
the reference checkpoint scores to seven significant figures, each deletion
is checked rather than argued.

By feature area, it removed the other encoders (ESM2, MPNN light-attention,
site-masked, MLM, identity); the datasets this pipeline has no input for
(HERMES structures, antibody binding, ProteinGym and Tsuboyama DMS, MLM
corruption, MPNN features); the neutral-model family and the molevol serving
it; branch-length optimization; and the heads and losses the paper does not
train. Three registries now state what this repository can build rather than
what `dasm2` could: one encoder, two heads (`SelectionFactorHead` for
`dasm-85k` and `dasm-293k`, `PerSiteParentChildHead` for `ci-484k`), one loss.

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

The prune took out 534 of them on its own — 1,235 flags before, **701**
after — which is why it came before the comment review rather than after.
What is left is in files this pipeline actually runs, so each one needs
reading rather than deleting with its module.
