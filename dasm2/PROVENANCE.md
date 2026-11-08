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

## What was dropped on the way in, and what is still to drop

Dropped already: everything `import dasm2.train_command` does not reach —
118 of the package's 325 modules, including the analysis dashboards, the
other CLI commands, and the test fixtures.

Still carried, and still to go: the closure is inflated by eager registry
imports that load every head, dataset and loss, so most of what remains is
reachable but never run. The prune is deliberately *after* the gates, not
before — once Gate 1 scores the reference checkpoint to six decimal places,
every deletion can be checked rather than reasoned about.

Also still here: five hardcoded `/fh/fast/...` cluster paths, in `config.py`,
`data/discovery.py` and `data/local.py`. All of them are group-shared
directories rather than anyone's home, and viral-dasm reaches none of them —
`config.yaml` names every path this pipeline reads, so `dasm2`'s own
artifact-root and registry defaults are never consulted. They go with the
code that consults them.

## The comment flags

`dasm2`'s comments carry a great deal of detail about work that is not this
paper: results, measurements and design arguments from other projects, and
1,522 references to private `dasm2-experiments` issue numbers. Generic
explanation stays. Anything that reports a result or an argument from
elsewhere is collected in `dasm2/COMMENT-FLAGS.md` for a decision before
this repository goes public.
