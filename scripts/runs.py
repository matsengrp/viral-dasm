"""The seven training runs, and the budget every one of them gets.

Separate from ``scripts/lib.py`` on purpose. ``lib`` is a declared input of
all six corpus rules, and it is small enough that Snakemake records its
sha256 rather than its mtime, so adding a training constant there would
rebuild every corpus. Nothing below is read by a corpus rule.

Python rather than ``config.yaml`` for the same kind of reason, pointing the
other way: ``config.yaml``'s claim is to be the complete list of paths the
pipeline *reads*, and none of these names a path. The run list is a decision
this repository makes, not an input it receives.

Stdlib only. ``lib`` reads and validates ``config.yaml`` at import, and a
rule that wanted these constants would then have to declare that file too.
"""

from __future__ import annotations

#: Effective parameters per model: allocated, less the parameters that can
#: never receive a gradient. Settled in matsengrp/viral-dasm#1 under "The
#: models the pipeline trains"; ``scripts/check_params.py`` builds each model
#: and reproduces the number rather than trusting this table.
#:
#: The never-gradient term is different in kind for the two families. A
#: transformer allocates ``positional_embedding.weight`` at
#: ``(pe_max_len, d_pos)`` -- 5000 rows by the encoder's own default -- while
#: the corpus's longest amino-acid sequence is 1,273, so the rows beyond it
#: are never indexed. ``ci-484k``'s head allocates ``[1273, 20, 20]`` and
#: zaps the parent->parent diagonal in every forward, so 1,273 x 20 of its
#: entries carry no gradient either.
#:
#: Named by the count because `size_large` and `center` mean nothing outside
#: the scan they came from.
MODELS: dict[str, int] = {
    "dasm-1077k": 1_076_756,
    "dasm-293k": 292_628,
    "dasm-85k": 84_884,
    "ci-484k": 483_740,
}

#: For the ``{model}`` wildcard constraint. Not optional: corpus names contain
#: underscores, so ``{model}_{corpus}`` is ambiguous without it.
MODEL_NAMES = tuple(MODELS)

#: The seven ``(model, corpus)`` pairs, and not the cross product of the two
#: axes. The three transformers and the baseline run at full depth, where the
#: size ladder and the pooling contrast are read; only ``dasm-85k`` runs the
#: thinned rungs, because that is the architecture the sampling-depth ladder
#: trained at every depth.
RUNS: tuple[tuple[str, str], ...] = (
    ("dasm-1077k", "full_depth"),
    ("dasm-293k", "full_depth"),
    ("dasm-85k", "full_depth"),
    ("ci-484k", "full_depth"),
    ("dasm-85k", "cap_50000_0"),
    ("dasm-85k", "cap_20000_0"),
    ("dasm-85k", "cap_5000_0"),
)

#: A budget guard, not the stopping rule. Every run is meant to stop when
#: ``ReduceLROnPlateau`` has driven the learning rate below 1e-6; this is the
#: value all three reference campaigns used, and it does bite -- three arms of
#: the capacity scan stopped on it, which is why that campaign raised its
#: combined arm to 700,000.
#:
#: A run that exhausts it leaves **no log record at all**: ``trainer.py``
#: breaks on ``self._cycle_step >= budget.max_train_steps`` before
#: ``_should_stop_early`` is called, and ``log_early_stopping`` fires only
#: inside the branch that call opens. ``scripts/check_run.py`` detects the
#: absence rather than a reason string.
MAX_TRAIN_STEPS = 500_000

#: Evaluations cadence, in optimizer steps. The reference curves have one
#: point every 2,000 steps, and plateau patience is counted in evaluations,
#: so this sets where a run stops as surely as the learning rate does.
EVAL_EVERY_STEPS = 2_000

#: The CPU wiring smoke: four optimizer steps, with the cadence low enough
#: that the cycle contains an evaluation. ``TrainingConfig`` rejects
#: ``max_train_steps < eval_every_steps`` -- "a cycle must contain at least
#: one evaluation" -- which is why the cadence is a template placeholder at
#: all.
SMOKE_STEPS = 4
SMOKE_EVAL_EVERY_STEPS = 2

#: Peak RSS for one training process, in MB. Snakemake co-schedules on this,
#: and enforces a budget only when the invocation supplies ``--resources
#: mem_mb=``. Sized for the dataloader workers and the row-group cache, which
#: dominate: the model itself is single-digit MB on the host.
TRAIN_MEM_MB = 24_000

#: Peak RSS for the CPU smoke. Much smaller: 400 rows, no cache budget raised
#: above the library default.
SMOKE_MEM_MB = 8_000


def run_id(model: str, corpus: str) -> str:
    """The stem every artifact of one run is named with.

    One function so the rendered config, the run directory, the log and the
    readout row cannot disagree about what a run is called.
    """
    if (model, corpus) not in RUNS:
        raise KeyError(
            f"({model!r}, {corpus!r}) is not one of the seven runs: "
            f"{sorted(RUNS)}. The run list is seven named pairs, not the "
            f"cross product of four models and four corpora."
        )
    return f"{model}_{corpus}"
