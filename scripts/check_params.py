# ruff: noqa: I001 -- import order in this file is load-bearing; see below.
"""Build a rendered config's model and check it against the name it was given.

Each of the four models in this pipeline is **named by its effective
parameter count**: what it allocates, less the parameters that can never
receive a gradient. The count is how a reader of the paper identifies a
model, so it is the thing to check, and checking it means building the model
rather than reading the config's numbers back out.

What this catches is the realistic way to train the wrong model for five
days: a template transcribed from the wrong arm. A geometry that is off by
one field renders, validates and trains perfectly well; its parameter count
does not match the name on the run.

**The never-gradient term is derived from the config, not tabulated.** For a
transformer it is ``(pe_max_len - corpus_max_aa_seq_len) * d_pos``, over the
rows of ``encoder.positional_embedding.weight`` that no position in this
corpus ever indexes. 5,000 is only ``EncoderConfig.pe_max_len``'s default and
no template states it, so writing it down here would be a second copy of a
library default -- the failure mode ``tests/test_training_defaults.py``
exists to prevent. For ``ci-484k`` the term is the parent->parent diagonal of
its ``[max_aa_seq_len, 20, 20]`` table, ``max_aa_seq_len * 20``, which
``zap_predictions_along_diagonal`` overwrites on every forward.

**It also makes the assertions the deleted ``check_rendered_config.py`` made
about the paths.** ``render_config.py`` checks only that the five censoring
paths exist, not that they are the ones ``config.yaml`` names, and losing the
reversion source -- roughly 40x the other two on Spike -- is the most
damaging thing that can go wrong here without anything failing. A run that
censors something else trains fine and reports a number for a condition it
did not run.

GISAID: reads a corpus's footer metadata for one integer; no rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first: it is what puts this repository's root on `sys.path`, and
# `import dasm2` below is served by the venv's editable install -- which
# points at another checkout -- until it has.
from lib import (
    CONFIG,
    assert_dasm2_is_this_repo,
    corpus,
    corpus_max_aa_seq_len,
    mat_pcp_path,
)
from runs import MODELS

# `dasm2` next, above anything that pulls pyarrow in. Arrow fixes its C++
# default memory pool at first import and `dasm2/__init__.py` steers it off
# mimalloc; whichever import lands first wins. See
# `tests/test_import_order.py`.
import dasm2  # noqa: F401  (imported for its side effects)

from dasm2.config import load_config_from_yaml
from dasm2.models.factory import create_model_from_config
from dasm2.sequence.constants import NUM_AMINO_ACIDS

assert_dasm2_is_this_repo()

#: The transformer parameter whose rows beyond the corpus length are dead.
POSITIONAL_EMBEDDING_KEY = "encoder.positional_embedding.weight"

#: The per-site head whose parent->parent diagonal is zapped every forward.
PER_SITE_FACTORS_KEY = "log_selection_factors"


def never_gradient_params(model, max_aa_seq_len: int) -> tuple[int, list[str]]:
    """Parameters the corpus and the head between them can never reach.

    Returns the count and the names of the tensors it came from, so a key
    rename shows up as an empty list rather than as a silently-zero term.
    """
    dead = 0
    sources: list[str] = []
    for name, param in model.named_parameters():
        if name.endswith(POSITIONAL_EMBEDDING_KEY):
            # Rows past the corpus's longest amino-acid sequence are never
            # indexed, so they never receive a gradient.
            dead += (param.shape[0] - max_aa_seq_len) * param.shape[1]
            sources.append(name)
        elif name.endswith(PER_SITE_FACTORS_KEY):
            # The parent->parent entry at every site: overwritten with 0 by
            # `zap_predictions_along_diagonal` before the loss sees it.
            dead += param.shape[0] * NUM_AMINO_ACIDS
            sources.append(name)
    return dead, sources


def check_paths(config) -> list[str]:
    """The five censoring fields, against ``config.yaml`` and against disk."""
    failures = []
    for key in (
        "censored_mutations_file",
        "censored_clades_file",
        "masked_nt_sites_file",
        "reference_sequence_file",
    ):
        expected = mat_pcp_path(key)
        value = getattr(config, key)
        if value is None or Path(value) != expected:
            failures.append(
                f"{key} is {value!r}, not {str(expected)!r}, so this run "
                f"censors something other than what the paper's runs censored."
            )
        elif not Path(value).exists():
            failures.append(f"{key} points at {value}, which does not exist.")
    expected_mode = CONFIG["mat_pcp"]["censor_mode"]
    if str(config.censor_mode).rsplit(".", 1)[-1].lower() != expected_mode:
        failures.append(
            f"censor_mode is {config.censor_mode!r}, not {expected_mode!r}."
        )
    return failures


def check_raw(raw: dict) -> list[str]:
    """Two blocks a rendered config must not carry."""
    failures = []
    if raw.get("neutral_model") is not None:
        failures.append(
            "this config carries a neutral_model block. The prepared corpus "
            "already carries nt_rates and nt_csps, so the model would be "
            "loaded and then never consulted -- the config would describe a "
            "choice the run did not make."
        )
    if raw.get("scheduler") is not None:
        failures.append(
            "this config states a scheduler block. Every run in this "
            "repository stops when the plateau default's reductions cross "
            "the learning-rate floor, and an explicit block here would have "
            "to be checked against that path rather than assumed."
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    max_aa_seq_len = corpus_max_aa_seq_len(corpus(args.corpus).path)
    raw = yaml.safe_load(args.config.read_text())
    config = load_config_from_yaml(args.config)

    # CPU: this builds the model to count it and never runs a batch through
    # it, so it must not wait on, or occupy, a card a training run wants.
    model = create_model_from_config(config.model, device="cpu")
    allocated = sum(p.numel() for p in model.parameters())
    dead, sources = never_gradient_params(model, max_aa_seq_len)
    effective = allocated - dead
    expected = MODELS[args.model]

    print(f"model:   {args.model} on {args.corpus}")
    print(f"corpus max_aa_seq_len: {max_aa_seq_len}")
    print(
        f"  parameters: {allocated:,} allocated, {dead:,} never-gradient, "
        f"{effective:,} effective (expected {expected:,})"
    )

    failures = []
    if not sources:
        failures.append(
            f"no never-gradient parameters were found. Every model here has "
            f"some -- a transformer's oversized positional table, or the "
            f"per-site head's zapped diagonal -- so either "
            f"{POSITIONAL_EMBEDDING_KEY!r} and {PER_SITE_FACTORS_KEY!r} have "
            f"both been renamed, or this is not one of the four models."
        )
    if effective != expected:
        failures.append(
            f"{args.model} has {effective:,} effective parameters, not the "
            f"{expected:,} its name claims. The name is how the paper "
            f"identifies this model, so a run under this config would be "
            f"reported as a model it is not."
        )
    failures += check_paths(config)
    failures += check_raw(raw)

    report = {
        "model": args.model,
        "corpus": args.corpus,
        "config": str(args.config),
        "corpus_max_aa_seq_len": max_aa_seq_len,
        "allocated_params": allocated,
        "never_gradient_params": dead,
        "never_gradient_sources": sources,
        "effective_params": effective,
        "expected_effective_params": expected,
        "passed": not failures,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))

    if failures:
        raise SystemExit(
            "parameter check failed:\n" + "\n".join(f"  - {f}" for f in failures)
        )
    print("parameter check passed")


if __name__ == "__main__":
    main()
