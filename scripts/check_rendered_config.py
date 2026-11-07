"""Assert a rendered training config, and the defaults it does not state.

Two kinds of check, and the second is the reason this file exists.

**What the config says.** Geometry, optimizer, batch sizes, the five censoring
paths, the early-stopping threshold. A rendered file is machine-written, so
these mostly guard against a template edit, but the censor paths also get an
existence check: a path that does not resolve censors nothing, trains
perfectly well, and reports a number for a condition it did not run.

**What the config does not say.** The reference run stops at step 436,000, and
nothing in any config file says so. It stops because three library defaults
compose: ``Trainer.min_learning_rate`` 1e-6, plateau ``factor`` 0.5, and
plateau ``eval_patience`` 10. 1e-3 x 0.5^10 = 9.77e-7 < 1e-6, and that is the
entire derivation. Patience 30 never fires -- the last improvement is 12 evals
before the stop, nowhere near 30.

A vendored trainer that defaulted ``min_learning_rate`` to 0, or ``factor`` to
0.1, would stop somewhere else and nothing in the loss curve would say why.
So all three are read off this tree's own objects and asserted. This is also
the check that Gate 2 explicitly cannot make: the reference's LR is constant
1e-3 through step 130,000, so a constant-LR run produces an identical first
ten evals.

The file also refuses a configured ``neutral_model``. This corpus carries
``nt_rates`` and ``nt_csps``, so a neutral model would be loaded and then
never used -- the config would describe a choice the run did not make.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import CONFIG, assert_dasm2_is_this_repo, mat_pcp_path

from dasm2.config import (
    ReduceOnPlateauSchedulerConfig,
    load_config_from_yaml,
)
from dasm2.training.trainer import Trainer

assert_dasm2_is_this_repo()

#: The three undeclared values that decide where the run stops, and the LR
#: path they imply from the configured 1e-3.
EXPECTED_MIN_LEARNING_RATE = 1e-6
EXPECTED_PLATEAU_FACTOR = 0.5
EXPECTED_PLATEAU_EVAL_PATIENCE = 10

#: Geometry of the one architecture M1 trains.
EXPECTED_ENCODER = {
    "nhead": 4,
    "d_model_per_head": 8,
    "num_layers": 5,
    "dim_feedforward": 128,
    "dropout": 0.1,
    "vocab_size": 21,
    "d_pos": 16,
    "flash_attention": False,
}

EXPECTED_SCALARS = {
    "eval_every_steps": 2000,
    "early_stopping_patience": 30,
    "early_stopping_min_delta": 1.0e-4,
    "batch_size": 128,
    "val_batch_size": 256,
    "num_dataloader_workers": 6,
    "save_training_state": True,
}


def lr_path(initial: float, factor: float, floor: float) -> list[float]:
    """Every learning rate the plateau scheduler can reach before the floor."""
    path, current = [initial], initial
    while current >= floor:
        current *= factor
        path.append(current)
    return path


def check_undeclared_defaults() -> tuple[list[str], dict]:
    """The three values no config file states, read off this tree."""
    failures = []
    signature = inspect.signature(Trainer.__init__)
    min_lr = signature.parameters["min_learning_rate"].default
    plateau = ReduceOnPlateauSchedulerConfig()

    measured = {
        "min_learning_rate": min_lr,
        "plateau_factor": plateau.factor,
        "plateau_eval_patience": plateau.eval_patience,
    }
    for name, value, expected in (
        ("Trainer.min_learning_rate", min_lr, EXPECTED_MIN_LEARNING_RATE),
        ("plateau factor", plateau.factor, EXPECTED_PLATEAU_FACTOR),
        (
            "plateau eval_patience",
            plateau.eval_patience,
            EXPECTED_PLATEAU_EVAL_PATIENCE,
        ),
    ):
        if value != expected:
            failures.append(
                f"{name} defaults to {value!r}, not {expected!r}. This value "
                f"appears in no config file and it is one of the three that "
                f"together decide where the run stops; a run under it would "
                f"stop somewhere other than the reference's 436,000 steps "
                f"with nothing in the loss curve to say why."
            )
    return failures, measured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--expect-max-train-steps",
        type=int,
        default=None,
        help="the step budget this rendered config should carry",
    )
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text())
    config = load_config_from_yaml(args.config)
    failures: list[str] = []
    measured: dict = {}

    encoder = config.model.encoder
    for field, expected in EXPECTED_ENCODER.items():
        value = getattr(encoder, field)
        measured[f"encoder.{field}"] = value
        if value != expected:
            failures.append(f"encoder.{field} is {value!r}, not {expected!r}.")

    for field, expected in EXPECTED_SCALARS.items():
        value = getattr(config, field)
        measured[field] = value
        if value != expected:
            failures.append(f"{field} is {value!r}, not {expected!r}.")

    if args.expect_max_train_steps is not None:
        measured["max_train_steps"] = config.max_train_steps
        if config.max_train_steps != args.expect_max_train_steps:
            failures.append(
                f"max_train_steps is {config.max_train_steps}, not "
                f"{args.expect_max_train_steps}."
            )

    # The five censoring fields, against config.yaml and against the disk.
    expected_paths = {
        key: mat_pcp_path(key)
        for key in (
            "censored_mutations_file",
            "censored_clades_file",
            "masked_nt_sites_file",
            "reference_sequence_file",
        )
    }
    for key, expected_path in expected_paths.items():
        value = getattr(config, key)
        measured[key] = str(value) if value is not None else None
        if value is None or Path(value) != expected_path:
            failures.append(
                f"{key} is {value!r}, not {str(expected_path)!r}, so this run "
                f"censors something other than what the paper's runs censored."
            )
        elif not Path(value).exists():
            failures.append(f"{key} points at {value}, which does not exist.")
    expected_mode = CONFIG["mat_pcp"]["censor_mode"]
    measured["censor_mode"] = str(config.censor_mode)
    if str(config.censor_mode).rsplit(".", 1)[-1].lower() != expected_mode:
        failures.append(
            f"censor_mode is {config.censor_mode!r}, not {expected_mode!r}."
        )

    # A configured neutral model would be loaded and never used, because this
    # corpus carries nt_rates and nt_csps. Raise rather than reproduce that.
    if raw.get("neutral_model") is not None:
        failures.append(
            "this config carries a neutral_model block. The prepared corpus "
            "already carries nt_rates and nt_csps, so the model would be "
            "loaded and then never consulted -- the config would describe a "
            "choice the run did not make."
        )

    # No scheduler block, so the plateau default applies and the LR path below
    # is what the run follows.
    measured["scheduler_block"] = raw.get("scheduler")
    if raw.get("scheduler") is not None:
        failures.append(
            "this config states a scheduler block. The reference run has none, "
            "so its LR path is the plateau default's; an explicit block here "
            "would have to be checked against that path rather than assumed."
        )

    default_failures, default_measured = check_undeclared_defaults()
    failures += default_failures
    measured |= default_measured

    path = lr_path(
        config.optimizer.learning_rate,
        default_measured["plateau_factor"],
        default_measured["min_learning_rate"],
    )
    measured["lr_path"] = path
    print(
        f"  LR path to the floor ({len(path)} values): "
        f"{', '.join(f'{v:.6g}' for v in path)}"
    )
    print(f"  reductions to reach the floor: {len(path) - 1}")

    report = {
        "config": str(args.config),
        "measured": measured,
        "passed": not failures,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))

    if failures:
        raise SystemExit(
            "rendered-config check failed:\n" + "\n".join(f"  - {f}" for f in failures)
        )
    print("rendered-config check passed")


if __name__ == "__main__":
    main()
