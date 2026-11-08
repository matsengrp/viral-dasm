"""Gate 1: score the reference checkpoint in this tree, and check six things.

The strongest check in M1, and it costs no GPU-hours. One fixed-weight pass
over fold-0 validation exercises the Parquet backend, the split, all three
censoring sources, the transform chain, the encoder, the SF projection, the
head, the loss and the denominator -- essentially the whole port, without an
optimizer step.

What it asserts, in the order it learns them:

1. **Fold-0 split sizes** are 1,435,362 / 159,412. Seconds, no GPU. Catches a
   wrong ``cv_seed``, a wrong ``num_folds``, an off-by-one fold, or a
   non-numpy RNG.
2. **The censor banner.** Four integers the trainer prints before training
   starts: UShER entries and non-empty clades, global nucleotide sites, and
   distinct reference bases. The reversion source is roughly 40x the other two
   on Spike, so losing it is the most damaging possible prune error -- and it
   shows up here as ``4/4 bases present`` going missing.
3. **The state-dict key set and per-key shapes** of a model built fresh from
   this repository's rendered config, against the reference checkpoint's.
   Stronger than a scalar parameter count at the same cost: two wrong
   architectures can share a count. It also proves the published checkpoint is
   loadable by this code, which the release milestone depends on.
4. **The effective parameter count**, 84,884. Allocated is 144,516; the
   difference is ``encoder.positional_embedding.weight``, allocated (5000, 16)
   against a corpus whose longest sequence is 1,273 amino acids, so 59,632
   rows of it never receive a gradient.
5. **The validation loss**, 7.766710 to within 1e-4. Tight on purpose: at
   fixed weights the flash/dense difference is zero to six decimal places.
   Measured on this checkpoint and this corpus -- flash 7.766709, dense
   7.766710. Per-element differences are real (max |dlog SF| ~5.3e-3 on random
   weights, since flash-attn casts fp32 to bf16) and average out over the
   pass.
6. **The loss-mask position count**, 158,137. The sharpest integer in the
   system: it moves if any censoring source is dropped, if the nt mask is
   missed, if the reference sequence is wrong, if ``censor_mode`` flips, or if
   the fold is off by one. In each of those cases the loss itself moves by
   ~1e-3, which is detectable only once you have a tolerance; the count moves
   by thousands.

GISAID: reads the corpus, emits scalars and a pass/fail report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import (
    batch_size_for,
    build_sequential_loader,
    evaluate,
    head_key_of,
)
from lib import (
    CV_FOLD_INDEX,
    CV_NUM_FOLDS,
    CV_SEED,
    DATALOADER_WORKERS,
    SELF_CHECK_TOLERANCE,
    Corpus,
    assert_dasm2_is_this_repo,
    corpus,
    reference_run,
)

from dasm2.config import CvSplitConfig, load_config_from_yaml
from dasm2.data.evolutionary_dataset import _resolve_parquet_train_val
from dasm2.models.factory import create_model_from_config
from dasm2.serialization.checkpoints import (
    _attach_task_losses_for_inference,
    load_model_components,
)
from dasm2.train_command import (
    _resolve_censor_table,
    _resolve_reference_bases,
)

assert_dasm2_is_this_repo()

#: The banner the three censoring sources print, as (name, regex, expected).
#: Hard-coded against the v12 Spike corpus; a different mat-pcp run gives
#: different numbers and this gate is specific to the one the paper uses.
BANNER_EXPECTATIONS = (
    ("usher_entries", r"UShER censoring: mode=\w+, (\d+) entries", 1892),
    ("usher_clades", r"entries across (\d+)/\d+ clades", 37),
    ("usher_clades_total", r"entries across \d+/(\d+) clades", 56),
    ("global_nt_sites", r"plus (\d+) global nucleotide sites", 18),
    ("reference_nt", r"reference (\d+) nt", 3819),
    ("reference_codons", r"reference \d+ nt \((\d+) codons\)", 1273),
    ("distinct_bases", r"(\d+)/4 bases present", 4),
)

#: Expected censor mode, asserted from the banner rather than from the config
#: object, so a mode that is read but not applied cannot pass.
EXPECTED_CENSOR_MODE = "both"

#: Allocated parameters, and the slice of them that never receives a gradient.
EXPECTED_ALLOCATED_PARAMS = 144516
DEAD_PARAM_KEY = "encoder.positional_embedding.weight"

#: Longest amino-acid sequence in this corpus. Every parent is 3,819 nt.
MAX_AA_SEQ_LEN = 1273


def check_split_sizes(spec: Corpus, cv_split: CvSplitConfig) -> tuple[list[str], dict]:
    """Fold sizes, taken from the library's own resolver rather than redrawn.

    Compared against ``spec``'s recorded sizes rather than a module-level pair,
    because the draw sizes itself from the corpus's row count: every arm has
    its own.
    """
    _paths, _nick, train_filter, val_filter, _h, _m = _resolve_parquet_train_val(
        str(spec.path), cv_split=cv_split
    )
    n_train = int(np.asarray(train_filter).sum())
    n_val = int(np.asarray(val_filter).sum())
    print(
        f"  split: {n_train:,} train / {n_val:,} val "
        f"(expected {spec.n_train:,} / {spec.n_val:,})"
    )
    failures = []
    if (n_train, n_val) != (spec.n_train, spec.n_val):
        failures.append(
            f"fold {CV_FOLD_INDEX} of {CV_NUM_FOLDS} at seed {CV_SEED} gives "
            f"{n_train:,}/{n_val:,}, not {spec.n_train:,}/{spec.n_val:,}. Every "
            f"number below would then be measured on a different held-out set."
        )
    return failures, {"n_train": n_train, "n_val": n_val}


def check_censor_banner(config) -> tuple[list[str], dict, object, object]:
    """Resolve censoring, capturing and checking what it prints.

    Returns the failures, the parsed integers, and the two resolved objects, so
    the caller does not resolve them a second time.
    """
    lines: list[str] = []

    def capture(message: str) -> None:
        lines.append(str(message))
        print(str(message))

    censor_table = _resolve_censor_table(config, capture)
    reference_bases = _resolve_reference_bases(config, capture)
    banner = "\n".join(lines)

    failures, measured = [], {}
    for name, pattern, expected in BANNER_EXPECTATIONS:
        match = re.search(pattern, banner)
        if match is None:
            failures.append(
                f"the censor banner never printed {name!r} (pattern "
                f"{pattern!r}). A source that prints nothing is a source that "
                f"did not load."
            )
            continue
        value = int(match.group(1))
        measured[name] = value
        if value != expected:
            failures.append(
                f"{name} is {value}, not {expected}. The loss below would move "
                f"by ~1e-3 and the loss-mask count by thousands."
            )
    mode = re.search(r"mode=(\w+)", banner)
    if mode is None or mode.group(1) != EXPECTED_CENSOR_MODE:
        failures.append(
            f"censor_mode reads {mode.group(1) if mode else None!r}, not "
            f"{EXPECTED_CENSOR_MODE!r}."
        )
    measured["censor_mode"] = mode.group(1) if mode else None
    return failures, measured, censor_table, reference_bases


def build_from_config_and_load(
    config, checkpoint: Path, device: str
) -> tuple[list[str], dict, object]:
    """Build the model from **this repository's** config, load the reference weights.

    Deliberately not ``load_model_only``. That helper rebuilds the
    architecture from the config embedded *in the checkpoint*, which says
    ``flash_attention: true``, so on a host without flash-attn it raises
    before any weight is read. The published checkpoint therefore is not
    loadable by a flash-free reader the way the obvious call loads it --
    which is the property the release milestone depends on.

    Doing it this way instead checks the thing worth checking: that
    viral-dasm's own config-to-architecture path produces the architecture
    the reference was trained with, and that the reference's weights drop
    into it strictly. ``load_state_dict(strict=True)`` across the flash flag
    is what makes the checkpoint scorable on a laptop.

    The key-set and per-key-shape comparison is stronger than a scalar
    parameter count at the same cost: two wrong architectures can share a
    count.
    """
    failures = []

    # Raw weights, with no model construction, so nothing needs flash-attn.
    metadata, checkpoint_state = load_model_components(checkpoint, device=device)
    fresh = create_model_from_config(config.model, device=device)
    _attach_task_losses_for_inference(fresh, config)
    fresh_state = fresh.state_dict()

    missing = sorted(set(checkpoint_state) - set(fresh_state))
    extra = sorted(set(fresh_state) - set(checkpoint_state))
    if missing or extra:
        failures.append(
            f"a model built from this repository's config has a different "
            f"state-dict key set from the reference checkpoint: the "
            f"checkpoint has {missing} that it does not, and it has {extra} "
            f"that the checkpoint does not."
        )
    shape_mismatches = {
        key: {
            "checkpoint": tuple(checkpoint_state[key].shape),
            "config": tuple(fresh_state[key].shape),
        }
        for key in set(checkpoint_state) & set(fresh_state)
        if checkpoint_state[key].shape != fresh_state[key].shape
    }
    if shape_mismatches:
        failures.append(
            f"per-key shapes differ between the reference checkpoint and a "
            f"model built from this repository's config: {shape_mismatches}."
        )
    if not failures:
        fresh.load_state_dict(checkpoint_state, strict=True)

    allocated = sum(p.numel() for p in fresh.parameters())
    dead = 0
    for name, param in fresh.named_parameters():
        if name.endswith(DEAD_PARAM_KEY):
            # Rows beyond the corpus's longest amino-acid sequence are never
            # indexed, so they never receive a gradient.
            dead += (param.shape[0] - MAX_AA_SEQ_LEN) * param.shape[1]
    effective = allocated - dead
    print(
        f"  parameters: {allocated:,} allocated, {dead:,} never-gradient, "
        f"{effective:,} effective"
    )
    if allocated != EXPECTED_ALLOCATED_PARAMS:
        failures.append(
            f"the model allocates {allocated:,} parameters, not "
            f"{EXPECTED_ALLOCATED_PARAMS:,}; this is a different architecture."
        )
    if dead == 0:
        failures.append(
            f"no never-gradient parameters were found under {DEAD_PARAM_KEY!r}. "
            f"Either the key was renamed or the positional table is no longer "
            f"oversized, and the effective count is then not comparable to the "
            f"one the paper reports."
        )

    # What the checkpoint itself says it was trained under, recorded rather
    # than asserted: the weights are the same either way, and this is the line
    # that says the cross-flag load actually happened.
    trained_flash = (
        metadata.get("training_config", {})
        .get("model", {})
        .get("encoder", {})
        .get("flash_attention")
    )
    print(
        f"  checkpoint was trained with flash_attention={trained_flash}; "
        f"loaded here into flash_attention="
        f"{config.model.encoder.flash_attention}"
    )

    return (
        failures,
        {
            "allocated_params": allocated,
            "dead_params": dead,
            "effective_params": effective,
            "state_dict_keys": len(checkpoint_state),
            "checkpoint_flash_attention": trained_flash,
        },
        fresh,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dasm-85k")
    parser.add_argument("--corpus", default="full_depth")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="this repository's rendered training config",
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=DATALOADER_WORKERS)
    args = parser.parse_args()

    spec = corpus(args.corpus)
    reference = reference_run(args.model)
    config = load_config_from_yaml(args.config)
    cv_split = CvSplitConfig(
        num_folds=CV_NUM_FOLDS, fold_index=CV_FOLD_INDEX, seed=CV_SEED
    )

    failures: list[str] = []
    measured: dict = {}

    print(f"Gate 1: {args.model} on {spec.name}, fold {CV_FOLD_INDEX}")
    print(f"  checkpoint: {reference.best_checkpoint.name}")
    print(f"  flash_attention: {config.model.encoder.flash_attention}")
    print(f"  torch.backends.mha fastpath: {torch.backends.mha.get_fastpath_enabled()}")
    if torch.backends.mha.get_fastpath_enabled():
        failures.append(
            "the fused MHA fastpath is enabled. It engages in eval mode under "
            "no_grad, which is exactly this pass, and at val_batch_size 256 "
            "its attention tensor is 6.6 GB per layer -- under the int32 "
            "cutoff, so it does not crash, it just gets slow and fat. "
            "dasm2/__init__.py's import-time disable_mha_fastpath() did not run."
        )

    split_failures, split_measured = check_split_sizes(spec, cv_split)
    failures += split_failures
    measured |= split_measured

    banner_failures, banner_measured, _table, _bases = check_censor_banner(config)
    failures += banner_failures
    measured |= banner_measured

    arch_failures, arch_measured, module = build_from_config_and_load(
        config, reference.best_checkpoint, args.device
    )
    failures += arch_failures
    measured |= arch_measured
    if arch_measured["effective_params"] != reference.effective_params:
        failures.append(
            f"effective parameters are {arch_measured['effective_params']:,}, "
            f"not {reference.effective_params:,}."
        )

    head = head_key_of(module)
    loader = build_sequential_loader(
        config, spec.path, cv_split, batch_size_for(config), "val", args.num_workers
    )
    print("  scoring fold-0 validation ...", flush=True)
    val_loss, n_positions = evaluate(module, head, loader, args.device)
    delta = val_loss - reference.best_val_loss
    print(
        f"  val_loss: {val_loss:.6f} vs reference {reference.best_val_loss:.6f} "
        f"(delta {delta:+.2e})"
    )
    print(
        f"  loss-mask positions: {n_positions:,} "
        f"(expected {reference.n_loss_positions:,})"
    )
    measured |= {"val_loss": val_loss, "n_loss_positions": n_positions}

    if abs(delta) > SELF_CHECK_TOLERANCE:
        failures.append(
            f"this tree scores {val_loss:.6f} where the reference run logged "
            f"{reference.best_val_loss:.6f}, a difference of {delta:+.2e} "
            f"against a tolerance of {SELF_CHECK_TOLERANCE:.0e}. At fixed "
            f"weights flash and dense agree to six decimal places, so this is "
            f"not the attention setting. Check the censoring resolution and "
            f"the loss denominator first."
        )
    if n_positions != reference.n_loss_positions:
        failures.append(
            f"the pass scored {n_positions:,} loss-mask positions, not "
            f"{reference.n_loss_positions:,}."
        )

    report = {
        "model": args.model,
        "corpus": args.corpus,
        "checkpoint": str(reference.best_checkpoint),
        "measured": measured,
        "tolerance": SELF_CHECK_TOLERANCE,
        "passed": not failures,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))

    if failures:
        raise SystemExit("Gate 1 failed:\n" + "\n".join(f"  - {f}" for f in failures))
    print("Gate 1 passed")


if __name__ == "__main__":
    main()
