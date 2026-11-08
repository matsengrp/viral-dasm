"""Training command implementation for DASM2 CLI.

This module orchestrates training by:
1. Resolving config name to ModelConfig
2. Loading per-head datasets using explicit --data mapping
3. Creating output directory
4. Delegating to core trainer infrastructure
5. Saving checkpoint with embedded metadata at training completion
"""

import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pandas as pd
import torch
import yaml
from torch.utils.data import ConcatDataset, Dataset, Subset

from dasm2.config import (
    CvSplitConfig,
    HeadConfig,
    HookConfig,
    LossConfig,
    TaskConfig,
    TrainingConfig,
    WarmupPolynomialSchedulerConfig,
    get_config,
)
from dasm2.data.evolutionary_dataset import (
    DatasetBundle,
    load_whichmut_train_val,
    warn_if_parquet_dataset_mode_mismatch,
)
from dasm2.data.local import compute_md5
from dasm2.data.parquet_backend import CacheBudgetSpec
from dasm2.data.registry import (
    BINDING_CLASSIFICATION_HEAD_TYPES,
    CONTRASTIVE_BINDING_HEAD_TYPES,
    DELTA_AFFINITY_HEAD_TYPES,
    DMS_HEAD_TYPES,
    MLM_HEAD_TYPES,
    MPNN_WHICHMUT_HEAD_TYPES,
    STRUCTURAL_HEAD_TYPES,
    is_parquet_data_spec,
    is_pickle_data_spec,
)
from dasm2.heads.types import HeadType
from dasm2.losses.factory import create_loss
from dasm2.losses.types import (
    BINDING_CLASSIFICATION_LOSS_TYPES,
    EVOLUTIONARY_LOSS_TYPES,
    WHICHMUT_LOSS_TYPES,
    LossType,
)
from dasm2.metadata import SourceCheckpointInfo, create_training_metadata_info
from dasm2.models.core.attention import (
    EntmaxMultiheadAttention,
    FlashMultiheadAttention,
    RotaryMultiheadAttention,
)
from dasm2.models.factory import create_model_from_config
from dasm2.sequence.masking import (
    CensorMode,
    CensorSpec,
    load_censor_table,
    load_reference_base_idxs,
    parse_masked_nt_sites_file,
    parse_problematic_sites_file,
    with_global_codon_sites,
    with_global_nt_sites,
)
from dasm2.serialization.checkpoints import (
    load_model_components,
    load_model_metadata,
    migrate_legacy_state_dict,
    resume_training,
)
from dasm2.training import Task, create_trainer
from dasm2.training.compute_backend import (
    AccelerateBackend,
    ComputeBackend,
    MixedPrecision,
    SingleGPUBackend,
    compute_effective_batch_size,
)
from dasm2.training import Trainer  # noqa: F401  (freeze_task_head annotation)
from dasm2.training.hooks import CheckpointSaveHook
from dasm2.training.hooks.base import GradientCombineHook, TrainingHook
from dasm2.training.hooks.factory import create_hook
from dasm2.training.logging import TrainingLogger
from dasm2.training.types import StepBudget
from dasm2.utils.device import pick_device
from dasm2.utils.torch_utils import set_deterministic_mode, set_seed

# Maximum number of parameter keys to display when loading checkpoints
_MAX_KEYS_TO_DISPLAY = 5

DEFAULT_TRAIN_FRACTION = 0.8


# Training's main process runs single-threaded. This pin used to happen as a
# side effect of importing the BL hooks; it is kept explicit here so moving
# that pin into the BL workers doesn't change training's
# floating-point reduction order.
TRAINING_NUM_THREADS = 1


def _rank_guarded_print(is_main: bool) -> Callable[..., None]:
    """Return ``print`` if this is the main process, else a silent no-op."""
    return print if is_main else (lambda *a, **kw: None)


_CV_SPLIT_LOGGER_NAME = "dasm2.data.evolutionary_dataset.cv_split"
_PARQUET_BACKEND_LOGGER_NAME = "dasm2.data.parquet_backend"


def _attach_info_print_handler(logger_name: str, print_fn: Callable[..., None]) -> None:
    """Route a library logger's INFO records through ``print_fn``.

    Library callers keep Python's default silence-by-default behavior; CLI
    ``train_model`` runs attach this handler so selected INFO lines surface in
    the run log via the rank-guarded ``_print``: the CV-split helpers' held-out
    group ids and per-fold counts, and the Parquet backend's cache budget and
    its source.

    Idempotent: re-invoking with the same ``print_fn`` does not stack handlers.
    """
    target = logging.getLogger(logger_name)
    for existing in target.handlers:
        if getattr(existing, "_dasm2_print_handler", False):
            return

    class _PrintHandler(logging.Handler):
        _dasm2_print_handler = True

        def emit(self, record: logging.LogRecord) -> None:
            print_fn(self.format(record))

    handler = _PrintHandler(level=logging.INFO)
    target.addHandler(handler)
    if target.level == logging.NOTSET or target.level > logging.INFO:
        target.setLevel(logging.INFO)


# AAH (Amino Acid Hybrid) is the only head that takes two datasets;
# expansion happens at task-construction time (one Task per loss type,
# both pointing at the same head) — see ``_create_tasks_from_config``.
#
# Multi-loss head extension checklist: a future second head that emits
# multiple tasks on a single head_config currently requires explicit
# updates at *every* AAH-aware site in this module. The contact points
# (referenced by function name — line numbers drift):
#
#   * ``_parse_head_data_config`` — validates the per-head path count
#     (AAH alone permits ``len(paths) == 2``).
#   * ``_validate_head_datasets_match_config`` — accepts colon-suffixed
#     dataset keys (``"AAHWhichmutHead:whichmut"`` /
#     ``"AAHWhichmutHead:aao"``) for multi-loss heads.
#   * ``_create_tasks_from_config`` — emits multiple ``Task`` entries
#     for the same head_config and pairs each with the right
#     ``LossConfig``.
#   * AAH branch in ``train_model``'s ``head_datasets`` loop — head-keyed
#     dispatch that writes the ``f"{head_type_str}:whichmut"`` /
#     ``f"{head_type_str}:aao"`` entries into ``head_datasets``. The loader
#     it called upstream is not in this fork.
#   * ``suffix_to_loss_type`` (inside ``_create_tasks_from_config``) —
#     colon-suffix → ``LossType`` map.
#
# Until this is generalised into a per-head ``tasks_for_head`` registry,
# every site above needs to grow a new branch when a second multi-loss
# head appears. The discriminated-union ``LossConfig`` is
# already shaped to absorb the per-task data; only the dataset-side
# wiring is still head-keyed. Down-stream call sites compare to
# ``HeadType.AAH_WHICHMUT_HEAD`` directly rather than via a dispatch
# frozenset.

# AAH dataset-key suffixes (the part after the ``:`` in keys like
# ``"AAHWhichmutHead:whichmut"``). These are not equal to the matching
# ``LossType.value`` strings (e.g. ``LossType.AAO_WHICHMUT.value`` is
# ``"aao_whichmut"``, the suffix is ``"aao"``) so the mapping is explicit.
# Producer (``train_model``) and consumer (``_create_tasks_from_config``)
# share this single constant so a rename can't desync them silently.
_AAH_WHICHMUT_SUFFIX = "whichmut"
_AAH_AAO_SUFFIX = "aao"
_AAH_SUFFIX_TO_LOSS_TYPE: Dict[str, LossType] = {
    _AAH_WHICHMUT_SUFFIX: LossType.WHICHMUT,
    _AAH_AAO_SUFFIX: LossType.AAO_WHICHMUT,
}


def _maybe_relaunch_under_accelerate(
    multi_gpu: bool,
    num_gpus: Optional[int],
    gpus: Union[str, Tuple[int, ...], None] = None,
) -> None:
    """Re-exec under ``accelerate launch`` if not already inside one.

    When ``--multi_gpu`` is passed from the CLI but ``LOCAL_RANK`` is absent,
    we are the outer (non-distributed) process. Re-exec ourselves under
    ``accelerate launch -m dasm2.cli`` so every rank runs the same entry point.

    Args:
        multi_gpu: Whether multi-GPU training was requested.
        num_gpus: Number of GPUs (None = all available).
        gpus: Comma-separated GPU IDs (e.g. "1,4"). Sets
            ``CUDA_VISIBLE_DEVICES`` before relaunching so Accelerate only
            sees the specified GPUs. Useful on shared machines.

    The relaunch always requests ``--main_process_port 0`` so concurrent runs
    on one host get distinct rendezvous ports instead of colliding on 29500.

    Raises:
        SystemExit: Propagates the exit code from the ``accelerate launch``
            subprocess so the CLI reports success/failure correctly.
    """
    if not multi_gpu or "LOCAL_RANK" in os.environ:
        return
    if gpus is not None:
        # Fire may parse "0,1,2,3" as a tuple of ints; coerce back to str.
        if not isinstance(gpus, str):
            gpus = ",".join(str(g) for g in gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    # Port 0 tells Accelerate to pick a free port rather than defaulting to
    # 29500. Two runs launched on one host otherwise both resolve to 29500 --
    # neither sees it taken yet -- and the one that loses the bind race dies at
    # rendezvous with EADDRINUSE. Hit while launching two training arms
    # seconds apart.
    cmd = ["accelerate", "launch", "--multi_gpu", "--main_process_port", "0"]
    if num_gpus is not None:
        cmd += ["--num_processes", str(num_gpus)]
    cmd += ["-m", "dasm2.cli"] + sys.argv[1:]
    raise SystemExit(subprocess.call(cmd))


def _is_main_process() -> bool:
    """Return True if this is the main process (rank 0 or single-GPU).

    Uses LOCAL_RANK environment variable set by accelerate. Falls back to
    True when not in a distributed context.
    """
    return os.environ.get("LOCAL_RANK", "0") == "0"


def _detect_attention_backend(model: torch.nn.Module) -> str:
    """Return a human-readable string describing the attention backend in use."""
    for module in model.modules():
        if isinstance(module, FlashMultiheadAttention):
            return "FlashMultiheadAttention (flash-attn v2)"
        if isinstance(module, EntmaxMultiheadAttention):
            return "EntmaxMultiheadAttention (entmax)"
        if isinstance(module, RotaryMultiheadAttention):
            return "RotaryMultiheadAttention (rotary)"
        if isinstance(module, torch.nn.MultiheadAttention):
            return "nn.MultiheadAttention (softmax)"
    return "unknown (no attention module found)"


def _validate_head_datasets_match_config(
    config_name: str,
    config_heads: Set[str],
    provided_heads: Set[str],
) -> None:
    """Validate that provided datasets match config heads exactly.

    For AAH heads, the provided_heads may contain suffixed keys
    like "AAHWhichmutHead:whichmut" and "AAHWhichmutHead:aao".
    These are normalized to the base head type for validation.

    Args:
        config_name: Name of the config (for error messages)
        config_heads: Set of head type strings from config
        provided_heads: Set of head type strings from provided data
            (may include suffixed keys for multi-dataset heads)

    Raises:
        ValueError: If there are missing or extra heads
    """
    # Normalize provided_heads by stripping suffixes for multi-dataset heads
    normalized_provided: Set[str] = set()
    for head_key in provided_heads:
        if ":" in head_key:
            # Multi-dataset head key like "AAHWhichmutHead:whichmut"
            base_head = head_key.split(":")[0]
            normalized_provided.add(base_head)
        else:
            normalized_provided.add(head_key)

    missing = config_heads - normalized_provided
    extra = normalized_provided - config_heads

    if missing:
        raise ValueError(
            f"Missing datasets for heads: {sorted(missing)}. "
            f"Config '{config_name}' requires: {sorted(config_heads)}"
        )
    if extra:
        raise ValueError(
            f"Unknown heads in head_datasets: {sorted(extra)}. "
            f"Config '{config_name}' has: {sorted(config_heads)}"
        )


def _reject_untrainable_heads(training_config: TrainingConfig) -> None:
    """Fail fast on a config whose model has a head with nothing to train.

    ``NeutralHead`` is the parameter-free f ≡ 1 model: it emits a constant
    that does not depend on any parameter, so a training run would die deep
    in the trainer (an empty optimizer parameter list, or a backward through
    a tensor with no grad) instead of naming the problem. Such a model is
    materialized with ``save_parameter_free_checkpoint`` instead.

    Raises:
        ValueError: If any configured head is a ``NeutralHead``.
    """
    if any(h.head_type is HeadType.NEUTRAL_HEAD for h in training_config.model.heads):
        raise ValueError(
            "dasm2 train cannot train a NeutralHead: it has no parameters, so "
            "there is nothing to optimize. Write the f ≡ 1 model as a "
            "checkpoint with dasm2.serialization.checkpoints."
            "save_parameter_free_checkpoint and score that instead."
        )


def _validate_resume_cli_args(
    resume: Optional[str],
    checkpoint: Optional[str],
    start_cycle: int,
    loader_cycle_start: Optional[int],
    recompute_branch_lengths_on_resume: bool,
) -> None:
    """Fail fast on invalid --resume parameter combinations.

    ``start_cycle``, ``loader_cycle_start``, and the optimizer/scheduler/RNG
    state they all depend on are restored from the checkpoint's own
    ``TrainingState`` when ``--resume`` is set — passing them explicitly
    alongside ``--resume`` would either be redundant or silently overridden,
    so fail fast rather than let one of the two win unpredictably
    (Constitution IV).

    Args:
        resume: Path to a resumable checkpoint, or None.
        checkpoint: Path to a weights-only checkpoint for fine-tuning, or None.
        start_cycle: Cycle to start from (0-indexed CLI value, before any
            resume-derived override).
        loader_cycle_start: DDP shuffle epoch to seed before training, or None.
        recompute_branch_lengths_on_resume: Whether the BLO resume-fallback
            flag was requested.

    Raises:
        ValueError: If ``resume`` is combined with ``checkpoint``, a non-zero
            ``start_cycle``, or ``loader_cycle_start``; or if
            ``recompute_branch_lengths_on_resume`` is set but neither
            ``resume`` nor a non-zero ``start_cycle`` could ever trigger it.

    Note:
        ``--multi_gpu`` is *not* checked here. It used to be rejected because
        ``resume_training()`` hard-wired a ``SingleGPUBackend``; it now takes
        the backend the run is launched under. The world size
        itself is validated by :func:`_validate_resume_world_size_matches`,
        which needs the backend and so runs later.

        ``freeze_encoder`` is not checked here either:
        ``TrainingConfig.freeze_encoder`` is now part of the checkpoint's
        recorded config, so ``resume_training()`` applies it (or not)
        exactly as the interrupted run had it, and a ``--freeze_encoder``
        override that conflicts with the checkpoint's own value is caught
        by :func:`_validate_resume_overrides_dont_conflict` alongside
        ``batch_size`` and friends.
    """
    if resume is not None:
        if checkpoint is not None:
            raise ValueError(
                "--resume and --checkpoint are mutually exclusive: --checkpoint "
                "loads weights only (fine-tuning, resets optimizer/LR schedule); "
                "--resume restores full training state via resume_training(). "
                "Use --resume to continue an interrupted run."
            )
        if start_cycle != 0:
            raise ValueError(
                "--resume determines start_cycle from the checkpoint's saved "
                "training state; do not also pass --start_cycle "
                f"(got start_cycle={start_cycle})."
            )
        if loader_cycle_start is not None:
            raise ValueError(
                "--resume restores loader_cycle from the checkpoint's saved "
                "training state; do not also pass --loader_cycle_start "
                f"(got loader_cycle_start={loader_cycle_start})."
            )
    if recompute_branch_lengths_on_resume and resume is None and start_cycle == 0:
        raise ValueError(
            "--recompute_branch_lengths_on_resume only has an effect when "
            "training starts mid-pipeline (--resume, or --checkpoint with "
            "--start_cycle > 0); got resume=None, start_cycle=0."
        )


def _validate_resume_checkpoint_config(
    checkpoint_path: Path,
    training_config: TrainingConfig,
    checkpoint_training_config: TrainingConfig,
) -> None:
    """Fail fast when --resume's config is incompatible with the checkpoint.

    Two independent checks: the task list must match, and the checkpoint must not have been trained with a
    feature ``resume_training()`` doesn't support restoring.

    ``resume_training()`` reconstructs the model from the checkpoint's own
    ``training_config.model`` and attaches ``task_losses`` built from the
    *current* run's ``tasks``. A mismatched task list (different
    head_type/loss_type pairs, different count, or different order) would
    otherwise produce a model whose task_losses/heads don't correspond to
    the checkpoint's optimizer state, rather than a clear error.

    Also fails fast on the *other* direction of the ``combine_gradient_hooks``
    resume gap: the caller already rejects a *current* run config that sets
    it under ``--resume`` (``resume_training()`` never threads it through),
    but that only catches the current config. A checkpoint that was itself
    trained *with* ``combine_gradient_hooks``, resumed with a config that
    omits it, would otherwise sail past that check and only fail later —
    inside ``Trainer.restore_training_state``'s presence-mismatch assertion
    — after the model, datasets, and logger are already built. Checking the
    checkpoint's own config here catches it before any of that setup starts.
    (``head_optimizer`` had the same one-sided gap before its resume
    support landed; ``resume_training()`` now rebuilds it from the
    checkpoint's own config the same way it does the main optimizer, so
    there is nothing to check here for it anymore.)

    Args:
        checkpoint_path: Path to the resume checkpoint (for the error message).
        training_config: The current run's resolved TrainingConfig.
        checkpoint_training_config: TrainingConfig reconstructed from the
            checkpoint's own metadata.

    Raises:
        ValueError: If the (head_type, loss_type) task pairs differ.
        NotImplementedError: If the checkpoint was trained with
            ``combine_gradient_hooks``, which ``resume_training()`` does not
            thread through.
    """
    if checkpoint_training_config.combine_gradient_hooks:
        raise NotImplementedError(
            f"--resume checkpoint {checkpoint_path} was trained with "
            "combine_gradient_hooks set, but resume_training() does not "
            "thread combine_gradient_hooks through — a resumed run would "
            "silently drop it. Start a fresh run instead, or drop "
            "combine_gradient_hooks from the checkpoint's own training run."
        )
    current_tasks = [
        (tc.head_type.value, tc.loss_config.loss_type.value)
        for tc in training_config.tasks
    ]
    checkpoint_tasks = [
        (tc.head_type.value, tc.loss_config.loss_type.value)
        for tc in checkpoint_training_config.tasks
    ]
    if current_tasks != checkpoint_tasks:
        raise ValueError(
            f"--resume checkpoint {checkpoint_path} was trained with tasks "
            f"{checkpoint_tasks} but the current config specifies tasks "
            f"{current_tasks}. Resume requires an unchanged task "
            "configuration (same head_type/loss_type pairs in the same "
            "order) so the checkpoint's model, optimizer, and task_losses "
            "modules line up with this run's tasks. Use --checkpoint for "
            "weights-only fine-tuning with a different task configuration "
            "instead."
        )


def _validate_resume_world_size_matches(
    checkpoint_path: Path,
    checkpoint_num_processes: Optional[int],
    run_num_processes: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> None:
    """Fail fast when --resume's world size differs from the checkpoint's.

    Effective batch size is
    ``batch_size x gradient_accumulation_steps x num_processes``, so resuming
    a world-size-4 checkpoint on one GPU quarters it while continuing an LR
    schedule that was annealed for the original — and the step budget covers
    proportionally less data, since steps are taken in lockstep on every
    rank. Neither is visible in any log line. Elastic resume (deliberately
    changing world size) would mean rescaling both, which is out of scope;
    reject it with both numbers named instead.

    Args:
        checkpoint_path: Path to the resume checkpoint (for the error message).
        checkpoint_num_processes: World size recorded in the checkpoint's
            ``training_metadata``, or None for checkpoints written before
            it was recorded. None skips the check — a missing field means
            "unknown", and assuming 1 would reject legitimate resumes of
            every older multi-GPU checkpoint.
        run_num_processes: World size of the current run's backend.
        batch_size: Per-device batch size read off the *checkpoint's* own
            config, not this run's. ``resume_training`` rebuilds the trainer
            from the checkpoint's config, so that is the number both effective
            batch sizes are actually computed from — and the two configs can
            legitimately differ here, because
            ``_validate_resume_overrides_dont_conflict`` only compares values
            the CLI passed explicitly, leaving a YAML-declared difference to
            be silently overridden by the checkpoint.
        gradient_accumulation_steps: Accumulation steps, likewise from the
            checkpoint's config.

    Raises:
        ValueError: If the two world sizes differ.
    """
    if (
        checkpoint_num_processes is None
        or checkpoint_num_processes == run_num_processes
    ):
        return
    checkpoint_effective = compute_effective_batch_size(
        batch_size, gradient_accumulation_steps, checkpoint_num_processes
    )
    run_effective = compute_effective_batch_size(
        batch_size, gradient_accumulation_steps, run_num_processes
    )
    raise ValueError(
        f"--resume checkpoint {checkpoint_path} was trained at world size "
        f"{checkpoint_num_processes} but this run has world size "
        f"{run_num_processes}. That changes the effective batch size from "
        f"{checkpoint_effective} to {run_effective} "
        f"(batch_size={batch_size} x gradient_accumulation_steps="
        f"{gradient_accumulation_steps} x num_processes) while continuing an "
        f"LR schedule annealed for the original, and makes a fixed "
        f"max_train_steps cover a different amount of data. Resume at world "
        f"size {checkpoint_num_processes}, or start a fresh run from these "
        f"weights with --checkpoint (which resets the optimizer and LR "
        f"schedule)."
    )


def _validate_resume_mixed_precision_matches(
    checkpoint_path: Path,
    checkpoint_mixed_precision: Optional[str],
    run_mixed_precision: str,
) -> None:
    """Fail fast when --resume's precision mode differs from the checkpoint's.

    ``--mixed_precision`` is honored on resume, which is exactly why a
    mismatch has to be caught: resuming a bf16-trained run in fp32 (or the
    reverse) silently changes the numerics of every forward and backward pass
    under a restored optimizer state and LR schedule, and — unlike the world
    size — nothing downstream would ever fail. The flag is a launch-time
    argument, not part of ``TrainingConfig``, so ``resume_training`` cannot
    recover it from the checkpoint's config; it is recorded in
    ``training_metadata.mixed_precision`` instead.

    Args:
        checkpoint_path: Path to the resume checkpoint (for the error message).
        checkpoint_mixed_precision: Backend precision mode recorded in the
            checkpoint's ``training_metadata``, or None for checkpoints
            written before it was recorded. None skips the check, for the
            same reason it does in :func:`_validate_resume_world_size_matches`:
            absent means unknown, not ``"no"``.
        run_mixed_precision: This run's ``backend.mixed_precision``.

    Raises:
        ValueError: If the two modes differ.
    """
    if (
        checkpoint_mixed_precision is None
        or checkpoint_mixed_precision == run_mixed_precision
    ):
        return
    remedy = (
        "drop --mixed_precision"
        if checkpoint_mixed_precision == "no"
        else f"pass --mixed_precision {checkpoint_mixed_precision}"
    )
    raise ValueError(
        f"--resume checkpoint {checkpoint_path} was trained with "
        f"mixed_precision={checkpoint_mixed_precision!r} but this run has "
        f"mixed_precision={run_mixed_precision!r}. That changes the numerics "
        f"of every forward and backward pass while continuing the "
        f"checkpoint's own optimizer state and LR schedule, and nothing later "
        f"in the run would fail because of it. To continue this run, "
        f"{remedy}; to deliberately change precision, start a fresh run from "
        f"these weights with --checkpoint (which resets the optimizer and LR "
        f"schedule). Note this is the Accelerate backend's mode from "
        f"--mixed_precision, not training_config.use_mixed_precision, which "
        f"rides in the config and is already restored from the checkpoint."
    )


# Maps a train_model_from_datasets CLI/function parameter name to how to
# read the corresponding value off a TrainingConfig, for
# _validate_resume_overrides_dont_conflict. Only parameters that
# resume_training() actually re-derives from the CHECKPOINT's own config
# (rather than from the tasks/hooks built before resume_training() is
# called) need a conflict check — see that function's docstring.
_RESUME_OVERRIDE_CHECKPOINT_PATHS: Dict[str, Callable[[TrainingConfig], Any]] = {
    "max_train_steps": lambda cfg: cfg.max_train_steps,
    "eval_every_steps": lambda cfg: cfg.eval_every_steps,
    "max_eval_steps": lambda cfg: cfg.max_eval_steps,
    "batch_size": lambda cfg: cfg.batch_size,
    "val_batch_size": lambda cfg: cfg.val_batch_size,
    "gradient_accumulation_steps": lambda cfg: cfg.gradient_accumulation_steps,
    "early_stopping_patience": lambda cfg: cfg.early_stopping_patience,
    "num_dataloader_workers": lambda cfg: cfg.num_dataloader_workers,
    "max_tokens_per_batch": lambda cfg: cfg.max_tokens_per_batch,
    "learning_rate": lambda cfg: cfg.optimizer.learning_rate,
    "weight_decay": lambda cfg: cfg.optimizer.weight_decay,
    "optimizer_name": lambda cfg: cfg.optimizer.optimizer_name,
    "freeze_encoder": lambda cfg: cfg.freeze_encoder,
}


def _validate_resume_overrides_dont_conflict(
    checkpoint_path: Path,
    checkpoint_training_config: TrainingConfig,
    overrides: Dict[str, Any],
) -> None:
    """Fail fast on a --resume override that would be silently ignored.

    ``resume_training()`` rebuilds the model's frozen/trainable parameters,
    the optimizer, scheduler, StepBudget, and batch sizes from the
    CHECKPOINT's own ``training_config``, not from this run's CLI overrides
    (hook-level params like ``freeze_branch_lengths`` and BL-optimization
    knobs are unaffected — they're baked into the ``tasks`` list built
    before ``resume_training()`` runs, so those do take effect). Passing
    e.g. ``--learning_rate`` alongside ``--resume`` would
    otherwise be accepted without complaint and then silently have no
    effect — the same silent-divergence failure class that resumable
    training state exists to fix, just one layer up (a mismatched *intent*
    instead of a mismatched *restore*).

    Accepts an override when it matches the checkpoint's own value exactly
    (this is what makes the Snakemake idempotency pattern in
    ``docs/guides/training-a-model.md`` work: the same command line is
    reused for the initial attempt and the ``--resume`` retry, and real
    Snakefiles in this repo pass flags like ``--batch-size`` on every
    invocation). Raises when it would silently change resumed behavior.

    Args:
        checkpoint_path: Path to the resume checkpoint (for the error message).
        checkpoint_training_config: TrainingConfig reconstructed from the
            checkpoint's own metadata.
        overrides: Mapping of parameter name (a key of
            ``_RESUME_OVERRIDE_CHECKPOINT_PATHS``) to the value passed on
            this call. ``None`` means "not passed" and is never a conflict.

    Raises:
        ValueError: If any override differs from the checkpoint's value.
    """
    conflicts = []
    for name, passed_value in overrides.items():
        if passed_value is None:
            continue
        checkpoint_value = _RESUME_OVERRIDE_CHECKPOINT_PATHS[name](
            checkpoint_training_config
        )
        if passed_value != checkpoint_value:
            conflicts.append(
                f"{name}={passed_value!r} (checkpoint has {checkpoint_value!r})"
            )

    if conflicts:
        raise ValueError(
            f"--resume checkpoint {checkpoint_path} was trained with different "
            "hyperparameters than requested on this run. resume_training() "
            "rebuilds the optimizer/scheduler/step-budget from the "
            "CHECKPOINT's own config, so these overrides would be silently "
            f"ignored rather than applied: {'; '.join(conflicts)}. Remove "
            "them (they have no effect on resume) or omit --resume to start "
            "a fresh run with these values."
        )


def _create_model_and_maybe_load_checkpoint(
    training_config: TrainingConfig,
    backend: ComputeBackend,
    checkpoint: Optional[str],
    _print: Callable[..., None],
) -> Tuple[Any, Optional[SourceCheckpointInfo]]:
    """Build the model from config, optionally fine-tuning from a checkpoint.

    Only called on the non-``--resume`` path — ``resume_training()`` builds
    its own model from the checkpoint's own ``training_config`` instead (see
    ``train_model_from_datasets``).

    Args:
        training_config: This run's resolved TrainingConfig.
        backend: Compute backend (used for the model's init device).
        checkpoint: Path to a weights-only checkpoint for fine-tuning, or
            None for a fresh model.
        _print: Rank-guarded print function.

    Returns:
        Tuple of (model, source_checkpoint_info). ``source_checkpoint_info``
        is None unless ``checkpoint`` was provided.

    Raises:
        FileNotFoundError: If ``checkpoint`` names a path that doesn't exist.
    """
    model = create_model_from_config(
        training_config.model, device=backend.model_init_device
    )

    # Log attention backend so training runs have a definitive record
    _print(f"  Attention backend: {_detect_attention_backend(model)}")

    if checkpoint is None:
        return model, None

    # Load checkpoint weights (for fine-tuning / sequential training)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    _print(f"Loading weights from checkpoint: {checkpoint_path}")
    metadata, state_dict = load_model_components(
        checkpoint_path, device=backend.model_init_device
    )

    # Migrate legacy state dict keys (e.g., self_attn → self_attention) so
    # warm-starting from an older checkpoint doesn't silently miss every
    # renamed parameter. ``--resume`` already migrates via resume_training();
    # this is the fine-tuning path's equivalent.
    serialization_version = metadata.get("serialization_version", 1)
    state_dict = migrate_legacy_state_dict(state_dict, model, serialization_version)

    # Load with strict=False to allow partial matching (different heads)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        _print(f"  New parameters (randomly initialized): {len(missing_keys)}")
        for key in missing_keys[:_MAX_KEYS_TO_DISPLAY]:
            _print(f"    - {key}")
        if len(missing_keys) > _MAX_KEYS_TO_DISPLAY:
            _print(f"    ... and {len(missing_keys) - _MAX_KEYS_TO_DISPLAY} more")
    if unexpected_keys:
        _print(f"  Checkpoint parameters not used: {len(unexpected_keys)}")
        for key in unexpected_keys[:_MAX_KEYS_TO_DISPLAY]:
            _print(f"    - {key}")
        if len(unexpected_keys) > _MAX_KEYS_TO_DISPLAY:
            _print(f"    ... and {len(unexpected_keys) - _MAX_KEYS_TO_DISPLAY} more")
        # Identify dropped heads by extracting head names from parameter keys
        # Keys look like "heads.SelectionFactorHead.linear.weight"
        dropped_heads = {
            key.split(".")[1] for key in unexpected_keys if key.startswith("heads.")
        }
        if dropped_heads:
            warnings.warn(
                f"The following heads from the source checkpoint are not in "
                f"your fine-tuning config and will not appear in the saved "
                f"checkpoint: {sorted(dropped_heads)}. To preserve a head "
                f"without training it, include it in your config with "
                f"loss_weight: 0.",
                stacklevel=1,
            )
    if not missing_keys and not unexpected_keys:
        _print("  All parameters loaded successfully (exact match)")

    # Create provenance info for fine-tuning
    checkpoint_md5 = compute_md5(checkpoint_path)
    source_checkpoint_info = SourceCheckpointInfo(
        path=str(checkpoint_path),
        md5=checkpoint_md5,
    )
    _print(f"  Source checkpoint MD5: {checkpoint_md5}")

    return model, source_checkpoint_info


def _apply_cli_overrides(
    training_config: TrainingConfig,
    *,
    neutral_model_name: Optional[str] = None,
    multihit_model_name: Optional[str] = None,
    dataset_mode: Optional[str] = None,
    cv_fold: Optional[int] = None,
    cv_num_folds: Optional[int] = None,
    cv_seed: Optional[int] = None,
    cv_group_col: Optional[str] = None,
    cv_binding_split_strategy: Optional[str] = None,
) -> TrainingConfig:
    """Apply CLI overrides to training config.

    Covers neutral model fields, dataset mode, and on-the-fly CV fold
    selection. CV overrides build a fresh ``CvSplitConfig`` (validated
    via Pydantic) so a single YAML config can drive K independent
    fold-index runs from a Snakefile without YAML edits per fold.
    """
    neutral_model_updates = {
        k: v
        for k, v in [
            ("model_name", neutral_model_name),
            ("multihit_model_name", multihit_model_name),
        ]
        if v is not None
    }
    if neutral_model_updates and training_config.neutral_model is not None:
        training_config.neutral_model = training_config.neutral_model.model_copy(
            update=neutral_model_updates
        )
    if dataset_mode is not None:
        training_config = training_config.model_copy(
            update={"dataset_mode": dataset_mode}
        )
    if (cv_fold is None) != (cv_num_folds is None):
        raise ValueError(
            "cv_fold and cv_num_folds must be provided together "
            f"(got cv_fold={cv_fold}, cv_num_folds={cv_num_folds})."
        )
    if cv_fold is None and (
        cv_seed is not None
        or cv_group_col is not None
        or cv_binding_split_strategy is not None
    ):
        raise ValueError(
            "cv_seed, cv_group_col, and cv_binding_split_strategy require "
            "cv_fold and cv_num_folds to be set."
        )
    if cv_fold is not None and cv_num_folds is not None:
        # Every field not passed on the CLI keeps the YAML cv_split block's
        # value (if any), then the CvSplitConfig default — letting the user
        # pass --cv-fold alone with a YAML cv_split block carrying group_col /
        # seed / binding_split_strategy / val_num_folds.
        existing = training_config.cv_split
        cv_kwargs: Dict[str, Any] = existing.model_dump() if existing else {}
        cv_kwargs.update(num_folds=cv_num_folds, fold_index=cv_fold)
        cli_overrides = {
            "seed": cv_seed,
            "group_col": cv_group_col,
            "binding_split_strategy": cv_binding_split_strategy,
        }
        cv_kwargs.update({k: v for k, v in cli_overrides.items() if v is not None})
        new_cv = CvSplitConfig(**cv_kwargs)
        training_config = training_config.model_copy(update={"cv_split": new_cv})
    return training_config







def freeze_task_head(trainer: "Trainer", task: "Task", tasks: List["Task"]) -> bool:
    """Freeze ``task``'s head parameters, unless another task shares the head.

    Inlined from the deleted ``dasm2/training/hooks/overfit_guard.py`` (#4),
    whose ``OverfitGuardHook`` was the other caller. ``requires_grad`` is not
    part of ``state_dict``, so a resumed run re-derives the freeze here for a
    task whose ``loss_weight`` is already ``0.0``.

    Returns:
        True if parameters were frozen, False if skipped due to head sharing.
    """
    sharing_tasks = [t for t in tasks if t.head_type == task.head_type]
    if len(sharing_tasks) > 1:
        return False
    head = trainer.unwrapped_model.prediction_heads[task.head_type.value]
    for parameter in head.parameters():
        parameter.requires_grad = False
    return True


def train_model_from_datasets(  # noqa: C901
    config_name: str,
    head_datasets: Dict[str, Tuple[Any, Any]],
    neutral_model: None,
    output_dir: str = "checkpoints",
    max_train_steps: Optional[int] = None,
    eval_every_steps: Optional[int] = None,
    max_eval_steps: Optional[int] = None,
    cycles: int = 4,
    batch_size: Optional[int] = None,
    val_batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    optimizer_name: Optional[str] = None,
    device: Optional[str] = None,
    notes: Optional[str] = None,
    seed: int = 42,
    checkpoint_every_n_evals: Optional[int] = None,
    checkpoint_every_n_steps: Optional[int] = None,
    no_log: bool = False,
    checkpoint: Optional[str] = None,
    gradient_accumulation_steps: Optional[int] = None,
    early_stopping_patience: Optional[int] = None,
    num_dataloader_workers: Optional[int] = None,
    multi_gpu: bool = False,
    mixed_precision: Optional[MixedPrecision] = None,
    training_config: Optional[TrainingConfig] = None,
    start_cycle: int = 0,
    loader_cycle_start: Optional[int] = None,
    cache_dir: str = ".dasm2_cache",
    max_tokens_per_batch: Optional[int] = None,
    bl_optimization_num_workers: Optional[int] = None,
    bl_optimization_max_chunk_size: Optional[int] = None,
    freeze_branch_lengths: Optional[bool] = None,
    freeze_branch_lengths_after_first_cycle: Optional[bool] = None,
    init_branch_lengths_from_input: Optional[bool] = None,
    save_training_state: Optional[bool] = None,
    resume: Optional[str] = None,
    recompute_branch_lengths_on_resume: bool = False,
    freeze_encoder: Optional[bool] = None,
    bl_optimization_parallel: bool = True,
) -> None:
    """Train a DASM2 model from in-memory Dataset objects.

    Core training function for programmatic workflows. Accepts pre-constructed
    Dataset objects (e.g., BranchLengthDatasetBundle from the unified pipeline).

    For CLI usage, use train_model() which handles data loading from files/registry.

    Args:
        config_name: Named config (e.g., '8m', '77k') or path to config YAML.
            Used for loading model architecture and default training parameters.
        head_datasets: Mapping of head type string to (train_dataset, val_dataset) tuple.
            Keys are head type strings (e.g., "SelectionFactorHead", "DMSFunctionalScoreHead").
            Values are tuples of compatible Dataset objects for that head type.
            The first train dataset's `.nickname` attribute is used for checkpoint
            filenames and training logs.
        neutral_model: Neutral model instance for checkpoint metadata.
            Should match the model used to compute nt_rates/nt_csps in
            evolutionary datasets. For Whichmut heads, used only for metadata.
        output_dir: Directory to save model checkpoints.
        max_train_steps: Optimizer steps per ``_train_for_steps`` invocation
            (per task per cycle in sequential mode, per cycle in joint mode).
            If None, uses config value.
        eval_every_steps: Validation cadence in optimizer steps. If None, uses
            config value.
        max_eval_steps: Maximum micro-batches per evaluation; None exhausts
            the validation loader. If None, uses config value.
        cycles: Number of training cycles. Use >1 for branch length optimization
            with the DASM-family losses (LossType.DASM / .QDASM / .HYBRID).
            For Whichmut-family losses, typically use cycles=1.
        batch_size: Per-device training batch size (used together with
            ``gradient_accumulation_steps`` to set the effective batch size).
            If None, uses config value.
        val_batch_size: Per-device validation batch size. Validation has no
            backward pass, so this typically runs 2-8x larger than
            ``batch_size`` on the same device. If None, uses config value.
        learning_rate: Learning rate for optimizer. If None, uses config value.
        weight_decay: Weight decay for regularization. If None, uses config value.
        optimizer_name: Optimizer type (RMSprop, Adam, SGD). If None, uses config value.
        device: Device preference for training (e.g., "cuda", "cpu").
            Ignored when multi_gpu=True (accelerate manages device placement).
        notes: Optional notes about this training run (saved in checkpoint).
        seed: Random seed for reproducibility.
        checkpoint_every_n_evals: If set, save checkpoints every N evaluations
            during training (in addition to cycle-end checkpoints).
        checkpoint_every_n_steps: If set, also save every N optimizer steps,
            keeping only the newest two such saves.
        no_log: Disable training logging (default: enabled).
        checkpoint: Path to checkpoint file to load weights from for fine-tuning.

            **Weight Loading**: Weights are loaded with ``strict=False``, allowing
            partial matching. Encoder weights transfer; new heads are randomly
            initialized; checkpoint heads not in your config are discarded.

            **Metadata Behavior**: The saved checkpoint's metadata is reconstructed
            from your training config, not preserved from the source. Only heads
            in your config appear in the saved metadata. The ``source_checkpoint``
            field tracks provenance (path + MD5) back to the original.

            **Important**: Heads not in your config are permanently removed from
            the fine-tuned checkpoint. To preserve a head without training it,
            include it in your config with ``loss_weight: 0``.
        gradient_accumulation_steps: Number of micro-batches to accumulate gradients
            over before performing an optimizer step. If None, uses config value.
        early_stopping_patience: Stop if validation loss doesn't improve for this
            many evaluations. If None, uses config value.
        multi_gpu: Enable multi-GPU training via HuggingFace accelerate.
            When True, must be launched with ``accelerate launch``.
        mixed_precision: Mixed precision mode ("fp16", "bf16", or None).
            Requires multi_gpu=True.
        training_config: Pre-resolved TrainingConfig. When provided, skips
            reloading from YAML (preserves runtime config modifications such
            as PerSiteChildHead wildtype_sequence resolution).
        start_cycle: Cycle to start from (0-indexed). >0 resumes mid-pipeline:
            whole cycles are skipped and the BLO ``pre_cycle`` hook hard-loads
            cached branch lengths for cycle ``start_cycle`` (raises on miss).
            Combine with ``checkpoint=`` to load a model-only checkpoint. 0
            (default) trains normally from cycle 0.
        loader_cycle_start: If set, seed the DDP shuffle epoch
            (``Trainer._loader_cycle``) before training so a resumed cycle
            reproduces the exact data order a continuous run would have had.
            ``_loader_cycle`` advances once per loader exhaust (~once per
            epoch), not once per cycle, so a faithful cycle-N resume must
            preload its cycle-N-entry value. None (default) leaves it at 0.
        cache_dir: Directory for cached artifacts (branch lengths). Defaults to
            ``.dasm2_cache`` (CWD-relative), matching production. Point at an
            isolated dir for a debug resume so the post-cycle BLO hook can't
            overwrite the shared cache (the cache key has no cycle component).
        max_tokens_per_batch: If set, override the config's token-budget batching
            cap (padded tokens per micro-batch). None uses the config value.
        bl_optimization_num_workers: If set, override the config's post-cycle
            branch-length-optimization worker count. None uses the config value.
        bl_optimization_max_chunk_size: If set, override the config's post-cycle
            branch-length-optimization chunk-size cap (host-RAM bound). None uses
            the config value.
        freeze_branch_lengths_after_first_cycle: If set, override the config's
            ``freeze_branch_lengths_after_first_cycle``. True lets cycle-0
            ``pre_cycle`` set the branch lengths and then holds them fixed for
            the rest of the run. What
            cycle 0 does is handler-dependent: with a calibrated
            ``bl_init_strategy`` it runs the mutation-frequency init and
            optimizes nothing; without one it falls through to the cache and
            then to model-based optimization. Mutually exclusive with
            ``freeze_branch_lengths``, which also freezes cycle 0.
        freeze_branch_lengths: If set, override the config's
            ``freeze_branch_lengths``. True disables all branch-length
            re-estimation (pre_cycle and post_cycle both become no-ops), so
            branch lengths already present in the input data are used
            unchanged throughout training. None uses the config value.
        init_branch_lengths_from_input: If set, override the config's
            ``init_branch_lengths_from_input``. True skips cycle 0's
            branch-length cold start, so cycle 0 trains at the input's
            ``branch_length`` column, and re-optimizes after every cycle as
            usual. Mutually exclusive with both freezes.
        save_training_state: If set, override the config's
            ``save_training_state``. True makes ``CheckpointSaveHook`` save
            full resumable checkpoints (optimizer, scheduler, RNG, step
            counters) instead of weights only, so they can later be passed
            to ``resume=``. None uses the config value, EXCEPT when
            ``resume`` is set: then None defaults to True (a resumed run
            keeps producing resumable checkpoints unless explicitly told
            not to, so a second interruption can still be resumed) — pass
            False explicitly to opt back out.
        freeze_encoder: If set, override the config's ``freeze_encoder``.
            True sets ``requires_grad = False`` on every encoder parameter
            before the optimizer is built, so only head parameters are
            trained (head-only fine-tuning). None uses the config value. On
            ``--resume``, the checkpoint's own ``freeze_encoder`` value is
            what actually applies (``resume_training()`` rebuilds the model
            from the checkpoint's config); passing an explicit value here
            that conflicts with the checkpoint's raises rather than being
            silently ignored.
        resume: Path to a resumable checkpoint (one saved with
            ``save_training_state=True``). Reconstructs the Trainer via
            ``resume_training()``, restoring optimizer/scheduler state, RNG,
            and step counters, and continues from the checkpoint's saved
            cycle/task/step — unlike ``checkpoint=``, which loads weights
            only and resets the optimizer/LR schedule. Mutually exclusive
            with ``checkpoint``, ``start_cycle`` (non-zero), and
            ``loader_cycle_start`` (all three are derived from the
            checkpoint's own training state). Works with ``multi_gpu``, but
            only at the world size and
            ``mixed_precision`` mode the checkpoint was trained at — a
            different world size changes the effective batch size under a
            restored LR schedule, a different precision mode changes the
            numerics, and either raises. The current run's ``tasks``
            (head_type/loss_type pairs) must match the checkpoint's — a
            mismatch raises.
        recompute_branch_lengths_on_resume: If True, a cache-miss in the
            branch-length optimizer's resume-entry-point (hit whenever
            training starts mid-pipeline at ``start_cycle > 0``, whether via
            ``resume`` or the ``checkpoint``+``start_cycle`` combo) falls
            back to model-based re-optimization instead of raising. Use
            this only to extend a completed run past its originally planned
            final cycle (no historical cache exists to protect); leave False
            (default) when resuming an interrupted run, where the hard
            raise guards against silently diverging from that run's own
            branch-length history. Has no effect unless training starts
            mid-pipeline.

    Raises:
        ValueError: If head types in head_datasets don't match config.
        ValueError: If ``resume`` is combined with ``checkpoint``, a non-zero
            ``start_cycle``, or ``loader_cycle_start``; if
            ``recompute_branch_lengths_on_resume`` is set but can never take
            effect; if the checkpoint predates the resumable-checkpoint
            format (missing ``training_config`` metadata); if the
            checkpoint's tasks don't match the current run's tasks; if the
            checkpoint's recorded world size or ``mixed_precision`` mode
            differs from this run's; or if an override such as
            ``batch_size``, ``learning_rate``, or ``freeze_encoder`` is
            passed alongside ``--resume`` with a value that conflicts with
            the checkpoint's own recorded config.
        FileNotFoundError: If ``resume`` or ``checkpoint`` names a path that
            doesn't exist.

    Example:
        >>> from dasm2.train_command import train_model_from_datasets
        >>> from dasm2.data.evolutionary_dataset import create_whichmut_dataset
        >>> # Create datasets via unified pipeline (user's responsibility)
        >>> train_bundle = create_whichmut_dataset(train_df, ...)
        >>> val_bundle = create_whichmut_dataset(val_df, ...)
        >>> train_model_from_datasets(
        ...     config_name="8m",
        ...     head_datasets={"SelectionFactorHead": (train_bundle, val_bundle)},
        ...     neutral_model=neutral_model,
        ... )
    """
    # Validate multi-GPU parameter combinations
    if mixed_precision is not None and not multi_gpu:
        raise ValueError(
            f"mixed_precision={mixed_precision!r} requires multi_gpu=True "
            f"(mixed precision is only supported via AccelerateBackend)"
        )
    if multi_gpu and device is not None:
        raise ValueError(
            "device cannot be specified with multi_gpu=True "
            "(accelerate manages device placement)"
        )

    _validate_resume_cli_args(
        resume,
        checkpoint,
        start_cycle,
        loader_cycle_start,
        recompute_branch_lengths_on_resume,
    )

    # A resumable checkpoint's own config defaults save_training_state to
    # False, so a bare `--resume ckpt.zip` (no explicit --save_training_state)
    # would otherwise silently produce a WEIGHTS-ONLY checkpoint this time --
    # inheriting non-resumability from an explicitly-resumable parent, so a
    # second interruption couldn't be resumed. Default to True on resume
    # unless the caller explicitly overrides it either way.
    if resume is not None and save_training_state is None:
        save_training_state = True

    # See TRAINING_NUM_THREADS; set before any model/data creation.
    torch.set_num_threads(TRAINING_NUM_THREADS)
    # Set random seeds for reproducibility before any model/data creation
    set_seed(seed)

    # Load base config from YAML (unless a pre-resolved config was provided,
    # e.g. with PerSiteChildHead wildtype_sequence resolved from manifest)
    if training_config is None:
        training_config = get_config(config_name)

    # Fail fast (CONSTITUTION IV): resume_training() does not thread
    # combine_gradient_hooks through, so this combination can never work --
    # catch it before model creation, dataset loading, or logger setup. This
    # only catches the *current* run's config; the checkpoint's own config
    # (loaded further down) could also carry combine_gradient_hooks with no
    # trace of it in this run's config, which _validate_resume_checkpoint_config
    # checks separately once that metadata is available.
    if training_config.combine_gradient_hooks and resume is not None:
        raise NotImplementedError(
            "combine_gradient_hooks is not yet supported with --resume "
            "(resume_training() does not thread it through) — start a "
            "fresh run instead, or drop combine_gradient_hooks from the "
            "config."
        )

    # head_optimizer used to have the same unsupported-with-
    # --resume gap as combine_gradient_hooks above, but resume_training()
    # now rebuilds it from the CHECKPOINT's own config the same way it
    # rebuilds the main optimizer — this run's own ``training_config.head_optimizer``
    # is simply ignored on the --resume path, exactly like
    # ``training_config.optimizer``/``training_config.scheduler`` already are.

    _reject_untrainable_heads(training_config)

    head_configs = training_config.model.heads

    # Validate data covers exactly all heads in config
    config_heads = {h.head_type.value for h in head_configs}
    provided_heads = set(head_datasets.keys())
    _validate_head_datasets_match_config(config_name, config_heads, provided_heads)

    # Resume: validate the current run's tasks match the checkpoint's before
    # any data wrapping/model creation. resume_training() rebuilds the model
    # from the CHECKPOINT's own training_config and attaches task_losses from
    # THIS run's tasks — a mismatch there would silently produce a model
    # whose task_losses/heads don't correspond to the checkpoint's, rather
    # than a clear error.
    if resume is not None:
        resume_path = Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        checkpoint_metadata = load_model_metadata(resume_path)
        if "training_config" not in checkpoint_metadata:
            raise ValueError(
                f"--resume checkpoint {resume_path} is missing 'training_config' "
                "metadata; it may predate the resumable-checkpoint format "
                "and cannot be resumed."
            )
        checkpoint_training_config = TrainingConfig(
            **checkpoint_metadata["training_config"]
        )
        # Read here (the metadata is already in hand) but validated further
        # down, once the backend exists to compare against. Both are None for
        # any checkpoint written before they were recorded.
        checkpoint_training_metadata = checkpoint_metadata.get("training_metadata", {})
        checkpoint_num_processes = checkpoint_training_metadata.get("num_processes")
        checkpoint_mixed_precision = checkpoint_training_metadata.get("mixed_precision")
        _validate_resume_checkpoint_config(
            resume_path, training_config, checkpoint_training_config
        )
        # Built from the map's own keys (rather than a hand-written literal)
        # so the two can never drift out of sync -- every key here is a
        # same-named local (a parameter of this function). Must be a plain
        # `for` loop, not a dict/set/list comprehension: comprehensions get
        # their own scope pre-3.12 (PEP 709 only inlines them into the
        # enclosing frame from 3.12 on), so `locals()` called inside one
        # sees only the comprehension's own locals (just `name`) on 3.11
        # and raises KeyError -- verified by CI, which caught exactly this
        # cross-version divergence when the earlier comprehension form
        # passed on 3.12 (this dev box's version) but failed on 3.11.
        resume_overrides: Dict[str, Any] = {}
        for override_name in _RESUME_OVERRIDE_CHECKPOINT_PATHS:
            resume_overrides[override_name] = locals()[override_name]
        _validate_resume_overrides_dont_conflict(
            resume_path, checkpoint_training_config, resume_overrides
        )

    # Build update dictionaries for parameter overrides (filtering out None values)
    training_updates: Dict[str, Any] = {}
    if max_train_steps is not None:
        training_updates["max_train_steps"] = max_train_steps
    if eval_every_steps is not None:
        training_updates["eval_every_steps"] = eval_every_steps
    if max_eval_steps is not None:
        training_updates["max_eval_steps"] = max_eval_steps
    if batch_size is not None:
        training_updates["batch_size"] = batch_size
    if val_batch_size is not None:
        training_updates["val_batch_size"] = val_batch_size
    if gradient_accumulation_steps is not None:
        training_updates["gradient_accumulation_steps"] = gradient_accumulation_steps
    if early_stopping_patience is not None:
        training_updates["early_stopping_patience"] = early_stopping_patience
    if num_dataloader_workers is not None:
        training_updates["num_dataloader_workers"] = num_dataloader_workers
    if max_tokens_per_batch is not None:
        training_updates["max_tokens_per_batch"] = max_tokens_per_batch
    if bl_optimization_num_workers is not None:
        training_updates["bl_optimization_num_workers"] = bl_optimization_num_workers
    if bl_optimization_max_chunk_size is not None:
        training_updates["bl_optimization_max_chunk_size"] = (
            bl_optimization_max_chunk_size
        )
    if freeze_branch_lengths is not None:
        training_updates["freeze_branch_lengths"] = freeze_branch_lengths
    if freeze_branch_lengths_after_first_cycle is not None:
        training_updates["freeze_branch_lengths_after_first_cycle"] = (
            freeze_branch_lengths_after_first_cycle
        )
    if init_branch_lengths_from_input is not None:
        training_updates["init_branch_lengths_from_input"] = (
            init_branch_lengths_from_input
        )
    if save_training_state is not None:
        training_updates["save_training_state"] = save_training_state
    if freeze_encoder is not None:
        training_updates["freeze_encoder"] = freeze_encoder

    optimizer_updates: Dict[str, Any] = {}
    if learning_rate is not None:
        optimizer_updates["learning_rate"] = learning_rate
    if weight_decay is not None:
        optimizer_updates["weight_decay"] = weight_decay
    if optimizer_name is not None:
        optimizer_updates["optimizer_name"] = optimizer_name

    # Apply overrides to TrainingConfig
    if training_updates:
        training_config = training_config.model_copy(update=training_updates)
    if optimizer_updates:
        training_config.optimizer = training_config.optimizer.model_copy(
            update=optimizer_updates
        )

    # Apply matmul precision before model construction so all FP32 matmuls
    # (including those outside the BF16 autocast context) benefit from TF32.
    torch.set_float32_matmul_precision(training_config.float32_matmul_precision)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create compute backend
    backend: ComputeBackend
    if multi_gpu:
        backend = AccelerateBackend(mixed_precision=mixed_precision)
        if backend.is_main_process:
            print(f"Multi-GPU training enabled: {backend.num_processes} processes")
            if mixed_precision:
                print(f"Mixed precision: {mixed_precision}")
    else:
        backend = SingleGPUBackend(str(pick_device(device)))

    # Convenience: only rank 0 prints status messages in multi-GPU mode.
    _print = _rank_guarded_print(backend.is_main_process)

    _print(f"Random seed: {seed}")

    # World size and precision mode are launch-environment facts, so these
    # cannot ride in _validate_resume_overrides_dont_conflict above: that runs
    # before the backend exists, and its override table is keyed on a
    # None-means-not-passed convention that a world size (always present,
    # defaulting to 1) and a precision mode (likewise, "no") cannot satisfy.
    # Here the backend exists and no model has been constructed yet, which is
    # the earliest correct point.
    #
    # The batch-size numbers in the world-size message come from the
    # CHECKPOINT's config, not this run's: resume_training() rebuilds the
    # trainer from the former, and the two can differ silently whenever the
    # difference was declared in YAML rather than passed on the CLI.
    if resume is not None:
        _validate_resume_world_size_matches(
            resume_path,
            checkpoint_num_processes,
            backend.num_processes,
            checkpoint_training_config.batch_size,
            checkpoint_training_config.gradient_accumulation_steps,
        )
        _validate_resume_mixed_precision_matches(
            resume_path,
            checkpoint_mixed_precision,
            backend.mixed_precision,
        )

    # Create model on the backend's init device (CPU for accelerate, GPU for
    # single). Skipped for --resume: resume_training() rebuilds the model
    # from the CHECKPOINT's own training_config below, not from this run's.
    source_checkpoint_info: Optional[SourceCheckpointInfo] = None
    if resume is None:
        model, source_checkpoint_info = _create_model_and_maybe_load_checkpoint(
            training_config, backend, checkpoint, _print
        )

    # Freeze encoder parameters if requested (for head-only fine-tuning).
    # Reads the post-override TrainingConfig value (not the raw parameter)
    # so it matches whatever _validate_resume_overrides_dont_conflict just
    # checked, and so the same field resume_training() consults from the
    # checkpoint's own config is the single source of truth here too.
    # Guarded by ``resume is None`` like the model-creation block above:
    # on --resume there is no local ``model`` to freeze here at all --
    # resume_training() applies the checkpoint's own freeze_encoder value
    # to the model it rebuilds internally (before create_optimizer runs).
    if resume is None and training_config.freeze_encoder:
        frozen_count = sum(1 for _ in model.encoder.parameters())
        model.freeze_encoder()
        _print(
            f"Encoder frozen: {frozen_count} parameters. "
            f"Only head parameters will be trained."
        )

    # Log dataset sizes
    for head_type_str, (train_item, val_item) in head_datasets.items():
        if isinstance(train_item, DatasetBundle):
            assert isinstance(val_item, DatasetBundle), (
                f"Mismatched types for {head_type_str}: "
                f"train is DatasetBundle but val is {type(val_item).__name__}"
            )
            train_len = len(train_item.dataset)  # type: ignore[arg-type]
            val_len = len(val_item.dataset)  # type: ignore[arg-type]
        else:
            train_len = len(train_item)  # type: ignore[arg-type]
            val_len = len(val_item)  # type: ignore[arg-type]
        _print(f"Dataset for {head_type_str}: Train={train_len}, Val={val_len}")

    # Create tasks from head configs with per-head datasets
    tasks = _create_tasks_from_config(
        head_configs=head_configs,
        head_datasets=head_datasets,
        output_path=output_path,
        config_name=config_name,
        notes=notes,
        checkpoint_every_n_evals=checkpoint_every_n_evals,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        is_main_process=backend.is_main_process,
        task_configs=training_config.tasks,
        save_training_state=training_config.save_training_state,
    )

    # On --resume, carry over a task's checkpoint-recorded loss_weight when
    # it is 0.0: OverfitGuardHook mutates Task.loss_weight live (never back
    # into the YAML/CLI config), so `tasks` above was just rebuilt from
    # `training_config.tasks`' original (pre-freeze) value. Without this,
    # a task the guard had already frozen comes back with its original
    # nonzero loss_weight and un-frozen parameters, re-arming the guard for
    # another `patience` evals -- see OverfitGuardHook's docstring.
    # `_validate_resume_checkpoint_config` above already guarantees
    # `tasks` and `checkpoint_training_config.tasks` are the same length and
    # in the same (head_type, loss_type) order, so a positional zip is safe.
    if resume is not None:
        for task, checkpoint_task_config in zip(
            tasks, checkpoint_training_config.tasks
        ):
            if checkpoint_task_config.loss_weight == 0.0:
                task.loss_weight = 0.0

    # Effective batch size accounts for all parallelism dimensions:
    # per-GPU batch × gradient accumulation steps × number of DDP processes
    effective_batch_size = compute_effective_batch_size(
        training_config.batch_size,
        training_config.gradient_accumulation_steps,
        backend.num_processes,
    )

    # Create logger unless disabled (only on main process to avoid duplicate logs)
    logger = None
    if not no_log and backend.is_main_process:
        env_metadata = create_training_metadata_info(notes=notes)

        logger_metadata = {
            "config_name": config_name,
            "model_config": training_config.model.model_dump(),
            "optimizer_config": training_config.optimizer.model_dump(),
            "neutral_model_config": (
                neutral_model.to_config().model_dump() if neutral_model else None
            ),
            "training_params": {
                "max_train_steps": training_config.max_train_steps,
                "eval_every_steps": training_config.eval_every_steps,
                "max_eval_steps": training_config.max_eval_steps,
                "num_cycles": cycles,
                "per_device_batch_size": training_config.batch_size,
                "batch_size": effective_batch_size,
                "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
                "num_processes": backend.num_processes,
                "min_learning_rate": 1e-6,
                "float32_matmul_precision": training_config.float32_matmul_precision,
                "use_mixed_precision": training_config.use_mixed_precision,
            },
            "tasks": [
                {
                    "head_type": task.head_type.value,
                    "dataset_nickname": getattr(
                        task.train_dataset,
                        "nickname",
                        type(task.train_dataset).__name__,
                    ),
                    "train_size": len(task.train_dataset),  # type: ignore[arg-type]
                    "val_size": len(task.val_dataset),  # type: ignore[arg-type]
                }
                for task in tasks
            ],
            "environment": env_metadata.model_dump(),
        }

        logger = TrainingLogger.create(
            output_dir=output_dir,
            config_name=config_name,
            metadata=logger_metadata,
        )
        _print(f"Logging to: {logger.log_path}")

    # Build the StepBudget consumed by the trainer's step-based loop. The
    # polynomial scheduler reads ``total_training_steps`` internally from
    # ``step_budget.max_train_steps`` (per cycle).
    scheduler_config = training_config.scheduler
    step_budget = StepBudget(
        max_train_steps=training_config.max_train_steps,
        eval_every_steps=training_config.eval_every_steps,
        max_eval_steps=training_config.max_eval_steps,
    )
    if isinstance(scheduler_config, WarmupPolynomialSchedulerConfig):
        _print(
            f"Polynomial scheduler: {step_budget.max_train_steps} optimizer steps per cycle"
        )

    # Per-cycle budget summary. Multi-task runs joint training (one
    # ``_train_for_steps`` over all tasks); single-task uses the same
    # budget on the sequential path.
    assert tasks, "_create_tasks_from_config must produce at least one task"
    mode = "joint" if len(tasks) > 1 else "sequential"
    budget_summary = f"max_train_steps={step_budget.max_train_steps} per cycle ({mode})"

    # start_cycle/start_cycle_task_index/start_cycle_step for the
    # multi_cycle_train(_joint) call below. --resume overwrites all three
    # from the checkpoint's own saved position; otherwise start_cycle is the
    # (validated-zero-unless-resuming) CLI value and the other two stay 0
    # (only the debug --checkpoint+--start_cycle combo skips whole cycles;
    # it has no notion of a mid-cycle task/step position).
    # Top-level (not per-task) gradient-combination hooks,
    # e.g. PCGradHook — resolved once here rather than per-task, since
    # combining gradients is inherently cross-task. Empty by default, which
    # reproduces today's plain-sum combination exactly (Trainer's own
    # no-hook branch).
    # PCGradHook was the only ``GradientCombineHook``, and it went with the
    # multi-task heads it combined gradients for (#4). An empty list is the
    # plain-sum combination the Trainer's own no-hook branch already does; a
    # config that asks for one fails here rather than being ignored.
    if training_config.combine_gradient_hooks:
        raise ValueError(
            "combine_gradient_hooks is configured, but this fork registers "
            "none. The gradient-combination hooks were removed with the "
            "multi-task heads (#4)."
        )
    combine_gradient_hooks: List[GradientCombineHook] = []

    start_cycle_task_index = 0
    start_cycle_step = 0
    if resume is not None:
        _print(f"Resuming from checkpoint: {resume_path}")
        resume_result = resume_training(
            filepath=resume_path,
            tasks=tasks,
            cache_dir=Path(cache_dir),
            enable_cache=True,
            backend=backend,
        )
        trainer = resume_result.trainer
        if logger is not None:
            trainer.logger = logger
        _print(
            f"  Attention backend: {_detect_attention_backend(trainer.unwrapped_model)}"
        )
        start_cycle = resume_result.start_cycle
        start_cycle_task_index = resume_result.start_cycle_task_index
        start_cycle_step = resume_result.start_cycle_step
        _print(
            f"Resumed at cycle={start_cycle}, "
            f"task_index={start_cycle_task_index}, "
            f"step={start_cycle_step} "
            "(optimizer/scheduler/RNG state restored)"
        )
    else:
        _print(f"Creating trainer with config: {config_name}")
        _print(
            f"Training params: {budget_summary}, "
            f"eval_every_steps={step_budget.eval_every_steps}, "
            f"max_eval_steps={step_budget.max_eval_steps}, "
            f"batch_size={effective_batch_size}, "
            f"per_device_batch_size={training_config.batch_size}, "
            f"gradient_accumulation_steps={training_config.gradient_accumulation_steps}, "
            f"num_processes={backend.num_processes}, "
            f"lr={training_config.optimizer.learning_rate}, "
            f"optimizer={training_config.optimizer.optimizer_name}, "
            f"float32_matmul_precision={training_config.float32_matmul_precision}"
        )
        if training_config.use_mixed_precision:
            _print("  Mixed precision: BF16 enabled")
        if training_config.gradient_accumulation_steps > 1:
            _print(
                f"  Gradient accumulation: {training_config.gradient_accumulation_steps} steps"
            )
        if training_config.head_optimizer is not None:
            _print(
                f"  head_optimizer on "
                f"{training_config.head_optimizer_head_type}: "
                f"lr={training_config.head_optimizer.learning_rate}, "
                f"weight_decay={training_config.head_optimizer.weight_decay}"
            )
        trainer = create_trainer(
            model=model,
            neutral_model=neutral_model,
            tasks=tasks,
            batch_size=training_config.batch_size,
            val_batch_size=training_config.val_batch_size,
            optimizer_config=training_config.optimizer,
            backend=backend,
            config_name=config_name,
            logger=logger,
            gradient_clip_val=training_config.gradient_clip_val,
            early_stopping_patience=training_config.early_stopping_patience,
            early_stopping_min_delta=training_config.early_stopping_min_delta,
            monitor_task=training_config.monitor_task,
            source_checkpoint=source_checkpoint_info,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            scheduler_config=scheduler_config,
            use_mixed_precision=training_config.use_mixed_precision,
            step_budget=step_budget,
            num_dataloader_workers=training_config.num_dataloader_workers,
            cache_dir=cache_dir,
            max_tokens_per_batch=training_config.max_tokens_per_batch,
            dataloader_prefetch_factor=training_config.dataloader_prefetch_factor,
            dataloader_pin_memory=training_config.dataloader_pin_memory,
            data_seed=seed,
            combine_gradient_hooks=combine_gradient_hooks,
            head_optimizer_config=training_config.head_optimizer,
            head_optimizer_head_type=training_config.head_optimizer_head_type,
        )

        # Debug knob for reproducing a cycle-1 stall. ``start_cycle`` skips
        # whole cycles, threading into the BLO ``pre_cycle`` hook so cycle
        # ``start_cycle`` hard-loads its cached branch lengths (raises on
        # miss). A model-only checkpoint loaded via ``checkpoint=`` above +
        # ``start_cycle>0`` lets a debug run resume mid-pipeline without
        # ``training_state.pth``. ``loader_cycle_start`` seeds the DDP
        # shuffle epoch (``_loader_cycle``) before training so a resumed
        # cycle reproduces the exact per-cycle data order a continuous run
        # would have had (it advances once per loader exhaust, ~once per
        # epoch, not once per cycle). Both default to no-ops; --resume
        # restores loader_cycle from the checkpoint's TrainingState instead
        # (validated mutually exclusive with loader_cycle_start above).
        if loader_cycle_start is not None:
            _print(
                f"Seeding loader_cycle (DDP shuffle epoch) to {loader_cycle_start} "
                "before training (resume/debug)"
            )
            trainer._loader_cycle = loader_cycle_start

    # Re-derive OverfitGuardHook's freeze for any task whose loss_weight is
    # already 0.0. ``requires_grad`` isn't part of a checkpoint's
    # state_dict, so on --resume of a run the hook already froze, tasks are
    # rebuilt fresh (requires_grad=True) even though loss_weight=0.0
    # persists via this run's own config -- leaving weight_decay free to
    # erode the "frozen" head again (see OverfitGuardHook's docstring in
    # dasm2/training/hooks/overfit_guard.py). This also covers a task
    # declared with loss_weight=0.0 from the start on a fresh run.
    for task in tasks:
        if task.loss_weight == 0.0:
            freeze_task_head(trainer, task, tasks)

    # Train model using the step-based multi-cycle wrapper. For --resume,
    # ``trainer.step_budget`` is the checkpoint's own (authoritative) budget,
    # which may differ from the local ``step_budget`` built from this run's
    # CLI/config values (resume_training() ignores those for trainer state).
    # Both branches above always construct the trainer with a step_budget.
    assert trainer.step_budget is not None, (
        "trainer.step_budget unset; both create_trainer() and resume_training() "
        "always construct the trainer with a StepBudget"
    )
    _print(
        f"Starting training: {trainer.step_budget.max_train_steps} steps × "
        f"{cycles} cycles ({'joint' if len(tasks) > 1 else 'sequential'})"
        + (f", start_cycle={start_cycle}" if start_cycle else "")
    )
    try:
        if len(tasks) > 1:
            _print(f"  Using joint training for {len(tasks)} tasks")
            trainer.multi_cycle_train_joint(
                num_cycles=cycles,
                start_cycle=start_cycle,
                start_cycle_step=start_cycle_step,
            )
        else:
            trainer.multi_cycle_train(
                num_cycles=cycles,
                start_cycle=start_cycle,
                start_cycle_task_index=start_cycle_task_index,
                start_cycle_step=start_cycle_step,
            )

        _print(f"\n{'=' * 70}")
        _print("Training complete!")
        _print(f"{'=' * 70}")
    finally:
        trainer.shutdown()


def _parse_head_data_config(
    data: Dict[str, Any],
) -> Dict[HeadType, List[str]]:
    """Parse head data configuration into validated format.

    Handles both single-dataset heads (string values) and multi-dataset
    heads like AAHWhichmutHead (list values).

    Args:
        data: Mapping of head type string to data path(s).
            Can be str for single-dataset heads or List[str] for multi-dataset heads.

    Returns:
        Mapping of HeadType to list of data paths

    Raises:
        ValueError: If head type is invalid or wrong number of paths provided
    """
    result: Dict[HeadType, List[str]] = {}
    for head_type_str, data_paths in data.items():
        try:
            head_type = HeadType(head_type_str)
        except ValueError:
            valid_types = [h.value for h in HeadType]
            raise ValueError(
                f"Unknown head type: {head_type_str}. Valid types: {valid_types}"
            )

        # Normalize to list
        if isinstance(data_paths, str):
            paths = [data_paths]
        elif isinstance(data_paths, list):
            paths = data_paths
        else:
            raise ValueError(
                f"Invalid data format for {head_type_str}: expected str or list, "
                f"got {type(data_paths).__name__}"
            )

        # Validate path count for gauge-anchored heads (AAH = AA Hybrid)
        if head_type == HeadType.AAH_WHICHMUT_HEAD:
            if len(paths) != 2:
                raise ValueError(
                    f"{head_type_str} requires exactly 2 dataset paths "
                    f"[whichmut_path, aao_path], got {len(paths)}"
                )
        elif len(paths) != 1:
            raise ValueError(
                f"{head_type_str} requires exactly 1 dataset path, got {len(paths)}"
            )

        result[head_type] = paths

    return result






def _resolve_censor_table(
    training_config: "TrainingConfig",
    _print: Callable[..., None],
) -> Optional[Dict[str, "CensorSpec"]]:
    """Load the configured censor table, or None when censoring is off.

    Merges mat-pcp's global mask into every clade when one is configured -- either
    ``masked_nt_sites_file`` (nucleotide-granular, preferred) or the
    older ``codon_sites_to_mask_file``, never both. Every mask file reaches
    the loss through one table, because a clade-independent site mask is what
    ``CensorSpec.site_nt_indices`` already means; see
    :func:`~dasm2.sequence.masking.with_global_nt_sites`.

    Args:
        training_config: Config carrying the censoring paths and the mode.
        _print: Rank-guarded print function.

    Reported rather than silent, because a censored run and an uncensored one
    are otherwise indistinguishable from the training output: censoring changes
    no tensor shape, raises no warning, and produces a perfectly plausible loss
    curve.
    """
    if training_config.censored_mutations_file is None:
        return None
    assert training_config.censored_clades_file is not None  # config validator
    table = load_censor_table(
        training_config.censored_mutations_file,
        training_config.censored_clades_file,
    )
    entries = sum(spec.num_entries for spec in table.values())
    nonempty = sum(1 for spec in table.values() if not spec.is_empty)
    _print(
        f"  UShER censoring: mode={training_config.censor_mode}, "
        f"{entries} entries across {nonempty}/{len(table)} clades "
        f"({training_config.censored_mutations_file})"
    )
    if entries == 0:
        raise ValueError(
            f"{training_config.censored_mutations_file} censors nothing for any "
            f"of its {len(table)} clades. That reproduces the uncensored "
            f"baseline exactly, so configuring it is almost certainly a mistake "
            f"-- most likely the wrong gene's table. Remove the setting to train "
            f"uncensored deliberately."
        )

    # What protects the guard above from the merge is that `entries` and
    # `nonempty` are computed from the *pre-merge* table, not the placement of
    # this block: after merging, every clade has entries, so a count taken
    # afterwards would read as healthy for a per-clade table that had gone missing
    # or belonged to another gene. Keep those two `sum()`s above the merge.
    # `max_aa_seq_len` is set only by heads that size a per-site table from it
    # (`PerSiteParentChildHead`). The transformer configs deliberately leave it
    # unset -- `create_aa_tensors` truncates to it silently, so a value there
    # would cap sequence length rather than describe it. So it is a bound when
    # present and absent otherwise, never a requirement: demanding it would
    # refuse a legitimate transformer config.
    #
    # Nothing is unguarded when it is absent. `CensorMaskedMutations` checks
    # `_max_nt_index >= 3 * num_codons` against every sample and raises naming
    # the gene mismatch, so the too-large direction is caught on the first
    # batch instead of at construction.
    num_codons = training_config.model.max_aa_seq_len
    # Reported separately from the per-clade counts above, not merged into them:
    # after either merge no clade is empty, so a combined "nonempty/total clades"
    # line would read as healthy even if the per-clade table had gone missing
    # entirely.
    # One per branch, in that branch's own units: the codon branch prints a
    # codon site and the nucleotide branch a nucleotide index, so a single
    # shared string would be wrong in one of them.
    nt_extent = (
        f"of {3 * num_codons} nt"
        if num_codons is not None
        else "of ? nt, corpus length checked per sample"
    )
    codon_extent = (
        f"of {num_codons}"
        if num_codons is not None
        else "corpus length checked per sample"
    )

    # `elif`, not a second `if`: the two files are two granularities of one mask
    # and the config validator rejects setting both. Spelled as a branch anyway so
    # that a future edit loosening the validator cannot silently apply the codon
    # mask on top of the nucleotide one, where it would win by containment.
    if training_config.masked_nt_sites_file is not None:
        nt_indices = parse_masked_nt_sites_file(training_config.masked_nt_sites_file)
        table = with_global_nt_sites(table, nt_indices, num_codons)
        _print(
            f"  ... plus {len(nt_indices)} global nucleotide sites censored in "
            f"every clade (max index {max(nt_indices)} {nt_extent}; "
            f"{training_config.masked_nt_sites_file})"
        )
    elif training_config.codon_sites_to_mask_file is not None:
        codon_sites = parse_problematic_sites_file(
            str(training_config.codon_sites_to_mask_file)
        )
        table = with_global_codon_sites(table, codon_sites, num_codons)
        _print(
            f"  ... plus {len(codon_sites)} global codon sites censored in every "
            f"clade (max site {max(codon_sites) + 1} {codon_extent}; "
            f"{training_config.codon_sites_to_mask_file})"
        )
    return table


def _resolve_reference_bases(
    training_config: "TrainingConfig",
    _print: Callable[..., None],
) -> Optional["torch.Tensor"]:
    """Load the configured reference sequence, or None when the rule is off.

    A sibling of :func:`_resolve_censor_table` and deliberately independent of
    it: the reversion rule is clade-independent, so this returns a reference
    whether or not a per-clade table is configured, and neither guard borrows
    from the other.

    Args:
        training_config: Config carrying ``reference_sequence_file`` and the mode.
        _print: Rank-guarded print function.

    Raises:
        ValueError: If the reference is identical to nothing the corpus can
            differ from -- see below -- or if the FASTA is unusable; see
            :func:`~dasm2.sequence.masking.load_reference_base_idxs`.

    Note:
        The "censors nothing" guard here cannot be the table's guard. A censor
        table censoring nothing is visible from the table alone (zero entries),
        but whether a *reference* censors anything depends on the parents, which
        this function never sees. What it can check is the one degenerate case
        that is detectable up front and that is also the bit-exactness test's
        *pass* condition: a reference of a single repeated base cannot be a real
        coding sequence, so it is a mis-specified input rather than a deliberate
        choice. Everything else is left to the ``audit_censor`` rule, which reads
        the corpus and reports the realized rate.
    """
    if training_config.reference_sequence_file is None:
        return None

    # `max_aa_seq_len` is a bound when present and absent otherwise, for the same
    # reason `_resolve_censor_table` treats it that way: the transformer configs
    # deliberately leave it unset. The transform's own per-sample check is exact
    # equality, so a wrong-gene reference is caught on the first batch regardless.
    reference = load_reference_base_idxs(
        training_config.reference_sequence_file,
        num_codons=training_config.model.max_aa_seq_len,
    )
    distinct = int(torch.unique(reference).numel())
    _print(
        f"  Reversion censoring: mode={training_config.censor_mode}, "
        f"reference {reference.numel()} nt ({reference.numel() // 3} codons), "
        f"{distinct}/4 bases present "
        f"({training_config.reference_sequence_file})"
    )
    if distinct <= 1:
        raise ValueError(
            f"{training_config.reference_sequence_file} is a single repeated "
            f"base over {reference.numel()} nt. That is not a coding sequence, "
            f"so this is a mis-specified reference rather than a deliberate "
            f"choice. Remove the setting to train without reversion censoring."
        )
    return reference


def _load_whichmut_head_data(
    data_path: str,
    loss_type: LossType,
    dataset_mode: Literal["precomputed", "streaming"],
    cv_split: Optional[CvSplitConfig] = None,
    censor_table: Optional[Dict[str, "CensorSpec"]] = None,
    censor_mode: CensorMode = "both",
    reference_base_idxs: Optional[torch.Tensor] = None,
    cache_budget: CacheBudgetSpec = CacheBudgetSpec(),
) -> Tuple[Any, Any]:
    """Load train/val data for the whichmut task from a Parquet corpus.

    Parquet is the only format this fork reads. The CSV and pickle arms
    went with the neutral models they needed: every one of them built its
    rates from a ``RatesCSPNeutralModel``, and viral-DASM takes neutral
    rates from the mat-pcp censor tables instead, so ``neutral_model`` is
    ``None`` on every path and those arms could only have raised. The 3Di
    wrapping arm went with ``data/tdi_whichmut.py`` in the same cut.

    Args:
        data_path: Data specification string; must name a Parquet corpus.
        loss_type: The whichmut loss variant. ``WHICHMUT`` is the only one
            ``LOSS_REGISTRY`` can build, so this is checked, not dispatched
            on.
        dataset_mode: "precomputed" or "streaming".
        cv_split: Optional cross-validation fold spec. Bypasses the
            ``in_train`` column and registry holdouts to compute the split
            on the fly.
        censor_table: Optional UShER censor table, applied to both
            splits.
        censor_mode: Which half of censoring to apply.
        reference_base_idxs: Clade-founder bases for reversion censoring.
        cache_budget: Parquet row-group cache RAM budget inputs.

    Returns:
        Tuple of (train_item, val_item).
    """
    if loss_type is not LossType.WHICHMUT:
        raise ValueError(
            f"loss_type={loss_type.value!r} is not something this repository "
            f"can train: LOSS_REGISTRY holds {LossType.WHICHMUT.value!r} "
            f"alone. The other whichmut variant, "
            f"{LossType.AAO_WHICHMUT.value!r}, needs the AAO transform chain "
            f"and a neutral model, neither of which this fork carries."
        )
    if not is_parquet_data_spec(data_path):
        raise ValueError(
            f"{data_path!r} is not a Parquet data spec. The CSV and pickle "
            f"loaders needed a RatesCSPNeutralModel to build whichmut rates; "
            f"viral-DASM reads them from the mat-pcp censor tables, so those "
            f"loaders went with the neutral models and Parquet is the only "
            f"format left."
        )
    warn_if_parquet_dataset_mode_mismatch(dataset_mode)
    return load_whichmut_train_val(
        data_path,
        neutral_model=None,
        cv_split=cv_split,
        censor_table=censor_table,
        censor_mode=censor_mode,
        reference_base_idxs=reference_base_idxs,
        cache_budget=cache_budget,
    )


def train_model(  # noqa: C901
    config_name: str,
    data: Dict[str, Any],
    output_dir: str = "checkpoints",
    max_train_steps: Optional[int] = None,
    eval_every_steps: Optional[int] = None,
    max_eval_steps: Optional[int] = None,
    cycles: int = 4,
    batch_size: Optional[int] = None,
    val_batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    optimizer_name: Optional[str] = None,
    device: Optional[str] = None,
    notes: Optional[str] = None,
    seed: int = 42,
    deterministic: bool = False,
    neutral_model_name: Optional[str] = None,
    multihit_model_name: Optional[str] = None,
    checkpoint_every_n_evals: Optional[int] = None,
    checkpoint_every_n_steps: Optional[int] = None,
    no_log: bool = False,
    checkpoint: Optional[str] = None,
    gradient_accumulation_steps: Optional[int] = None,
    early_stopping_patience: Optional[int] = None,
    dataset_mode: Optional[str] = None,
    num_dataloader_workers: Optional[int] = None,
    multi_gpu: bool = False,
    num_gpus: Optional[int] = None,
    gpus: Union[str, Tuple[int, ...], None] = None,
    mixed_precision: Optional[MixedPrecision] = None,
    freeze_encoder: Optional[bool] = None,
    bl_optimization_parallel: bool = True,
    cv_fold: Optional[int] = None,
    cv_num_folds: Optional[int] = None,
    cv_seed: Optional[int] = None,
    cv_group_col: Optional[str] = None,
    cv_binding_split_strategy: Optional[str] = None,
    start_cycle: int = 0,
    loader_cycle_start: Optional[int] = None,
    cache_dir: str = ".dasm2_cache",
    max_tokens_per_batch: Optional[int] = None,
    bl_optimization_num_workers: Optional[int] = None,
    bl_optimization_max_chunk_size: Optional[int] = None,
    freeze_branch_lengths: Optional[bool] = None,
    freeze_branch_lengths_after_first_cycle: Optional[bool] = None,
    init_branch_lengths_from_input: Optional[bool] = None,
    save_training_state: Optional[bool] = None,
    resume: Optional[str] = None,
    recompute_branch_lengths_on_resume: bool = False,
    # Deprecated — kept so ``--epochs`` on the CLI produces a clear error
    # rather than a confusing "unrecognized argument". Rejected at entry.
    epochs: Optional[int] = None,
) -> None:
    """Train a DASM2 model (CLI interface).

    Loads data from files/registry and delegates to train_model_from_datasets().
    Config values are used as defaults - CLI arguments override only when explicitly provided.

    Args:
        config_name: Named config (e.g., '8m', '77k') or path to config YAML.
        data: Mapping of head type to data path. The dict must have exactly one entry
            per head in the config. Keys are head type strings (e.g., "SelectionFactorHead").
            Values are data paths or nicknames. For a ``dms_matrix_calibration`` loss the
            value is a dataset CSV path or a registered nickname (e.g.
            ``tsuboyama_megascale``), optionally followed by ``:<split_csv>``.
        output_dir: Directory to save model checkpoints.
        max_train_steps: Optimizer steps per ``_train_for_steps`` invocation
            (per task per cycle in sequential mode, per cycle in joint mode).
            If None, uses config value.
        eval_every_steps: Validation cadence in optimizer steps. If None, uses
            config value.
        max_eval_steps: Maximum micro-batches per evaluation; None exhausts
            the validation loader. If None, uses config value.
        cycles: Number of training cycles (use >1 for branch length optimization).
        batch_size: Per-device training batch size (used together with
            ``gradient_accumulation_steps`` to set the effective batch size).
            If None, uses config value.
        val_batch_size: Per-device validation batch size. Validation has no
            backward pass, so this typically runs 2-8x larger than
            ``batch_size`` on the same device. If None, uses config value.
        learning_rate: Learning rate for optimizer. If None, uses config value.
        weight_decay: Weight decay for regularization. If None, uses config value.
        optimizer_name: Optimizer type (RMSprop, Adam, SGD). If None, uses config value.
        device: Device preference for training.
            Ignored when multi_gpu=True (accelerate manages device placement).
        notes: Optional notes about this training run.
        seed: Random seed for reproducibility.
        deterministic: Pin intra-op threads to 1 and enable
            torch.use_deterministic_algorithms (see
            dasm2.utils.torch_utils.set_deterministic_mode). Off by default
            (can slow training); turn on when exact rerun-to-rerun
            reproducibility matters more than speed.
        neutral_model_name: Base SHM model name (e.g., 'ThriftyHumV0.2-59').
            If None, uses config value.
        multihit_model_name: Multihit correction model name.
            If provided, enables multihit correction. If None, uses config value.
        checkpoint_every_n_evals: If set, save checkpoints every N evaluations
            during training (in addition to cycle-end checkpoints).
        checkpoint_every_n_steps: If set, also save every N optimizer steps,
            keeping only the newest two such saves.
        no_log: Disable training logging (default: enabled).
        checkpoint: Path to checkpoint file to load weights from for fine-tuning.
            See ``train_model_from_datasets`` for detailed behavior documentation
            regarding weight loading, metadata reconstruction, and head preservation.
        gradient_accumulation_steps: Number of micro-batches to accumulate gradients
            over before performing an optimizer step. If None, uses config value.
        early_stopping_patience: Stop if validation loss doesn't improve for this
            many evaluations. If None, uses config value.
        dataset_mode: Dataset pipeline mode ("precomputed" or "streaming").
            "precomputed" tokenizes and transforms all samples at init (fast training,
            more memory). "streaming" runs transforms on-the-fly (lower memory, hidden
            by model compute on large models). If None, uses config value.
        multi_gpu: Enable multi-GPU training via HuggingFace accelerate.
            Automatically re-launches under ``accelerate launch`` when needed.
        num_gpus: Number of GPUs to use for multi-GPU training. If None,
            uses all available GPUs. Requires multi_gpu=True.
        gpus: Comma-separated GPU IDs to use (e.g. "1,4"). Sets
            ``CUDA_VISIBLE_DEVICES`` before relaunching under Accelerate, so
            only the specified GPUs are visible. Useful on shared machines.
            Implies ``--multi_gpu``.
        mixed_precision: Mixed precision mode ("fp16", "bf16", or None).
        start_cycle: Cycle to start from (0-indexed). >0 resumes mid-pipeline,
            loading cached branch lengths for cycle ``start_cycle`` via the BLO
            hook; combine with ``checkpoint=`` for a model-only resume. See
            ``train_model_from_datasets`` for details.
        loader_cycle_start: If set, preload the DDP shuffle epoch so a resumed
            cycle reproduces a continuous run's data order. See
            ``train_model_from_datasets`` for details.
        cache_dir: Directory for cached artifacts (branch lengths). Defaults to
            ``.dasm2_cache``. Use an isolated dir for a debug resume so the
            post-cycle BLO hook can't overwrite the shared cache.
        max_tokens_per_batch: If set, override the config's token-budget batching
            cap (padded tokens per micro-batch). None uses the config value.
        bl_optimization_num_workers: If set, override the config's post-cycle
            branch-length-optimization worker count (bounds host RAM). None uses
            the config value.
        bl_optimization_max_chunk_size: If set, override the config's post-cycle
            branch-length-optimization chunk-size cap (bounds host RAM). None uses
            the config value.
        freeze_encoder: If set, override the config's ``freeze_encoder``.
            True sets ``requires_grad = False`` on every encoder parameter
            before the optimizer is built (head-only fine-tuning). None
            uses the config value. Recorded in the checkpoint, so
            ``--resume`` continues with the same frozen/unfrozen encoder
            the interrupted run had; an explicit ``--freeze_encoder`` that
            conflicts with the checkpoint's own value raises.
        freeze_branch_lengths_after_first_cycle: If set, override the config's
            ``freeze_branch_lengths_after_first_cycle``. True lets cycle-0
            ``pre_cycle`` set the branch lengths and then holds them fixed for
            the rest of the run. What
            cycle 0 does is handler-dependent: with a calibrated
            ``bl_init_strategy`` it runs the mutation-frequency init and
            optimizes nothing; without one it falls through to the cache and
            then to model-based optimization. Mutually exclusive with
            ``freeze_branch_lengths``, which also freezes cycle 0.
        freeze_branch_lengths: If set, override the config's
            ``freeze_branch_lengths``. True disables all branch-length
            re-estimation for the run. None uses the config value.
        init_branch_lengths_from_input: If set, override the config's
            ``init_branch_lengths_from_input``. True skips cycle 0's
            branch-length cold start, so cycle 0 trains at the input's
            ``branch_length`` column, and re-optimizes after every cycle as
            usual. Mutually exclusive with both freezes.
        save_training_state: If set, override the config's
            ``save_training_state``. True makes checkpoints resumable
            (usable with ``--resume``) instead of weights-only. None uses
            the config value, EXCEPT when ``--resume`` is set: then None
            defaults to True so a resumed run keeps producing resumable
            checkpoints (a second interruption can still be resumed) —
            pass ``--save_training_state False`` explicitly to opt out.
        resume: Path to a resumable checkpoint (saved with
            ``save_training_state: true`` in the config, or
            ``--save_training_state`` on that earlier run). Reconstructs
            the Trainer via ``resume_training()``, restoring the optimizer,
            LR/ReduceLROnPlateau schedule, RNG state, and step counters, and
            continues from the checkpoint's saved cycle/task/step — unlike
            ``--checkpoint``, which loads weights only and resets the
            optimizer/LR schedule. Mutually exclusive with ``--checkpoint``,
            a non-zero ``--start_cycle``, and ``--loader_cycle_start``.
            Combines with ``--multi_gpu``, provided the world size matches
            the one the checkpoint was trained at. See
            ``train_model_from_datasets`` for details.
        recompute_branch_lengths_on_resume: If set, a branch-length cache
            miss when resuming mid-pipeline (via ``--resume`` or the debug
            ``--checkpoint``+``--start_cycle`` combo) falls back to
            model-based re-optimization instead of raising. Use only to
            extend a completed run past its originally planned final cycle.
            See ``train_model_from_datasets`` for details.

    Examples:
        # Single-GPU training (default)
        # dasm2 train dasm_1m '{"SelectionFactorHead": "tst"}'
        # dasm2 train dasm_1m '{"SelectionFactorHead": "tst"}' --device cuda:0

        # Multi-GPU training (uses all available GPUs)
        # dasm2 train dasm_1m '{"SelectionFactorHead": "tst"}' --multi_gpu

        # Multi-GPU with specific GPU count
        # dasm2 train dasm_1m '{"SelectionFactorHead": "tst"}' --multi_gpu --num_gpus 2

        # Multi-GPU on specific GPUs (shared machine)
        # dasm2 train dasm_1m '{"SelectionFactorHead": "tst"}' --gpus 1,4
    """
    # Set before any data/model creation so it also pins threading for the
    # data-loading path (in-process, not just the trainer).
    torch.set_num_threads(TRAINING_NUM_THREADS)
    if deterministic:
        set_deterministic_mode()

    # Deprecated --epochs: reject up front with a pointer at the new flags
    # so users don't see a cryptic "unrecognized argument" or a silently
    # ignored override.
    if epochs is not None:
        raise ValueError(
            "--epochs was removed when training became step-based. "
            "Pass --max_train_steps to control the per-cycle optimizer-step "
            "budget and --eval_every_steps to control the validation "
            "cadence. See `dasm2 train --help` for the full flag list."
        )

    # Fire parses "0,1,2,3" as a tuple; coerce back to comma-separated str.
    if isinstance(gpus, tuple):
        gpus = ",".join(str(g) for g in gpus)

    # Validate num_gpus requires multi_gpu
    if num_gpus is not None and not multi_gpu and gpus is None:
        raise ValueError("--num_gpus requires --multi_gpu")

    # CV-fold overrides require both index and num_folds together so a
    # half-applied override (e.g. one Snakemake rule passing only --cv-fold)
    # does not silently fall back to the YAML defaults.
    if (cv_fold is None) != (cv_num_folds is None):
        raise ValueError(
            "--cv-fold and --cv-num-folds must be passed together "
            "(or both omitted). Got "
            f"cv_fold={cv_fold!r}, cv_num_folds={cv_num_folds!r}."
        )
    if cv_seed is not None and cv_fold is None:
        raise ValueError(
            "--cv-seed without --cv-fold has no effect; pass --cv-fold "
            "and --cv-num-folds to override the YAML cv_split block."
        )
    if cv_group_col is not None and cv_fold is None:
        raise ValueError(
            "--cv-group-col without --cv-fold has no effect; pass --cv-fold "
            "and --cv-num-folds to override the YAML cv_split block."
        )
    if cv_binding_split_strategy is not None and cv_fold is None:
        raise ValueError(
            "--cv-binding-split-strategy without --cv-fold has no effect; pass "
            "--cv-fold and --cv-num-folds to override the YAML cv_split block."
        )

    # Re-exec under accelerate if needed (no-op when already launched or single-GPU)
    # --gpus implies --multi_gpu
    _maybe_relaunch_under_accelerate(multi_gpu or gpus is not None, num_gpus, gpus)

    # In multi-GPU mode, only rank 0 should print status messages.
    # Uses env-var check because the backend doesn't exist yet at this point;
    # train_model_from_datasets creates it after data loading.
    _print = _rank_guarded_print(_is_main_process())

    # Load base config from YAML (for neutral model settings)
    training_config = get_config(config_name)

    # Apply CLI overrides to config
    training_config = _apply_cli_overrides(
        training_config,
        neutral_model_name=neutral_model_name,
        multihit_model_name=multihit_model_name,
        dataset_mode=dataset_mode,
        cv_fold=cv_fold,
        cv_num_folds=cv_num_folds,
        cv_seed=cv_seed,
        cv_group_col=cv_group_col,
        cv_binding_split_strategy=cv_binding_split_strategy,
    )

    if training_config.cv_split is not None:
        cv = training_config.cv_split
        _print(
            f"CV split active: fold {cv.fold_index} of {cv.num_folds} "
            f"(group_col={cv.group_col!r}, seed={cv.seed}, "
            f"binding_split_strategy={cv.binding_split_strategy!r}, "
            f"val_num_folds={cv.val_num_folds!r}). "
            "Overrides any baked in_train column / registry holdouts."
        )
        # Route the helper's CV-split log lines (e.g. held-out group ids) to
        # _print so they're visible at command level. Library callers don't
        # hit this branch and keep standard logging silence-by-default.
        _attach_info_print_handler(_CV_SPLIT_LOGGER_NAME, _print)
        # cv_split is currently only threaded into the evolutionary,
        # standard-whichmut, MLM, binding-classification, and
        # contrastive-binding head loaders.
        # The other head families (MPNN_WHICHMUT, DMS, STRUCTURAL,
        # DELTA_AFFINITY, AAH) call internal loaders
        # that do not accept ``cv_split``, so silently passing them
        # ``cv_split`` from the YAML config would have no effect. Fail
        # loudly instead.
        # The dispatch is loss-type-keyed; resolve each head's effective
        # ``loss_type`` from the ``tasks: [...]`` block. A head with two
        # losses expands to two task entries sharing one head_type, so
        # iterate tasks rather than heads.
        cv_supported_loss_types = (
            EVOLUTIONARY_LOSS_TYPES
            | WHICHMUT_LOSS_TYPES
            | {LossType.MLM}
            | BINDING_CLASSIFICATION_LOSS_TYPES
            | {LossType.CONTRASTIVE_BINDING}
        )
        cv_unsupported_set: set[str] = set()
        for tc in training_config.tasks:
            ht = tc.head_type
            # AAH takes a dedicated dispatch branch that doesn't accept
            # cv_split, even though its loss_type would otherwise look
            # "supported". (MPNN_WHICHMUT heads route through their
            # ``whichmut_residual`` loss, which isn't in the supported
            # set, so the loss-type check below excludes them naturally.)
            if ht == HeadType.AAH_WHICHMUT_HEAD:
                cv_unsupported_set.add(ht.value)
                continue
            if tc.loss_config.loss_type not in cv_supported_loss_types:
                cv_unsupported_set.add(ht.value)
        cv_unsupported = sorted(cv_unsupported_set)
        if cv_unsupported:
            supported_losses = sorted(t.value for t in cv_supported_loss_types)
            raise NotImplementedError(
                f"cv_split is not yet supported for head types: "
                f"{cv_unsupported}. cv_split currently routes through "
                f"loss types: {supported_losses} (the union of "
                f"EVOLUTIONARY_LOSS_TYPES, WHICHMUT_LOSS_TYPES, "
                f"{{LossType.MLM}}, BINDING_CLASSIFICATION_LOSS_TYPES, and "
                f"{{LossType.CONTRASTIVE_BINDING}})."
            )
        # val_num_folds makes fold_index a test fold that only the contrastive
        # loader honors; every other loader would treat it as a validation
        # fold, so a mixed (joint) config would silently split differently
        # per head.
        contrastive_task_present = any(
            tc.head_type in CONTRASTIVE_BINDING_HEAD_TYPES
            for tc in training_config.tasks
        )
        if cv.val_num_folds is not None and any(
            tc.head_type not in CONTRASTIVE_BINDING_HEAD_TYPES
            for tc in training_config.tasks
        ):
            raise ValueError(
                "cv_split.val_num_folds (held-out test fold) is supported only "
                "when every task is a ContrastiveBindingHead task; got tasks "
                f"{[tc.head_type.value for tc in training_config.tasks]}."
            )
        # MLM only supports group-aware CV — check here too (reusing the
        # same validator dasm2.data.mlm.create_mlm_dataset/load_mlm_train_val
        # call) so the error surfaces before any data loads, not partway
        # through Phase 4.

    # Validate head types before loading any data (fail fast)
    _reject_untrainable_heads(training_config)
    config_heads = {h.head_type.value for h in training_config.model.heads}
    provided_heads = set(data.keys())
    _validate_head_datasets_match_config(config_name, config_heads, provided_heads)

    # ``TrainingConfig.neutral_model`` is typed to admit nothing but None in
    # this fork, so there is no model to build: the data carries precomputed
    # neutral outputs from the mat-pcp censor tables (#4).
    neutral_model: None = None
    _print("No neutral model — data carries precomputed neutral outputs")

    codon_pref_log: Optional[torch.Tensor] = None

    # Parse head data config (validates types and formats)
    parsed_data = _parse_head_data_config(data)

    # Dataset pipeline mode: "precomputed" (default) or "streaming"
    dataset_mode = training_config.dataset_mode
    if dataset_mode == "streaming":
        _print("Dataset mode: streaming (transforms run on-the-fly per __getitem__)")

    # Map head_type -> declared LossConfigs from the v4 ``tasks: [...]``
    # block. AAH yields two LossConfigs for the same head_type so the
    # dispatch below recognises both whichmut and aao_whichmut variants.
    task_loss_configs_by_head: Dict[HeadType, List[LossConfig]] = {}
    for tc in training_config.tasks:
        task_loss_configs_by_head.setdefault(tc.head_type, []).append(tc.loss_config)

    # Load per-head datasets using unified pipeline (returns DatasetBundle / BranchLengthDatasetBundle)
    head_datasets: Dict[str, Tuple[Any, Any]] = {}
    _attach_info_print_handler(_PARQUET_BACKEND_LOGGER_NAME, _print)
    # CLI overrides reach training_config only in train_model_from_datasets,
    # after data loading, so honor the --num_dataloader_workers flag here.
    cache_budget = CacheBudgetSpec(
        budget_gb=training_config.parquet_cache_budget_gb,
        num_dataloader_workers=(
            num_dataloader_workers
            if num_dataloader_workers is not None
            else training_config.num_dataloader_workers
        ),
    )
    for head_type, data_paths in parsed_data.items():
        head_type_str = head_type.value

        data_path = data_paths[0]
        _print(f"Loading data for {head_type_str}: {data_path}")

        train_item: Any
        val_item: Any

        # Dataset dispatch is loss-keyed. The head-keyed arms -- MPNN
        # whichmut, DMS, structural, delta-affinity, contrastive and
        # binding classification, MLM, and the two-dataset AAH head --
        # went with the datasets they loaded (#4).
        head_loss_configs = task_loss_configs_by_head[head_type]
        head_loss_type = head_loss_configs[0].loss_type
        is_whichmut = head_loss_type in WHICHMUT_LOSS_TYPES
        if is_whichmut:
            train_item, val_item = _load_whichmut_head_data(
                data_path,
                head_loss_type,
                dataset_mode,
                cv_split=training_config.cv_split,
                censor_table=_resolve_censor_table(training_config, _print),
                censor_mode=training_config.censor_mode,
                reference_base_idxs=_resolve_reference_bases(training_config, _print),
                cache_budget=cache_budget,
            )
        else:
            raise ValueError(
                f"No data loader for head type: {head_type.value} "
                f"(loss_type={head_loss_type.value}). "
                f"Supported evolutionary loss types: "
                f"{[t.value for t in EVOLUTIONARY_LOSS_TYPES]}; "
                f"whichmut loss types: {[t.value for t in WHICHMUT_LOSS_TYPES]}; "
                f"architecture-determined sets: "
                f"mpnn_whichmut={[h.value for h in MPNN_WHICHMUT_HEAD_TYPES]}, "
                f"dms={[h.value for h in DMS_HEAD_TYPES]}, "
                f"structural={[h.value for h in STRUCTURAL_HEAD_TYPES]}, "
                f"delta_affinity={[h.value for h in DELTA_AFFINITY_HEAD_TYPES]}, "
                f"contrastive_binding={[h.value for h in CONTRASTIVE_BINDING_HEAD_TYPES]}, "
                f"binding_classification={[h.value for h in BINDING_CLASSIFICATION_HEAD_TYPES]}, "
                f"mlm={[h.value for h in MLM_HEAD_TYPES]}, "
                f"aah=[{HeadType.AAH_WHICHMUT_HEAD.value}]"
            )

        head_datasets[head_type_str] = (train_item, val_item)

    # Delegate to core training function.
    train_model_from_datasets(
        config_name=config_name,
        head_datasets=head_datasets,
        neutral_model=neutral_model,
        output_dir=output_dir,
        max_train_steps=max_train_steps,
        eval_every_steps=eval_every_steps,
        max_eval_steps=max_eval_steps,
        cycles=cycles,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        device=device,
        notes=notes,
        seed=seed,
        checkpoint_every_n_evals=checkpoint_every_n_evals,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        no_log=no_log,
        checkpoint=checkpoint,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_dataloader_workers=num_dataloader_workers,
        multi_gpu=multi_gpu or gpus is not None,
        mixed_precision=mixed_precision,
        training_config=training_config,
        early_stopping_patience=early_stopping_patience,
        start_cycle=start_cycle,
        loader_cycle_start=loader_cycle_start,
        cache_dir=cache_dir,
        max_tokens_per_batch=max_tokens_per_batch,
        bl_optimization_num_workers=bl_optimization_num_workers,
        bl_optimization_max_chunk_size=bl_optimization_max_chunk_size,
        freeze_branch_lengths=freeze_branch_lengths,
        freeze_branch_lengths_after_first_cycle=freeze_branch_lengths_after_first_cycle,
        init_branch_lengths_from_input=init_branch_lengths_from_input,
        save_training_state=save_training_state,
        resume=resume,
        recompute_branch_lengths_on_resume=recompute_branch_lengths_on_resume,
        freeze_encoder=freeze_encoder,
        bl_optimization_parallel=bl_optimization_parallel,
    )




def _create_tasks_from_config(
    head_configs: Sequence[HeadConfig],
    head_datasets: Dict[str, Tuple[Any, Any]],
    output_path: Path,
    config_name: str,
    notes: Optional[str],
    task_configs: Sequence[TaskConfig],
    checkpoint_every_n_evals: Optional[int] = None,
    checkpoint_every_n_steps: Optional[int] = None,
    *,
    is_main_process: bool = True,
    save_training_state: bool = False,
) -> List[Task]:
    """Create training tasks from head configurations with per-head datasets.

    Each entry in ``task_configs`` (one per (head, loss) pair) becomes a
    ``Task`` paired with its declarative ``LossConfig``. AAH yields two
    ``TaskConfig`` entries sharing the same ``head_type`` pointer; their
    ``loss_type`` selects the matching ``:whichmut`` or ``:aao`` dataset
    key.

    DASM heads get branch length optimization hooks (other heads like
    Whichmut don't use branch lengths). The last task gets a checkpoint
    save hook.

    Values in ``head_datasets`` can be either raw ``Dataset`` objects
    (eager pipeline) or ``DatasetBundle``/``BranchLengthDatasetBundle``
    objects (unified pipeline). These are passed directly to
    ``Task(train_data=..., val_data=...)`` — the Task's computed
    properties handle extracting dataset, collator, and worker_init_fn
    from bundles.

    Args:
        head_configs: List of head configurations from TrainingConfig.
        head_datasets: Dict mapping dataset key -> (train, val) pair.
            Values can be ``Dataset``, ``DatasetBundle``, or
            ``BranchLengthDatasetBundle``.
        output_path: Directory for checkpoints.
        config_name: Config name for checkpoint naming.
        notes: Optional notes for checkpoints.
        task_configs: Declarative ``TaskConfig`` list from
            ``TrainingConfig.tasks``; one entry per (head, loss) pair.
        checkpoint_every_n_evals: If set, save checkpoints every N evaluations
            via ``CheckpointSaveHook.on_eval``.
        checkpoint_every_n_steps: If set, save every N optimizer steps via
            ``CheckpointSaveHook.on_step``.
        bl_optimization_num_workers: Worker count for the post-cycle
            ``BranchLengthOptimizer`` (None = auto). From
            ``TrainingConfig.bl_optimization_num_workers``.
        bl_optimization_max_chunk_size: Upper bound on sequences per BL
            optimization chunk (host-RAM cap). From
            ``TrainingConfig.bl_optimization_max_chunk_size``.
        freeze_branch_lengths: If True, the ``BranchLengthOptimizer`` hook
            never re-estimates branch lengths (pre_cycle and post_cycle both
            become no-ops). From ``TrainingConfig.freeze_branch_lengths``.
        freeze_branch_lengths_after_first_cycle: If True, branch lengths are
            set once by cycle-0 ``pre_cycle`` and then held fixed (post_cycle
            becomes a no-op; pre_cycle still runs). Cycle 0 runs the handler's
            ``bl_init_strategy`` where it has one, and otherwise falls through
            to the cache and model-based optimization. From
            ``TrainingConfig.freeze_branch_lengths_after_first_cycle``.
        init_branch_lengths_from_input: If True, cycle-0 ``pre_cycle`` keeps
            the input branch lengths and ``post_cycle`` re-optimizes as usual.
            From ``TrainingConfig.init_branch_lengths_from_input``.
        save_training_state: If True, ``CheckpointSaveHook`` saves full
            resumable checkpoints (optimizer, scheduler, RNG, step counters)
            instead of weights only. From
            ``TrainingConfig.save_training_state``.
        recompute_branch_lengths_on_resume: If True, the
            ``BranchLengthOptimizer``'s resume-entry-point cache miss falls
            back to model-based re-optimization instead of raising. See
            ``BranchLengthOptimizer.__init__`` for when this is appropriate.

    Returns:
        List of Task objects ready for training.
    """
    if not task_configs:
        raise ValueError(
            "TrainingConfig.tasks is empty. Pre-v4 YAMLs derived a default "
            "loss per head implicitly; v4 requires an explicit ``tasks: [...]`` "
            "block. Run "
            "``pixi run python scripts/migrate_yaml_to_v4.py --apply`` to migrate."
        )

    tasks: List[Task] = []

    checkpoint_hook = CheckpointSaveHook(
        output_dir=str(output_path),
        notes=notes,
        checkpoint_every_n_evals=checkpoint_every_n_evals,
        save_training_state=save_training_state,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
    )

    # Map head_type -> head_config. The v4 schema requires deduplicated
    # head entries, so a duplicate is a config-time bug.
    head_by_type: Dict[str, "HeadConfig"] = {}
    for hc in head_configs:
        ht = hc.head_type.value
        if ht in head_by_type:
            raise ValueError(
                f"Duplicate head_type {ht!r} in model.heads — v4 schema "
                f"requires deduplicated head entries."
            )
        head_by_type[ht] = hc

    task_head_types = {tc.head_type.value for tc in task_configs}
    missing = set(head_by_type) - task_head_types
    if missing:
        raise ValueError(
            f"Heads {sorted(missing)} have no matching entry in tasks. "
            f"Every head in model.heads must be paired with at least one "
            f"TaskConfig. Partially migrated YAMLs often land here — rerun "
            f"``pixi run python scripts/migrate_yaml_to_v4.py --apply`` to "
            f"regenerate the ``tasks: [...]`` block from the (deduplicated) "
            f"``model.heads`` list."
        )

    # Collect dataset keys per head_type (AAH exposes two: ``...:whichmut``
    # and ``...:aao``).
    head_type_to_dataset_keys: Dict[str, List[str]] = {}
    for dataset_key in head_datasets.keys():
        base_head = dataset_key.split(":", 1)[0] if ":" in dataset_key else dataset_key
        head_type_to_dataset_keys.setdefault(base_head, []).append(dataset_key)

    # Build the (head_config, dataset_key, loss_config, loss_weight) plan.
    # Each TaskConfig is matched to its head_config and to a not-yet-consumed
    # dataset key. AAH disambiguates by suffix: the ``:whichmut`` key pairs
    # with LossType.WHICHMUT and ``:aao`` with LossType.AAO_WHICHMUT.
    plan: List[
        Tuple[HeadConfig, str, LossConfig, float, Optional[int], List[HookConfig]]
    ] = []
    consumed: Dict[str, set] = {ht: set() for ht in head_by_type}
    for tc in task_configs:
        ht = tc.head_type.value
        if ht not in head_by_type:
            raise ValueError(
                f"task entry head_type {ht!r} has no matching entry in "
                f"model.heads (have {list(head_by_type)})."
            )
        hc = head_by_type[ht]
        available_keys = head_type_to_dataset_keys.get(ht, [ht])
        chosen_key: Optional[str] = None
        loss_type = tc.loss_config.loss_type
        # Currently the only colon-suffix keys come from AAH expansion
        # (``...:whichmut`` / ``...:aao``); pair each suffix to its loss type.
        # Module-level ``_AAH_SUFFIX_TO_LOSS_TYPE`` is the shared source of
        # truth between the producer (in ``train_model``) and this consumer.
        suffix_to_loss_type = _AAH_SUFFIX_TO_LOSS_TYPE
        for k in available_keys:
            if k in consumed[ht]:
                continue
            if ":" in k:
                suffix = k.split(":", 1)[1]
                expected_loss_type = suffix_to_loss_type.get(suffix)
                if expected_loss_type is None:
                    raise ValueError(
                        f"Unrecognized dataset-key suffix {suffix!r} on "
                        f"key {k!r}. Expected one of "
                        f"{sorted(suffix_to_loss_type)}."
                    )
                if loss_type == expected_loss_type:
                    chosen_key = k
                    break
            else:
                chosen_key = k
                break
        if chosen_key is None:
            raise ValueError(
                f"No dataset key available for task entry "
                f"head_type={ht!r}, loss_type={loss_type.value!r}. "
                f"Available: {available_keys}, consumed: {consumed[ht]}."
            )
        consumed[ht].add(chosen_key)
        plan.append(
            (
                hc,
                chosen_key,
                tc.loss_config,
                tc.loss_weight,
                tc.batch_size,
                tc.eval_hooks,
            )
        )

    total_tasks = len(plan)

    for task_idx, (
        head_config,
        dataset_key,
        loss_config,
        loss_weight,
        task_batch_size,
        task_eval_hook_configs,
    ) in enumerate(plan):
        head_type_str = head_config.head_type.value
        is_last_task = task_idx == total_tasks - 1
        train_item, val_item = head_datasets[dataset_key]

        # No pre-cycle hook survives the fork. Branch-length optimization
        # attached on ``loss_config.loss_type in EVOLUTIONARY_LOSS_TYPES``,
        # which is ``{DASM, QDASM, HYBRID}`` -- whichmut is not in it, so the
        # hook was built and never registered. It went with those losses (#4).
        pre_hooks: List[TrainingHook] = []
        post_hooks: List[TrainingHook] = []
        if is_last_task:
            post_hooks.append(checkpoint_hook)

        # Per-evaluation hooks: checkpointing on the last task, plus whatever
        # this task's own TaskConfig.eval_hooks configures (e.g.
        # OverfitGuardHook) — a per-task behavior, so it rides in as a hook
        # rather than a trainer-side conditional (see dasm2/training/hooks/
        # overfit_guard.py).
        eval_hooks: List[TrainingHook] = [
            create_hook(hook_config) for hook_config in task_eval_hook_configs
        ]
        intra_cycle_saves = (
            checkpoint_every_n_evals is not None or checkpoint_every_n_steps is not None
        )
        if intra_cycle_saves and is_last_task:
            eval_hooks.append(checkpoint_hook)

        task = Task(
            train_data=train_item,
            val_data=val_item,
            head_config=head_config,
            loss=create_loss(loss_config),
            loss_config=loss_config,
            loss_weight=loss_weight,
            batch_size=task_batch_size,
            pre_cycle_hooks=pre_hooks,
            post_cycle_hooks=post_hooks,
            eval_hooks=eval_hooks,
        )
        tasks.append(task)

        # Display suffix if multi-dataset head
        suffix = ""
        if ":" in dataset_key:
            suffix = f" ({dataset_key.split(':')[1]})"
        if is_main_process:
            print(f"  Task {task_idx + 1}: {head_type_str}{suffix}")
            if checkpoint_every_n_evals is not None and is_last_task:
                print(
                    f"    → Eval checkpointing every {checkpoint_every_n_evals} evals"
                )
            if checkpoint_every_n_steps is not None and is_last_task:
                print(
                    f"    → Step checkpointing every {checkpoint_every_n_steps} steps"
                )
            if task_eval_hook_configs:
                names = ", ".join(hc.type for hc in task_eval_hook_configs)
                print(f"    → Eval hooks: {names}")

    return tasks
