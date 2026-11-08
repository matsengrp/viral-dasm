"""Training state capture for checkpoint resume.

Captures the mutable training state an interrupted run needs to resume at
the optimizer step where it was saved, including partway through a data pass.
See :class:`TrainingState` for when the resumed weights match an
uninterrupted run's exactly.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Legacy field renames (pre-step-based-training schema).
#
# Callers construct TrainingState exclusively via ``TrainingState.from_dict``,
# so every serialized dict is normalized through ``_migrate_training_state_dict``
# before reaching the dataclass constructor. This is the *only* compat code
# we keep (Constitution VIII) — it is deliberate, scoped to in-flight resumable
# runs saved before the epoch→step migration.
_LEGACY_TRAINING_STATE_KEYS = {
    "global_batch": "global_step",
    "epochs_without_improvement": "evals_without_improvement",
    "global_epoch": "data_passes",  # int counter -> float counter
}


def _migrate_training_state_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a serialized TrainingState dict to the current schema.

    Renames legacy keys, provides defaults for new fields, and promotes
    ``data_passes`` to float. Old saves were always at cycle/task
    boundaries, so ``cycle_step``, ``cycle_task_index``, and ``eval_count``
    default to 0; ``loader_cycle`` is seeded from the (floored) data-pass
    counter so DDP shuffle determinism picks up where the old run left off.
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        out[_LEGACY_TRAINING_STATE_KEYS.get(k, k)] = v
    out.setdefault("cycle_step", 0)
    out.setdefault("cycle_task_index", 0)
    out.setdefault("eval_count", 0)
    out.setdefault("data_passes", 0.0)
    out.setdefault("loader_cycle", int(out["data_passes"]))
    # Absent in every checkpoint saved before head_optimizer existed, and
    # in any run that never configured one.
    out.setdefault("head_optimizer_state_dict", None)
    out.setdefault("head_learning_rate", None)
    # Absent in every checkpoint saved before head_scheduler existed, and
    # in any run whose head_optimizer (if any) pairs with a
    # non-reduce_on_plateau main scheduler (see Trainer._create_head_scheduler).
    out.setdefault("head_scheduler_state_dict", None)
    # Absent in every checkpoint saved before this field existed.
    out.setdefault("head_optimizer_head_type", None)
    # Older saves built every token-budget sampler with seed 0.
    out.setdefault("data_seed", 0)
    # Absent in every checkpoint saved before mid-pass resume existed;
    # Trainer._build_train_iters then restarts the saved pass with a warning.
    out.setdefault("train_iter_draws", None)
    out.setdefault("pass_start_rng_states", None)
    out.setdefault("rank_pass_positions", None)
    if isinstance(out["data_passes"], int):
        out["data_passes"] = float(out["data_passes"])
    return out


def as_cpu_byte_tensor(state: torch.Tensor) -> torch.Tensor:
    """Coerce an RNG state tensor to CPU, regardless of its checkpoint-load device.

    ``torch.random.get_rng_state()``/``torch.cuda.get_rng_state_all()``
    always produce ``torch.uint8`` tensors; ``map_location`` on checkpoint
    load only ever changes device, never dtype, so the dtype assertion below
    is a belt-and-suspenders check on that invariant rather than a case this
    function is expected to actually handle.
    """
    assert state.dtype == torch.uint8, (
        f"RNG state tensor has unexpected dtype {state.dtype} (expected "
        "torch.uint8); capture_rng_states() should only ever produce "
        "ByteTensors."
    )
    return state.cpu() if state.device.type != "cpu" else state


@dataclass
class TrainingState:
    """Complete mutable training state for checkpoint resume.

    Captures everything needed to resume training at the optimizer step where
    it was saved, including mid-cycle, mid-task and mid-pass.

    The resumed run's weights match an uninterrupted run's when each train
    loader's batch order is reproduced by its epoch and the pass-start RNG
    state (the global streams, any generator Accelerate attached to a
    shuffled loader, and whether each loader's persistent-worker iterator
    was reset rather than built). That holds for a single train loader, and
    for any number of loaders under token-budget batching, whose order is a
    function of ``data_seed + loader_cycle``. Under DDP each rank resumes at
    its own saved position (``rank_pass_positions``). It holds with or without
    DataLoader workers, provided the dataset and collate function draw no
    random numbers inside the workers.

    These cases are not reproduced exactly:

    - In joint training without token-budget batching, a loader's order can
      come from global-RNG draws made mid-pass (a shorter loader
      re-iterating, or a sampler's first draw landing after a forward pass),
      which the checkpoint does not hold.
    - DataLoader worker processes' own RNG streams are not checkpointed, so
      a collate function that draws inside workers (MLM masking, binding
      corruption) draws differently after a resume.
    - A DDP checkpoint saved before ``rank_pass_positions`` existed resumes
      every rank at rank 0's position and RNG streams (see
      :meth:`restore_rng_states`).

    Under DDP a batch order drawn from the CPU stream must also agree across
    ranks for the run to shard correctly at all, so ``Trainer`` raises at a
    pass start where it does not.

    Fields map to private Trainer attributes:
        global_step -> trainer._global_step  (optimizer steps across all cycles)
        current_cycle -> trainer._current_cycle
        cycle_step -> trainer._cycle_step  (step within current ``_train_for_steps``)
        cycle_task_index -> trainer._cycle_task_index  (task index in sequential mode)
        eval_count -> trainer._eval_count  (evals in current cycle)
        data_passes -> trainer._data_passes  (fractional dataset passes, diagnostic)
        loader_cycle -> trainer._loader_cycle  (authoritative DDP shuffle seed)
        data_seed -> trainer._data_seed  (token-budget batch-order base seed)
        best_val_loss -> trainer.best_val_loss
        evals_without_improvement -> trainer.evals_without_improvement
        best_model_state -> trainer.best_model_state
        optimizer_state_dict -> trainer.optimizer.state_dict()
        scheduler_state_dict -> trainer.scheduler.state_dict()
        rng_states -> torch/cuda/numpy/python RNG states
        head_optimizer_state_dict -> trainer.head_optimizer.state_dict() (None if unset)
        head_learning_rate -> trainer.head_learning_rate (None if unset)
        head_scheduler_state_dict -> trainer.head_scheduler.state_dict() (None if unset)
        head_optimizer_head_type -> trainer.head_optimizer_head_type.value (None if unset)
        learning_rate -> trainer.learning_rate (initial LR for the cycle-boundary reset)
        train_iter_draws -> trainer._train_iter_draws (None on older checkpoints)
        pass_start_rng_states -> trainer._pass_start_rng_states
        rank_pass_positions -> every rank's loader_cycle / draws / resets under DDP

    Construct exclusively via :meth:`from_dict` when deserializing so legacy
    schemas get migrated. The dataclass constructor is reserved for code that
    emits a TrainingState from live trainer attributes.
    """

    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Optional[Dict[str, Any]]
    global_step: int
    current_cycle: int
    cycle_step: int
    cycle_task_index: int
    eval_count: int
    data_passes: float
    loader_cycle: int
    data_seed: int
    best_val_loss: float
    evals_without_improvement: int
    best_model_state: Optional[Dict[str, Any]]
    rng_states: Dict[str, Any]
    learning_rate: float
    # State for an optional second optimizer scoped to one head's own
    # parameters. None on every run that doesn't configure one,
    # and on every checkpoint saved before this field existed (see
    # _migrate_training_state_dict above).
    head_optimizer_state_dict: Optional[Dict[str, Any]] = None
    head_learning_rate: Optional[float] = None
    # State for head_optimizer's own plateau scheduler. None
    # whenever head_optimizer is unset, or the main scheduler isn't
    # reduce_on_plateau (see Trainer._create_head_scheduler), or on any
    # checkpoint saved before this field existed.
    head_scheduler_state_dict: Optional[Dict[str, Any]] = None
    # Which head head_optimizer_state_dict belongs to, as HeadType.value. None whenever head_optimizer is unset,
    # and on any checkpoint saved before this field existed. Checked before
    # load_state_dict in Trainer.restore_training_state so restoring into a
    # trainer configured for a *different* head — whose parameters can
    # legitimately have the same shapes as the checkpoint's head — fails
    # loudly instead of silently applying the wrong optimizer state.
    head_optimizer_head_type: Optional[str] = None
    # The position inside the current data pass. One count per
    # train loader of the micro-batches its live iterator has yielded, plus
    # the RNG state just before those iterators were built (the global
    # streams, and under "loader_generators" each loader's Accelerate
    # synchronized generator state or None), so resume
    # can rebuild the same iterators and draw and discard the same batches.
    # Counted per micro-batch rather than per optimizer step, so gradient
    # accumulation is covered. None on any checkpoint saved before these
    # fields existed (see _migrate_training_state_dict above).
    train_iter_draws: Optional[List[int]] = None
    pass_start_rng_states: Optional[Dict[str, Any]] = None
    # Under DDP, every rank's own position, in rank order, as
    # {"loader_cycle", "train_iter_draws", "reset_iterators", "rng_states"}.
    # Ranks' passes and RNG streams drift apart (a token-budget pass can deal
    # them unequal batch counts), so rank 0's are not theirs. None on a
    # single process and on checkpoints saved before this field existed.
    rank_pass_positions: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingState":
        """Construct from a serialized dict, applying legacy migrations.

        This is the only supported construction path from serialized data.
        All loaders (``load_resumable_checkpoint``, legacy-fixture tests)
        must go through this classmethod so migration is structurally
        unavoidable.
        """
        return cls(**_migrate_training_state_dict(d))

    @staticmethod
    def capture_rng_states() -> Dict[str, Any]:
        """Capture current RNG states for torch, CUDA, numpy, and Python.

        Captured on whichever process calls it. The ``rng_states`` field is
        rank 0's; under DDP every rank's own is also saved (see
        :meth:`restore_rng_states`).

        The ``"cuda"`` entry is a *list* — one state per visible device on
        the saving process — because that is what
        ``torch.cuda.get_rng_state_all`` returns. Its length is therefore a
        property of the saving host's ``CUDA_VISIBLE_DEVICES``, not of the
        world size, and :meth:`restore_rng_states` checks it against the
        restoring host's device count.

        Returns:
            Dictionary with RNG states keyed by source name.
        """
        states: Dict[str, Any] = {
            "torch": torch.random.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            states["cuda"] = torch.cuda.get_rng_state_all()
        return states

    @staticmethod
    def restore_rng_states(rng_states: Dict[str, Any]) -> None:
        """Restore RNG states for torch, CUDA, numpy, and Python.

        RNG state tensors (torch's and CUDA's alike) must always be CPU
        ``torch.ByteTensor``s -- that's true regardless of which device the
        model/optimizer are restored to. But callers load
        ``training_state.pth`` with ``map_location=<model device>`` (e.g.
        ``resume_training(..., backend=SingleGPUBackend("cuda"))``), which
        moves every
        tensor in the pickle -- RNG state included -- onto that device.
        ``torch.random.set_rng_state``/``torch.cuda.set_rng_state_all`` then
        raise ``TypeError: RNG state must be a torch.ByteTensor`` on a CUDA
        tensor. Move back to CPU defensively rather than pushing a
        CPU-only ``map_location`` requirement onto every caller.

        **Under DDP each rank restores its own streams.** Ranks drift apart
        mid-run (per-rank batch shapes differ, dropout consumes the stream at
        different rates, and under token-budget batching ranks start passes
        at different steps), so ``Trainer.save_resumable`` gathers every
        rank's streams into ``rank_pass_positions`` and
        ``Trainer.restore_training_state`` hands each rank its own. A
        checkpoint saved before that holds only rank 0's streams
        (``rng_states``), and every rank restores those, as ``set_seed(seed)``
        does at launch; a rank whose stream had drifted then draws different
        dropout masks than it would have uninterrupted.

        Args:
            rng_states: Dictionary from capture_rng_states().

        Raises:
            ValueError: If the saved CUDA state list has a different number
                of entries than this process has visible devices.
        """
        torch.random.set_rng_state(as_cpu_byte_tensor(rng_states["torch"]))
        np.random.set_state(rng_states["numpy"])
        random.setstate(rng_states["python"])
        if "cuda" in rng_states and torch.cuda.is_available():
            # ``set_rng_state_all`` enumerates the list and writes device ``i``
            # per entry, so a longer list than this host has devices addresses
            # a nonexistent ordinal and dies with an opaque torch error.
            # Name both counts instead.
            saved = list(rng_states["cuda"])
            visible = torch.cuda.device_count()
            if len(saved) != visible:
                raise ValueError(
                    f"Checkpoint holds CUDA RNG state for {len(saved)} device(s) "
                    f"but this process sees {visible}. The saved list has one "
                    f"entry per visible device on the saving host, so restoring "
                    f"it here would address a device that does not exist. Resume "
                    f"with {len(saved)} visible device(s) (e.g. via "
                    f"CUDA_VISIBLE_DEVICES), or resume on CPU."
                )
            torch.cuda.set_rng_state_all([as_cpu_byte_tensor(s) for s in saved])
