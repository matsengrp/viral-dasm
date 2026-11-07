"""ZIP-based model serialization implementation."""

import dataclasses
import importlib.util
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Final, List, Optional, Tuple

import torch

from dasm2.data.neutral_rates_source import NeutralRatesSource
from dasm2.metadata import (
    CURRENT_SERIALIZATION_VERSION,
    BaseTrainingMetadata,
    SourceLaunchInfo,
)
from dasm2.models.factory import needs_sf_projection
from dasm2.serialization.training_state import TrainingState
from dasm2.serialization.zip_io import (
    model_archive_exists,
    read_model_archive,
    read_model_archive_with_extras,
    read_zip_metadata,
    write_model_archive,
)

if TYPE_CHECKING:
    from dasm2.config import TrainingConfig
    from dasm2.models.core.base import CoreModel
    from dasm2.training.compute_backend import ComputeBackend
    from dasm2.training.task import Task
    from dasm2.training.trainer import Trainer

# Patch script for checkpoints with an ambiguous sf_projection (see
# _assert_unambiguous_sf_projection below). Also referenced by
# scripts/detect_wiggle_regression.py's mirror check (#969) — single source
# of truth so a future rename only needs one edit.
WIGGLE_PATCH_SCRIPT_PATH: Final = (
    "experiments/compara-training/patch_checkpoint_wiggle.py"
)


@dataclasses.dataclass(frozen=True)
class ResumeResult:
    """Return value of :func:`resume_training`.

    Callers use field access so adding fields in the future does not break
    tuple-unpacking call sites. ``start_cycle_task_index`` is always 0 for
    joint-mode saves and for cycle-boundary saves; it is meaningful only when
    a sequential single-task run was interrupted mid-task.
    """

    trainer: "Trainer"
    start_cycle: int
    start_cycle_task_index: int
    start_cycle_step: int


def save_model(
    model: torch.nn.Module,
    metadata: BaseTrainingMetadata,
    filepath: Path,
) -> None:
    """Save model and metadata to ZIP archive.

    Args:
        model: PyTorch model to save
        metadata: Complete training metadata (Pydantic model)
        filepath: Output file path
    """
    write_model_archive(
        path=filepath,
        metadata=metadata,
        data=model.state_dict(),
        data_filename="model.pth",
    )


def load_model_metadata(filepath: Path) -> Dict[str, Any]:
    """Load only metadata without model weights.

    Args:
        filepath: Path to model checkpoint

    Returns:
        Metadata dictionary
    """
    return read_zip_metadata(filepath)


def load_model_components(
    filepath: Path, device: str = "cpu"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load metadata and model state dict components.

    Args:
        filepath: Path to model checkpoint
        device: Device to load tensors on

    Returns:
        Tuple of (metadata_dict, model_state_dict)
    """
    return read_model_archive(
        path=filepath,
        data_filename="model.pth",
        device=device,
        weights_only=True,
    )


def model_exists(filepath: Path) -> bool:
    """Check if valid model ZIP exists.

    Args:
        filepath: Path to check

    Returns:
        True if valid model checkpoint exists
    """
    return model_archive_exists(filepath, data_filename="model.pth")


def save_resumable_checkpoint(
    model: torch.nn.Module,
    metadata: BaseTrainingMetadata,
    training_state: TrainingState,
    filepath: Path,
) -> None:
    """Save model, metadata, and training state to ZIP archive for resume.

    Creates a ZIP archive with three entries:
    - metadata.yml: Human-readable YAML serialization of metadata
    - model.pth: Torch-serialized model state_dict
    - training_state.pth: Torch-serialized TrainingState for resume

    Args:
        model: PyTorch model to save
        metadata: Complete training metadata (Pydantic model)
        training_state: Mutable training state for resume
        filepath: Output file path
    """
    write_model_archive(
        path=filepath,
        metadata=metadata,
        data=model.state_dict(),
        data_filename="model.pth",
        extra_data={"training_state.pth": dataclasses.asdict(training_state)},
    )


def load_resumable_checkpoint(
    filepath: Path, device: str = "cpu"
) -> Tuple[Dict[str, Any], Dict[str, Any], TrainingState]:
    """Load metadata, model state dict, and training state from resumable checkpoint.

    Args:
        filepath: Path to resumable checkpoint .zip file
        device: Device to load tensors on

    Returns:
        Tuple of (metadata_dict, model_state_dict, training_state)

    Raises:
        ValueError: If checkpoint does not contain training_state.pth
    """
    # weights_only=False for training_state.pth is required because
    # TrainingState contains numpy arrays (RNG state) and Python tuples
    # that weights_only=True rejects. Safe because we only load our own
    # checkpoints.
    metadata, model_state_dict, extras = read_model_archive_with_extras(
        path=filepath,
        data_filename="model.pth",
        device=device,
        weights_only=True,
        extra_data_filenames={"training_state.pth": False},
    )

    training_state = TrainingState.from_dict(extras["training_state.pth"])

    return metadata, model_state_dict, training_state


def has_training_state(filepath: Path) -> bool:
    """Check if a checkpoint contains resumable training state.

    Args:
        filepath: Path to checkpoint .zip file

    Returns:
        True if the checkpoint contains training_state.pth
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return False
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            return "training_state.pth" in zf.namelist()
    except zipfile.BadZipFile:
        raise zipfile.BadZipFile(
            f"Checkpoint file is corrupt: {filepath}. "
            "This may indicate an interrupted save (OOM, preemption, SIGTERM)."
        )
    except OSError:
        raise OSError(f"Checkpoint file is unreadable: {filepath}")


# ---------------------------------------------------------------------------
# Legacy migration helpers (used by ``scripts/migrate_checkpoint_to_v5.py``
# and by tests in ``tests/serialization/`` and
# ``tests/training/test_legacy_checkpoint_migration.py``). The production
# loaders (``load_checkpoint``, ``resume_training``, ``load_model_only``)
# DO NOT call these — pre-v5 checkpoints are rejected at all three entry
# points with a pointer to the migration script.
# ---------------------------------------------------------------------------


def migrate_legacy_state_dict(
    state_dict: Dict[str, Any],
    model: torch.nn.Module,
    serialization_version: int,
) -> Dict[str, Any]:
    """Migrate legacy DASM2 checkpoint keys to current model layout.

    Two rename passes are applied per key when needed:
      - TransformerBlock rename (#379, #444): self_attn → self_attention,
        linear1/2 → feed_forward.linear1/2, norm1/2 → post_attn_norm/post_ffn_norm.
      - Shared SF projection rename (#513):
        prediction_heads.{Head}.prediction_layer.* → sf_projection.projection.*.

    .. deprecated::
        Post-#621 Task #7 the production loader rejects pre-v5 checkpoints
        outright. This helper survives only to support
        ``scripts/migrate_checkpoint_to_v5.py``: re-applying these renames as
        part of an offline migration so the resulting v5 checkpoint can be
        loaded normally.

    Compares checkpoint keys against the model's expected keys to decide
    whether migration is needed.

    Raises:
        ValueError: If migration produces duplicate keys or keys not expected
            by the model.
    """
    if serialization_version >= CURRENT_SERIALIZATION_VERSION:
        return state_dict

    expected_keys = set(model.state_dict().keys())
    if set(state_dict.keys()) == expected_keys:
        return state_dict

    # Issue #199, Exp 3: distance_proj is now bias-free. Drop the legacy bias
    # tensor when loading a pre-Exp-3 checkpoint into a current model.
    if (
        "encoder.distance_proj.bias" in state_dict
        and "encoder.distance_proj.bias" not in expected_keys
    ):
        state_dict = {
            k: v for k, v in state_dict.items() if k != "encoder.distance_proj.bias"
        }

    # Every key must already name a parameter of the model being loaded into.
    #
    # `dasm2` renamed keys here for two pre-#513/#379 layouts: the
    # `TransformerEncoderLayer` -> `TransformerBlock` rename, and the three
    # shared-SF heads whose `prediction_layer` moved into `sf_projection`.
    # Both migrations ran only for a key that did not already match, and the
    # checkpoints this repository loads predate neither: all 64 keys of the
    # `dasm-85k` reference match as written. The rewrites were also not
    # idempotent -- applied to a current key, the block migration produces
    # `feed_forward.feed_forward.linear1` -- so carrying them would have
    # claimed support for formats this repository has never read (#4).
    for key in state_dict:
        if key not in expected_keys:
            raise ValueError(
                f"Checkpoint key '{key}' names no parameter of this model. "
                "This fork loads the checkpoint layout its own code writes; "
                "it carries no migrations for older `dasm2` layouts."
            )
    return dict(state_dict)


# Sentinel for "not yet set" — distinct from None, which is a valid domain
# value (wiggle_beta=null means explicitly wiggle-free).
_UNSET: Final = object()

# The wiggle_beta default that DASMEvolHead carried pre-#513. Recorded as a
# named constant because the M4 branch injects it explicitly to avoid a hidden
# dependency on SelectionFactorProjection's post-#513 default staying aligned.
_PRE_513_DASM_EVOL_HEAD_DEFAULT_WIGGLE_BETA: Final = 0.3

# Historical set of head config classes that existed pre-#513 with shared SF.
# Intentionally an independent frozen literal, not derived from
# `_SF_HEAD_PREFIXES` in `dasm2/models/key_migration.py` — that constant
# records the historical state_dict key prefixes, a different historical
# assertion that happens to refer to the same three heads today. Coupling the
# two would create an implicit contract that may not survive future refactors.
# Do not add new entries.
_SF_HEAD_TYPES = frozenset({"DASMEvolHead", "WhichmutHead", "AAOWhichmutHead"})

# Historical set of pre-#513 SF heads whose configs had **no** `wiggle_beta`
# field — these heads were always wiggle-free by construction. Kept as a
# literal frozenset rather than `_SF_HEAD_TYPES - {"DASMEvolHead"}` because
# the membership records a separate historical fact (which heads lacked the
# field), not a present-day set relationship.
#
# After #513, the shared `SelectionFactorProjection` defaults
# `wiggle_beta=0.3`, which would silently apply wiggle to these heads' weights
# at inference. The migration treats their pre-#513 intent as `wiggle_beta=null`.
_NO_WIGGLE_HEAD_TYPES = frozenset({"WhichmutHead", "AAOWhichmutHead"})

# Heads that carried a `max_aa_seq_len` field during their early context-free
# baseline phase but are now truly context-free (a single scalar / a 20x20
# table) and reject the field. The migration strips it so legacy
# compara-training checkpoints load. Recorded as a literal frozenset (not
# derived from any other constant) because it records a separate historical
# fact: which heads had this transient field.
_CONTEXT_FREE_HEADS_NO_LEN = frozenset({"SingleValueHead", "SingleMatrixHead"})


def _infer_sf_projection_from_heads(
    heads: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Infer the post-#513 sf_projection block from pre-#513 head configs.

    Returns the dict to use for ``model["sf_projection"]``, or ``None`` if
    the metadata has no SF-consuming head and no injection is needed.

    Raises:
        ValueError: If two SF heads carry conflicting ``wiggle_beta`` values.
    """
    explicit_wiggle_beta: Any = _UNSET
    has_no_wiggle_head = False
    has_dasm_evol_head = False

    for head in heads:
        head_type = head.get("head_type")
        if head_type in _NO_WIGGLE_HEAD_TYPES:
            has_no_wiggle_head = True
        if head_type == "DASMEvolHead":
            has_dasm_evol_head = True
        if head_type in _SF_HEAD_TYPES and "wiggle_beta" in head:
            head_value = head["wiggle_beta"]
            if (
                explicit_wiggle_beta is not _UNSET
                and head_value != explicit_wiggle_beta
            ):
                raise ValueError(
                    "Conflicting wiggle_beta values across SF heads in pre-#513 "
                    f"checkpoint metadata: {explicit_wiggle_beta!r} vs "
                    f"{head_value!r}. All SF heads were expected to share a "
                    "single value."
                )
            explicit_wiggle_beta = head_value

    if explicit_wiggle_beta is not _UNSET:
        return {"wiggle_beta": explicit_wiggle_beta}
    if has_no_wiggle_head:
        return {"wiggle_beta": None}
    if has_dasm_evol_head:
        # Pre-#513 DASMEvolHead with wiggle_beta omitted — the head's pre-#513
        # default was 0.3 (netam-compatible). Inject explicitly so the migration's
        # intent doesn't depend on _create_sf_projection's auto-create default
        # staying aligned with the historical head default.
        return {"wiggle_beta": _PRE_513_DASM_EVOL_HEAD_DEFAULT_WIGGLE_BETA}
    return None


# Head types that legitimately retain a `wiggle_beta` field on their head
# config post-#621 (QDASM/Hybrid head-side activation parameter). The strip
# below must not touch these, otherwise a pre-v5 checkpoint with an explicit
# `wiggle_beta=0.5` would have the field popped and Pydantic would refill
# the default `0.3` on load — silent data loss on resume.
_WIGGLE_BETA_HEAD_TYPES = frozenset({"QDASMEvolHead", "HybridEvolHead"})


def _migrate_metadata(metadata: Dict[str, Any]) -> None:
    """Apply in-place metadata migrations for old checkpoint formats.

    .. deprecated::
        Post-#621 Task #7 the production loader rejects pre-v5 checkpoints
        outright. This helper survives only to support
        ``scripts/migrate_checkpoint_to_v5.py`` and the legacy-migration
        tests under ``tests/serialization/`` and
        ``tests/training/test_legacy_checkpoint_migration.py``.

    Issue #513: `wiggle_beta` moved from head configs to `sf_projection`.
    Issue #586: pre-#513 checkpoints must reconstruct with the same wiggle
    setting they were trained with. After #699 the auto-create default is
    `wiggle_beta=None` (wiggle-free), so DASMEvolHead checkpoints need
    explicit injection of `wiggle_beta=0.3` to preserve their trained
    behavior — without it the auto-create path would silently drop wiggle.

    Pre-#513:
        - `DASMEvolHead` had a `wiggle_beta` field on its config (default 0.3,
          can be set to `null` for no-wiggle trainings).
        - `WhichmutHead` and `AAOWhichmutHead` had no `wiggle_beta` field;
          they were always wiggle-free by construction.

    Migration:
        - If any `DASMEvolHead` config has an explicit `wiggle_beta`, that
          value wins. (Multiple SF heads with conflicting values raise.)
        - Otherwise, if the metadata only carries `WhichmutHead` /
          `AAOWhichmutHead`, inject `wiggle_beta: null` to preserve their
          original wiggle-free behavior.
        - Strip `wiggle_beta` from SF-collapsed head configs so
          `TrainingConfig` deserialization doesn't reject the extra field.
          QDASM/Hybrid retain `wiggle_beta` as a head-side field post-#621.
        - Skip injection entirely if `sf_projection` is already present
          (post-#513 passthrough).

    Issue #578: epoch -> step training-loop migration. ``epochs`` was
    removed from ``TrainingConfig`` and replaced by ``max_train_steps`` /
    ``eval_every_steps``. Old checkpoints carry the integer epoch count;
    we lift it to ``max_train_steps`` (and pin ``eval_every_steps`` to
    the same value) so the new validators accept it.

    Issue #578 also removed the ``num_workers`` DataLoader knob from
    ``TrainingConfig``. Late-pre-#578 checkpoints carry it as an extra
    top-level field; we strip it unconditionally since it has no
    architectural implication and only affected DataLoader parallelism
    at training time.

    Issue #199: distance encoding moved from the head to the encoder. Old
    ``DASMEvolHead`` configs may carry ``distance_embed_dim``,
    ``num_distance_bins``, and ``use_distance``; these were never read by the
    head's forward, so the head loop drops them.
    """
    tc = metadata.get("training_config")
    if tc is None:
        return
    model = tc.get("model")
    if model is None:
        return
    heads = model.get("heads", [])

    inferred = _infer_sf_projection_from_heads(heads)

    for head in heads:
        if head.get("head_type") not in _WIGGLE_BETA_HEAD_TYPES:
            head.pop("wiggle_beta", None)
        # Issue #199: distance encoding moved from head to encoder. Drop the
        # legacy head-level distance fields the current head config rejects.
        head.pop("distance_embed_dim", None)
        head.pop("num_distance_bins", None)
        head.pop("use_distance", None)

    if "epochs" in tc:
        epochs = tc.pop("epochs")
        tc.setdefault("max_train_steps", int(epochs))
        tc.setdefault("eval_every_steps", int(epochs))

    # `num_workers` (DataLoader parallelism) was removed from TrainingConfig
    # in the same #578 step-based-training migration. Strip silently — no
    # architectural implication, training-time-only knob.
    tc.pop("num_workers", None)

    for head in heads:
        if head.get("head_type") in _CONTEXT_FREE_HEADS_NO_LEN:
            head.pop("max_aa_seq_len", None)

    if "sf_projection" in model:
        return

    if inferred is not None:
        model["sf_projection"] = inferred


def _deduplicate_heads(training_config: Any) -> None:
    """Remove duplicate head configs by head_type (in-place).

    .. deprecated::
        Post-#621 Task #7 the production loader rejects pre-v5 checkpoints.
        This helper is retained for the migration script and its tests.

    Some pre-v4 checkpoints (e.g., AAHWhichmutHead) were trained with
    multi-dataset configs that list the same head type multiple times even
    though the model only has one head instance. v4 metadata deduplicates
    head_configs at ``export_metadata`` time so this is no longer needed
    on the load path.
    """
    import warnings

    seen: set[str] = set()
    unique_heads = []
    duplicates: list[str] = []
    for head_config in training_config.model.heads:
        head_type_str = head_config.head_type.value
        if head_type_str not in seen:
            unique_heads.append(head_config)
            seen.add(head_type_str)
        else:
            duplicates.append(head_type_str)
    if duplicates:
        warnings.warn(
            f"Checkpoint metadata contained duplicate head types: {duplicates}. "
            "Keeping first occurrence of each.",
            stacklevel=2,
        )
    training_config.model.heads = unique_heads


def _attach_task_losses_for_inference(model: "CoreModel", training_config: Any) -> None:
    """Attach a ``task_losses`` ``ModuleDict`` to ``model`` for inference loading.

    The v5 state_dict carries ``task_losses.<key>.*`` parameters/buffers that
    must land on a real submodule before ``model.load_state_dict`` is called.
    For inference paths (``load_checkpoint``) we have no live ``Task`` objects;
    instead we read the v4 ``training_config.tasks: [...]`` block, build a
    fresh ``Loss`` instance per entry, and assign them under the keys
    that :func:`compute_task_loss_keys` would produce.

    The Loss instances are constructed via ``create_loss(task_config.loss_config)``
    so their parameter shapes / buffer registrations match what was saved
    (e.g. siglip's ``logit_bias`` is a Parameter, InfoNCE's is a buffer).
    """
    # Local imports: serialization -> training is a circular dependency.
    from dasm2.losses import create_loss
    from dasm2.training.task import compute_task_loss_keys

    # No tasks at all (e.g. checkpoint was saved from a Trainer with empty
    # tasks list) → nothing to attach. The state_dict won't have task_losses
    # entries either.
    task_configs = list(training_config.tasks or [])
    if not task_configs:
        return

    losses = [create_loss(tc.loss_config) for tc in task_configs]
    keys = compute_task_loss_keys(
        [tc.head_type for tc in task_configs],
        [tc.loss_config.loss_type for tc in task_configs],
    )
    module_dict = torch.nn.ModuleDict(dict(zip(keys, losses)))
    # ``model.task_losses`` is the same attribute the trainer attaches; the
    # inference Trainer wrapper around this model never references it again.
    model.task_losses = module_dict
    # Move loss params to the model's device, which a parameter-free model
    # also reports (``CoreModel.device``).
    module_dict.to(model.device)


def _assert_supported_serialization_version(
    metadata: Dict[str, Any], filepath: Path
) -> None:
    """Reject pre-v5 checkpoints with a helpful pointer to the migration script.

    Post-#621 (Task #7) the loader assumes the checkpoint metadata is in the
    v5 layout (per-task ``LossConfig``, deduplicated ``model.heads``,
    head_types using post-collapse names) and the state_dict uses the
    post-collapse key naming (``task_losses.<key>.*`` rather than the legacy
    ``prediction_heads.<head>.{scale,logit_bias}``). The on-disk
    ``serialization_version`` is stamped to 5 to mark this layout.

    Older checkpoints (v1-v4) need to run through
    ``scripts/migrate_checkpoint_to_v5.py`` before they can be loaded; this
    function emits a one-line error pointing the user at that script.
    """
    sv = metadata.get("serialization_version")
    if sv is None:
        raise ValueError(
            f"Checkpoint metadata missing 'serialization_version' field: "
            f"{filepath}. Run "
            f"`pixi run python scripts/migrate_checkpoint_to_v5.py --apply "
            f"--paths {filepath}` to migrate it to the current "
            f"(v{CURRENT_SERIALIZATION_VERSION}) layout."
        )
    if sv < CURRENT_SERIALIZATION_VERSION:
        raise ValueError(
            f"Checkpoint at {filepath} has serialization_version={sv}, "
            f"but the current loader requires v{CURRENT_SERIALIZATION_VERSION} "
            f"(post-#621 Task #7: per-task LossConfig schema and "
            f"task_losses state-dict layout). Run "
            f"`pixi run python scripts/migrate_checkpoint_to_v5.py --apply "
            f"--paths {filepath}` to migrate the checkpoint in place."
        )


def _assert_unambiguous_sf_projection(
    training_config: "TrainingConfig", filepath: Path, *, resumable: bool
) -> None:
    """Reject a checkpoint whose stored ``sf_projection`` is ambiguous.

    Since PR #895 (closing #877), ``export_metadata`` derives
    ``sf_projection`` from the live model instance — and
    ``CoreModel._validate_sf_projection`` forbids an SF-consuming head with
    a null projection — so any checkpoint written through the real
    ``Trainer.save_checkpoint``/``export_metadata`` path on post-#895 code
    always stores an explicit, non-null ``sf_projection`` whenever an
    SF-consuming head is present. Before #895, ``export_metadata`` never
    passed ``sf_projection`` at all, so it *unconditionally* stored
    ``null`` for every checkpoint saved by that code, regardless of the
    model's actual ``wiggle_beta``. A stored ``sf_projection: null``
    alongside an SF-consuming head therefore ordinarily means the
    checkpoint was saved by code from before #895 merged (2026-07-20) —
    which includes any active experiment branch that hadn't rebased past
    that date, not just checkpoints that are chronologically old. A stale
    or dirty clone that had rebased past #895 but not picked up the fix,
    or metadata copied verbatim from such a checkpoint (as
    ``experiments/2026-07-10-sim-anchor/scripts/null_control.py`` does when
    re-saving a mutated copy of a base checkpoint), are secondary,
    unconfirmed ways to reach the same shape. Worked example: the
    sim-anchor truth model was saved 2026-07-11 — nine days *before* #895
    merged — so it is an ordinary pre-#895 checkpoint; its unresolvable
    ``commit_sha`` and ``dirty`` status are incidental to running from an
    active experiment clone, not evidence of clone staleness relative to
    #895. In every case the true ``wiggle_beta`` is unknowable from the
    metadata alone (#896) — reconstructing it anyway would silently apply
    whatever wiggle default the *loading* commit happens to have, rather
    than what the checkpoint actually trained with. This mechanism is
    exactly what produced the incidents behind #896, #964's Phase 0, and
    the PR #934 review.

    Args:
        training_config: Parsed ``TrainingConfig`` for the checkpoint.
        filepath: Path to the checkpoint, for the error message.
        resumable: Whether this checkpoint carries a ``training_state.pth``
            member — ``patch_checkpoint_wiggle.py`` refuses to touch such
            checkpoints, so they have no metadata-patch remedy.

    Raises:
        ValueError: If ``training_config.model.sf_projection`` is ``None``
            and the model's heads consume shared selection factors.
    """
    model_config = training_config.model
    if model_config.sf_projection is not None:
        return
    if not needs_sf_projection(model_config.heads):
        return

    if resumable:
        remedy = (
            f"This checkpoint carries a training_state.pth member, which "
            f"{WIGGLE_PATCH_SCRIPT_PATH} explicitly refuses to touch (its "
            f"docstring: it does not attempt to preserve that member) — there "
            f"is no metadata-patch remedy available here. Retrain from "
            f"scratch, or resume from an earlier, unambiguous checkpoint in "
            f"the same run."
        )
    else:
        remedy = (
            f"Run `python {WIGGLE_PATCH_SCRIPT_PATH} {filepath} "
            f"--wiggle-beta <value>` to patch the metadata in place with the "
            f"known value (see #896), or retrain."
        )

    raise ValueError(
        f"Checkpoint at {filepath} has an SF-consuming head but no explicit "
        f"sf_projection in its stored training_config (sf_projection: null). "
        f"This checkpoint was ordinarily written by code from before #895 "
        f"merged (2026-07-20), which stored sf_projection: null "
        f"unconditionally; a stale/dirty clone that hadn't picked up #895, "
        f"or metadata copied verbatim from such a checkpoint, are secondary, "
        f"unconfirmed possibilities. In every case its true wiggle_beta is "
        f"unknowable from the metadata alone. Reconstructing it would "
        f"silently use whatever wiggle_beta default the current commit "
        f"happens to have, which may not match how this checkpoint was "
        f"trained. {remedy}"
    )


def load_checkpoint(
    filepath: Path,
    device: str = "cpu",
    cache_dir: Optional[Path] = None,
    enable_cache: bool = True,
    allow_dense_attention_fallback: bool = False,
    neutral_model_path: Optional[Path] = None,
) -> "Trainer":
    """Load checkpoint as Trainer ready for evaluation.

    Reconstructs full training context from checkpoint metadata:
    - CoreModel with trained weights
    - Neutral model (from TrainingConfig)
    - Training configuration
    - Cache configuration

    Args:
        filepath: Path to checkpoint .zip file
        device: Device to load model on ("cpu" or "cuda")
        cache_dir: Cache directory for branch lengths/predictions (default: .dasm2_cache)
        enable_cache: Enable caching for branch lengths and predictions
        allow_dense_attention_fallback: Build dense attention for a checkpoint
            trained with ``flash_attention: true`` when the flash-attn kernel
            cannot run on ``device``. Same rule as in :func:`load_model_only`.
            When it fires, re-saving the returned Trainer records
            ``flash_attention: false``: the saved encoder config is read off
            the model, which is dense.
        neutral_model_path: Build the neutral model from this archive instead
            of the ``model_path`` recorded in the checkpoint's
            ``training_config.neutral_model``. The recorded path is absolute
            under whichever host's artifact root wrote it, so it is often
            missing elsewhere (#1473). The caller vouches that this is the
            same neutral; nothing here can check it. Re-saving the returned
            Trainer records this path.

    Returns:
        Trainer ready for evaluation with:
        - model: Reconstructed CoreModel with loaded weights
        - neutral_model: Reconstructed neutral model
        - cache: Configured for predictions and branch lengths

    Raises:
        ValueError: If checkpoint file format is invalid or missing required
            metadata, or if ``neutral_model_path`` is given for a checkpoint
            whose neutral model records no ``model_path``
        FileNotFoundError: If checkpoint file doesn't exist

    Examples:
        >>> from pathlib import Path
        >>> from dasm2.analysis.oe_plots import OEPlotter
        >>> from dasm2.data.evolutionary_dataset import create_evolutionary_dataset
        >>> trainer = load_checkpoint(Path("checkpoints/8m_final.zip"))
        >>> # Trainer ready for OE plotting
        >>> bundle = create_evolutionary_dataset("v1jaffe", trainer.neutral_model)
        >>> plotter = OEPlotter(trainer, bundle)
        >>> predictions_df = plotter.oe_plot_df
    """
    from dasm2.config import TrainingConfig

    # Load components using generic function
    metadata, state_dict = load_model_components(filepath, device=device)
    _assert_supported_serialization_version(metadata, Path(filepath))

    # Extract TrainingConfig
    if "training_config" not in metadata:
        raise ValueError(
            f"Checkpoint metadata missing 'training_config' key. "
            f"Available keys: {list(metadata.keys())}. "
            f"Note: This checkpoint may be from an old version. "
            f"Checkpoints must be created with the new TrainingConfig format."
        )

    training_config = TrainingConfig(**metadata["training_config"])
    _assert_unambiguous_sf_projection(training_config, Path(filepath), resumable=False)

    if allow_dense_attention_fallback:
        _fall_back_to_dense_attention(training_config, device)

    if neutral_model_path is not None:
        _override_neutral_model_path(
            training_config, Path(neutral_model_path), Path(filepath)
        )

    # World size and backend precision mode the checkpoint was trained at,
    # both None for anything written before issue #1027 recorded them.
    # Carried through so re-saving this trainer doesn't overwrite them with
    # the fabricated SingleGPUBackend's 1/"no" — including the None case,
    # which must stay None rather than become a fact the source never stated.
    source_training_metadata = metadata.get("training_metadata", {})
    # Where training's rates came from (issue #1299); None for checkpoints
    # that predate the field. Evaluation checks it against the eval file, and
    # a re-save carries it through, since this Trainer has no datasets.
    recorded_rates_source = metadata.get("neutral_rates_source")
    trainer = _build_inference_trainer(
        training_config,
        device=device,
        state_dict=state_dict,
        cache_dir=cache_dir,
        enable_cache=enable_cache,
        config_name=metadata.get("config_name"),
        source_launch_info=SourceLaunchInfo(
            num_processes=source_training_metadata.get("num_processes"),
            mixed_precision=source_training_metadata.get("mixed_precision"),
        ),
        source_neutral_rates_source=(
            NeutralRatesSource(**recorded_rates_source)
            if recorded_rates_source is not None
            else None
        ),
    )

    # Set checkpoint_id so prediction cache entries are unique per checkpoint.
    # Without this, different checkpoints of the same architecture (e.g.,
    # cycle0 vs final) would silently share cached predictions.
    trainer.checkpoint_id = Path(filepath).stem

    return trainer


def _build_inference_trainer(
    training_config: "TrainingConfig",
    device: str,
    state_dict: Optional[Dict[str, Any]],
    cache_dir: Optional[Path],
    enable_cache: bool,
    config_name: Optional[str],
    source_launch_info: SourceLaunchInfo,
    source_neutral_rates_source: Optional[NeutralRatesSource],
) -> "Trainer":
    """Build an evaluation ``Trainer`` from a ``TrainingConfig``.

    Shared by :func:`load_checkpoint` (which passes the saved
    ``state_dict``) and :func:`save_parameter_free_checkpoint` (which has no
    weights to load, so passes ``None``).
    """
    # Local imports: serialization -> training is a circular dependency
    # (training.trainer imports serialization.checkpoints.save_model at module level).
    from dasm2.config import create_neutral_model_from_config
    from dasm2.models.factory import create_model_from_config
    from dasm2.training.compute_backend import SingleGPUBackend
    from dasm2.training.trainer import Trainer

    # Create model from configuration
    model = create_model_from_config(training_config.model, device=device)

    # Reconstruct ``model.task_losses`` from the v4 ``tasks: [...]`` block so
    # state_dict's ``task_losses.<key>.*`` entries land on real submodules at
    # ``load_state_dict`` time. The keys produced by ``Trainer._task_loss_key``
    # depend on the *list* of tasks (multi-task vs single-task), so we
    # materialise the full list here even though inference only needs the
    # parameters' values.
    _attach_task_losses_for_inference(model, training_config)

    # Load trained weights — the v5 contract is that state_dict keys match
    # the model's expected layout exactly. Pre-v5 checkpoints fail
    # ``load_checkpoint``'s `_assert_supported_serialization_version` gate
    # with a pointer to the migration script.
    if state_dict is not None:
        model.load_state_dict(state_dict)

    # Create neutral model from config (if present)
    neutral_model = (
        create_neutral_model_from_config(training_config.neutral_model)
        if training_config.neutral_model is not None
        else None
    )

    # Initialize Trainer with all components (empty tasks for evaluation only)
    cache_dir_str = str(cache_dir) if cache_dir is not None else ".dasm2_cache"
    # ``tasks=[]``: evaluation needs no live Tasks, and building them is not
    # even possible here — a ``Task`` bundles the dataset it trains on
    # (DESIGN D5) and a checkpoint carries no datasets. The parsed
    # ``TaskConfig`` list therefore rides along in ``source_task_configs``
    # so a re-save doesn't drop the ``tasks: [...]`` block (issue #974); see
    # ``Trainer.__init__`` for why that fallback is needed.
    # ``source_launch_info`` rides along for the same reason: the backend
    # below is fabricated, so its world size of 1 and precision mode of "no"
    # would overwrite the recorded launch facts on any re-save (issue #1027).
    return Trainer(
        model=model,
        neutral_model=neutral_model,
        tasks=[],  # No tasks needed for evaluation
        batch_size=training_config.batch_size,
        val_batch_size=training_config.val_batch_size,
        backend=SingleGPUBackend(device),
        cache_dir=cache_dir_str if enable_cache else None,
        config_name=config_name,
        source_task_configs=list(training_config.tasks or []),
        source_launch_info=source_launch_info,
        source_neutral_rates_source=source_neutral_rates_source,
    )


def save_parameter_free_checkpoint(
    training_config: "TrainingConfig",
    filepath: Path,
    config_name: Optional[str] = None,
    notes: Optional[str] = None,
) -> Path:
    """Write a model with no learnable parameters as a loadable checkpoint.

    A model with nothing to train, such as ``IdentityEncoder`` +
    ``NeutralHead`` (the f ≡ 1 model), is fully determined by its config. This
    writes it as an ordinary checkpoint, so :func:`load_checkpoint` and
    every command built on it (``oe-plot``, ``likelihood``, branch-length
    refitting) score it like a trained model under the kernel its
    ``tasks: [...]`` block configures.

    Args:
        training_config: The model, tasks and neutral model to write.
        filepath: Output ``.zip`` path; its parent directory is created.
        config_name: Recorded as the checkpoint's ``config_name``.
        notes: Recorded in the checkpoint's training metadata.

    Returns:
        ``filepath``.

    Raises:
        ValueError: If the model or its task losses have any learnable
            parameter. Saving an untrained initialization as a checkpoint
            would score random weights as if they were a model.
    """
    trainer = _build_inference_trainer(
        training_config,
        device="cpu",
        state_dict=None,
        cache_dir=None,
        enable_cache=False,
        config_name=config_name,
        # Never launched: no world size or precision mode to record.
        source_launch_info=SourceLaunchInfo(),
        # Never trained, so no training rates to record.
        source_neutral_rates_source=None,
    )
    parameters = [name for name, _ in trainer.unwrapped_model.named_parameters()]
    if parameters:
        raise ValueError(
            "save_parameter_free_checkpoint writes only models with no "
            f"learnable parameters, but this one has {len(parameters)} "
            f"(first: {parameters[:5]}). Train it instead."
        )
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    save_model(trainer.unwrapped_model, trainer.export_metadata(notes=notes), filepath)
    return filepath


def flash_attention_available(device: str = "cuda") -> bool:
    """True if the flash-attn kernel can actually run on ``device``.

    Two conditions, not one. The package has to be importable, *and* the model
    has to be headed for a CUDA device: flash-attn ships CUDA kernels only, so
    on a host where it is installed a CPU load sails past the import and then
    dies inside the forward pass with ``NotImplementedError: Could not run
    'flash_attn::_flash_attn_varlen_forward' with arguments from the 'CPU'
    backend``. Testing importability alone makes the dense fallback fire on a
    host without the package and stay silent on a host with it -- exactly
    inverted from what the fallback exists for, and the failure lands at
    inference rather than at load.
    """
    return importlib.util.find_spec("flash_attn") is not None and device.startswith(
        "cuda"
    )


def _fall_back_to_dense_attention(
    training_config: "TrainingConfig", device: str
) -> None:
    """Switch ``training_config``'s encoder to dense attention if flash can't run.

    The rule behind ``allow_dense_attention_fallback`` in :func:`load_model_only`
    and :func:`load_checkpoint`; see the former's docstring for why it is safe.
    """
    from dasm2.config import EncoderConfig

    # `EncoderConfig` is the only member of the encoder union that carries a
    # flash-attention flag, so the isinstance check is the type narrowing and
    # the "nothing to fall back from" case at once.
    encoder = training_config.model.encoder
    if (
        isinstance(encoder, EncoderConfig)
        and encoder.flash_attention
        and not flash_attention_available(device)
    ):
        encoder.flash_attention = False


def _override_neutral_model_path(
    training_config: "TrainingConfig", neutral_model_path: Path, filepath: Path
) -> None:
    """Point ``training_config``'s neutral model at ``neutral_model_path``.

    The rule behind ``neutral_model_path`` in :func:`load_checkpoint`. Only a
    recorded ``model_path`` is replaced: a neutral named from the catalog or
    built from inline parameters has no path to be stale, and swapping one in
    would silently change which neutral the checkpoint loads.
    """
    neutral_config = training_config.neutral_model
    if neutral_config is None or getattr(neutral_config, "model_path", None) is None:
        raise ValueError(
            f"neutral_model_path was given, but checkpoint {filepath} records "
            f"no neutral model_path to override (neutral_model: "
            f"{neutral_config!r})."
        )
    # model_copy, as train_command.py does for model_name overrides.
    training_config.neutral_model = neutral_config.model_copy(
        update={"model_path": neutral_model_path}
    )


def load_model_only(
    filepath: Path,
    device: str = "cpu",
    allow_dense_attention_fallback: bool = False,
) -> "torch.nn.Module":
    """Load just the trained CoreModel from a checkpoint, skipping the neutral model.

    Use this for analysis or inference workflows that only need the model's
    forward pass (e.g. computing selection factors). It avoids the
    neutral-model construction step in :func:`load_checkpoint`, which can
    fail when a checkpoint's ``training_config.neutral_model.model_path``
    points to an absolute path that doesn't exist on the current machine
    (a common cross-machine portability issue). When the neutral model is
    needed too, pass ``neutral_model_path`` to :func:`load_checkpoint`.

    Args:
        filepath: Path to checkpoint .zip file.
        device: Device to load the model on (``"cpu"`` or ``"cuda"``).
        allow_dense_attention_fallback: Load a checkpoint trained with
            ``flash_attention: true`` on a host where the flash-attn kernel is
            absent, by building dense attention instead. Flash and dense
            attention are two implementations of the same operation over the
            same weights, so this changes the kernel, not the model
            (``test_flash_matches_dense_attention_at_spike_length`` pins the
            agreement to ``atol=1e-2`` with identical weights loaded
            ``strict=True``). The flag is only ever honoured in the *off*
            direction, and only when the kernel cannot run on ``device`` -- see
            :func:`flash_attention_available`, which means "importable AND
            headed for CUDA", not merely "installed". A CUDA load on a host with
            the kernel therefore keeps using it. Off by default: silently
            swapping kernels is the caller's decision to make, not this
            function's.

    Returns:
        ``CoreModel`` with trained weights loaded, in eval-ready state.
        The neutral model, optimizer, and Trainer plumbing are not
        reconstructed — only the model itself.

    Raises:
        ValueError: If the checkpoint metadata is missing ``training_config``.
        FileNotFoundError: If ``filepath`` does not exist.
    """
    # Local imports: serialization -> training is a circular dependency
    # via Trainer; the neutral-model factory has its own import chain we
    # explicitly want to avoid here.
    from dasm2.config import TrainingConfig
    from dasm2.models.factory import create_model_from_config

    metadata, state_dict = load_model_components(filepath, device=device)
    _assert_supported_serialization_version(metadata, Path(filepath))

    if "training_config" not in metadata:
        raise ValueError(
            f"Checkpoint metadata missing 'training_config' key. "
            f"Available keys: {list(metadata.keys())}. "
            f"Note: This checkpoint may be from an old version. "
            f"Checkpoints must be created with the new TrainingConfig format."
        )

    training_config = TrainingConfig(**metadata["training_config"])
    _assert_unambiguous_sf_projection(training_config, Path(filepath), resumable=False)

    if allow_dense_attention_fallback:
        _fall_back_to_dense_attention(training_config, device)

    model = create_model_from_config(training_config.model, device=device)

    # The state_dict carries ``task_losses.<key>.*`` entries (e.g. the
    # contrastive loss's ``logit_bias``); they need real submodules to land on,
    # exactly as in ``load_checkpoint``.
    _attach_task_losses_for_inference(model, training_config)

    model.load_state_dict(state_dict)
    return model


def resume_training(
    filepath: Path,
    tasks: List["Task"],
    cache_dir: Optional[Path] = None,
    enable_cache: bool = True,
    backend: Optional["ComputeBackend"] = None,
) -> ResumeResult:
    """Load resumable checkpoint as Trainer ready to continue training.

    Reconstructs full training context from checkpoint metadata and restores
    all mutable training state (optimizer, scheduler, counters, RNG, best model).

    Works single- or multi-GPU: pass the ``backend`` the run is launched
    under and the model/optimizer are prepared for it. Resuming at a world
    size other than the one the checkpoint was trained at is *not* supported
    and is rejected upstream by ``_validate_resume_world_size_matches`` —
    the restored LR schedule was annealed for the original effective batch
    size, and rescaling it is a separate design question (issue #1027).

    Rebuilds ``head_optimizer`` (issue #1220) from the checkpoint's own
    ``training_config.head_optimizer``/``head_optimizer_head_type`` the same
    way it rebuilds the main optimizer — this run's own tasks/backend are
    the only inputs that come from the caller; every optimizer/scheduler
    setting, including ``head_optimizer``, is the checkpoint's own. Raises
    ``ValueError`` if ``head_optimizer`` is set under a multi-process
    backend, mirroring ``create_trainer``'s own guard.

    The trainer's ``checkpoint_id`` stays ``None``, so the resume reads and
    writes the same branch-length cache names as the run it resumes (#1876).

    Returns a :class:`ResumeResult` dataclass. Pass the fields to
    :meth:`Trainer.multi_cycle_train` (or ``multi_cycle_train_joint``) to
    resume from the correct cycle/task/step::

        result = resume_training(path, tasks=tasks)
        result.trainer.multi_cycle_train(
            num_cycles=N,
            start_cycle=result.start_cycle,
            start_cycle_task_index=result.start_cycle_task_index,
            start_cycle_step=result.start_cycle_step,
        )

    Args:
        filepath: Path to resumable checkpoint .zip file
        tasks: List of Task objects for training (must match original training tasks)
        cache_dir: Cache directory for branch lengths/predictions (default: .dasm2_cache)
        enable_cache: Enable caching for branch lengths and predictions
        backend: Compute backend to resume under, and the single source of
            truth for device placement — there is deliberately no separate
            ``device`` argument to contradict it. Defaults to
            ``SingleGPUBackend("cpu")``; pass ``SingleGPUBackend("cuda")`` to
            resume on a GPU, or an ``AccelerateBackend`` to resume a
            multi-GPU run, in which case the model and optimizer are
            DDP-wrapped via ``backend.prepare_model_and_optimizer``, exactly
            as ``create_trainer`` does for a fresh run. Threading the backend
            in *without* that wrap would be worse than the old hard-wired
            ``SingleGPUBackend``: every rank would train an independent model
            on its own data shard with no gradient all-reduce, silently
            (issue #1027).

    Returns:
        :class:`ResumeResult` with ``trainer``, ``start_cycle``,
        ``start_cycle_task_index``, and ``start_cycle_step``. When the save
        landed at a cycle boundary the task index and step are 0 and the
        caller will resume from the top of the next cycle.

    Raises:
        ValueError: If checkpoint does not contain training_state.pth
        ValueError: If checkpoint format is invalid or missing required metadata
        FileNotFoundError: If checkpoint file doesn't exist
    """
    # Local imports: serialization -> training is a circular dependency
    from dasm2.config import TrainingConfig, create_neutral_model_from_config
    from dasm2.models.factory import create_model_from_config
    from dasm2.training.compute_backend import SingleGPUBackend
    from dasm2.training.head_optimizer_state import (
        assert_head_optimizer_supported_by_backend,
    )
    from dasm2.training.optimizer import create_optimizer
    from dasm2.training.task import build_task_loss_module_dict
    from dasm2.training.trainer import Trainer
    from dasm2.training.types import StepBudget

    # Same CPU default as before the backend became the only way to say where
    # to load, so a bare resume behaves identically to a bare fresh run.
    if backend is None:
        backend = SingleGPUBackend("cpu")

    # ``model_init_device`` is the backend's answer to "where do weights go
    # before prepare?" — the device itself for single-GPU, "cpu" under
    # accelerate, which owns placement. Used for the state-dict map_location
    # too, so nothing is staged on a device prepare is about to move it off.
    init_device = backend.model_init_device

    # Load all components including training state
    metadata, model_state_dict, training_state = load_resumable_checkpoint(
        filepath, device=init_device
    )
    _assert_supported_serialization_version(metadata, Path(filepath))

    if "training_config" not in metadata:
        raise ValueError(
            f"Checkpoint metadata missing 'training_config' key. "
            f"Available keys: {list(metadata.keys())}."
        )

    training_config = TrainingConfig(**metadata["training_config"])
    _assert_unambiguous_sf_projection(training_config, Path(filepath), resumable=True)

    if training_config.head_optimizer is not None:
        assert_head_optimizer_supported_by_backend(backend)

    # Create model — keys match exactly under the v5 contract.
    model = create_model_from_config(training_config.model, device=init_device)

    # Attach per-task loss modules to the model so their parameters live in
    # ``model.parameters()`` AND in ``model.state_dict()``. The optimizer is
    # built next and would otherwise miss them; the v5 state_dict carries
    # ``task_losses.<key>.*`` entries that ``load_state_dict`` needs to map
    # onto these modules. The Trainer constructor re-registers the same
    # instances idempotently — ``nn.ModuleDict`` does not deep-copy values,
    # so parameter ``id()`` values survive the rebuild and the optimizer's
    # param group remains valid.
    model.task_losses = build_task_loss_module_dict(tasks)

    # Now that the model has its full v5 layout, load weights.
    model.load_state_dict(model_state_dict)

    # Re-apply the encoder freeze the interrupted run had, if any. Must
    # happen here, before create_optimizer builds its param groups from
    # requires_grad-filtered parameters below, and before the DDP wrap in
    # backend.prepare_model_and_optimizer — flipping requires_grad after
    # the wrap deadlocks DDP's reducer, which registers every
    # requires-grad parameter at construction time (issue #1076).
    if training_config.freeze_encoder:
        model.freeze_encoder()

    # Create neutral model
    neutral_model = (
        create_neutral_model_from_config(training_config.neutral_model)
        if training_config.neutral_model is not None
        else None
    )

    config_name = metadata.get("config_name")

    # head_optimizer_module is resolved before backend.prepare_model_and_optimizer
    # below, while ``model`` is still the plain (unwrapped) CoreModel — mirrors
    # create_trainer (issue #1220; PR #1223 review: resume_training() didn't
    # thread head_optimizer through at all until now).
    head_optimizer_module = (
        model.require_head(training_config.head_optimizer_head_type)
        if training_config.head_optimizer_head_type is not None
        else None
    )

    # Create optimizer from config (state will be restored via load_state_dict()).
    # Excludes head_optimizer's own parameters, if any, same as create_trainer.
    optimizer = create_optimizer(
        model, training_config.optimizer, exclude_module=head_optimizer_module
    )
    head_optimizer = (
        create_optimizer(head_optimizer_module, training_config.head_optimizer)
        if head_optimizer_module is not None
        and training_config.head_optimizer is not None
        else None
    )

    # DDP wrap + device placement, mirroring create_trainer. No-op for
    # SingleGPUBackend. Must come after ``model.task_losses`` is attached
    # above, per the wrapper-bypass contract that
    # ``tests/integration/_loss_params_ddp_worker.py`` pins, and after
    # ``load_state_dict`` because wrapping renames state_dict keys.
    model, optimizer = backend.prepare_model_and_optimizer(model, optimizer)  # type: ignore[assignment]
    if head_optimizer is not None:
        head_optimizer = backend.prepare_optimizer(head_optimizer)

    cache_dir_str = str(cache_dir) if cache_dir is not None else ".dasm2_cache"
    trainer = Trainer(
        model=model,
        neutral_model=neutral_model,
        tasks=tasks,
        batch_size=training_config.batch_size,
        val_batch_size=training_config.val_batch_size,
        backend=backend,
        optimizer=optimizer,
        cache_dir=cache_dir_str if enable_cache else None,
        config_name=config_name,
        gradient_clip_val=training_config.gradient_clip_val,
        early_stopping_patience=training_config.early_stopping_patience,
        early_stopping_min_delta=training_config.early_stopping_min_delta,
        monitor_task=training_config.monitor_task,
        scheduler_config=training_config.scheduler,
        use_mixed_precision=training_config.use_mixed_precision,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        num_dataloader_workers=training_config.num_dataloader_workers,
        max_tokens_per_batch=training_config.max_tokens_per_batch,
        dataloader_prefetch_factor=training_config.dataloader_prefetch_factor,
        dataloader_pin_memory=training_config.dataloader_pin_memory,
        step_budget=StepBudget(
            max_train_steps=training_config.max_train_steps,
            eval_every_steps=training_config.eval_every_steps,
            max_eval_steps=training_config.max_eval_steps,
        ),
        head_optimizer=head_optimizer,
        head_optimizer_module=head_optimizer_module,
        head_optimizer_head_type=training_config.head_optimizer_head_type,
    )

    # Restore full training state (optimizer, scheduler, counters, RNG, best model)
    trainer.restore_training_state(training_state)

    # If the save landed mid-cycle (cycle_step > 0), resume from that cycle
    # and position. Otherwise we finished cycle N and should resume at N+1.
    mid_cycle = training_state.cycle_step > 0 or training_state.cycle_task_index > 0
    start_cycle = (
        training_state.current_cycle if mid_cycle else training_state.current_cycle + 1
    )
    return ResumeResult(
        trainer=trainer,
        start_cycle=start_cycle,
        start_cycle_task_index=training_state.cycle_task_index,
        start_cycle_step=training_state.cycle_step,
    )
