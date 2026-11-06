"""Training metadata capture for DASM2 core functionality."""

import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Literal, NamedTuple, Optional, Union

import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict

from dasm2.config import TrainingConfig
from dasm2.data.neutral_rates_source import NeutralRatesSource

# Single source of truth for the on-disk serialization version. The default
# field on `BaseTrainingMetadata` and the gate in
# `dasm2.serialization.checkpoints.load_model_components` must reference this
# constant — bumping it in one place without the other is precisely the bug
# fixed in #609 (the SF refactor changed the layout but didn't bump v3).
CURRENT_SERIALIZATION_VERSION = 5


Vcs = Literal["git", "jj"]


class GitInfo(BaseModel):
    """Version-control metadata for the code that ran.

    Filled from ``git`` when it can name the revision. In a ``jj`` workspace
    that ``git`` cannot see (a non-colocated workspace), ``commit_sha`` is
    ``jj``'s ``@`` commit, ``status`` is ``"jj-snapshot"`` (the commit
    snapshots the working copy, and may exist only in the local ``jj`` repo
    until it is pushed), ``branch`` the nearest ancestor bookmark, and
    ``remote_url`` jj's ``origin``. Every field is ``"unknown"`` when neither
    tool can name the revision. No new key records which tool was used: the
    schema is kept to these four keys so older code that parses metadata
    strictly (``extra="forbid"``) still reads new metadata.
    """

    commit_sha: str
    branch: str
    status: str
    remote_url: str

    model_config = ConfigDict(extra="forbid")


class VcsRevision(NamedTuple):
    """The revision of the code in a working copy; see :func:`get_vcs_revision`."""

    vcs: Vcs
    """The tool that named the revision."""
    commit: str
    """``git``'s ``HEAD``, or ``jj``'s ``@`` commit id."""
    dirty: bool
    """Whether the working copy differs from ``commit``: tracked changes or
    untracked files that are not ignored. Always False under ``jj``, whose
    ``@`` commit is a snapshot of the working copy itself."""


class SystemInfo(BaseModel):
    """System and environment metadata."""

    python_version: str
    platform: str
    hostname: str
    torch_version: str
    cuda_available: bool
    cuda_version: str
    gpu_count: int
    gpu_names: List[str]

    model_config = ConfigDict(extra="forbid")


class TrainingMetadataInfo(BaseModel):
    """Training session metadata (git, system, command, notes).

    ``num_processes`` and ``mixed_precision`` are *launch-environment* facts:
    what the run was actually launched under, which no config records. Both
    are deliberately here and not on ``TrainingConfig``: that is the
    YAML-facing declarative config (DESIGN D7), where either one would be a
    value a user could write down with no effect on the launch.

    ``num_processes`` is the DDP world size the run was training at. It is
    *not* the same thing as ``system.gpu_count``, which is the saving host's
    visible-device count and diverges from the world size under
    ``--num_gpus``, ``--gpus``, or ``CUDA_VISIBLE_DEVICES``.

    ``mixed_precision`` is the *backend's* precision mode from
    ``--mixed_precision``, in Accelerate's spelling: ``"no"``, ``"fp16"``,
    or ``"bf16"``. It is *not* ``training_config.use_mixed_precision``, which
    is the trainer's own BF16 autocast around the forward pass — that one is
    config-declared and so already round-trips inside ``training_config``.
    Left as a plain ``str`` rather than a ``Literal`` so a checkpoint written
    under a mode this version doesn't know (e.g. ``"fp8"``) still loads.

    ``None`` means "not recorded" for either field — every checkpoint written
    before issue #1027 is in that state, and the resume validators treat it
    as unknown and skip their check rather than assuming a value. Note that
    "not recorded" and "recorded as off" are therefore distinguishable:
    ``mixed_precision`` is ``"no"``, never ``None``, when a run records it.
    """

    git: GitInfo
    system: SystemInfo
    command: str
    notes: str
    num_processes: Optional[int] = None
    mixed_precision: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class SourceLaunchInfo(BaseModel):
    """Launch facts read off a loaded checkpoint, to be re-emitted verbatim.

    In-memory only — never serialized. A ``Trainer`` built by
    ``load_checkpoint`` is handed a *fabricated* ``SingleGPUBackend``, so
    reading these facts off its backend at write time would rewrite a
    world-size-4 bf16 checkpoint's metadata to world size 1 and ``"no"`` on
    any load-mutate-save workflow — the round-trip defect class of issue #974
    through a different door. Such a Trainer carries the parsed values here
    instead and ``export_metadata`` prefers them (issue #1027).

    Presence of this object is what marks a Trainer as carrying: ``None`` on
    the Trainer means "the backend is live, read the facts off it", while a
    ``None`` *field* on a carried instance means "the source checkpoint
    predates the field and recorded nothing" and is re-emitted as ``None``.
    Collapsing that second case to the fabricated backend's value would
    manufacture a fact the source never claimed.
    """

    num_processes: Optional[int] = None
    mixed_precision: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class SourceCheckpointInfo(BaseModel):
    """Provenance info for fine-tuned models.

    When a model is fine-tuned from an existing checkpoint, this records
    the source checkpoint's path and hash for reproducibility.
    """

    path: str  # Original path used during fine-tuning
    md5: str  # Hash of the entire checkpoint ZIP file

    model_config = ConfigDict(extra="forbid")


class BaseTrainingMetadata(BaseModel):
    """Common metadata for checkpoints and artifacts.

    Contains everything needed for provenance tracking and validation:
    - serialization_version: Format version for forward compatibility
    - training_config: Full model/training configuration
    - config_name: Optional named config identifier
    - training_metadata: Git, system, command, and notes
    - created_at: Timestamp when metadata was created (informational only,
      not used for cache key computation)
    - source_checkpoint: Optional provenance info for fine-tuned models
    - neutral_rates_source: Where training's per-site neutral rates came from
      (see ``NeutralRatesSource``). ``None`` for checkpoints that predate the
      field, and for runs with no rate-consuming dataset.

    Serialization version history:
        1: Original format (nn.TransformerEncoderLayer key names)
        2: TransformerBlock key names (PR #379): self_attn → self_attention,
           linear1/2 → feed_forward.linear1/2
        3: Position-aware norm names (#444): norm1 → post_attn_norm,
           norm2 → post_ffn_norm. Note that v3 is layout-ambiguous:
           checkpoints stamped with v3 may carry *either* the old per-head
           ``prediction_heads.{Head}.prediction_layer.*`` keys (pre-#513)
           or the new shared ``sf_projection.projection.*`` keys (post-#513,
           because the SF refactor never bumped the version stamp).
        4: Shared SF projection (#513, version stamp added in #616):
           prediction_heads.{Head}.prediction_layer.* moved to
           sf_projection.projection.*. v4 is the first version that
           *guarantees* the post-#513 layout.
        5: Task/loss separation (#621): per-task ``LossConfig`` is owned by
           the new ``training_config.tasks: [...]`` block; head_configs are
           deduplicated; loss-owned parameters live under
           ``state_dict["task_losses.<key>.*"]`` (was ``prediction_heads.<head>.*``).
           Pre-v5 checkpoints must be migrated via
           ``scripts/migrate_checkpoint_to_v5.py``; the loader rejects them.
    """

    serialization_version: int = CURRENT_SERIALIZATION_VERSION
    training_config: TrainingConfig
    config_name: Optional[str] = None
    training_metadata: TrainingMetadataInfo
    created_at: str
    source_checkpoint: Optional[SourceCheckpointInfo] = None
    neutral_rates_source: Optional[NeutralRatesSource] = None

    model_config = ConfigDict(extra="forbid")


class ArtifactMetadata(BaseTrainingMetadata):
    """Artifact-specific metadata extending base training metadata.

    Adds artifact_type for cache identification. The created_at timestamp
    is inherited from BaseTrainingMetadata.
    """

    artifact_type: str

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_base_metadata(
        cls,
        artifact_type: str,
        base_metadata: "BaseTrainingMetadata",
    ) -> "ArtifactMetadata":
        """Create artifact metadata from base training metadata.

        Args:
            artifact_type: Type of artifact (e.g., "branch_lengths", "predictions")
            base_metadata: Base metadata from trainer.export_metadata()

        Returns:
            Complete ArtifactMetadata with current timestamp
        """
        return cls(
            artifact_type=artifact_type,
            created_at=str(pd.Timestamp.now()),
            training_config=base_metadata.training_config,
            config_name=base_metadata.config_name,
            training_metadata=base_metadata.training_metadata,
            source_checkpoint=base_metadata.source_checkpoint,
            neutral_rates_source=base_metadata.neutral_rates_source,
        )


def _vcs_output(
    argv: List[str], repo: Optional[Union[str, Path]]
) -> Optional["subprocess.CompletedProcess[str]"]:
    """Run a ``git``/``jj`` query in ``repo``; None if the tool is missing."""
    try:
        return subprocess.run(argv, cwd=repo, capture_output=True, text=True)
    except FileNotFoundError:
        return None


def _vcs_stdout(argv: List[str], repo: Optional[Union[str, Path]]) -> Optional[str]:
    """Stripped stdout of a successful, non-empty query, else None."""
    result = _vcs_output(argv, repo)
    if result is None or result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip()


def get_vcs_revision(
    repo: Optional[Union[str, Path]] = None,
) -> Optional[VcsRevision]:
    """Name the revision of the code checked out at ``repo``.

    Asks ``git`` first (``HEAD``, and ``git status --porcelain`` for tracked
    changes and untracked files). When ``git`` cannot name it -- e.g. in a
    non-colocated ``jj`` workspace, which has no ``.git`` -- asks ``jj`` for
    its ``@`` commit. That command snapshots the working copy, so the commit
    pins uncommitted edits too and the revision is never dirty. The snapshot
    exists only in the local ``jj`` repo until it is pushed.

    Args:
        repo: Directory inside the working copy; the current directory if None.

    Returns:
        The revision, or None if neither tool can name one (not a
        repository, or neither tool installed).
    """
    commit = _vcs_stdout(["git", "rev-parse", "HEAD"], repo)
    if commit is not None:
        changes = _vcs_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], repo
        )
        dirty = changes is None or changes.returncode != 0 or bool(changes.stdout)
        return VcsRevision("git", commit, dirty)
    commit = _vcs_stdout(
        ["jj", "log", "-r", "@", "--no-graph", "-T", "commit_id"], repo
    )
    if commit is not None:
        return VcsRevision("jj", commit, False)
    return None


def _jj_origin_url(remote_list: str) -> Optional[str]:
    """The ``origin`` URL from ``jj git remote list`` output (``name url`` lines)."""
    for line in remote_list.splitlines():
        name, _, url = line.strip().partition(" ")
        if name == "origin" and url.strip():
            return url.strip()
    return None


def get_git_info(*, cwd: str | Path | None = None) -> GitInfo:
    """Get version-control metadata for a working copy.

    See :func:`get_vcs_revision` for how the revision is found, and
    :class:`GitInfo` for what each field means under each tool.

    Args:
        cwd: Checkout to inspect; defaults to the process working directory.

    Returns:
        GitInfo with current repository state
    """
    revision = get_vcs_revision(cwd)
    if revision is None:
        return GitInfo(
            commit_sha="unknown",
            branch="unknown",
            status="unknown",
            remote_url="unknown",
        )
    if revision.vcs == "git":
        branch = _vcs_stdout(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
        remote_url = _vcs_stdout(["git", "remote", "get-url", "origin"], cwd)
    else:
        # The @ revision was just snapshotted by get_vcs_revision; don't
        # snapshot again (a concurrent edit would make the fields disagree).
        branch = _vcs_stdout(
            [
                "jj",
                "log",
                "--ignore-working-copy",
                "-r",
                "latest(::@ & bookmarks())",
                "--no-graph",
                "-T",
                'local_bookmarks.map(|b| b.name()).join(",")',
            ],
            cwd,
        )
        remotes = _vcs_stdout(
            ["jj", "git", "remote", "list", "--ignore-working-copy"], cwd
        )
        remote_url = _jj_origin_url(remotes) if remotes is not None else None
    if revision.vcs == "jj":
        # Not "clean": the snapshot may exist only in this local jj repo.
        status = "jj-snapshot"
    else:
        status = "dirty" if revision.dirty else "clean"
    return GitInfo(
        commit_sha=revision.commit,
        branch=branch or "unknown",
        status=status,
        remote_url=remote_url or "unknown",
    )


def get_system_info() -> SystemInfo:
    """Get system and environment metadata.

    Returns:
        SystemInfo with current system state
    """
    # Get GPU information
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_names: List[str] = []

    if cuda_available:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        cuda_version = torch.version.cuda or "unknown"
    else:
        cuda_version = "not_available"

    return SystemInfo(
        python_version=sys.version,
        platform=platform.platform(),
        hostname=platform.node(),
        torch_version=str(torch.__version__),
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
    )


def create_training_metadata_info(
    command: Optional[List[str]] = None,
    notes: Optional[str] = None,
    launch_info: Optional[SourceLaunchInfo] = None,
) -> TrainingMetadataInfo:
    """Create training metadata info with git and system information.

    Args:
        command: Command line arguments used to start training
        notes: Optional training notes
        launch_info: World size and backend precision mode of the run, or None
            when the caller has no backend to read them off (the run logger's
            environment block, artifact metadata). Both stay unrecorded then.

    Returns:
        TrainingMetadataInfo with git, system, command, notes, and launch facts
    """
    launch_info = launch_info or SourceLaunchInfo()
    return TrainingMetadataInfo(
        git=get_git_info(),
        system=get_system_info(),
        command=_format_command(command),
        notes=notes or "",
        num_processes=launch_info.num_processes,
        mixed_precision=launch_info.mixed_precision,
    )


def _format_command(command: Optional[List[str]]) -> str:
    """Format command line arguments as string."""
    if command is None:
        # Try to get from sys.argv
        command = sys.argv

    return " ".join(command)


def create_artifact_metadata(
    artifact_type: str,
    training_config: TrainingConfig,
    config_name: Optional[str] = None,
    command: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> ArtifactMetadata:
    """Create complete artifact metadata.

    Args:
        artifact_type: Type of artifact (e.g., "branch_lengths", "predictions")
        training_config: Full training configuration
        config_name: Optional named config identifier
        command: Command line arguments
        notes: Optional notes

    Returns:
        Complete ArtifactMetadata ready for serialization
    """
    return ArtifactMetadata(
        artifact_type=artifact_type,
        created_at=str(pd.Timestamp.now()),
        training_config=training_config,
        config_name=config_name,
        training_metadata=create_training_metadata_info(command=command, notes=notes),
    )
