"""Shared naming utilities for checkpoints and artifacts.

This module provides pure functions for generating consistent filenames across
checkpoints and training artifacts. Centralizes naming logic to ensure consistency
and prevent duplication between checkpoint saving and artifact caching.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from dasm2.config import EncoderConfig, TrainingConfig

#: Literal prefix marking the run-token component of a checkpoint filename.
RUN_TOKEN_PREFIX = "run"

#: Matches the ``_{YYYYmmdd-HHMMSS}_{hex}`` tail that
#: :meth:`dasm2.training.logging.TrainingLogger.create` appends to every run ID.
_RUN_ID_TAIL_PATTERN = re.compile(r"_(\d{8}-\d{6})_([0-9a-f]+)$")

#: A run token as it appears inside a checkpoint filename (see
#: :func:`run_token_from_run_id`). Kept in sync with what that function emits.
RUN_TOKEN_BODY = rf"{RUN_TOKEN_PREFIX}\d{{8}}-\d{{6}}-[0-9a-f]+"

#: Matches a run token embedded in a checkpoint filename.
RUN_TOKEN_PATTERN = re.compile(rf"_({RUN_TOKEN_BODY})_")

#: The optional ``_{run_token}`` segment that :func:`build_checkpoint_filename`
#: places between a checkpoint's suffix and its ``_{hash}_{date}`` trailer.
#: Callers matching a suffix against that trailer must allow for it, and must
#: allow for its absence: checkpoints predating the token, or written by a run
#: with no structured logger, carry no token.
OPTIONAL_RUN_TOKEN = rf"(?:_{RUN_TOKEN_BODY})?"


def run_token_from_run_id(run_id: Optional[str]) -> Optional[str]:
    """Derive the filename-safe run token for a training run.

    Checkpoint filenames carry this token so that two attempts writing into
    the same directory produce distinguishable files, and so any checkpoint
    can be traced back to the ``logs/{run_id}_log.jsonl`` that describes it
    The token keeps the run ID's timestamp so filenames sort
    chronologically within a naming group, unlike the metadata hash.

    Args:
        run_id: Logger run ID, e.g. ``"dasm_16m_20260729-101530_a1b2c3"``.
            ``None`` when training ran without a structured logger.

    Returns:
        Token such as ``"run20260729-101530-a1b2c3"``, or ``None`` when
        *run_id* is absent or does not follow ``TrainingLogger``'s
        ``{prefix}_{YYYYmmdd-HHMMSS}_{hex}`` shape. Returning ``None`` rather
        than raising keeps a long training run from dying at its save call
        over a cosmetic naming concern; the checkpoint is simply written
        without a token, exactly as it was before this component existed.

    Examples:
        >>> run_token_from_run_id("dasm_16m_20260729-101530_a1b2c3")
        'run20260729-101530-a1b2c3'
        >>> run_token_from_run_id("hand-rolled-id") is None
        True
    """
    if run_id is None:
        return None
    match = _RUN_ID_TAIL_PATTERN.search(run_id)
    if match is None:
        return None
    return f"{RUN_TOKEN_PREFIX}{match.group(1)}-{match.group(2)}"


def parse_run_token(filename: str) -> Optional[str]:
    """Extract the run token from a checkpoint filename, if it carries one.

    Args:
        filename: Checkpoint filename or basename.

    Returns:
        The run token, or ``None`` for checkpoints written before run tokens
        existed or by a run whose logger was disabled.

    Examples:
        >>> parse_run_token("m_H-d_cycle3_best_run20260729-101530-a1b2c3_9612_20260729.zip")
        'run20260729-101530-a1b2c3'
        >>> parse_run_token("m_H-d_cycle3_9612_20260729.zip") is None
        True
    """
    match = RUN_TOKEN_PATTERN.search(filename)
    return match.group(1) if match else None


def generate_model_identifier(
    config_name: Optional[str],
    encoder_config: "EncoderConfig",
) -> str:
    """Generate model identifier from config name or architecture.

    Extracts config_name if available, otherwise generates architecture description
    from encoder configuration.

    Args:
        config_name: Optional config name (e.g., "dasm_8m", "77k")
        encoder_config: Encoder configuration for fallback generation

    Returns:
        Model identifier string (e.g., "dasm_8m" or "6L320D" or "ESM2-640D")

    Examples:
        >>> # With config name
        >>> generate_model_identifier("dasm_8m", encoder_config)
        "dasm_8m"
        >>> # Without config name (fallback to architecture)
        >>> generate_model_identifier(None, encoder_config)
        "6L320D"  # 6 layers, 320 d_model
        >>> # ESM2 encoder (no num_layers)
        >>> generate_model_identifier(None, esm2_encoder_config)
        "ESM2-640D"
        >>> # ESM2 encoder with a 5-layer, 128-wide decoder
        >>> generate_model_identifier(None, esm2_decoder_config)
        "ESM2-640D+5L128D"
    """
    if config_name is not None:
        return Path(config_name).stem

    # Handle ESM2 encoders, naming ESM2's own width and any decoder after it
    if encoder_config.type == "ESM2Encoder":
        identifier = f"ESM2-{encoder_config.esm2_hidden_dim}D"
        decoder = encoder_config.decoder
        if decoder is not None:
            identifier += f"+{decoder.num_layers}L{decoder.d_model}D"
        return identifier

    # Handle MPNN encoders (no num_layers)
    if encoder_config.type == "MPNNLightAttentionEncoder":
        d_model = encoder_config.d_model
        return f"MPNN-{d_model}D"

    # Handle IdentityEncoder (no transformer, no num_layers).
    # Used by encoder-free heads; the head's parameter shape is the more
    # meaningful identifier and is captured by the config filename instead.
    if encoder_config.type == "IdentityEncoder":
        return "Identity"

    # Generate from architecture: {num_layers}L{d_model}D
    num_layers = encoder_config.num_layers  # type: ignore[union-attr]
    d_model = encoder_config.d_model
    return f"{num_layers}L{d_model}D"


def generate_head_dataset_identifier(
    head_dataset_pairs: List[Tuple[str, str]],
) -> str:
    """Generate identifier from head-dataset pairs.

    Creates a string showing which head is trained on which dataset, using
    the format: HeadType-dataset+HeadType-dataset+...

    Args:
        head_dataset_pairs: List of (head_type_value, dataset_nickname) tuples.
            head_type_value is the string value from HeadType enum (e.g., "SelectionFactorHead").
            dataset_nickname is the human-readable dataset identifier.

    Returns:
        Head-dataset identifier string

    Examples:
        >>> # Single head
        >>> generate_head_dataset_identifier([("SelectionFactorHead", "v1tang")])
        "SelectionFactorHead-v1tang"
        >>> # Multiple heads
        >>> generate_head_dataset_identifier([
        ...     ("SelectionFactorHead", "v1jaffe"),
        ...     ("DMSFunctionalScoreHead", "megascale")
        ... ])
        "SelectionFactorHead-v1jaffe+DMSFunctionalScoreHead-megascale"
    """
    if not head_dataset_pairs:
        return "unknown_head"

    parts = [f"{head_type}-{dataset}" for head_type, dataset in head_dataset_pairs]
    return "+".join(parts)


def compute_metadata_hash(
    training_config: "TrainingConfig", hash_length: int = 8
) -> str:
    """Compute deterministic hash from training configuration.

    The hash is computed from the full TrainingConfig serialized to JSON with
    sorted keys for determinism. Same configuration always produces the same hash.

    Args:
        training_config: Complete training configuration
        hash_length: Number of hex characters to return (default: 8 = 16M values)

    Returns:
        Hex string of specified length

    Examples:
        >>> compute_metadata_hash(training_config)
        "a3b2c1d4"
        >>> compute_metadata_hash(training_config, hash_length=16)
        "a3b2c1d4e5f6g7h8"
    """
    # Serialize config to JSON with sorted keys for determinism
    config_dict = training_config.model_dump()
    config_json = json.dumps(config_dict, sort_keys=True, default=str)

    # Compute SHA-256 hash and truncate to requested length
    full_hash = hashlib.sha256(config_json.encode()).hexdigest()
    return full_hash[:hash_length]


def build_checkpoint_filename(
    config_name: Optional[str],
    encoder_config: "EncoderConfig",
    head_dataset_pairs: List[Tuple[str, str]],
    training_config: "TrainingConfig",
    suffix: Optional[str] = None,
    run_token: Optional[str] = None,
    hash_length: int = 8,
) -> str:
    """Build complete checkpoint filename from components.

    Format:
    ``{config_name}_{HeadType-dataset+...}[_{suffix}][_{run_token}]_{hash}_{date}.zip``

    The suffix (e.g., "cycle0_best") is inserted before the run token when
    provided. The run token identifies the training run that wrote the file,
    so two attempts sharing an output directory stay distinguishable; it sits
    ahead of the hash so that lexicographic sorting within a naming group is
    chronological rather than hash-ordered. The date (YYYYMMDD
    format) is appended after the hash for easy identification of when the
    checkpoint was created.

    Args:
        config_name: Optional config name (e.g., "dasm_8m")
        encoder_config: Encoder configuration for model identifier
        head_dataset_pairs: List of (head_type_value, dataset_nickname) tuples
        training_config: Complete training configuration for hash
        suffix: Optional suffix marking which save this is (e.g.,
            "cycle0_best" for a post-cycle save, "cycle0_step500" for an
            on-eval one)
        run_token: Optional run token from :func:`run_token_from_run_id`.
            Omitted when training ran without a structured logger.
        hash_length: Hash length (default: 8)

    Returns:
        Complete checkpoint filename with .zip extension

    Examples:
        >>> # No suffix, no run token
        >>> build_checkpoint_filename(
        ...     config_name="dasm_8m",
        ...     encoder_config=encoder_config,
        ...     head_dataset_pairs=[("SelectionFactorHead", "v1tang")],
        ...     training_config=training_config,
        ... )
        "dasm_8m_SelectionFactorHead-v1tang_a3b2c1d4_20260127.zip"
        >>> # Post-cycle checkpoint from a logged run
        >>> build_checkpoint_filename(
        ...     ..., suffix="cycle0_best", run_token="run20260127-101530-a1b2c3"
        ... )
        "dasm_8m_SelectionFactorHead-v1tang_cycle0_best_run20260127-101530-a1b2c3_a3b2c1d4_20260127.zip"
    """
    model_id = generate_model_identifier(config_name, encoder_config)
    head_dataset_id = generate_head_dataset_identifier(head_dataset_pairs)
    metadata_hash = compute_metadata_hash(training_config, hash_length)
    date_str = datetime.now().strftime("%Y%m%d")

    # Build filename parts
    parts = [model_id, head_dataset_id]
    if suffix is not None:
        parts.append(suffix)
    if run_token is not None:
        parts.append(run_token)
    parts.append(metadata_hash)
    parts.append(date_str)

    return "_".join(parts) + ".zip"
