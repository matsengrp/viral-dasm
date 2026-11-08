"""I/O utilities for dasm2.

Shared helpers for reading biological file formats and locating external
tool binaries.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from Bio import SeqIO

from dasm2.data.local import get_config_path, get_user_config


def find_phyz() -> Path:
    """Locate the phyz binary.

    Resolution order:

    1. ``phyz_binary`` from the user config (``~/.config/dasm2/config.yaml``).
    2. ``shutil.which("phyz")`` (i.e. anywhere on ``PATH``).

    No machine-specific source-tree paths are baked in — set
    ``phyz_binary`` in the user config, or put phyz on ``PATH``.

    Returns:
        Path to the phyz binary.

    Raises:
        FileNotFoundError: If phyz cannot be found via either route. The
            message names both the config field and the resolved config-file
            path so users know exactly where to set it.
    """
    config = get_user_config()
    if config.phyz_binary:
        configured = Path(config.phyz_binary).expanduser()
        if configured.is_file():
            return configured
        raise FileNotFoundError(
            f"phyz binary not found at configured path: {configured}\n"
            f"  (from {get_config_path()} → phyz_binary)"
        )

    on_path = shutil.which("phyz")
    if on_path is not None:
        return Path(on_path)

    raise FileNotFoundError(
        "phyz binary not found. Either:\n"
        "  - put phyz on PATH, or\n"
        f"  - set `phyz_binary: /abs/path/to/phyz` in {get_config_path()}"
    )


def read_fasta_first_sequence(fasta_path: Path) -> str:
    """Read the first sequence from a FASTA file.

    Delegates parsing to BioPython ``SeqIO`` (already a project dependency)
    and uppercases the result.

    Args:
        fasta_path: Path to a FASTA file.

    Returns:
        Nucleotide or amino acid sequence string (uppercased).

    Raises:
        ValueError: If the file contains no sequence data.
    """
    with open(fasta_path) as handle:
        record = next(SeqIO.parse(handle, "fasta"), None)
    if record is None or len(record.seq) == 0:
        raise ValueError(f"No sequence found in {fasta_path}")
    return str(record.seq).upper()
