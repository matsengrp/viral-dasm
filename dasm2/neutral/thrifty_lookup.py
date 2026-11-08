"""Global 13-mer Thrifty lookup table for QDASM Option 1a.

ThriftyHumV0.2-59 has a bounded 13-nt receptive field (3-mer embeddings +
``Conv1d(kernel_size=11)``). The rate and CSP at position $\\ell$ are pure
functions of the 13-mer centered on $\\ell$. This module provides helpers to
build and query a global lookup table indexed by 13-mer, enabling O(1)
"evaluate Thrifty at any intermediate codon" during QDASM Q-matrix
construction.

Index convention:
    ``idx = base[0] * 4^12 + base[1] * 4^11 + ... + base[12]``
where A=0, C=1, G=2, T=3. N is not allowed in table keys; boundary sites
(13-mer contains N) use cached full-segment Thrifty rescans for each source
codon instead (``neutral.codon_boundary``).

Table shape: ``[4^13, 5]`` float32 — 1 rate + 4 CSP probabilities per entry.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Tuple

import torch

from dasm2.utils.artifacts import resolve_artifact_path

# Base-4 nucleotide convention matching kmer_encoding._BASE_TO_INT
# (A=0, C=1, G=2, T=3, N=4).
KMER_LEN = 13
CENTER_POS = KMER_LEN // 2  # index 6
TABLE_SIZE = 4**KMER_LEN  # 67,108,864
NUM_OUTPUT_FIELDS = 5  # 1 rate + 4 CSP probs

# Powers of 4 for base-4 encoding: [4^12, 4^11, ..., 4^0].
# Canonical encoding reference for table construction and direct queries.
KMER_POWERS = 4 ** torch.arange(KMER_LEN - 1, -1, -1, dtype=torch.long)


def encode_13mer_to_index(nt_indices: torch.Tensor) -> torch.Tensor:
    """Encode 13-nt sequences (as base-4 indices) into scalar table indices.

    Args:
        nt_indices: Long tensor of shape ``(..., 13)`` with entries in {0,1,2,3}.
            Callers are responsible for ensuring no N (value 4) is present.

    Returns:
        Long tensor of shape ``(...,)`` with entries in ``[0, 4^13)``.
    """
    if nt_indices.shape[-1] != KMER_LEN:
        raise ValueError(
            f"Expected last dim = {KMER_LEN}, got shape {tuple(nt_indices.shape)}"
        )
    powers = KMER_POWERS.to(device=nt_indices.device, dtype=nt_indices.dtype)
    result: torch.Tensor = (nt_indices * powers).sum(dim=-1)
    return result


def string_to_nt_indices(kmer: str) -> torch.Tensor:
    """Encode an ACGT-only string as a long tensor of base indices."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    try:
        return torch.tensor([mapping[b] for b in kmer], dtype=torch.long)
    except KeyError as exc:
        raise ValueError(f"Unsupported base {exc.args[0]!r} in {kmer!r}") from exc


def resolve_thrifty_13mer_table_path(spec: str | Path) -> Path:
    """Resolve a 13-mer table specifier to a concrete on-disk path.

    Lookup order:

    1. ``spec`` as-is (absolute or cwd-relative). Used by cluster configs
       that pin the canonical shared copy.
    2. ``get_artifacts_dir() / "thrifty_13mer" / <basename(spec)>``. Honors
       the user's ``~/.config/dasm2/config.yaml:artifacts_dir`` setting.
    3. ``get_experiment_artifact_root() / "thrifty_13mer" / <basename(spec)>``
       — the documented single source of truth (DESIGN.md D10).

    Returns the first regular file (directories are not artifacts). Raises ``FileNotFoundError`` listing
    every attempted location plus a pointer to the precompute notebook.
    """
    return resolve_artifact_path(
        spec,
        "thrifty_13mer",
        description="Thrifty 13-mer table not found",
        instructions=(
            "Build it with experiments/2026-04-01-qdasm-new-q/notebooks/"
            "build_thrifty_13mer_table.py, then set artifacts_dir in "
            "~/.config/dasm2/config.yaml or pass an absolute path."
        ),
    )


def load_thrifty_13mer_table(path: str | Path) -> torch.Tensor:
    """Load a precomputed 13-mer table.

    Args:
        path: Path to ``.pt`` tensor file. Resolved via
            ``resolve_thrifty_13mer_table_path`` so a bare filename works for
            users who configure ``artifacts_dir``.

    Returns:
        Tensor of shape ``[4^13, 5]`` float32.
    """
    resolved = resolve_thrifty_13mer_table_path(path)
    table: torch.Tensor = torch.load(resolved, map_location="cpu", weights_only=True)
    if table.shape != (TABLE_SIZE, NUM_OUTPUT_FIELDS):
        raise ValueError(
            f"Expected table shape ({TABLE_SIZE}, {NUM_OUTPUT_FIELDS}); "
            f"got {tuple(table.shape)}"
        )
    return table


@lru_cache(maxsize=16)
def _sha256_file_version(path: str, mtime_ns: int, size: int) -> str:
    """Hash one immutable file version; stat fields form the cache key."""
    del mtime_ns, size
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a potentially multi-gigabyte file."""
    stat = path.stat()
    return _sha256_file_version(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def load_thrifty_13mer_table_metadata(path: str | Path) -> dict[str, Any]:
    """Load and validate a 13-mer table's provenance sidecar.

    The table builder writes ``<stem>.json`` next to ``<stem>.pt``. Requiring
    that sidecar when a configured neutral-model identity must be checked keeps
    a raw tensor from silently masquerading as a table built from another model.

    Args:
        path: Path to the ``.pt`` table, resolved like
            :func:`load_thrifty_13mer_table`.

    Returns:
        Parsed metadata containing a non-empty ``model_name`` whose
        ``sha256_tensor_file`` matches the resolved tensor artifact.
    """
    resolved = resolve_thrifty_13mer_table_path(path)
    metadata_path = resolved.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Thrifty 13-mer table metadata not found at {metadata_path}. "
            f"The table source model cannot be verified. If {resolved.name} was "
            "copied from another location (a node-local copy for DDP, say), "
            "copy the matching .json sidecar next to it — that is the usual "
            "cause and the whole fix. Only rebuild the table with "
            "build_thrifty_13mer_table.py if no sidecar exists anywhere."
        )
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read Thrifty 13-mer table metadata at {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Thrifty 13-mer table metadata at {metadata_path} must be a JSON object"
        )
    model_name = metadata.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(
            f"Thrifty 13-mer table metadata at {metadata_path} must contain a "
            "non-empty string model_name"
        )
    expected_sha256 = metadata.get("sha256_tensor_file")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(
            f"Thrifty 13-mer table metadata at {metadata_path} must contain a "
            "64-character sha256_tensor_file"
        )
    actual_sha256 = _sha256_file(resolved)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"Thrifty 13-mer table SHA256 mismatch for {resolved}: metadata "
            f"records {expected_sha256}, computed {actual_sha256}. The sidecar "
            "does not describe this tensor file."
        )
    return metadata


def query_13mer_table(
    table: torch.Tensor, kmer_indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Look up (rate, csp) for a batch of 13-mer indices.

    Args:
        table: ``[4^13, 5]`` table from ``load_thrifty_13mer_table``.
        kmer_indices: Long tensor of arbitrary shape with entries in
            ``[0, 4^13)``.

    Returns:
        Tuple of (rates, csps):
            rates: same shape as kmer_indices.
            csps: same shape as kmer_indices with trailing dim 4.
    """
    entries = table[kmer_indices]  # (..., 5)
    rates = entries[..., 0]
    csps = entries[..., 1:]
    return rates, csps
