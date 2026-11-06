"""Shared frozen per-residue embedding I/O (issue #802).

Extracted from ``contrastive_binding.py`` so head-specific dataset modules
(``contrastive_binding.py``, ``pairwise_binding.py``) share one on-disk
loader and one process-local LRU cache for per-residue embeddings, rather
than each defining its own with the same bound (Constitution VI: "Extend,
Don't Duplicate"). Both modules import these functions directly.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

# LRU cache bound: embedding tensors can be large ([~hundreds, D] float32).
# Most SAbDab-scale runs use a few thousand distinct entities, which fits
# comfortably; capping avoids unbounded accumulation across experiments in
# the same process (e.g. sweeps, test runners).
EMBEDDING_CACHE_MAXSIZE = 4096


def load_residue_embedding(npz_path: Path) -> np.ndarray:
    """Load a per-residue embedding array from an ``.npz`` file.

    Expects the array under key ``"embeddings"`` of shape ``[L, D]``. Any
    additional keys in the NPZ (e.g. ``"sequence"``) are ignored.
    """
    with np.load(npz_path) as npz:
        if "embeddings" not in npz.files:
            raise ValueError(
                f"Embedding NPZ {npz_path} missing 'embeddings' key. "
                f"Found: {list(npz.files)}"
            )
        arr = npz["embeddings"]
    if arr.ndim != 2:
        raise ValueError(
            f"Embedding NPZ {npz_path} must hold a 2-D array [L, D], "
            f"got shape {arr.shape}."
        )
    return np.ascontiguousarray(arr, dtype=np.float32)


@lru_cache(maxsize=EMBEDDING_CACHE_MAXSIZE)
def cached_residue_tensor(npz_path_str: str) -> torch.Tensor:
    """Process-local cache of per-residue embeddings by absolute path string.

    Bounded at ``EMBEDDING_CACHE_MAXSIZE`` so cross-experiment accumulation
    in the same process doesn't grow forever. Per-worker when used from a
    ``DataLoader`` with ``num_workers > 0``.
    """
    return torch.from_numpy(load_residue_embedding(Path(npz_path_str)))


def resolve_embedding_path(embedding_dir: Path, key: str) -> Path:
    """Resolve ``key`` to ``{embedding_dir}/{key}.npz``, or raise."""
    path = embedding_dir / f"{key}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Embedding not found for key '{key}' at {path}.")
    return path


__all__ = [
    "EMBEDDING_CACHE_MAXSIZE",
    "load_residue_embedding",
    "cached_residue_tensor",
    "resolve_embedding_path",
]
