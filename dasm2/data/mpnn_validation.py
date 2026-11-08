"""Shared validation for MPNN feature NPZ files.

Centralizes shape, dimension, and NaN/Inf checks used by both
MPNNWhichmutDataset and MPNNDMSDataset during construction-time validation.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

# Expected dimensions for validation
MPNN_FEATURE_DIM = 256  # 2 decoder layers × 128
MPNN_LOGLIK_DIM = 20  # 20 standard amino acids


def load_and_validate_mpnn_npz(
    npz_path: Path,
    structure_id: str,
) -> Tuple[Tensor, Tensor]:
    """Load and validate MPNN features and loglikelihoods from an NPZ file.

    Checks that:
    - features shape is [L, 256]
    - loglikelihoods shape is [L, 20]
    - features and loglikelihoods have matching length L
    - No NaN or Inf values in either tensor

    Args:
        npz_path: Path to the NPZ file.
        structure_id: Structure identifier (for error messages).

    Returns:
        Tuple of (features [L, 256], loglikelihoods [L, 20]) as float tensors.

    Raises:
        FileNotFoundError: If NPZ file does not exist.
        ValueError: If shapes are wrong or values contain NaN/Inf.
    """
    if not npz_path.exists():
        raise FileNotFoundError(
            f"MPNN features not found for structure_id='{structure_id}': {npz_path}"
        )

    npz_data = np.load(npz_path)
    features = torch.from_numpy(npz_data["features"]).float()
    loglikelihoods = torch.from_numpy(npz_data["loglikelihoods"]).float()

    # Shape validation
    if features.ndim != 2 or features.shape[1] != MPNN_FEATURE_DIM:
        raise ValueError(
            f"Expected features shape [L, {MPNN_FEATURE_DIM}] for "
            f"structure '{structure_id}', got {list(features.shape)}"
        )
    if loglikelihoods.ndim != 2 or loglikelihoods.shape[1] != MPNN_LOGLIK_DIM:
        raise ValueError(
            f"Expected loglikelihoods shape [L, {MPNN_LOGLIK_DIM}] for "
            f"structure '{structure_id}', got {list(loglikelihoods.shape)}"
        )
    if features.shape[0] != loglikelihoods.shape[0]:
        raise ValueError(
            f"Feature length ({features.shape[0]}) != loglikelihoods length "
            f"({loglikelihoods.shape[0]}) for structure '{structure_id}'"
        )

    # NaN/Inf validation
    if not torch.all(torch.isfinite(features)):
        raise ValueError(f"NaN or Inf in features for structure '{structure_id}'")
    if not torch.all(torch.isfinite(loglikelihoods)):
        raise ValueError(f"NaN or Inf in loglikelihoods for structure '{structure_id}'")

    return features, loglikelihoods
