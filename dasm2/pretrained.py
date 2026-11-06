"""Pretrained models for DASM2.

Provides utility functions for pretrained models used in DASM2.

Coverage relative to ``netam.pretrained`` (DASM2 does not import netam; netam is
a test-only reference):
    - Neutral models: resolved from the dasm2 model catalog
      (``model_catalog.yaml`` via ``dasm2.model_catalog``).
    - Multihit-correction models: the ``PRETRAINED_MULTIHIT_MODELS`` weights
      below (migrated from netam).
    - Selection models (DNSM/DASM): intentionally not mirrored — DASM2 trains
      its own selection models and never loads netam's pretrained selection
      weights.
"""

from typing import Dict, List, Optional, Tuple

import torch

from dasm2.neutral.hit_class import HitClassModel

# Pretrained multihit (hit-class) correction weights, migrated from netam so that
# neutral-model resolution no longer depends on netam.pretrained. Each entry is a
# (hit-class-1, hit-class-2, hit-class-3) log-rate-adjustment tuple. Names describe
# the origin of the multihit model and are not otherwise interpreted.
#
# These are kept as in-code constants rather than model_catalog.yaml entries: the
# catalog is for file-based archives that need downloading/integrity-checking,
# whereas these are three floats with no archive to resolve. To add another
# multihit model, append a tuple here (not a catalog entry).
PRETRAINED_MULTIHIT_MODELS: Dict[str, Tuple[float, float, float]] = {
    # Trained using the notebook
    # thrifty-experiments-1/human/multihit_model_exploration.ipynb
    "ThriftyHumV0.2-59-hc-tangshm": (-0.1626, 0.0692, 0.5076),
    "ThriftyHumV0.2-59-hc-shmoof": (-0.2068, 0.1603, 0.6317),
    # Fit against the S5F heavy-chain neutral model (S5F-HH-classic-v1), whose
    # rate table is standardized to mean rate 1 on load (#866); the correction
    # absorbs that model's codon-level structure, so it was re-fit after the
    # rescale. Fit via scripts/fit_multihit.py on the data-central Tang
    # out-of-frame PCPs
    # (tang-deepshm-oof_igh_pcp_2024-08-09_MASKED_NI_noN_no-naive.csv.gz,
    # md5 792d2fb775348f73c949fbe699394a76; site_count=500, train_frac=0.8,
    # bl_cycles=1). The factors are the exact concave MLE at the fitted branch
    # lengths, so the fit is deterministic -- no learning rate, epochs or seed.
    # A single shared correction reused for the light-chain model
    # (S5F-HKL-classic-v1); see docs/concepts/neutral-models.md.
    "S5F-HH-classic-v1-hc-tangshm": (-0.0898, 0.4913, 1.1625),
}


def load_multihit(
    model_name: str,
    device: Optional[torch.device] = None,
) -> HitClassModel:
    """Load a pretrained multihit model using dasm2's HitClassModel.

    Reads weights from ``PRETRAINED_MULTIHIT_MODELS`` (migrated from netam's
    registry) into dasm2's native HitClassModel implementation.

    Args:
        model_name: Name of the pretrained multihit model
            (e.g., "ThriftyHumV0.2-59-hc-tangshm")
        device: Optional torch device to place model on

    Returns:
        HitClassModel with pretrained weights

    Raises:
        ValueError: If model_name is not found in pretrained registry
    """
    if model_name not in PRETRAINED_MULTIHIT_MODELS:
        raise ValueError(
            f"Model {model_name} not found in pretrained multihit models. "
            f"Available: {list(PRETRAINED_MULTIHIT_MODELS.keys())}"
        )

    weights: Tuple[float, float, float] = PRETRAINED_MULTIHIT_MODELS[model_name]
    model = HitClassModel.from_weights(weights)

    if device is not None:
        model = model.to(device)

    return model


def get_default_multihit_model(
    device: Optional[torch.device] = None,
) -> Optional[HitClassModel]:
    """Get the default multihit model for testing and production use.

    Loads the thrifty model's multihit variant (ThriftyHumV0.2-59-hc-tangshm),
    which provides multihit correction for neutral evolutionary modeling.

    Args:
        device: Optional torch device to place model on. If None, the model
               stays on the default device.

    Returns:
        Loaded HitClassModel instance, or None if no pretrained models
        are available.

    Raises:
        ValueError: If the required default model is not found in available
                   pretrained models.
    """
    if not PRETRAINED_MULTIHIT_MODELS:
        return None

    # Use Thrifty model's multihit variant as default
    default_name = "ThriftyHumV0.2-59-hc-tangshm"
    if default_name not in PRETRAINED_MULTIHIT_MODELS:
        raise ValueError(
            f"Required model {default_name} not found in pretrained models"
        )

    return load_multihit(default_name, device=device)


def list_available_models() -> Dict[str, List[str]]:
    """List all available pretrained models.

    Neutral models come from the dasm2 model catalog (``model_catalog.yaml``);
    multihit correction models come from ``PRETRAINED_MULTIHIT_MODELS``.

    Returns:
        Dictionary with model categories as keys and model names as values:
        - 'neutral_models': List of available neutral evolution models
        - 'multihit_models': List of available multihit correction models
    """
    from dasm2.model_catalog import list_models  # noqa: PLC0415

    return {
        "neutral_models": [e.name for e in list_models(model_type="neutral_model")],
        "multihit_models": list(PRETRAINED_MULTIHIT_MODELS.keys()),
    }


# No re-exports - import functions directly:
# from dasm2.pretrained import get_default_multihit_model, load_multihit
