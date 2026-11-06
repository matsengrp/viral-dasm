"""netam weight loading utilities for DASM2."""

import os
import warnings
from typing import Any, Dict, Tuple

import torch
import yaml

from dasm2.config import (
    EncoderConfig,
    ModelConfig,
    SelectionFactorHeadConfig,
    SelectionFactorProjectionConfig,
)
from dasm2.models.key_migration import migrate_transformer_block_keys


def load_netam_data(crepe_path: str) -> Tuple[ModelConfig, Dict]:
    """Load netam data and convert to DASM2 format (no model creation).

    Args:
        crepe_path: Path to netam crepe files without extension (e.g., "model" for "model.pth"/"model.yml")

    Returns:
        Tuple of (DASM2 ModelConfig, netam_state_dict)
    """
    # Validate files exist and load data
    pth_path, yml_path = _validate_netam_files(crepe_path)
    netam_state_dict = _load_netam_state_dict(pth_path)
    model_config = _load_netam_config(yml_path)

    return model_config, netam_state_dict


def _validate_netam_files(crepe_path: str) -> Tuple[str, str]:
    """Validate netam files exist and return paths."""
    pth_path = f"{crepe_path}.pth"
    yml_path = f"{crepe_path}.yml"

    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Required netam weight file not found: {pth_path}")
    if not os.path.exists(yml_path):
        raise FileNotFoundError(f"Required netam config file not found: {yml_path}")

    return pth_path, yml_path


def _load_netam_state_dict(pth_path: str) -> Dict[str, Any]:
    """Load netam state_dict from .pth file."""
    return torch.load(pth_path, map_location="cpu")  # type: ignore[no-any-return]


def map_netam_state_dict(
    netam_state_dict: Dict[str, Any], encoder_config: EncoderConfig
) -> Dict[str, Any]:
    """Map netam state_dict keys to DASM2 structure.

    Key transformations for encoder weights (applied after prefix prepend):

    - ``self_attn`` → ``self_attention`` (TransformerBlock naming)
    - ``linear1`` → ``feed_forward.linear1`` (FFN wrapped in submodule)
    - ``linear2`` → ``feed_forward.linear2`` (FFN wrapped in submodule)
    - ``norm1`` → ``post_attn_norm`` (Post-LN norm naming)
    - ``norm2`` → ``post_ffn_norm`` (Post-LN norm naming)

    Example:
        netam key: ``encoder.layers.0.self_attn.in_proj_weight``
        → prefixed: ``encoder.transformer_encoder.layers.0.self_attn.in_proj_weight``
        → rewritten: ``encoder.transformer_encoder.layers.0.self_attention.in_proj_weight``

    Args:
        netam_state_dict: State dict from netam model
        encoder_config: DASM2 encoder configuration for validation

    Returns:
        State dict with DASM2-compatible keys
    """
    mapped_dict = {}
    for key, value in netam_state_dict.items():
        # Map netam keys to DASM2 structure
        if key.startswith("amino_acid_embedding"):
            mapped_key = f"encoder.{key}"
            mapped_dict[mapped_key] = value
        elif key.startswith("pos_encoder"):
            mapped_key = f"encoder.{key}"
            mapped_dict[mapped_key] = value
        elif key.startswith("encoder"):
            # netam: encoder.layers.X.* -> DASM2: encoder.transformer_encoder.layers.X.*
            # Step 1: prepend transformer_ prefix
            mapped_key = f"encoder.transformer_{key}"
            # Step 2: rewrite submodule names for TransformerBlock
            mapped_key = migrate_transformer_block_keys(mapped_key)
            mapped_dict[mapped_key] = value
        elif key.startswith("encoder_layer"):
            # netam has a single encoder_layer template, but we skip it in favor of encoder.layers.X
            continue
        elif key.startswith("linear"):
            # Map netam prediction head to shared SF projection
            # netam: linear.* -> DASM2: sf_projection.projection.*
            if key == "linear.weight":
                mapped_dict["sf_projection.projection.weight"] = value
            elif key == "linear.bias":
                mapped_dict["sf_projection.projection.bias"] = value
        else:
            # Copy other keys as-is (though there shouldn't be any)
            mapped_dict[key] = value

    return mapped_dict


# netam model classes that apply wiggle(x, 0.3) to selection factors.
_WIGGLE_MODEL_CLASSES = frozenset(
    {
        "TransformerBinarySelectionModelWiggleAct",
        "BidirectionalTransformerBinarySelectionModelWiggleAct",
    }
)

# The wiggle_beta value used by netam wiggle model classes. Same historical
# value as _PRE_513_DASM_EVOL_HEAD_DEFAULT_WIGGLE_BETA in checkpoints.py.
_NETAM_WIGGLE_BETA: float = 0.3


def _infer_sf_projection_from_model_class(
    netam_config_dict: Dict[str, Any],
) -> SelectionFactorProjectionConfig | None:
    """Infer sf_projection config from the netam model class.

    netam wiggle model classes (e.g. TransformerBinarySelectionModelWiggleAct)
    apply wiggle(x, 0.3) to the prediction layer output. To reproduce the
    same forward pass in DASM2 we must set wiggle_beta=0.3 on the shared
    SelectionFactorProjection. Non-wiggle model classes get the default
    (wiggle_beta=None), so no sf_projection config is needed.
    """
    model_class = netam_config_dict.get("model_class", "")
    if model_class in _WIGGLE_MODEL_CLASSES:
        return SelectionFactorProjectionConfig(wiggle_beta=_NETAM_WIGGLE_BETA)
    if model_class:
        warnings.warn(
            f"Unrecognised netam model_class {model_class!r}; assuming "
            f"wiggle-free projection. Add to _WIGGLE_MODEL_CLASSES in "
            f"dasm2/models/netam_loader.py if this class uses wiggle(x, 0.3).",
            stacklevel=2,
        )
    return None


def _load_netam_config(yml_path: str) -> ModelConfig:
    """Load configuration from netam .yml file."""
    with open(yml_path, "r") as f:
        netam_config_dict = yaml.safe_load(f)  # Raw netam format dict

    # Convert netam YAML dict to validated Pydantic configs
    encoder_config = netam_to_encoder_config(netam_config_dict)
    head_config = create_default_dasm_head_config()
    sf_projection = _infer_sf_projection_from_model_class(netam_config_dict)

    # Create ModelConfig with architecture only (no training config)
    return ModelConfig(
        encoder=encoder_config,
        heads=[head_config],
        sf_projection=sf_projection,
    )


def netam_to_encoder_config(netam_config_dict: Dict[str, Any]) -> EncoderConfig:
    """Convert netam transformer config to DASM2 EncoderConfig.

    Note:
        netam models always use sinusoidal positional encoding (pos_encoding="sinusoidal")
        and standard softmax attention (entmax_alpha=None).
        For sparse attention models, use EncoderConfig(entmax_alpha=1.5).
    """
    try:
        model_params = netam_config_dict["model_hyperparameters"]

        # Handle netam's dual naming conventions: try preferred key first,
        # fall back to legacy key only if preferred is absent (not falsy)
        num_layers = (
            model_params["n_layers"]
            if "n_layers" in model_params
            else model_params["layer_count"]
        )
        dropout = (
            model_params["dropout_prob"]
            if "dropout_prob" in model_params
            else model_params["dropout"]
        )

        return EncoderConfig(
            nhead=model_params["nhead"],
            d_model_per_head=model_params["d_model_per_head"],
            num_layers=num_layers,
            dim_feedforward=model_params["dim_feedforward"],
            dropout=dropout,
            vocab_size=model_params["known_token_count"],
        )
    except KeyError as e:
        raise ValueError(
            f"Missing required netam config parameter: {e}. "
            f"Available keys: {list(netam_config_dict.get('model_hyperparameters', {}).keys())}"
        ) from e


def create_default_dasm_head_config() -> SelectionFactorHeadConfig:
    """Standard SF (DASM/Whichmut) evolutionary head configuration."""
    return SelectionFactorHeadConfig(
        input_dim=None,  # Will be auto-set by ModelConfig validator
    )
