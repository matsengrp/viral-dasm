"""Model factory for creating complete DASM2 models from configuration."""

from typing import Dict, Iterable, List, Type, Union, cast

import torch

from dasm2.config import (
    EncoderConfig,
    ESM2EncoderConfig,
    HeadConfig,
    IdentityEncoderConfig,
    ModelConfig,
    MPNNLightAttentionEncoderConfig,
    SelectionFactorProjectionConfig,
)
from dasm2.heads.factory import HEAD_REGISTRY
from dasm2.heads.types import HeadType
from dasm2.models.core.base import CoreModel
from dasm2.models.core.esm2 import ESM2Encoder
from dasm2.models.core.esm2_core_model import ESM2CoreModel
from dasm2.models.core.identity import IdentityEncoder
from dasm2.models.core.mlm_core_model import MLMCoreModel
from dasm2.models.core.mpnn_core_model import MPNNCoreModel
from dasm2.models.core.mpnn_light_attention import MPNNLightAttentionEncoder
from dasm2.models.core.sf_projection import SelectionFactorProjection
from dasm2.models.core.site_masked_core_model import SiteMaskedCoreModel
from dasm2.models.core.transformer import TransformerEncoder

# Registry pattern for encoder types
ENCODER_REGISTRY: Dict[str, Type[torch.nn.Module]] = {
    "TransformerEncoder": TransformerEncoder,
    "ESM2Encoder": ESM2Encoder,
    "MPNNLightAttentionEncoder": MPNNLightAttentionEncoder,
    "IdentityEncoder": IdentityEncoder,
}


def create_model_from_config(
    model_config: ModelConfig, device: str = "cuda"
) -> CoreModel:
    """Create a complete model from configuration.

    Uses ESM2CoreModel when encoder is ESM2EncoderConfig,
    MPNNCoreModel when encoder is MPNNLightAttentionEncoderConfig,
    otherwise uses CoreModel (handles standard, concat-pos, and tdi encoders).

    Args:
        model_config: Model configuration with encoder and heads
        device: Device to place model on ("cuda" or "cpu")

    Returns:
        Initialized CoreModel (or subclass) with encoder and prediction heads

    Raises:
        ValueError: If configuration is invalid or heads cannot be created
    """
    # Validate head/encoder compatibility before building anything expensive
    # (ESM2/MPNN encoder construction can download pretrained weights).
    # MLMCoreModel reuses CoreModel.forward()/_build_encoder_input verbatim
    # (see its module docstring) and has no ESM2/MPNN-specific input
    # handling, so it only makes sense paired with a plain TransformerEncoder
    # (EncoderConfig) — reject any other encoder type explicitly rather than
    # silently building an MLMCoreModel that would feed MLM-vocabulary token
    # ids into an encoder built for a different vocabulary/input shape.
    is_mlm = needs_masked_marginal_core_model(model_config.heads)
    if is_mlm and not isinstance(model_config.encoder, EncoderConfig):
        raise ValueError(
            f"MLMHead (masked-marginal scoring) requires a plain "
            f"TransformerEncoder, got {type(model_config.encoder).__name__}. "
            f"MLMCoreModel has no ESM2/MPNN-specific encoder handling."
        )

    # Create encoder based on configuration
    encoder = _create_encoder_from_config(model_config.encoder, device)

    # Create SF projection if configured or if SF-consuming heads are present
    sf_projection = _create_sf_projection(model_config.sf_projection, model_config)

    # Select CoreModel class. Head-type dispatch takes priority over
    # encoder-type dispatch: MLM and site-masked configs use a plain
    # TransformerEncoder (the same encoder-config type any DASM config
    # uses), so encoder type alone can't tell their checkpoints from a DASM
    # one; only the head's requires_site_masked_core_model or
    # requires_masked_marginal_scoring flag can.
    if needs_site_masked_core_model(model_config.heads):
        # ModelConfig.validate_heads has already required this to be the only
        # head, on a TransformerEncoder with the mask token in its vocabulary.
        model: CoreModel = SiteMaskedCoreModel(
            encoder=encoder,
            prediction_heads=model_config.heads,
            sf_projection=sf_projection,
        )
    elif is_mlm:
        model = MLMCoreModel(
            encoder=encoder,
            prediction_heads=model_config.heads,
            sf_projection=sf_projection,
        )
    elif isinstance(model_config.encoder, ESM2EncoderConfig):
        model = ESM2CoreModel(
            encoder=encoder,
            prediction_heads=model_config.heads,
            sf_projection=sf_projection,
        )
    elif isinstance(model_config.encoder, MPNNLightAttentionEncoderConfig):
        model = MPNNCoreModel(
            encoder=encoder,
            prediction_heads=model_config.heads,
            sf_projection=sf_projection,
        )
    else:
        model = CoreModel(
            encoder=encoder,
            prediction_heads=model_config.heads,
            sf_projection=sf_projection,
        )

    # Apply ThermoMPNN warm-start if configured
    if (
        isinstance(model_config.encoder, MPNNLightAttentionEncoderConfig)
        and model_config.encoder.thermompnn_weights_path is not None
    ):
        _apply_thermompnn_warmstart(model, model_config)

    model.to(device)
    return model


def _apply_thermompnn_warmstart(model: CoreModel, model_config: ModelConfig) -> None:
    """Apply ThermoMPNN warm-start to encoder and WhichmutResidualHead(s).

    Args:
        model: Model with MPNNLightAttentionEncoder.
        model_config: Config with thermompnn_weights_path set.
    """
    from dasm2.config import WhichmutResidualHeadConfig
    from dasm2.heads.types import HeadType
    from dasm2.heads.whichmut_residual import WhichmutResidualHead
    from dasm2.models.core.mpnn_light_attention import MPNNLightAttentionEncoder

    encoder_config = model_config.encoder
    assert isinstance(encoder_config, MPNNLightAttentionEncoderConfig)
    assert encoder_config.thermompnn_weights_path is not None

    # Load encoder weights, get head output weights back
    assert isinstance(model.encoder, MPNNLightAttentionEncoder)
    head_weight, head_bias = model.encoder.load_thermompnn_weights(
        encoder_config.thermompnn_weights_path
    )

    # Apply to WhichmutResidualHead prediction layers
    for head_config in model_config.heads:
        if head_config.head_type != HeadType.WHICHMUT_RESIDUAL_HEAD:
            continue
        assert isinstance(head_config, WhichmutResidualHeadConfig)

        head = model.require_head(HeadType.WHICHMUT_RESIDUAL_HEAD)
        assert isinstance(head, WhichmutResidualHead)

        if head_config.use_residual:
            # Residual mode: delta=0 at init (already zero-init from constructor).
            pass
        else:
            # Non-residual: warm-start from ThermoMPNN output layer.
            head.prediction_layer.weight.data.copy_(head_weight[:20, :])
            assert head.prediction_layer.bias is not None
            head.prediction_layer.bias.data.copy_(head_bias[:20])


def head_types_need_sf_projection(head_types: Iterable[str]) -> bool:
    """Check if any raw head_type string consumes shared selection factors.

    Dict-metadata equivalent of :func:`needs_sf_projection`, for callers that
    only have raw ``head_type`` strings (e.g. unparsed checkpoint metadata,
    as in ``scripts/detect_wiggle_regression.py``) rather than constructed
    ``HeadConfig`` objects. Both derive from the same HEAD_REGISTRY +
    consumes_shared_sf ClassVar (no hardcoded set) — this is the shared core
    ``needs_sf_projection`` delegates to.

    Args:
        head_types: Raw head_type strings (``HeadType`` values also work,
            since ``HeadType`` is a ``str`` subclass).
    """
    for head_type in head_types:
        head_cls = HEAD_REGISTRY.get(cast(HeadType, head_type))
        if head_cls is not None and head_cls.consumes_shared_sf:
            return True
    return False


def needs_sf_projection(heads: List[HeadConfig]) -> bool:
    """Check if any head consumes shared selection factors.

    Derives from HEAD_REGISTRY + consumes_shared_sf ClassVar (no hardcoded set).

    Args:
        heads: List of head configurations to check.
    """
    return head_types_need_sf_projection(hc.head_type for hc in heads)


def head_types_need_masked_marginal_core_model(head_types: Iterable[str]) -> bool:
    """Check if any raw head_type string requires the masked-marginal adapter.

    Dict-metadata equivalent of :func:`needs_masked_marginal_core_model`,
    mirroring :func:`head_types_need_sf_projection`'s raw-string pattern for
    callers that only have unparsed ``head_type`` strings.

    Args:
        head_types: Raw head_type strings (``HeadType`` values also work,
            since ``HeadType`` is a ``str`` subclass).
    """
    for head_type in head_types:
        head_cls = HEAD_REGISTRY.get(cast(HeadType, head_type))
        if head_cls is not None and head_cls.requires_masked_marginal_scoring:
            return True
    return False


def needs_masked_marginal_core_model(heads: List[HeadConfig]) -> bool:
    """Check if any head requires the MLMCoreModel masked-marginal adapter.

    Derives from HEAD_REGISTRY + requires_masked_marginal_scoring ClassVar
    (no hardcoded head-type set) — mirrors needs_sf_projection's pattern.

    Args:
        heads: List of head configurations to check.
    """
    return head_types_need_masked_marginal_core_model(hc.head_type for hc in heads)


def needs_site_masked_core_model(heads: List[HeadConfig]) -> bool:
    """Check if any head requires the SiteMaskedCoreModel (issue #1550).

    Derives from HEAD_REGISTRY + requires_site_masked_core_model ClassVar,
    mirroring needs_masked_marginal_core_model.
    """
    for hc in heads:
        head_cls = HEAD_REGISTRY.get(hc.head_type)
        if head_cls is not None and head_cls.requires_site_masked_core_model:
            return True
    return False


def _create_sf_projection(
    config: Union[SelectionFactorProjectionConfig, None],
    model_config: ModelConfig,
) -> Union[SelectionFactorProjection, None]:
    """Create SelectionFactorProjection from config, or auto-create if needed.

    Auto-creates a default projection when SF-consuming heads are present
    but no explicit sf_projection config is given. This preserves backward
    compatibility with existing YAML configs that don't include sf_projection.

    Args:
        config: Explicit SF projection config, or None.
        model_config: Full model config (for auto-detection).

    Returns:
        Initialized SelectionFactorProjection, or None.
    """
    if config is not None:
        if config.input_dim is None:
            raise ValueError(
                "sf_projection.input_dim must be set "
                "(should be auto-set by ModelConfig.validate_heads)"
            )
        return SelectionFactorProjection(
            input_dim=config.input_dim,
            num_amino_acids=config.num_amino_acids,
            wiggle_beta=config.wiggle_beta,
            link=config.link,
        )

    # Auto-create if any head consumes shared SF
    if needs_sf_projection(model_config.heads):
        return SelectionFactorProjection(input_dim=model_config.encoder.d_model)

    return None


def _create_encoder_from_config(
    encoder_config: Union[
        EncoderConfig,
        ESM2EncoderConfig,
        MPNNLightAttentionEncoderConfig,
        IdentityEncoderConfig,
    ],
    device: str,
) -> torch.nn.Module:
    """Create encoder from configuration using registry pattern.

    Args:
        encoder_config: Encoder configuration
        device: Device to place encoder on

    Returns:
        Initialized encoder

    Raises:
        ValueError: If encoder type is unknown
    """
    encoder_class = ENCODER_REGISTRY.get(encoder_config.type)
    if encoder_class is None:
        available_encoders = list(ENCODER_REGISTRY.keys())
        raise ValueError(
            f"Unknown encoder type: '{encoder_config.type}'. "
            f"Available encoders: {available_encoders}"
        )

    return encoder_class(encoder_config)
