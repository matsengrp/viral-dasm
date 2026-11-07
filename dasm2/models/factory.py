"""Model factory for creating complete DASM2 models from configuration."""

from typing import Dict, Iterable, List, Type, Union, cast

import torch

from dasm2.config import (
    EncoderConfig,
    HeadConfig,
    ModelConfig,
    SelectionFactorProjectionConfig,
)
from dasm2.heads.factory import HEAD_REGISTRY
from dasm2.heads.types import HeadType
from dasm2.models.core.base import CoreModel
from dasm2.models.core.sf_projection import SelectionFactorProjection
from dasm2.models.core.transformer import TransformerEncoder

#: One encoder. ESM2, MPNN light-attention and the identity passthrough went
#: with their core models (#4); the paper trains the codon transformer.
ENCODER_REGISTRY: Dict[str, Type[torch.nn.Module]] = {
    "TransformerEncoder": TransformerEncoder,
}


def create_model_from_config(
    model_config: ModelConfig, device: str = "cuda"
) -> CoreModel:
    """Create a complete model from configuration.

    Uses ESM2CoreModel when encoder is ESM2EncoderConfig,
    MPNNCoreModel when encoder is MPNNLightAttentionEncoderConfig,
    otherwise uses CoreModel (handles standard and concat-pos encoders).

    Args:
        model_config: Model configuration with encoder and heads
        device: Device to place model on ("cuda" or "cpu")

    Returns:
        Initialized CoreModel (or subclass) with encoder and prediction heads

    Raises:
        ValueError: If configuration is invalid or heads cannot be created
    """
    encoder = _create_encoder_from_config(model_config.encoder, device)

    # Create SF projection if configured or if SF-consuming heads are present
    sf_projection = _create_sf_projection(model_config.sf_projection, model_config)

    # One core model. The site-masked, MLM, ESM2 and MPNN variants went with
    # the heads and encoders that selected them (#4).
    model = CoreModel(
        encoder=encoder,
        prediction_heads=model_config.heads,
        sf_projection=sf_projection,
    )

    model.to(device)
    return model


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
    encoder_config: EncoderConfig,
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
