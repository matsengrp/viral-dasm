"""Registry for structural embedding data (HERMES archives).

This module defines a YAML-based registry for structural embedding datasets,
following the same ``ManifestRegistry`` pattern used by the functional and
evolutionary registries.  Structural embedding manifests use the discriminator
``dasm2_structural_embedding_data_schema: 1`` and point to tar.gz archives
(or directories) containing pre-computed per-residue structural embeddings.

Example YAML::

    dasm2_structural_embedding_data_schema: 1
    description: >
      HERMES stability-tuned model embeddings on antibody chains only.
    data:
      path: ../../../data_root/.../hermes_...tar.gz
    model: hermes_py_050_ft_cdna117k_ddg_st
    variant: antibody_only
    last_updated: 2026-02-10

Usage::

    from dasm2.data.structural_embedding_registry import (
        list_structural_embeddings,
        resolve_structural_embedding_path,
    )

    print(list_structural_embeddings())
    path = resolve_structural_embedding_path("hermes_sabdab_ab_stability")
"""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dasm2.data.discovery import get_manifest_index
from dasm2.data.local import PathWithMD5, VerifiedPath
from dasm2.data.manifests import coerce_date_to_str


class StructuralEmbeddingConfig(BaseModel):
    """Configuration for a structural embedding dataset loaded from YAML.

    Attributes:
        dasm2_structural_embedding_data_schema: Discriminator key and schema
            version.  Must be present in every YAML file belonging to this
            registry.
        description: Human-readable dataset description.
        source_manifest: Optional path to the upstream source manifest.
        data: Path to the embedding archive (tar.gz) or directory, as a
            :class:`PathWithMD5` (accepts string or ``{path, md5}`` dict).
            Paths are relative to the YAML file location.
        model: HERMES model identifier
            (e.g. ``hermes_py_050_ft_cdna117k_ddg_st``).
        variant: Dataset variant (e.g. ``antibody_only``, ``full_complex``).
        last_updated: Optional ISO-format date string.
        yaml_dir: Directory containing the source YAML file (set during
            loading by :class:`ManifestRegistry`).
    """

    model_config = ConfigDict(extra="ignore")

    dasm2_structural_embedding_data_schema: int = Field(ge=1)
    description: Optional[str] = None
    source_manifest: Optional[str] = None
    data: PathWithMD5
    model: str
    variant: str
    last_updated: Optional[str] = None
    yaml_dir: Optional[Path] = Field(default=None, exclude=True)

    @field_validator("last_updated", mode="before")
    @classmethod
    def coerce_date(cls, v: object) -> Optional[str]:
        """Coerce ``datetime.date`` from YAML parsing to ISO string."""
        return coerce_date_to_str(v)

    def resolve_data_path(self) -> VerifiedPath:
        """Resolve the data path relative to the YAML file location.

        If the path is absolute it is used directly.  If relative,
        ``yaml_dir`` must be set (typically via
        :class:`ManifestRegistry` auto-discovery).

        Returns:
            :class:`VerifiedPath` with absolute path and optional MD5.

        Raises:
            ValueError: If path is relative and ``yaml_dir`` is not set.
        """
        data_path = Path(self.data.path)
        if data_path.is_absolute():
            return VerifiedPath(data_path.resolve(), self.data.md5)
        if self.yaml_dir is None:
            raise ValueError(
                f"Cannot resolve relative path '{self.data.path}': yaml_dir not set. "
                "Use absolute paths or load config via the registry."
            )
        return self.data.resolve(self.yaml_dir)


# ---------------------------------------------------------------------------
# Registry (delegates to unified ManifestIndex)
# ---------------------------------------------------------------------------

_DISCRIMINATOR = "dasm2_structural_embedding_data_schema"


def get_structural_embedding_config(name: str) -> StructuralEmbeddingConfig:
    """Get a registered structural embedding configuration by name.

    Args:
        name: Registry name (YAML file stem, e.g.
            ``"hermes_sabdab_ab_stability"``).

    Returns:
        The dataset configuration.

    Raises:
        KeyError: If no dataset with this name is registered.
    """
    config: StructuralEmbeddingConfig = get_manifest_index().get_typed(
        name, _DISCRIMINATOR
    )
    return config


def resolve_structural_embedding_path(name: str) -> Path:
    """Resolve a structural embedding name to a filesystem path.

    Resolution order:

    1. Direct path -- if *name* is an existing file or directory it is
       returned as-is.
    2. Registry lookup -- look up the name in the unified manifest index
       and resolve the ``data`` field.

    Args:
        name: Registry name, directory path, or tar.gz path.

    Returns:
        Absolute path to the embedding archive or directory.

    Raises:
        FileNotFoundError: If the name cannot be resolved.
    """
    # Try as direct path first (backward compatibility)
    direct = Path(name).expanduser()
    if direct.exists():
        return direct.resolve()

    # Registry lookup
    index = get_manifest_index()
    typed_config: Optional[StructuralEmbeddingConfig] = index.get_typed_optional(
        name, _DISCRIMINATOR
    )
    if typed_config is not None:
        return typed_config.resolve_data_path().raw_path

    # Not found
    available = list_structural_embeddings()
    available_display = ", ".join(available[:10])
    if len(available) > 10:
        available_display += f", ... ({len(available)} total)"

    raise FileNotFoundError(
        f"Structural embedding '{name}' not found.\n"
        f"Available: {available_display or '(none)'}\n"
        f"Or provide a valid path to a directory or tar.gz file."
    )


def list_structural_embeddings() -> List[str]:
    """List all registered structural embedding names.

    Returns:
        Sorted list of registered names.
    """
    return get_manifest_index().list_names(_DISCRIMINATOR)


def is_structural_embedding_available(name: str) -> bool:
    """Check whether a structural embedding is registered.

    Args:
        name: Registry name to check.

    Returns:
        ``True`` if the name exists in the registry, ``False`` otherwise.
    """
    return get_manifest_index().get_typed_optional(name, _DISCRIMINATOR) is not None
