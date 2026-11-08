"""Generic YAML manifest registry for DASM2 data sources.

This module provides :class:`ManifestRegistry`, a reusable generic registry
that discovers, validates, and caches Pydantic-modelled YAML manifests from
the filesystem.  It also exposes shared utility functions used by all
manifest-based registries (viral fitness, functional, PCP datasets).

Two construction modes:

* **Lazy** (``discovery_paths_fn``): Paths are resolved and scanned on first
  access (``get``, ``list_names``, ``__contains__``, ``__len__``).
* **Eager** (``scan_paths``): Paths are scanned immediately at construction.

Example::

    from dasm2.data.manifests import ManifestRegistry

    registry: ManifestRegistry[MyConfig] = ManifestRegistry(
        discriminator_key="my_schema",
        config_class=MyConfig,
        discovery_paths_fn=lambda: [Path(".")],
    )
    config = registry.get("my-dataset")
"""

import datetime
import logging
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import yaml
from pydantic import BaseModel

from dasm2.data.local import PathWithMD5, VerifiedPath

logger = logging.getLogger(__name__)

_C = TypeVar("_C", bound=BaseModel)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def coerce_date_to_str(v: Any) -> Optional[str]:
    """Coerce ``datetime.date`` (from YAML parsing) to ISO-format string.

    Args:
        v: Value to coerce – may be ``None``, a ``datetime.date``, or a string.

    Returns:
        ISO-format date string, or ``None`` if *v* is ``None``.
    """
    if v is None:
        return None
    if isinstance(v, datetime.date):
        return v.isoformat()
    return str(v)


def resolve_extends(
    data: Dict[str, Any], yaml_path: Path, discriminator_key: str
) -> Dict[str, Any]:
    """Resolve ``extends`` inheritance for a YAML config dict.

    Loads the base manifest, merges base fields as defaults, then overlays
    child-specific fields.  Chained extends (base extending another base)
    are not supported.

    Args:
        data: Raw YAML dict that may contain an ``extends`` field.
        yaml_path: Absolute path to the YAML file containing *data*.
        discriminator_key: Schema discriminator key name (e.g.,
            ``"dasm2_fitness_data_schema"``).

    Returns:
        Merged dict with base fields inherited and ``extends`` cleared.

    Raises:
        ValueError: If the extends target is not found or itself extends.
    """
    extends = data.get("extends")
    if not extends:
        return data

    base_path = (yaml_path.parent / extends).resolve()
    if not base_path.exists():
        raise ValueError(
            f"extends target '{extends}' not found "
            f"(resolved to {base_path}) in {yaml_path}"
        )

    with open(base_path) as f:
        base_data = yaml.safe_load(f)

    if not isinstance(base_data, dict):
        raise ValueError(f"Base manifest {base_path} is not a valid YAML dict")

    if base_data.get("extends"):
        raise ValueError(
            f"Chained extends not allowed: {yaml_path} extends {base_path} "
            f"which itself extends {base_data['extends']}"
        )

    # Base fields provide defaults; child fields override
    merged: Dict[str, Any] = {}
    for key, value in base_data.items():
        if key in (discriminator_key, "extends"):
            continue
        if value is not None:
            merged[key] = value
    for key, value in data.items():
        if key == "extends":
            continue
        if value is not None:
            merged[key] = value
    # Always use child's discriminator
    merged[discriminator_key] = data[discriminator_key]
    return merged


def resolve_path_with_md5(
    path_field: Optional[PathWithMD5],
    yaml_dir: Optional[Path],
    field_name: str,
) -> VerifiedPath:
    """Resolve a :class:`PathWithMD5` field relative to the YAML directory.

    Args:
        path_field: ``PathWithMD5`` value to resolve.
        yaml_dir: Directory of source YAML (``None`` if not set via loader).
        field_name: Name of the field being resolved (for error messages).

    Returns:
        :class:`VerifiedPath` with absolute path and optional MD5.

    Raises:
        ValueError: If *path_field* is ``None`` or cannot be resolved.
    """
    if path_field is None:
        raise ValueError(f"No {field_name} path specified in config")
    if yaml_dir is None:
        file_path = Path(path_field.path)
        if file_path.is_absolute():
            return VerifiedPath(file_path.resolve(), path_field.md5)
        raise ValueError(
            f"Cannot resolve relative path '{path_field.path}': yaml_dir not set. "
            "Load config via the appropriate load function."
        )
    return path_field.resolve(yaml_dir)


def load_yaml_config(
    yaml_path: Path,
    discriminator_key: str,
    config_class: Type[_C],
) -> _C:
    """Load a YAML config with ``extends`` inheritance.

    Args:
        yaml_path: Resolved absolute path to the YAML file.
        discriminator_key: Expected schema discriminator key name.
        config_class: Pydantic model class to instantiate.

    Returns:
        Instantiated config with ``yaml_dir`` set (if the model has the field).

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        ValueError: If the YAML is invalid or missing the discriminator.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: expected dict, got {type(data)}")

    if discriminator_key not in data:
        raise ValueError(f"Missing discriminator '{discriminator_key}' in {yaml_path}.")

    data = resolve_extends(data, yaml_path, discriminator_key)
    config = config_class(**data)

    # Set yaml_dir if the model has the field
    if "yaml_dir" in type(config).model_fields:
        config.yaml_dir = yaml_path.parent  # type: ignore[attr-defined]

    return config


# ---------------------------------------------------------------------------
# ManifestRegistry
# ---------------------------------------------------------------------------


class ManifestRegistry(Generic[_C]):
    """Generic registry for Pydantic-modelled YAML manifests.

    Supports two construction modes:

    * **Lazy** (``discovery_paths_fn``): Paths are resolved and files scanned
      on first read access (``get``, ``list_names``, ``__contains__``,
      ``__len__``).
    * **Eager** (``scan_paths``): Files are scanned immediately during
      ``__init__``.

    Exactly one of *discovery_paths_fn* or *scan_paths* must be provided.

    Args:
        discriminator_key: YAML key that identifies files belonging to this
            registry (e.g. ``"dasm2_fitness_data_schema"``).
        config_class: Pydantic ``BaseModel`` subclass to validate YAML data.
        discovery_paths_fn: Callable returning a list of directories to scan
            (lazy mode).
        scan_paths: List of directories to scan immediately (eager mode).
        recursive: If ``True``, use ``rglob`` instead of ``glob`` for YAML
            discovery.  Default ``False``.
        post_load: Optional callback
            ``(raw_config, name, yaml_path) -> C`` applied after
            ``load_yaml_config``.  Useful to transform raw configs into
            resolved forms (e.g. ``ResolvedDatasetConfig``).
        on_error: Error handling policy for scanning.  ``"raise"`` (default)
            fails fast on validation errors and collects all duplicate names
            then raises a single ``ValueError``.  ``"skip"`` silently ignores
            validation errors **and** duplicates.
    """

    def __init__(
        self,
        discriminator_key: str,
        config_class: Type[BaseModel],
        *,
        discovery_paths_fn: Optional[Callable[[], List[Path]]] = None,
        scan_paths: Optional[List[Path]] = None,
        recursive: bool = False,
        post_load: Optional[Callable[[Any, str, Path], _C]] = None,
        on_error: Literal["skip", "raise"] = "raise",
    ) -> None:
        if (discovery_paths_fn is None) == (scan_paths is None):
            raise ValueError(
                "Exactly one of discovery_paths_fn or scan_paths must be provided"
            )

        self._discriminator_key = discriminator_key
        self._config_class = config_class
        self._discovery_paths_fn = discovery_paths_fn
        self._recursive = recursive
        self._post_load = post_load
        self._on_error = on_error

        self._registry: Dict[str, _C] = {}
        self._loaded = False

        # Eager mode: scan immediately
        if scan_paths is not None:
            self._scan(scan_paths)
            self._loaded = True

    # -- Public API ---------------------------------------------------------

    def get(self, name: str) -> _C:
        """Look up a config by name.

        Args:
            name: Dataset / manifest name (YAML file stem).

        Returns:
            The registered config.

        Raises:
            KeyError: If *name* is not registered.
        """
        self._ensure_loaded()
        if name not in self._registry:
            available = sorted(self._registry.keys())
            available_display = ", ".join(available[:10])
            if len(available) > 10:
                available_display += f", ... ({len(available)} total)"
            raise KeyError(
                f"'{name}' not found in registry.\n"
                f"Available: {available_display or '(none)'}"
            )
        return self._registry[name]

    def get_optional(self, name: str) -> Optional[_C]:
        """Look up a config by name, returning ``None`` if not found."""
        self._ensure_loaded()
        return self._registry.get(name)

    def list_names(self) -> List[str]:
        """Return a sorted list of all registered names."""
        self._ensure_loaded()
        return sorted(self._registry.keys())

    def register(self, name: str, config: _C) -> None:
        """Manually register a config.

        Does **not** trigger lazy loading, so callers can register entries
        before auto-discovery runs.

        Args:
            name: Unique name for the config.
            config: The config instance.

        Raises:
            ValueError: If *name* is already registered.
        """
        if name in self._registry:
            raise ValueError(f"'{name}' is already registered")
        self._registry[name] = config

    def clear(self) -> None:
        """Reset the registry, clearing all entries and the loaded flag."""
        self._registry.clear()
        self._loaded = False

    def __contains__(self, name: str) -> bool:
        self._ensure_loaded()
        return name in self._registry

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._registry)

    # -- Internals ----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Trigger lazy loading if not already done."""
        if self._loaded:
            return
        assert self._discovery_paths_fn is not None
        paths = self._discovery_paths_fn()
        self._scan(paths)
        self._loaded = True

    def _scan(self, scan_paths: List[Path]) -> None:
        """Scan directories for YAML files and populate the registry.

        Args:
            scan_paths: Directories to scan.
        """
        # Track duplicates for on_error="raise"
        name_to_files: Dict[str, List[Path]] = {}

        for scan_path in scan_paths:
            if not scan_path.exists():
                continue

            glob_method = scan_path.rglob if self._recursive else scan_path.glob
            yaml_files = sorted(glob_method("*.yaml"))

            for yaml_file in yaml_files:
                result = self._try_load_file(yaml_file)
                if result is None:
                    continue

                name, config = result

                if self._on_error == "raise":
                    # Track for duplicate detection
                    if name not in name_to_files:
                        name_to_files[name] = []
                    name_to_files[name].append(yaml_file)
                    self._registry[name] = config
                else:
                    # on_error="skip": first wins, silently ignore duplicates
                    if name not in self._registry:
                        self._registry[name] = config

        # Check for duplicates when on_error="raise"
        if self._on_error == "raise":
            duplicates = {
                n: files for n, files in name_to_files.items() if len(files) > 1
            }
            if duplicates:
                dup_messages = []
                for n, files in duplicates.items():
                    file_list = "\n    ".join(str(f) for f in files)
                    dup_messages.append(f"  '{n}' defined in:\n    {file_list}")
                raise ValueError(
                    "Duplicate dataset names found across registry paths:\n"
                    + "\n".join(dup_messages)
                )

    def _try_load_file(self, yaml_file: Path) -> Optional[Tuple[str, _C]]:
        """Attempt to load a single YAML file.

        Returns ``(name, config)`` on success, or ``None`` if the file should
        be skipped (not our type, or validation failed under ``on_error="skip"``).
        """
        resolved_path = yaml_file.resolve()

        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError:
            if self._on_error == "raise":
                raise
            return None

        if not isinstance(data, dict):
            return None

        if self._discriminator_key not in data:
            return None

        # File HAS our discriminator → apply on_error policy from here
        name = resolved_path.stem
        try:
            data = resolve_extends(data, resolved_path, self._discriminator_key)
            config = self._config_class(**data)

            # Set yaml_dir if the model has the field
            if "yaml_dir" in type(config).model_fields:
                config.yaml_dir = resolved_path.parent  # type: ignore[attr-defined]

            # Apply post_load transform if provided
            if self._post_load is not None:
                config = self._post_load(config, name, resolved_path)

            return (name, config)  # type: ignore[return-value]

        except Exception:
            if self._on_error == "raise":
                raise
            logger.warning(
                "Skipping %s due to validation error", yaml_file, exc_info=True
            )
            return None
