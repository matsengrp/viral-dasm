"""YAML-based dataset registry for evolutionary PCP data.

This module provides backward-compatible convenience functions for looking up
evolutionary PCP datasets (``dasm2_pcp_data_schema``).  Under the hood,
everything delegates to the unified :class:`~dasm2.data.discovery.ManifestIndex`
which performs a single filesystem scan across all data types.

Usage::

    from dasm2.data.yaml_registry import get_registry

    registry = get_registry()
    config = registry.get("v1tang")
    print(config.data.path)
"""

from pathlib import Path
from typing import List

from dasm2.data.discovery import (
    ScanPath,
    TypedManifestView,
    create_manifest_index,
    get_manifest_index,
    reset_manifest_index,
)
from dasm2.data.registry_schema import ResolvedDatasetConfig

_DISCRIMINATOR = "dasm2_pcp_data_schema"


def get_registry() -> TypedManifestView[ResolvedDatasetConfig]:
    """Get the global evolutionary PCP dataset registry.

    Returns a view of the unified :class:`ManifestIndex` filtered to
    ``dasm2_pcp_data_schema`` entries.

    Test data and CWD are scanned non-recursively (flat). User-configured
    registry paths are expanded to include all subdirectories so that
    manifests organized into folders (e.g. ``evolutionary/``, ``viral/``)
    are discovered automatically.

    Returns:
        A :class:`TypedManifestView` with the same interface as
        the previous ``ManifestRegistry[ResolvedDatasetConfig]``.
    """
    return TypedManifestView(get_manifest_index(), _DISCRIMINATOR)


def create_dataset_registry(
    scan_paths: List[Path],
) -> TypedManifestView[ResolvedDatasetConfig]:
    """Create a dataset registry with explicit scan paths (for CLI/tests).

    Args:
        scan_paths: Directories to scan for YAML files (non-recursive).

    Returns:
        A :class:`TypedManifestView` filtered to evolutionary PCP datasets.
    """
    index = create_manifest_index([ScanPath(p, recursive=False) for p in scan_paths])
    return TypedManifestView(index, _DISCRIMINATOR)


def reset_registry() -> None:
    """Reset the global registry.

    Forces re-discovery of YAML files on next ``get_registry()`` call.
    Useful for testing.
    """
    reset_manifest_index()
