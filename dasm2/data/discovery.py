"""Unified manifest discovery for all DASM2 data types.

Provides a single :class:`ManifestIndex` that scans the filesystem once
and routes YAML files to the appropriate config parser based on discriminator
keys.  This replaces the redundant per-type filesystem scanning that was
previously done by each registry module independently.

Adding a new data type requires only:

1. A Pydantic config class (in its own module).
2. One :class:`DataTypeDescriptor` entry in :func:`_build_descriptors`.

Usage::

    from dasm2.data.discovery import get_manifest_index

    index = get_manifest_index()

    # Look up by name across all types
    manifest = index.get("v1tang50k")
    print(manifest.discriminator)  # "dasm2_pcp_data_schema"

    # List all datasets of a specific type
    viral_names = index.list_names("dasm2_viral_pcp_data_schema")

    # Resolve data path for any dataset
    path = index.resolve_data_path("s2Spike21K-2022-06-01-v1")
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import yaml
from pydantic import BaseModel, ValidationError

from dasm2.data.local import get_registry_paths, get_test_data_dir, get_xdg_config_dir
from dasm2.data.manifests import resolve_extends

logger = logging.getLogger(__name__)

_C = TypeVar("_C", bound=BaseModel)

# Hard wall-clock bound for scanning a single ScanPath (see
# ManifestIndex._scan_one_path). registry_paths can point at a slow or
# unresponsive network mount. Calibrated empirically (#1008): a real,
# eventually-successful recursive scan of the sshfs-mounted /fh/fast
# registry_paths entry took 162s under ordinary pax load (9% CPU, almost
# entirely I/O wait) — a much shorter bound would misfire on legitimate
# slow-but-working access, turning "slow" into "wrong" (a stale/incomplete
# index) rather than "eventually correct". 240s gives that real case
# ~50% headroom while still bounding the true-hang case (which is what this
# guard exists to stop) to a known, finite wait.
#
# This budget is PER scan path, not a total across the whole scan — with
# today's single-entry registry_paths that's moot, but a user config with
# several slow entries could in principle accumulate past pyproject.toml's
# global per-test pytest-timeout bound (300s) before this guard even gets a
# chance to skip the later ones, triggering that harsher hard-abort instead
# of a clean per-path skip. Not addressed here since it doesn't apply to any
# known real config; if it ever does, make this a total deadline shared
# across all of _scan's paths rather than resetting per path.
_SCAN_PATH_TIMEOUT_SECONDS = 240.0


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataTypeDescriptor:
    """Bundles type-specific implementation details for a manifest type.

    Each discriminator maps to a descriptor that tells the unified scanner
    how to parse, transform, and display manifests of that type.

    Attributes:
        discriminator: YAML key that identifies files of this type
            (e.g. ``"dasm2_viral_pcp_data_schema"``).
        display_name: Human-readable name for CLI grouping
            (e.g. ``"Viral PCP"``).
        config_class: Pydantic model class for validation.
        post_load: Optional transform applied after initial parsing.
            Signature: ``(raw_config, name, yaml_path) -> config``.
        data_path_resolver: Optional callable to extract data path from a
            parsed config.  Signature: ``(config) -> Path``.
        size_reader: Optional callable to measure dataset size for CLI display.
            Signature: ``(data_path, exact) -> Optional[SizeResult]``.
            When ``None``, the CLI falls back to the default line-count reader.
    """

    discriminator: str
    display_name: str
    config_class: Type[BaseModel]
    post_load: Optional[Callable[[Any, str, Path], Any]] = None
    data_path_resolver: Optional[Callable[[Any], Path]] = None
    size_reader: Optional[Callable] = None


@dataclass
class DiscoveredManifest:
    """A YAML file that matched a known discriminator.

    Attributes:
        name: Filename stem used as lookup key.
        discriminator: Which discriminator matched.
        yaml_path: Absolute path to the YAML file.
        config: Parsed and validated config object.
    """

    name: str
    discriminator: str
    yaml_path: Path
    config: Any  # BaseModel subclass; specific type depends on discriminator


@dataclass(frozen=True)
class ScanPath:
    """A directory to scan with recursive/non-recursive control.

    Attributes:
        path: Directory path to scan.
        recursive: If ``True``, use ``rglob`` to find YAML files in all
            subdirectories.  If ``False``, use ``glob`` for immediate
            children only.
    """

    path: Path
    recursive: bool = True


# ---------------------------------------------------------------------------
# ManifestIndex
# ---------------------------------------------------------------------------


class ManifestIndex:
    """Unified index over all manifest types.

    Scans filesystem paths once and routes each YAML file to the appropriate
    config parser based on discriminator keys.  Names must be unique across
    **all** manifest types.

    Args:
        descriptors: Map of discriminator key to :class:`DataTypeDescriptor`.
        scan_paths: Directories to scan.  Each :class:`ScanPath` controls
            whether scanning is recursive.  Plain :class:`Path` objects
            are also accepted and will be scanned recursively.
    """

    def __init__(
        self,
        descriptors: Dict[str, DataTypeDescriptor],
        scan_paths: List[ScanPath],
    ) -> None:
        self._descriptors = descriptors
        self._index: Dict[str, DiscoveredManifest] = {}
        self._skipped_scan_paths: List[Path] = []
        self._scan(scan_paths)

    # -- Public API ---------------------------------------------------------

    def get(self, name: str) -> DiscoveredManifest:
        """Look up by name across all registries.

        Args:
            name: Dataset / manifest name (YAML file stem).

        Returns:
            The discovered manifest.

        Raises:
            KeyError: If *name* is not registered. If a scan path timed out
                during index construction (see :attr:`skipped_scan_paths`),
                the message names it — *name* may be defined there but
                wasn't discovered.
        """
        if name not in self._index:
            available = sorted(self._index.keys())
            available_display = ", ".join(available[:10])
            if len(available) > 10:
                available_display += f", ... ({len(available)} total)"
            skipped_note = ""
            if self._skipped_scan_paths:
                skipped_list = ", ".join(str(p) for p in self._skipped_scan_paths)
                skipped_note = (
                    f"\nNote: scanning timed out and was skipped for: "
                    f"{skipped_list} — '{name}' may be defined there but "
                    f"wasn't discovered this build."
                )
            raise KeyError(
                f"'{name}' not found in manifest index.\n"
                f"Available: {available_display or '(none)'}"
                f"{skipped_note}"
            )
        return self._index[name]

    @property
    def skipped_scan_paths(self) -> List[Path]:
        """Scan paths that timed out and were skipped during this index build."""
        return list(self._skipped_scan_paths)

    def get_typed(self, name: str, discriminator: str) -> Any:
        """Look up by name within a specific discriminator.

        Args:
            name: Dataset / manifest name.
            discriminator: Expected discriminator key.

        Returns:
            The config object (typed for the discriminator).

        Raises:
            KeyError: If *name* is not found or has the wrong discriminator.
        """
        manifest = self.get(name)
        if manifest.discriminator != discriminator:
            desc = self._descriptors[manifest.discriminator]
            raise KeyError(
                f"'{name}' is a {desc.display_name} dataset "
                f"(discriminator: {manifest.discriminator}), "
                f"not {discriminator}"
            )
        return manifest.config

    def get_optional(self, name: str) -> Optional[DiscoveredManifest]:
        """Look up by name, returning ``None`` if not found."""
        return self._index.get(name)

    def get_typed_optional(self, name: str, discriminator: str) -> Optional[Any]:
        """Look up by name within a discriminator, returning ``None`` if not found."""
        manifest = self._index.get(name)
        if manifest is None or manifest.discriminator != discriminator:
            return None
        return manifest.config

    def list_names(self, discriminator: Optional[str] = None) -> List[str]:
        """List all names, optionally filtered by discriminator.

        Args:
            discriminator: If provided, only names matching this type.

        Returns:
            Sorted list of names.
        """
        if discriminator is None:
            return sorted(self._index.keys())
        return sorted(
            name for name, m in self._index.items() if m.discriminator == discriminator
        )

    def resolve_data_path(self, name: str) -> Path:
        """Resolve name to a data file path across all registries.

        Uses the ``data_path_resolver`` from the matching
        :class:`DataTypeDescriptor`.

        Args:
            name: Dataset / manifest name.

        Returns:
            Absolute path to the data file.

        Raises:
            KeyError: If *name* is not found.
            ValueError: If the descriptor has no ``data_path_resolver``.
        """
        manifest = self.get(name)
        desc = self._descriptors[manifest.discriminator]
        if desc.data_path_resolver is None:
            raise ValueError(
                f"No data path resolver configured for "
                f"{desc.display_name} ({desc.discriminator})"
            )
        resolved: Path = desc.data_path_resolver(manifest.config)
        return resolved

    def register(
        self,
        name: str,
        discriminator: str,
        config: Any,
        yaml_path: Optional[Path] = None,
    ) -> None:
        """Manually register a manifest entry.

        Args:
            name: Unique name.
            discriminator: Discriminator key.
            config: Config object.
            yaml_path: Optional path to YAML file.

        Raises:
            ValueError: If *name* is already registered.
        """
        if name in self._index:
            raise ValueError(f"'{name}' is already registered in the manifest index")
        self._index[name] = DiscoveredManifest(
            name=name,
            discriminator=discriminator,
            yaml_path=yaml_path or Path("<manual>"),
            config=config,
        )

    @property
    def descriptors(self) -> Dict[str, DataTypeDescriptor]:
        """Access the registered data type descriptors."""
        return self._descriptors

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def __len__(self) -> int:
        return len(self._index)

    # -- Internals ----------------------------------------------------------

    def _scan(self, scan_paths: List[ScanPath]) -> None:
        """Single scan across all paths (recursive or not per ScanPath).

        Each path is scanned via :meth:`_scan_one_path`, which bounds it to
        ``_SCAN_PATH_TIMEOUT_SECONDS``; a path that times out is skipped
        (logged) rather than hanging this method indefinitely.
        """
        name_to_files: Dict[str, List[Path]] = {}

        for sp in scan_paths:
            try:
                found = self._scan_one_path(sp)
            except TimeoutError:
                # registry_paths (dasm2/data/local.py, from the user's
                # ~/.config/dasm2/config.yaml) can point at a slow or
                # unresponsive network mount (e.g. an sshfs mount that stalls
                # on directory walks). Without this guard, a single bad mount
                # hangs any process that lazily builds the global
                # ManifestIndex (get_manifest_index()) — including pytest
                # runs and `dasm2.cli` invocations — indefinitely, with no
                # way to distinguish "still scanning" from "stuck forever".
                # See #827 (test-layer isolation fixture) and #1008 (this
                # production-side guard; reproduced hanging at two different
                # points in a full-suite run via this exact code path).
                logger.warning(
                    "Registry scan path %s did not respond within %.0fs "
                    "(possibly a slow or unresponsive network mount); "
                    "skipping it for this index build.",
                    sp.path,
                    _SCAN_PATH_TIMEOUT_SECONDS,
                )
                self._skipped_scan_paths.append(sp.path)
                continue
            except PermissionError as exc:
                # Same root cause as the TimeoutError case above (a flaky
                # network mount), different symptom: a read-only sshfs mount
                # that's degraded/reconnecting can raise PermissionError
                # ("Operation not permitted") on a directory walk instead of
                # hanging (observed scanning a read-only /fh/fast sshfs mount
                # from a workstation, issue #814). Deliberately narrower than
                # OSError: a misconfigured registry_paths entry (typo'd path,
                # a real local-permissions bug) should still surface as an
                # error rather than being silently attributed to "network
                # flakiness" and skipped.
                logger.warning(
                    "Registry scan path %s raised %s (possibly a slow or "
                    "unresponsive network mount); skipping it for this "
                    "index build.",
                    sp.path,
                    exc,
                )
                self._skipped_scan_paths.append(sp.path)
                continue
            for name, manifest, yaml_file in found:
                if name not in name_to_files:
                    name_to_files[name] = []
                    self._index[name] = manifest
                name_to_files[name].append(yaml_file)

        # Check for duplicates across all types
        duplicates = {n: files for n, files in name_to_files.items() if len(files) > 1}
        if duplicates:
            dup_messages = []
            for n, files in sorted(duplicates.items()):
                file_list = "\n    ".join(str(f) for f in files)
                dup_messages.append(f"  '{n}' defined in:\n    {file_list}")
            raise ValueError(
                "Duplicate manifest names found across registry paths:\n"
                + "\n".join(dup_messages)
            )

    def _scan_one_path(
        self, sp: ScanPath
    ) -> List[Tuple[str, DiscoveredManifest, Path]]:
        """Scan a single path for YAML manifests, bounded by a hard timeout.

        Runs the filesystem walk (``exists`` + ``glob``/``rglob``, plus
        per-file loading) on a daemon thread and gives it
        ``_SCAN_PATH_TIMEOUT_SECONDS`` to finish. Blocking filesystem
        syscalls release the GIL, so a stuck thread doesn't stall anything
        else; using a daemon thread (rather than e.g. a
        ``ThreadPoolExecutor``, whose non-daemon workers are joined at
        interpreter exit) means an abandoned scan can't hang process exit
        either.

        Raises:
            TimeoutError: If the scan doesn't finish in time.
            Exception: Any exception raised while walking or parsing (e.g.
                ``yaml.YAMLError`` from a malformed manifest), re-raised with
                its original type and message once the scan thread finishes.
        """
        results: List[Tuple[str, DiscoveredManifest, Path]] = []
        # A single-element list, not a plain variable, so the nested
        # function below can record an exception via .append() without a
        # `nonlocal` rebind.
        error: List[Exception] = []

        def _run() -> None:
            try:
                results.extend(self._walk_and_load(sp))
            except Exception as exc:  # noqa: BLE001 - re-raised below
                error.append(exc)

        # Daemon so a scan that never returns (a genuinely, permanently
        # hung mount) can't block interpreter exit; each timeout leaves one
        # more thread parked on the blocked syscall for the process's
        # lifetime, which is an acceptable tradeoff for a guard whose whole
        # point is "don't hang forever" over "never leak a thread".
        thread = threading.Thread(
            target=_run, daemon=True, name=f"manifest-scan-{sp.path}"
        )
        thread.start()
        thread.join(timeout=_SCAN_PATH_TIMEOUT_SECONDS)
        if thread.is_alive():
            raise TimeoutError(
                f"scanning {sp.path} did not complete within "
                f"{_SCAN_PATH_TIMEOUT_SECONDS:.0f}s"
            )
        if error:
            raise error[0]
        return results

    def _walk_and_load(
        self, sp: ScanPath
    ) -> List[Tuple[str, DiscoveredManifest, Path]]:
        """Walk one ScanPath and load each matching YAML file.

        No timeout handling here — this is the part of
        :meth:`_scan_one_path` that actually does filesystem I/O, split out
        so it can be exercised directly without going through threading
        machinery.
        """
        results: List[Tuple[str, DiscoveredManifest, Path]] = []
        if not sp.path.exists():
            return results
        glob_method = sp.path.rglob if sp.recursive else sp.path.glob
        for yaml_file in sorted(glob_method("*.yaml")):
            loaded = self._try_load_file(yaml_file)
            if loaded is not None:
                name, manifest = loaded
                results.append((name, manifest, yaml_file))
        return results

    def _try_load_file(
        self, yaml_file: Path
    ) -> Optional[Tuple[str, DiscoveredManifest]]:
        """Attempt to load a single YAML file, routing by discriminator.

        Returns ``(name, manifest)`` on success, or ``None`` if no known
        discriminator matched.
        """
        resolved_path = yaml_file.resolve()

        try:
            with open(yaml_file) as f:
                raw_data = yaml.safe_load(f)
        except yaml.YAMLError:
            raise  # fail fast

        if not isinstance(raw_data, dict):
            return None

        # Find matching discriminator
        for disc_key, descriptor in self._descriptors.items():
            if disc_key not in raw_data:
                continue

            # Discriminator matched — non-validation failures are real errors
            name = resolved_path.stem
            try:
                data = resolve_extends(raw_data, resolved_path, disc_key)
                config = descriptor.config_class(**data)
            except ValidationError as exc:
                short = "; ".join(f"{e['loc']}: {e['msg']}" for e in exc.errors())
                logger.warning(
                    "Skipping %s: validation error for discriminator '%s': %s",
                    yaml_file,
                    disc_key,
                    short,
                )
                return None

            # Set yaml_dir if the model has the field
            if "yaml_dir" in type(config).model_fields:
                config.yaml_dir = resolved_path.parent  # type: ignore[attr-defined]

            # Apply post_load transform if provided
            if descriptor.post_load is not None:
                config = descriptor.post_load(config, name, resolved_path)

            return (
                name,
                DiscoveredManifest(
                    name=name,
                    discriminator=disc_key,
                    yaml_path=resolved_path,
                    config=config,
                ),
            )

        return None  # No matching discriminator


# ---------------------------------------------------------------------------
# TypedManifestView — drop-in replacement for per-type ManifestRegistry usage
# ---------------------------------------------------------------------------


class TypedManifestView(Generic[_C]):
    """Read-through view of a :class:`ManifestIndex` filtered to one discriminator.

    Provides the same duck-type interface as
    :class:`~dasm2.data.manifests.ManifestRegistry` so existing code that calls
    ``registry.get(name)`` / ``name in registry`` / ``registry.list_names()``
    continues to work unchanged.

    Args:
        index: The backing manifest index.
        discriminator: Discriminator to filter on.
    """

    def __init__(self, index: ManifestIndex, discriminator: str) -> None:
        self._index = index
        self._discriminator = discriminator

    def get(self, name: str) -> _C:
        """Look up a config by name within this type.

        Provides a helpful error if *name* exists under a different discriminator.

        Args:
            name: Dataset / manifest name.

        Returns:
            The config.

        Raises:
            KeyError: If not found or wrong discriminator.
        """
        config: Optional[_C] = self._index.get_typed_optional(name, self._discriminator)
        if config is not None:
            return config

        # Check if name exists under a different type
        manifest = self._index.get_optional(name)
        if manifest is not None:
            other_desc = self._index.descriptors[manifest.discriminator]
            raise KeyError(
                f"'{name}' is a {other_desc.display_name} dataset "
                f"(discriminator: {manifest.discriminator}), "
                f"not available in this registry ({self._discriminator})."
            )

        # Not found anywhere
        available = self.list_names()
        available_display = ", ".join(available[:10])
        if len(available) > 10:
            available_display += f", ... ({len(available)} total)"
        raise KeyError(
            f"'{name}' not found in registry.\n"
            f"Available: {available_display or '(none)'}"
        )

    def get_optional(self, name: str) -> Optional[_C]:
        """Look up by name, returning ``None`` if not found or wrong type."""
        return self._index.get_typed_optional(name, self._discriminator)  # type: ignore[return-value]

    def list_names(self) -> List[str]:
        """Return sorted list of all names in this type."""
        return self._index.list_names(self._discriminator)

    def register(self, name: str, config: _C) -> None:
        """Manually register a config entry."""
        self._index.register(name, self._discriminator, config)

    def __contains__(self, name: str) -> bool:
        return self._index.get_typed_optional(name, self._discriminator) is not None

    def __len__(self) -> int:
        return len(self.list_names())


# ---------------------------------------------------------------------------
# Data type descriptors (built lazily to avoid circular imports)
# ---------------------------------------------------------------------------


def _build_descriptors() -> Dict[str, DataTypeDescriptor]:
    """Build the descriptor registry.

    One descriptor. viral-DASM resolves nothing from a manifest -- every path
    it reads is written out in ``config.yaml`` -- so the viral-PCP, fitness,
    functional and structural-embedding registries went with the datasets
    they described (#4). What remains is the evolutionary PCP schema, which
    the Parquet loader still consults to name a run.

    Uses a late import to avoid a circular dependency with the config module.
    """
    from dasm2.data.registry_schema import DatasetConfig, ResolvedDatasetConfig

    def _resolve_dataset_config(
        raw: DatasetConfig, name: str, yaml_path: Path
    ) -> ResolvedDatasetConfig:
        return ResolvedDatasetConfig.from_config(raw, name, yaml_path)

    def _evol_data_path(config: ResolvedDatasetConfig) -> Path:
        return config.data.raw_path

    def _generic_data_path(config: Any) -> Path:
        resolved: Path = config.resolve_data_path().raw_path
        return resolved

    return {
        "dasm2_pcp_data_schema": DataTypeDescriptor(
            discriminator="dasm2_pcp_data_schema",
            display_name="Evolutionary PCP",
            config_class=DatasetConfig,
            post_load=_resolve_dataset_config,
            data_path_resolver=_evol_data_path,
        ),
    }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


def _get_default_scan_paths() -> List[ScanPath]:
    """Get unified scan paths for all manifest types.

    Returns:
        List of :class:`ScanPath` entries in scan order:
        1. Package test data directory (non-recursive)
        2. Current working directory (non-recursive)
        3. User-configured ``registry_paths`` (recursive)
        4. Legacy ``~/.dasm2/functional_datasets`` (non-recursive, if exists)
    """
    paths: List[ScanPath] = []

    # 1. Package test data directory (non-recursive)
    test_data_dir = get_test_data_dir()
    if test_data_dir.exists():
        paths.append(ScanPath(test_data_dir, recursive=False))

    # 2. Current working directory (non-recursive)
    paths.append(ScanPath(Path.cwd(), recursive=False))

    # 3. User-configured paths (recursive — this is where data-central lives)
    for p in get_registry_paths():
        paths.append(ScanPath(p, recursive=True))

    # 4. Legacy functional_datasets dir (non-recursive) — check both XDG and legacy
    xdg_func_dir = get_xdg_config_dir() / "functional_datasets"
    legacy_func_dir = Path.home() / ".dasm2" / "functional_datasets"
    for func_dir in (xdg_func_dir, legacy_func_dir):
        if func_dir.exists():
            paths.append(ScanPath(func_dir, recursive=False))
            break

    return paths


_GLOBAL_INDEX: Optional[ManifestIndex] = None


def get_manifest_index() -> ManifestIndex:
    """Get the global manifest index (lazy-loaded singleton).

    The index is built on first access by scanning all configured paths
    recursively.  All YAML files are read once and routed to the appropriate
    config parser by discriminator key.

    If a scan path timed out (see ``ManifestIndex._scan_one_path``,
    ``_SCAN_PATH_TIMEOUT_SECONDS``, #1008), the resulting *degraded* index —
    missing whatever that path would have contributed — is still cached
    here, deliberately: an immediate retry would just re-pay the same
    timeout against a mount that's already shown itself to be slow or
    unresponsive. Call :func:`reset_manifest_index` to force a fresh scan
    once the underlying mount is known to be healthy again.

    Returns:
        The global :class:`ManifestIndex`.
    """
    global _GLOBAL_INDEX
    if _GLOBAL_INDEX is None:
        _GLOBAL_INDEX = ManifestIndex(
            descriptors=_build_descriptors(),
            scan_paths=_get_default_scan_paths(),
        )
    return _GLOBAL_INDEX


def create_manifest_index(scan_paths: List[ScanPath]) -> ManifestIndex:
    """Create a manifest index with explicit scan paths.

    Useful for tests and CLI that need custom discovery paths.

    Args:
        scan_paths: Directories to scan (as :class:`ScanPath` objects).

    Returns:
        A new :class:`ManifestIndex` (not the global singleton).
    """
    return ManifestIndex(
        descriptors=_build_descriptors(),
        scan_paths=scan_paths,
    )


def reset_manifest_index() -> None:
    """Reset the global manifest index.

    Forces re-scan on next :func:`get_manifest_index` call.
    Useful for testing.
    """
    global _GLOBAL_INDEX
    _GLOBAL_INDEX = None
