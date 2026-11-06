"""Local configuration and path expansion for DASM2.

This module handles user configuration and provides path expansion utilities.
The user config file is optional and supports:

- registry_paths: Additional directories to scan for dataset YAML files
- artifacts_dir: Directory for saving checkpoints, logs, etc.
- experiment_artifact_root: Root directory for experiment outputs
- remote_host: Remote server hostname for `make remote-*` commands
- remote_dir: Remote directory path for `make remote-*` commands
- phyz_binary: Absolute path to the phyz executable (cross-tool pipelines)
- superfamily_pcp_root: Path to the superfamily-pcp checkout (superfamily experiments)

Config file location (checked in order):

1. ``$XDG_CONFIG_HOME/dasm2/config.yaml`` (defaults to ``~/.config/dasm2/config.yaml``)
2. ``~/.dasm2/config.yaml`` (legacy fallback)

Example config.yaml::

    registry_paths:
      - /fh/fast/matsen_e/shared/dasm2-data/registry
      - ~/my-datasets/registry

    artifacts_dir: ~/dasm2-artifacts

    experiment_artifact_root: ~/dasm2-experiments

    remote_host: my-server
    remote_dir: /home/user/re/dasm2-experiments

    phyz_binary: /home/user/re/phyz/zig-out/bin/phyz

    superfamily_pcp_root: ~/re/superfamily-pcp
"""

import hashlib
import os
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class UserConfig(BaseModel):
    """User configuration from ``~/.config/dasm2/config.yaml``.

    Attributes:
        registry_paths: Additional directories to scan for dataset YAML files.
        artifacts_dir: Directory for saving checkpoints, logs, etc.
        experiment_artifact_root: Root directory for experiment outputs, read
            by :func:`dasm2.config.get_experiment_artifact_root`.
        remote_host: Remote server hostname for ``make remote-*`` commands.
        remote_dir: Remote directory path for ``make remote-*`` commands.
        phyz_binary: Absolute path to the phyz executable. Used by
            :func:`dasm2.io.find_phyz` when phyz is not on ``PATH``.
        superfamily_pcp_root: Path to the superfamily-pcp checkout. Read by
            :func:`get_superfamily_pcp_root`.
    """

    registry_paths: List[str] = Field(default_factory=list)
    artifacts_dir: Optional[str] = None
    experiment_artifact_root: Optional[str] = None
    remote_host: Optional[str] = None
    remote_dir: Optional[str] = None
    phyz_binary: Optional[str] = None
    superfamily_pcp_root: Optional[str] = None

    @field_validator("registry_paths", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> List[str]:
        """Ensure registry_paths is a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)


# Cache for user config
_USER_CONFIG: Optional[UserConfig] = None

# Legacy config location
_LEGACY_CONFIG_PATH = Path.home() / ".dasm2" / "config.yaml"


def get_xdg_config_dir(app: str = "dasm2") -> Path:
    """Return the XDG config directory for *app*, falling back to ``~/.config/app``.

    Uses ``$XDG_CONFIG_HOME`` if set and non-empty, otherwise defaults to
    ``~/.config``.

    Args:
        app: Application name subdirectory (default ``"dasm2"``).

    Returns:
        Path to the XDG config directory for the application.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / app
    return Path.home() / ".config" / app


def get_config_path() -> Path:
    """Resolve the config file path using XDG Base Directory with legacy fallback.

    Resolution order:

    1. ``$XDG_CONFIG_HOME/dasm2/config.yaml``
       (defaults to ``~/.config/dasm2/config.yaml``)
    2. ``~/.dasm2/config.yaml`` (legacy fallback)

    Returns:
        Path to the config file (may not exist).
    """
    xdg_path = get_xdg_config_dir() / "config.yaml"

    if xdg_path.exists():
        return xdg_path
    if _LEGACY_CONFIG_PATH.exists():
        return _LEGACY_CONFIG_PATH

    # Neither exists; prefer XDG location for error messages / new installs
    return xdg_path


def _load_user_config() -> UserConfig:
    """Load user configuration from the resolved config path.

    Returns:
        UserConfig with settings from file, or defaults if file doesn't exist.
    """
    global _USER_CONFIG
    if _USER_CONFIG is not None:
        return _USER_CONFIG

    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            _USER_CONFIG = UserConfig(**data)
        except (yaml.YAMLError, ValidationError) as e:
            raise ValueError(
                f"Error parsing {config_path}: {e}\n"
                f"Please check your config file format."
            ) from e
    else:
        _USER_CONFIG = UserConfig()

    return _USER_CONFIG


def get_user_config() -> UserConfig:
    """Get user configuration.

    Returns:
        UserConfig from config file or defaults.
    """
    return _load_user_config()


def get_registry_paths() -> List[Path]:
    """Get list of registry paths to scan for dataset YAML files.

    Returns:
        List of paths from user config, with ~ expanded.
    """
    config = get_user_config()
    return [Path(p).expanduser() for p in config.registry_paths]


def get_artifacts_dir() -> Optional[Path]:
    """Get configured artifacts directory.

    Returns:
        Path to artifacts directory from config, or None if not configured.
    """
    config = get_user_config()
    if config.artifacts_dir:
        return Path(config.artifacts_dir).expanduser()
    return None


def get_superfamily_pcp_root() -> Path:
    """Get the configured superfamily-pcp checkout directory.

    Returns:
        ``superfamily_pcp_root`` from the user config, with ``~`` expanded.

    Raises:
        FileNotFoundError: If the key is unset or does not name a directory.
            The message names the key and the resolved config-file path.
    """
    config = get_user_config()
    if not config.superfamily_pcp_root:
        raise FileNotFoundError(
            "superfamily-pcp checkout not configured. Set "
            f"`superfamily_pcp_root: /path/to/superfamily-pcp` in {get_config_path()}"
        )
    root = Path(config.superfamily_pcp_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"superfamily-pcp checkout not found at configured path: {root}\n"
            f"  (from {get_config_path()} → superfamily_pcp_root)"
        )
    return root


def get_test_data_dir() -> Path:
    """Get path to package test data directory.

    Returns:
        Path to dasm2/tst_data directory.
    """
    return Path(str(files("dasm2") / "tst_data"))


def compute_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file efficiently.

    Streams the file to avoid loading it entirely into memory.

    Args:
        file_path: Path to the file to hash.
        chunk_size: Size of chunks to read (default 8KB).

    Returns:
        Lowercase hexadecimal MD5 hash string.

    Raises:
        FileNotFoundError: If file doesn't exist.
        IOError: If file cannot be read.
    """
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def compute_sha256(file_path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hash of a file, streaming.

    Sits next to :func:`compute_md5` because manifests use MD5 for integrity
    against a recorded value, while provenance blobs written *by* us record
    SHA-256. Callers that need to prove which input produced an artifact should
    use this one.

    Args:
        file_path: Path to the file to hash.
        chunk_size: Size of chunks to read (default 1MB).

    Returns:
        Lowercase hexadecimal SHA-256 hash string.

    Raises:
        FileNotFoundError: If file doesn't exist.
        IOError: If file cannot be read.
    """
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_md5(file_path: Path, expected_md5: str) -> None:
    """Verify file MD5 matches expected value.

    Args:
        file_path: Path to the file to verify.
        expected_md5: Expected MD5 hash (case-insensitive).

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If MD5 doesn't match.
    """
    actual_md5 = compute_md5(file_path)
    if actual_md5.lower() != expected_md5.lower():
        raise ValueError(
            f"MD5 mismatch for {file_path}:\n"
            f"  Expected: {expected_md5.lower()}\n"
            f"  Actual:   {actual_md5}\n"
            f"File may be corrupted or modified."
        )


def resolve_path(yaml_dir: Path, relative_path: str) -> Path:
    """Resolve a path relative to a YAML file's directory.

    Args:
        yaml_dir: Directory containing the YAML file.
        relative_path: Path string (may be relative or absolute).

    Returns:
        Resolved absolute path.
    """
    p = Path(relative_path)
    if p.is_absolute():
        return p.resolve()
    return (yaml_dir / relative_path).resolve()


def resolve_path_with_fallback(
    spec: str,
    registry_lookup: Callable[[str], Optional[Path]],
    item_type: str,
    list_available: Callable[[], List[str]],
) -> Path:
    """Resolve a path specification using direct path or registry lookup.

    Tries to resolve the spec as a direct file path first, then falls back
    to registry lookup. Provides helpful error messages listing available
    options if resolution fails.

    Args:
        spec: File path or registered nickname to resolve.
        registry_lookup: Function that looks up spec in registry, returns
            resolved Path or None if not found.
        item_type: Human-readable type for error messages (e.g., "Dataset").
        list_available: Function that returns list of available nicknames.

    Returns:
        Resolved absolute Path.

    Raises:
        ValueError: If spec is empty or whitespace-only.
        FileNotFoundError: If path doesn't exist and nickname not registered.
    """
    # Fail fast on empty specification
    if not spec or not spec.strip():
        raise ValueError(
            f"{item_type} specification cannot be empty. "
            f"Provide a file path or registered nickname."
        )

    # Try as direct path first
    path = Path(spec).expanduser()
    if path.exists():
        return path.resolve()

    # Try registry lookup
    resolved = registry_lookup(spec)
    if resolved is not None:
        return resolved

    # Not found - provide helpful error
    available = list_available()
    available_display = ", ".join(available[:10])
    if len(available) > 10:
        available_display += f", ... ({len(available)} total)"

    raise FileNotFoundError(
        f"{item_type} '{spec}' not found.\n"
        f"Available: {available_display or '(none)'}\n"
        f"Or provide a valid path."
    )


class VerifiedPath:
    """A resolved path with optional MD5 that verifies lazily on first access.

    This class bundles a file path with an optional MD5 checksum. When the
    `path` property is accessed, the MD5 is verified (if provided) on the
    first access only. Subsequent accesses return the path without re-verifying.

    Attributes:
        md5: The expected MD5 checksum, or None if not specified.

    Example:
        >>> vp = VerifiedPath(Path("/data/file.csv"), md5="abc123...")
        >>> vp.md5  # Does NOT trigger verification
        'abc123...'
        >>> vp.path  # Triggers verification on first access
        PosixPath('/data/file.csv')
        >>> vp.path  # No verification on subsequent access
        PosixPath('/data/file.csv')
    """

    def __init__(self, path: Path, md5: Optional[str] = None):
        """Initialize a VerifiedPath.

        Args:
            path: The resolved absolute path to the file.
            md5: Optional MD5 checksum for integrity verification.
        """
        self._path = path
        self._md5 = md5
        self._verified = False

    @property
    def path(self) -> Path:
        """Get the path, verifying MD5 on first access if provided.

        Returns:
            The file path.

        Raises:
            FileNotFoundError: If file doesn't exist (on first access with MD5).
            ValueError: If MD5 doesn't match (on first access with MD5).
        """
        if not self._verified:
            if self._md5:
                verify_md5(self._path, self._md5)
            self._verified = True
        return self._path

    @property
    def md5(self) -> Optional[str]:
        """Get the expected MD5 checksum (does not trigger verification)."""
        return self._md5

    @property
    def raw_path(self) -> Path:
        """Get the path without triggering MD5 verification.

        Use this when you need the path value without verifying the file exists
        or its checksum matches (e.g., for error reporting when a file is missing).
        """
        return self._path

    def __repr__(self) -> str:
        """Return string representation."""
        return f"VerifiedPath({self._path!r}, md5={self._md5!r})"


def reset_config_cache() -> None:
    """Reset the cached user config.

    Useful for testing when config file changes.
    """
    global _USER_CONFIG
    _USER_CONFIG = None


class PathWithMD5(BaseModel):
    """A file path with optional MD5 checksum for integrity verification.

    This is a shared utility for YAML-based registry schemas. Paths are stored
    as strings (relative to the YAML file location) and resolved at load time.

    Accepts either a string path or a dict with {path, md5} keys:
        - "data/file.csv" -> PathWithMD5(path="data/file.csv", md5=None)
        - {path: "data/file.csv", md5: "abc..."} -> PathWithMD5(path="data/file.csv", md5="abc...")

    Attributes:
        path: Relative path to the file (relative to YAML file location).
        md5: Optional MD5 checksum for file integrity verification.
    """

    path: str
    md5: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_string_input(cls, data: Any) -> Any:
        """Accept a plain string as shorthand for {path: string}."""
        if isinstance(data, str):
            return {"path": data}
        return data

    @field_validator("path")
    @classmethod
    def path_not_empty(cls, v: str) -> str:
        """Validate that path is not empty."""
        if not v or not v.strip():
            raise ValueError("path cannot be empty")
        return v

    @field_validator("md5")
    @classmethod
    def md5_valid_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate MD5 hash format if provided."""
        if v is None:
            return v
        v = v.strip().lower()
        if len(v) != 32:
            raise ValueError(f"MD5 hash must be 32 characters, got {len(v)}")
        if not all(c in "0123456789abcdef" for c in v):
            raise ValueError("MD5 hash must contain only hexadecimal characters")
        return v

    def resolve(self, yaml_dir: Path) -> VerifiedPath:
        """Resolve this path relative to a YAML directory.

        Args:
            yaml_dir: Directory containing the source YAML file.

        Returns:
            VerifiedPath with absolute path and optional MD5 for lazy verification.
        """
        return VerifiedPath(resolve_path(yaml_dir, self.path), self.md5)
