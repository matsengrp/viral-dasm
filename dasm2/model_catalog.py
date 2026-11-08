"""YAML-based model catalog for portable model resolution.

Provides a registry of named models (neutral models, TDI token directories)
that can be resolved to local file paths. Models are downloaded on demand
from GitHub when not found in the local cache.

The catalog is defined in ``model_catalog.yaml`` (bundled as package data)
and maps human-readable names to download URLs with SHA256 integrity checks.

Usage:
    >>> from dasm2.model_catalog import resolve_model, list_models
    >>> path = resolve_model("TabularRates-SCoV2-v1")
    >>> models = list_models(model_type="neutral_model")
"""

import functools
import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
from importlib.resources import files
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

ModelType = Literal["neutral_model", "tdi_tokens"]

logger = logging.getLogger(__name__)


class CatalogEntry(BaseModel):
    """A single model entry in the catalog.

    Attributes:
        name: Unique human-readable model identifier (e.g., "TabularRates-SCoV2-v1").
        model_type: Category string (e.g., "neutral_model", "tdi_tokens").
        path: Relative path within the cache directory (e.g., "neutral_models/model.zip").
        url: Optional download URL for auto-fetching.
        sha256: Optional SHA256 hex digest for integrity verification.
        description: Optional human-readable description.
    """

    name: str = Field(description="Unique model identifier")
    model_type: ModelType = Field(description="Model category")
    path: str = Field(description="Relative path within cache directory")
    url: Optional[str] = Field(default=None, description="Download URL")
    sha256: Optional[str] = Field(default=None, description="SHA256 hex digest")
    description: Optional[str] = Field(
        default=None, description="Human-readable description"
    )


class ModelCatalog(BaseModel):
    """Top-level catalog containing versioned model entries.

    Attributes:
        version: Catalog format version (currently 1).
        models: List of model entries.
    """

    version: int = Field(description="Catalog format version")
    models: List[CatalogEntry] = Field(description="Model entries")

    @functools.cached_property
    def models_by_name(self) -> Dict[str, CatalogEntry]:
        """Index of model entries by name for O(1) lookup."""
        return {e.name: e for e in self.models}


def get_model_cache_dir() -> Path:
    """Return the local cache directory for downloaded models.

    Uses ``$XDG_CACHE_HOME/dasm2/models/`` (falling back to
    ``~/.cache/dasm2/models/`` when ``XDG_CACHE_HOME`` is unset).

    Returns:
        Path to the cache directory (created on first access).
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        base = Path(xdg_cache)
    else:
        base = Path.home() / ".cache"
    cache_dir = base / "dasm2" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@functools.lru_cache(maxsize=1)
def load_catalog() -> ModelCatalog:
    """Load the model catalog from the bundled ``model_catalog.yaml``.

    Returns:
        Parsed and validated ModelCatalog.

    Raises:
        FileNotFoundError: If the catalog YAML is not found in package data.
        ValueError: If the catalog YAML is invalid.
    """
    catalog_path = Path(str(files("dasm2").joinpath("model_catalog.yaml")))
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Model catalog not found at {catalog_path}. "
            "Ensure dasm2 is installed with package data."
        )
    with open(catalog_path) as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Empty model catalog at {catalog_path}")
    return ModelCatalog(**data)


def _compute_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(path: Path, expected: str, name: str) -> None:
    """Verify a file's SHA256 digest, raising on mismatch.

    Args:
        path: File to verify.
        expected: Expected lowercase hex digest.
        name: Model name for error messages.

    Raises:
        ValueError: If the computed digest does not match expected.
    """
    actual = _compute_sha256(path)
    if actual != expected:
        raise ValueError(
            f"SHA256 mismatch for downloaded '{name}'. "
            f"Expected: {expected}, got: {actual}"
        )


def _get_github_token() -> Optional[str]:
    """Get GitHub token for private repo access.

    Checks in order:
    1. ``GITHUB_TOKEN`` environment variable
    2. ``GH_TOKEN`` environment variable
    3. ``gh auth token`` CLI output (if ``gh`` is installed and authenticated)

    Returns:
        Token string or None if no token is available.
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token

    # Fall back to GitHub CLI
    import subprocess

    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # gh not installed or timed out
        pass

    return None


_AUTH_SETUP_HINT = (
    "To download from private GitHub repos, authenticate using one of:\n"
    "  1. GitHub CLI (recommended): gh auth login\n"
    "  2. Environment variable: export GITHUB_TOKEN=ghp_..."
)


def _download_file(url: str, dest: Path) -> None:
    """Download a file from a URL, with optional GitHub token auth.

    For private GitHub repos, adds an Authorization header using a token
    from environment variables or the GitHub CLI (see ``_get_github_token``).

    Args:
        url: Source URL.
        dest: Local destination path.

    Raises:
        urllib.error.URLError: If download fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    token = _get_github_token()
    req = urllib.request.Request(url)

    if token:
        req.add_header("Authorization", f"token {token}")

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            with open(dest, "wb") as out:
                shutil.copyfileobj(response, out)
    except urllib.error.HTTPError as e:
        if e.code == 401 and token:
            # Retry without token (repo may have gone public)
            logger.info("Auth failed with token, retrying without auth")
            try:
                req_noauth = urllib.request.Request(url)
                with urllib.request.urlopen(req_noauth, timeout=300) as response:
                    with open(dest, "wb") as out:
                        shutil.copyfileobj(response, out)
            except urllib.error.HTTPError as retry_err:
                raise urllib.error.HTTPError(
                    retry_err.url,
                    retry_err.code,
                    f"Download failed for {url} (tried with and without auth): "
                    f"{retry_err.reason}",
                    retry_err.headers,
                    retry_err.fp,
                ) from retry_err
        elif e.code in (401, 403) and not token:
            raise urllib.error.HTTPError(
                e.url,
                e.code,
                f"Authentication required to download {url}.\n{_AUTH_SETUP_HINT}",
                e.headers,
                e.fp,
            ) from e
        else:
            raise


def get_catalog_entry(name: str) -> Optional[CatalogEntry]:
    """Look up a catalog entry by name.

    Args:
        name: Model name as listed in the catalog.

    Returns:
        The matching CatalogEntry, or None if not found.
    """
    return load_catalog().models_by_name.get(name)


def resolve_model(name: str) -> Path:
    """Resolve a catalog model name to a local file path.

    Resolution steps:
    1. Look up ``name`` in the catalog.
    2. Check if ``{cache_dir}/{entry.path}`` exists locally; return if cached.
    3. If not found and ``url`` is set, download and verify SHA256.
    4. For ``.tar.gz`` URLs, extract after download.
    5. If still missing, raise an error with setup instructions.

    Args:
        name: Model name as listed in the catalog (e.g., "TabularRates-SCoV2-v1").

    Returns:
        Absolute path to the resolved model file or directory.

    Raises:
        KeyError: If ``name`` is not in the catalog.
        FileNotFoundError: If the model cannot be found or downloaded.
        ValueError: If SHA256 verification fails.
    """
    catalog = load_catalog()

    # Look up entry
    entry = catalog.models_by_name.get(name)
    if entry is None:
        available = sorted(catalog.models_by_name.keys())
        raise KeyError(
            f"Model '{name}' not found in catalog. Available models: {available}"
        )

    cache_dir = get_model_cache_dir()
    local_path = cache_dir / entry.path

    # Check if already cached (integrity was verified at download time)
    if local_path.exists():
        logger.info(f"Model '{name}' found in cache: {local_path}")
        return local_path

    # Try to download
    if entry.url:
        logger.info(f"Downloading model '{name}' from {entry.url}")
        is_tarball = entry.url.endswith(".tar.gz") or entry.url.endswith(".tgz")

        if is_tarball:
            # Download to a temp file, then extract
            local_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(suffix=".tar.gz", dir=local_path.parent)
            os.close(fd)
            tarball_path = Path(tmp_name)
            try:
                _download_file(entry.url, tarball_path)

                if entry.sha256:
                    _verify_sha256(tarball_path, entry.sha256, name)

                # Extract
                local_path.mkdir(parents=True, exist_ok=True)
                with tarfile.open(tarball_path) as tf:
                    tf.extractall(path=local_path, filter="data")
                logger.info(f"Extracted '{name}' to {local_path}")
            except Exception:
                # Clean up partial extraction to avoid stale cache entries
                if local_path.exists():
                    shutil.rmtree(local_path, ignore_errors=True)
                raise
            finally:
                if tarball_path.exists():
                    tarball_path.unlink()
        else:
            _download_file(entry.url, local_path)

            if entry.sha256:
                try:
                    _verify_sha256(local_path, entry.sha256, name)
                except ValueError:
                    local_path.unlink(missing_ok=True)
                    raise

            logger.info(f"Downloaded '{name}' to {local_path}")

        return local_path

    # No URL and not cached
    raise FileNotFoundError(
        f"Model '{name}' not found in cache ({local_path}) and no download URL "
        f"is configured. To populate the cache manually:\n"
        f"  1. Clone https://github.com/matsengrp/dasm2-models\n"
        f"  2. Copy the model file to {local_path}\n\n"
        f"If the model has a download URL but you're seeing this error, check "
        f"that the model_catalog.yaml entry has a 'url' field.\n\n"
        f"{_AUTH_SETUP_HINT}"
    )


def list_models(model_type: Optional[ModelType] = None) -> List[CatalogEntry]:
    """List models in the catalog, optionally filtered by type.

    Args:
        model_type: If set, only return entries matching this type
            (e.g., "neutral_model").

    Returns:
        List of matching CatalogEntry objects.
    """
    catalog = load_catalog()
    if model_type is None:
        return catalog.models
    return [e for e in catalog.models if e.model_type == model_type]
