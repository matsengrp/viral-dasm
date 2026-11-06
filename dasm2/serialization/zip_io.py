"""Generic ZIP archive I/O for Pydantic metadata + torch data.

Provides two archive patterns:
1. Model archives (write_model_archive/read_model_archive): metadata.yml + torch state_dict
2. Data archives (write_data_archive/read_data_archive): metadata.yml + arbitrary named files
"""

import warnings
import zipfile
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml
from pydantic import BaseModel


def _make_yaml_safe(obj: Any) -> Any:
    """Recursively convert non-YAML-safe types for serialization.

    Converts Enum values to strings and Path objects to strings so that
    yaml.dump() produces output compatible with yaml.safe_load().

    Args:
        obj: Any Python object (dict, list, or primitive)

    Returns:
        Copy of obj with Enum and Path instances converted to strings
    """
    if isinstance(obj, dict):
        return {
            (k.value if isinstance(k, Enum) else k): _make_yaml_safe(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        # Tuples must become lists, not just recurse element-wise: yaml.dump's
        # default Dumper serializes a bare tuple as a `!!python/tuple` node,
        # which yaml.safe_load (the reader) cannot construct (issue #501
        # follow-up -- multihit_log_factors is the first Tuple-typed config
        # field ever round-tripped through this path).
        return [_make_yaml_safe(item) for item in obj]
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def _validate_zip_path_for_read(path: Path) -> None:
    """Validate path for reading a ZIP archive.

    Args:
        path: Path to validate

    Raises:
        ValueError: If file is not a .zip file
        FileNotFoundError: If file doesn't exist
    """
    if path.suffix != ".zip":
        raise ValueError(f"Expected .zip file, got: {path.suffix}")
    if not path.exists():
        raise FileNotFoundError(f"Archive not found: {path}")


def _prepare_path_for_write(path: Path) -> None:
    """Prepare path for writing, creating parent directories.

    Args:
        path: Path to prepare
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _serialize_metadata(metadata: BaseModel) -> bytes:
    """Serialize Pydantic metadata to YAML bytes.

    Args:
        metadata: Pydantic model to serialize

    Returns:
        UTF-8 encoded YAML bytes
    """
    metadata_dict = metadata.model_dump()
    yaml_safe_metadata = _make_yaml_safe(metadata_dict)
    return yaml.dump(yaml_safe_metadata, default_flow_style=False).encode("utf-8")


def _read_metadata_from_zip(zf: zipfile.ZipFile) -> Dict[str, Any]:
    """Read and parse metadata.yml from an open ZIP file.

    Args:
        zf: Open ZipFile object

    Returns:
        Metadata dictionary (empty dict if metadata.yml is empty)
    """
    metadata = yaml.safe_load(zf.read("metadata.yml"))
    return metadata if metadata is not None else {}


# =============================================================================
# Model Archives (torch-based)
# =============================================================================


def write_model_archive(
    path: Union[str, Path],
    metadata: BaseModel,
    data: Any,
    data_filename: str = "data.pt",
    extra_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Write Pydantic metadata + torch data to ZIP archive.

    Creates a ZIP archive containing:
    - metadata.yml: Human-readable YAML serialization of metadata
    - {data_filename}: Torch-serialized data
    - Any additional torch-serialized entries from extra_data

    Args:
        path: Output file path
        metadata: Pydantic model to serialize as YAML
        data: Any torch-serializable data (state_dict, tensor, etc.)
        data_filename: Name for data file in archive (default: "data.pt")
        extra_data: Optional dict mapping archive filenames to torch-serializable
            data. Each entry is saved as a separate file in the archive.
    """
    path = Path(path)
    _prepare_path_for_write(path)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.yml", _serialize_metadata(metadata))

        buffer = BytesIO()
        torch.save(data, buffer, _use_new_zipfile_serialization=False)
        zf.writestr(data_filename, buffer.getvalue())

        if extra_data:
            reserved = {"metadata.yml", data_filename}
            overlap = reserved & set(extra_data.keys())
            if overlap:
                raise ValueError(
                    f"extra_data keys conflict with reserved archive entries: {overlap}"
                )
            for filename, entry_data in extra_data.items():
                entry_buffer = BytesIO()
                torch.save(
                    entry_data, entry_buffer, _use_new_zipfile_serialization=False
                )
                zf.writestr(filename, entry_buffer.getvalue())


def read_zip_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """Read only metadata.yml from ZIP archive.

    Useful for inspection without loading heavy data.

    Args:
        path: Path to ZIP archive

    Returns:
        Metadata dictionary from YAML

    Raises:
        ValueError: If file is not a .zip file
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    _validate_zip_path_for_read(path)

    with zipfile.ZipFile(path, "r") as zf:
        return _read_metadata_from_zip(zf)


def read_model_archive(
    path: Union[str, Path],
    data_filename: str = "data.pt",
    device: str = "cpu",
    weights_only: bool = False,
) -> Tuple[Dict[str, Any], Any]:
    """Read metadata and torch data from ZIP archive.

    Args:
        path: Path to ZIP archive
        data_filename: Name of data file in archive (default: "data.pt")
        device: Device to load tensors on (default: "cpu")
        weights_only: If True, only load tensor weights (safer but less flexible)

    Returns:
        Tuple of (metadata_dict, data)

    Raises:
        ValueError: If file is not a .zip file
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    _validate_zip_path_for_read(path)

    with zipfile.ZipFile(path, "r") as zf:
        metadata = _read_metadata_from_zip(zf)
        data = torch.load(
            BytesIO(zf.read(data_filename)),
            map_location=device,
            weights_only=weights_only,
        )

    return metadata, data


def read_model_archive_with_extras(
    path: Union[str, Path],
    data_filename: str = "data.pt",
    device: str = "cpu",
    weights_only: bool = False,
    extra_data_filenames: Optional[Dict[str, bool]] = None,
) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """Read metadata, torch data, and extra entries from ZIP archive.

    Like read_model_archive but also loads additional named entries.

    Args:
        path: Path to ZIP archive
        data_filename: Name of data file in archive (default: "data.pt")
        device: Device to load tensors on (default: "cpu")
        weights_only: If True, only load tensor weights for main data
        extra_data_filenames: Dict mapping archive filenames to their
            weights_only setting. Each entry is loaded and returned by name.

    Returns:
        Tuple of (metadata_dict, data, extras_dict) where extras_dict
        maps filenames to their loaded torch data.

    Raises:
        ValueError: If file is not a .zip file or required files are missing
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    _validate_zip_path_for_read(path)

    extras: Dict[str, Any] = {}
    with zipfile.ZipFile(path, "r") as zf:
        if extra_data_filenames:
            archive_files = zf.namelist()
            missing = set(extra_data_filenames.keys()) - set(archive_files)
            if missing:
                raise ValueError(
                    f"Archive missing required files: {missing}. "
                    f"Available: {archive_files}"
                )

        metadata = _read_metadata_from_zip(zf)
        data = torch.load(
            BytesIO(zf.read(data_filename)),
            map_location=device,
            weights_only=weights_only,
        )

        if extra_data_filenames:
            for filename, entry_weights_only in extra_data_filenames.items():
                extras[filename] = torch.load(
                    BytesIO(zf.read(filename)),
                    map_location=device,
                    weights_only=entry_weights_only,
                )

    return metadata, data, extras


def model_archive_exists(
    path: Union[str, Path], data_filename: str = "data.pt"
) -> bool:
    """Check if valid model ZIP archive exists with expected files.

    Args:
        path: Path to check
        data_filename: Expected data file name in archive

    Returns:
        True if valid archive with metadata.yml and data file exists
    """
    path = Path(path)
    if not path.exists():
        return False

    try:
        with zipfile.ZipFile(path, "r") as zf:
            return "metadata.yml" in zf.namelist() and data_filename in zf.namelist()
    except (zipfile.BadZipFile, OSError) as e:
        warnings.warn(
            f"File exists but is corrupt or unreadable: {path} ({e})",
            stacklevel=2,
        )
        return False


# =============================================================================
# Data Archives (non-torch, arbitrary files)
# =============================================================================


def write_data_archive(
    path: Union[str, Path],
    metadata: BaseModel,
    files: Dict[str, bytes],
) -> None:
    """Write Pydantic metadata + arbitrary data files to ZIP archive.

    Creates a ZIP archive containing:
    - metadata.yml: Human-readable YAML serialization of metadata
    - Additional files specified in the files dict

    This is useful for data-driven models that don't use torch weights,
    such as empirical rate models loaded from CSV files.

    Args:
        path: Output file path
        metadata: Pydantic model to serialize as YAML
        files: Dict mapping filenames to file contents as bytes

    Raises:
        ValueError: If any filename is 'metadata.yml' (reserved)
    """
    if "metadata.yml" in files:
        raise ValueError("'metadata.yml' is reserved and cannot be in files dict")

    path = Path(path)
    _prepare_path_for_write(path)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.yml", _serialize_metadata(metadata))

        for filename, content in files.items():
            zf.writestr(filename, content)


def read_data_archive(
    path: Union[str, Path],
    file_names: List[str],
) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    """Read metadata and named files from ZIP archive.

    Args:
        path: Path to ZIP archive
        file_names: List of filenames to read from archive

    Returns:
        Tuple of (metadata_dict, files_dict) where files_dict maps
        filename to bytes content

    Raises:
        ValueError: If file is not a .zip file
        FileNotFoundError: If file doesn't exist
        KeyError: If any requested file is not in the archive
    """
    path = Path(path)
    _validate_zip_path_for_read(path)

    with zipfile.ZipFile(path, "r") as zf:
        metadata = _read_metadata_from_zip(zf)

        archive_files = zf.namelist()
        files = {}
        for name in file_names:
            if name not in archive_files:
                raise KeyError(
                    f"File '{name}' not found in archive. Available: {archive_files}"
                )
            files[name] = zf.read(name)

    return metadata, files


def data_archive_exists(path: Union[str, Path], file_names: List[str]) -> bool:
    """Check if valid data ZIP archive exists with expected files.

    Args:
        path: Path to check
        file_names: Expected file names in archive (in addition to metadata.yml)

    Returns:
        True if valid archive with metadata.yml and all named files exists
    """
    path = Path(path)
    if not path.exists():
        return False

    try:
        with zipfile.ZipFile(path, "r") as zf:
            archive_files = zf.namelist()
            if "metadata.yml" not in archive_files:
                return False
            return all(name in archive_files for name in file_names)
    except (zipfile.BadZipFile, OSError) as e:
        warnings.warn(
            f"File exists but is corrupt or unreadable: {path} ({e})",
            stacklevel=2,
        )
        return False
