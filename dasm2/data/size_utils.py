"""Size measurement utilities for dataset manifests.

Provides size readers that measure dataset sizes for CLI display.  Each reader
returns a :class:`SizeResult` with a human-readable value, a column label,
and a numeric sort key.

Two built-in readers are provided:

* :func:`read_line_count` — estimates or counts CSV lines (default).
* :func:`read_file_size` — reports raw file size (for archives).
"""

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Default sample size for line count estimation (4 MB of decompressed data)
_DEFAULT_SAMPLE_BYTES = 4 * 1024 * 1024

# Below this estimated line count, use exact counting instead
_EXACT_COUNT_THRESHOLD = 1000


def count_lines(file_path: Path) -> int:
    """Count lines in a file exactly (handles gzip compression).

    Args:
        file_path: Path to file

    Returns:
        Number of lines in file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a regular file
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot count lines in non-existent file: {file_path}")

    if not file_path.is_file():
        raise ValueError(
            f"Cannot count lines in non-file path: {file_path} (is directory or special file)"
        )

    if file_path.suffix == ".gz":
        with gzip.open(file_path, "rt") as f:
            return sum(1 for _ in f)
    else:
        with open(file_path) as f:
            return sum(1 for _ in f)


def estimate_line_count(
    file_path: Path, sample_decompressed_bytes: int = _DEFAULT_SAMPLE_BYTES
) -> int:
    """Estimate line count by sampling and extrapolating.

    For gzip files, reads until a target amount of decompressed data, tracks
    the actual compression ratio, and extrapolates.

    For uncompressed files, samples the first portion and extrapolates.

    Args:
        file_path: Path to file (handles gzip compression)
        sample_decompressed_bytes: Read until this many decompressed bytes
            (default 4MB)

    Returns:
        Estimated number of data lines (excluding header)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a regular file or has no data lines
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot estimate lines in non-existent file: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Cannot estimate lines in non-file path: {file_path} (is directory or special file)"
        )

    file_size = file_path.stat().st_size

    if file_path.suffix == ".gz":
        # For gzip: track decompressed bytes and measure compression ratio
        with open(file_path, "rb") as raw_file:
            with gzip.GzipFile(fileobj=raw_file) as gz:
                lines_read = 0
                decompressed_bytes = 0

                # Read lines until we have enough decompressed data
                for raw_line in gz:
                    decompressed_bytes += len(raw_line)
                    lines_read += 1
                    if decompressed_bytes >= sample_decompressed_bytes:
                        break

                if lines_read == 0:
                    raise ValueError(f"File appears empty: {file_path}")

                # Get compressed bytes consumed (includes some buffer overshoot)
                compressed_bytes_read = raw_file.tell()

                # Estimate total decompressed size using observed ratio
                compression_ratio = decompressed_bytes / compressed_bytes_read
                estimated_total_decompressed = file_size * compression_ratio

                # Estimate lines using average line length from sample
                avg_line_length = decompressed_bytes / lines_read
                estimated_total_lines = int(
                    estimated_total_decompressed / avg_line_length
                )
    else:
        # For uncompressed: sample first portion of file
        sample_bytes = min(sample_decompressed_bytes, file_size)
        lines_read = 0
        bytes_read = 0

        with open(file_path) as f:
            for text_line in f:
                bytes_read += len(text_line)
                lines_read += 1
                if bytes_read >= sample_bytes:
                    break

        if lines_read == 0 or bytes_read == 0:
            raise ValueError(f"File appears empty: {file_path}")

        # Extrapolate based on bytes read
        ratio = file_size / bytes_read
        estimated_total_lines = int(lines_read * ratio)

    # Return data lines (subtract 1 for header)
    return max(0, estimated_total_lines - 1)


def format_line_count(count: int) -> str:
    """Format a line count with appropriate suffix.

    Formatting rules:
    - Less than 1000: show exact number
    - 1000-999999: show as Xk (rounded to nearest thousand)
    - 1000000+: show as X.XM (rounded to nearest hundred thousand)

    Args:
        count: Number of lines

    Returns:
        Formatted string like "125k", "1.2M", or "500"
    """
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        thousands = round(count / 1000)
        return f"{thousands}k"
    else:
        millions = round(count / 100_000) / 10
        return f"{millions:.1f}M"


@dataclass
class SizeResult:
    """Result from a size_reader function.

    Attributes:
        label: Column header, e.g. "Lines Est", "Size".
        value: Formatted display value, e.g. "120k", "4.3 GB".
        sort_key: Numeric value for sorting (line count or bytes).
    """

    label: str
    value: str
    sort_key: float


def format_file_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable file size.

    Args:
        size_bytes: Number of bytes.

    Returns:
        Formatted string like "4.3 GB", "125 MB", "8 KB", or "512 B".
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.0f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"


def read_line_count(data_path: Path, exact: bool) -> Optional[SizeResult]:
    """Read line count for a data file (estimate or exact).

    Args:
        data_path: Path to the data file. Must exist and be a regular file.
        exact: If True, count lines exactly. If False, estimate (but use exact
            for small files below the threshold).

    Returns:
        SizeResult with line count, or None if file is unreadable.
    """
    try:
        if exact:
            line_count = count_lines(data_path)
            return SizeResult(
                label="Lines",
                value=format_line_count(line_count),
                sort_key=float(line_count),
            )
        else:
            line_count = estimate_line_count(data_path)
            is_estimate = True
            if line_count < _EXACT_COUNT_THRESHOLD:
                line_count = count_lines(data_path)
                is_estimate = False
            return SizeResult(
                label="Lines Est" if is_estimate else "Lines",
                value=format_line_count(line_count),
                sort_key=float(line_count),
            )
    except (OSError, IOError):
        return None


def read_file_size(data_path: Path, exact: bool) -> Optional[SizeResult]:
    """Read file size for a data file.

    Args:
        data_path: Path to the data file. Must exist and be a regular file.
        exact: Unused (file size is always exact), accepted for protocol
            compatibility.

    Returns:
        SizeResult with file size, or None if file is unreadable.
    """
    try:
        size_bytes = data_path.stat().st_size
        return SizeResult(
            label="Size",
            value=format_file_size(size_bytes),
            sort_key=float(size_bytes),
        )
    except (OSError, IOError):
        return None


#: Type alias for size reader callables used by DataTypeDescriptor.
SizeReader = Callable[[Path, bool], Optional[SizeResult]]
