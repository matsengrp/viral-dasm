"""Resolve precomputed artifacts from explicit paths and configured storage."""

from pathlib import Path


def resolve_artifact_path(
    spec: str | Path,
    subdirectory: str,
    *,
    description: str,
    instructions: str,
    search_storage: bool = True,
) -> Path:
    """Find a file locally, in user artifacts, then in shared artifact storage.

    Args:
        spec: Explicit path or artifact filename; expands the user home directory.
        subdirectory: Artifact family directory under each configured storage root.
        description: Artifact description used in a missing-file error.
        instructions: Recovery instructions appended to a missing-file error.
        search_storage: Whether to try the basename under configured roots.
            Disable for explicit custom paths that must not fall back elsewhere.

    Returns:
        First existing file in lookup order.

    Raises:
        FileNotFoundError: No candidate is a file; lists all attempted paths.
    """
    from dasm2.config import get_experiment_artifact_root
    from dasm2.data.local import get_artifacts_dir

    spec_path = Path(spec).expanduser()
    candidates = [spec_path]
    if search_storage:
        artifacts = get_artifacts_dir()
        if artifacts is not None:
            candidates.append(artifacts / subdirectory / spec_path.name)
        candidates.append(
            Path(get_experiment_artifact_root()).expanduser()
            / subdirectory
            / spec_path.name
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    attempted = "\n  ".join(map(str, candidates))
    raise FileNotFoundError(f"{description}. Tried:\n  {attempted}\n{instructions}")
