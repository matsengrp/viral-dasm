"""Training artifact implementations for caching expensive computations."""

import hashlib
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Sized,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from torch import Tensor

from dasm2.metadata import ArtifactMetadata
from dasm2.serialization.naming import generate_model_identifier
from dasm2.serialization.zip_io import read_model_archive, write_model_archive

if TYPE_CHECKING:
    from dasm2.data.evolutionary_dataset import BranchLengthDatasetBundle
    from dasm2.heads.types import HeadType
    from dasm2.training.trainer import Trainer

# Generic type variable for training artifacts
T = TypeVar("T", bound=Sized)
"""Generic type parameter for TrainingArtifact container types."""


@runtime_checkable
class CacheableDataset(Protocol):
    """Protocol for datasets that can be cached.

    Datasets must have a nickname attribute for cache key generation, and a
    row_filter_digest that tells apart datasets built from one data spec with
    different row filters.
    """

    nickname: str
    """Dataset identifier used for cache filenames."""

    row_filter_digest: Optional[str]
    """Digest of the selected rows, or ``None`` when the dataset is unfiltered."""

    def __len__(self) -> int:
        """Return number of items in dataset."""
        ...


def _extract_value(data: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    """Extract value from nested dict using path tuple.

    Args:
        data: Nested dictionary to extract from
        path: Tuple of keys forming the path to the value

    Returns:
        Value at the specified path

    Raises:
        KeyError: If path doesn't exist in data
    """
    for key in path:
        data = data[key]
    return data


def _generate_model_identifier_from_metadata(metadata: ArtifactMetadata) -> str:
    """Generate model identifier from metadata for cache filenames.

    Wrapper around shared generate_model_identifier that extracts values from metadata.

    Args:
        metadata: Complete artifact metadata

    Returns:
        Model identifier string (e.g., "77k" or "8L256D")
    """
    return generate_model_identifier(
        config_name=metadata.config_name,
        encoder_config=metadata.training_config.model.encoder,
    )


def _generate_head_identifier(head_names: List[str]) -> str:
    """Generate head identifier from list of head names.

    Args:
        head_names: List of prediction head type strings (HeadType.value)

    Returns:
        Head identifier string (e.g., "SelectionFactorHead" or "DMSFunctionalScoreHead")
    """
    if not head_names:
        return "unknown_head"

    # Use first head name (most common case: single head)
    # For multiple heads, join with underscore
    return "_".join(sorted(head_names))


def _build_cache_filename(
    head_name: str,
    model_id: str,
    dataset_nickname: str,
    split: str,
    artifact_type: str,
    checkpoint_id: Optional[str] = None,
    row_filter_digest: Optional[str] = None,
) -> str:
    """Build cache filename from components.

    Format: {head_name}_{model_id}[_{checkpoint_digest}][_rf{row_filter_digest}]_{dataset_nickname}_{split}_{artifact_type}.zip

    When checkpoint_id is provided, an 8-hex-char SHA-256 digest of it (same
    truncation convention as ``naming.compute_metadata_hash``) is included in
    the filename to disambiguate predictions from different checkpoints of
    the same architecture. This prevents a bug where cycle0's predictions
    would be silently reused for cycle1, cycle2, etc. The digest, not the raw
    id, is embedded: ``checkpoint_id`` is set to a checkpoint's full filename
    stem (``serialization/checkpoints.py``), which for joint multi-head,
    multi-dataset training is already long, and embedding it raw here on top
    of the other components previously produced filenames over the 255-byte
    limit most Linux filesystems enforce, so ``Path.exists()`` raised
    ``OSError: [Errno 36] File name too long`` instead of a cache miss.

    Args:
        head_name: Prediction head identifier
        model_id: Model identifier (e.g., "77k" or "8L256D")
        dataset_nickname: Dataset identifier
        split: Dataset split ("train", "val", "test")
        artifact_type: Type of artifact ("branch_lengths", "predictions")
        checkpoint_id: Optional checkpoint-specific identifier (e.g., checkpoint
            filename stem). When provided, its digest ensures different
            checkpoints of the same architecture produce distinct cache
            entries.
        row_filter_digest: Optional digest of the dataset's selected rows
            (``ParquetBackend.row_filter_digest``). The nickname comes from
            the data spec alone, so without it two equal-sized shards of one
            spec shared a cache entry and the second was served the first's
            branch lengths. Omitted from the name when None, so
            unfiltered datasets keep their filenames. It goes before the
            nickname so that the filename's suffix,
            ``{dataset_nickname}_{split}_{artifact_type}.zip``, which
            experiment scripts glob for, is unchanged.

    Returns:
        Cache filename string
    """
    prefix = f"{head_name}_{model_id}"
    if checkpoint_id is not None:
        checkpoint_digest = hashlib.sha256(checkpoint_id.encode()).hexdigest()[:8]
        prefix = f"{prefix}_{checkpoint_digest}"
    if row_filter_digest is not None:
        prefix = f"{prefix}_rf{row_filter_digest}"
    return f"{prefix}_{dataset_nickname}_{split}_{artifact_type}.zip"


def _is_cacheable_dataset(dataset: Any) -> bool:
    """Check if dataset should be cached.

    Datasets must implement CacheableDataset protocol and have a non-default
    nickname. Datasets with nickname="unnamed_dataset" are not cached to avoid
    collisions.

    The check uses ``hasattr`` rather than ``isinstance``: from Python 3.12,
    ``isinstance`` against a runtime-checkable Protocol skips ``__getattr__``,
    so it rejects delegating wrappers such as ``ESM2DatasetWrapper``.

    Args:
        dataset: Dataset to check (should implement CacheableDataset protocol)

    Returns:
        True if dataset should be cached, False for the default
        "unnamed_dataset" nickname

    Raises:
        TypeError: If dataset lacks a CacheableDataset attribute.
    """
    missing = [
        attr
        for attr in ("nickname", "row_filter_digest", "__len__")
        if not hasattr(dataset, attr)
    ]
    if missing:
        raise TypeError(
            f"{type(dataset).__name__} reached the BL cache without {missing} "
            "(CacheableDataset)"
        )
    nickname: str = dataset.nickname
    return nickname != "unnamed_dataset"


class TrainingArtifact(Generic[T]):
    """Generic container for cacheable training artifacts with metadata.

    Container for cached training artifacts that can be validated against
    context and persisted to disk. The generic parameter T represents
    the type of data being cached (e.g., Tensor for branch lengths).

    Subclasses should define nested ValidationFields classes for training
    and evaluation validation levels, and set class attributes for
    artifact_type_name and default_validation_level.

    Args:
        artifact_type: Type of artifact (e.g., "branch_lengths", "predictions")
        data: The actual artifact data
        metadata: Complete artifact metadata for validation
    """

    # Class attributes - subclasses must override these
    artifact_type_name: ClassVar[str] = "unknown"
    """Artifact type identifier for cache filenames."""

    # Validation paths - subclasses override for different validation strictness
    validation_paths: ClassVar[List[Tuple[str, ...]]] = [
        ("training_config",),
    ]
    """Paths to extract and compare for cache validation."""

    def __init__(self, artifact_type: str, data: T, metadata: ArtifactMetadata) -> None:
        """Initialize training artifact.

        Args:
            artifact_type: Type of artifact (e.g., "branch_lengths", "predictions")
            data: The actual artifact data
            metadata: Complete artifact metadata for validation
        """
        self.artifact_type = artifact_type
        self.data = data
        self.metadata = metadata

    def save(self, path: Path) -> None:
        """Save artifact with metadata to ZIP archive.

        Creates a ZIP archive containing:
        - metadata.yml: Human-readable YAML serialization
        - data.pt: Torch-serialized artifact data

        Args:
            path: File path to save to
        """
        write_model_archive(
            path=path,
            metadata=self.metadata,
            data=self.data,
            data_filename="data.pt",
        )

    @classmethod
    def load(cls, path: Path) -> "TrainingArtifact[T]":
        """Load artifact from ZIP archive.

        Args:
            path: File path to load from

        Returns:
            New TrainingArtifact instance with loaded data

        Raises:
            FileNotFoundError: If artifact file doesn't exist
            ValueError: If file is not a .zip file
        """
        metadata_dict, data = read_model_archive(
            path=path,
            data_filename="data.pt",
            device="cpu",
            weights_only=False,
        )

        # Reconstruct ArtifactMetadata from dict
        metadata = ArtifactMetadata(**metadata_dict)

        return cls(metadata.artifact_type, data, metadata)

    @classmethod
    def try_load(cls, path: Path) -> Optional["TrainingArtifact[T]"]:
        """Try to load artifact, return None if file doesn't exist or fails.

        Args:
            path: File path to load from

        Returns:
            TrainingArtifact instance if successful, None otherwise
        """
        try:
            return cls.load(path)
        except (FileNotFoundError, ValueError):
            return None
        except Exception as e:
            print(f"Warning: Unexpected error loading artifact: {e}")
            return None

    @classmethod
    def validate_metadata(
        cls,
        stored: ArtifactMetadata,
        expected: ArtifactMetadata,
    ) -> bool:
        """Validate metadata compatibility between stored and expected.

        Extracts values at specified paths and compares them.

        Args:
            stored: Metadata from cached artifact
            expected: Metadata from current context

        Returns:
            True if all values at validation paths match, False otherwise
        """
        stored_dict = stored.model_dump()
        expected_dict = expected.model_dump()

        return all(
            _extract_value(stored_dict, path) == _extract_value(expected_dict, path)
            for path in cls.validation_paths
        )

    @classmethod
    def try_load_for_dataset(
        cls,
        cache_dir: Path,
        trainer: "Trainer",
        dataset: "BranchLengthDatasetBundle",
        split: str,
        head_type: Optional["HeadType"] = None,
    ) -> Optional[T]:
        """Try to load cached artifact for dataset.

        Args:
            cache_dir: Directory containing cached artifacts
            trainer: Trainer instance (provides metadata)
            dataset: Dataset to load artifact for
            split: Split identifier ("train", "val", "test")
            head_type: Optional HeadType for head-specific artifacts

        Returns:
            Cached data if found and valid, None otherwise
        """
        if not _is_cacheable_dataset(dataset):
            return None

        # Get metadata from trainer and create artifact metadata
        base_metadata = trainer.export_metadata()
        expected_metadata = ArtifactMetadata.from_base_metadata(
            artifact_type=cls.artifact_type_name,
            base_metadata=base_metadata,
        )

        # Build cache key
        model_id = _generate_model_identifier_from_metadata(expected_metadata)
        head_name = head_type.value if head_type else "all_heads"
        checkpoint_id = getattr(trainer, "checkpoint_id", None)
        cache_key = _build_cache_filename(
            head_name=head_name,
            model_id=model_id,
            dataset_nickname=dataset.nickname,
            split=split,
            artifact_type=cls.artifact_type_name,
            checkpoint_id=checkpoint_id,
            row_filter_digest=dataset.row_filter_digest,
        )

        cache_path = cache_dir / cache_key
        if not cache_path.exists():
            return None

        try:
            # Load artifact
            loaded = cls.load(cache_path)

            # Validate using our validation method
            if cls.validate_metadata(loaded.metadata, expected_metadata):
                # Check shape
                data = loaded.data
                if len(data) != len(dataset):
                    print(
                        f"Cache miss: shape mismatch for {dataset.nickname} "
                        f"(cached: {len(data)}, dataset: {len(dataset)})"
                    )
                    return None

                print(
                    f"✓ Cache hit: Loading {cls.artifact_type_name} for {dataset.nickname} {split}"
                )
                return data
        except Exception as e:
            print(f"Warning: Cache load failed: {e}")

        return None

    @classmethod
    def save_for_dataset(
        cls,
        cache_dir: Path,
        result: T,
        trainer: "Trainer",
        dataset: "BranchLengthDatasetBundle",
        split: str,
        head_type: Optional["HeadType"] = None,
    ) -> None:
        """Save artifact for dataset.

        Args:
            cache_dir: Directory to store cached artifacts
            result: Computed data to cache
            trainer: Trainer instance (provides metadata)
            dataset: Dataset artifact was computed for
            split: Split identifier ("train", "val", "test")
            head_type: Optional HeadType for head-specific artifacts
        """
        if not _is_cacheable_dataset(dataset):
            return

        # Get metadata from trainer and create artifact metadata
        base_metadata = trainer.export_metadata()
        metadata = ArtifactMetadata.from_base_metadata(
            artifact_type=cls.artifact_type_name,
            base_metadata=base_metadata,
        )

        # Build cache key
        model_id = _generate_model_identifier_from_metadata(metadata)
        head_name = head_type.value if head_type else "all_heads"
        checkpoint_id = getattr(trainer, "checkpoint_id", None)
        cache_key = _build_cache_filename(
            head_name=head_name,
            model_id=model_id,
            dataset_nickname=dataset.nickname,
            split=split,
            artifact_type=cls.artifact_type_name,
            checkpoint_id=checkpoint_id,
            row_filter_digest=dataset.row_filter_digest,
        )

        # Create artifact using subclass's from_tensor
        artifact = cls.from_tensor(result, metadata)

        # Save to disk
        cache_path = cache_dir / cache_key
        artifact.save(cache_path)
        print(f"✓ Cached {cls.artifact_type_name} for {dataset.nickname}")

    @classmethod
    def from_tensor(cls, data: T, metadata: ArtifactMetadata) -> "TrainingArtifact[T]":
        """Create artifact from data tensor.

        Args:
            data: The tensor data to wrap
            metadata: Complete artifact metadata

        Returns:
            New artifact instance
        """
        return cls(cls.artifact_type_name, data, metadata)

    def to_tensor(self) -> T:
        """Get artifact data as tensor.

        Returns:
            The underlying tensor data
        """
        return self.data


class BranchLengthArtifact(TrainingArtifact[Tensor]):
    """Artifact for caching optimized branch lengths with context validation.

    Stores branch length optimizations from evolutionary training with
    Pydantic-based context validation to ensure cache validity.

    The data is stored as a simple tensor for efficient serialization.
    """

    artifact_type_name: ClassVar[str] = "branch_lengths"


def _loss_normalizations_by_head(
    metadata: ArtifactMetadata,
) -> Dict[str, set[Optional[str]]]:
    """Map each task's head type to the ``normalization`` its loss uses.

    Losses with no ``normalization`` field map to ``None``. A set per head,
    because one head can carry several tasks.
    """
    by_head: Dict[str, set[Optional[str]]] = {}
    for task in metadata.training_config.tasks:
        by_head.setdefault(task.head_type.value, set()).add(
            getattr(task.loss_config, "normalization", None)
        )
    return by_head


class LenientBranchLengthArtifact(BranchLengthArtifact):
    """Branch length artifact with lenient validation.

    Validates only what changes the branch-length objective: the
    neutral_model config and, for every head both configs train, the loss
    ``normalization`` (``DASMLossConfig``'s endpoint kernel).
    This allows reuse across training configurations that differ only in
    fields that leave the fitted lengths' meaning unchanged, such as hook
    configs.
    """

    validation_paths: ClassVar[List[Tuple[str, ...]]] = [
        ("training_config", "neutral_model"),
    ]

    @classmethod
    def validate_metadata(
        cls,
        stored: ArtifactMetadata,
        expected: ArtifactMetadata,
    ) -> bool:
        """Check the neutral model and each shared head's loss normalization.

        Args:
            stored: Metadata from cached artifact
            expected: Metadata from current context

        Returns:
            True if the neutral models match and every head present in both
            configs uses the same loss normalization, False otherwise
        """
        if not super().validate_metadata(stored, expected):
            return False
        stored_norms = _loss_normalizations_by_head(stored)
        expected_norms = _loss_normalizations_by_head(expected)
        return all(
            stored_norms[head] == expected_norms[head]
            for head in stored_norms.keys() & expected_norms.keys()
        )


class PredictionArtifact(TrainingArtifact[Tensor]):
    """Artifact for caching model predictions on a dataset.

    Stores raw prediction tensors from model heads (e.g., selection factors).
    The predictions can be post-processed into different formats as needed.

    Predictions always require exact match of training configuration since
    they depend on all model parameters.
    """

    artifact_type_name: ClassVar[str] = "predictions"


class ComputationCache:
    """File-based cache directory manager for training artifacts.

    Manages the cache directory for training artifacts. Individual artifact
    classes handle their own loading/saving logic via try_load_for_dataset
    and save_for_dataset class methods.

    Args:
        cache_dir: Directory to store cached artifacts
    """

    def __init__(self, cache_dir: str = ".dasm2_cache") -> None:
        """Initialize computation cache.

        Args:
            cache_dir: Directory to store cached artifacts
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clear_cache(self) -> None:
        """Clear all cached artifacts from disk.

        Removes all files in the cache directory.
        """
        for cache_file in self.cache_dir.glob("*"):
            if cache_file.is_file():
                cache_file.unlink()
