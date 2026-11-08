"""Frozen observed-context Thrifty tables and startup coverage validation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Sequence,
)

import numpy as np
import torch
from torch import Tensor

from dasm2.codon_table import NUM_SENSE_CODONS
from dasm2.neutral.base import RatesCSPNeutralModel
from dasm2.neutral.codon_boundary import SENSE_CODONS
from dasm2.neutral.kmer_encoding import KmerSequenceEncoder, encode_sequences_vectorized
from dasm2.sequence.constants import NT_SEPARATOR
from dasm2.utils.artifacts import resolve_artifact_path

if TYPE_CHECKING:
    from dasm2.neutral.thrifty import ThriftySHMModel

logger = logging.getLogger(__name__)
_FORMAT_VERSION = 1
DEFAULT_MAX_DECODED_CONTEXTS = 128
_CONTEXT_BATCH_SIZE = 64
_PROGRESS_INTERVAL = 4096
_NT_PER_CODON = 3
_NUM_BASES = 4
_NUM_TARGETS = _NUM_BASES - 1
_NUM_FEATURES = 1 + _NUM_BASES
_PRECISION_DTYPES = {"float16": torch.float16, "float32": torch.float32}
MAX_RELATIVE_TRANSITION_ERROR = 0.0005
_NON_ACGT = re.compile("[^ACGT]")
_TARGETS = torch.tensor(
    [
        [[i for i, base in enumerate("ACGT") if base != source] for source in codon]
        for codon in SENSE_CODONS
    ]
)
ContextKey = tuple[str, int, int]


def boundary_sites(raw_parent: str) -> Iterator[tuple[str, int]]:
    """Yield scored source codons whose 15-nt windows cross an edge or ambiguity.

    Enumerate the four segment-edge codons and the codons within six bases of
    each ambiguous nucleotide. This matches build_codon_nt_windows without
    allocating a tensor at every site during a complete corpus scan.

    Args:
        raw_parent: Parent nucleotide sequence with optional chain separators.

    Yields:
        Full segment and nucleotide offset of each complete sense codon requiring
        boundary features, ordered by segment and then position.
    """
    for segment in raw_parent.split(NT_SEPARATOR):
        starts = {0, 3, 3 * ((len(segment) - 3) // 3), 3 * ((len(segment) - 6) // 3)}
        for match in _NON_ACGT.finditer(segment):
            nt = match.start()
            starts.update(
                range(max(0, 3 * ((nt - 6) // 3)), min(len(segment) - 2, nt + 7), 3)
            )
        for start in sorted(starts):
            if start >= 0 and segment[start : start + 3] in SENSE_CODONS:
                yield segment, start


def model_fingerprint(model: ThriftySHMModel) -> str:
    """Bind table entries to the encoder, source order, architecture and weights."""
    # This is a full artifact identity including tensor bytes, not the truncated
    # TrainingConfig identifier from compute_metadata_hash. Keep the v1 byte
    # encoding stable so already-published tables remain loadable.
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "version": _FORMAT_VERSION,
                "site_count": model.site_count,
                "architecture": type(model.inference_model).__name__,
                "hyperparameters": model.inference_model.hyperparameters,
                "sources": SENSE_CODONS,
            },
            sort_keys=True,
        ).encode()
    )
    for name, value in sorted(model.inference_model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def custom_table_instructions(model_name: str) -> str:
    """Actionable recovery shared by missing-file and missing-context failures."""
    return (
        "Build a custom table from ALL training and testing PCP inputs:\n"
        f"  dasm2 build-boundary-table 'YOUR_PCP_DATA' --model-name={model_name} "
        "--output=/path/to/custom-boundaries.npz\n"
        "Replace YOUR_PCP_DATA with a registered nickname, file path, or '+'-joined "
        "PCP specs. Set neutral_model.boundary_table_path to the resulting file "
        "in the training YAML, or pass --boundary-table-path for oe-plot/likelihood. "
        "See docs/guides/thrifty-boundary-tables.md."
    )


def resolve_boundary_table_path(model: ThriftySHMModel) -> Path:
    """Find an explicit custom file or the common table in lab artifact storage."""
    spec = Path(
        model.boundary_table_path or f"{model.name}-common-v{_FORMAT_VERSION}.npz"
    ).expanduser()
    return resolve_artifact_path(
        spec,
        "thrifty_boundary",
        description="Thrifty boundary table is required before loading global-kmer data",
        instructions=custom_table_instructions(model.name),
        search_storage=model.boundary_table_path is None or spec.parent == Path("."),
    )


def validate_context_key(key: object, model: ThriftySHMModel) -> ContextKey:
    """Validate flank width and integer CNN padding before sorting or inference.

    Accept JSON lists as well as in-memory tuples. Raise ValueError for malformed
    keys or keys whose dimensions do not match the model's receptive field.
    """
    conv_radius = model.conv_radius
    flank_width = 2 * model.context_radius
    if not isinstance(key, (tuple, list)) or len(key) != 3:
        raise ValueError(
            f"Invalid boundary context key: expected ({flank_width}-character flanks, "
            f"integer left pad in [0, {conv_radius}], integer right pad in [0, {conv_radius}]); "
            f"received {key!r}"
        )
    flanks, left, right = key
    if (
        not isinstance(flanks, str)
        or len(flanks) != flank_width
        or type(left) is not int
        or type(right) is not int
        or not 0 <= left <= conv_radius
        or not 0 <= right <= conv_radius
    ):
        raise ValueError(
            f"Invalid boundary context key: expected ({flank_width}-character flanks, "
            f"integer left pad in [0, {conv_radius}], integer right pad in [0, {conv_radius}]); "
            f"received {key!r}"
        )
    return flanks, left, right


class ThriftyBoundaryTable:
    """Immutable context index with compressed source nucleotide transition rates.

    One entry covers all 61 sense codons and contains unadjusted transition rates.
    The artifact has no pickle payload. Copies of a neutral model share this
    read-only tensor instead of duplicating the complete table per dataset.
    """

    def __init__(
        self, keys: Sequence[Hashable], features: Tensor, metadata: dict[str, Any]
    ):
        """Construct a table; use load() for validated on-disk artifacts."""
        if (
            not isinstance(metadata.get("model_name"), str)
            or not metadata["model_name"]
        ):
            raise ValueError(
                f"Boundary metadata requires nonempty model_name; received {metadata.get('model_name')!r}"
            )
        self._index = {key: i for i, key in enumerate(keys)}
        self._features = features
        self.metadata = metadata
        self._decoded: OrderedDict[int, Tensor] = OrderedDict()

    def __deepcopy__(self, memo: dict[int, Any]) -> "ThriftyBoundaryTable":
        """Share frozen storage when dataset transforms copy their neutral model."""
        return self

    def __len__(self) -> int:
        """Number of distinct observed flank contexts."""
        return len(self._index)

    def __contains__(self, key: Hashable) -> bool:
        """Whether all source codons are available for this context."""
        return key in self._index

    def get(self, key: Hashable) -> Tensor:
        """Decode source-specific rates/CSPs, with a bounded decoded-entry cache."""
        try:
            index = self._index[key]
        except KeyError:
            raise ValueError(
                f"Uncovered Thrifty boundary context {key!r}. "
                "Dataset coverage must be validated before training or evaluation.\n"
                + custom_table_instructions(self.metadata["model_name"])
            ) from None
        if index not in self._decoded:
            transitions = self._features[index].float()
            rates = transitions.sum(-1, keepdim=True)
            csps = torch.zeros(
                (NUM_SENSE_CODONS, _NT_PER_CODON, _NUM_BASES), dtype=torch.float32
            )
            csps.scatter_(-1, _TARGETS, transitions / rates)
            if len(self._decoded) >= DEFAULT_MAX_DECODED_CONTEXTS:
                self._decoded.popitem(last=False)
            self._decoded[index] = torch.cat((rates, csps), dim=-1)
        self._decoded.move_to_end(index)
        return self._decoded[index]

    @classmethod
    def load(cls, path: Path, model: ThriftySHMModel) -> "ThriftyBoundaryTable":
        """Check format, model identity, key uniqueness and all transition values."""
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            features = torch.from_numpy(archive["features"])
        if metadata.get("version") != _FORMAT_VERSION or metadata.get(
            "model_fingerprint"
        ) != model_fingerprint(model):
            raise ValueError(
                f"Boundary table {path} does not match this Thrifty model/encoder or table format.\n"
                + custom_table_instructions(model.name)
            )
        if not isinstance(metadata.get("contexts"), list):
            raise ValueError(
                f"Boundary table {path}: expected a contexts list, received {metadata.get('contexts')!r}"
            )
        keys = [validate_context_key(key, model) for key in metadata.pop("contexts")]
        precision = metadata.get("precision")
        if precision not in _PRECISION_DTYPES:
            raise ValueError(
                f"Boundary table {path}: expected precision float16 or float32, received {precision!r}"
            )
        expected_dtype = _PRECISION_DTYPES[precision]
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"Boundary table {path}: expected unique keys, received {len(keys) - len(set(keys))} duplicates"
            )
        expected_shape = (len(keys), NUM_SENSE_CODONS, _NT_PER_CODON, _NUM_TARGETS)
        if features.shape != expected_shape:
            raise ValueError(
                f"Boundary table {path}: expected transition shape {expected_shape}, received {tuple(features.shape)}"
            )
        if features.dtype != expected_dtype:
            raise ValueError(
                f"Boundary table {path}: expected transition dtype {expected_dtype}, received {features.dtype}"
            )
        # Validate in chunks without expanding the entire compact table to float32.
        for chunk in features.split(4096):
            if (
                not torch.isfinite(chunk).all()
                or (chunk < 0).any()
                or (chunk.float().sum(-1) <= 0).any()
            ):
                raise ValueError(f"Invalid boundary transition rates in {path}")
        return cls(keys, features, metadata)


def context_features(model: ThriftySHMModel, keys: list[ContextKey]) -> Tensor:
    """Evaluate only the encoded receptive fields for a batch with equal CNN pads.

    Encode N-containing nucleotide windows, then remove entries outside the real
    encoder tensor before applying the CNN. Its zero-vector padding therefore
    occurs at exactly the original boundaries. This avoids processing the other
    ~487 tensor positions for every source/context in a large offline build.
    """
    if not keys:
        return torch.empty((0, NUM_SENSE_CODONS, _NT_PER_CODON, _NUM_FEATURES))
    left, right = keys[0][1:]
    if any(key[1:] != (left, right) for key in keys):
        raise ValueError("Context feature batches must have identical CNN padding")
    radius = len(keys[0][0]) // 2
    windows = [
        flanks[:radius] + codon + flanks[radius:]
        for flanks, _, _ in keys
        for codon in SENSE_CODONS
    ]
    kmer_half = model.kmer_length // 2
    encoder = KmerSequenceEncoder(model.kmer_length, len(windows[0]))
    encoded, masks, modifiers = encode_sequences_vectorized(windows, encoder)
    lo, hi = kmer_half + left, len(windows[0]) - kmer_half - right
    start = radius - lo
    with torch.no_grad():
        rates, logits = model.inference_model(
            encoded[:, lo:hi], masks[:, lo:hi], modifiers[:, lo:hi]
        )
        features = torch.cat(
            (
                rates[:, start : start + 3, None],
                logits[:, start : start + 3].softmax(-1),
            ),
            dim=-1,
        )
    return features.reshape(
        len(keys), NUM_SENSE_CODONS, _NT_PER_CODON, _NUM_FEATURES
    ).cpu()


def write_boundary_table(
    model: ThriftySHMModel,
    contexts: Iterable[Hashable],
    output: Path,
    provenance: dict[str, Any],
    precision: Literal["float16", "float32"] = "float16",
) -> dict[str, Any]:
    """Write compact source-specific nucleotide transition intensities atomically.

    Store only the three possible target rates at each source nucleotide. The
    default float16 precision bounds relative error of each positive transition
    by 0.05%; reject underflow/overflow or larger error. Products for three-hit
    codon transitions consequently differ by at most 0.151%. Float32 is available
    for users who require full precision. Neither representation stores Q's
    diagonal, forbidden self CSPs, parent-specific values or chain adjustments.
    """
    if output.exists():
        raise FileExistsError(
            f"Boundary table already exists: {output}; choose a new output path"
        )
    if precision not in ("float16", "float32"):
        raise ValueError("precision must be float16 or float32")
    keys = sorted(validate_context_key(key, model) for key in contexts)
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate boundary context keys")
    packed, max_error = _pack_transitions(model, keys, precision)
    metadata = {
        "version": _FORMAT_VERSION,
        "model_name": model.name,
        "model_fingerprint": model_fingerprint(model),
        "contexts": keys,
        "provenance": provenance,
        "precision": precision,
        "max_relative_transition_error": max_error,
    }
    _publish_atomically(output, packed, metadata)
    return {
        "contexts": len(keys),
        "feature_bytes": packed.numel() * packed.element_size(),
        "file_bytes": output.stat().st_size,
        "model_fingerprint": metadata["model_fingerprint"],
        "max_relative_transition_error": max_error,
        "precision": precision,
    }


def _pack_transitions(
    model: ThriftySHMModel,
    keys: list[ContextKey],
    precision: Literal["float16", "float32"],
) -> tuple[Tensor, float]:
    """Evaluate context batches and enforce the stored-transition error bound."""
    packed = torch.empty(
        (len(keys), NUM_SENSE_CODONS, _NT_PER_CODON, _NUM_TARGETS),
        dtype=_PRECISION_DTYPES[precision],
    )
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key[1:]].append(i)
    max_error = 0.0
    done = 0
    next_report = _PROGRESS_INTERVAL
    for indices in groups.values():
        for offset in range(0, len(indices), _CONTEXT_BATCH_SIZE):
            batch = indices[offset : offset + _CONTEXT_BATCH_SIZE]
            features = context_features(model, [keys[i] for i in batch])
            transitions = features[..., :1] * features[..., 1:].gather(
                -1, _TARGETS[None].expand(len(batch), -1, -1, -1)
            )
            stored = transitions.to(packed.dtype)
            error = (
                (
                    (stored.float() - transitions).abs()
                    / transitions.clamp_min(torch.finfo(torch.float32).tiny)
                )
                .max()
                .item()
            )
            if (
                not torch.isfinite(stored).all()
                or error > MAX_RELATIVE_TRANSITION_ERROR
            ):
                raise ValueError(
                    "Boundary transition quantization exceeds 0.05% or overflows; rebuild with --precision=float32"
                )
            max_error = max(max_error, error)
            packed[batch] = stored
            done += len(batch)
            if done >= next_report or done == len(keys):
                logger.info("Computed %d/%d boundary contexts", done, len(keys))
                next_report = done + _PROGRESS_INTERVAL
    return packed, max_error


def _publish_atomically(output: Path, packed: Tensor, metadata: dict[str, Any]) -> None:
    """Publish a complete artifact once, without overwriting another builder.

    Atomic mkdir serializes the existence check and rename on shared NFS; a
    bare replace could overwrite a competing writer between check and rename.
    The temporary file lives on the destination filesystem for atomic rename.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    # Rename a complete file within the destination filesystem (including NFS).
    # Serialize publication so simultaneous builders cannot overwrite each other.
    # A failed writer removes both its temporary data and its publication lock.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".boundary-") as temp:
        temporary_path = Path(temp) / "table.npz"
        np.savez_compressed(
            temporary_path, metadata=json.dumps(metadata), features=packed.numpy()
        )
        temporary_path.chmod(0o644)
        lock = output.with_name(output.name + ".lock")
        try:
            lock.mkdir()
        except FileExistsError:
            raise FileExistsError(
                f"Boundary publication lock exists: {lock}. Another builder holds, "
                "or died holding, this lock. Remove it only after confirming no "
                "builder is running, then retry."
            ) from None
        try:
            if output.exists():
                raise FileExistsError(f"Boundary table already exists: {output}")
            temporary_path.replace(output)
        finally:
            lock.rmdir()


class FrozenSourceCodonLookup:
    """Adapt a frozen Thrifty table to the sparse boundary record builder."""

    def __init__(
        self, neutral_model: RatesCSPNeutralModel, table: ThriftyBoundaryTable
    ):
        """Bind a model's context keys to an already validated immutable table."""
        self.neutral_model = neutral_model
        self.table = table

    def get(self, segment: str, start: int) -> Tensor:
        """Look up all source features without an inference fallback."""
        return self.table.get(
            self.neutral_model.source_codon_context_key(segment, start)
        )


def validate_boundary_coverage(
    model: Optional[RatesCSPNeutralModel], parents: Iterable[str], label: str
) -> None:
    """Scan every input before dataset preprocessing or worker iteration starts.

    Other neutral model families retain their own full-segment source oracle.
    A missing context (not an uncommon source codon) aborts with a custom-table
    command. Scanning needs only parent strings, so Parquet callers can stream
    selected parent columns without reading rates, CSPs or training tensors.

    Args:
        model: Neutral model used by the global-kmer dataset, optionally wrapped
            with multihit correction.
        parents: All selected parent sequences, including chain separators.
        label: Dataset description included in errors and coverage logging.

    Returns:
        None. Retains a validated table on Thrifty models for later lookups.

    Raises:
        FileNotFoundError: The required boundary artifact is missing.
        ValueError: No model was supplied, the artifact is invalid or incompatible,
            a parent exceeds the encoder limit, or an observed context is missing.
    """
    if model is None:
        raise ValueError("Global k-mer data requires a neutral_model")
    model.validate_source_coverage(parents, label)


def validate_thrifty_coverage(
    model: ThriftySHMModel, parents: Iterable[str], label: str
) -> None:
    """Load the selected table and reject missing contexts before preprocessing."""
    table = model.load_boundary_table()
    observed = set()
    missing = set()
    rows = 0
    for parent in parents:
        rows += 1
        for key in parent_context_keys(model, parent, label, rows):
            if key not in observed:
                observed.add(key)
                if key not in table:
                    missing.add(key)
    if missing:
        raise ValueError(
            f"Thrifty boundary coverage failed for {label}: {len(missing)} missing "
            f"contexts among {len(observed)} contexts in {rows} PCP rows. "
            f"Examples: {sorted(missing, key=repr)[:3]}. No training/evaluation steps have started.\n"
            + custom_table_instructions(model.name)
        )
    logger.info(
        "Thrifty boundary coverage: %s, %d PCP rows, %d contexts, zero missing",
        label,
        rows,
        len(observed),
    )


def parent_context_keys(
    model: ThriftySHMModel, parent: str, label: str, row: int
) -> Iterator[ContextKey]:
    """Validate encoder length, then enumerate a parent's boundary contexts."""
    lengths = [len(segment) for segment in parent.split(NT_SEPARATOR)]
    if any(length > model.site_count for length in lengths):
        raise ValueError(
            f"Thrifty input in {label}, PCP row {row}, has segment lengths {lengths}; "
            f"expected each <= site_count={model.site_count}. "
            "Filter overlength parent/child pairs before training/evaluation, as DASM does. "
            "A custom boundary table cannot extend the neutral model's encoder."
        )
    for segment, start in boundary_sites(parent):
        yield model.source_codon_context_key(segment, start)
