"""Native DASM2 implementation of Thrifty SHM rate prediction.

This module provides a native implementation of the Thrifty somatic hypermutation
(SHM) model, eliminating the runtime dependency on netam for inference while
maintaining full compatibility with netam checkpoints.

Two CNN architectures are supported, matching netam's pretrained Thrifty models:
    - JoinedRSCNNModel (e.g. ThriftyHumV0.2-59): shared k-mer embedding + CNN,
      separate rate (r_linear) and CSP (s_linear) output heads.
    - IndepRSCNNModel (e.g. ThriftyHumV0.2-20/-45): fully independent rate and
      CSP branches (each with its own embedding, CNN, and head).

Architecture (from ThriftyHumV0.2-59, JoinedRSCNNModel):
    - 3-mer encoding of parent sequence (kmer_length: 3)
    - shared CNN layers, separate rate/CSP output heads
        - kmer_embedding: 64 kmers → embedding_dim (7)
        - conv: 1D CNN (filter_count=19, kernel_size=11)
        - r_linear: → log_rates
        - s_linear: → CSP logits
    - Outputs: rates = exp(log_rates), csps = softmax(logits)
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, ValidationError

from dasm2.neutral.base import RatesCSPNeutralModel, SourceCodonFeatureLookup
from dasm2.neutral.boundary_table import (
    ContextKey,
    FrozenSourceCodonLookup,
    ThriftyBoundaryTable,
    custom_table_instructions,
    resolve_boundary_table_path,
    validate_thrifty_coverage,
)
from dasm2.neutral.kmer_encoding import (
    KmerSequenceEncoder,
    encode_sequences,
    encode_sequences_vectorized,
    generate_kmers,
)
from dasm2.serialization.zip_io import (
    data_archive_exists,
    read_data_archive,
    write_data_archive,
)

if TYPE_CHECKING:
    from dasm2.config import ThriftySHMNeutralModelConfig

logger = logging.getLogger(__name__)

# Name of the torch state-dict file stored inside ThriftySHM zip archives.
_WEIGHTS_FILENAME = "weights.pth"

# The pretrained Thrifty SHM models repackaged from netam's thrifty-models
# v0.2.0 release and published as catalog entries (see model_catalog.yaml and
# scripts/build_thrifty_neutral_models.py). Defined here so build tooling and
# tests share a single source of truth without importing from ``scripts``.
PRETRAINED_THRIFTY_MODEL_NAMES = [
    "ThriftyHumV0.2-20",
    "ThriftyHumV0.2-45",
    "ThriftyHumV0.2-59",
]


class ThriftySHMArchiveMetadata(BaseModel):
    """Metadata for a ThriftySHMModel zip archive.

    Stored as ``metadata.yml`` inside the archive alongside the ``weights.pth``
    torch state dict. Carries the architecture hyperparameters needed to
    reconstruct the ``JoinedRSCNNModel`` or ``IndepRSCNNModel`` and the encoder
    ``site_count``.
    """

    model_type: Literal["ThriftySHMModel"] = Field(
        default="ThriftySHMModel",
        description="Model class identifier for validation",
    )
    model_name: str = Field(description="Human-readable model name")
    architecture: Literal["joined", "indep"] = Field(
        default="joined",
        description=(
            "CNN architecture: 'joined' (shared rate/CSP CNN, e.g. "
            "ThriftyHumV0.2-59) or 'indep' (independent rate/CSP CNNs, e.g. "
            "ThriftyHumV0.2-20/-45)"
        ),
    )
    archive_version: int = Field(
        default=1,
        description="Archive format version for future compatibility",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the model and its provenance",
    )
    kmer_length: int = Field(description="K-mer length (typically 3)")
    embedding_dim: int = Field(description="K-mer embedding dimension")
    filter_count: int = Field(description="Number of convolutional filters")
    kernel_size: int = Field(description="Convolutional kernel size")
    dropout_prob: float = Field(default=0.1, description="Dropout probability")
    site_count: int = Field(gt=0, description="Maximum sequence length for encoding")


class JoinedRSCNNModel(nn.Module):
    """CNN model that predicts both mutation rates and conditional substitution probs.

    This architecture shares embedding and convolutional layers between rate and CSP
    prediction, with separate linear output heads. This is the native DASM2 equivalent
    of netam's JoinedRSCNNModel.

    Architecture:
        kmer_embedding → conv → dropout → (r_linear, s_linear)
    """

    def __init__(
        self,
        kmer_length: int,
        embedding_dim: int,
        filter_count: int,
        kernel_size: int,
        dropout_prob: float = 0.1,
    ):
        """Initialize the JoinedRSCNNModel.

        Args:
            kmer_length: Length of k-mers (typically 3 for Thrifty models)
            embedding_dim: Dimension of k-mer embeddings
            filter_count: Number of convolutional filters
            kernel_size: Size of convolutional kernel
            dropout_prob: Dropout probability
        """
        super().__init__()

        self.kmer_length = kmer_length
        all_kmers = generate_kmers(kmer_length)
        self.kmer_count = len(all_kmers)

        self.kmer_embedding = nn.Embedding(self.kmer_count, embedding_dim)
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filter_count,
            kernel_size=kernel_size,
            padding="same",
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.r_linear = nn.Linear(in_features=filter_count, out_features=1)
        self.s_linear = nn.Linear(in_features=filter_count, out_features=4)

    def forward(
        self,
        encoded_parents: torch.Tensor,
        masks: torch.Tensor,
        wt_base_modifier: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing rates and CSP logits.

        Args:
            encoded_parents: K-mer encoded sequences, shape (batch, seq_len)
            masks: Valid position mask, shape (batch, seq_len)
            wt_base_modifier: Wild-type base modifier for CSPs, shape (batch, seq_len, 4)

        Returns:
            Tuple of (rates, csp_logits):
                - rates: Mutation rates, shape (batch, seq_len)
                - csp_logits: CSP logits (NOT softmaxed), shape (batch, seq_len, 4)
        """
        # Embedding and convolution
        kmer_embeds = self.kmer_embedding(encoded_parents)
        kmer_embeds = kmer_embeds.permute(0, 2, 1)  # (batch, embed, seq)
        conv_out = F.relu(self.conv(kmer_embeds))
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)  # (batch, seq, filter)

        # Rate prediction
        log_rates = self.r_linear(conv_out).squeeze(-1)
        rates = torch.exp(log_rates * masks)

        # CSP prediction
        csp_logits = self.s_linear(conv_out)
        csp_logits *= masks.unsqueeze(-1)
        csp_logits += wt_base_modifier

        return rates, csp_logits

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Return model hyperparameters."""
        return {
            "kmer_length": self.kmer_length,
            "embedding_dim": self.kmer_embedding.embedding_dim,
            "filter_count": self.conv.out_channels,
            "kernel_size": self.conv.kernel_size[0],
            "dropout_prob": self.dropout.p,
        }


class IndepRSCNNModel(nn.Module):
    """CNN model with fully independent rate and CSP branches.

    The native DASM2 equivalent of netam's IndepRSCNNModel: unlike
    JoinedRSCNNModel, the rate (``r_``) and CSP (``s_``) predictors share no
    weights — each has its own k-mer embedding, convolution, and output head.
    Used by the ThriftyHumV0.2-20 and ThriftyHumV0.2-45 pretrained models.

    Architecture (per branch):
        kmer_embedding → conv → dropout → linear
    """

    def __init__(
        self,
        kmer_length: int,
        embedding_dim: int,
        filter_count: int,
        kernel_size: int,
        dropout_prob: float = 0.1,
    ):
        """Initialize the IndepRSCNNModel.

        Args:
            kmer_length: Length of k-mers (typically 3 for Thrifty models)
            embedding_dim: Dimension of k-mer embeddings
            filter_count: Number of convolutional filters
            kernel_size: Size of convolutional kernel
            dropout_prob: Dropout probability
        """
        super().__init__()

        self.kmer_length = kmer_length
        all_kmers = generate_kmers(kmer_length)
        self.kmer_count = len(all_kmers)

        # Independent rate (r_) branch
        self.r_kmer_embedding = nn.Embedding(self.kmer_count, embedding_dim)
        self.r_conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filter_count,
            kernel_size=kernel_size,
            padding="same",
        )
        self.r_dropout = nn.Dropout(dropout_prob)
        self.r_linear = nn.Linear(in_features=filter_count, out_features=1)

        # Independent CSP (s_) branch
        self.s_kmer_embedding = nn.Embedding(self.kmer_count, embedding_dim)
        self.s_conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filter_count,
            kernel_size=kernel_size,
            padding="same",
        )
        self.s_dropout = nn.Dropout(dropout_prob)
        self.s_linear = nn.Linear(in_features=filter_count, out_features=4)

    def forward(
        self,
        encoded_parents: torch.Tensor,
        masks: torch.Tensor,
        wt_base_modifier: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing rates and CSP logits via independent branches.

        Args:
            encoded_parents: K-mer encoded sequences, shape (batch, seq_len)
            masks: Valid position mask, shape (batch, seq_len)
            wt_base_modifier: Wild-type base modifier for CSPs, shape (batch, seq_len, 4)

        Returns:
            Tuple of (rates, csp_logits):
                - rates: Mutation rates, shape (batch, seq_len)
                - csp_logits: CSP logits (NOT softmaxed), shape (batch, seq_len, 4)
        """
        # Rate branch
        r_kmer_embeds = self.r_kmer_embedding(encoded_parents)
        r_kmer_embeds = r_kmer_embeds.permute(0, 2, 1)  # (batch, embed, seq)
        r_conv_out = F.relu(self.r_conv(r_kmer_embeds))
        r_conv_out = self.r_dropout(r_conv_out)
        r_conv_out = r_conv_out.permute(0, 2, 1)  # (batch, seq, filter)
        log_rates = self.r_linear(r_conv_out).squeeze(-1)
        rates = torch.exp(log_rates * masks)

        # CSP branch
        s_kmer_embeds = self.s_kmer_embedding(encoded_parents)
        s_kmer_embeds = s_kmer_embeds.permute(0, 2, 1)  # (batch, embed, seq)
        s_conv_out = F.relu(self.s_conv(s_kmer_embeds))
        s_conv_out = self.s_dropout(s_conv_out)
        s_conv_out = s_conv_out.permute(0, 2, 1)  # (batch, seq, filter)
        csp_logits = self.s_linear(s_conv_out)
        csp_logits *= masks.unsqueeze(-1)
        csp_logits += wt_base_modifier

        return rates, csp_logits

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Return model hyperparameters (read from the rate branch)."""
        return {
            "kmer_length": self.kmer_length,
            "embedding_dim": self.r_kmer_embedding.embedding_dim,
            "filter_count": self.r_conv.out_channels,
            "kernel_size": self.r_conv.kernel_size[0],
            "dropout_prob": self.r_dropout.p,
        }


# Maps the ``architecture`` field stored in archive metadata to the nn.Module
# that implements it. Both modules share the (rates, csp_logits) forward
# signature, so ThriftySHMModel is otherwise architecture-agnostic.
_ARCHITECTURES: Dict[str, type] = {
    "joined": JoinedRSCNNModel,
    "indep": IndepRSCNNModel,
}


def _architecture_name(model: nn.Module) -> Literal["joined", "indep"]:
    """Return the metadata ``architecture`` key for a model instance."""
    if isinstance(model, IndepRSCNNModel):
        return "indep"
    if isinstance(model, JoinedRSCNNModel):
        return "joined"
    raise ValueError(f"Unsupported Thrifty architecture: {type(model).__name__}")


class ThriftySHMModel(RatesCSPNeutralModel):
    """Native DASM2 implementation of Thrifty SHM rate prediction.

    This class provides a native implementation of Thrifty models loaded from
    portable zip archives. It eliminates the runtime dependency on netam for
    inference while maintaining full output compatibility.

    Example:
        >>> from dasm2.model_catalog import resolve_model
        >>> path = resolve_model("ThriftyHumV0.2-59")
        >>> model = ThriftySHMModel.from_catalog("ThriftyHumV0.2-59", path)
        >>> rates, csps = model.compute_rates_and_csps(["ACGTACGTACGT"])
    """

    def __init__(
        self,
        model: Union[JoinedRSCNNModel, IndepRSCNNModel],
        site_count: int,
        model_name: str = "unnamed",
        boundary_table_path: Optional[str] = None,
    ):
        """Initialize ThriftySHMModel from component layers.

        Use ThriftySHMModel.from_catalog() to load a model from the catalog, or
        from_pretrained() to load directly from a zip archive.

        Args:
            model: JoinedRSCNNModel or IndepRSCNNModel instance with loaded weights
            site_count: Maximum sequence length for encoding
            model_name: Human-readable model identifier
            boundary_table_path: Custom frozen boundary table, or the common artifact.
        """
        self._model = model
        self._model.eval()  # Always in eval mode for inference
        self._site_count = site_count
        self._model_name = model_name
        self._boundary_table_path = boundary_table_path
        self._boundary_table: Optional["ThriftyBoundaryTable"] = None

        # Create encoder for sequence preprocessing
        self._encoder = KmerSequenceEncoder(
            kmer_length=model.kmer_length,
            site_count=site_count,
        )

    @property
    def inference_model(self) -> Union[JoinedRSCNNModel, IndepRSCNNModel]:
        """Frozen CNN used for encoded inference and checkpoint fingerprinting."""
        return self._model

    @property
    def kmer_length(self) -> int:
        """Nucleotide width of the encoder's k-mers."""
        return self._encoder.kmer_length

    @property
    def conv_radius(self) -> int:
        """Encoded positions on either side of a CNN prediction."""
        return int(self.inference_model.hyperparameters["kernel_size"]) // 2

    @property
    def context_radius(self) -> int:
        """Nucleotide flank width including encoder k-mer overhang."""
        return self.conv_radius + self.kmer_length // 2

    def validate_source_coverage(self, parents: Iterable[str], label: str) -> None:
        """Require frozen entries for every selected parent before preprocessing."""
        validate_thrifty_coverage(self, parents, label)

    def source_codon_lookup(self) -> SourceCodonFeatureLookup:
        """Require a prevalidated table; never silently evaluate missing entries."""
        if self.boundary_table is None:
            raise ValueError(
                "Thrifty boundary table is not loaded. Validate source coverage "
                "before constructing a global-kmer dataset, or explicitly select "
                "the oracle for simulation/audits.\n"
                + custom_table_instructions(self.name)
            )
        return FrozenSourceCodonLookup(self, self.boundary_table)

    def set_boundary_table_path(self, path: str) -> None:
        """Select a custom table and invalidate the previously loaded artifact."""
        self.boundary_table_path = path

    @property
    def boundary_table(self) -> Optional["ThriftyBoundaryTable"]:
        """Previously validated table, without performing I/O or inference."""
        return self._boundary_table

    def load_boundary_table(self) -> "ThriftyBoundaryTable":
        """Load and validate the selected artifact once, retaining frozen storage.

        Raises:
            FileNotFoundError: No selected artifact can be found.
            ValueError: The artifact is incompatible with this model or malformed.
        """
        if self._boundary_table is None:
            self._boundary_table = ThriftyBoundaryTable.load(
                resolve_boundary_table_path(self), self
            )
        return self._boundary_table

    @property
    def boundary_table_path(self) -> Optional[str]:
        """Custom boundary artifact, or None for the shared common table."""
        return self._boundary_table_path

    @boundary_table_path.setter
    def boundary_table_path(self, path: Optional[str]) -> None:
        """Invalidate loaded lookup state when selecting a different artifact."""
        self._boundary_table_path = path
        self._boundary_table = None

    @property
    def name(self) -> str:
        """Human-readable identifier for this neutral model."""
        return self._model_name

    @property
    def site_count(self) -> int:
        """Maximum sequence length this model can handle."""
        return self._site_count

    def to_config(self) -> "ThriftySHMNeutralModelConfig":
        """Reconstruct ThriftySHMNeutralModelConfig from neutral model state.

        ThriftySHM configs are catalog-only: ``model_name`` must resolve via
        ``resolve_model()`` for the config to round-trip on another machine
        (unlike TabularRates/GTRCodon, ThriftySHM has no ``model_path`` escape
        hatch). Prefers the catalog name recorded by ``from_catalog()``; falls
        back to the archive metadata name only if it is itself a catalog entry.

        Returns:
            ThriftySHMNeutralModelConfig matching current neutral model

        Raises:
            ValueError: If the resolved name is not a catalog entry, since the
                resulting config would parse but fail at load time.
        """
        # Import here to avoid circular dependency
        from dasm2.config import ThriftySHMNeutralModelConfig  # noqa: PLC0415
        from dasm2.model_catalog import get_catalog_entry  # noqa: PLC0415

        name = self._catalog_name or self._model_name
        if get_catalog_entry(name) is None:
            raise ValueError(
                f"Cannot build a portable config for ThriftySHMModel '{name}': "
                "it is not an entry in model_catalog.yaml, so the resulting "
                "config could not be resolved at load time. Load the model via "
                "from_catalog() with a name listed in the catalog."
            )

        return ThriftySHMNeutralModelConfig(
            model_name=name,
            boundary_table_path=self.boundary_table_path,
            multihit_model_name=None,
        )

    def save_pretrained(self, path: Union[str, Path], description: str = "") -> None:
        """Save the model to a portable zip archive.

        Creates a zip archive containing:
        - metadata.yml: Architecture hyperparameters, site_count, and model name
        - weights.pth: torch state dict of the JoinedRSCNNModel or IndepRSCNNModel

        Args:
            path: Output path for the zip archive (should end in .zip)
            description: Optional human-readable provenance description

        Example:
            model.save_pretrained("ThriftyHumV0.2-59.zip")
        """
        path = Path(path)

        hyperparameters = self._model.hyperparameters
        metadata = ThriftySHMArchiveMetadata(
            model_name=self._model_name,
            architecture=_architecture_name(self._model),
            description=description,
            kmer_length=hyperparameters["kmer_length"],
            embedding_dim=hyperparameters["embedding_dim"],
            filter_count=hyperparameters["filter_count"],
            kernel_size=hyperparameters["kernel_size"],
            dropout_prob=hyperparameters["dropout_prob"],
            site_count=self._site_count,
        )

        weights_buffer = BytesIO()
        torch.save(self._model.state_dict(), weights_buffer)

        write_data_archive(
            path, metadata, {_WEIGHTS_FILENAME: weights_buffer.getvalue()}
        )
        logger.info(f"Saved ThriftySHMModel '{self._model_name}' to {path}")

    @classmethod
    def from_pretrained(
        cls, model_path: Union[str, Path], *, boundary_table_path: Optional[str] = None
    ) -> "ThriftySHMModel":
        """Load a Thrifty model from a pretrained zip archive.

        Loads the model architecture and weights from a zip archive created by
        ``save_pretrained()`` (see ``scripts/build_thrifty_neutral_models.py``
        for repackaging netam crepes), producing a native DASM2 model instance.

        Args:
            model_path: Path to zip archive created by save_pretrained()
            boundary_table_path: Optional custom frozen boundary artifact.

        Returns:
            Initialized ThriftySHMModel with loaded weights

        Raises:
            FileNotFoundError: If the archive doesn't exist or is missing files
            ValueError: If the archive format is invalid
        """
        path = Path(model_path)
        logger.info(f"Loading ThriftySHMModel from {path}")

        if not data_archive_exists(path, [_WEIGHTS_FILENAME]):
            raise FileNotFoundError(
                f"Invalid or missing archive at {path}. "
                f"Expected zip with metadata.yml and {_WEIGHTS_FILENAME}."
            )

        raw_metadata, files = read_data_archive(path, [_WEIGHTS_FILENAME])

        # Validate the raw dict against the same Pydantic schema used to write it,
        # so reads fail fast on malformed metadata and write/read share one typed
        # model. ``model_type`` (Literal), ``architecture`` (Literal, default
        # "joined"), ``dropout_prob`` (default 0.1), and ``site_count`` (gt=0) are
        # all enforced here, including pydantic defaults for archives that omit the
        # optional fields.
        try:
            metadata = ThriftySHMArchiveMetadata.model_validate(raw_metadata)
        except ValidationError as exc:
            raise ValueError(f"Archive at {path} has invalid metadata: {exc}") from exc

        model = _ARCHITECTURES[metadata.architecture](
            kmer_length=metadata.kmer_length,
            embedding_dim=metadata.embedding_dim,
            filter_count=metadata.filter_count,
            kernel_size=metadata.kernel_size,
            dropout_prob=metadata.dropout_prob,
        )

        # weights_only=True for security; the stored state dict is already
        # filtered to the architecture's expected keys at archive-build time.
        state_dict = torch.load(
            BytesIO(files[_WEIGHTS_FILENAME]), map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)

        return cls(model, metadata.site_count, metadata.model_name, boundary_table_path)

    @classmethod
    def from_catalog(
        cls,
        catalog_name: str,
        archive_path: Union[str, Path],
        *,
        boundary_table_path: Optional[str] = None,
    ) -> "ThriftySHMModel":
        """Load catalog weights with a custom boundary path configured at creation."""
        model = cls.from_pretrained(
            archive_path, boundary_table_path=boundary_table_path
        )
        model._catalog_name = catalog_name
        return model

    def source_codon_context_key(self, sequence: str, codon_start: int) -> ContextKey:
        """Key the finite flanks and CNN padding for all source-codon rescans.

        All 61 sources replace the center codon, so the observed parent codon
        is deliberately absent from the key. The nucleotide footprint includes
        both the k-mer overhang and convolution radius. Missing nucleotides use
        N, while separate counts preserve zero-vector convolution padding at
        the encoder tensor edges. Thus a real left edge cannot alias a run of
        Ns inside a sequence. Right-edge counts also preserve site_count
        truncation, independently of the biological segment's length.

        Args:
            sequence: Original full segment.
            codon_start: Nucleotide offset of a complete encoded source codon.

        Returns:
            Flank window with its center removed, plus left/right CNN pad counts.

        Raises:
            ValueError: The requested codon is not fully within encoded output.
        """
        if not 0 <= codon_start <= min(len(sequence), self.site_count) - 3:
            raise ValueError(
                f"Source codon at nt {codon_start} is not fully within segment "
                f"length {len(sequence)} and Thrifty site_count {self.site_count}"
            )
        conv_radius = self.conv_radius
        radius = self.context_radius
        lo, hi = codon_start - radius, codon_start + 3 + radius
        window = (
            "N" * max(0, -lo)
            + sequence[max(0, lo) : min(len(sequence), hi)]
            + "N" * max(0, hi - len(sequence))
        )
        flanks = window[:radius] + window[radius + 3 :]
        return (
            flanks,
            max(0, conv_radius - codon_start),
            max(0, codon_start + 3 + conv_radius - self.site_count),
        )

    def _rates_and_csps_for_encoded(
        self,
        encoded_parents: torch.Tensor,
        masks: torch.Tensor,
        wt_base_modifiers: torch.Tensor,
        batch_sequences: List[str],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run the model on an encoded batch and trim outputs to sequence lengths.

        Shared by ``compute_rates_and_csps`` and
        ``compute_rates_and_csps_vectorized``, which differ only in how the batch
        is encoded.

        Args:
            encoded_parents: K-mer encoded sequences, shape (batch, seq_len).
            masks: Valid-position mask, shape (batch, seq_len).
            wt_base_modifiers: Wild-type base modifier, shape (batch, seq_len, 4).
            batch_sequences: The sequences encoded above, used to trim per-sequence.

        Returns:
            Tuple of (nt_rates_list, nt_csps_list), one tensor per input sequence.
        """
        with torch.no_grad():
            rates, csp_logits = self._model(encoded_parents, masks, wt_base_modifiers)
            csps = torch.softmax(csp_logits, dim=-1)

        nt_rates_list = [rates[i, : len(seq)] for i, seq in enumerate(batch_sequences)]
        nt_csps_list = [csps[i, : len(seq)] for i, seq in enumerate(batch_sequences)]
        return nt_rates_list, nt_csps_list

    def compute_rates_and_csps(
        self,
        sequences: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute neutral rates and CSPs for one or more sequences.

        This is the reference path that processes the whole input at once. For
        large datasets prefer ``compute_rates_and_csps_vectorized``, which encodes
        ~100x faster and batches to bound peak memory.

        Args:
            sequences: Single sequence string or list of sequences
            context: Reserved for future use (e.g., structural context). Currently
                unused but included for interface compatibility with AbstractNeutralModel.

        Returns:
            Tuple of (nt_rates_list, nt_csps_list) as lists of tensors, one per sequence:
                - nt_rates_list: Each tensor has shape (seq_length,)
                - nt_csps_list: Each tensor has shape (seq_length, 4)
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        encoded_parents, masks, wt_base_modifiers = encode_sequences(
            sequences, self._encoder
        )
        return self._rates_and_csps_for_encoded(
            encoded_parents, masks, wt_base_modifiers, sequences
        )

    def compute_rates_and_csps_vectorized(
        self,
        sequences: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
        batch_size: int = 4096,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute neutral rates and CSPs using vectorized numpy encoding.

        Drop-in replacement for ``compute_rates_and_csps`` that uses
        ``encode_sequences_vectorized`` instead of ``encode_sequences``.
        Produces identical outputs but ~100x faster encoding on large batches,
        processing in chunks of ``batch_size`` to bound peak memory.

        Args:
            sequences: Single sequence string or list of sequences.
            context: Reserved for future use (interface compatibility).
            batch_size: Maximum sequences to process at once (default: 4096).

        Returns:
            Tuple of (nt_rates_list, nt_csps_list) — identical to
            ``compute_rates_and_csps``.
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        nt_rates_list: List[torch.Tensor] = []
        nt_csps_list: List[torch.Tensor] = []

        for batch_start in range(0, len(sequences), batch_size):
            batch_end = min(batch_start + batch_size, len(sequences))
            batch_sequences = sequences[batch_start:batch_end]

            # Vectorized encoding (~100x faster than encode_sequences)
            encoded_parents, masks, wt_base_modifiers = encode_sequences_vectorized(
                batch_sequences, self._encoder
            )
            batch_rates, batch_csps = self._rates_and_csps_for_encoded(
                encoded_parents, masks, wt_base_modifiers, batch_sequences
            )
            nt_rates_list.extend(batch_rates)
            nt_csps_list.extend(batch_csps)

        return nt_rates_list, nt_csps_list
