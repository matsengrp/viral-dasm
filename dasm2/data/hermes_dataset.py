"""HERMES dataset for sequence-structure alignment.

This dataset loads sequences with their corresponding HERMES structural embeddings
for structural guidance learning. It provides paired (sequence, structure) examples
to guide sequence embeddings toward structurally-aware representations.

Currently optimized for antibody structures with heavy (H) and light (L) chain
handling, including chain reordering and separator token insertion. Future versions
may generalize to arbitrary multi-chain proteins.

"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from dasm2.data.hermes import (
    _is_tarfile_path,
    load_all_from_directory,
    load_all_from_tarfile,
    normalize_structural_embeddings_with_mask,
)
from dasm2.sequence.constants import AA_AMBIG_IDX, AA_PADDING_TOKEN, AA_SEPARATOR
from dasm2.sequence.conversion import three_letter_aa_to_one_letter
from dasm2.tokenizers import AminoAcidTokenizer
from dasm2.utils.tensor import stack_heterogeneous

logger = logging.getLogger(__name__)


class HERMESDataset(Dataset):
    """Dataset providing sequences with HERMES structural embeddings for structural guidance.

    This dataset loads sequences that have corresponding PDB structures in HERMES,
    providing both sequence and structural embeddings for structural guidance learning.

    Unlike EvolutionaryDataset which provides parent-child pairs, HERMESDataset
    provides single sequences with their structural representations.

    Multi-Chain Handling (Antibody-Specific):
        This dataset is currently designed for antibody structures with heavy (H)
        and/or light (L) chains. When chain_filter specifies multiple chains
        (e.g., ["H", "L"]), chains are concatenated with separator tokens following
        the unified tokenization design:
        - Sequences: "EVQLV...^DIQMT..." (H chain, separator, L chain)
        - Embeddings: [H_embs, zeros, L_embs] with zero vector at separator position
        - Masks: False at separator position (excluded from loss computation)
        - Chain Reordering: Chains are reordered to match chain_filter order for
          consistency across PDB files (some have L→H order, others H→L)
        - Partial Chains: Allows H-only ("H_seq^") or L-only ("^L_seq") structures,
          but fails if neither requested chain is present

        Note: Future versions may generalize to arbitrary multi-chain proteins beyond
        antibodies.

    Example:
        >>> from pathlib import Path
        >>> from dasm2.data.structural_embedding_registry import resolve_structural_embedding_path
        >>> # Resolve path using registry (external to dataset)
        >>> hermes_dir = resolve_structural_embedding_path("hermes_sabdab_ab_stability")
        >>> # Auto-filters to valid PDBs with requested chains
        >>> dataset = HERMESDataset(
        ...     hermes_model_dir=hermes_dir,
        ...     chain_filter=["H", "L"],  # Antibody heavy + light chains with separator
        ... )
        >>> batch = dataset[0]
        >>> print(batch.keys())
        dict_keys(['aa_parents_idxs', 'structure_embeddings', 'mask', 'pdb_id'])
        >>> batch["structure_embeddings"].shape[1] == HERMES_EMBEDDING_DIM
        True
        >>> # Separator position is masked
        >>> batch["mask"][separator_idx]  # False at separator
        False
    """

    row_filter_digest: Optional[str] = None
    """Never row-filtered; part of the ``CacheableDataset`` protocol (#1672)."""

    def __init__(
        self,
        hermes_model_dir: Union[Path, str],
        chain_filter: Optional[List[str]] = None,
        pdb_ids: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        device: str = "cpu",
        nickname: Optional[str] = None,
    ):
        """Initialize structural dataset.

        Args:
            hermes_model_dir: Path to HERMES model directory or .tar.gz archive.
                Can be a Path object or string (e.g., "~/data/hermes/model.tar.gz").
            chain_filter: Optional list of chain IDs to include (e.g., ["H", "L"] for
                         antibody heavy and light chains). If None, loads all chains
                         for each PDB. When multiple chains specified, they are
                         reordered to match chain_filter order and separated with
                         separator tokens.
            pdb_ids: Optional list of specific PDB IDs to load. If None, automatically
                     filters to valid PDBs using get_valid_pdb_ids(), excluding empty
                     PDBs and those missing all requested chains (when chain_filter
                     is provided).
            max_length: Optional maximum sequence length (for padding/truncation)
            device: Device for tensors (default: "cpu", moved to GPU by trainer)
            nickname: Human-readable identifier for cache filenames. If None, auto-generated
                     from hermes_model_dir name and chain_filter (e.g., "hermes_py_000_H_L").

        Raises:
            FileNotFoundError: If HERMES model directory/archive not found
            ValueError: If no valid structures can be loaded

        Example:
            >>> from pathlib import Path
            >>> # Load from unpacked directory
            >>> dataset = HERMESDataset(
            ...     hermes_model_dir="data/hermes/hermes_py_000",
            ...     chain_filter=["H", "L"],
            ... )
            >>> # Load from tar.gz archive (sequential read, no unpacking needed)
            >>> dataset = HERMESDataset(
            ...     hermes_model_dir="~/data/hermes/model.tar.gz",
            ...     chain_filter=["H", "L"],
            ... )
        """
        super().__init__()

        # Convert to Path and expand user
        hermes_model_dir = Path(hermes_model_dir).expanduser()

        # Validate path exists
        if not hermes_model_dir.exists():
            raise FileNotFoundError(f"HERMES model path not found: {hermes_model_dir}")

        # Determine if this is a tar.gz archive
        self._is_tarfile = _is_tarfile_path(hermes_model_dir)

        if not self._is_tarfile and not hermes_model_dir.is_dir():
            raise ValueError(
                f"HERMES model path is not a directory or tar.gz: {hermes_model_dir}"
            )

        self.hermes_model_dir = hermes_model_dir
        self.chain_filter = chain_filter
        self.max_length = max_length
        self.device = device

        # Compute nickname from path and chain_filter if not provided
        if nickname is None:
            chain_suffix = "_".join(chain_filter) if chain_filter else "all"
            # For tar.gz, use stem without .tar.gz extension
            if self._is_tarfile:
                base_name = hermes_model_dir.stem
                if base_name.endswith(".tar"):
                    base_name = base_name[:-4]
            else:
                base_name = hermes_model_dir.name
            self.nickname = f"{base_name}_{chain_suffix}"
        else:
            self.nickname = nickname

        # Initialize tokenizer for amino acid encoding
        self.tokenizer = AminoAcidTokenizer()

        # Store user-provided pdb_ids (may be None for auto-discovery)
        # PDB ID discovery is deferred to _load_all_structures to avoid
        # scanning files twice (once for discovery, once for loading)
        self._requested_pdb_ids = pdb_ids

        # Pre-load all data (structural embeddings are small, ~1-5MB per PDB)
        # Data is stored as stacked tensors for efficient batching
        self._load_all_structures()

    def _segment_by_chains_from_metadata(
        self, embeddings: torch.Tensor, metadata: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Extract individual chain segments from metadata.

        Args:
            embeddings: Full embeddings tensor [seq_len, 256]
            metadata: DataFrame with "chain" and "resname" columns indicating
                     chain ID and amino acid name per residue

        Returns:
            Dict mapping chain_id -> {"sequence": str, "embeddings": torch.Tensor}

        Raises:
            ValueError: If embeddings length doesn't match metadata
        """
        # Fail-fast validation: detect data corruption early
        if len(metadata) != len(embeddings):
            raise ValueError(
                f"Data corruption: metadata length ({len(metadata)}) doesn't match "
                f"embeddings length ({len(embeddings)})"
            )

        # Reset index to ensure positions match embeddings tensor
        metadata_reset = metadata.reset_index(drop=True)

        # Segment by chain using pandas groupby
        chain_segments: Dict[str, Dict[str, Any]] = {}
        for chain_id, group in metadata_reset.groupby("chain", sort=False):
            # Get residue names for this chain and convert to 1-letter codes
            chain_resnames = group["resname"].tolist()
            sequence = "".join(
                aa if len(aa) == 1 else three_letter_aa_to_one_letter(aa)
                for aa in chain_resnames
            )

            # Extract embeddings for this chain using positional indices
            indices = group.index.tolist()
            chain_embeddings = embeddings[indices]

            chain_segments[str(chain_id)] = {
                "sequence": sequence,
                "embeddings": chain_embeddings,
            }

        return chain_segments

    def _reorder_and_concatenate_chains(
        self,
        chain_segments: Dict[str, Dict[str, Any]],
        embeddings: torch.Tensor,
        metadata: pd.DataFrame,
    ) -> tuple[str, torch.Tensor]:
        """Reorder chains and concatenate with separator tokens.

        When chain_filter specifies multiple chains (e.g., ["H", "L"] for antibody
        heavy and light chains), this reorders chains to match chain_filter order
        and inserts separator tokens between chains following the unified tokenization
        design (H^^^L pattern in amino acid space becomes H^L).

        Chain reordering ensures consistency across different PDB files - some PDBs
        store chains in L→H order while others use H→L order. By reordering to match
        chain_filter, we guarantee consistent sequence structure regardless of PDB
        storage order.

        Args:
            chain_segments: Dict mapping chain_id -> {"sequence": str, "embeddings": torch.Tensor}
            embeddings: Original full embeddings tensor (for dtype/shape reference)
            metadata: DataFrame with "chain" column (for validation)

        Returns:
            Tuple of (separated_sequence, separated_embeddings) where:
            - separated_sequence: Chains in chain_filter order with separators (e.g., "H_seq^L_seq")
            - separated_embeddings: [total_len, 256] with zero vectors at separator positions

        Example:
            >>> # Input: chain_segments with L and H chains
            >>> chain_segments = {
            ...     "L": {"sequence": "DIQM", "embeddings": torch.randn(3, 256)},
            ...     "H": {"sequence": "EVQLV", "embeddings": torch.randn(5, 256)}
            ... }
            >>> # chain_filter = ["H", "L"] - requested order
            >>> sep_seq, sep_embs = dataset._reorder_and_concatenate_chains(chain_segments, embs, metadata)
            >>> sep_seq
            "EVQLV^DIQM"  # Reordered to H^L (requested order)
            >>> sep_embs.shape
            torch.Size([9, 256])  # 5 + 1 + 3
            >>> (sep_embs[5] == 0).all()  # Separator position is zero vector
            True
        """
        # Single chain or no chain filter: return first available chain as-is
        if self.chain_filter is None or len(self.chain_filter) <= 1:
            # Return first chain found
            first_chain = next(iter(chain_segments.values()))
            return first_chain["sequence"], first_chain["embeddings"]

        # Reorder chains and insert separators
        # IMPORTANT: Separators mark chain junctions even when chains are missing.
        # This ensures consistent sequence structure for model compatibility.
        # Examples below assume chain_filter=["H", "L"] is requested:
        # - H+L both present: "H_seq^L_seq"
        # - H-only present: "H_seq^" (trailing separator marks where L would be)
        # - L-only present: "^L_seq" (leading separator marks where H would be)
        sep_parts: List[str] = []
        sep_embeddings_parts: List[torch.Tensor] = []

        for i, chain_id in enumerate(self.chain_filter):
            # Add chain segment if present
            if chain_id in chain_segments:
                sep_parts.append(chain_segments[chain_id]["sequence"])
                sep_embeddings_parts.append(chain_segments[chain_id]["embeddings"])

            # Add separator after each chain except the last
            # (even if chain is missing, preserve junction position)
            if i < len(self.chain_filter) - 1:
                sep_parts.append(AA_SEPARATOR)
                # Zero vector at separator position
                sep_embeddings_parts.append(
                    torch.zeros((1, embeddings.shape[1]), dtype=embeddings.dtype)
                )

        separated_sequence = "".join(sep_parts)
        separated_embeddings = torch.cat(sep_embeddings_parts, dim=0)

        # Fail-fast if result is separator-only (no actual sequence data)
        # This happens when none of the requested chains are present in the PDB
        if separated_sequence.strip("^") == "":
            present_chains = sorted(metadata["chain"].unique().tolist())
            raise ValueError(
                f"No requested chains found in structure. "
                f"Requested: {self.chain_filter}, "
                f"Present: {present_chains}. "
                f"Result would be separator-only with no sequence data."
            )

        return separated_sequence, separated_embeddings

    def _load_all_structures(self) -> None:
        """Load all HERMES structures and convert to sequence format.

        This pre-loads all structural data to avoid repeated disk I/O during training.
        HERMES embeddings are relatively small (~256 floats per residue), so loading
        all structures is practical for most training scenarios.

        For tar.gz archives, reads the entire archive sequentially in a single pass
        for efficiency (avoids repeated decompression).

        Data is collected into lists first, then stacked with padding using
        stack_heterogeneous to enable efficient batching of variable-length sequences.

        Raises:
            FileNotFoundError: If any PDB file is missing
            ValueError: If any PDB has invalid data (e.g., empty embeddings, missing chains)

        Note:
            Follows fail-fast philosophy - any error in loading a PDB will raise an
            exception rather than silently skipping. This ensures data quality and
            makes problems immediately visible.
        """
        # Load raw HERMES data in a single pass (same approach for directory and tar.gz)
        # Pass pdb_ids=None to auto-discover all valid PDBs
        pdb_ids_arg = self._requested_pdb_ids

        if self._is_tarfile:
            all_hermes_data = load_all_from_tarfile(
                self.hermes_model_dir,
                chain_filter=self.chain_filter,
                pdb_ids=pdb_ids_arg,
            )
        else:
            all_hermes_data = load_all_from_directory(
                self.hermes_model_dir,
                chain_filter=self.chain_filter,
                pdb_ids=pdb_ids_arg,
            )

        # Update pdb_ids with what was actually loaded
        self.pdb_ids = list(all_hermes_data.keys())

        # Process loaded data into sequence format
        aa_parents_idxs_list: List[torch.Tensor] = []
        structure_embeddings_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        pdb_ids_list: List[str] = []

        for pdb_id, hermes_data in all_hermes_data.items():
            # Get structural embeddings (already aligned to sequence)
            structure_embeddings = (
                hermes_data.embeddings
            )  # [seq_len, HERMES_EMBEDDING_DIM]

            # Segment by chains from metadata
            chain_segments = self._segment_by_chains_from_metadata(
                structure_embeddings, hermes_data.metadata
            )

            # Reorder chains and insert separators if multiple chains requested
            # (e.g., H^L for antibody heavy+light chains)
            # Chains are reordered to match chain_filter order for consistency
            sequence, structure_embeddings = self._reorder_and_concatenate_chains(
                chain_segments, hermes_data.embeddings, hermes_data.metadata
            )

            # Validate non-empty sequence
            if len(sequence) == 0:
                raise ValueError(
                    f"PDB {pdb_id} has zero residues after chain filtering "
                    f"(chain_filter={self.chain_filter})"
                )

            # Convert sequence to indices using tokenizer
            aa_idxs = self.tokenizer.tokenize([sequence])[0]  # [seq_len]

            # Create mask: True for valid positions, False for padding and separators
            # aa_idxs < AA_AMBIG_IDX excludes both padding (20) and separator (21)
            mask = aa_idxs < AA_AMBIG_IDX

            # Apply max_length if specified
            if self.max_length is not None and len(aa_idxs) > self.max_length:
                aa_idxs = aa_idxs[: self.max_length]
                structure_embeddings = structure_embeddings[: self.max_length]
                mask = mask[: self.max_length]

            # Normalize structural embeddings (required by structural guidance loss)
            structure_embeddings = normalize_structural_embeddings_with_mask(
                structure_embeddings.unsqueeze(0), mask.unsqueeze(0)
            ).squeeze(0)

            aa_parents_idxs_list.append(aa_idxs)
            structure_embeddings_list.append(structure_embeddings)
            mask_list.append(mask)
            pdb_ids_list.append(pdb_id)

        # Fail fast if no structures loaded
        if len(pdb_ids_list) == 0:
            raise ValueError(
                f"No valid structures loaded from {self.hermes_model_dir}.\n"
                f"Attempted to load {len(self.pdb_ids)} PDB IDs with chain_filter={self.chain_filter}.\n"
                f"Check that PDB IDs exist and have the requested chains."
            )

        # Stack with padding using stack_heterogeneous
        self.aa_parents_idxs = (
            stack_heterogeneous(aa_parents_idxs_list, pad_value=float(AA_PADDING_TOKEN))
            .long()
            .to(self.device)
        )
        self.structure_embeddings = stack_heterogeneous(
            structure_embeddings_list, pad_value=0.0
        ).to(self.device)
        self.mask = stack_heterogeneous(mask_list, pad_value=0.0).bool().to(self.device)
        self._pdb_ids = pdb_ids_list  # Keep as list of strings

        logger.info(
            f"Loaded {len(self._pdb_ids)} HERMES structures from {self.hermes_model_dir.name} "
            f"with chain_filter={self.chain_filter}"
        )

    def __len__(self) -> int:
        """Return number of sequence-structure pairs."""
        return len(self._pdb_ids)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get a single sequence-structure pair.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Dict with keys:
            - aa_parents_idxs: [max_seq_len] amino acid sequence indices (includes separator if
              multi-chain, padded to max length with AA_PADDING_TOKEN)
            - structure_embeddings: [max_seq_len, HERMES_EMBEDDING_DIM] L2-normalized HERMES structural
              embeddings (zero vectors at separator and padding positions)
            - mask: [max_seq_len] valid position indicators (False at separators and padding)
            - pdb_id: str, PDB identifier (metadata)

        Example:
            >>> # Single chain
            >>> batch = dataset[0]
            >>> batch["aa_parents_idxs"].shape
            torch.Size([max_len])  # Padded to max length in dataset
            >>> batch["structure_embeddings"].shape
            torch.Size([max_len, 256])  # 256 == HERMES_EMBEDDING_DIM
            >>> batch["mask"].sum() <= max_len  # Valid positions <= max length
            True
            >>> batch["pdb_id"]
            '1adq'
            >>> # Multi-chain (H+L)
            >>> dataset_hl = HERMESDataset(model_dir, chain_filter=["H", "L"])
            >>> batch = dataset_hl[0]
            >>> # Length = max_len, mask excludes separators and padding
            >>> batch["mask"].sum() < len(batch["aa_parents_idxs"])  # Separator/padding masked
            True
        """
        return {
            "aa_parents_idxs": self.aa_parents_idxs[idx],
            "structure_embeddings": self.structure_embeddings[idx],
            "mask": self.mask[idx],
            "pdb_id": self._pdb_ids[idx],
        }

    def to(self, device: str) -> "HERMESDataset":
        """Move all tensors to specified device.

        Args:
            device: Target device ("cpu", "cuda", etc.)

        Returns:
            Self (for chaining)
        """
        self.device = device
        self.aa_parents_idxs = self.aa_parents_idxs.to(device)
        self.structure_embeddings = self.structure_embeddings.to(device)
        self.mask = self.mask.to(device)
        return self
