"""HERMES structural embeddings data loading.

This module provides utilities for loading HERMES structural embeddings
from CSV + NPY file pairs. Each PDB ID has:
- CSV: Residue metadata + 20 amino acid logits (ΔG predictions)
- NPY: [num_residues, HERMES_EMBEDDING_DIM] embeddings parallel with CSV rows

Supports loading from both unpacked directories and .tar.gz archives.
When loading from tar.gz, files are read sequentially for efficiency.
"""

import io
import tarfile
import warnings
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch


def _is_tarfile_path(path: Path) -> bool:
    """Check if a path points to a tar.gz archive."""
    return path.suffix == ".gz" and path.stem.endswith(".tar")


def _process_hermes_pair(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    pdb_id: str,
    chain_filter: Optional[List[str]],
) -> Optional["HermesData"]:
    """Process a CSV/NPY pair and return HermesData if valid.

    This is the unified processing logic shared between directory and tar.gz loading.
    Handles validation, dimension checks, and chain filtering.

    Args:
        df: DataFrame from CSV with residue metadata
        embeddings: NumPy array from NPY with structural embeddings
        pdb_id: PDB identifier for warnings
        chain_filter: Optional list of chain IDs to filter to

    Returns:
        HermesData if valid and passes filters, None if should be skipped
    """
    # Skip empty DataFrames
    if len(df) == 0:
        return None

    # Check chain filter early (before validation)
    if chain_filter is not None:
        available_chains = set(df["chain"].unique())
        requested_chains = set(chain_filter)
        if not (available_chains & requested_chains):
            return None

    # Validate 2D array
    if embeddings.ndim != 2:
        warnings.warn(f"Skipping {pdb_id}: embeddings not 2D")
        return None

    # Validate alignment
    if len(df) != embeddings.shape[0]:
        warnings.warn(
            f"Skipping {pdb_id}: CSV has {len(df)} rows, "
            f"NPY has {embeddings.shape[0]} rows"
        )
        return None

    # Validate embedding dimension
    if embeddings.shape[1] != HERMES_EMBEDDING_DIM:
        warnings.warn(
            f"Skipping {pdb_id}: expected {HERMES_EMBEDDING_DIM}D embeddings, "
            f"got {embeddings.shape[1]}D"
        )
        return None

    embeddings_tensor = torch.from_numpy(embeddings).float()

    # Apply chain filtering to the data
    if chain_filter is not None:
        chain_mask = df["chain"].isin(chain_filter)
        if not chain_mask.any():
            return None
        df = df[chain_mask].copy()
        chain_mask_tensor = torch.from_numpy(chain_mask.values)
        embeddings_tensor = embeddings_tensor[chain_mask_tensor]

    return HermesData(metadata=df, embeddings=embeddings_tensor)


def _update_result_dict(
    result: Dict[str, "HermesData"],
    pdb_id: str,
    hermes_data: "HermesData",
) -> None:
    """Update result dictionary, keeping entry with more chains on duplicates.

    Args:
        result: Dictionary to update in place
        pdb_id: PDB identifier
        hermes_data: HermesData to potentially add
    """
    if pdb_id in result:
        existing_chains = set(result[pdb_id].metadata["chain"].unique())
        new_chains = set(hermes_data.metadata["chain"].unique())
        if len(new_chains) <= len(existing_chains):
            return
    result[pdb_id] = hermes_data


def _parse_pdb_id_from_basename(basename: str, model_name: str) -> Optional[str]:
    """Parse PDB ID from a HERMES file basename.

    Args:
        basename: Filename like "model__pdb__chains.csv" or "model__pdb-embeddings.npy"
        model_name: Expected model name prefix

    Returns:
        PDB ID if valid format, None otherwise
    """
    if not basename.startswith(model_name + "__"):
        return None

    # Remove extension and -embeddings suffix
    stem = basename.split(".")[0].replace("-embeddings", "")
    parts = stem.split("__")
    if len(parts) < 2:
        return None

    return parts[1]


def _get_model_name_from_tarfile(tar_path: Path) -> str:
    """Extract model name from tar.gz filename.

    E.g., 'chothia_antibodies_pdbfiles_hermes_py_050_ft_cdna117k_ddg_st.tar.gz'
    -> 'hermes_py_050_ft_cdna117k_ddg_st'
    """
    # Remove .tar.gz suffix
    base = tar_path.stem  # removes .gz
    if base.endswith(".tar"):
        base = base[:-4]  # removes .tar

    # The model name is typically the last part after 'pdbfiles_'
    if "_pdbfiles_" in base:
        return base.split("_pdbfiles_")[-1]
    # Fallback: use the whole base name
    return base


class HermesArchiveIndex(NamedTuple):
    """Index of files in a HERMES tar.gz archive.

    Attributes:
        model_name: The HERMES model name (e.g., 'hermes_py_050_ft_cdna117k_ddg_st')
        pdb_files: Dict mapping pdb_id -> list of (csv_member_name, npy_member_name) tuples
        chains_by_pdb: Dict mapping pdb_id -> set of chain IDs found in any file for that PDB
    """

    model_name: str
    pdb_files: Dict[str, List[Tuple[str, str]]]
    chains_by_pdb: Dict[str, set]


def _build_tarfile_index(
    tar_path: Path, chain_filter: Optional[List[str]] = None
) -> HermesArchiveIndex:
    """Build an index of HERMES files in a tar.gz archive.

    Scans the archive once to identify all CSV/NPY pairs and their chain content.
    This is done sequentially for efficiency with gzip compression.

    Args:
        tar_path: Path to the tar.gz archive
        chain_filter: Optional chain filter - if provided, only index files
            containing at least one of these chains

    Returns:
        HermesArchiveIndex with file mappings and chain information
    """
    model_name = _get_model_name_from_tarfile(tar_path)
    pdb_files: Dict[str, List[Tuple[str, str]]] = {}
    chains_by_pdb: Dict[str, set] = {}

    # First pass: find all CSV files and check their chains
    csv_members: Dict[str, str] = {}  # member_name -> pdb_id
    requested_chains = set(chain_filter) if chain_filter else None

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue

            name = member.name
            # Handle both 'dir/file.csv' and 'file.csv' formats
            basename = Path(name).name

            if not basename.startswith(model_name + "__"):
                continue

            if basename.endswith(".csv"):
                # Parse: model__pdb__chains.csv or model__pdb.csv
                stem = basename[:-4]  # remove .csv
                parts = stem.split("__")
                if len(parts) < 2:
                    continue
                pdb_id = parts[1]

                # Read CSV to check chains
                f = tar.extractfile(member)
                if f is None:
                    continue
                df = pd.read_csv(f, usecols=["chain"])

                if len(df) == 0:
                    continue

                available_chains = set(df["chain"].unique())

                # Apply chain filter
                if requested_chains is not None:
                    if not (available_chains & requested_chains):
                        continue

                csv_members[name] = pdb_id

                # Track chains for this PDB
                if pdb_id not in chains_by_pdb:
                    chains_by_pdb[pdb_id] = set()
                chains_by_pdb[pdb_id].update(available_chains)

            elif basename.endswith("-embeddings.npy"):
                # Will match with CSV in second pass
                pass

    # Second pass: match CSV files with their NPY counterparts
    with tarfile.open(tar_path, "r:gz") as tar:
        npy_members = {}
        for member in tar:
            if member.isfile() and member.name.endswith("-embeddings.npy"):
                npy_members[member.name] = member.name

    # Build the pdb_files mapping
    for csv_name, pdb_id in csv_members.items():
        # Construct expected NPY name
        npy_name = csv_name[:-4] + "-embeddings.npy"

        if npy_name in npy_members:
            if pdb_id not in pdb_files:
                pdb_files[pdb_id] = []
            pdb_files[pdb_id].append((csv_name, npy_name))

    return HermesArchiveIndex(
        model_name=model_name, pdb_files=pdb_files, chains_by_pdb=chains_by_pdb
    )


def load_all_from_tarfile(
    tar_path: Path,
    chain_filter: Optional[List[str]] = None,
    pdb_ids: Optional[List[str]] = None,
) -> Dict[str, "HermesData"]:
    """Load all HERMES data from a tar.gz archive in a single sequential pass.

    This is the efficient way to load from tar.gz - reads the entire archive
    once sequentially, avoiding repeated decompression.

    Args:
        tar_path: Path to the tar.gz archive
        chain_filter: Optional list of chain IDs to filter to
        pdb_ids: Optional list of specific PDB IDs to load. If None, loads all.

    Returns:
        Dict mapping pdb_id -> HermesData for each loaded PDB

    Raises:
        FileNotFoundError: If archive doesn't exist
        ValueError: If no valid data found
    """
    if not tar_path.exists():
        raise FileNotFoundError(f"HERMES archive not found: {tar_path}")

    model_name = _get_model_name_from_tarfile(tar_path)
    requested_pdb_ids = set(pdb_ids) if pdb_ids else None

    # Collect CSV and NPY data in memory
    csv_data: Dict[str, pd.DataFrame] = {}  # member_name -> DataFrame
    npy_data: Dict[str, np.ndarray] = {}  # member_name -> array
    csv_to_pdb: Dict[str, str] = {}  # csv_member_name -> pdb_id

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue

            name = member.name
            basename = Path(name).name

            pdb_id = _parse_pdb_id_from_basename(basename, model_name)
            if pdb_id is None:
                continue

            # Skip if not in requested PDB list
            if requested_pdb_ids is not None and pdb_id not in requested_pdb_ids:
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            if basename.endswith(".csv"):
                try:
                    df = pd.read_csv(f)
                    if len(df) == 0:
                        # Skip empty DataFrames (header only, no data rows)
                        continue
                    csv_data[name] = df
                    csv_to_pdb[name] = pdb_id
                except pd.errors.EmptyDataError:
                    # Skip malformed/empty CSV files
                    warnings.warn(f"Skipping malformed CSV in archive: {name}")
                    continue
                except pd.errors.ParserError as e:
                    # Skip CSVs that fail to parse
                    warnings.warn(f"Skipping unparseable CSV in archive: {name} ({e})")
                    continue
            elif basename.endswith("-embeddings.npy"):
                try:
                    content = f.read()
                    npy_data[name] = np.load(io.BytesIO(content))
                except (EOFError, ValueError) as e:
                    # Skip malformed/empty NPY files
                    warnings.warn(f"Skipping malformed NPY in archive: {name} ({e})")
                    continue

    # Match CSV with NPY and build HermesData objects
    result: Dict[str, HermesData] = {}

    for csv_name, df in csv_data.items():
        npy_name = csv_name[:-4] + "-embeddings.npy"
        if npy_name not in npy_data:
            continue

        pdb_id = csv_to_pdb[csv_name]
        hermes_data = _process_hermes_pair(df, npy_data[npy_name], pdb_id, chain_filter)
        if hermes_data is not None:
            _update_result_dict(result, pdb_id, hermes_data)

    if not result:
        raise ValueError(
            f"No valid HERMES data found in {tar_path} "
            f"with chain_filter={chain_filter}, pdb_ids={pdb_ids}"
        )

    return result


def get_valid_pdb_ids_from_tarfile(
    tar_path: Path,
    chain_filter: Optional[List[str]] = None,
) -> List[str]:
    """Get valid PDB IDs from a tar.gz archive.

    Args:
        tar_path: Path to the tar.gz archive
        chain_filter: Optional list of chain IDs - only returns PDBs with
            at least one of these chains

    Returns:
        Sorted list of valid PDB IDs
    """
    index = _build_tarfile_index(tar_path, chain_filter=chain_filter)
    return sorted(index.pdb_files.keys())


# HERMES model configuration
HERMES_EMBEDDING_DIM = 256  # Structural embedding dimension from HERMES model


def load_all_from_directory(
    hermes_dir: Path,
    chain_filter: Optional[List[str]] = None,
    pdb_ids: Optional[List[str]] = None,
) -> Dict[str, "HermesData"]:
    """Load all HERMES data from a directory.

    Scans all CSV/NPY file pairs, loading and filtering by chains in memory.

    Args:
        hermes_dir: Path to HERMES model directory
        chain_filter: Optional list of chain IDs to filter to. Returns PDBs
            with at least one of the specified chains.
        pdb_ids: Optional list of specific PDB IDs to load. If None, loads all.

    Returns:
        Dict mapping pdb_id -> HermesData for each loaded PDB

    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no valid data found
    """
    if not hermes_dir.exists():
        raise FileNotFoundError(f"HERMES directory not found: {hermes_dir}")

    model_name = hermes_dir.name
    requested_pdb_ids = set(pdb_ids) if pdb_ids else None

    result: Dict[str, HermesData] = {}

    # Find all CSV files
    csv_pattern = f"{model_name}__*.csv"

    for csv_path in hermes_dir.glob(csv_pattern):
        # Parse PDB ID from filename
        parts = csv_path.stem.split("__")
        if len(parts) < 2:
            continue
        pdb_id = parts[1]

        # Skip if not in requested PDB list
        if requested_pdb_ids is not None and pdb_id not in requested_pdb_ids:
            continue

        # Check NPY file exists before loading CSV
        npy_path = csv_path.with_name(csv_path.stem + "-embeddings.npy")
        if not npy_path.exists():
            continue

        # Load CSV and NPY
        df = pd.read_csv(csv_path)
        embeddings = np.load(npy_path)

        hermes_data = _process_hermes_pair(df, embeddings, pdb_id, chain_filter)
        if hermes_data is not None:
            _update_result_dict(result, pdb_id, hermes_data)

    if not result:
        raise ValueError(
            f"No valid HERMES data found in {hermes_dir} "
            f"with chain_filter={chain_filter}, pdb_ids={pdb_ids}"
        )

    return result


class HermesData(NamedTuple):
    """HERMES structural embeddings with metadata.

    Attributes:
        metadata: Residue metadata (PDB, chain, resname, resnum, insertion_code, logits)
        embeddings: Structural embeddings of shape [num_residues, HERMES_EMBEDDING_DIM]
    """

    metadata: pd.DataFrame
    embeddings: torch.Tensor


def _find_hermes_files_for_pdb(
    hermes_model_dir: Path, pdb_id: str
) -> List[Tuple[Path, Path]]:
    """Find all HERMES CSV/NPY file pairs for a PDB ID.

    Searches for files matching `{model}__{pdb_id}*.csv` pattern, which handles
    both legacy format (`{model}__{pdb}.csv`) and chain-suffix format
    (`{model}__{pdb}__{chains}.csv`).

    Args:
        hermes_model_dir: Path to HERMES model directory
        pdb_id: PDB identifier

    Returns:
        List of (csv_path, npy_path) tuples for all matching file pairs

    Raises:
        FileNotFoundError: If no files found for this PDB ID
    """
    model_name = hermes_model_dir.name

    # Find all CSV files for this PDB (any chain suffix or none)
    # Pattern matches: model__pdb.csv, model__pdb__H_L.csv, model__pdb__P_M.csv, etc.
    csv_pattern = f"{model_name}__{pdb_id}*.csv"
    csv_files = list(hermes_model_dir.glob(csv_pattern))

    # Filter to exclude files where pdb_id is a prefix of another PDB
    # e.g., searching for "1a4j" shouldn't match "1a4j2"
    valid_pairs = []
    for csv_path in csv_files:
        # Extract the part after model__pdb to verify exact match
        stem = csv_path.stem  # e.g., "model__1adq" or "model__1adq__H_L"
        parts = stem.split("__")
        if len(parts) >= 2 and parts[1] == pdb_id:
            # Find corresponding NPY file
            npy_path = csv_path.with_name(csv_path.stem + "-embeddings.npy")
            if npy_path.exists():
                valid_pairs.append((csv_path, npy_path))

    if not valid_pairs:
        raise FileNotFoundError(
            f"No HERMES files found for PDB '{pdb_id}' in '{model_name}'"
        )

    return valid_pairs


def _load_raw_hermes_data(
    csv_path: Path, npy_path: Path, pdb_id: str
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """Load and validate raw HERMES data from a CSV + NPY file pair.

    Handles file loading, alignment validation, and dimension checks.

    Args:
        csv_path: Path to CSV file with residue metadata
        npy_path: Path to NPY file with embeddings
        pdb_id: PDB identifier (for error messages)

    Returns:
        Tuple of (metadata DataFrame, embeddings tensor)

    Raises:
        FileNotFoundError: If CSV or NPY file not found
        ValueError: If CSV and NPY have mismatched row counts or wrong dimension
    """
    # Check existence
    if not csv_path.exists():
        raise FileNotFoundError(f"HERMES CSV not found: {csv_path}")
    if not npy_path.exists():
        raise FileNotFoundError(f"HERMES embeddings not found: {npy_path}")

    # Load data
    metadata_df = pd.read_csv(csv_path)
    embeddings_np = np.load(npy_path)

    # Validate array is 2D before accessing shape[1]
    if embeddings_np.ndim != 2:
        raise ValueError(
            f"Invalid embedding array for PDB {pdb_id}:\n"
            f"  Expected 2D array, got {embeddings_np.ndim}D array with shape {embeddings_np.shape}\n"
            f"HERMES embeddings should be [num_residues, {HERMES_EMBEDDING_DIM}]."
        )

    # Validate alignment
    if len(metadata_df) != embeddings_np.shape[0]:
        raise ValueError(
            f"Residue count mismatch for PDB {pdb_id}:\n"
            f"  CSV has {len(metadata_df)} residues\n"
            f"  NPY has {embeddings_np.shape[0]} residues\n"
            f"CSV and NPY files must have aligned residue data."
        )

    # Validate embedding dimension
    if embeddings_np.shape[1] != HERMES_EMBEDDING_DIM:
        raise ValueError(
            f"Unexpected embedding dimension for PDB {pdb_id}:\n"
            f"  Expected {HERMES_EMBEDDING_DIM}, got {embeddings_np.shape[1]}\n"
            f"HERMES embeddings should be {HERMES_EMBEDDING_DIM}-dimensional."
        )

    # Convert to torch tensor
    embeddings_tensor = torch.from_numpy(embeddings_np).float()

    return metadata_df, embeddings_tensor


def _filter_by_chains(
    metadata: pd.DataFrame,
    embeddings: torch.Tensor,
    chain_filter: List[str],
    pdb_id: str,
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """Filter metadata and embeddings to specified chains.

    Args:
        metadata: Full metadata DataFrame
        embeddings: Full embeddings tensor
        chain_filter: List of chain IDs to include
        pdb_id: PDB ID for error messages

    Returns:
        Tuple of (filtered metadata DataFrame, filtered embeddings tensor)

    Raises:
        ValueError: If no residues found for specified chains
    """
    # Filter DataFrame
    chain_mask = metadata["chain"].isin(chain_filter)

    # Check if any chains found
    if not chain_mask.any():
        available_chains = sorted(metadata["chain"].unique())
        raise ValueError(
            f"No residues found for chain(s) {chain_filter} in PDB {pdb_id}.\n"
            f"Available chains: {available_chains}"
        )

    # Apply filter
    filtered_metadata = metadata[chain_mask].copy()
    # Convert pandas Series to torch tensor for indexing
    chain_mask_tensor = torch.from_numpy(chain_mask.values)
    filtered_embeddings = embeddings[chain_mask_tensor]

    return filtered_metadata, filtered_embeddings


def load_hermes_embeddings_for_pdb(
    hermes_model_dir: Path,
    pdb_id: str,
    chain_filter: Optional[List[str]] = None,
) -> HermesData:
    """Load HERMES embeddings and metadata for a PDB ID.

    Finds available files for the PDB and loads from the first file that contains
    the requested chains. Chain filtering is always done in memory based on the
    actual 'chain' column in the CSV, regardless of filename conventions.

    Args:
        hermes_model_dir: Path to HERMES model directory (e.g., hermes_py_000/)
        pdb_id: PDB identifier (e.g., "1adq")
        chain_filter: Optional list of chain IDs to include. If None, returns all chains.

    Returns:
        HermesData containing:
        - metadata: DataFrame with columns: pdb, chain, resname, resnum, insertion_code, logit_G, ..., logit_W
          (filtered to specified chains if chain_filter provided)
        - embeddings: Tensor of shape [num_filtered_residues, HERMES_EMBEDDING_DIM] with structural embeddings
          where num_filtered_residues equals len(metadata) after filtering

    Raises:
        FileNotFoundError: If no HERMES files found for this PDB ID
        ValueError: If CSV and NPY have mismatched row counts or wrong dimension
        ValueError: If specified chains not found in any available file

    Example:
        >>> # Load all chains
        >>> data = load_hermes_embeddings_for_pdb(model_dir, "1adq")
        >>> # Load only heavy chain
        >>> data_h = load_hermes_embeddings_for_pdb(model_dir, "1adq", chain_filter=["H"])
        >>> # Load heavy + light (exclude antigen)
        >>> data_hl = load_hermes_embeddings_for_pdb(model_dir, "1adq", chain_filter=["H", "L"])
    """
    # Find all available file pairs for this PDB
    file_pairs = _find_hermes_files_for_pdb(hermes_model_dir, pdb_id)

    # Try each file pair until we find one with the requested chains
    last_error: Optional[Exception] = None
    for csv_path, npy_path in file_pairs:
        metadata, embeddings = _load_raw_hermes_data(csv_path, npy_path, pdb_id)

        # If no chain filter, return first file's data
        if chain_filter is None:
            return HermesData(metadata=metadata, embeddings=embeddings)

        # Check if this file has any of the requested chains
        available_chains = set(metadata["chain"].unique())
        requested_chains = set(chain_filter)
        if available_chains & requested_chains:
            # This file has at least one requested chain - filter and return
            try:
                metadata, embeddings = _filter_by_chains(
                    metadata, embeddings, chain_filter, pdb_id
                )
                return HermesData(metadata=metadata, embeddings=embeddings)
            except ValueError as e:
                last_error = e
                continue

    # No file had the requested chains
    if last_error:
        raise last_error
    raise ValueError(f"No HERMES file for PDB {pdb_id} contains chains {chain_filter}")


def list_chains_for_pdb(
    hermes_model_dir: Path,
    pdb_id: str,
) -> List[str]:
    """List available chains for a PDB ID.

    Scans all HERMES files for this PDB ID and returns the union of all chains
    found across different file variants.

    Args:
        hermes_model_dir: Path to HERMES model directory
        pdb_id: PDB identifier

    Returns:
        Sorted list of chain identifiers (e.g., ["A", "H", "L"])

    Raises:
        FileNotFoundError: If PDB ID not found

    Example:
        >>> chains = list_chains_for_pdb(model_dir, "1adq")
        >>> print(chains)  # ["A", "H", "L"]
    """
    # Find all file pairs for this PDB
    file_pairs = _find_hermes_files_for_pdb(hermes_model_dir, pdb_id)

    # Collect chains from all files
    all_chains: Set[str] = set()
    for csv_file, _ in file_pairs:
        df = pd.read_csv(csv_file, usecols=["chain"])
        all_chains.update(df["chain"].unique().tolist())

    return sorted(all_chains)


def list_available_pdb_ids(hermes_model_dir: Path) -> List[str]:
    """List all PDB IDs available in a HERMES model directory.

    Args:
        hermes_model_dir: Path to HERMES model directory

    Returns:
        Sorted list of unique PDB IDs (extracted from CSV filenames)

    Note:
        Scans filesystem on every call. For repeated lookups, consider caching the result.
        HERMES files may have multiple chain variants per PDB (e.g., __H_L and __P_M),
        so PDB IDs are deduplicated.
    """
    model_name = hermes_model_dir.name
    csv_pattern = f"{model_name}__*.csv"

    pdb_ids: Set[str] = set()
    for csv_file in hermes_model_dir.glob(csv_pattern):
        # Extract PDB ID from filename: hermes_py_000__1adq__H_L.csv -> 1adq
        pdb_id = csv_file.stem.split("__")[1]
        pdb_ids.add(pdb_id)

    return sorted(pdb_ids)


def get_valid_pdb_ids(
    hermes_model_dir: Union[Path, str],
    chain_filter: Optional[List[str]] = None,
) -> List[str]:
    """Get PDB IDs that have valid, non-empty data for the given chain filter.

    Scans all HERMES files and checks the actual 'chain' column content to
    determine which PDBs have the requested chains. This works with both
    unpacked directories and .tar.gz archives.

    Filters out:
    - Empty PDBs (0 residues)
    - PDBs with no matching chains (when chain_filter provided)

    Args:
        hermes_model_dir: Path to HERMES model directory or .tar.gz archive
        chain_filter: Optional list of chain IDs. If provided, only returns
            PDB IDs that have at least one of the specified chains.

    Returns:
        Sorted list of valid PDB IDs

    Note:
        Scans filesystem/archive and reads CSV chain columns on every call.
        For repeated lookups, consider caching the result.

    Example:
        >>> from pathlib import Path
        >>> hermes_dir = Path("data/hermes/hermes_py_000")
        >>> # Get all non-empty PDBs
        >>> all_valid = get_valid_pdb_ids(hermes_dir)
        >>> # Get PDBs with H and/or L chains (for antibody training)
        >>> valid_hl = get_valid_pdb_ids(hermes_dir, chain_filter=["H", "L"])
        >>> # Also works with tar.gz archives
        >>> valid_hl = get_valid_pdb_ids("data/hermes/model.tar.gz", chain_filter=["H", "L"])
    """
    hermes_model_dir = Path(hermes_model_dir).expanduser()

    # Handle tar.gz archives
    if _is_tarfile_path(hermes_model_dir):
        return get_valid_pdb_ids_from_tarfile(hermes_model_dir, chain_filter)

    model_name = hermes_model_dir.name

    # Find all CSV files in the directory
    csv_pattern = f"{model_name}__*.csv"

    # Track valid PDBs (use set to deduplicate across multiple files per PDB)
    valid_pdb_ids: Set[str] = set()
    requested_chains = set(chain_filter) if chain_filter else None

    for csv_file in hermes_model_dir.glob(csv_pattern):
        # Extract PDB ID from filename: hermes_py_000__1adq__H_L.csv -> 1adq
        parts = csv_file.stem.split("__")
        if len(parts) < 2:
            continue
        pdb_id = parts[1]

        # Skip if we've already validated this PDB
        if pdb_id in valid_pdb_ids:
            continue

        # Read only chain column for efficiency (avoids loading logit columns)
        df = pd.read_csv(csv_file, usecols=["chain"])

        # Skip empty PDBs
        if len(df) == 0:
            continue

        # Check chain filter if specified
        if requested_chains is not None:
            available_chains = set(df["chain"].unique())
            if not (available_chains & requested_chains):
                # No overlap with requested chains
                continue

        valid_pdb_ids.add(pdb_id)

    return sorted(valid_pdb_ids)


def normalize_structural_embeddings_with_mask(
    embeddings: torch.Tensor, mask: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """Normalize HERMES structural embeddings, preserving zeros at separator positions.

    HERMES structural embeddings contain zero vectors at separator positions (e.g.,
    between heavy and light chains in antibodies). This function normalizes only
    valid positions while keeping separator positions as zero vectors.

    Uses vectorized operations for efficiency.

    Args:
        embeddings: Structural embeddings of shape [batch, seq_len, embed_dim]
        mask: Boolean mask of shape [batch, seq_len] indicating valid positions
                (True=valid, False=separator/padding with zero vectors)
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        Normalized embeddings with zero vectors preserved at separator positions

    Example:
        >>> struct_emb = torch.randn(2, 5, 256)
        >>> struct_emb[0, 2, :] = 0.0  # Separator position (zero vector)
        >>> mask = torch.tensor([[True, True, False, True, True],
        ...                      [True, True, True, True, True]])
        >>> normalized = normalize_structural_embeddings_with_mask(struct_emb, mask)
        >>> (normalized[0, 2] == 0).all()  # Zero vector preserved
        True
    """
    # Clone and detach to avoid modifying input
    normalized = embeddings.clone().detach()

    # Expand mask to match embedding dimensions [batch, seq_len, 1]
    mask_expanded = mask.unsqueeze(-1)

    # Compute norms: [batch, seq_len, 1]
    norms = torch.norm(normalized, p=2, dim=-1, keepdim=True)

    # Only normalize valid positions indicated by mask
    # At separator positions (mask=False), keep zeros
    normalized = torch.where(
        mask_expanded,
        normalized / (norms + epsilon),  # Normalize valid positions
        torch.zeros_like(normalized),  # Keep zeros at separator positions
    )

    return normalized
