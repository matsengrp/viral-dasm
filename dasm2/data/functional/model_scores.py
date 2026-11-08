"""Model score extraction utilities for functional analysis.

Converts model outputs (HERMES logits, ESM2 masked LM logits, etc.) to
DMS-style matrices for comparison with experimental data.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

from dasm2.data.functional.dms_matrix import mask_wt_to_nan
from dasm2.data.hermes import HermesData
from dasm2.sequence.constants import AA_STR_SORTED


def hermes_logits_to_dms_matrix(
    hermes_data: HermesData,
    reference_sequence: str,
    chain_order: Optional[List[str]] = None,
    set_wt_nan: bool = True,
    aa_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract HERMES AA logits as DMS-style matrix.

    HERMES outputs ΔG values where lower (more negative) = more favorable.
    This function negates the logits so higher = better, matching the DMS
    convention used by experimental data and other models.

    Args:
        hermes_data: HermesData from ``load_hermes_embeddings_for_pdb()``.
        reference_sequence: Expected AA sequence for validation. Length must
            match the number of residues in hermes_data.
        chain_order: Optional list of chain IDs specifying the order to
            concatenate chains. If None, uses the order in the metadata.
        set_wt_nan: Whether to set wild-type positions to NaN (default True).
        aa_order: Order of amino acids for rows. Defaults to AA_STR_SORTED.

    Returns:
        DataFrame of shape (20, L) with negated logits (higher = better).
        Index is amino acids, columns are position integers 0..L-1.

    Raises:
        ValueError: If logit columns are missing, sequence length doesn't
            match, or reference sequence doesn't match HERMES residues.
    """
    if aa_order is None:
        aa_order = list(AA_STR_SORTED)

    metadata = hermes_data.metadata
    logit_cols = [f"logit_{aa}" for aa in AA_STR_SORTED]

    # Validate logit columns exist
    missing_cols = [col for col in logit_cols if col not in metadata.columns]
    if missing_cols:
        raise ValueError(
            f"Missing HERMES logit columns: {missing_cols}. "
            f"Available columns: {list(metadata.columns)}"
        )

    # Order by chain if specified
    if chain_order is not None:
        ordered_dfs = []
        for chain in chain_order:
            chain_df = metadata[metadata["chain"] == chain]
            if len(chain_df) == 0:
                raise ValueError(
                    f"Chain '{chain}' not found in HERMES data. "
                    f"Available chains: {sorted(metadata['chain'].unique())}"
                )
            ordered_dfs.append(chain_df)
        metadata = pd.concat(ordered_dfs, ignore_index=True)

    num_residues = len(metadata)

    # Validate sequence length
    if num_residues != len(reference_sequence):
        raise ValueError(
            f"HERMES data has {num_residues} residues, "
            f"but reference_sequence has {len(reference_sequence)} positions"
        )

    # Validate residue identity matches reference.
    # HERMES CSVs may use 3-letter codes ('ALA') or 1-letter codes ('A')
    # depending on the PDB file format. Handle both.
    hermes_resnames = metadata["resname"].tolist()
    _THREE_TO_ONE = _build_three_to_one()
    _VALID_ONE_LETTER = set(AA_STR_SORTED)
    for i, (resname, expected_aa) in enumerate(
        zip(hermes_resnames, reference_sequence)
    ):
        if resname in _THREE_TO_ONE:
            hermes_aa = _THREE_TO_ONE[resname]
        elif resname in _VALID_ONE_LETTER:
            hermes_aa = resname
        else:
            warnings.warn(
                f"Unknown residue code '{resname}' at position {i} "
                f"(not a standard 3-letter or 1-letter amino acid code), "
                f"skipping validation",
                stacklevel=2,
            )
            continue
        if hermes_aa != expected_aa:
            raise ValueError(
                f"Residue mismatch at position {i}: "
                f"HERMES has {resname} ({hermes_aa}), "
                f"expected {expected_aa}"
            )

    # Extract logit matrix: (num_residues, 20)
    logit_matrix = metadata[logit_cols].values

    # Negate: HERMES uses ΔG (lower = better), DMS convention is higher = better
    negated = -logit_matrix

    # Create DataFrame with AA rows × position columns (transpose)
    matrix = pd.DataFrame(
        negated.T,
        index=aa_order,
        columns=list(range(num_residues)),
    )

    # Set WT positions to NaN
    if set_wt_nan:
        matrix = mask_wt_to_nan(matrix, reference_sequence)

    return matrix


def _build_three_to_one() -> Dict[str, str]:
    """Build mapping from 3-letter amino acid codes to 1-letter codes.

    Returns:
        Dict mapping e.g. 'ALA' -> 'A', 'CYS' -> 'C', etc.
    """
    from Bio.Data.IUPACData import protein_letters_3to1

    return {k.upper(): v.upper() for k, v in protein_letters_3to1.items()}


def _build_esm2_aa_index_map(
    tokenizer: Any,
) -> Dict[str, int]:
    """Build mapping from 1-letter amino acid codes to ESM2 vocabulary indices.

    Args:
        tokenizer: HuggingFace ESM2 tokenizer (from AutoTokenizer).

    Returns:
        Dict mapping e.g. 'A' -> 5, 'C' -> 23, etc. for the 20 standard
        amino acids present in the ESM2 vocabulary.

    Raises:
        ValueError: If any standard amino acid is missing from the vocabulary.
    """
    vocab = tokenizer.get_vocab()
    aa_to_esm_idx: Dict[str, int] = {}
    missing = []
    for aa in AA_STR_SORTED:
        if aa in vocab:
            aa_to_esm_idx[aa] = vocab[aa]
        else:
            missing.append(aa)
    if missing:
        raise ValueError(f"ESM2 vocabulary missing standard amino acids: {missing}")
    return aa_to_esm_idx


def esm2_aa_vocab_indices(tokenizer: Any) -> List[int]:
    """ESM2 vocabulary indices for the 20 standard amino acids, in
    :data:`~dasm2.sequence.constants.AA_STR_SORTED` order.

    Thin wrapper around :func:`_build_esm2_aa_index_map` that returns the
    ordered index list directly, since every caller immediately does
    ``[aa_to_esm_idx[aa] for aa in AA_STR_SORTED]`` -- shared by this
    module's wild-type-marginal scorer (:func:`esm2_logits_to_dms_matrix`)
    and :class:`dasm2.analysis.esm2_zeroshot.Esm2ZeroShotSelectionModel`'s
    masked-marginal scorer.

    Args:
        tokenizer: HuggingFace ESM2 tokenizer (from AutoTokenizer).

    Returns:
        List of 20 ESM2 vocabulary indices, ordered to match
        ``AA_STR_SORTED``.

    Raises:
        ValueError: If any standard amino acid is missing from the
            vocabulary (via :func:`_build_esm2_aa_index_map`).
    """
    aa_to_esm_idx = _build_esm2_aa_index_map(tokenizer)
    return [aa_to_esm_idx[aa] for aa in AA_STR_SORTED]


def load_esm2_for_masked_lm(
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    device: str = "cpu",
) -> Tuple[Any, Any]:
    """Load ESM2 masked language model and tokenizer.

    Loads ``EsmForMaskedLM`` (which includes the LM head for per-position
    amino acid logits) rather than ``EsmModel`` (hidden states only).

    Args:
        model_name: HuggingFace ESM2 model identifier.
        device: Device for the model ("cpu", "cuda", etc.).

    Returns:
        Tuple of (model, tokenizer) ready for inference.
    """
    from transformers import AutoTokenizer, EsmForMaskedLM

    model = EsmForMaskedLM.from_pretrained(model_name).to(device)  # type: ignore[arg-type]
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def esm2_logits_to_dms_matrix(
    aa_sequence: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    device: str = "cpu",
    set_wt_nan: bool = True,
) -> pd.DataFrame:
    """Extract ESM2 wild-type marginal logits as a DMS-style matrix.

    Runs the full (unmasked) sequence through ``EsmForMaskedLM`` and extracts
    the per-position logits for the 20 standard amino acids. Higher logit
    means the model considers that amino acid more probable at that position.

    This uses "wild-type marginals": a single forward pass with the unmasked
    sequence. Each position's logits are conditioned on the true residues at
    all other positions. This is sufficient for detecting codon table bias
    (systematic score differences between single- vs multi-nucleotide
    mutations) and is much faster than masked marginals (L forward passes).

    Args:
        aa_sequence: Amino acid sequence (1-letter codes, standard 20 only).
        model: Pre-loaded ``EsmForMaskedLM`` instance. If None, loads from
            ``model_name``. Pass a pre-loaded model to avoid reloading when
            calling this function multiple times.
        tokenizer: Pre-loaded HuggingFace tokenizer matching the model. Required
            if ``model`` is provided; ignored if ``model`` is None.
        model_name: HuggingFace model identifier. Only used if ``model`` is None.
        device: Device for inference. Only used if ``model`` is None.
        set_wt_nan: Whether to set wild-type positions to NaN (default True).

    Returns:
        DataFrame of shape (20, L) with logits (higher = more probable).
        Index is ``AA_STR_SORTED``, columns are position integers 0..L-1.

    Raises:
        ValueError: If sequence contains non-standard amino acids, or if
            ``model`` is provided without ``tokenizer``.
    """
    if model is not None and tokenizer is None:
        raise ValueError("tokenizer is required when model is provided")

    # Validate sequence
    invalid_aas = set(aa_sequence) - set(AA_STR_SORTED)
    if invalid_aas:
        raise ValueError(
            f"Sequence contains non-standard amino acids: {invalid_aas}. "
            f"Only standard 20 ({AA_STR_SORTED}) are supported."
        )

    # Load model if not provided
    if model is None:
        model, tokenizer = load_esm2_for_masked_lm(model_name, device)
    assert tokenizer is not None  # guaranteed by check above

    # Build AA -> ESM2 vocab index mapping
    esm_indices = esm2_aa_vocab_indices(tokenizer)

    # Tokenize: ESM2 adds <cls> prefix and <eos> suffix
    encoded = tokenizer(
        aa_sequence,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    attention_mask = encoded["attention_mask"].to(next(model.parameters()).device)

    # Forward pass (single pass, wild-type marginals)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # logits shape: (1, seq_len+2, vocab_size) — includes <cls> and <eos>
        logits = outputs.logits[0]

    # Strip <cls> (position 0) and <eos> (last position) to get (L, vocab_size)
    seq_logits = logits[1:-1]

    assert seq_logits.shape[0] == len(aa_sequence), (
        f"Expected {len(aa_sequence)} positions after stripping special tokens, "
        f"got {seq_logits.shape[0]}"
    )

    # Extract logits for the 20 standard amino acids: (L, 20)
    aa_logits = seq_logits[:, esm_indices].cpu().numpy()

    # Build DataFrame: (20, L) with AA rows × position columns
    matrix = pd.DataFrame(
        aa_logits.T,
        index=list(AA_STR_SORTED),
        columns=list(range(len(aa_sequence))),
    )

    if set_wt_nan:
        matrix = mask_wt_to_nan(matrix, aa_sequence)

    return matrix
