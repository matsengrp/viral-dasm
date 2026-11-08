"""Concrete tokenizer implementations."""

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from transformers import AutoTokenizer

from dasm2.sequence.constants import AA_AMBIG_IDX, TOKEN_STR_SORTED, TOKEN_TO_IDX
from dasm2.sequence.mlm_constants import (
    MLM_AMBIG_IDX,
    MLM_CLS_IDX,
    MLM_CLS_TOKEN,
    MLM_MASK_IDX,
    MLM_MASK_TOKEN,
    MLM_PAD_IDX,
    MLM_PAD_TOKEN,
    MLM_TOKENS,
)


class AminoAcidTokenizer:
    """Tokenizer for amino acid sequences.

    Converts amino acid sequences to integer token tensors and vice versa.
    Uses standard 20 amino acids plus ambiguous 'X' token for padding/unknown residues.
    Token indices correspond to alphabetically sorted amino acids.
    """

    def __init__(self) -> None:
        """Initialize amino acid tokenizer with netam-compatible vocabulary.

        Creates bidirectional mappings between tokens and indices.
        Vocabulary includes 20 standard amino acids plus ambiguous 'X' token
        plus reserved tokens like '^' separator (netam-compatible: 22 tokens total).
        """
        # Create mapping from token to index using complete netam-compatible vocabulary
        self.token_to_idx = dict(TOKEN_TO_IDX)

        # Create reverse mapping
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

    def tokenize(self, sequences: List[str]) -> Tensor:
        """Convert amino acid sequences to token tensors.

        Args:
            sequences: List of amino acid sequences as strings

        Returns:
            Token tensor of shape [num_sequences, max_length] with integer indices.
            Sequences are right-padded with ambiguous token (AA_AMBIG_IDX).
            Invalid amino acids are replaced with ambiguous token.

        Note:
            Empty input returns empty tensor with shape [0, 0].
        """
        if not sequences:
            return torch.empty(0, 0, dtype=torch.long)

        # Find maximum sequence length
        max_len = max(len(seq) for seq in sequences)

        # Create tensor for tokenized sequences
        tokens = torch.full(
            (len(sequences), max_len),
            fill_value=AA_AMBIG_IDX,  # Padding with ambiguous token
            dtype=torch.long,
        )

        # Tokenize each sequence
        for i, sequence in enumerate(sequences):
            for j, char in enumerate(sequence):
                char_upper = char.upper()
                if char_upper not in self.token_to_idx:
                    # For now, use ambiguous token but could be made stricter
                    # raise ValueError(f"Invalid token '{char}' at position {j} in sequence {i}")
                    tokens[i, j] = AA_AMBIG_IDX
                else:
                    tokens[i, j] = self.token_to_idx[char_upper]

        return tokens

    def detokenize(self, tokens: Tensor) -> List[str]:
        """Convert token tensors back to amino acid sequences.

        Args:
            tokens: Token tensor of shape [num_sequences, seq_length] with integer indices

        Returns:
            List of amino acid sequence strings. Sequences are truncated at first
            padding token (ambiguous token). Invalid token indices are skipped.
        """
        sequences = []

        for token_seq in tokens:
            sequence = ""
            for token_idx in token_seq:
                idx = token_idx.item()
                if idx in self.idx_to_token:
                    # Stop at padding tokens (explicit check)
                    if idx == AA_AMBIG_IDX:
                        break
                    token = self.idx_to_token[idx]
                    sequence += token
            sequences.append(sequence)

        return sequences

    @property
    def vocab_size(self) -> int:
        """Size of token vocabulary including ambiguous and reserved tokens.

        Returns:
            Total vocabulary size (22: 20 standard amino acids + X ambiguous + ^ separator)
        """
        return len(TOKEN_STR_SORTED)


"""Module-level singleton for amino acid tokenization.

AminoAcidTokenizer is stateless (vocabulary is derived from constants), so a
single shared instance avoids redundant dict construction. Import this instead
of creating per-module ``AminoAcidTokenizer()`` instances.
"""
AA_TOKENIZER = AminoAcidTokenizer()


_MLM_RAW_INPUT_ALLOWED_CHARS = frozenset(TOKEN_STR_SORTED)
"""Single characters ``MLMTokenizer.tokenize`` recognizes as one of the 20
amino acids, ``X`` (ambiguous), or ``^`` (separator) — exactly the existing
22-token alphabet. Anything else falls back to ``MLM_AMBIG_IDX`` unless it
matches a multi-character special token first (see ``_MLM_SPECIAL_TOKENS``)."""

_MLM_SPECIAL_TOKENS: Tuple[Tuple[str, int], ...] = (
    (MLM_MASK_TOKEN, MLM_MASK_IDX),
    (MLM_PAD_TOKEN, MLM_PAD_IDX),
    (MLM_CLS_TOKEN, MLM_CLS_IDX),
)
"""Multi-character vocabulary tokens, checked at each scan position before
falling back to single-character lookup — the same longest-match-first
scheme HuggingFace's ``PreTrainedTokenizer.tokenize`` uses (a trie split
over ``unique_no_split_tokens``) to keep a multi-character vocabulary entry
from being shredded into its individual characters."""


class MLMTokenizer:
    """Tokenizer for the 25-token MLM vocabulary.

    Independent of ``AA_TOKENIZER``'s 22-token singleton so existing DASM
    checkpoints and heads are unaffected (see ``dasm2/sequence/mlm_constants.py``).
    Every sequence is prefixed with a single ``CLS`` token. Heavy/light pairs
    are expected to already be joined with ``^`` (see
    ``dasm2.data.functional_registry.make_paired_seq`` /
    ``dasm2.data.dataset_utils.concat_chains_with_separator``, the same
    join-before-tokenize convention ``TokenizeParent``/
    ``tokenize_single_aa_sequence`` use for the evolutionary-dataset
    pipeline) before being passed in here; single-chain sequences carry no
    separator at all.

    ``tokenize()`` does not pad: it returns one variable-length tensor per
    input sequence, mirroring ``tokenize_single_aa_sequence``
    (``dasm2/tensors_of_seqs.py``) — batch-level padding across
    different-length sequences is ``MaskedTransformCollator``'s job, not the
    tokenizer's.
    """

    def __init__(self) -> None:
        """Initialize the MLM tokenizer.

        Creates bidirectional mappings between tokens and indices for the
        25-token MLM vocabulary (20 standard amino acids + ambiguous ``X`` +
        ``^`` separator + ``<mask>`` + ``<pad>`` + ``<cls>``).
        """
        self.token_to_idx = {token: idx for idx, token in enumerate(MLM_TOKENS)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

    def tokenize(self, sequences: List[str]) -> List[Tensor]:
        """Convert amino acid sequences to token tensors, with a CLS prefix.

        Args:
            sequences: List of amino acid sequences as strings. Paired
                heavy/light sequences must already be ``^``-joined.

        Returns:
            One 1-D LongTensor per input sequence — no padding, so
            sequences of different lengths return differently-shaped
            tensors. Every tensor is prefixed with ``MLM_CLS_IDX``. Each
            remaining position is filled by scanning left to right: a
            multi-character special token (``<mask>``/``<pad>``/``<cls>``)
            matching at the current position is consumed as one token
            (mirroring the trie-based "no-split-tokens" pre-pass real
            tokenizers such as HuggingFace's ``PreTrainedTokenizer`` /
            ESM's use, so a multi-character vocabulary entry is never
            shredded into individual characters); otherwise one character
            is consumed and mapped to its amino acid index, or to
            ``MLM_AMBIG_IDX`` if it isn't in the 22-token alphabet.
        """
        return [self._tokenize_one(sequence) for sequence in sequences]

    def _tokenize_one(self, sequence: str) -> Tensor:
        token_idxs = [MLM_CLS_IDX]
        i = 0
        n = len(sequence)
        while i < n:
            for special_token, special_idx in _MLM_SPECIAL_TOKENS:
                if sequence.startswith(special_token, i):
                    token_idxs.append(special_idx)
                    i += len(special_token)
                    break
            else:
                char_upper = sequence[i].upper()
                token_idxs.append(
                    self.token_to_idx[char_upper]
                    if char_upper in _MLM_RAW_INPUT_ALLOWED_CHARS
                    else MLM_AMBIG_IDX
                )
                i += 1
        return torch.tensor(token_idxs, dtype=torch.long)

    def detokenize(self, token_seqs: Union[Tensor, List[Tensor]]) -> List[str]:
        """Convert token tensors back to amino acid sequences.

        Inverts ``tokenize()``: the token at position 0 is always the
        auto-prepended ``<cls>`` prefix and is dropped, and a trailing run
        of ``<pad>`` (batch padding, added by ``MaskedTransformCollator``,
        never genuine content) is stripped. Every other position —
        including a ``<mask>`` left by masking, or a literal
        multi-character special token ``tokenize()`` matched mid-sequence —
        round-trips back to its own token string, the same way an amino
        acid character does.

        Args:
            token_seqs: Either a list of 1-D tensors (``tokenize()``'s own
                output) or a 2-D ``[num_sequences, seq_length]`` tensor (a
                collator-padded batch) — both iterate to one 1-D row per
                sequence.

        Returns:
            List of amino acid sequence strings.
        """
        sequences = []

        for token_seq in token_seqs:
            idxs = [token_idx.item() for token_idx in token_seq]

            end = len(idxs)
            while end > 0 and idxs[end - 1] == MLM_PAD_IDX:
                end -= 1
            idxs = idxs[:end]

            if idxs and idxs[0] == MLM_CLS_IDX:
                idxs = idxs[1:]

            sequences.append("".join(self.idx_to_token.get(idx, "") for idx in idxs))

        return sequences

    @property
    def vocab_size(self) -> int:
        """Size of the MLM token vocabulary.

        Returns:
            Total vocabulary size (25: 20 standard amino acids + X ambiguous +
            ^ separator + <mask> + <pad> + <cls>).
        """
        return len(MLM_TOKENS)


"""Module-level singleton for MLM tokenization.

MLMTokenizer is stateless (vocabulary is derived from constants), so a single
shared instance avoids redundant dict construction. Import this instead of
creating per-module ``MLMTokenizer()`` instances.
"""
MLM_TOKENIZER = MLMTokenizer()


class ESM2Tokenizer:
    """Tokenizer for ESM2 models using HuggingFace tokenizer.

    ESM2 uses a different tokenization scheme than DASM:
    - 33 tokens (vs DASM's 22)
    - Frequency-sorted vocabulary (vs alphabetical)
    - Adds <cls> prefix and <eos> suffix automatically
    - Uses <pad> token (idx 1) for padding

    This tokenizer wraps HuggingFace's AutoTokenizer for use with DASM2
    data pipelines and ESM2Encoder.

    Note:
        ESM2Encoder.forward() strips <cls>/<eos> from embeddings, so the
        returned embedding positions align with input sequence positions
        despite the tokenizer adding special tokens.
    """

    def __init__(self, model_name: str = "facebook/esm2_t30_150M_UR50D") -> None:
        """Initialize ESM2 tokenizer.

        Args:
            model_name: HuggingFace ESM2 model identifier
        """
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def create_encoder_batch(
        self,
        sequences: List[str],
        device: str = "cpu",
    ) -> dict[str, Tensor]:
        """Create encoder input batch with token_ids and attention masks.

        This is the primary tokenization method. Returns both token_ids and
        masks in the format expected by ESM2Encoder.forward().

        Args:
            sequences: List of amino acid sequences as strings
            device: Device for output tensors ("cpu", "cuda", "cuda:0", etc.)

        Returns:
            Dict with 'token_ids' and 'masks' tensors ready for ESM2Encoder

        Raises:
            ValueError: If device is invalid or unavailable
        """
        if not sequences:
            return {
                "token_ids": torch.empty(0, 0, dtype=torch.long, device=device),
                "masks": torch.empty(0, 0, dtype=torch.bool, device=device),
            }

        encoded = self._tokenizer(
            sequences,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        try:
            token_ids = encoded["input_ids"].to(device)
            masks = encoded["attention_mask"].bool().to(device)
        except RuntimeError as e:
            raise ValueError(f"Failed to move tensor to device '{device}': {e}") from e

        return {"token_ids": token_ids, "masks": masks}

    def detokenize(self, tokens: Tensor) -> List[str]:
        """Convert token tensors back to amino acid sequences.

        Args:
            tokens: Token tensor of shape [num_sequences, seq_length]

        Returns:
            List of amino acid sequences with special tokens removed
        """
        # Decode and remove special tokens
        sequences = self._tokenizer.batch_decode(tokens, skip_special_tokens=True)
        # Remove spaces that HuggingFace adds between tokens
        return [seq.replace(" ", "") for seq in sequences]

    @property
    def vocab_size(self) -> int:
        """Size of ESM2 token vocabulary (33 for ESM2 models)."""
        return int(self._tokenizer.vocab_size)

    def _get_special_token_id(self, token_attr: str) -> int:
        """Get special token ID with validation.

        Args:
            token_attr: Name of token attribute (e.g., "pad_token_id")

        Returns:
            Token ID as integer

        Raises:
            ValueError: If tokenizer doesn't have the specified token
        """
        token_id: Optional[int] = getattr(self._tokenizer, token_attr, None)
        if token_id is None:
            raise ValueError(f"Tokenizer {self.model_name} has no {token_attr}")
        return token_id

    @property
    def pad_token_id(self) -> int:
        """Padding token index (1 for ESM2)."""
        return self._get_special_token_id("pad_token_id")

    @property
    def cls_token_id(self) -> int:
        """CLS token index (0 for ESM2)."""
        return self._get_special_token_id("cls_token_id")

    @property
    def eos_token_id(self) -> int:
        """EOS token index (2 for ESM2)."""
        return self._get_special_token_id("eos_token_id")
