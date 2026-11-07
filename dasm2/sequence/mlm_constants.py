"""MLM (masked-language-model) token vocabulary.

Extends the amino acid alphabet with ``MASK``, ``PAD``, and ``CLS`` tokens for
MLM training. This vocabulary lives alongside (not in place of) the existing
22-token ``TOKEN_STR_SORTED`` (``dasm2/sequence/constants.py``) — existing DASM
checkpoints and heads are built against those exact indices and must not
shift. Follows the same append-a-new-alphabet-module pattern already used for
the 3Di constants that left with the structural-token support.

See issue #884 / #851 for the full design rationale (why MASK, PAD, and
ambiguous/`X` must be three distinct tokens for an MLM objective).
"""

from typing import Tuple

from dasm2.sequence.constants import AA_SEPARATOR, AA_STR_SORTED

# Special tokens new to the MLM vocabulary. Bracketed multi-character names
# (BERT/HuggingFace convention) rather than single punctuation characters —
# readable in logs/debugging and can never collide with an amino acid code.
# They can only ever be produced by the tokenizer itself (CLS prefix) or the
# collator (MASK/PAD); MLMTokenizer.tokenize() never reads them out of raw
# input text (see _MLM_RAW_INPUT_ALLOWED_CHARS in dasm2/tokenizers.py).
MLM_MASK_TOKEN = "<mask>"
"""Marks a position the model must predict; applied only by the training-time
masking procedure (``MaskedTransformCollator``), never present in raw input."""

MLM_PAD_TOKEN = "<pad>"
"""Batch padding, distinct from ``X`` (ambiguous) so a real ambiguous residue
and a padding position are never the same index."""

MLM_CLS_TOKEN = "<cls>"
"""Prepended once per sequence; gives a fixed whole-sequence pooled position."""

# 25-token MLM vocabulary: 20 AA + X (ambiguous) + ^ (separator) + MASK + PAD + CLS.
# A tuple, not a concatenated string, since the special tokens are now
# multi-character. The first 22 entries are in the same order as
# TOKEN_STR_SORTED, so their indices (0-21) match exactly; MASK/PAD/CLS are
# appended at the end (22-24).
MLM_TOKENS: Tuple[str, ...] = tuple(AA_STR_SORTED) + (
    "X",
    AA_SEPARATOR,
    MLM_MASK_TOKEN,
    MLM_PAD_TOKEN,
    MLM_CLS_TOKEN,
)

MLM_AMBIG_IDX = MLM_TOKENS.index("X")
MLM_SEPARATOR_IDX = MLM_TOKENS.index(AA_SEPARATOR)
MLM_MASK_IDX = MLM_TOKENS.index(MLM_MASK_TOKEN)
MLM_PAD_IDX = MLM_TOKENS.index(MLM_PAD_TOKEN)
MLM_CLS_IDX = MLM_TOKENS.index(MLM_CLS_TOKEN)

MLM_IGNORE_INDEX = -100
"""``F.cross_entropy``'s ``ignore_index`` sentinel for non-masked positions in
``batch["mlm_labels"]``. Shared by ``MaskedTransformCollator`` (producer,
``dasm2/data/masking.py``) and ``MLMLoss`` (consumer, ``dasm2/losses/mlm.py``)
so the two can't silently desync."""
