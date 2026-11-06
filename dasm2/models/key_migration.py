"""State dict key migration utilities for model architecture changes."""

# Head types whose prediction_layer was moved to the shared sf_projection.
# Issue #513: Shared Selection Factor Projection.
# Intentionally frozen — represents the historical set of heads that had
# prediction_layer in pre-#513 checkpoints. Do not add new entries.
_SF_HEAD_PREFIXES = (
    "prediction_heads.DASMEvolHead.prediction_layer.",
    "prediction_heads.WhichmutHead.prediction_layer.",
    "prediction_heads.AAOWhichmutHead.prediction_layer.",
)


def migrate_transformer_block_keys(key: str) -> str:
    """Apply TransformerBlock key renames to a single state_dict key.

    PR #379 replaced nn.TransformerEncoderLayer with TransformerBlock, renaming:
        - self_attn -> self_attention
        - linear1 -> feed_forward.linear1
        - linear2 -> feed_forward.linear2

    Norm/dropout renames (Post-LN, which is what netam checkpoints use):
        - norm1 -> post_attn_norm
        - norm2 -> post_ffn_norm

    Pre-LN keys (``pre_attn_norm``, ``pre_ffn_norm``) are not present in legacy
    checkpoints and pass through unchanged — no migration needed for them.

    This is the shared rename logic used by both netam_loader.map_netam_state_dict
    and load_checkpoint for legacy DASM2 checkpoints.
    """
    _RENAMES = {
        ".self_attn.": ".self_attention.",
        ".linear1.": ".feed_forward.linear1.",
        ".linear2.": ".feed_forward.linear2.",
        ".norm1.": ".post_attn_norm.",
        ".norm2.": ".post_ffn_norm.",
    }
    for old, new in _RENAMES.items():
        key = key.replace(old, new, 1)
    return key


def migrate_sf_projection_keys(key: str) -> str:
    """Remap old per-head prediction_layer keys to shared sf_projection.

    Issue #513: prediction_heads.{Head}.prediction_layer.* → sf_projection.projection.*

    Only remaps keys from SF-consuming heads (DASM, Whichmut, AAO Whichmut).
    """
    for prefix in _SF_HEAD_PREFIXES:
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            return f"sf_projection.projection.{suffix}"
    return key


def unmigrate_transformer_block_keys(key: str) -> str:
    """Reverse of migrate_transformer_block_keys: DASM2 keys → PyTorch keys.

    Only valid for Post-LN checkpoints (which is what nn.TransformerEncoder
    and netam produce). Pre-LN attributes (``pre_attn_norm``, ``pre_ffn_norm``)
    have no PyTorch counterpart and are not handled here.

    Used in tests to copy weights from a DASM2 TransformerStack back into a
    nn.TransformerEncoder for numerical equivalence checks.
    """
    _RENAMES = {
        ".self_attention.": ".self_attn.",
        ".feed_forward.linear1.": ".linear1.",
        ".feed_forward.linear2.": ".linear2.",
        ".post_attn_norm.": ".norm1.",
        ".post_ffn_norm.": ".norm2.",
    }
    for old, new in _RENAMES.items():
        key = key.replace(old, new, 1)
    return key
