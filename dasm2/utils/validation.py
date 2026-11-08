"""Batch validation helpers used by head/loss forward paths.

The pre-#621 mixin home for this helper was ``dasm2/heads/mixins.py``; the
mixins were deleted in #621 Task #4 and this helper moved here so the few
remaining head forwards (and ``ContrastiveBindingHead``,
``PooledPredictionHead`` subclasses) can keep their fail-fast batch-key
checks without resurrecting the mixins module.
"""

from typing import Any, Dict, List


def validate_batch_keys(
    batch: Dict[str, Any],
    required_keys: List[str],
    context: str,
    dataset_hint: str,
) -> None:
    """Validate that required keys exist in a batch dictionary.

    Args:
        batch: Batch dictionary to validate
        required_keys: List of keys that must be present
        context: Description of the calling context (e.g.,
            ``"ContrastiveBindingHead.forward()"``).
        dataset_hint: Hint about which dataset should be used (e.g.,
            ``"create_contrastive_binding_dataset"``).

    Raises:
        ValueError: If any required keys are missing.
    """
    missing = [k for k in required_keys if k not in batch]
    if missing:
        raise ValueError(
            f"{context} missing required batch keys: {missing}. "
            f"Got keys: {list(batch.keys())}. "
            f"Ensure {dataset_hint} was used."
        )
