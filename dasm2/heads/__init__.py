"""Prediction head implementations.

Import heads explicitly from specific modules (no package-level re-exports):
- from dasm2.heads.base import PredictionHead, SharedSelectionFactorHead
- from dasm2.heads.selection_factor import SelectionFactorHead
- from dasm2.heads.qdasm import QDASMEvolHead
- from dasm2.heads.per_site_child import PerSiteChildHead
- from dasm2.heads.dms import DMSFunctionalScoreHead
- from dasm2.heads.delta_affinity import DeltaAffinityHead
- from dasm2.heads.hybrid import HybridEvolHead
- from dasm2.heads.types import HeadType
- from dasm2.heads.factory import create_head

Heads do not own loss mixins. Loss math lives in
``dasm2/losses/`` (one ``Loss`` class per ``LossType``); the trainer
attaches them to tasks via ``Task.loss``.

``DASMEvolHead`` / ``WhichmutHead`` / ``AAOWhichmutHead`` are collapsed
into ``SelectionFactorHead`` (single
identity-forward head consuming the shared SF tensor).

Note: Head classes are NOT re-exported from this package to avoid circular
dependency with dasm2.config. Use explicit imports from submodules instead.
"""

from typing import List

# Minimal package structure - no re-exports to avoid circular dependencies
__all__: List[str] = []
