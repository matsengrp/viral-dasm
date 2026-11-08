"""DASM2: Deep learning framework for modeling biological sequence evolution.

Import components directly from their modules:
- from dasm2.config import ModelConfig, TrainingConfig, etc.
- from dasm2.artifacts import TrainingArtifact
- from dasm2.heads.selection_factor import SelectionFactorHead
- from dasm2.heads.types import HeadType
- from dasm2.molevol.neutral_evolution import neutral_codon_probs
"""

from dasm2.arrow_pool import use_system_arrow_pool

# First, before anything imports pyarrow: Arrow fixes its C++ default pool at
# import time. See the function's docstring.
use_system_arrow_pool()

from dasm2.mha_fastpath import disable_mha_fastpath  # noqa: E402

# Applied at import so it covers every entry point -- CLI, Snakemake, notebooks,
# tests -- and models that bypass DASM2's own encoder. See the function's
# docstring for why this is process-wide rather than call-scoped.
disable_mha_fastpath()
