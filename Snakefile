"""viral-dasm: from mat-pcp outputs to every figure in the manuscript.

Every path this pipeline reads is in ``config.yaml``. Nothing is resolved from
a manifest, a registry or an artifact-root default: a reader with their own
GISAID access edits that one file to point at their own mat-pcp run.

Run from the repository root with the environment already active -- rules call
bare commands, not ``uv run`` or ``.venv/bin/``.

Milestone 1 (issue #2) vendored the training code and the gates that prove it
still trains the model it was ported from. Milestone 2 (issue #6) adds the
stage before it: building a training corpus, and its neutral model, from a
mat-pcp run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(workflow.basedir) / "scripts"))

from lib import (  # noqa: E402
    ARTIFACT_ROOT,
    BUILT_CORPUS_NAMES,
    CANARY_MODEL,
    CANARY_STEPS,
    CORPUS_NAMES,
    CUTOFF,
    FAST_TEST_PARQUET,
    FULL_DEPTH,
    GENE,
    PREP_MEM_MB,
    TRAIN_SEED,
    VERSION,
    corpus,
)

include: "workflow/corpus.smk"
include: "workflow/gates.smk"


rule all:
    """Everything the pipeline produces so far. Later milestones extend this."""
    input:
        rules.gates.input,
        rules.corpora.input,
