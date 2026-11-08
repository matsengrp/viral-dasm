"""viral-dasm: from mat-pcp outputs to every figure in the manuscript.

Every path this pipeline reads is in ``config.yaml``. Nothing is resolved from
a manifest, a registry or an artifact-root default: a reader with their own
GISAID access edits that one file to point at their own mat-pcp run.

Run from the repository root with the environment already active -- rules call
bare commands, not ``uv run`` or ``.venv/bin/``.

Milestone 1 (issue #2) is what exists so far: the vendored training code, and
the gates that prove it still trains the model ``dasm2-experiments`` trained.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(workflow.basedir) / "scripts"))

from lib import (  # noqa: E402
    ARTIFACT_ROOT,
    CANARY_MODEL,
    CANARY_STEPS,
    FAST_TEST_PARQUET,
    FULL_DEPTH,
    TRAIN_SEED,
)

include: "workflow/corpus.smk"
include: "workflow/gates.smk"


rule all:
    """Everything M1 produces. Later milestones extend this."""
    input:
        rules.gates.input,
