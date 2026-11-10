"""viral-dasm: from mat-pcp outputs to every figure in the manuscript.

Every path this pipeline reads is in ``config.yaml``. Nothing is resolved from
a manifest, a registry or an artifact-root default: a reader with their own
GISAID access edits that one file to point at their own mat-pcp run.

Run from the repository root with the environment already active -- rules call
bare commands, not ``uv run`` or ``.venv/bin/``.

Milestone 1 (issue #2) vendored the training code and showed it still trains
the model it was ported from. Milestone 2 (issue #6) added the stage before
it: building a training corpus, and its neutral model, from a mat-pcp run.
Issue #8 then removed the apparatus that checked the port, so what remains is
pipeline rather than scaffolding. Milestone 3 (issue #14) added the stage
after it: the paper's seven models, each stopping on the learning-rate floor
and each read back out of its own log into ``data/runs.csv``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(workflow.basedir) / "scripts"))

from lib import (  # noqa: E402
    ARTIFACT_ROOT,
    BUILT_CORPUS_NAMES,
    CORPUS_NAMES,
    CUTOFF,
    GENE,
    PREP_MEM_MB,
    SMOKE_CORPUS,
    VERSION,
    corpus,
    corpus_max_aa_seq_len,
    mat_pcp_path,
)

include: "workflow/corpus.smk"
include: "workflow/train.smk"
include: "workflow/score.smk"


rule all:
    """Everything the pipeline produces so far.

    Every corpus the paper trains on, each one checked against the row order
    its fold-0 draw depends on, plus the neutral models they carry, and then
    the seven models trained on them -- each with its stopping point read out
    of its own log and checked.
    """
    input:
        rules.corpora.input,
        rules.models.input,
        rules.scored.input,
