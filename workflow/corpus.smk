"""The corpora the pipeline trains on, and the guard that identifies them.

``full_depth`` is pinned rather than built. The paper holds out fold 0 of a
10-fold ``cv_seed`` 42 split, and that split is a row-positional draw: the
index set comes from ``(seed, num_folds, n_rows)`` and which PCP sits at each
index comes from the file's row order. Rebuilding the file would reshuffle the
partition even if every row survived. See ``scripts/check_corpus.py``.

M2 adds the three thinned arms here. Those have no reference curve to match
and draw their own folds, so they are built, not pinned.
"""


rule check_corpus:
    """Row count, fold-0 split sizes, row-order digest and md5.

    Every later milestone inherits this. It is the reason a rebuilt
    ``full_depth`` cannot silently become the paper's corpus.
    """
    output:
        report=f"{ARTIFACT_ROOT}/corpus_guard/{{corpus}}.json",
        membership=f"{ARTIFACT_ROOT}/corpus_guard/{{corpus}}_fold0_membership.npz",
    log:
        f"{ARTIFACT_ROOT}/logs/check_corpus_{{corpus}}.log",
    shell:
        "python scripts/check_corpus.py"
        " --corpus {wildcards.corpus}"
        " --report {output.report}"
        " --membership-output {output.membership}"
        " > {log} 2>&1"
