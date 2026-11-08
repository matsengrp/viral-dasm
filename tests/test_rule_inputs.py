"""Every rule declares the code it runs, so editing that code reruns it.

Snakemake decides a job is up to date by comparing its declared inputs'
mtimes against its outputs. Code a rule executes but does not declare is
invisible to that comparison: edit it and the rule reports nothing to be
done, leaving an output produced by a version of the code that no longer
exists. The corpora this pipeline writes are what the paper's models train
on, so a corpus that silently fails to rebuild is the expensive failure.

**Direct imports, not the transitive closure, and never a glob.** A glob
over the vendored package would be complete and would also make a comment
edit anywhere in it invalidate the corpora -- and, through them,
the training runs below. The cost is asymmetric: minutes of CPU for a real
rebuild, GPU-days for a false one. What covers the residual transitive risk
is ``scripts/check_corpus.py``, which gates each corpus's row-order digest
against the value pinned in ``config.yaml`` and so fires on a corpus whose
rows moved however that happened.

**Extra declarations pass.** This asserts imports are a subset of
declarations, not equality. ``build_neutral`` declares
``dasm2/neutral/tabular_rates.py``, which its script does not import
directly, and that over-declaration is harmless.
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import pytest
import snakemake.parser as smk_parser
from snakemake.sourcecache import SourceCache, infer_source_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_SMK = REPO_ROOT / "workflow" / "corpus.smk"

#: Each rule that runs repository code, and the file that code starts in.
#:
#: For a rule with a ``shell:`` directive the entry point is the script the
#: shell command runs. ``render_era_map`` has a ``run:`` block instead, which
#: is not a file; its entry point is the one sibling script that block
#: imports.
ENTRY_POINTS = {
    "build_neutral": "scripts/build_era_neutral_models.py",
    "verify_neutral": "scripts/verify_era_neutral_models.py",
    "render_era_map": "scripts/write_era_config.py",
    "prepare_parquet": "scripts/prepare_viral_parquet.py",
    "check_corpus": "scripts/check_corpus.py",
    "audit_censor": "scripts/audit_censor.py",
}

#: Rules with no entry point of their own, and why.
EXEMPT = {
    "corpora": "an aggregator: it has no shell or run directive and runs no code",
    "validate_tests": (
        "its entry point is the test suite, whose dasm2 imports are a "
        "different question -- the suite declares every test file instead"
    ),
}

#: Read and validated at ``scripts/lib.py`` import time rather than imported
#: by anything, so no walk of the import graph can find it. Stated as a
#: closure on the required set: a rule that needs ``lib`` needs what ``lib``
#: reads.
CONFIG = "config.yaml"
LIB = "scripts/lib.py"


class _StubWorkflow:
    """Everything ``snakemake.parser.parse`` touches, and nothing more.

    ``parse`` translates the ``.smk`` dialect into plain Python that ``ast``
    can read -- rules become functions carrying ``@workflow.rule(name=...)``
    and ``@workflow.input(...)`` decorators. It needs no config, no DAG and no
    workdir lock, which matters because this suite runs inside the
    ``validate_tests`` job: that job already holds the lock, so a test that
    went through ``snakemake.api`` or ``--dryrun`` would deadlock against the
    run that invoked it.

    ``snakemake.parser`` is an internal API. Verified against snakemake
    9.27.1; if an upgrade breaks this, the two routes that do *not* work are
    ``import snakemake.workflow`` (``ImportError`` from a partially
    initialized module unless ``snakemake.api`` is imported first, and
    reaching its rules means building a DAG) and ``ast.parse`` on the raw
    ``.smk`` (``SyntaxError`` at ``wildcard_constraints:``, which is not
    Python).
    """

    def __init__(self, cache_dir: Path):
        self.sourcecache = SourceCache(cache_dir)
        self.modifier = None
        self.rulecount = 0


def declared_inputs(smk_path: Path) -> dict[str, set[str]]:
    """Each rule's literally-declared input paths, by rule name.

    Non-literal entries -- ``era_map=ERA_MAP``, ``pcp=lambda wc: ...``,
    ``validate_tests``' generator -- are dropped, which is right: every path
    this test reasons about is written out as a string.
    """
    with tempfile.TemporaryDirectory() as tmp:
        code = smk_parser.parse(
            infer_source_file(str(smk_path)), _StubWorkflow(Path(tmp)), {}
        )[0]

    found: dict[str, set[str]] = {}
    for node in ast.parse(code).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        name: str | None = None
        paths: set[str] = set()
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            attr = func.attr if isinstance(func, ast.Attribute) else None
            if attr == "rule":
                for keyword in decorator.keywords:
                    if keyword.arg == "name":
                        name = ast.literal_eval(keyword.value)
            elif attr == "input":
                args = list(decorator.args) + [k.value for k in decorator.keywords]
                for arg in args:
                    try:
                        value = ast.literal_eval(arg)
                    except ValueError:
                        continue
                    if isinstance(value, str):
                        paths.add(value)
        if name is not None:
            found[name] = paths
    return found


def imported_names(path: Path) -> set[str]:
    """Every module name the file imports directly, dotted names intact.

    ``tests/test_import_order.py``'s ``first_import_lines`` cannot stand in
    here: it keys its dict by ``name.split(".")[0]``, which collapses
    ``dasm2.data.local``, ``dasm2.io`` and a bare ``import dasm2`` into one
    entry -- exactly the distinction this test exists to make.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
    return names


def resolve(name: str) -> str | None:
    """A module name as a repository-relative path, or None if it is external.

    ``dasm2.a.b`` resolves to ``dasm2/a/b.py``, falling back to
    ``dasm2/a/b/__init__.py``; a bare name matching a sibling script resolves
    to ``scripts/<name>.py``. Anything else -- the standard library, torch,
    pandas -- is not this repository's code and is not declared.
    """
    if name == "dasm2" or name.startswith("dasm2."):
        stem = Path(*name.split("."))
        for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
            if (REPO_ROOT / candidate).is_file():
                return str(candidate)
        return None
    sibling = Path("scripts") / f"{name}.py"
    return str(sibling) if (REPO_ROOT / sibling).is_file() else None


def required_inputs(entry_point: str) -> set[str]:
    """The entry point, the repository code it imports, and ``config.yaml``."""
    required = {entry_point}
    required.update(
        path
        for path in map(resolve, imported_names(REPO_ROOT / entry_point))
        if path is not None
    )
    if LIB in required:
        required.add(CONFIG)
    return required


@pytest.mark.parametrize("rule", sorted(ENTRY_POINTS), ids=str)
def test_rule_declares_the_code_it_runs(rule: str):
    declared = declared_inputs(CORPUS_SMK)[rule]
    missing = sorted(required_inputs(ENTRY_POINTS[rule]) - declared)
    assert not missing, (
        f"rule {rule} runs {ENTRY_POINTS[rule]}, which imports "
        f"{', '.join(missing)}, but does not declare {'it' if len(missing) == 1 else 'them'} "
        f"as inputs. Snakemake compares only declared inputs' mtimes, so "
        f"editing {'that file' if len(missing) == 1 else 'those files'} "
        f"leaves this rule reporting nothing to be done and its outputs "
        f"produced by code that no longer exists. Add "
        f"{'it' if len(missing) == 1 else 'them'} to the rule's input block."
    )


def test_every_rule_is_classified():
    """A rule added later fails here until someone says what it runs.

    Without this, the parametrization above silently stops covering the
    workflow the moment a rule is added -- the failure mode this whole file
    exists to prevent, one level up.
    """
    parsed = set(declared_inputs(CORPUS_SMK))
    classified = set(ENTRY_POINTS) | set(EXEMPT)
    assert parsed == classified, (
        f"workflow/corpus.smk holds rules {sorted(parsed - classified)} that "
        f"this test does not classify, and classifies "
        f"{sorted(classified - parsed)} that it does not hold. Either give "
        f"the rule an ENTRY_POINTS entry naming the file its shell or run "
        f"directive starts in, or an EXEMPT entry saying why it runs no "
        f"repository code."
    )
