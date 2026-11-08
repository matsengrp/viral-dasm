"""Every rule that runs a script declares the code that script runs.

Snakemake decides a job is up to date by comparing its declared inputs'
mtimes against its outputs. Code a rule executes but does not declare is
invisible to that comparison: edit it and the rule reports nothing to be
done, leaving an output produced by a version of the code that no longer
exists. The corpora this pipeline writes are what the paper's models train
on, so a corpus that silently fails to rebuild is the expensive failure.

**Direct imports, not the transitive closure, and never a glob.** A glob
over ``dasm2`` would be complete and would also make a comment edit anywhere
in the package invalidate the corpora -- and, through them, the training runs
below. The cost is asymmetric: minutes of CPU for a real rebuild, GPU-days
for a false one. What covers the residual transitive risk is
``scripts/check_corpus.py``, which gates each corpus's row-order digest
against the value pinned in ``config.yaml`` and so fires on a corpus whose
rows moved however that happened.

The closure stops at the entry point's own imports. A module the entry point
imports may import more repository code -- ``scripts/lib.py`` imports
``dasm2``, and ``scripts/build_era_neutral_models.py`` imports
``dasm2.neutral.scov2_rates`` -- and that second hop is deliberately not
walked. ``config.yaml`` is the one exception, because ``lib`` reads and
validates it at import rather than importing it, so no walk of the import
graph could ever reach it.

**Extra declarations pass.** This asserts imports are a subset of
declarations, not equality. ``build_neutral`` declares
``dasm2/neutral/tabular_rates.py``, which its script does not import
directly, and that over-declaration is harmless.
"""

from __future__ import annotations

import ast
import functools
import tempfile
from pathlib import Path

import pytest
import snakemake.parser as smk_parser
from snakemake.sourcecache import SourceCache, infer_source_file

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every file that can define a rule. Globbed rather than listed so a
#: workflow file added later is covered the day it appears, which is the
#: point: M3 adds the training rules, whose outputs cost GPU-days.
RULE_FILES = sorted(
    [REPO_ROOT / "Snakefile"] + list((REPO_ROOT / "workflow").glob("*.smk"))
)

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

#: Rules with no entry-point script of their own, and why.
EXEMPT = {
    "all": "an aggregator: no shell or run directive, so it runs no code",
    "corpora": "an aggregator: no shell or run directive, so it runs no code",
    "validate_tests": (
        "its entry point is the test suite, which runs against all of dasm2 "
        "by design. Declaring that would be the glob this file rejects, so "
        "the rule declares every test file instead and relies on the suite's "
        "own assertions rather than on mtimes"
    ),
}

#: Read and validated at ``scripts/lib.py`` import time rather than imported,
#: so no walk of the import graph can find it. Stated as a closure on the
#: required set: a rule that needs ``lib`` needs what ``lib`` reads.
CONFIG_PATH = "config.yaml"
LIB_PATH = "scripts/lib.py"

#: The input key a rule gives its entry-point script, where it has one.
#: Checked against ``ENTRY_POINTS`` so the two cannot drift apart.
SCRIPT_KEYS = ("script", "era_lib")


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
    9.27.1. If an upgrade breaks this, the two routes that do *not* work are
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


def _literal_strings(decorator: ast.Call) -> dict[str, str]:
    """The decorator's string-literal arguments, by keyword where it has one.

    Non-literal entries -- ``era_map=ERA_MAP``, ``pcp=lambda wc: ...``,
    ``validate_tests``' generator -- are dropped, which is right: every path
    this file reasons about is written out as a string. A literal list is
    dropped too, which would show up as a missing declaration rather than as
    a silent pass.
    """
    found: dict[str, str] = {}
    args: list[tuple[str, ast.expr]] = [
        (f"_{i}", a) for i, a in enumerate(decorator.args)
    ]
    args += [(k.arg or "_", k.value) for k in decorator.keywords]
    for key, node in args:
        try:
            value = ast.literal_eval(node)
        except (ValueError, TypeError):
            continue
        if isinstance(value, str):
            found[key] = value
    return found


@functools.cache
def declared_inputs() -> dict[str, dict[str, str]]:
    """Each rule's literally-declared input paths, keyed by rule then by key.

    Parsed once for the whole module: the tests below would otherwise
    re-parse every workflow file a dozen times over.
    """
    found: dict[str, dict[str, str]] = {}
    for rule_file in RULE_FILES:
        with tempfile.TemporaryDirectory() as tmp:
            code = smk_parser.parse(
                infer_source_file(str(rule_file)), _StubWorkflow(Path(tmp)), {}
            )[0]
        for node in ast.parse(code).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            name: str | None = None
            paths: dict[str, str] = {}
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                func = decorator.func
                attr = func.attr if isinstance(func, ast.Attribute) else None
                if attr == "rule":
                    name = _literal_strings(decorator).get("name")
                elif attr == "input":
                    paths.update(_literal_strings(decorator))
            if name is not None:
                found[name] = paths
    return found


def imported_names(path: Path) -> set[str]:
    """Every module name the file imports directly, dotted names intact.

    A ``from`` import contributes both its module and each name under it, so
    ``from dasm2.sequence import masking`` reaches
    ``dasm2/sequence/masking.py`` rather than stopping at the package. Names
    that are not modules -- ``from dasm2.io import read_fasta_first_sequence``
    -- resolve to nothing and drop out.

    ``tests/test_import_order.py``'s ``first_import_lines`` cannot stand in
    here: it keys its dict by ``name.split(".")[0]``, which collapses
    ``dasm2.data.local``, ``dasm2.io`` and a bare ``import dasm2`` into one
    entry -- exactly the distinction this file exists to make.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def resolve(name: str) -> str | None:
    """A module name as a repository-relative path, or None if it is external.

    ``dasm2.a.b`` resolves to ``dasm2/a/b.py``, falling back to
    ``dasm2/a/b/__init__.py``; a bare name matching a sibling script resolves
    to ``scripts/<name>.py``. Anything else -- the standard library, torch,
    pandas, and the attribute names a ``from`` import pulls out of a module --
    is not this repository's code and is not declared.
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
    """The entry point, the repository code it imports, and ``config.yaml``.

    ``config.yaml`` rides along with ``scripts/lib.py`` rather than being
    found by the walk; see this module's docstring for why it has to.
    """
    required = {entry_point}
    required.update(
        path
        for path in map(resolve, imported_names(REPO_ROOT / entry_point))
        if path is not None
    )
    if LIB_PATH in required:
        required.add(CONFIG_PATH)
    return required


@pytest.mark.parametrize("rule", sorted(ENTRY_POINTS))
def test_rule_declares_the_code_it_runs(rule: str):
    """The subset check.

    It cannot pass vacuously: ``required_inputs`` always contains the entry
    point itself, so a parse that returned no declarations for this rule
    fails here rather than reporting nothing to check.
    """
    declared = set(declared_inputs()[rule].values())
    missing = sorted(required_inputs(ENTRY_POINTS[rule]) - declared)
    them = "it" if len(missing) == 1 else "them"
    assert not missing, (
        f"rule {rule} runs {ENTRY_POINTS[rule]}, which imports "
        f"{', '.join(missing)}, but does not declare {them} as inputs. "
        f"Snakemake compares only declared inputs' mtimes, so editing {them} "
        f"leaves this rule reporting nothing to be done and its outputs "
        f"produced by code that no longer exists. Add {them} to the rule's "
        f"input block."
    )


@pytest.mark.parametrize("rule", sorted(ENTRY_POINTS))
def test_entry_point_matches_the_rule_declaration(rule: str):
    """``ENTRY_POINTS`` is hand-written; the rule is what actually runs.

    Without this the two drift silently: point a rule's shell command at a
    different script and the table above goes on checking the old script's
    imports, passing while testing nothing that runs.
    """
    declared = declared_inputs()[rule]
    named = {declared[key] for key in SCRIPT_KEYS if key in declared}
    assert named == {ENTRY_POINTS[rule]}, (
        f"rule {rule} declares {sorted(named) or 'no script'} under "
        f"{' or '.join(SCRIPT_KEYS)}, but ENTRY_POINTS says its entry point "
        f"is {ENTRY_POINTS[rule]}. Whichever is stale, the import check above "
        f"is reading a file the rule does not run."
    )


def test_every_rule_is_classified():
    """A rule added later fails here until someone says what it runs.

    Covers ``Snakefile`` and every ``workflow/*.smk``, not just the corpus
    stage, so M3's training rules are in scope the moment they are written --
    which is where this matters most, their outputs being GPU-days each.
    """
    parsed = set(declared_inputs())
    classified = set(ENTRY_POINTS) | set(EXEMPT)
    assert parsed == classified, (
        f"{', '.join(str(p.relative_to(REPO_ROOT)) for p in RULE_FILES)} hold "
        f"rules {sorted(parsed - classified)} that this test does not "
        f"classify, and it classifies {sorted(classified - parsed)} that they "
        f"do not hold. Either give the rule an ENTRY_POINTS entry naming the "
        f"file its shell or run directive starts in, or an EXEMPT entry "
        f"saying why it runs no repository code."
    )
