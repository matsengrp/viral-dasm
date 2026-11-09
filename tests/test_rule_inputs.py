"""Every rule that runs a script declares the code that script runs.

Snakemake decides a job is up to date by comparing its declared inputs
against its outputs. Code a rule executes but does not declare is invisible
to that comparison: edit it and the rule reports nothing to be done, leaving
an output produced by a version of the code that no longer exists. The
corpora this pipeline writes are what the paper's models train on, so a
corpus that silently fails to rebuild is the expensive failure.

(The comparison is on content, not mtime: snakemake 9 records each input's
sha256 in ``.snakemake/metadata`` and reruns on ``Updated input files``. A
``touch`` changes nothing. Declaring the file is what puts it in that record
at all.)

**Direct imports, not the transitive closure, and never a glob.** A glob over
``dasm2`` would be complete and would also make a comment edit anywhere in the
package invalidate the corpora -- and, through them, the training runs below.
The cost is asymmetric: minutes of CPU for a real rebuild, GPU-days for a
false one. What covers the residual transitive risk is
``scripts/check_corpus.py``, which gates each corpus's row-order digest
against the value pinned in ``config.yaml``.

The closure stops at the entry point's own imports, and stops at *modules*.
``from dasm2.neutral import tabular_rates`` requires ``tabular_rates.py`` and
not ``dasm2/neutral/__init__.py``, even though that file executes on import,
for the same reason the second import hop is not walked. A bare ``import
dasm2`` is different: the package itself is then the direct import, and
``dasm2/__init__.py`` is required. ``config.yaml`` is the one thing required
without being imported at all, because ``scripts/lib.py`` reads and validates
it at import, so no walk of the import graph could reach it.

**Extra declarations pass.** This asserts imports are a subset of
declarations, not equality.
"""

from __future__ import annotations

import ast
import functools
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import snakemake.parser as smk_parser
from snakemake.sourcecache import SourceCache, infer_source_file

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_SNAKEFILE = REPO_ROOT / "Snakefile"

#: Each rule that runs repository code, and the file that code starts in.
#: Checked against what the rule actually commands, not just against its
#: input keys; see ``commanded_entry_points``.
ENTRY_POINTS = {
    "build_neutral": "scripts/build_era_neutral_models.py",
    "verify_neutral": "scripts/verify_era_neutral_models.py",
    "render_era_map": "scripts/write_era_config.py",
    "prepare_parquet": "scripts/prepare_viral_parquet.py",
    "check_corpus": "scripts/check_corpus.py",
    "audit_censor": "scripts/audit_censor.py",
    "render_config": "scripts/render_config.py",
    "render_smoke_config": "scripts/render_config.py",
    "check_params": "scripts/check_params.py",
    "smoke": "scripts/train.py",
    "train": "scripts/train.py",
    "read_run": "scripts/read_run.py",
    "check_run": "scripts/check_run.py",
    "runs_table": "scripts/runs_table.py",
}

#: Rules that execute nothing. Asserted, not asserted-by-assertion: each must
#: carry snakemake's ``norun`` marker, so listing a rule here cannot excuse it
#: from the import check while it quietly runs a shell command.
AGGREGATORS = ("all", "corpora", "models")

#: Rules that do run code and are still not import-checked, with the reason.
#: Each must carry a ``shellcmd``, so an entry here describes a real
#: code-running rule rather than hiding one.
DELIBERATE_EXEMPTIONS = {
    "validate_tests": (
        "its entry point is the test suite, which runs against all of dasm2 "
        "by design. Declaring that would be the glob this file rejects, so "
        "the rule declares every test file instead and relies on the suite's "
        "own assertions"
    ),
}

#: Read and validated at ``scripts/lib.py`` import time rather than imported.
CONFIG_PATH = "config.yaml"
LIB_PATH = "scripts/lib.py"

#: ``{input.key}`` in a shell command. The first one is the script python runs.
INPUT_REF = re.compile(r"\{input\.([A-Za-z_][A-Za-z0-9_]*)\}")


class _StubWorkflow:
    """Everything ``snakemake.parser.parse`` touches, and nothing more.

    ``parse`` translates the ``.smk`` dialect into plain Python that ``ast``
    can read -- rules become functions carrying ``@workflow.rule(name=...)``,
    ``@workflow.input(...)``, ``@workflow.shellcmd(...)`` and
    ``@workflow.norun()`` decorators. It needs no config, no DAG and no
    workdir lock, which matters because this suite runs inside the
    ``validate_tests`` job: that job already holds the lock, so a test that
    went through ``snakemake.api`` or ``--dryrun`` would deadlock against the
    run that invoked it.

    ``snakemake.parser`` is an internal API. Verified against snakemake
    9.27.1. The two routes that do *not* work are ``import snakemake.workflow``
    (``ImportError`` from a partially initialized module unless
    ``snakemake.api`` is imported first, and reaching its rules means building
    a DAG) and ``ast.parse`` on the raw ``.smk`` (``SyntaxError`` at
    ``wildcard_constraints:``, which is not Python).
    """

    def __init__(self, cache_dir: Path):
        self.sourcecache = SourceCache(cache_dir)
        self.modifier = None
        self.rulecount = 0


@dataclass(frozen=True)
class ParsedRule:
    """One rule, as much of it as this file reasons about."""

    inputs: dict[str, str] = field(default_factory=dict)
    shellcmd: str | None = None
    norun: bool = False
    run_imports: frozenset[str] = frozenset()


def _to_python(path: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        return smk_parser.parse(
            infer_source_file(str(path)), _StubWorkflow(Path(tmp)), {}
        )[0]


def _literal_strings(decorator: ast.Call) -> dict[str, str]:
    """The decorator's string-literal arguments, by keyword where it has one.

    Non-literal entries -- ``era_map=ERA_MAP``, ``pcp=lambda wc: ...``,
    ``validate_tests``' generator -- are dropped, which is right: every path
    this file reasons about is written out as a string. Positional arguments
    get keys that cannot collide with a real keyword.
    """
    found: dict[str, str] = {}
    args: list[tuple[str, ast.expr]] = [
        (f"#{i}", a) for i, a in enumerate(decorator.args)
    ]
    args += [(k.arg or "#kwargs", k.value) for k in decorator.keywords]
    for key, node in args:
        try:
            value = ast.literal_eval(node)
        except (ValueError, TypeError):
            continue
        if isinstance(value, str):
            found[key] = value
    return found


@functools.cache
def parsed_rules() -> dict[str, ParsedRule]:
    """Every rule reachable from ``Snakefile``, by name.

    File discovery follows ``include:`` rather than globbing a directory, so
    a rule file added anywhere -- ``workflow/rules/train.smk``, say -- is
    covered by construction rather than by the glob happening to match.
    """
    found: dict[str, ParsedRule] = {}
    pending = [ROOT_SNAKEFILE]
    seen: set[Path] = set()
    while pending:
        rule_file = pending.pop()
        if rule_file in seen:
            continue
        seen.add(rule_file)
        tree = ast.parse(_to_python(rule_file))

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "include"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                pending.append(REPO_ROOT / node.args[0].value)

        assert "userule" not in ast.dump(tree), (
            f"{rule_file.relative_to(REPO_ROOT)} uses `use rule ... as ...`. "
            f"A derived rule carries no `@workflow.rule(name=...)`, so it is "
            f"invisible to this file and would escape every check below. "
            f"Teach this parser to read derived rules before adding one."
        )

        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            name: str | None = None
            rule = ParsedRule()
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                func = decorator.func
                attr = func.attr if isinstance(func, ast.Attribute) else None
                if attr == "rule":
                    name = _literal_strings(decorator).get("name")
                elif attr == "input":
                    rule = ParsedRule(
                        {**rule.inputs, **_literal_strings(decorator)},
                        rule.shellcmd,
                        rule.norun,
                        rule.run_imports,
                    )
                elif attr == "shellcmd":
                    shell = _literal_strings(decorator).get("#0")
                    rule = ParsedRule(rule.inputs, shell, rule.norun, rule.run_imports)
                elif attr == "norun":
                    rule = ParsedRule(
                        rule.inputs, rule.shellcmd, True, rule.run_imports
                    )
            if name is not None:
                found[name] = ParsedRule(
                    rule.inputs,
                    rule.shellcmd,
                    rule.norun,
                    frozenset(_imports_in(node)),
                )
    return found


def _imports_in(node: ast.AST) -> set[str]:
    """Module names imported anywhere under ``node``."""
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Import):
            names.update(alias.name for alias in child.names)
        elif isinstance(child, ast.ImportFrom):
            assert not child.level, (
                f"relative import at line {child.lineno}; this file resolves "
                f"module names to repository paths and cannot follow one."
            )
            if child.module:
                names.add(child.module)
                names.update(f"{child.module}.{a.name}" for a in child.names)
    return names


def imported_names(path: Path) -> set[str]:
    """Every module name the file imports directly, dotted names intact.

    ``tests/test_import_order.py``'s ``first_import_lines`` cannot stand in
    here: it keys its dict by ``name.split(".")[0]``, which collapses
    ``dasm2.data.local``, ``dasm2.io`` and a bare ``import dasm2`` into one
    entry -- exactly the distinction this file exists to make.
    """
    return _imports_in(ast.parse(path.read_text()))


def resolve(name: str) -> str | None:
    """A module name as a repository-relative path, or None if it is external.

    ``dasm2.a.b`` resolves to ``dasm2/a/b.py`` and to nothing else: a package
    ``__init__`` is reached only by a bare ``import dasm2``, which names the
    package itself. A bare name matching a sibling script resolves to
    ``scripts/<name>.py``. Everything else -- the standard library, torch,
    pandas, and the attribute names a ``from`` import pulls out of a module --
    is not this repository's code and is not declared.
    """
    if name == "dasm2":
        return "dasm2/__init__.py"
    if name.startswith("dasm2."):
        module = Path(*name.split(".")).with_suffix(".py")
        return str(module) if (REPO_ROOT / module).is_file() else None
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
    if LIB_PATH in required:
        required.add(CONFIG_PATH)
    return required


def commanded_entry_points(rule: ParsedRule) -> set[str]:
    """The repository scripts the rule's own command actually starts.

    For a ``shell:`` rule, the first ``{input.key}`` in the command, resolved
    through the rule's declarations -- that is the file python runs. For a
    ``run:`` rule, whichever sibling scripts the block imports. Reading the
    command rather than an input key named ``script`` is what stops the two
    from drifting: a rule could declare ``script="scripts/a.py"`` and run
    ``python scripts/b.py``, and a key-only check would pass.
    """
    if rule.shellcmd is not None:
        match = INPUT_REF.search(rule.shellcmd)
        if match is None:
            return set()
        declared = rule.inputs.get(match.group(1))
        return {declared} if declared else set()
    return {p for p in map(resolve, rule.run_imports) if p is not None}


@pytest.mark.parametrize("rule", sorted(ENTRY_POINTS))
def test_rule_declares_the_code_it_runs(rule: str):
    """The subset check.

    It cannot pass vacuously: ``required_inputs`` always contains the entry
    point itself, so a parse that returned no declarations for this rule
    fails here rather than reporting nothing to check.
    """
    declared = set(parsed_rules()[rule].inputs.values())
    missing = sorted(required_inputs(ENTRY_POINTS[rule]) - declared)
    them = "it" if len(missing) == 1 else "them"
    assert not missing, (
        f"rule {rule} runs {ENTRY_POINTS[rule]}, which imports "
        f"{', '.join(missing)}, but does not declare {them} as inputs. "
        f"Snakemake tracks only declared inputs, so editing {them} leaves "
        f"this rule reporting nothing to be done and its outputs produced by "
        f"code that no longer exists. Add {them} to the rule's input block."
    )


@pytest.mark.parametrize("rule", sorted(ENTRY_POINTS))
def test_entry_point_is_what_the_rule_commands(rule: str):
    """``ENTRY_POINTS`` is hand-written; the command is what actually runs."""
    commanded = commanded_entry_points(parsed_rules()[rule])
    assert ENTRY_POINTS[rule] in commanded, (
        f"ENTRY_POINTS says rule {rule} starts at {ENTRY_POINTS[rule]}, but "
        f"its own shell command or run block starts "
        f"{sorted(commanded) or 'nothing this file can resolve'}. Whichever "
        f"is stale, the import check is reading a file the rule does not run."
    )


@pytest.mark.parametrize("rule", AGGREGATORS)
def test_aggregators_run_nothing(rule: str):
    """An aggregator is excused from the import check by running no code.

    Asserted rather than asserted-by-listing: without this, moving a rule
    into ``AGGREGATORS`` would excuse it from every check above while it went
    on running a shell command.
    """
    parsed = parsed_rules()[rule]
    assert parsed.norun and parsed.shellcmd is None, (
        f"rule {rule} is listed as an aggregator but carries "
        f"{'a shell command' if parsed.shellcmd else 'a run block'}. An "
        f"aggregator runs no code; this one does, so it needs an "
        f"ENTRY_POINTS entry instead."
    )


@pytest.mark.parametrize("rule", sorted(DELIBERATE_EXEMPTIONS))
def test_deliberate_exemptions_do_run_code(rule: str):
    """The other exemption: a rule that runs code and is still not checked.

    The assertion is the opposite of the one above. An entry here is a claim
    that a code-running rule is deliberately unchecked, so a rule that turned
    out to run nothing belongs in ``AGGREGATORS`` and its recorded reason is
    describing something that no longer exists.
    """
    parsed = parsed_rules()[rule]
    assert parsed.shellcmd is not None and not parsed.norun, (
        f"rule {rule} is recorded as a deliberate exemption -- "
        f"{DELIBERATE_EXEMPTIONS[rule]} -- but runs no command. Move it to "
        f"AGGREGATORS."
    )


def test_every_rule_is_classified():
    """A rule added later fails here until someone says what it runs.

    Covers every file reachable from ``Snakefile`` by ``include:``, so M3's
    training rules are in scope the moment they are written -- which is where
    this matters most, their outputs being GPU-days each.
    """
    parsed = set(parsed_rules())
    classified = set(ENTRY_POINTS) | set(AGGREGATORS) | set(DELIBERATE_EXEMPTIONS)
    assert parsed == classified, (
        f"the workflow holds rules {sorted(parsed - classified)} that this "
        f"test does not classify, and it classifies "
        f"{sorted(classified - parsed)} that it does not hold. Give the rule "
        f"an ENTRY_POINTS entry naming the file its command starts in, or an "
        f"AGGREGATORS/DELIBERATE_EXEMPTIONS entry saying why it is not "
        f"import-checked."
    )
