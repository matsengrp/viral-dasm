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

**The data a script opens is checked too, where the path is one
``config.yaml`` names.** Code reaches this repository's own files by importing
them, and the import walk above is what finds those. It reaches mat-pcp's files
by calling ``lib.mat_pcp_path("<key>")``, which no import walk can see: the key
is a string, and ``config.yaml`` gives the mat-pcp root and each file's name
rather than its content. A rule that reads one of those and does not declare it
has the same defect -- re-emit the table, and the rule reports nothing to be
done -- and it is the defect this file's own subject matter made easy to miss,
``rule score`` having read six of them while declaring three.

The declared side is the key as the rule's own input block spells it, or as a
module-level helper the block names does: ``founders=founder_fastas`` is how a
rule declares a directory's *contents*, which is what ``config.yaml`` names the
directory for. The same one-hop boundary as above applies on the reading side:
a key reached only through a ``lib`` helper -- ``clade_roster()`` calls
``mat_pcp_path("clade_founders_dir")`` -- is not seen here, and a key passed as
a variable is not seen at all. What is checked is every key the code a rule
actually runs names as a literal.

That second blind spot has already cost something, so it is worth naming.
``lib.master_table_path`` picks its key out of a dict keyed by era, so no
script anywhere names ``master_table_file`` as a literal, and ``build_neutral``
and ``verify_neutral`` both read the fitted rate tables while declaring
neither. This check did not catch it -- a review did -- and the declarations
are now written out by hand. ``check_params.py`` loops over
``TRAINING_CONFIG_FILES`` the same way, which is harmless only because it
compares paths rather than reading files. Resolving one hop into an imported
module would close both; it is not done here.

**A path written down is not a file read.** ``scripts/render_config.py``
renders four of these paths into a training config and never opens them, so the
rendered config follows ``config.yaml`` and not the tables. Those keys are
listed in :data:`PATH_ONLY_KEYS` with the reason, and each listing is itself
asserted against the script.
"""

from __future__ import annotations

import ast
import functools
import re
import tempfile
from dataclasses import dataclass, field, replace
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
    "train_eval": "scripts/train_eval.py",
    "runs_table": "scripts/runs_table.py",
    "check_clades": "scripts/check_clades.py",
    "score": "scripts/score_run.py",
    "scored_table": "scripts/scored_table.py",
    # The panel rules run a notebook as their script. `marimo export html`
    # executes every cell, so the notebook is the entry point and its imports
    # are the code the rule runs.
    "agreement_mutations": "notebooks/agreement_mutations.py",
    "agreement_ladders": "notebooks/agreement_ladders.py",
    "compose_agreement": "notebooks/agreement.py",
}

#: Rules that execute nothing. Asserted, not asserted-by-assertion: each must
#: carry snakemake's ``norun`` marker, so listing a rule here cannot excuse it
#: from the import check while it quietly runs a shell command.
AGGREGATORS = ("all", "corpora", "models", "scored", "panels", "figures")

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
    "publish_scored_table": (
        "it runs `cp` and no repository code, so there is no entry point to "
        "read imports from. What it copies is `rule score`'s output and what "
        "it copies to is a path this repository tracks; neither depends on a "
        "module whose edit should invalidate the copy"
    ),
}

#: Read and validated at ``scripts/lib.py`` import time rather than imported.
CONFIG_PATH = "config.yaml"
LIB_PATH = "scripts/lib.py"

#: ``lib.mat_pcp_path`` -- the one way a script names a mat-pcp file. Matched
#: as a call with a literal first argument, so the key cannot be picked up out
#: of a docstring or an error message that happens to quote it.
MAT_PCP_CALL = "mat_pcp_path"


@dataclass(frozen=True)
class PathOnly:
    """Keys an entry point names without opening the files they point at."""

    keys: frozenset[str]
    reason: str


#: Per entry point, the keys whose *path* it uses and whose *content* cannot
#: reach its output. Listing one here is a claim about the script, and
#: ``test_path_only_keys_are_still_named`` holds the claim to the script.
PATH_ONLY_KEYS = {
    "scripts/render_config.py": PathOnly(
        frozenset(
            {
                "censored_mutations_file",
                "censored_clades_file",
                "masked_nt_sites_file",
                "reference_sequence_file",
            }
        ),
        "it writes these four paths into the rendered training config as "
        "strings and never opens the files, so the rendered config follows "
        "`config.yaml` and not the tables. Declaring them would also put "
        "mat-pcp's tree upstream of a `train` input, which is the false "
        "rebuild this file exists to prevent",
    ),
}

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
    mat_pcp_keys: frozenset[str] = frozenset()


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


def mat_pcp_keys_in(node: ast.AST) -> set[str]:
    """The mat-pcp keys ``mat_pcp_path("<key>")`` names anywhere under ``node``.

    Both spellings count: ``mat_pcp_path(...)`` where the function is imported
    by name, and ``lib.mat_pcp_path(...)`` where the module is. A key computed
    rather than written out is skipped, which is a real blind spot and is
    recorded in this file's docstring.
    """
    keys: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not child.args:
            continue
        func = child.func
        called = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )
        if called != MAT_PCP_CALL:
            continue
        try:
            key = ast.literal_eval(child.args[0])
        except (ValueError, TypeError):
            continue
        if isinstance(key, str):
            keys.add(key)
    return keys


def _declared_mat_pcp_keys(
    decorator: ast.Call, helpers: dict[str, frozenset[str]]
) -> set[str]:
    """The keys an ``@workflow.input(...)`` decorator declares.

    Written out in the block, or reached by a module-level helper the block
    names. The helper case is not a convenience: ``config.yaml`` gives a
    directory, so a rule that wants its contents has to glob, and the glob is
    the function. Resolving it is what makes ``founders=founder_fastas`` a
    declaration of ``clade_founders_dir`` rather than an opaque callable.
    """
    keys = mat_pcp_keys_in(decorator)
    values = list(decorator.args) + [k.value for k in decorator.keywords]
    for value in values:
        if isinstance(value, ast.Name):
            keys |= helpers.get(value.id, frozenset())
    return keys


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

        # Module-level helpers first, since a rule may name one defined
        # below it. A rule is a function carrying `@workflow.rule(name=...)`;
        # everything else at this level is the file's own code.
        helpers = {
            node.name: frozenset(mat_pcp_keys_in(node))
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not _rule_name(node)
        }

        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            name = _rule_name(node)
            rule = ParsedRule()
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                func = decorator.func
                attr = func.attr if isinstance(func, ast.Attribute) else None
                if attr == "input":
                    rule = replace(
                        rule,
                        inputs={**rule.inputs, **_literal_strings(decorator)},
                        mat_pcp_keys=rule.mat_pcp_keys
                        | frozenset(_declared_mat_pcp_keys(decorator, helpers)),
                    )
                elif attr == "shellcmd":
                    shell = _literal_strings(decorator).get("#0")
                    rule = replace(rule, shellcmd=shell)
                elif attr == "norun":
                    rule = replace(rule, norun=True)
            if name is not None:
                found[name] = replace(rule, run_imports=frozenset(_imports_in(node)))
    return found


def _rule_name(node: ast.FunctionDef) -> str | None:
    """The rule a function defines, or None if it is not a rule at all."""
    for decorator in node.decorator_list:
        func = decorator.func if isinstance(decorator, ast.Call) else None
        if isinstance(func, ast.Attribute) and func.attr == "rule":
            return _literal_strings(decorator).get("name")
    return None


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


def _run_block_functions(rule: ParsedRule, entry_point: str) -> set[str] | None:
    """The functions a ``run:`` block imports from its entry point, or None.

    None for a ``shell:`` rule, which runs the module as a script and so
    reaches its ``main()``. For a ``run:`` rule the distinction is load-bearing:
    ``rule render_era_map`` imports ``build_era_map`` and ``render_era_config``
    from ``scripts/write_era_config.py`` and never calls its ``main()``, which
    is the only place that module reads ``era_fit_summary_file``. Scanning the
    whole module would require the rule to declare a file it cannot read, and a
    declaration like that is a false rebuild trigger -- the defect this file
    exists to prevent, pointing the other way.
    """
    if rule.shellcmd is not None:
        return None
    module = Path(entry_point).stem
    return {
        name.split(".", 1)[1]
        for name in rule.run_imports
        if name.startswith(f"{module}.")
    }


def mat_pcp_keys_read_by(rule_name: str) -> set[str]:
    """The mat-pcp keys the code this rule actually runs names as literals."""
    rule = parsed_rules()[rule_name]
    entry_point = ENTRY_POINTS[rule_name]
    tree = ast.parse((REPO_ROOT / entry_point).read_text())
    wanted = _run_block_functions(rule, entry_point)
    if wanted is None:
        return mat_pcp_keys_in(tree)
    keys: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            keys |= mat_pcp_keys_in(node)
    return keys


@pytest.mark.parametrize("rule", sorted(ENTRY_POINTS))
def test_rule_declares_the_mat_pcp_files_it_reads(rule: str):
    """The same subset check, for the data side.

    Unlike the import check this one can pass vacuously -- most entry points
    name no mat-pcp file -- so what keeps it honest is
    ``test_path_only_keys_are_still_named`` below and the fact that the keys
    are read out of the script rather than listed here.
    """
    entry_point = ENTRY_POINTS[rule]
    read = mat_pcp_keys_read_by(rule)
    exempt = PATH_ONLY_KEYS.get(entry_point)
    if exempt is not None:
        read -= exempt.keys
    missing = sorted(read - parsed_rules()[rule].mat_pcp_keys)
    them = "it" if len(missing) == 1 else "them"
    assert not missing, (
        f"rule {rule} runs {entry_point}, which reads mat-pcp's "
        f"{', '.join(missing)}, but does not declare {them}. `config.yaml` "
        f"names the mat-pcp root and not the contents, so a re-emitted table "
        f"leaves this rule reporting nothing to be done against an output "
        f"built from the old one. Declare {them} with mat_pcp_path(...) in "
        f"the input block, or -- for a directory -- with a helper that globs "
        f"it; if the rule only writes the path down, add it to PATH_ONLY_KEYS "
        f"with the reason."
    )


@pytest.mark.parametrize("entry_point", sorted(PATH_ONLY_KEYS))
def test_path_only_keys_are_still_named(entry_point: str):
    """An exemption describes a call the script still makes.

    Without this, a key that stopped being used, or a script that started
    opening the file instead of writing its path, would leave a standing
    exemption behind -- and the second of those is the failure the exemption
    would then hide.
    """
    named = mat_pcp_keys_in(ast.parse((REPO_ROOT / entry_point).read_text()))
    stale = sorted(PATH_ONLY_KEYS[entry_point].keys - named)
    assert not stale, (
        f"{entry_point} is exempted for {stale} -- "
        f"{PATH_ONLY_KEYS[entry_point].reason} -- but no longer names "
        f"{'it' if len(stale) == 1 else 'them'} at all. Drop the entry."
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


# The retrain hazard. Both halves of it: the edge that scheduled a retrain of
# seven finished runs on 2026-11-13, and the refusal that now stands behind
# wherever else such an edge might appear.


def test_the_test_gate_is_a_sibling_of_the_corpora_not_a_precondition():
    """`tests_passed.done` must not sit upstream of a corpus.

    It did, as `prepare_parquet`'s input, and the chain ran on to training:
    a test edit rewrote the corpus, which rewrote `corpus_guard`, which
    `train` declares un-ancient and *should* retrain on. So a one-line test
    edit scheduled a retrain of seven runs that had finished days earlier,
    and the only thing that stopped it was every GPU being busy.

    The gate itself is kept -- `rule corpora` asks for the marker, so the
    suite runs in the default DAG and an edited test still reruns pytest --
    but as a sibling, exactly as `rule models` treats the smokes.

    Read off the `.smk` source rather than through `parsed_rules`, which
    keeps only plain string literals and so cannot see an input built from
    an f-string. Pinned structurally rather than by dry-running the DAG,
    because the failure is one line anyone can reintroduce and what it
    reaches is days of GPU time.
    """
    marker = "tests_passed.done"
    rule_header = re.compile(r"^rule ([A-Za-z_][A-Za-z0-9_]*):", re.MULTILINE)
    consumers: set[str] = set()
    producers: set[str] = set()
    for smk in sorted((REPO_ROOT / "workflow").glob("*.smk")):
        text = smk.read_text()
        starts = [(m.group(1), m.start()) for m in rule_header.finditer(text)]
        for index, (name, begin) in enumerate(starts):
            finish = starts[index + 1][1] if index + 1 < len(starts) else len(text)
            body = text[begin:finish]
            # Only the declaration blocks, not the prose around them: these
            # files discuss this marker at length, and a docstring naming it
            # is the opposite of a problem.
            for section, found in (("input", consumers), ("output", producers)):
                block = re.search(
                    rf"\n    {section}:\n(.*?)(?=\n    [a-z_]+:\n|\Z)",
                    body,
                    re.DOTALL,
                )
                if block and marker in _without_comments(block.group(1)):
                    found.add(name)

    assert producers == {"validate_tests"}, (
        f"{marker} is written by {sorted(producers)}; it is the test "
        f"suite's own marker and nothing else should produce it."
    )
    assert consumers == {"corpora"}, (
        f"{marker} is an input of {sorted(consumers)}. Only `corpora` may "
        f"ask for it, as the aggregator that keeps the suite in the default "
        f"DAG. A rule that *produces* something training reads puts a test "
        f"edit upstream of a multi-day run."
    )
    assert "corpora" in AGGREGATORS and parsed_rules()["corpora"].norun, (
        "`corpora` runs a command, so asking it for the marker is not the "
        "sibling arrangement this test is checking for."
    )


def _without_comments(block: str) -> str:
    """A declaration block with its `#` comments removed."""
    return "\n".join(line.split("#", 1)[0] for line in block.splitlines())


def test_train_is_handed_the_guard_that_refuses_a_finished_run():
    """`rule train` passes `--refuse-if-complete`, and its shell uses it.

    The parameter is the whole mechanism: `scripts/train.py` cannot protect
    a run whose guard path it is never given, and a `params` entry that no
    shell interpolates is silently inert -- the rule would look guarded and
    train anyway.
    """
    shell = parsed_rules()["train"].shellcmd or ""
    assert "{params.completed}" in shell, (
        f"rule train's shell is {shell!r}, which never passes the "
        f"completion guard to scripts/train.py, so every finished run is "
        f"unprotected."
    )
    source = (REPO_ROOT / "workflow" / "train.smk").read_text()
    assert "completed=completion_guard," in source, (
        "rule train interpolates {params.completed} but declares no "
        "`completed` param, so snakemake would fail the rule rather than "
        "guard it."
    )
    assert "--refuse-if-complete" in source, (
        "`completion_guard` no longer passes --refuse-if-complete, so the "
        "param expands to something scripts/train.py ignores."
    )
