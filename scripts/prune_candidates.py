"""Which vendored ``dasm2`` modules never run, and who still imports them.

Reads the JSON that ``scripts/trace_executed_modules.py`` writes for each
phase and sorts every file under ``dasm2/`` into three buckets.

**The classification rule, which is the whole measurement.** A file "ran" if
the code object of a named ``def`` in it was entered. Class bodies, module
bodies, lambdas and comprehensions do not count, because every imported
module executes those and counting them would mark the entire closure live.
Three consequences follow, and each has already produced a wrong answer once:

* A module whose only contribution is a constant, a ``Protocol`` or a
  dataclass has no named ``def`` to run. It is reported separately and is
  never a candidate.
* A class registered in a factory executes its body at import and its methods
  only when something builds it. ``heads/per_site_parent_child.py`` reads as a
  candidate for that reason while ``ci-484k`` needs it.
* A function that runs once at import to build a module-level constant marks
  its file as live even when nothing calls it afterwards.
  ``data/functional/codon_neighbor.py`` reads as live for that reason while
  the only chain reaching it is itself being deleted.

So the output is a candidate list, not a delete list. The importer map is the
part to act on: a candidate no live module names can go on a static argument,
and a candidate a live module names has to be read first.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRACKED_LIST = REPO / "data/prune/delete_candidates.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "traces", nargs="+", type=Path, help="JSON from trace_executed_modules.py"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"refresh {TRACKED_LIST.relative_to(REPO)}",
    )
    args = parser.parse_args()

    executed: set[tuple[str, str]] = set()
    for path in args.traces:
        for entry in json.loads(path.read_text()):
            file, _, name = entry.partition("::")
            executed.add((file, name))

    ran_names: dict[str, set[str]] = defaultdict(set)
    for file, name in executed:
        ran_names[file].add(name)

    files = sorted(str(p.relative_to(REPO)) for p in (REPO / "dasm2").rglob("*.py"))
    lines = {f: len((REPO / f).read_text().splitlines()) for f in files}

    live, candidates, no_functions = [], [], []
    for f in files:
        defined = _function_names(REPO / f)
        if not defined:
            no_functions.append(f)
        elif ran_names[f] & defined:
            live.append(f)
        else:
            candidates.append(f)

    def block(label: str, group: list[str]) -> str:
        return f"  {label:<40}{len(group):>4} files {sum(lines[f] for f in group):>8,} lines"

    print(f"{len(files)} files, {sum(lines.values()):,} lines")
    print(block("a function body ran", live))
    print(block("no function body ever ran", candidates))
    print(block("no function bodies to run", no_functions))

    importers = _import_graph(files, lines)
    candidate_set = set(candidates)
    live_importers = {
        f: sorted(
            i
            for i in importers[f]
            if i not in candidate_set and not i.endswith("__init__.py")
        )
        for f in candidates
    }
    free = [f for f in candidates if not live_importers[f]]
    held = [f for f in candidates if live_importers[f]]

    print()
    print(block("  of those, no live module imports it", free))
    print(block("  of those, a live module imports it", held))

    print("\ncandidates by directory")
    by_dir = defaultdict(list)
    for f in candidates:
        rel = f[len("dasm2/") :]
        by_dir[rel.rsplit("/", 1)[0] if "/" in rel else "(root)"].append(f)
    for d in sorted(by_dir, key=lambda k: -sum(lines[f] for f in by_dir[k])):
        group = by_dir[d]
        print(
            f"  {d:<18}{len(group):>4} files {sum(lines[f] for f in group):>8,} lines"
        )

    print("\ncandidates a live module imports, and which module")
    for f in sorted(held, key=lambda f: -lines[f]):
        print(f"  {lines[f]:>5}  {f}")
        for i in live_importers[f]:
            print(f"           <- {i}")

    if args.write:
        TRACKED_LIST.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            "# lines  path  <- live modules that import it (blank: none)",
            f"# {len(candidates)} files, {sum(lines[f] for f in candidates):,} lines",
            "# regenerate: python scripts/prune_candidates.py <traces> --write",
            "",
        ]
        rows += [
            f"{lines[f]:>6}  {f}  {' '.join(live_importers[f])}".rstrip()
            for f in candidates
        ]
        TRACKED_LIST.write_text("\n".join(rows) + "\n")
        print(f"\nwrote {TRACKED_LIST.relative_to(REPO)}")


def _function_names(path: Path) -> set[str]:
    """Named ``def``s in the file, excluding bodies that are only a stub.

    A ``Protocol`` method whose body is ``...`` can never be entered, so a
    file holding nothing else belongs in the "no function bodies to run"
    bucket rather than reading as an unreached candidate.
    """
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            body = body[1:]  # drop the docstring
        if len(body) == 1 and isinstance(body[0], (ast.Pass, ast.Raise)):
            continue
        if (
            len(body) == 1
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and body[0].value.value is Ellipsis
        ):
            continue
        names.add(node.name)
    return names


def _import_graph(files: list[str], lines: dict[str, int]) -> dict[str, set[str]]:
    """For each file, the set of files that import it.

    Static, and relative imports are resolved against the importing module's
    package, so ``from .foo import bar`` inside ``dasm2/data/`` resolves to
    ``dasm2/data/foo.py``. ``from dasm2.x import y`` records an edge to
    ``dasm2/x/y.py`` when that file exists and to ``dasm2/x.py`` otherwise,
    since the name may be either a submodule or a symbol.
    """
    by_module = {}
    for f in files:
        module = f[:-3].replace("/", ".")
        by_module[module.removesuffix(".__init__")] = f

    importers: dict[str, set[str]] = defaultdict(set)
    for f in files:
        me = f[:-3].replace("/", ".").removesuffix(".__init__")
        tree = ast.parse((REPO / f).read_text())
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = me.split(".")
                    keep = (
                        len(parts)
                        - node.level
                        + (1 if f.endswith("__init__.py") else 0)
                    )
                    package = ".".join(parts[:keep])
                    base = f"{package}.{base}" if base else package
                targets = [base, *(f"{base}.{a.name}" for a in node.names)]
            for target in targets:
                hit = by_module.get(target)
                if hit is not None and hit != f:
                    importers[hit].add(f)
    return importers


if __name__ == "__main__":
    main()
