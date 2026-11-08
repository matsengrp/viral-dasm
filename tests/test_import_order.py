"""No pipeline script may import pyarrow before `dasm2`.

Arrow fixes its C++ default memory pool the first time it is imported, and
`dasm2/__init__.py` calls `use_system_arrow_pool()` at package import to steer
it off mimalloc -- which, under pyarrow 25, requests transparent huge pages.
Whichever import lands first wins.

`pandas` pulls pyarrow in transitively, so a script that does `import pandas`
above `import dasm2` silently loses the setting. `dasm2` does print a warning,
but it goes to a Snakemake rule's log file where nobody reads it.

This is one of the two import-time side effects #2 calls "the easiest thing
here to lose". `tests/test_fork.py` checks the side effects still exist; this
checks the scripts actually benefit from them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((REPO_ROOT / "scripts").glob("*.py"))

#: Modules that pull pyarrow in, directly or transitively.
ARROW_IMPORTERS = frozenset({"pyarrow", "pandas"})


def first_import_lines(path: Path) -> dict[str, int]:
    """The line number of the first import of each top-level module."""
    seen: dict[str, int] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module] if node.module and not node.level else []
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            seen.setdefault(root, node.lineno)
    return seen


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_dasm2_is_imported_before_pyarrow(path: Path):
    lines = first_import_lines(path)
    arrow = {m: lines[m] for m in ARROW_IMPORTERS if m in lines}
    if not arrow:
        pytest.skip(f"{path.name} imports neither pyarrow nor pandas")
    if "dasm2" not in lines:
        pytest.skip(f"{path.name} does not import dasm2")
    earliest_arrow = min(arrow.values())
    assert lines["dasm2"] < earliest_arrow, (
        f"{path.name} imports {min(arrow, key=arrow.get)} at line "
        f"{earliest_arrow}, before dasm2 at line {lines['dasm2']}. Arrow fixes "
        f"its default memory pool at first import, so dasm2's "
        f"use_system_arrow_pool() has already lost by then. Move the dasm2 "
        f"import above it."
    )
