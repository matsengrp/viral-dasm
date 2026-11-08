"""Two orderings every pipeline script's import block has to get right.

**`lib` before `dasm2`.** `scripts/lib.py` is what puts the repository root
on `sys.path`. Until it has, `import dasm2` is served by the venv's editable
install, which points at whichever checkout it was installed from -- a script
run puts `scripts/` on `sys.path[0]`, never the repository root. Three
scripts once imported `dasm2` above `lib` for the Arrow reason below, and one
of them duly ran against the other checkout. `lib` imports only the standard
library and `yaml`, so it cannot itself lose the Arrow race.

**`dasm2` before pyarrow.**

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


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_lib_is_imported_before_dasm2(path: Path):
    """`lib` puts the repository root on `sys.path`; `dasm2` needs it there.

    `lib.py` itself is exempt: it *is* the fix, and it imports `dasm2` only
    inside `assert_dasm2_is_this_repo`, after the insert at module level.
    """
    if path.name == "lib.py":
        pytest.skip("lib.py performs the insert")
    lines = first_import_lines(path)
    if "dasm2" not in lines:
        pytest.skip(f"{path.name} does not import dasm2")
    assert "lib" in lines, (
        f"{path.name} imports dasm2 at line {lines['dasm2']} and never imports "
        f"lib, so nothing has put the repository root on sys.path. The "
        f"editable install resolves dasm2 to the checkout it was installed "
        f"from, which need not be this one."
    )
    assert lines["lib"] < lines["dasm2"], (
        f"{path.name} imports dasm2 at line {lines['dasm2']}, above lib at "
        f"line {lines['lib']}. lib is what inserts the repository root into "
        f"sys.path, so dasm2 resolves through the editable install instead "
        f"and this script measures another checkout's code. Move the lib "
        f"import above it -- lib pulls in no Arrow, so the ordering below "
        f"still holds."
    )
