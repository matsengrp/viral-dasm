"""This repository's `dasm2` is its own fork, and stays that way.

The vendored package keeps the name `dasm2` so every module reads exactly as
it does upstream and a diff against the source commit stays meaningful. That
makes "no upstream dasm2" impossible to enforce by making the import fail, so
it is enforced the other way round: the import has to land *here*, and nothing
in the dependency list may drag a different one in.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dasm2_resolves_inside_this_repository():
    """`import dasm2` must be the fork, not a package installed alongside it."""
    import dasm2

    resolved = Path(dasm2.__file__).resolve()
    assert resolved.is_relative_to(REPO_ROOT), (
        f"`import dasm2` resolved to {resolved}, outside {REPO_ROOT}. Something "
        f"installed in this environment is shadowing the vendored fork, so the "
        f"gates would be measuring code this repository does not contain."
    )


def test_no_upstream_dasm2_dependency():
    """Nothing in pyproject.toml may install a `dasm2`."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    declared = list(pyproject["project"]["dependencies"])
    for group in pyproject.get("dependency-groups", {}).values():
        declared += [entry for entry in group if isinstance(entry, str)]
    offenders = [d for d in declared if d.split("[")[0].strip().startswith("dasm2")]
    assert not offenders, (
        f"{offenders} would install a dasm2 alongside the fork. The paper's "
        f"code should keep reproducing the paper's numbers, which a dependency "
        f"that moves does not."
    )


def test_provenance_records_the_source_commit():
    """The fork says where it came from, in a full 40-character SHA."""
    import re

    text = (REPO_ROOT / "dasm2" / "PROVENANCE.md").read_text()
    assert re.search(r"\b[0-9a-f]{40}\b", text), (
        "dasm2/PROVENANCE.md names no 40-character commit SHA. Without one "
        "there is no way to diff this tree against what it was taken from."
    )


def test_import_time_side_effects_are_carried():
    """The two things a restructured entry point drops silently.

    `dasm2/__init__.py` calls `use_system_arrow_pool()` and
    `disable_mha_fastpath()` at package import. Both are inert modules unless
    something calls them, and both are called exactly once, as an import side
    effect, so a vendored tree that moves its entry point loses them with no
    error anywhere.

    `disable_mha_fastpath` is the one that bites here. The fused MHA path
    engages only in eval mode under `no_grad`, which is exactly in-training
    validation. At `val_batch_size: 256`, nhead 4 and L 1,273 its attention
    tensor is 256 x 4 x 1273^2 = 1.66e9 floats, 6.6 GB per layer -- under the
    2^31-1 int32 cutoff, so it does not crash. It just gets slow and fat, and
    nothing in the loss curve says so.
    """
    import torch

    import dasm2  # noqa: F401  (imported for its side effects)

    assert torch.backends.mha.get_fastpath_enabled() is False, (
        "the fused MHA fastpath is still enabled after importing dasm2, so "
        "`disable_mha_fastpath()` did not run."
    )


def test_arrow_pool_side_effect_precedes_pyarrow():
    """Arrow fixes its C++ default pool at import, so the order matters."""
    from dasm2 import arrow_pool

    assert hasattr(arrow_pool, "use_system_arrow_pool"), (
        "dasm2.arrow_pool lost `use_system_arrow_pool`; dasm2/__init__.py calls "
        "it before anything imports pyarrow, because Arrow fixes its C++ "
        "default pool at import time."
    )
    source = (REPO_ROOT / "dasm2" / "__init__.py").read_text()
    pool_at = source.index("use_system_arrow_pool()")
    fastpath_at = source.index("disable_mha_fastpath()")
    assert pool_at < fastpath_at, (
        "use_system_arrow_pool() must be called before anything else in "
        "dasm2/__init__.py that could pull pyarrow in transitively."
    )


@pytest.mark.parametrize(
    "module",
    [
        "dasm2.train_command",
        "dasm2.config",
        "dasm2.training.trainer",
        "dasm2.serialization.checkpoints",
        "dasm2.data.evolutionary_dataset",
        "dasm2.data.parquet_backend",
        "dasm2.models.core.transformer",
        "dasm2.heads.selection_factor",
        "dasm2.losses.whichmut_loss",
        "dasm2.losses.whichmut_losses",
        "dasm2.losses.factory",
        "dasm2.training.log_reader",
    ],
)
def test_training_path_imports(module):
    """Every module M1's scope names is present and importable.

    A prune that removes one of these fails here rather than several minutes
    into a training run.

    ``dasm2.training.train_argv`` was on this list and is not any more. It
    built argv for ``dasm2/cli.py``, which this repository never vendored, so
    #2's own scope table recorded it as carried-but-unused. The prune (#4)
    removed it, which is what that note anticipated.
    """
    __import__(module)
    assert module in sys.modules


def test_a_script_run_resolves_dasm2_inside_this_repository():
    """`python scripts/<x>.py` must import the fork, not an installed copy.

    `test_dasm2_resolves_inside_this_repository` above passes under pytest
    because pytest puts the rootdir on `sys.path`. A script run does not: it
    puts the *script's* directory there, so the repository root is absent and
    `_EditableFinder` -- which sits after `PathFinder` in `sys.meta_path` --
    resolves `dasm2` to whatever checkout the venv's editable install points
    at. When this repository is a `git worktree` of that checkout, the gates
    then exercise the other tree's code and pass against it.

    `scripts/lib.py` puts the repository root on `sys.path` to prevent that.
    This is the test that would have caught it: it reproduces a script run
    exactly, from a working directory that is not the repository.
    """
    probe = (
        "import sys;"
        f" sys.path.insert(0, {str(REPO_ROOT / 'scripts')!r});"
        " import lib;"
        " import dasm2;"
        " print(dasm2.__file__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tempfile.gettempdir(),
        capture_output=True,
        text=True,
        check=True,
    )
    resolved = Path(result.stdout.strip()).resolve()
    assert resolved.is_relative_to(REPO_ROOT), (
        f"a script run resolved `dasm2` to {resolved}, outside {REPO_ROOT}"
    )


def test_no_dasm2_submodule_resolves_outside_this_repository():
    """A module deleted here must not be served from another checkout.

    The editable finder maps the *name* `dasm2` to the directory it was
    installed from and resolves submodules against that, not against the
    parent package's `__path__`. So deleting `dasm2/foo.py` here does not
    make `import dasm2.foo` fail while that path still has one: `PathFinder`
    finds nothing and the editable finder serves the other copy. The prune
    deleted `positional_encoding.py` and the gates kept passing on the other
    checkout's copy of it.
    """
    probe = (
        "import sys;"
        f" sys.path.insert(0, {str(REPO_ROOT / 'scripts')!r});"
        " import lib;"
        " import dasm2.train_command;"
        " lib.assert_dasm2_is_this_repo();"
        " print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tempfile.gettempdir(),
        capture_output=True,
        text=True,
        check=False,  # the assertion below reports the guard's own message
    )
    assert result.returncode == 0, result.stderr.strip().splitlines()[-1:]
