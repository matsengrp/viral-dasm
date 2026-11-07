"""This repository's `dasm2` is its own fork, and stays that way.

The vendored package keeps the name `dasm2` so every module reads exactly as
it does upstream and a diff against the source commit stays meaningful. That
makes "no upstream dasm2" impossible to enforce by making the import fail, so
it is enforced the other way round: the import has to land *here*, and nothing
in the dependency list may drag a different one in.
"""

from __future__ import annotations

import sys
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
