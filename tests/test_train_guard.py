"""The refusal that stands between a stale DAG and a finished run.

`scripts/train.py` resumes from its latest checkpoint rather than starting
over, so a run scheduled by mistake does not no-op -- it continues a model
the paper has already reported and leaves no copy of what it reported. The
seven campaign runs cost days of GPU time across two torch versions and
cannot be reproduced exactly, so the asymmetry is total: the check costs a
stat call and the mistake costs the result.

This file is about the refusal alone. That the DAG hands it the right path
is `tests/test_rule_inputs.py`, and that a test edit no longer schedules
training at all is the same file's sibling-gate check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import lib  # noqa: F401
from train import refuse_if_complete

import dasm2  # noqa: F401


def _guard(tmp_path: Path, name: str, **fields) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(fields))
    return path


def test_a_finished_run_is_refused(tmp_path):
    """The case the whole file exists for."""
    guard = _guard(
        tmp_path,
        "dasm-85k_cap_5000_0",
        passed=True,
        steps_to_stop=240000,
        stop_reason="LR 9.77e-07 < 1.00e-06",
    )
    with pytest.raises(SystemExit) as caught:
        refuse_if_complete(guard)
    message = str(caught.value)
    # The message has to carry the way out, or the next person deletes the
    # guard file to get past it -- which is the one response that destroys
    # the evidence that the run finished.
    assert "retrain=dasm-85k_cap_5000_0" in message
    assert "240000" in message


def test_a_run_with_no_guard_trains(tmp_path):
    """What a new model, a new corpus or a rule under test looks like.

    The protection is scoped to runs that have finished, not to training as
    such: a guard file that does not exist is not evidence of anything, and
    refusing on it would make the next campaign unrunnable.
    """
    refuse_if_complete(tmp_path / "nothing-here.json")
    refuse_if_complete(None)


def test_a_run_that_did_not_pass_trains(tmp_path):
    """A failed run is exactly the one worth retrying."""
    refuse_if_complete(_guard(tmp_path, "run", passed=False, failures=["lr floor"]))


def test_a_guard_that_cannot_be_read_refuses_rather_than_trains(tmp_path):
    """An unreadable guard is not permission.

    The file is the only evidence that the run finished, so losing track of
    it means stopping and looking rather than overwriting the thing it was
    describing. A `try` that swallowed this would turn a corrupted byte into
    a silent retrain -- the failure mode this module exists to prevent,
    reached by the code meant to prevent it.
    """
    corrupt = tmp_path / "run.json"
    corrupt.write_text("{not json")
    with pytest.raises(SystemExit, match="cannot read the run guard"):
        refuse_if_complete(corrupt)


def test_the_campaigns_own_guards_would_all_be_refused():
    """Against the real files, not a fixture.

    A guard whose schema drifted -- `passed` renamed, or nested a level
    down -- would leave every fixture above green and every real run
    unprotected.
    """
    import panel_spec  # noqa: F401  (for ARTIFACT_ROOT's import side effects)
    from lib import ARTIFACT_ROOT
    from runs import RUNS, run_id

    directory = Path(ARTIFACT_ROOT) / "run_guard"
    guards = [directory / f"{run_id(m, c)}.json" for m, c in RUNS]
    present = [g for g in guards if g.is_file()]
    if not present:
        pytest.skip(f"no run guards under {directory}")
    for guard in present:
        if not json.loads(guard.read_text()).get("passed"):
            continue
        with pytest.raises(SystemExit, match="has already trained"):
            refuse_if_complete(guard)


def test_main_actually_calls_the_refusal(tmp_path):
    """The guard is wired into `main`, not merely defined beside it.

    A `refuse_if_complete` that nothing calls passes every test above: the
    function works, the rule passes the path, and training proceeds anyway.
    So this one goes through the real entry point, argparse included, and
    asserts the process refuses.

    A subprocess rather than a call, because `main` is the thing under test
    and it ends in `train_model`. The refusal fires before the config is
    read or a card is claimed, so nothing here needs a GPU or a corpus --
    which is itself part of the contract: a guard that only fired after the
    data loaded would be a guard that costs a run's startup to say no.
    """
    import subprocess

    guard = _guard(tmp_path, "already-done", passed=True, steps_to_stop=1234)
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            # Paths that do not exist: reaching them would mean the refusal
            # came too late, and the error would name them instead.
            "--config",
            str(tmp_path / "no-config.yaml"),
            "--parquet",
            str(tmp_path / "no-corpus.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
            "--refuse-if-complete",
            str(guard),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        # A non-zero exit is the result under test, not an error.
        check=False,
    )
    assert result.returncode != 0, (
        f"train.py exited 0 on a finished run. stdout:\n{result.stdout}"
    )
    assert "has already trained" in result.stderr, (
        f"train.py failed for some other reason:\n{result.stderr[-2000:]}"
    )
