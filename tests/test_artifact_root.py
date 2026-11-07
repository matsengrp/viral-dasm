"""`VIRAL_DASM_ARTIFACT_ROOT` redirects where the pipeline writes.

The override's whole purpose is to keep a second checkout from writing into a
live run's output directory. A silent failure — the variable set but ignored,
or honoured but pointing somewhere unexpected — produces exactly the collision
it exists to prevent, days into a run, so it is asserted rather than trusted.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Read ``ARTIFACT_ROOT`` in a fresh interpreter. ``scripts/lib.py`` resolves
#: it once at import, so a test that imported it could not then see the
#: override take effect.
PROBE = (
    "import sys; sys.path.insert(0, 'scripts');"
    " from lib import ARTIFACT_ROOT; print(ARTIFACT_ROOT)"
)


def _artifact_root(env_value: str | None) -> str:
    env = dict(os.environ)
    env.pop("VIRAL_DASM_ARTIFACT_ROOT", None)
    if env_value is not None:
        env["VIRAL_DASM_ARTIFACT_ROOT"] = env_value
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_default_comes_from_config():
    import yaml

    with (REPO_ROOT / "config.yaml").open() as handle:
        configured = yaml.safe_load(handle)["artifact_root"]
    assert _artifact_root(None) == configured


def test_override_wins():
    assert _artifact_root("/tmp/viral-dasm-test-root") == "/tmp/viral-dasm-test-root"


def test_override_is_not_a_config_key():
    """The override must not be settable in `config.yaml`.

    A committed value would follow the branch into whatever checkout opened
    it next, which is the opposite of what a per-checkout override is for.
    """
    import yaml

    with (REPO_ROOT / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    assert "viral_dasm_artifact_root" not in config
