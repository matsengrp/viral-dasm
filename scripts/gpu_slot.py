"""One run per card, claimed by the training process that will use it.

``dasm2.utils.device.pick_device`` treats any *string* preference as explicit
and hands back ``torch.device(str)`` unexamined, so ``--device cuda`` resolves
to whichever card is current -- card 0, absent ``CUDA_VISIBLE_DEVICES``. Seven
training runs scheduled in parallel would pile onto one card. This module is
what picks the card instead, and ``scripts/train.py`` passes the result
through as ``--device cuda:{slot}``.

``CUDA_VISIBLE_DEVICES`` is not the mechanism, and cannot be: ``train.py``
imports ``dasm2.train_command``, and so torch, at module scope, so by the time
any code here could set the variable torch has already enumerated the devices.

**The claim is held by the training process itself** rather than by a wrapper,
so the lock's lifetime is the run's lifetime and release-on-exit is a ``with``
block rather than a signal handler someone has to get right.

The lock loop is ``dasm2-experiments``' ``dasm2/utils/gpu_dispatch.py::
acquire_gpu_slot``: an ``O_CREAT | O_EXCL`` create under a shared directory,
which works on NFS where ``flock`` does not. What is new here is the filter
below.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` is what puts this repository's root on `sys.path`, so the deferred
# `dasm2` import in `candidate_cards` resolves to the vendored fork rather
# than to whichever checkout the venv's editable install points at. Imported
# here rather than there because the insert has to have happened first, and
# because `tests/test_import_order.py` reads the order off this file.
from lib import assert_dasm2_is_this_repo

#: ``nvidia-smi``'s own timeout, in seconds. A hung driver query must not hold
#: up a launch indefinitely; an unanswered query means no card can be shown to
#: be free, which is the conservative answer.
SMI_TIMEOUT_S = 30


def _nvidia_smi(query: str, extra: list[str] | None = None) -> list[str] | None:
    """``nvidia-smi`` CSV rows for one query, or None if it cannot be run."""
    command = [
        "nvidia-smi",
        f"--query-{query}",
        "--format=csv,noheader,nounits",
        *(extra or []),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _process_owner(pid: str) -> str | None:
    """The uid owning ``pid``, or None if it cannot be determined."""
    try:
        return str(Path("/proc", pid).stat().st_uid)
    except (OSError, ValueError):
        return None


def cards_with_another_users_process(
    compute_apps: list[str] | None, gpu_uuids: list[str] | None, my_uid: str
) -> set[int]:
    """Card indices carrying a process this user does not own.

    ``compute_apps`` are ``gpu_uuid, pid`` rows from
    ``nvidia-smi --query-compute-apps``; ``gpu_uuids`` are ``uuid`` rows from
    ``--query-gpu``, in card-index order.

    **A card whose occupant cannot be identified is excluded.** Two ways that
    happens, and both mean the same thing here: a process in another PID
    namespace gives a pid that resolves to nothing under ``/proc``, and a
    process that exited between the two queries has no ``/proc`` entry either.
    The first is somebody else's job and the second is nobody's, but a launch
    is a multi-day commitment and the cost of skipping a free card is one
    queued run.

    A query that could not be run at all excludes every card, for the same
    reason: nothing has been shown to be free.
    """
    if compute_apps is None or gpu_uuids is None:
        return set(range(len(gpu_uuids or [])))
    index_of = {uuid: i for i, uuid in enumerate(gpu_uuids)}
    occupied: set[int] = set()
    for row in compute_apps:
        parts = [field.strip() for field in row.split(",")]
        if len(parts) < 2:
            continue
        uuid, pid = parts[0], parts[1]
        card = index_of.get(uuid)
        if card is None:
            continue
        owner = _process_owner(pid)
        if owner is None or owner != my_uid:
            occupied.add(card)
    return occupied


def _stale(lock_file: Path) -> bool:
    """Whether a lock file names a pid that is no longer running."""
    try:
        pid = int(lock_file.read_text().strip().split()[-1])
    except (OSError, ValueError, IndexError):
        return False
    return not Path("/proc", str(pid)).exists()


def candidate_cards() -> list[int]:
    """Cards this user may take, least used first.

    Order comes from ``dasm2.utils.device.rank_cuda_gpus_by_usage``, which
    ranks on utilization, then memory, then process count. The filter is
    :func:`cards_with_another_users_process`.
    """
    from dasm2.utils.device import rank_cuda_gpus_by_usage

    assert_dasm2_is_this_repo()

    uuids = _nvidia_smi("gpu=uuid")
    if uuids is None:
        return []
    apps = _nvidia_smi("compute-apps=gpu_uuid,pid")
    excluded = cards_with_another_users_process(apps, uuids, str(os.getuid()))
    allowed = [i for i in range(len(uuids)) if i not in excluded]
    ranked = rank_cuda_gpus_by_usage(allowed=allowed)
    return ranked if ranked else allowed


@contextmanager
def claim_card(lock_dir: Path, job_name: str) -> Iterator[int]:
    """Hold one card for the duration of the block, and yield its index.

    Args:
        lock_dir: Shared directory holding one lock file per card. Created if
            missing; it must be visible to every host that launches runs.
        job_name: Written into the lock file beside this process's pid, so a
            stuck lock names what is holding it.

    Yields:
        The card index, to be passed on as ``cuda:{index}``.

    Raises:
        RuntimeError: If no card is both free of another user's work and
            unlocked.
    """
    lock_dir.mkdir(parents=True, exist_ok=True)
    cards = candidate_cards()
    if not cards:
        raise RuntimeError(
            f"no GPU is available for {job_name}. Either nvidia-smi could not "
            f"be queried, or every card carries a process this user does not "
            f"own. A card whose occupant cannot be identified is treated as "
            f"taken, so this also fires when the driver reports a process "
            f"whose pid resolves to nothing."
        )

    held: Path | None = None
    card: int | None = None
    for candidate in cards:
        lock_file = lock_dir / f"gpu_{candidate}.lock"
        if lock_file.exists() and _stale(lock_file):
            lock_file.unlink(missing_ok=True)
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, f"{job_name} {os.getpid()}".encode())
        os.close(fd)
        held, card = lock_file, candidate
        break

    if held is None or card is None:
        raise RuntimeError(
            f"every candidate card {cards} is already locked under {lock_dir} "
            f"by another run of this pipeline. Snakemake's `--resources gpu=` "
            f"should be at most the number of cards this host has free."
        )

    try:
        yield card
    finally:
        held.unlink(missing_ok=True)
