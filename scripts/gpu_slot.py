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

**A card is identified by its UUID, never by an index.** ``nvidia-smi``
numbers every physical card on the host; torch numbers only the cards
``CUDA_VISIBLE_DEVICES`` leaves visible, from zero. The two agree only when
that variable is unset, and the disagreement is not safely loud: with
``CUDA_VISIBLE_DEVICES=2,3``, ``cuda:0`` is physical card 2, so a lock naming
card 0 would guard one card while the run used another -- and could land on
another user's. So the lock is keyed on the physical index, which every
process on the host agrees about, while what is yielded is the torch ordinal
for that same UUID. A card torch cannot address is not a candidate.

**The claim is held by the training process itself** rather than by a wrapper,
so the lock's lifetime is the run's lifetime and release-on-exit is a ``with``
block rather than a signal handler someone has to get right.

The lock itself is an ``O_CREAT | O_EXCL`` create under a shared directory,
which works on NFS where ``flock`` does not.

**The lock is host-local.** :func:`_stale` resolves a pid against this
host's ``/proc``, so a live lock written by another host would read as
stale, and the file name carries no hostname. One host at a time.
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
    """Whether a lock file names a pid that is no longer running.

    An unreadable or unparsable lock counts as *live*, not stale. Treating a
    truncated file as free would let a second run take a card a first one is
    using; the cost of the safe direction is a lock someone clears by hand.
    A recycled pid also reads as live, which is the same safe direction.
    """
    try:
        pid = int(lock_file.read_text().strip().split()[-1])
    except (OSError, ValueError, IndexError):
        return False
    return not Path("/proc", str(pid)).exists()


def torch_ordinals_by_uuid() -> dict[str, int]:
    """Map each card's UUID to the ordinal torch addresses it by.

    ``nvidia-smi`` reports ``GPU-<uuid>``; torch reports the bare UUID. Both
    are normalised to lowercase hex with no prefix. Cards outside
    ``CUDA_VISIBLE_DEVICES`` are simply absent, which is what makes this the
    filter for "can this process actually use that card".
    """
    import torch

    if not torch.cuda.is_available():
        return {}
    ordinals: dict[str, int] = {}
    for ordinal in range(torch.cuda.device_count()):
        uuid = getattr(torch.cuda.get_device_properties(ordinal), "uuid", None)
        if uuid is not None:
            ordinals[_normalise_uuid(str(uuid))] = ordinal
    return ordinals


def _normalise_uuid(uuid: str) -> str:
    """``GPU-47648ED2-...`` and ``47648ed2-...`` to one spelling."""
    return uuid.strip().removeprefix("GPU-").removeprefix("gpu-").lower()


def candidate_cards() -> tuple[list[int], list[str]]:
    """Cards this user may take, least used first, and the host's uuid list.

    Order comes from ``dasm2.utils.device.rank_cuda_gpus_by_usage``, which
    ranks on utilization, then memory, then process count. The filter is
    :func:`cards_with_another_users_process`, plus "torch can address it":
    a card ``CUDA_VISIBLE_DEVICES`` hides has no ordinal to hand back.

    Returns both the candidate indices and the uuid list they index into, so
    a caller never re-queries ``nvidia-smi`` to turn an index back into a
    uuid. Two queries can disagree -- the driver can fail on the second, or a
    card can change state between them -- and an index resolved against the
    wrong list is the one error this module exists to prevent.
    """
    from dasm2.utils.device import rank_cuda_gpus_by_usage

    assert_dasm2_is_this_repo()

    uuids = _nvidia_smi("gpu=uuid")
    if uuids is None:
        return [], []
    apps = _nvidia_smi("compute-apps=gpu_uuid,pid")
    excluded = cards_with_another_users_process(apps, uuids, str(os.getuid()))
    visible = torch_ordinals_by_uuid()
    allowed = [
        i
        for i in range(len(uuids))
        if i not in excluded and _normalise_uuid(uuids[i]) in visible
    ]
    # `rank_cuda_gpus_by_usage` takes `allowed`, but it is re-filtered rather
    # than trusted: an unranked order would otherwise reach the caller as if
    # it had been ranked, and ranking is what keeps two launches off one card
    # when neither holds a lock yet.
    ranked = [i for i in rank_cuda_gpus_by_usage(allowed=allowed) if i in allowed]
    return (ranked or allowed), uuids


def _claim_one(lock_file: Path, job_name: str) -> bool:
    """Take one card's lock, clearing a dead holder's, without a race.

    The obvious form -- ``if exists() and _stale(): unlink()`` then an
    ``O_EXCL`` create -- is not atomic. Two processes can both judge the same
    lock stale; the first unlinks and creates, the second then unlinks *that
    live lock* and creates its own, and both train on one card. Here the
    unlink is conditional on the file still being the one that was judged, by
    comparing ``st_ino`` before and after: a lock replaced in between is a
    different inode, so the loser leaves it alone and moves to another card.
    """
    try:
        before = lock_file.stat().st_ino
    except FileNotFoundError:
        before = None
    if before is not None:
        if not _stale(lock_file):
            return False
        try:
            # Only unlink if nothing replaced it since the staleness verdict.
            if lock_file.stat().st_ino == before:
                lock_file.unlink()
        except (FileNotFoundError, OSError):
            return False
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    try:
        os.write(fd, f"{job_name} {os.getpid()}".encode())
    finally:
        os.close(fd)
    return True


@contextmanager
def claim_card(lock_dir: Path, job_name: str) -> Iterator[int]:
    """Hold one card for the duration of the block, and yield its ordinal.

    The lock is named for the card's **physical** index, which every process
    on the host agrees about. What is yielded is the **torch** ordinal for
    that same card, which is what ``cuda:{n}`` means inside this process. See
    this module's docstring for why those are not the same number.

    Args:
        lock_dir: Shared directory holding one lock file per card. Created if
            missing; it must be visible to every host that launches runs.
        job_name: Written into the lock file beside this process's pid, so a
            stuck lock names what is holding it.

    Yields:
        The torch device ordinal, to be passed on as ``cuda:{ordinal}``.

    Raises:
        RuntimeError: If no card is both free of another user's work and
            unlocked.
    """
    lock_dir.mkdir(parents=True, exist_ok=True)
    cards, uuids = candidate_cards()
    if not cards:
        raise RuntimeError(
            f"no GPU is available for {job_name}. Either nvidia-smi could not "
            f"be queried, every card carries a process this user does not "
            f"own, or CUDA_VISIBLE_DEVICES hides the cards that are free. A "
            f"card whose occupant cannot be identified is treated as taken, "
            f"so this also fires when the driver reports a process whose pid "
            f"resolves to nothing."
        )

    ordinals = torch_ordinals_by_uuid()

    held: Path | None = None
    card: int | None = None
    for candidate in cards:
        lock_file = lock_dir / f"gpu_{candidate}.lock"
        if _claim_one(lock_file, job_name):
            held, card = lock_file, candidate
            break

    if held is None:
        raise RuntimeError(
            f"every candidate card {cards} is already locked under {lock_dir} "
            f"by another run of this pipeline. Snakemake's `--resources gpu=` "
            f"should be at most the number of cards this host has free."
        )

    try:
        uuid = _normalise_uuid(uuids[card])
        if uuid not in ordinals:
            raise RuntimeError(
                f"card {card} ({uuid}) was a candidate but torch cannot "
                f"address it. CUDA_VISIBLE_DEVICES changed between the two "
                f"queries, which makes every index here untrustworthy."
            )
        yield ordinals[uuid]
    finally:
        held.unlink(missing_ok=True)
