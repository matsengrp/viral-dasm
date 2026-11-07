"""Forensic diagnostics knobs for the trainer, resolved from the environment.

These are opt-in *debugging* toggles surfaced while diagnosing the issue-#724
cycle-1 crash. Unlike the run-shaping DataLoader knobs (``dataloader_prefetch_
factor`` / ``dataloader_pin_memory``, now ``TrainingConfig`` fields), these are
throwaway forensics, so they stay environment-driven rather than baked into the
serialized training config. Collecting them here gives a single place to see
"what environment variables affect a training run's forensics":

- ``DASM2_STEP_WATCHDOG_SECONDS`` — per-step stall-watchdog timeout (float).
- ``DASM2_LOG_LONG_BATCH_CODONS`` — padded-seq-len threshold for long-batch logs.
- ``DASM2_CAPTURE_WINDOW_DETAIL`` — valid-codon threshold for per-row capture.
- ``DASM2_CAPTURE_WINDOW_STEPS`` — ``<cycle>:<lo>-<hi>`` gate for that capture.

Related knob resolved in its own subsystem (not here, since it has no path to the
trainer): ``DASM2_PARQUET_CACHE_BUDGET_GB`` — row-group cache RAM cap, see
:mod:`dasm2.data.parquet_backend`.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple


def _step_watchdog_seconds() -> Optional[float]:
    """Per-step stall-watchdog timeout from ``DASM2_STEP_WATCHDOG_SECONDS``.

    When set to a positive number, the training loop arms
    ``faulthandler.dump_traceback_later`` before each optimizer step and cancels
    it after. A step that runs longer than the timeout dumps every thread's stack
    to stderr (and so into the training log) without exiting, turning a silent
    multi-minute stall — a stuck collective, a wedged data load, the issue-#724
    hang — into a traceback that pins the stuck operation, instead of an opaque
    NCCL watchdog timeout. Returns ``None`` (disabled) when unset or invalid.
    """
    raw = os.environ.get("DASM2_STEP_WATCHDOG_SECONDS")
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def _long_batch_log_codons() -> Optional[int]:
    """Padded-seq-len threshold (codons) for per-step long-batch forensics.

    Set ``DASM2_LOG_LONG_BATCH_CODONS=<int>`` to print, on EVERY rank and BEFORE
    the forward, a flushed line for any accumulation window whose padded
    sequence length reaches the threshold, with that rank's CUDA memory. This
    catches the heavy long-sequence batches that crash one rank's GPU under DDP
    load imbalance — the line is flushed pre-compute, so it survives
    even a SIGSEGV in the step it describes. Unset/non-positive ⇒ disabled.
    """
    raw = os.environ.get("DASM2_LOG_LONG_BATCH_CODONS")
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _capture_window_detail_codons() -> Optional[int]:
    """Valid-codon threshold for per-sequence window-detail forensics.

    Set ``DASM2_CAPTURE_WINDOW_DETAIL=<int>`` to dump, on EVERY rank and BEFORE
    the forward, one flushed ``[window-row]`` line per sequence whose true
    (non-pad) length reaches the threshold OR whose parent ambiguous-codon ("N")
    fraction is high — fingerprinting the long / low-information / high-N PCPs
    that drive the cycle-1 slow-batch crash. Flushed pre-compute, the lines
    survive a crash in the window they describe and can be matched back to
    parquet rows offline. Unset/non-positive ⇒ disabled.
    """
    raw = os.environ.get("DASM2_CAPTURE_WINDOW_DETAIL")
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _capture_window_steps() -> Optional[Tuple[int, int, int]]:
    """Optional ``(cycle, step_lo, step_hi)`` gate for window-detail capture.

    Set ``DASM2_CAPTURE_WINDOW_STEPS=<cycle>:<lo>-<hi>`` to confine the per-row
    ``[window-row]`` dump to a single 0-indexed ``cycle`` and an inclusive,
    1-indexed step range (matching the ``step=`` field printed in the
    forensics). Without it the dump fires for every qualifying window in every
    cycle — ~1M flushed lines on a full Compara run, which both bloats the log
    and perturbs the timing it is meant to measure. Confining it to the crash
    window (e.g. ``1:1700-1850`` for the issue-#724 cycle-1 ~step-1792 crash)
    keeps the I/O overhead out of the steps under study. Unset/malformed ⇒ no
    gate (capture fires for every qualifying window).
    """
    raw = os.environ.get("DASM2_CAPTURE_WINDOW_STEPS")
    if not raw:
        return None
    try:
        cycle_str, span = raw.split(":", 1)
        lo_str, hi_str = span.split("-", 1)
        cycle, lo, hi = int(cycle_str), int(lo_str), int(hi_str)
    except (ValueError, AttributeError):
        return None
    if cycle < 0 or lo > hi:
        return None
    return (cycle, lo, hi)


@dataclass(frozen=True)
class TrainerDiagnostics:
    """Resolved forensic-diagnostics settings for one training run.

    Built once per ``_train_for_steps`` invocation via :meth:`from_env`; each
    field is ``None`` (disabled) unless its environment variable is set to a
    valid value. See the module docstring for the variable names.
    """

    step_watchdog_seconds: Optional[float] = None
    long_batch_log_codons: Optional[int] = None
    capture_window_detail_codons: Optional[int] = None
    capture_window_steps: Optional[Tuple[int, int, int]] = None

    @classmethod
    def from_env(cls) -> "TrainerDiagnostics":
        """Resolve every forensic knob from the environment in one place."""
        return cls(
            step_watchdog_seconds=_step_watchdog_seconds(),
            long_batch_log_codons=_long_batch_log_codons(),
            capture_window_detail_codons=_capture_window_detail_codons(),
            capture_window_steps=_capture_window_steps(),
        )
