"""Device selection and GPU utilities for DASM2."""

import inspect
import os
import resource
import subprocess
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch


def find_least_used_cuda_gpu() -> Optional[int]:
    """Find the least utilized CUDA GPU for optimal resource allocation.

    Uses a priority system to select the best available GPU:
    1. GPU with lowest utilization (if any GPU is being used)
    2. GPU with lowest memory usage (if any GPU has allocated memory)
    3. GPU with fewest running processes (if there are differences)

    Returns:
        GPU index (0-based) of least used GPU, or None if all GPUs are idle

    Raises:
        subprocess.CalledProcessError: If nvidia-smi command fails
    """
    gpu_metrics = _query_gpu_metrics()
    if not gpu_metrics:
        return None
    return _select_best_gpu(gpu_metrics)


def rank_cuda_gpus_by_usage(
    allowed: Optional[Sequence[int]] = None,
) -> List[int]:
    """Rank CUDA GPUs from least to most used.

    Same priority order as ``find_least_used_cuda_gpu`` (utilization,
    then memory, then process count), but returns the full ranking
    rather than just the top pick. When all candidate GPUs are
    completely idle, the natural index order is preserved.

    Args:
        allowed: Optional candidate set; indices outside this set are
            excluded. ``None`` ranks all visible GPUs.

    Returns:
        GPU indices in priority order (least used first). Empty list if
        ``nvidia-smi`` cannot be queried or no candidate exists.
    """
    metrics = _query_gpu_metrics()
    if metrics is None:
        return []

    n_gpus = len(metrics["utilization"])
    if allowed is None:
        candidates = list(range(n_gpus))
    else:
        candidates = [i for i in allowed if 0 <= i < n_gpus]
    if not candidates:
        return []

    utilization = metrics["utilization"]
    memory_used = metrics["memory_used"]
    process_counts = metrics["process_counts"]
    uuids = metrics["uuids"]

    return sorted(
        candidates,
        key=lambda i: (
            utilization[i],
            memory_used[i],
            process_counts[uuids[i]],
            i,
        ),
    )


def _query_gpu_metrics() -> Optional[Dict[str, Any]]:
    """Collect comprehensive GPU metrics for intelligent device selection.

    Combines GPU utilization, memory usage, and process counts into a unified
    metrics dictionary for optimal GPU selection.

    Returns:
        Dictionary with GPU UUIDs, utilization percentages, memory usage,
        process counts, and UUID-to-index mapping. None if nvidia-smi fails.
    """
    # Query GPU utilization and memory usage
    utilization_data = _query_gpu_utilization_and_memory()
    if not utilization_data:
        return None

    uuids, utilization, memory_used = utilization_data

    # Query process counts
    process_counts = _query_gpu_process_counts(uuids)
    if process_counts is None:
        return None

    return {
        "uuids": uuids,
        "utilization": utilization,
        "memory_used": memory_used,
        "process_counts": process_counts,
        "uuid_to_index": {uuid: idx for idx, uuid in enumerate(uuids)},
    }


def _query_gpu_utilization_and_memory() -> Optional[
    Tuple[List[str], List[int], List[int]]
]:
    """Extract GPU utilization and memory data using nvidia-smi.

    Returns:
        Tuple of (UUIDs, utilization_percentages, memory_used_mb) or None if failed
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=gpu_uuid,utilization.gpu,memory.used",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("Error running nvidia-smi.")
        return None

    lines = result.stdout.strip().split("\n")
    gpu_data = [line.split(", ") for line in lines]

    uuids = [gpu[0] for gpu in gpu_data]
    utilization = [int(gpu[1]) for gpu in gpu_data]
    memory_used = [int(gpu[2]) for gpu in gpu_data]

    return uuids, utilization, memory_used


def _query_gpu_process_counts(uuids: List[str]) -> Optional[Counter[str]]:
    """Count active processes per GPU using nvidia-smi.

    Returns:
        Counter mapping GPU UUIDs to process counts, or None if query failed
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,name",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("Error running nvidia-smi.")
        return None

    process_entries = [
        line.split(", ") for line in result.stdout.strip().split("\n") if line
    ]

    # Count the number of processes per GPU
    gpu_counts = Counter({uuid: 0 for uuid in uuids})
    gpu_counts.update(
        [uuid for uuid, proc_name in process_entries if proc_name != "[Not Found]"]
    )

    return gpu_counts


def _select_best_gpu(metrics: Dict[str, Any]) -> Optional[int]:
    """Select optimal GPU using prioritized criteria.

    Priority order: lowest utilization > lowest memory > fewest processes.
    Returns None if all GPUs are completely idle.
    """
    utilization = metrics["utilization"]
    memory_used = metrics["memory_used"]
    process_counts = metrics["process_counts"]
    uuid_to_index = metrics["uuid_to_index"]

    # Priority 1: Choose by utilization if any GPU is being used
    if max(utilization) > 0:
        print("GPU chosen via utilization")
        return utilization.index(min(utilization))  # type: ignore[no-any-return]

    # Priority 2: Choose by memory usage if any GPU has allocated memory
    if max(memory_used) > 0:
        print("GPU chosen via memory")
        return memory_used.index(min(memory_used))  # type: ignore[no-any-return]

    # Priority 3: Choose by process count if there are differences
    if len(set(process_counts.values())) > 1:
        print("GPU chosen via process count")
        return min(  # type: ignore[no-any-return]
            uuid_to_index[uuid]
            for uuid, count in process_counts.items()
            if count == min(process_counts.values())
        )

    return None  # All GPUs are idle


def pick_device(gpu_preference: Optional[Union[str, int]] = None) -> torch.device:
    """Select optimal PyTorch device with intelligent GPU selection.

    Implements a sophisticated device selection strategy:
    - String preference: Direct device specification (strong preference)
    - Integer preference: Weak GPU preference with intelligent fallback
    - No preference: Automatic selection (CUDA > MPS > CPU)

    For CUDA, attempts to find least utilized GPU first, falling back to
    preference-based or random selection if all GPUs are idle.

    Args:
        gpu_preference: Device string (e.g., "cuda:1") or GPU index preference

    Returns:
        PyTorch device object configured for optimal performance

    Raises:
        RuntimeError: If CUDA device is inaccessible or preference is invalid
    """
    if _is_explicit_device_preference(gpu_preference):
        return torch.device(str(gpu_preference))  # Cast to str for torch.device

    gpu_index = gpu_preference if isinstance(gpu_preference, int) else None
    return _select_best_available_device(gpu_index)


def _is_explicit_device_preference(preference: Optional[Union[str, int]]) -> bool:
    """Check if preference is an explicit device string (strong preference)."""
    return preference is not None and isinstance(preference, str)


def _select_best_available_device(gpu_index_preference: Optional[int]) -> torch.device:
    """Choose optimal device with fallback hierarchy: CUDA > MPS > CPU."""
    if _is_cuda_available():
        return _select_cuda_device(gpu_index_preference)
    elif torch.backends.mps.is_available():
        print("Using Metal Performance Shaders")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def _is_cuda_available() -> bool:
    """Verify CUDA availability and initialization status."""
    if not torch.backends.cudnn.is_available():
        return False

    try:
        torch._C._cuda_init()
        return True
    except Exception:  # CUDA may not be available
        return False


def _select_cuda_device(gpu_index_preference: Optional[int]) -> torch.device:
    """Select CUDA device with intelligent GPU selection and accessibility testing."""
    which_gpu = find_least_used_cuda_gpu()

    if which_gpu is None:
        # All GPUs are idle, use preference or random selection
        if gpu_index_preference is None:
            which_gpu = np.random.randint(torch.cuda.device_count())
        else:
            which_gpu = gpu_index_preference % torch.cuda.device_count()

    print(f"Using CUDA GPU {which_gpu}")
    device = torch.device(f"cuda:{which_gpu}")

    # Test device accessibility
    torch.ones(1).to(device)
    return device


def print_tensor_devices(scope: str = "local") -> None:
    """Debug utility to print device locations of all PyTorch tensors.

    Scans the specified scope for PyTorch tensors and prints their device
    assignments. Useful for debugging device placement issues in multi-GPU
    or mixed CPU/GPU computations.

    Args:
        scope: Variable scope to scan ('local' or 'global')

    Raises:
        ValueError: If scope is not 'local' or 'global'
    """
    if scope == "local":
        frame = inspect.currentframe()
        if frame and frame.f_back:
            variables = frame.f_back.f_locals
        else:
            variables = {}
    elif scope == "global":
        variables = globals()
    else:
        raise ValueError("Scope must be 'local' or 'global'.")

    for var_name, var_value in variables.items():
        if isinstance(var_value, torch.Tensor):
            print(f"{var_name}: {var_value.device}")


def get_memory_usage_mb() -> float:
    """Get peak memory usage of current process in megabytes.

    Returns:
        Peak memory usage in MB using system resource tracking
    """
    # Returns the peak memory usage in MB
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024  # Convert from KB to MB


def get_current_rss_mb() -> Optional[float]:
    """Get *current* resident set size of this process in megabytes.

    Complements :func:`get_memory_usage_mb`, which reports ``ru_maxrss`` — a
    process-lifetime high-water mark that can only rise. Diagnosing whether a
    loop accumulates memory needs the instantaneous value: a high-water mark
    cannot show memory being released, so a flat ``ru_maxrss`` and a falling
    RSS look identical through it (issue #988).

    Reads ``/proc/self/statm`` rather than taking a ``psutil`` dependency for
    two integers. That file is Linux-only, which is where the GPU fleet runs.

    Returns:
        Current RSS in MB, or None on a platform without ``/proc/self/statm``.
    """
    return _rss_mb_for_pid("self")


#: Read by :func:`mem_available_bytes` / :func:`mem_total_bytes`; a module
#: attribute so tests can point it at a fixture file.
_MEMINFO_PATH = "/proc/meminfo"


def _meminfo_bytes(field: str) -> Optional[int]:
    """One ``/proc/meminfo`` field in bytes, or None if unreadable."""
    try:
        with open(_MEMINFO_PATH, "r") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith(f"{field}:"):
            # Format: "MemAvailable:   12345678 kB"
            return int(line.split()[1]) * 1024
    return None


def mem_available_bytes() -> Optional[int]:
    """The kernel's ``MemAvailable`` estimate in bytes, or None if unreadable.

    ``MemAvailable`` (not ``MemFree``) counts reclaimable page cache, which is
    what a new allocation can actually get. Sizes the DataLoader prefetch
    guard (issue #1272).
    """
    return _meminfo_bytes("MemAvailable")


def mem_total_bytes() -> Optional[int]:
    """The host's ``MemTotal`` in bytes, or None if unreadable.

    Sizes the Parquet row-group cache (issue #1272).
    """
    return _meminfo_bytes("MemTotal")


def _rss_mb_for_pid(pid: Union[int, str]) -> Optional[float]:
    """Resident set size of one process in MB, or None if it is unreadable.

    Args:
        pid: PID, or the literal ``"self"``.

    Returns:
        RSS in MB, or None on a non-Linux platform or a process that exited
        between being listed and being read.
    """
    try:
        with open(f"/proc/{pid}/statm", "r") as handle:
            resident_pages = int(handle.read().split()[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return None
    return resident_pages * page_size / (1024**2)


def get_process_tree_rss_mb() -> Optional[float]:
    """Get current RSS of this process *and all its descendants*, in MB.

    :func:`get_current_rss_mb` reads ``/proc/self/statm``, which — like
    ``getrusage(RUSAGE_SELF)`` — excludes child processes by construction. That
    was harmless while the memory-sensitive sweeps ran at ``num_workers=0``,
    but the streaming fetch path forks up to ``MAX_FETCH_WORKERS`` DataLoader
    workers, so a self-only reading now omits a real term of the host footprint
    the #988 success criteria are written against. ``RUSAGE_CHILDREN`` is not a
    substitute: it reports the largest single *reaped* child, and persistent
    DataLoader workers are only reaped at loader teardown.

    Walks ``/proc/<pid>/task/<tid>/children`` and sums per-process RSS. Two
    caveats worth stating, since this is an instrument:

    - Copy-on-write pages shared with forked children are counted once per
      process, so for ``fork``ed workers this over-counts — it is an upper
      bound on the private footprint, not a proportional-set-size figure.
    - The walk is not atomic; a process that exits mid-walk drops out.

    Returns:
        Summed RSS in MB, or None on a platform without ``/proc``.
    """
    own = _rss_mb_for_pid("self")
    if own is None:
        return None
    total = own
    pending = _child_pids(os.getpid())
    seen: Set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        rss = _rss_mb_for_pid(pid)
        if rss is not None:
            total += rss
        pending.extend(_child_pids(pid))
    return total


def _child_pids(pid: int) -> List[int]:
    """Direct children of ``pid``, read from the per-thread ``children`` files.

    Args:
        pid: Parent PID.

    Returns:
        Child PIDs; empty if the process is gone or ``/proc`` is unavailable.
    """
    children: List[int] = []
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return children
    for tid in tids:
        try:
            with open(f"/proc/{pid}/task/{tid}/children", "r") as handle:
                children.extend(int(token) for token in handle.read().split())
        except (OSError, ValueError):
            continue
    return children
