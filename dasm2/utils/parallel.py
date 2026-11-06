"""Multiprocessing and parallelization utilities for DASM2."""

import multiprocessing as mp
from collections import deque
from functools import wraps
from itertools import islice, repeat
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sized,
    Tuple,
    TypeVar,
    Union,
)

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dasm2.utils.batch_operations import move_batch_to_device


def force_spawn() -> None:
    """Force the spawn start method for multiprocessing.

    This is necessary to avoid conflicts with the internal OpenMP-based thread pool in
    PyTorch. The spawn method starts a fresh Python interpreter process, avoiding
    thread-safety issues with CUDA contexts and OpenMP.

    Raises:
        RuntimeError: If multiprocessing context has already been set
    """
    mp.set_start_method("spawn", force=True)


T = TypeVar("T")
R = TypeVar("R", bound=Sized)

# Timeout (seconds) for polling pending results.  Short enough to detect
# worker death promptly; long enough to avoid busy-waiting.
_POLL_TIMEOUT = 5.0


def chunked_async_spawn_map(
    func: Callable[[T], R],
    chunks: Iterable[T],
    num_workers: int,
    total_items: Optional[int] = None,
    desc: str = "Processing",
    result_len: Callable[[R], int] = len,
    max_pending: Optional[int] = None,
    initializer: Optional[Callable[[], None]] = None,
) -> List[R]:
    """Apply *func* to each item from a lazy *chunks* iterable in parallel.

    This is the recommended way to parallelise CPU-bound work that is fed by
    an expensive generator (e.g. one that runs GPU forward passes).  It
    improves on ``pool.imap`` / ``pool.map`` in three ways:

    1. **Bounded memory via back-pressure.**  ``pool.imap`` eagerly drains its
       input iterable from an internal thread, so passing a generator does
       *not* limit how many chunks are materialised.  Here, at most
       *max_pending* chunks are in flight at any time — the main process
       pauses the generator when enough chunks are queued.

    2. **Pipeline overlap.**  Unlike ``pool.map`` (which blocks until every
       item in the batch finishes), chunks are dispatched to workers the
       instant the generator yields them.  GPU chunk-generation and CPU
       optimisation therefore run concurrently.  The default
       ``max_pending = 2 * num_workers`` ensures a buffer of pre-generated
       chunks so that workers never stall waiting for chunk preparation.

    3. **Spawn context.**  A ``spawn`` pool is used unconditionally.  ``fork``
       would copy any initialised CUDA context into every worker, causing
       severe contention even for CPU-only work.

    Results are returned **in generator order** (not completion order), so the
    caller can safely concatenate them.  An empty iterable returns ``[]``.

    Args:
        func: Worker function applied to each chunk.  Must be picklable
            (top-level function, not a closure or lambda).
        chunks: Iterable of work items.  Only consumed as fast as workers
            can process — safe to pass an unbounded generator.
        num_workers: Size of the ``spawn`` worker pool.
        total_items: Total number of *items* (not chunks) for the progress
            bar.  Each result contributes ``result_len(result)`` items.
            ``None`` shows an indeterminate bar.  Example: for 100 sequences
            split into 10-item chunks, pass ``total_items=100``.
        desc: Progress bar description.
        result_len: Callable that returns the number of items in a single
            result, used to advance the progress bar.  Defaults to ``len``.
        max_pending: Maximum number of chunks in flight (submitted but not
            yet harvested).  Defaults to ``2 * num_workers``, which keeps
            a buffer of pre-generated chunks so workers never stall waiting
            for the generator.
        initializer: Optional picklable callable run once in each worker
            at startup (``Pool(initializer=...)``).

    Returns:
        Ordered list of results, one per chunk, in generator order.

    Raises:
        Any exception raised by *func* in a worker process — re-raised in the
        main process when that chunk's result is harvested.
    """
    if max_pending is None:
        max_pending = 2 * num_workers

    ctx = mp.get_context("spawn")
    all_results: List[R] = []
    pending: deque[mp.pool.AsyncResult] = deque()

    with (
        ctx.Pool(num_workers, initializer=initializer) as pool,
        # smoothing=0 gives an overall-average rate rather than per-step EMA,
        # which is more stable for variable-length chunks.
        tqdm(total=total_items, desc=desc, delay=2.0, smoothing=0) as bar,
    ):

        def _collect_ready() -> None:
            """Harvest completed results from the head of the queue."""
            while pending and pending[0].ready():
                result = pending.popleft().get()
                all_results.append(result)
                bar.update(result_len(result))

        for chunk in chunks:
            async_result = pool.apply_async(func, (chunk,))
            pending.append(async_result)

            _collect_ready()

            # Back-pressure: pause the producer until the in-flight count
            # drops below max_pending.  The pre-generated buffer
            # (max_pending − num_workers chunks) absorbs generator latency
            # so workers always have work ready.
            #
            # Note: _collect_ready only harvests from the head (to preserve
            # order), so a slow head-chunk can stall the producer even when
            # later chunks have finished.  The timeout prevents an infinite
            # hang if a worker dies, but does not fully solve head-of-line
            # blocking.  This is acceptable for our workload where chunks
            # are uniformly sized.
            while len(pending) >= max_pending:
                pending[0].wait(timeout=_POLL_TIMEOUT)
                _collect_ready()

        # Drain remaining results in order.
        while pending:
            result = pending.popleft().get()
            all_results.append(result)
            bar.update(result_len(result))

    return all_results


def batched_over_dataset(
    function: Callable,
    batch_arg_idx: int = 0,
    batch_size: int = 32,
    device: str = "cpu",
    collate_fn: Optional[Callable] = None,
) -> Callable:
    """Wrap a batch-processing function to process entire datasets.

    Transforms a function that operates on a single batch into one that
    processes an entire dataset, handling batching and concatenation automatically.
    Results are moved to CPU before concatenation.

    Args:
        function: Function to wrap.
        batch_arg_idx: Positional index of the batch argument (default: 0).
        batch_size: Batch size for processing.
        device: Device to move batches to (default: "cpu").
        collate_fn: Optional collation function for the DataLoader. Required
            for datasets with variable-length items (e.g., ``TransformDataset``).

    Returns:
        Wrapped function that accepts a dataset at batch_arg_idx position,
        processes in batches, and returns concatenated results (tensor or dict of tensors).

    Example:
        >>> batched_forward = batched_over_dataset(model.forward, batch_size=64)
        >>> all_outputs = batched_forward(dataset, head_names=[HeadType.SELECTION_FACTOR_HEAD])
        >>> all_factors = all_outputs[HeadType.SELECTION_FACTOR_HEAD]
    """

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if len(args) <= batch_arg_idx:
            raise ValueError(
                f"Expected at least {batch_arg_idx + 1} positional arguments, got {len(args)}"
            )

        dataset = args[batch_arg_idx]
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"Expected Dataset at position {batch_arg_idx}, got {type(dataset).__name__}"
            )

        # Create dataloader. For eager datasets default collate works since
        # they pre-pad; unified TransformDataset requires a TransformCollator.
        loader_kwargs: Dict[str, Any] = {}
        if collate_fn is not None:
            loader_kwargs["collate_fn"] = collate_fn
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
        )

        results: List[Any] = []

        for batch in dataloader:
            # Move batch to device
            batch = move_batch_to_device(batch, device)

            # Reconstruct args with batch instead of dataset
            new_args = args[:batch_arg_idx] + (batch,) + args[batch_arg_idx + 1 :]
            result = function(*new_args, **kwargs)

            # Move result to CPU
            if isinstance(result, dict):
                result = {k: v.cpu() for k, v in result.items()}
            elif isinstance(result, torch.Tensor):
                result = result.cpu()

            results.append(result)

        # Concatenate results
        if not results:
            raise ValueError("No results to concatenate (empty dataset?)")

        if isinstance(results[0], dict):
            return {k: torch.cat([r[k] for r in results]) for k in results[0].keys()}
        else:
            return torch.cat(results)

    return wrapper


# from https://docs.python.org/3.11/library/itertools.html#itertools-recipes
# avoiding walrus:
def chunked(
    iterable: Iterable[Union[str, int, torch.Tensor]], n: int
) -> Iterable[List[Union[str, int, torch.Tensor]]]:
    """Chunk data into lists of length n. The last chunk may be shorter.

    Generator that yields consecutive chunks of the input iterable, each containing
    at most n elements. The final chunk may contain fewer than n elements.

    Args:
        iterable: Input data to chunk
        n: Maximum chunk size

    Yields:
        Lists of at most n elements from the input iterable
    """
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


def chunk_function(
    first_chunkable_idx: int = 0,
    default_chunk_size: int = 2048,
    progress_bar_name: Optional[str] = None,
) -> Callable:
    """Decorator for automatic function input chunking with batch processing.

    Transforms functions to process large inputs in smaller chunks, enabling
    processing of datasets that don't fit in memory. Handles both single tensor
    and tuple of tensor outputs.

    The decorated function processes chunks sequentially and concatenates results.
    All positional arguments from first_chunkable_idx onward must be iterables
    of the same length.

    Args:
        first_chunkable_idx: Index of first argument to chunk (earlier args passed through)
        default_chunk_size: Default chunk size (can be overridden with chunk_size kwarg)
        progress_bar_name: Progress bar description (None for no progress bar)

    Returns:
        Decorator function that transforms the target function

    Example:
        >>> @chunk_function(progress_bar_name="Processing sequences")
        >>> def process_sequences(model, sequences, labels):
        >>>     return model(sequences), model.predict(labels)
        >>> # Automatically chunks sequences and labels, concatenates results
    """

    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if "chunk_size" in kwargs:
                chunk_size = kwargs.pop("chunk_size")
            else:
                chunk_size = default_chunk_size
            pre_chunk_args = args[:first_chunkable_idx]
            chunkable_args = args[first_chunkable_idx:]

            results = []
            if progress_bar_name is None:
                progargs: Dict[str, Any] = {"disable": True}
            else:
                progargs = {"desc": progress_bar_name}
            bar = tqdm(total=len(chunkable_args[0]), delay=2.0, **progargs)
            for chunked_args in zip(
                *(chunked(arg, chunk_size) for arg in chunkable_args)
            ):
                bar.update(len(chunked_args[0]))
                results.append(function(*pre_chunk_args, *chunked_args, **kwargs))
            if isinstance(results[0], tuple):
                return tuple(torch.cat(tensors) for tensors in zip(*results))
            else:
                return torch.cat(results)

        return wrapper

    return decorator


def _apply_args_and_kwargs(
    func: Callable,
    pre_chunk_args: Tuple[Any, ...],
    chunked_args: Any,
    kwargs: Dict[str, Any],
) -> Any:
    """Helper function to apply arguments to a function in multiprocessing context.

    Used internally by parallelize_function to reconstruct function calls
    in worker processes.
    """
    return func(*pre_chunk_args, *chunked_args, **kwargs)


def parallelize_function(
    function: Callable,
    first_chunkable_idx: int = 0,
    max_workers: int = 10,
    min_chunk_size: int = 1000,
) -> Callable:
    """Wrap function for CPU-based multiprocessing with automatic load balancing.

    Creates a multiprocessing wrapper for CPU-bound functions, automatically
    distributing work across available cores. Designed for CPU-only operations
    to avoid GPU context conflicts.

    Dynamically adjusts worker count based on data size to maintain efficient
    chunk sizes. Falls back to sequential processing for small datasets.

    Args:
        function: Function to parallelize (must be CPU-compatible)
        first_chunkable_idx: Index of first argument to distribute across workers
        max_workers: Maximum number of worker processes
        min_chunk_size: Minimum chunk size per worker (auto-adjusts worker count)

    Returns:
        Wrapped function with identical interface plus automatic parallelization

    Note:
        Uses spawn multiprocessing method to avoid PyTorch/CUDA context issues.
        Not suitable for GPU operations.
    """

    max_worker_count = min(mp.cpu_count() // 2, max_workers)
    if max_worker_count <= 1:
        return function
    force_spawn()

    @wraps(function)
    def wrapper(*args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        if len(args) <= first_chunkable_idx:
            raise ValueError(
                f"Function {function.__name__} cannot be parallelized without chunkable arguments"
            )
        pre_chunk_args = args[:first_chunkable_idx]
        chunkable_args = args[first_chunkable_idx:]
        min_worker_count = len(chunkable_args[0]) // min_chunk_size

        worker_count = min(min_worker_count, max_worker_count)
        if worker_count <= 1:
            return function(*args, **kwargs)

        chunk_size = (len(chunkable_args[0]) // worker_count) + 1
        chunked_args = list(zip(*(chunked(arg, chunk_size) for arg in chunkable_args)))
        with mp.Pool(worker_count) as pool:
            results = pool.starmap(
                _apply_args_and_kwargs,
                list(
                    zip(
                        repeat(function),
                        repeat(pre_chunk_args),
                        chunked_args,
                        repeat(kwargs),
                    )
                ),
            )
            pool.close()
            pool.join()
        if isinstance(results[0], tuple):
            concatenated = tuple(torch.cat(tensors) for tensors in zip(*results))
        else:
            concatenated = torch.cat(results)  # type: ignore[assignment]

        del results

        return concatenated

    return wrapper


def _apply_func_to_df_chunk(
    df_chunk: pd.DataFrame,
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Apply function to DataFrame chunk using pandas apply with optional progress.

    Helper function for parallel_df_apply that processes a chunk of DataFrame
    rows in a worker process. Uses tqdm progress_apply if requested.
    """
    if kwargs.pop("use_progress_apply", False):
        return df_chunk.progress_apply(func, axis=1, args=args, **kwargs)  # type: ignore[operator]
    else:
        return df_chunk.apply(func, axis=1, args=args, **kwargs)


def parallel_df_apply(
    df: pd.DataFrame,
    func: Callable[[pd.Series], Any],
    max_workers: int = 10,
    min_chunk_size: int = 1000,
    parallelize: bool = True,
    force_parallel: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Apply function to DataFrame rows with automatic parallelization.

    Efficiently processes large DataFrames by distributing rows across multiple
    processes. Preserves original DataFrame index order, including sparse or
    non-contiguous indices.

    Automatically determines optimal parallelization strategy based on data size
    and available resources. Falls back to sequential processing for small datasets.

    Args:
        df: DataFrame to process
        func: Function to apply to each row (receives pd.Series)
        max_workers: Maximum number of worker processes
        min_chunk_size: Minimum rows per worker (auto-adjusts worker count)
        parallelize: Enable parallel processing (False for sequential)
        force_parallel: Force specific number of workers (overrides auto-detection)
        use_progress_apply: Use tqdm progress bar (passed in kwargs)
        *args: Additional arguments passed to func
        **kwargs: Additional keyword arguments passed to func

    Returns:
        Series with results from applying func, preserving original index

    Note:
        Each worker receives a DataFrame chunk and uses pandas apply() internally.
    """
    data_length = len(df)

    max_worker_count = min(mp.cpu_count() // 2, max_workers)
    if (force_parallel is None) and (
        not parallelize or data_length < min_chunk_size or max_worker_count <= 1
    ):
        print("using sequential processing")
        # Fall back to sequential processing.
        if kwargs.pop("use_progress_apply", False):
            return df.progress_apply(func, axis=1, args=args, **kwargs)  # type: ignore[operator]
        else:
            return df.apply(func, axis=1, args=args, **kwargs)

    force_spawn()

    if force_parallel is not None:
        worker_count = force_parallel
    else:
        # Calculate optimal worker count based on data size.
        min_worker_count = data_length // min_chunk_size
        worker_count = min(min_worker_count, max_worker_count)

    chunk_size = (data_length // worker_count) + 1

    # Create DataFrame chunks preserving index.
    # Chunk rows by position but preserve original index.
    position_chunks = list(chunked(range(len(df)), chunk_size))
    df_chunks = [df.iloc[chunk_positions] for chunk_positions in position_chunks]  # type: ignore[index]

    with mp.Pool(worker_count) as pool:
        results = pool.starmap(
            _apply_func_to_df_chunk,
            list(zip(df_chunks, repeat(func), repeat(args), repeat(kwargs))),
        )

    # Concatenate results preserving original index order.
    # pd.concat maintains the index from each chunk, so sparse indices are preserved.
    return pd.concat(results)
