"""Process-level choice of Arrow's memory pool.

Kept as a leaf module (stdlib, plus pyarrow imported inside the function) so
``dasm2/__init__.py`` can apply it before anything else imports pyarrow.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

#: Arrow reads this once, when ``import pyarrow`` first resolves its default pool.
ARROW_POOL_ENV_VAR = "ARROW_DEFAULT_MEMORY_POOL"

ARROW_POOLS = ("system", "jemalloc", "mimalloc")


def use_system_arrow_pool() -> None:
    """Make Arrow allocate from the system allocator unless told otherwise.

    pyarrow 25 bundles mimalloc v3, which marks its arenas ``MADV_HUGEPAGE``.
    On a host with transparent huge pages at ``madvise``, each page fault in
    those arenas then compacts memory synchronously, and a DataLoader worker's
    multi-GB row-group cache stalls in the kernel. pyproject.toml
    requires pyarrow>=25, so every install has that mimalloc. The pools' costs
    are measured in ``experiments/2026-10-03-pyarrow-thp-pool``.

    There are two steps because pyarrow has two defaults. The environment
    variable sets Arrow's C++ default, which pyarrow fixes at import time.
    Parquet's reader properties use that default, so it is the only way to
    move every allocation off mimalloc. It works only if pyarrow has not been
    imported yet, and pandas imports it, so this runs first in
    ``dasm2/__init__.py``. ``set_memory_pool`` sets pyarrow's own default,
    which decoded columns come from, and covers the bulk when something
    imported pyarrow first. That fallback leaves Parquet's reader allocations
    on mimalloc (one hg mapping of about 25 MB per worker, against 8 GB
    without it), so it logs a warning. Forked DataLoader workers inherit both.

    An explicit ``ARROW_DEFAULT_MEMORY_POOL`` naming one of ``ARROW_POOLS``
    wins, which is how to opt back in to mimalloc.

    Raises:
        ValueError: If ``ARROW_DEFAULT_MEMORY_POOL`` names no Arrow pool.
    """
    pyarrow_preloaded = "pyarrow" in sys.modules
    requested = os.environ.setdefault(ARROW_POOL_ENV_VAR, "system")
    if requested not in ARROW_POOLS:
        raise ValueError(
            f"{ARROW_POOL_ENV_VAR}={requested!r} names no Arrow memory pool; "
            f"use one of {ARROW_POOLS}"
        )
    # Imported here, after the variable is set: a module-level import would
    # resolve Arrow's default pool before this function could choose it.
    import pyarrow as pa

    if pa.default_memory_pool().backend_name != requested:
        if pyarrow_preloaded:
            logger.warning(
                "pyarrow was imported before dasm2, so Parquet's reader still "
                "allocates from %s, not %s; under pyarrow 25 mimalloc requests "
                "transparent huge pages. Import dasm2 before pandas or pyarrow, "
                "or set %s before starting Python (#1368).",
                pa.default_memory_pool().backend_name,
                requested,
                ARROW_POOL_ENV_VAR,
            )
        pa.set_memory_pool(getattr(pa, f"{requested}_memory_pool")())
