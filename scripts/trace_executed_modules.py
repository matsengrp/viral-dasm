"""Record which ``dasm2`` function bodies execute during a run.

The vendored tree is an import closure, not a hand-picked subset: `dasm2`'s
package ``__init__`` modules eagerly import every head, dataset, loss and
encoder so their registries populate. Most of what is here is therefore
reachable without ever being run, and ``grep`` cannot tell the two apart.

This script can. It installs ``sys.setprofile`` and records, for every frame
entered under ``dasm2/``, the source file and the code object's name. Run it
once per phase; ``scripts/prune_candidates.py`` reads the phases together.

    python scripts/trace_executed_modules.py --out imports.json --phase imports
    python scripts/trace_executed_modules.py --out train.json \
        --script scripts/train.py -- --config ... --parquet ... --device cpu
    python scripts/trace_executed_modules.py --out gate1.json \
        --script scripts/gate1_score_reference.py -- --model dasm-85k ... \
        --num-workers 0

**Pass ``--num-workers 0``.** ``sys.setprofile`` is installed in this process
and this thread. A forked DataLoader worker inherits no profiler and writes no
output, so a trace taken at ``scripts/lib.py``'s ``DATALOADER_WORKERS`` of 6
silently omits everything the dataset, transform chain and collate do. The
script refuses a run whose arguments ask for workers rather than record a
result that is wrong in a direction the reader cannot see.

Two further blind spots are inherent and are documented in the issue that uses
this output rather than worked around here: a function reached only from a
worker *thread* leaves no record, and anything running after the ``finally``
that clears the profiler -- ``atexit``, ``__del__``, weakref finalizers -- is
not seen.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--phase",
        choices=["imports"],
        help="Trace only the imports scripts/train.py performs, and stop.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        help="A script under scripts/ to run end to end under the profiler.",
    )
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if (args.phase is None) == (args.script is None):
        parser.error("give exactly one of --phase imports or --script")

    passthrough = [a for a in args.rest if a != "--"]
    if args.script is not None:
        for complaint in _worker_complaints(passthrough):
            parser.error(complaint)

    dasm2_root = str(REPO / "dasm2")
    prefix_len = len(str(REPO)) + 1
    seen: set[str] = set()

    def profile(frame, event, _arg):
        if event != "call":
            return
        code = frame.f_code
        if code.co_filename.startswith(dasm2_root):
            seen.add(f"{code.co_filename[prefix_len:]}::{code.co_name}")

    # `lib` owns putting the repository root on `sys.path`; importing it is
    # how this script gets that, rather than repeating the insert and being
    # the copy that drifts. Without it the venv's editable install resolves
    # `dasm2` to whatever checkout it was installed from and this traces
    # that tree -- reporting every file of *this* one as dead.
    sys.path.insert(0, str(REPO / "scripts"))
    from lib import assert_dasm2_is_this_repo

    if args.phase == "imports":
        sys.setprofile(profile)
        try:
            import dasm2  # noqa: F401
            from dasm2.train_command import train_model  # noqa: F401
        finally:
            sys.setprofile(None)
        assert_dasm2_is_this_repo()
    else:
        script = (REPO / args.script).resolve()
        sys.argv = [str(script), *passthrough]
        sys.setprofile(profile)
        try:
            runpy.run_path(str(script), run_name="__main__")
        finally:
            sys.setprofile(None)

    if not seen:
        raise RuntimeError(
            "the profiler recorded nothing under "
            f"{dasm2_root}. Either the run imported `dasm2` from another "
            "checkout, or nothing ran. An empty trace reports the whole tree "
            "as dead, so it is an error rather than a result."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sorted(seen), indent=0))
    files = {s.split("::")[0] for s in seen}
    print(f"{len(seen)} code objects in {len(files)} files -> {args.out}")


def _worker_complaints(argv: list[str]) -> list[str]:
    """Every way these arguments would fork a DataLoader worker.

    Two routes reach the same place. ``gate1_score_reference.py`` takes
    ``--num-workers``, defaulting to ``scripts/lib.py``'s
    ``DATALOADER_WORKERS``; ``train.py`` takes none and reads
    ``num_dataloader_workers`` from the rendered config. Both are checked,
    because a trace that quietly loses the dataset path is worse than no
    trace: it reports the dataset as dead.
    """
    complaints = []

    workers = None
    for i, a in enumerate(argv):
        if a == "--num-workers" and i + 1 < len(argv):
            workers = int(argv[i + 1])
        elif a.startswith("--num-workers="):
            workers = int(a.split("=", 1)[1])
    if workers is not None:
        # An explicit flag wins over whatever the config says, so the config
        # is not consulted: gate1's canary config sets 6 and is overridden.
        return [f"--num-workers {workers}: pass 0 instead"] if workers else []

    for i, a in enumerate(argv):
        path = None
        if a == "--config" and i + 1 < len(argv):
            path = Path(argv[i + 1])
        elif a.startswith("--config="):
            path = Path(a.split("=", 1)[1])
        if path is None or not path.exists():
            continue
        for line in path.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "num_dataloader_workers" and value.strip() != "0":
                complaints.append(
                    f"{path} sets num_dataloader_workers: {value.strip()}; "
                    "trace against a config that sets 0"
                )

    return complaints


if __name__ == "__main__":
    main()
