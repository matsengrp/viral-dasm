"""Safe checkpoint selection over a training run's output directory.

Two questions get asked of a finished run's checkpoint directory, and this
module answers both without ever guessing:

``resolve_final_checkpoint``
    "Which file is this run's deliverable?" — the post-cycle checkpoint of
    the last cycle. This replaces the ``ls | grep -v '_cycle' | sort |
    tail -1`` shell idiom that Snakefiles used to inline.

``select_best_checkpoint``
    "Which cycle actually validated best?" — the trainer resets and re-tracks
    ``best_val_loss`` at the start of every cycle
    (``Trainer._reset_best_model_tracking``), so each post-cycle save is only
    the best *within its cycle*. A later cycle's warm restart can end above an
    earlier cycle's minimum, in which case the final cycle's file is not the
    run's best.

Why this is a module and not a one-liner (issue #1011, incident #950): the
shell idiom sorts on a metadata hash, which is not chronological, so a
directory holding two attempts' outputs resolved to an arbitrary one of them.
The correct checkpoint was then deleted as "stale". Every ambiguity this
module can detect is therefore raised, never resolved by ordering luck:

- Two attempts' post-cycle checkpoints for the same cycle → raise, listing
  each candidate's run token and mtime, and how to disambiguate.
- Two runs' logs reporting the same cycle index → raise. Non-overlapping
  cycle indices across logs are a genuine resume and get merged, which is how
  a best cycle that precedes the resume point stays findable.

Checkpoints written since #1011 carry a run token
(:func:`dasm2.serialization.naming.run_token_from_run_id`), so
``run_token=`` resolves those ambiguities outright. Legacy checkpoints
predate the token; they are still readable here, but a legacy directory with
two attempts in it can only be reported, not untangled.

A directory deliberately holding several *models* (rather than several
attempts at one) is separated by ``pattern=`` instead. That glob scopes the
logs as well as the checkpoint files — via the run tokens on the files it
matched — so a sibling model's cycle plan and cycle numbering stay out of
this model's safety checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from dasm2.serialization.naming import parse_run_token, run_token_from_run_id
from dasm2.training.log_reader import LOG_EXTENSION, find_logs, read_log

#: Suffix marker on a post-cycle save, whose weights are the best-validation
#: weights of that cycle (``Trainer._restore_best_model`` runs immediately
#: before the save).
POST_CYCLE_MARKER = "best"

#: Suffix marker on a step-cadence save (``CheckpointSaveHook.on_step``,
#: issue #1640): a resume point at a step that did not evaluate.
PERIODIC_MARKER = "periodic"

#: Step-cadence save: ``_[cycle{N}_]periodic_step{M}_``.
_PERIODIC_PATTERN = re.compile(rf"_(?:cycle\d+_)?{PERIODIC_MARKER}_step\d+_")

#: Post-cycle checkpoint written after issue #1011: ``_cycle{N}_best_``.
_CYCLE_BEST_PATTERN = re.compile(rf"_cycle(\d+)_{POST_CYCLE_MARKER}_")

#: On-eval intermediate save. Multi-cycle runs use ``_cycle{N}_step{M}_``;
#: single-cycle runs use a bare ``_step{M}_`` with no cycle component, which
#: is exactly why "has no ``_cycle``" was never a safe test for "is final".
_ON_EVAL_PATTERN = re.compile(r"_(?:cycle\d+_)?step\d+_")

#: Legacy post-cycle checkpoint for a non-final cycle: ``_cycle{N}_`` followed
#: by the metadata hash rather than a ``step``/``best`` component.
_LEGACY_CYCLE_PATTERN = re.compile(r"_cycle(\d+)_")


@dataclass(frozen=True)
class PostCycleCheckpoint:
    """A post-cycle checkpoint file and what its filename says about it."""

    path: Path
    #: 0-indexed cycle, or ``None`` for a legacy final-cycle save, which
    #: carried no cycle component at all.
    cycle: Optional[int]
    #: Run token, or ``None`` for checkpoints written before run tokens
    #: existed or by a run whose logger was disabled.
    run_token: Optional[str]
    #: Whether the filename carries the ``_best`` marker, i.e. was written
    #: after #1011. Legacy directories name only *non-final* cycles, so
    #: "highest cycle number present" means different things in the two eras.
    marked_best: bool

    @property
    def mtime(self) -> float:
        return self.path.stat().st_mtime

    def describe(self) -> str:
        """One-line diagnostic used in ambiguity errors."""
        cycle = "final (legacy, unnumbered)" if self.cycle is None else self.cycle
        return (
            f"{self.path.name} (cycle={cycle}, "
            f"run_token={self.run_token or 'none'}, mtime={self.mtime:.0f})"
        )


@dataclass(frozen=True)
class CycleBest:
    """A ``cycle_end`` record: one cycle's best validation loss and its run."""

    cycle: int
    best_val_loss: float
    run_id: str


@dataclass(frozen=True)
class SelectedCheckpoint:
    """Result of a selection, with the provenance that justified it."""

    path: Path
    #: 0-indexed cycle, or ``None`` for a legacy unnumbered final checkpoint.
    cycle: Optional[int]
    #: ``best_val_loss`` for that cycle, or ``None`` when no log recorded it.
    best_val_loss: Optional[float]
    run_token: Optional[str]


class AmbiguousCheckpointError(ValueError):
    """More than one checkpoint (or run) could satisfy the request.

    Raised rather than resolved: picking one by sort order is what destroyed
    a run's best weights in #950.
    """


def is_periodic_checkpoint(path: Path) -> bool:
    """Whether *path* is a step-cadence save (``periodic_step{M}``), named by filename alone."""
    return _PERIODIC_PATTERN.search(path.name) is not None


def _classify(path: Path) -> Optional[PostCycleCheckpoint]:
    """Return the post-cycle record for *path*, or None if it isn't one."""
    name = path.name
    run_token = parse_run_token(name)

    match = _CYCLE_BEST_PATTERN.search(name)
    if match is not None:
        return PostCycleCheckpoint(path, int(match.group(1)), run_token, True)

    # Intermediate on-eval and step-cadence saves are not post-cycle
    # checkpoints. Checked before the legacy patterns because both of those
    # would match them.
    if is_periodic_checkpoint(path) or _ON_EVAL_PATTERN.search(name):
        return None

    match = _LEGACY_CYCLE_PATTERN.search(name)
    if match is not None:
        return PostCycleCheckpoint(path, int(match.group(1)), run_token, False)

    # Neither a cycle nor a step component: a legacy final-cycle save.
    return PostCycleCheckpoint(path, None, run_token, False)


def find_post_cycle_checkpoints(
    checkpoints_dir: Path, pattern: str = "*.zip"
) -> list[PostCycleCheckpoint]:
    """Collect every post-cycle checkpoint in *checkpoints_dir*.

    Args:
        checkpoints_dir: Directory a training run wrote its checkpoints to.
        pattern: Glob restricting which files are considered, for directories
            that intentionally hold more than one model's output (e.g.
            ``"*_M0Head-*.zip"``).

    Returns:
        Post-cycle checkpoints, sorted by cycle then filename. On-eval
        intermediates are excluded.
    """
    found = [
        record
        for path in sorted(checkpoints_dir.glob(pattern))
        if (record := _classify(path)) is not None
    ]
    return sorted(
        found, key=lambda r: (r.cycle if r.cycle is not None else -1, r.path.name)
    )


def final_cycle_candidates(
    candidates: Iterable[PostCycleCheckpoint],
) -> list[PostCycleCheckpoint]:
    """The subset of *candidates* that could be a run's final-cycle save.

    Legacy directories number only the *non-final* cycles and leave the final
    save unsuffixed, so an unnumbered file is by definition a final one.
    Post-#1011 directories number every cycle and mark it ``_best``, so there
    the highest number wins. A directory holding both is a directory holding
    two attempts, and both eras' finals come back — callers decide whether to
    raise (:func:`resolve_final_checkpoint`) or offer the choice.

    Args:
        candidates: Post-cycle checkpoints from one directory.

    Returns:
        Final-cycle candidates. Empty when *candidates* holds only numbered
        legacy per-cycle saves, i.e. the run never wrote a final save.
    """
    candidates = list(candidates)
    legacy_finals = [c for c in candidates if c.cycle is None]
    marked = [c for c in candidates if c.marked_best and c.cycle is not None]
    if not marked:
        return legacy_finals
    final_cycle = max(c.cycle for c in marked if c.cycle is not None)
    return legacy_finals + [c for c in marked if c.cycle == final_cycle]


def final_cycle_checkpoint_paths(paths: Iterable[Path]) -> list[Path]:
    """Final-cycle checkpoints among *paths*, resolved per directory.

    For callers that sweep a tree of run directories at once (dashboards,
    cross-experiment summaries) and need one end-of-training checkpoint per
    run rather than every intermediate snapshot. Grouping by parent matters:
    "the final cycle" is only meaningful within a single run's directory.

    Args:
        paths: Checkpoint paths, possibly spanning many run directories.

    Returns:
        The final-cycle checkpoints, in the input's directory order. A
        directory holding two attempts contributes both, since there is no
        basis here for preferring one.
    """
    by_directory: dict[Path, list[PostCycleCheckpoint]] = {}
    for path in paths:
        record = _classify(path)
        if record is not None:
            by_directory.setdefault(path.parent, []).append(record)
    return [
        record.path
        for records in by_directory.values()
        for record in final_cycle_candidates(records)
    ]


def _filter_by_run_token(
    candidates: Iterable[PostCycleCheckpoint], run_token: Optional[str]
) -> list[PostCycleCheckpoint]:
    if run_token is None:
        return list(candidates)
    return [c for c in candidates if c.run_token == run_token]


def _exactly_one(
    candidates: list[PostCycleCheckpoint], what: str, checkpoints_dir: Path
) -> PostCycleCheckpoint:
    """Return the sole candidate, or explain why the caller must choose."""
    if len(candidates) == 1:
        return candidates[0]

    tokens = sorted({c.run_token or "none" for c in candidates})
    listing = "\n  ".join(c.describe() for c in candidates)
    raise AmbiguousCheckpointError(
        f"Expected exactly one {what} in {checkpoints_dir}, found "
        f"{len(candidates)} across run token(s) {tokens}:\n  {listing}\n"
        f"This directory holds more than one attempt's output. Pass "
        f"run_token= (CLI: --run-token) to name the attempt you want, or "
        f"move each attempt into its own output directory."
    )


def _log_scope_tokens(
    candidates: Iterable[PostCycleCheckpoint], run_token: Optional[str]
) -> Optional[set[str]]:
    """Run tokens whose logs describe *candidates*, or ``None`` for every log.

    ``pattern=`` narrows which checkpoint *files* are in play; the logs have to
    be narrowed the same way or a sibling model's plan and cycle indices decide
    this model's fate. A flat directory holding several models is exactly what
    ``pattern=`` is for, so this is the common case, not a corner one. Taking
    the tokens off the files that survived filtering keeps one filter — the
    glob — rather than a second pattern-to-log mapping to maintain.

    Returns ``None`` when any candidate is untokenised (legacy naming, or a run
    whose logger was disabled) and so cannot be tied to a log, and when there
    are no candidates at all: in both cases narrowing would drop the very log
    that describes the directory.
    """
    if run_token is not None:
        return {run_token}
    tokens = {c.run_token for c in candidates}
    if not tokens or None in tokens:
        return None
    return {token for token in tokens if token is not None}


def read_cycle_bests(
    checkpoints_dir: Path,
    pattern: str = "*.zip",
    run_token: Optional[str] = None,
) -> dict[int, CycleBest]:
    """Read the logs in *checkpoints_dir* and merge their per-cycle bests.

    Reading only the newest log (the pre-#1011 behaviour) hides every cycle
    that finished before a resume, so a best cycle preceding the resume point
    was invisible.

    Args:
        checkpoints_dir: Directory whose ``logs/`` holds the run's JSONL logs.
        pattern: Glob restricting which checkpoints are considered; the logs
            read are those of the attempts that wrote them.
        run_token: Restrict to the log of this one attempt.

    Returns:
        Mapping from 0-indexed cycle to its ``CycleBest``, merged across logs.
        Empty when the logs record no completed cycle.

    Raises:
        FileNotFoundError: If no training logs exist under *checkpoints_dir*.
        AmbiguousCheckpointError: If two logs report the same cycle index,
            which means two attempts trained the same cycle in this directory
            and there is no principled way to merge their bests.
    """
    candidates = _filter_by_run_token(
        find_post_cycle_checkpoints(checkpoints_dir, pattern), run_token
    )
    return _read_cycle_bests_for_tokens(
        checkpoints_dir, _log_scope_tokens(candidates, run_token)
    )


def _read_cycle_bests_for_tokens(
    checkpoints_dir: Path, tokens: Optional[set[str]]
) -> dict[int, CycleBest]:
    """``read_cycle_bests`` with the log scope already resolved."""
    bests: dict[int, CycleBest] = {}
    for log_path in _logs_for_tokens(checkpoints_dir, tokens):
        run_id = log_path.name.removesuffix(LOG_EXTENSION)
        for entry in read_log(log_path):
            if entry.get("type") != "cycle_end":
                continue
            cycle = entry["cycle"]
            previous = bests.get(cycle)
            if previous is not None and previous.run_id != run_id:
                raise AmbiguousCheckpointError(
                    f"Cycle {cycle} was trained by more than one run in "
                    f"{checkpoints_dir}: {previous.run_id} reported "
                    f"best_val_loss={previous.best_val_loss} and {run_id} "
                    f"reported best_val_loss={entry['best_val_loss']}. Two "
                    f"attempts share this directory. Pass run_token= (CLI: "
                    f"--run-token) to name the attempt you want, or move each "
                    f"attempt into its own output directory."
                )
            bests[cycle] = CycleBest(cycle, entry["best_val_loss"], run_id)

    return bests


def _logs_for_tokens(checkpoints_dir: Path, tokens: Optional[set[str]]) -> list[Path]:
    """Log files under *checkpoints_dir*, narrowed to *tokens* when given.

    Raises:
        FileNotFoundError: If the directory holds no logs at all, or none
            belonging to *tokens*.
    """
    log_paths = find_logs(str(checkpoints_dir))
    if not log_paths:
        raise FileNotFoundError(f"No training logs found under {checkpoints_dir}")
    if tokens is None:
        return log_paths

    matching = [
        path
        for path in log_paths
        if run_token_from_run_id(path.name.removesuffix(LOG_EXTENSION)) in tokens
    ]
    if not matching:
        raise FileNotFoundError(
            f"No training log for run token(s) {sorted(tokens)} under "
            f"{checkpoints_dir}; found {[p.name for p in log_paths]}"
        )
    return matching


def _expected_num_cycles(
    checkpoints_dir: Path, tokens: Optional[set[str]]
) -> Optional[int]:
    """Largest ``num_cycles`` the in-scope logs started out intending.

    Scoping matters: another attempt's more ambitious plan says nothing about
    whether this one finished, and reading it would fail a complete 4-cycle run
    because an abandoned 8-cycle attempt shares the directory. Within the
    scope there is no basis for choosing among the plans, so the largest stands
    and an unfinished attempt fails loudly.

    Returns ``None`` — no opinion — when no log is available to read.
    """
    try:
        log_paths = _logs_for_tokens(checkpoints_dir, tokens)
    except FileNotFoundError:
        return None
    planned = [
        entry["num_cycles"]
        for log_path in log_paths
        for entry in read_log(log_path)
        if entry.get("type") == "training_start"
    ]
    return max(planned) if planned else None


def resolve_final_checkpoint(
    checkpoints_dir: Path,
    pattern: str = "*.zip",
    run_token: Optional[str] = None,
) -> SelectedCheckpoint:
    """Resolve the run's final-cycle post-cycle checkpoint.

    Args:
        checkpoints_dir: Directory a training run wrote its checkpoints to.
        pattern: Glob restricting which files are considered; the "did it
            finish?" check reads only the logs of the attempts that wrote
            them, so a sibling model's plan cannot condemn this one.
        run_token: Select only this attempt's checkpoints. Required when the
            directory holds more than one attempt.

    Returns:
        The final cycle's post-cycle checkpoint.

    Raises:
        FileNotFoundError: If no post-cycle checkpoint matches, or if the
            logs show the run never reached its last cycle.
        AmbiguousCheckpointError: If more than one candidate survives
            filtering.
    """
    candidates = _filter_by_run_token(
        find_post_cycle_checkpoints(checkpoints_dir, pattern), run_token
    )
    if not candidates:
        raise FileNotFoundError(
            f"No post-cycle checkpoint matching {pattern!r}"
            + (f" for run token {run_token}" if run_token else "")
            + f" in {checkpoints_dir}"
        )

    final_candidates = final_cycle_candidates(candidates)
    if not final_candidates:
        raise FileNotFoundError(
            f"{checkpoints_dir} holds only per-cycle checkpoints "
            f"({sorted(c.cycle for c in candidates if c.cycle is not None)}) "
            f"and no final-cycle save; the run did not finish."
        )
    scope = _log_scope_tokens(candidates, run_token)
    # Only meaningful when every candidate is numbered: a legacy unsuffixed
    # final save is proof the run finished, whatever its cycle count was.
    numbered = [c.cycle for c in final_candidates if c.cycle is not None]
    if len(numbered) == len(final_candidates):
        _check_run_reached_final_cycle(checkpoints_dir, max(numbered), scope)

    chosen = _exactly_one(final_candidates, "final-cycle checkpoint", checkpoints_dir)
    return SelectedCheckpoint(
        path=chosen.path,
        cycle=chosen.cycle,
        best_val_loss=_best_val_loss_for(checkpoints_dir, chosen.cycle, scope),
        run_token=chosen.run_token,
    )


def _check_run_reached_final_cycle(
    checkpoints_dir: Path, last_saved_cycle: int, tokens: Optional[set[str]]
) -> None:
    """Refuse to pass off a mid-run cycle as the run's final checkpoint.

    Before #1011 the final save was the only unsuffixed file, so its absence
    made an unfinished run fail loudly. Now that every post-cycle save is
    numbered, "highest cycle present" would quietly stand in for it.

    *last_saved_cycle* and *tokens* must describe the same attempts, or a
    neighbouring attempt's plan decides this one's fate.
    """
    expected = _expected_num_cycles(checkpoints_dir, tokens)
    if expected is None:
        return
    if last_saved_cycle < expected - 1:
        raise FileNotFoundError(
            f"Training in {checkpoints_dir} planned {expected} cycles but the "
            f"highest post-cycle checkpoint is cycle {last_saved_cycle}; the "
            f"run did not reach its final cycle. Resume it, or select an "
            f"earlier cycle explicitly."
        )


def _best_val_loss_for(
    checkpoints_dir: Path, cycle: Optional[int], tokens: Optional[set[str]]
) -> Optional[float]:
    """Best validation loss the logs recorded for *cycle*, if any.

    Purely informational for ``resolve_final_checkpoint``, so a directory
    whose logs can't be read or reconciled reports ``None`` rather than
    defeating an otherwise unambiguous resolution.
    """
    if cycle is None:
        return None
    try:
        bests = _read_cycle_bests_for_tokens(checkpoints_dir, tokens)
    except (FileNotFoundError, AmbiguousCheckpointError):
        return None
    record = bests.get(cycle)
    return record.best_val_loss if record else None


def select_best_checkpoint(
    checkpoints_dir: Path,
    pattern: str = "*.zip",
    run_token: Optional[str] = None,
) -> SelectedCheckpoint:
    """Select the checkpoint of the cycle with the lowest ``best_val_loss``.

    Args:
        checkpoints_dir: Directory a training run wrote its checkpoints to.
        pattern: Glob restricting which files are considered; the cycles
            compared are those of the attempts that wrote them, so a sibling
            model's cycle numbering does not collide with this one's.
        run_token: Select only this attempt's checkpoints.

    Returns:
        The post-cycle checkpoint of the globally best-validating cycle.

    Raises:
        FileNotFoundError: If no logs or no matching checkpoint exist.
        ValueError: If no cycle has completed yet.
        AmbiguousCheckpointError: If two runs share the directory, or more
            than one checkpoint matches the winning cycle.
    """
    candidates = _filter_by_run_token(
        find_post_cycle_checkpoints(checkpoints_dir, pattern), run_token
    )
    bests = _read_cycle_bests_for_tokens(
        checkpoints_dir, _log_scope_tokens(candidates, run_token)
    )
    if not bests:
        raise ValueError(
            f"No completed cycles found in the logs under {checkpoints_dir} — "
            f"the run hasn't finished a full cycle yet"
        )

    best = min(bests.values(), key=lambda record: record.best_val_loss)
    is_final_cycle = best.cycle == max(bests)
    for_cycle = [c for c in candidates if c.cycle == best.cycle]
    if not for_cycle and is_final_cycle:
        # A legacy run's final save carries no cycle number.
        for_cycle = [c for c in candidates if c.cycle is None]
    if not for_cycle:
        raise FileNotFoundError(
            f"No post-cycle checkpoint for cycle {best.cycle} matching "
            f"{pattern!r} in {checkpoints_dir}, though its log reports "
            f"best_val_loss={best.best_val_loss}"
        )

    chosen = _exactly_one(
        for_cycle, f"post-cycle checkpoint for cycle {best.cycle}", checkpoints_dir
    )
    return SelectedCheckpoint(
        path=chosen.path,
        cycle=best.cycle,
        best_val_loss=best.best_val_loss,
        run_token=chosen.run_token,
    )
