"""Where per-site neutral rates come from, for training and for evaluation.

Issue #1299. A Parquet corpus either carries per-site ``nt_rates``/``nt_csps``
columns or it does not.

- **Column-carrying file:** the loader reads the columns, whatever the config
  names (DESIGN.md D8).
- **Rate-free file:** the configured neutral model computes the rates at load
  time, through ``ComputeNtRatesAndCsps``.

Either way, the file's columns decide the source, and the file never names
its own model.

The checkpoint records the *configured* model, which need not be the one that
produced training's rates. So it also records a ``NeutralRatesSource``, and
``check_evaluation_rates_source`` compares that record with the evaluation
file. Without the check, a checkpoint trained on columns could be scored
against a configured model on a rate-free twin, or a checkpoint trained on
the fly could be scored against another model's stored columns.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from dasm2.data.segment_handlers import SegmentHandler
    from dasm2.neutral.base import RatesCSPNeutralModel

logger = logging.getLogger(__name__)

NEUTRAL_RATE_COLUMN_BASES = ("nt_rates", "nt_csps")
"""Base names of the per-site neutral columns; segment suffixes are appended."""


class NeutralRatesSource(BaseModel):
    """Where a checkpoint's training rates came from.

    Runtime state, not config, so it sits beside ``training_config`` in the
    checkpoint metadata rather than inside it.

    Attributes:
        kind: ``"parquet_columns"`` when any training dataset read stored
            columns. ``"neutral_model"`` when every training dataset computed
            its rates from the configured model, as on a CSV corpus or a
            rate-free Parquet corpus.
        neutral_model_names: For ``parquet_columns``, the distinct
            ``neutral_model_name`` stamps of the files read. Informational
            only: a stamp never resolves a model.
    """

    kind: Literal["parquet_columns", "neutral_model"]
    neutral_model_names: List[str] = []

    model_config = ConfigDict(extra="forbid")


def combine_neutral_rates_sources(
    sources: Iterable[Optional[NeutralRatesSource]],
) -> Optional[NeutralRatesSource]:
    """Combine per-dataset rate sources into one record for the checkpoint.

    Any dataset that read columns makes the whole checkpoint
    ``parquet_columns``. That is the conservative choice: evaluation then
    refuses a rate-free file, rather than trusting a configured model that at
    least one task never trained against.

    Args:
        sources: One entry per training dataset. ``None`` means the dataset
            consumes no per-site rates (e.g. MLM).

    Returns:
        The combined record, or None if no dataset consumed rates.
    """
    known = [s for s in sources if s is not None]
    if not known:
        return None
    column_sources = [s for s in known if s.kind == "parquet_columns"]
    if not column_sources:
        return NeutralRatesSource(kind="neutral_model")
    names = sorted({n for s in column_sources for n in s.neutral_model_names})
    return NeutralRatesSource(kind="parquet_columns", neutral_model_names=names)


def _file_carries_neutral_columns(
    columns: set,
    path: Path,
    segment_handler: "SegmentHandler",
) -> bool:
    """Whether one Parquet file carries the neutral columns for its segments.

    Only segments whose ``parent`` column is present count. A heavy-only
    file under a paired handler has no light segment to carry rates for.

    Raises:
        ValueError: If the file has some of the columns but not all of them.
    """
    expected = [
        segment_handler.get_segment_column_name(base, seg_idx)
        for seg_idx in range(segment_handler.get_num_segments())
        if segment_handler.get_segment_column_name("parent", seg_idx) in columns
        for base in NEUTRAL_RATE_COLUMN_BASES
    ]
    found = [c for c in expected if c in columns]
    if not found:
        return False
    if len(found) == len(expected):
        return True
    missing = [c for c in expected if c not in columns]
    raise ValueError(
        f"Parquet file {path} has only some of its neutral rate columns: "
        f"found {found}, missing {missing}. A file must carry nt_rates and "
        "nt_csps for every segment it has, or none of them (a rate-free "
        "file, whose rates the configured neutral model computes at load)."
    )


def parquet_carries_neutral_columns(
    paths: Sequence[Path],
    file_column_names: Sequence[Sequence[str]],
    segment_handler: "SegmentHandler",
) -> bool:
    """Whether a Parquet spec carries per-site ``nt_rates``/``nt_csps``.

    Args:
        paths: The spec's files.
        file_column_names: Each file's schema column names, same order.
        segment_handler: Defines the segment suffixes.

    Returns:
        True if every file carries the columns, False if none does.

    Raises:
        ValueError: If a file carries only some of its columns, or if the
            files disagree.
    """
    per_file = [
        _file_carries_neutral_columns(set(names), path, segment_handler)
        for path, names in zip(paths, file_column_names, strict=True)
    ]
    if all(per_file):
        return True
    if not any(per_file):
        return False
    with_columns = [str(p) for p, has in zip(paths, per_file) if has]
    without_columns = [str(p) for p, has in zip(paths, per_file) if not has]
    raise ValueError(
        "Parquet spec mixes files with and without nt_rates/nt_csps columns, "
        "so one run would read rates from two sources. With columns: "
        f"{with_columns}. Without: {without_columns}. Package every file "
        "the same way."
    )


def _describe_neutral_model(neutral_model: Optional["RatesCSPNeutralModel"]) -> str:
    """Render a neutral model's config for an error or warning message."""
    if neutral_model is None:
        return "none"
    return str(neutral_model.to_config().model_dump(mode="json", exclude_none=True))


def check_evaluation_rates_source(
    checkpoint_source: Optional[NeutralRatesSource],
    eval_carries_columns: bool,
    neutral_model: Optional["RatesCSPNeutralModel"],
    eval_neutral_model_names: Sequence[str],
) -> None:
    """Refuse or flag an evaluation whose rates differ from training's source.

    - A ``parquet_columns`` checkpoint on a rate-free file raises. The
      rates would come from the configured model, which need not match the
      columns the checkpoint trained on.
    - A checkpoint without the field predates rate-free files, so its
      training rates came from columns or CSV. It is treated like
      ``parquet_columns`` for this check.
    - A ``neutral_model`` checkpoint on a column-carrying file warns, since
      the file's columns win. It warns rather than raises, so CSV-trained
      checkpoints stay evaluable on column-carrying parquets built with the
      same model.

    Args:
        checkpoint_source: The checkpoint's recorded source, or None.
        eval_carries_columns: Whether the evaluation file has rate columns.
        neutral_model: The checkpoint's neutral model, or None. Its config
            is described in the message.
        eval_neutral_model_names: The evaluation files' ``neutral_model_name``
            stamps, for the message.

    Raises:
        ValueError: On a column-trained (or unrecorded) checkpoint evaluated
            on a rate-free file.
    """
    if not eval_carries_columns:
        if checkpoint_source is None or checkpoint_source.kind == "parquet_columns":
            trained_on = (
                "an unrecorded source (the checkpoint predates "
                "neutral_rates_source, so its rates came from Parquet columns "
                "or CSV)"
                if checkpoint_source is None
                else "Parquet columns stamped "
                f"{checkpoint_source.neutral_model_names or ['<unstamped>']}"
            )
            raise ValueError(
                "Refusing to evaluate on a rate-free Parquet file: its rates "
                f"would come from the checkpoint's configured neutral model "
                f"({_describe_neutral_model(neutral_model)}), but training read "
                f"{trained_on}. "
                "Those need not be the same rates. Evaluate on a file packaged "
                "with the same rate columns, or retrain on a rate-free file."
            )
        return
    if checkpoint_source is not None and checkpoint_source.kind == "neutral_model":
        logger.warning(
            "Evaluating a checkpoint whose training rates came from its "
            "configured neutral model (%s) on a Parquet file that carries its "
            "own nt_rates/nt_csps columns (stamped %s). The file's columns "
            "win, so evaluation uses them, not the model training used. "
            "Evaluate on a rate-free file to score against the model.",
            _describe_neutral_model(neutral_model),
            list(eval_neutral_model_names) or ["<unstamped>"],
        )
