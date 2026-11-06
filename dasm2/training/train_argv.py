"""Shared builder for the `dasm2 train` argv.

Experiment Snakefiles historically duplicated a `_train_argv` helper that
assembles the CLI invocation for `dasm2 train`. This module factors out the
stable subset of flags that every one of those helpers passed identically,
so experiments no longer need to copy-paste (or drift on) the same argv
construction. Experiment-specific flags beyond this subset (e.g.
`--multi_gpu`, `--neutral_model_name`) are appended by the caller after
calling `build_train_argv`.
"""

from typing import Optional


def build_train_argv(
    config_file: str,
    head: str,
    dataset: str,
    output_dir: str,
    *,
    seed: int,
    cycles: int,
    max_train_steps: int,
    eval_every_steps: int,
    max_eval_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    val_batch_size: int,
    early_stopping_patience: int,
    notes: str,
    device: str,
    cv_num_folds: Optional[int] = None,
    cv_fold: Optional[int] = None,
    cv_group_col: Optional[str] = None,
    checkpoint_every_n_evals: Optional[int] = None,
) -> list[str]:
    """Build the argv for `dasm2 train`, over the stable flag subset every
    experiment Snakefile passes identically. `cv_*` and `checkpoint_every_n_evals` are
    appended only when not None (a run whose split comes from a file, not
    `--cv_*`, omits them).
    Experiment-specific flags beyond this subset are appended by the caller
    after calling this function -- this function does not try to cover
    every `train_model_from_datasets` kwarg.

    Raises:
        ValueError: If only some of `cv_num_folds`, `cv_fold` and `cv_group_col`
            are given -- a cross-validation run needs all three.
    """
    cv_given = [x is not None for x in (cv_num_folds, cv_fold, cv_group_col)]
    if any(cv_given) and not all(cv_given):
        raise ValueError(
            "cv_num_folds, cv_fold and cv_group_col must be given together or "
            "all omitted."
        )
    argv = [
        "dasm2",
        "train",
        config_file,
        f'{{"{head}": "{dataset}"}}',
        "--output_dir",
        output_dir,
        "--device",
        device,
        "--seed",
        str(seed),
        "--cycles",
        str(cycles),
        "--max_train_steps",
        str(max_train_steps),
        "--eval_every_steps",
        str(eval_every_steps),
        "--max_eval_steps",
        str(max_eval_steps),
        "--batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--val_batch_size",
        str(val_batch_size),
        "--early_stopping_patience",
        str(early_stopping_patience),
    ]
    if cv_num_folds is not None:
        argv += ["--cv_num_folds", str(cv_num_folds)]
    if cv_fold is not None:
        argv += ["--cv_fold", str(cv_fold)]
    if cv_group_col is not None:
        argv += ["--cv_group_col", cv_group_col]
    argv += ["--notes", notes]
    if checkpoint_every_n_evals is not None:
        argv += ["--checkpoint_every_n_evals", str(checkpoint_every_n_evals)]
    return argv
