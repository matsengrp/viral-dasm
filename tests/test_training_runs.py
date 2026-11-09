# ruff: noqa: I001 -- import order in this file is load-bearing; see below.
"""The checks around a training run, and what each of them catches.

Bug-catching rather than invariant-stating. Each test below names a way M3
can go wrong that nothing else in the pipeline would notice:

- a template transcribed from the wrong arm, so five days of GPU produce a
  model the paper would report under another name;
- a corpus or a library default moving the never-gradient term, so the
  effective count the name is built from stops meaning what it meant;
- a run that exhausted its step budget rather than converging, which the
  trainer records by writing *nothing*;
- a card already carrying another user's work;
- two attempts sharing an output directory, so which weights are "the"
  selected ones depends on glob order.

The four CPU smokes are the integration test and run in the DAG rather than
here: they need a corpus, and every corpus is GISAID-derived.

GISAID: nothing here reads a corpus. The one run record used as a positive
control is eleven aggregate scalars, inlined.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import render_config
from lib import select_checkpoint
from runs import MAX_TRAIN_STEPS, MODELS, RUNS, run_id

import dasm2  # noqa: F401  (imported for its side effects; see test_import_order)

from check_params import never_gradient_params
from check_run import failures_for
from gpu_slot import cards_with_another_users_process
from read_run import BUDGET_STOP_REASON, replay

from dasm2.config import TrainingConfig
from dasm2.models.factory import create_model_from_config

#: A legal corpus length. Every corpus in ``config.yaml`` reports 1,273.
MAX_AA_SEQ_LEN = 1273

#: A cadence a 500,000-step budget can hold; nothing here trains.
CADENCE = 2000


def render(name: str, edit=None) -> dict:
    """One template, rendered and parsed, optionally with a line changed."""
    text = (REPO_ROOT / "configs" / f"{name}.template.yaml").read_text()
    if edit is not None:
        text = edit(text)
    values = render_config.substitutions(
        MAX_TRAIN_STEPS,
        eval_every_steps=CADENCE,
        max_aa_seq_len=MAX_AA_SEQ_LEN if "{MAX_AA_SEQ_LEN}" in text else None,
    )
    return yaml.safe_load(render_config.render(text, values))


def effective_params(raw: dict, max_aa_seq_len: int = MAX_AA_SEQ_LEN) -> int:
    """What ``scripts/check_params.py`` would report for this config."""
    config = TrainingConfig(**raw)
    model = create_model_from_config(config.model, device="cpu")
    allocated = sum(p.numel() for p in model.parameters())
    dead, _ = never_gradient_params(model, max_aa_seq_len)
    return allocated - dead


def test_the_run_list_is_seven_named_pairs():
    """Not the cross product: the three larger models run at full depth only."""
    assert len(RUNS) == len(set(RUNS)) == 7
    assert sorted({m for m, _ in RUNS}) == sorted(MODELS)
    thinned = {c for m, c in RUNS if c != "full_depth"}
    assert {m for m, c in RUNS if c in thinned} == {"dasm-85k"}
    with pytest.raises(KeyError, match="not one of the seven"):
        run_id("ci-484k", "cap_5000_0")


def test_a_wrong_geometry_is_caught_by_the_parameter_count():
    """The realistic failure: a template transcribed from the wrong arm.

    ``d_pos`` 8 against ``dasm-85k``'s 16 renders, validates and trains
    perfectly well. It is a different model, and the only thing that says so
    is the count the name is built from. (8 rather than 32 because
    ``EncoderConfig`` rejects ``d_pos >= d_model`` outright, and a geometry
    the config layer already refuses is not the one worth catching here.)
    """
    honest = render("dasm-85k")
    assert effective_params(honest) == MODELS["dasm-85k"]

    wrong = render("dasm-85k", lambda t: t.replace("d_pos: 16", "d_pos: 8"))
    assert effective_params(wrong) != MODELS["dasm-85k"]


def test_the_never_gradient_term_follows_the_corpus_and_the_table():
    """It is derived from two inputs, so moving either must move it.

    ``pe_max_len`` is a library default no template states, and the corpus
    length is read from the corpus's own footer. A term written down as a
    constant would survive a change to either and quietly misreport the
    count the model is named for.
    """
    raw = render("dasm-85k")
    config = TrainingConfig(**raw)
    model = create_model_from_config(config.model, device="cpu")
    d_pos = config.model.encoder.d_pos

    base, sources = never_gradient_params(model, MAX_AA_SEQ_LEN)
    assert sources, "nothing was identified as never-gradient"

    shorter, _ = never_gradient_params(model, MAX_AA_SEQ_LEN - 100)
    assert shorter - base == 100 * d_pos

    raw["model"]["encoder"]["pe_max_len"] = 6000
    wider = create_model_from_config(TrainingConfig(**raw).model, device="cpu")
    widened, _ = never_gradient_params(wider, MAX_AA_SEQ_LEN)
    assert widened - base == 1000 * d_pos


def test_ci_484k_builds_with_no_encoder_parameters():
    """The model that cannot be built without the restored identity encoder.

    `#5`'s prune deleted ``IdentityEncoder``, its config class and its
    registry entry, so ``type: IdentityEncoder`` was a ValidationError until
    they came back. The head's ``[1273, 20, 20]`` table is the whole model.
    """
    raw = render("ci-484k")
    config = TrainingConfig(**raw)
    assert config.model.encoder.type == "IdentityEncoder"

    model = create_model_from_config(config.model, device="cpu")
    assert sum(p.numel() for p in model.encoder.parameters()) == 0
    assert effective_params(raw) == MODELS["ci-484k"]


def test_every_template_reaches_its_name():
    """All four, against the counts `scripts/runs.py` names them by."""
    for model in MODELS:
        assert effective_params(render(model)) == MODELS[model], model


def test_render_refuses_a_value_the_template_does_not_use():
    """``--max-aa-seq-len`` against a transformer is the silent-truncation bug.

    In other trees it truncates every sequence and yields a plausible loss
    curve on a prefix of the gene. Here the value has nowhere to go, and
    ``render`` says so rather than dropping it.
    """
    text = (REPO_ROOT / "configs" / "dasm-85k.template.yaml").read_text()
    assert "{MAX_AA_SEQ_LEN}" not in text
    values = render_config.substitutions(
        MAX_TRAIN_STEPS, eval_every_steps=CADENCE, max_aa_seq_len=MAX_AA_SEQ_LEN
    )
    with pytest.raises(KeyError, match="MAX_AA_SEQ_LEN"):
        render_config.render(text, values)


#: One finished run, as ``scripts/read_run.py`` wrote it: ``dasm-85k`` at
#: full depth on one L40S, stopping on the learning-rate floor at step
#: 480,000 after 36.3 hours. Inlined rather than read from the artifact root,
#: which is not in this repository and which no test may require.
FLOOR_STOP_ROW = {
    "model": "dasm-85k",
    "corpus": "full_depth",
    "evaluations": 240,
    "steps_to_stop": 480000,
    "stop_reason": "LR 9.77e-07 < 1.00e-06",
    "final_lr": 9.765625e-07,
    "data_passes": 42.803638309256286,
    "wall_clock_s": 130777.16801598202,
    "best_val_loss_logged": 7.76630163192749,
    "best_step_logged": 472000,
    "best_val_loss_saved": 7.766360282897949,
}


def test_a_floor_stop_passes():
    """The positive control, so the test below cannot pass vacuously."""
    assert failures_for(FLOOR_STOP_ROW) == []


def test_a_budget_stop_fails():
    """The trainer writes no record at all when the budget ends a run.

    ``trainer.py`` breaks on ``self._cycle_step >= budget.max_train_steps``
    *before* ``_should_stop_early`` is called, and ``log_early_stopping``
    fires only inside the branch that call opens -- so there is no
    ``"type": "early_stopping"`` event and no step-budget reason anywhere in
    ``dasm2/training/logging.py``. A fixture naming a step-budget
    ``stop_reason`` would exercise a branch the trainer cannot reach; the
    signature is the *absence* of the event, at a last step on the budget.
    """
    capped = FLOOR_STOP_ROW | {
        "stop_reason": BUDGET_STOP_REASON,
        "steps_to_stop": MAX_TRAIN_STEPS,
    }
    (failure,) = failures_for(capped)
    assert "budget" in failure


def test_a_patience_stop_fails_even_though_its_message_says_lr():
    """The reason a prefix test replaces the substring test.

    ``_should_stop_early`` appends ``min_delta=...`` to the patience message,
    so a patience stop's reason can contain the letters ``LR`` -- and
    patience is checked *before* the floor, so a run that stops improving
    ends under a different rule than the one the paper reports.
    """
    patience = FLOOR_STOP_ROW | {
        "steps_to_stop": 300000,
        "stop_reason": (
            "no improvement for 30 evaluations (patience=30, min_delta=0.001)"
        ),
    }
    (failure,) = failures_for(patience)
    assert "not on the learning-rate floor" in failure


def test_a_coarser_threshold_keeps_an_earlier_best_and_can_stop_sooner():
    """What ``min_delta`` does, and all it does.

    It gates best-model tracking and the patience counter, and nothing else:
    the plateau scheduler steps on the raw ``val_loss``. So the same descent
    replayed at a coarser threshold keeps an earlier, worse best, and its
    patience counter advances faster. That is what makes the two reference
    campaigns' stopping points derived columns rather than separate runs.
    """
    # One drop, then a monotone descent of 1e-6 per evaluation whose whole
    # remaining travel (3.9e-5) stays inside 1e-4. Every evaluation counts at
    # min_delta 0; none after the drop counts at 1e-4.
    steps = [2000 * (i + 1) for i in range(40)]
    losses = [8.0] + [7.5 - 1e-6 * i for i in range(1, 40)]

    fine = replay(losses, steps, 0.0)
    coarse = replay(losses, steps, 1e-4)

    assert fine["best_step"] == steps[-1]
    assert coarse["best_step"] == steps[1]
    assert coarse["best_val_loss"] > fine["best_val_loss"]
    # The floor would still be the finer run's stopping rule; the coarser one
    # ends 30 evaluations after its last accepted improvement instead.
    assert fine["patience_stop_step"] is None
    assert coarse["patience_stop_step"] == steps[31]


def test_a_card_with_another_users_process_is_excluded():
    """One lock-free card, one held by somebody else."""
    uuids = ["GPU-aaa", "GPU-bbb"]
    mine = str(Path("/proc/self").stat().st_uid)
    # pid 1 is init, owned by root; on any host this test runs on that is a
    # different user from the one running pytest.
    assert cards_with_another_users_process(["GPU-bbb, 1"], uuids, mine) == {1}


def test_a_card_whose_occupant_cannot_be_identified_is_excluded():
    """A pid in another namespace resolves to nothing, and that is enough.

    The alternative is launching a multi-day run onto a card somebody else is
    using; the cost of being wrong the other way is one queued run.
    """
    uuids = ["GPU-aaa", "GPU-bbb"]
    assert cards_with_another_users_process(["GPU-aaa, 999999999"], uuids, "0") == {0}


def test_an_unanswerable_nvidia_smi_excludes_every_card():
    """Nothing has been shown to be free, so nothing is taken."""
    assert cards_with_another_users_process(None, ["GPU-aaa", "GPU-bbb"], "0") == {
        0,
        1,
    }


def _touch_checkpoint(directory: Path, stem: str) -> Path:
    """A file named the way the trainer names a cycle's best checkpoint."""
    path = directory / f"{stem}_cycle0_best_run20260101-000000-aaaaaa.zip"
    with zipfile.ZipFile(path, "w"):
        pass
    return path


def test_checkpoint_selection_raises_on_zero_and_on_two(tmp_path: Path):
    """Two attempts in one directory is how two bests appear.

    Which one is "the" selected checkpoint is then a question about glob
    order, so it is raised rather than answered -- the failure mode that once
    cost a long run its best-validation weights.
    """
    with pytest.raises(FileNotFoundError, match="cycle0_best"):
        select_checkpoint(tmp_path)

    first = _touch_checkpoint(tmp_path, "a")
    assert select_checkpoint(tmp_path) == first

    _touch_checkpoint(tmp_path, "b")
    with pytest.raises(RuntimeError, match="2 best checkpoints"):
        select_checkpoint(tmp_path)


def test_the_readout_row_carries_every_column_the_table_writes(tmp_path: Path):
    """``runs_table`` refuses a row that does not match its column list.

    The committed table is the paper's record of these runs, so a field
    added to ``read_run.py`` must reach it deliberately rather than by
    appearing in a dict.
    """
    from runs_table import COLUMNS

    rows_dir = tmp_path / "readout"
    rows_dir.mkdir()
    for model, corpus_name in RUNS:
        row = dict.fromkeys(COLUMNS, 0)
        row["model"], row["corpus"] = model, corpus_name
        (rows_dir / f"{run_id(model, corpus_name)}.json").write_text(json.dumps(row))

    # One row grows a field nobody added to the table.
    stray = rows_dir / f"{run_id(*RUNS[0])}.json"
    row = json.loads(stray.read_text()) | {"best_val_loss_at_min_delta_1e5": 0}
    stray.write_text(json.dumps(row))

    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "runs_table.py"),
            "--rows-dir",
            str(rows_dir),
            "--output",
            str(tmp_path / "runs.csv"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "best_val_loss_at_min_delta_1e5" in result.stderr
