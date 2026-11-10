import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # The paper's seven runs: loss curves

    Validation and training curves for the seven `(model, corpus)` runs of
    `workflow/train.smk` — four architectures at full depth, and `dasm-85k`
    at each of the three thinned depths.

    This is a **monitoring** notebook. It reads whatever each run has logged
    so far and omits the runs that have not started, so it is readable
    mid-campaign. Re-run the cells to refresh.

    ## What is comparable here, and what is not

    **The four models at `full_depth` are directly comparable.** They train
    on the same corpus, hold out the same rows (`cv_split` in `config.yaml`),
    and differ only in architecture — so a difference between two of their
    validation curves at the same step is a paired difference on the same
    data. That is the architecture ladder, and it is the comparison panel A
    is for.

    **The four depths are not comparable by loss.** Each thinned corpus is
    its own Parquet with its own fold 0, so the four hold out *different*
    validation rows; the status table below gives each one's count.
    A lower loss on a smaller corpus is partly a statement about which rows
    were held out. Panel B plots them because the shape of each descent is
    worth seeing, not because the levels can be read against each other; the
    depth axis is read through DMS, as the sampling-depth ladder did.

    **The training curve is a different quantity from the validation curve,
    and the gap between them is not a generalization gap.** `avg_train_loss`
    is a running mean over training batches taken **with dropout active** and
    with the weights moving across the averaging window; `val_loss` is a
    clean eval-mode pass at a fixed point. Panel C plots the logged figure so
    the descent is visible, but the comparable number is
    `train_loss_eval_mode`, which the `train_eval` rule computes with a
    separate eval-mode pass over the training split once a run finishes. It
    lands in `results/runs.csv`, not here.

    **GISAID**: reads training logs under the artifact root. Keep any export
    of this notebook on the cluster.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import altair as alt
    import pandas as pd

    # `runs` owns the run list and `lib` the artifact root, so this notebook
    # cannot name a run the workflow did not train, or miss one it did.
    REPO_ROOT = Path(__file__).resolve().parent.parent
    for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

    from lib import ARTIFACT_ROOT
    from runs import MODELS, RUNS, run_id

    from dasm2.training.log_reader import get_run_eval_losses

    RUN_DIR = Path(ARTIFACT_ROOT) / "runs"
    return MODELS, RUNS, RUN_DIR, alt, get_run_eval_losses, pd, run_id


@app.cell
def _(MODELS, RUNS, RUN_DIR, get_run_eval_losses, pd, run_id):
    def _curve(model, corpus):
        """One run's evaluations, or None if it has logged none yet.

        `get_run_eval_losses` dedupes on `(cycle, cycle_step)`, later wins,
        which is what makes a resumed run's replayed evaluations count once.
        """
        directory = RUN_DIR / f"{model}_{corpus}"
        if not sorted(directory.glob("logs/*_log.jsonl")):
            return None
        frame = get_run_eval_losses(directory)
        if frame.empty:
            return None
        frame = frame.sort_values("global_step").copy()
        frame["run"] = run_id(model, corpus)
        frame["model"] = model
        frame["corpus"] = corpus
        frame["effective_params"] = MODELS[model]
        return frame

    FRAMES = {(m, c): _curve(m, c) for m, c in RUNS}
    MISSING = [run_id(m, c) for (m, c), f in FRAMES.items() if f is None]
    CURVES = (
        pd.concat([f for f in FRAMES.values() if f is not None], ignore_index=True)
        if any(f is not None for f in FRAMES.values())
        else pd.DataFrame()
    )
    return CURVES, MISSING


@app.cell
def _(CURVES, MISSING, RUNS, mo):
    mo.md(
        "**No run has logged an evaluation yet.**"
        if CURVES.empty
        else (
            f"{CURVES['run'].nunique()} of {len(RUNS)} runs have logged evaluations."
            + (f" Not started: {', '.join(MISSING)}." if MISSING else " All of them.")
        )
    )
    return


@app.cell
def _(CURVES, mo, pd):
    def _status(frame):
        rows = []
        for run, part in frame.groupby("run"):
            best = part.loc[part["val_loss"].idxmin()]
            last = part.iloc[-1]
            rows.append(
                {
                    "run": run,
                    "params": f"{int(part['effective_params'].iloc[0]):,}",
                    "evals": len(part),
                    "last step": f"{int(last['global_step']):,}",
                    "best val": f"{float(best['val_loss']):.6f}",
                    "@ step": f"{int(best['global_step']):,}",
                    "lr": f"{float(last['lr']):.2e}",
                    "data passes": f"{float(last['data_passes']):.1f}",
                }
            )
        return pd.DataFrame(rows)

    mo.ui.table(_status(CURVES)) if not CURVES.empty else mo.md("")
    return


@app.cell
def _(mo):
    ZOOM = mo.ui.checkbox(
        value=True, label="zoom the y-axis to the plotted range (hide zero)"
    )
    ZOOM
    return (ZOOM,)


@app.cell
def _(ZOOM, alt):
    def curve(frame, field, title, colour_by, colour_title):
        scale = alt.Scale(zero=False) if ZOOM.value else alt.Scale()
        return (
            alt.Chart(frame)
            .mark_line()
            .encode(
                x=alt.X("global_step:Q", title="optimizer step"),
                y=alt.Y(f"{field}:Q", title=title, scale=scale),
                color=alt.Color(f"{colour_by}:N", sort=None, title=colour_title),
                tooltip=["run:N", "global_step:Q", f"{field}:Q", "lr:Q"],
            )
            .properties(width=640, height=320)
        )

    return (curve,)


@app.cell
def _(CURVES, curve, mo):
    mo.md("## Panel A — the architecture ladder, at full depth")
    return


@app.cell
def _(CURVES, curve):
    _full = CURVES[CURVES["corpus"] == "full_depth"] if not CURVES.empty else CURVES
    (
        curve(_full, "val_loss", "validation loss (nats)", "model", "model")
        if not _full.empty
        else None
    )
    return


@app.cell
def _(mo):
    mo.md(
        "## Panel B — `dasm-85k` across the four depths\n\n"
        "Different held-out rows per corpus, so read the shape of each "
        "descent, not the levels against each other."
    )
    return


@app.cell
def _(CURVES, curve):
    _ladder = CURVES[CURVES["model"] == "dasm-85k"] if not CURVES.empty else CURVES
    (
        curve(_ladder, "val_loss", "validation loss (nats)", "corpus", "corpus")
        if not _ladder.empty
        else None
    )
    return


@app.cell
def _(mo):
    mo.md(
        "## Panel C — the logged training curve\n\n"
        "`avg_train_loss`, a running mean taken with dropout active. Not the "
        "quantity the gap is computed from; see the header."
    )
    return


@app.cell
def _(CURVES, curve):
    (
        curve(CURVES, "avg_train_loss", "logged avg_train_loss (nats)", "run", "run")
        if not CURVES.empty
        else None
    )
    return


if __name__ == "__main__":
    app.run()
