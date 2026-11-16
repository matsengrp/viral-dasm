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
    # DNMS against the BA.1 spike DMS

    **Exploratory, and not part of the pipeline.** Nothing here is a declared
    output, no rule produces it and no figure reads it. It exists to answer one
    question before an arm is built: on the one background DNMS was trained on,
    how well do its quantities predict a deep mutational scan?

    **What DNMS is.** Elkin & Zhu, *Commun Biol* 8 (2025) — ProtBERT fine-tuned
    on NCBI spike proteins. It reports three per-mutation quantities and one
    combination of them:

    | quantity | what it is |
    |---|---|
    | grammaticality | the model's probability for the mutant residue at that position |
    | semantic change | L2 distance between the parent's and the mutant's embeddings |
    | attention change | change in the attention matrix |
    | **DNMS rank-sum** | `rank(1.5 * rank(-change) + 3.0 * rank(prob) + 1.0 * rank(-attn))` |

    The weights are the authors' own CLI defaults (`bin/ParentChildMutate.py`),
    not `bin/MutationRankingResults.py`'s, which default to 1/1/1. **Every term
    is a rank within one parent's candidate set**, so the combination has no
    units and no zero — which is why it is drawn here before anyone decides
    what column it would fill.

    **Why BA.1 and only BA.1.** DNMS's language model is trained on sequences
    collected before **1 January 2022** (the corpus runs to January 2023, but
    those later sequences supply test mutations, not training). Of this paper's
    three DMS parents, only BA.1's construct is in that training set — checked
    by string membership against the released `train_seq.pkl`. XBB.1.5 appears
    once in the whole corpus and never in training; KP.3.1.1's lineage is
    absent entirely. So BA.1 is DNMS's best case, and the other two would
    measure something else. Even here the background is thin: 233 of the 6,256
    training sequences are clade 21K, all collected in the variant's first
    seven weeks.

    **It is also free.** The three released dictionaries already carry all
    24,073 substitutions on this parent — 1,267 positions x 19 — so nothing
    anywhere in this analysis loads a checkpoint or runs a forward pass.

    **The population is the pipeline's own.** `dms_models.homolog_cells("BA.1")`
    applies the training mask, so these numbers sit beside
    `results/panels/dms/predictor_correlations.csv` rather than beside a raw
    join of the published CSV.

    **GISAID**: reads published DMS tables and a third-party model's released
    value dictionaries. No sequence and no per-site count passes through here.
    """)


@app.cell
def _():
    import sys
    from pathlib import Path

    HERE = Path(__file__).resolve().parent
    REPO_ROOT = HERE.parents[1]

    #: **Precomputed, on purpose.** DNMS's three released dictionaries are
    #: 772 MB and hold about 8.7 million floats; the opcode gate that vets them
    #: before unpickling walks every byte in pure Python, which is about a
    #: minute of work. `marimo run` re-executes its notebook for every browser
    #: session, so that belongs in `prep_dnms_ba1.py` -- which did the gating
    #: and the mapping once -- and this reads its 1 MB output. Run that script
    #: first; this notebook is deliberately not able to rebuild it.
    SCORES = HERE / "dnms_ba1_scores.parquet"

    sys.path.insert(0, str(REPO_ROOT / "scripts"))

    # `lib` before anything heavy, as every script in this repository does:
    # `dasm2` is imported for its side effects and must come before pandas,
    # which pulls pyarrow in transitively.
    import lib
    from lib import assert_dasm2_is_this_repo

    import dasm2  # noqa: F401

    assert_dasm2_is_this_repo()

    import altair as alt
    import dms
    import dms_antigen
    import dms_models
    import pandas as pd
    from scipy.stats import rankdata, spearmanr

    # The faceted grid draws every measured cell four times over and again for
    # the union column, which is past Altair's default row ceiling.
    alt.data_transformers.disable_max_rows()

    return (
        REPO_ROOT,
        SCORES,
        alt,
        dms,
        dms_antigen,
        dms_models,
        lib,
        pd,
        rankdata,
        spearmanr,
    )


@app.cell
def _(SCORES, dms_antigen, lib, pd):
    scores = pd.read_parquet(SCORES)

    # The construct, for the summary below and for the stratification further
    # down. `prep_dnms_ba1.py` already used it to map DNMS's gap-stripped
    # positions onto WH1 sites; reading it again here is a FASTA parse, not a
    # second source of truth.
    gapped = dms_antigen.read_parents(lib.antigen_model().parents)["BA.1"]
    return gapped, scores


@app.cell
def _(dms_models, scores):
    cells, founder = dms_models.homolog_cells("BA.1")
    merged = cells.merge(scores, on=["aa_site", "mutant_aa"], how="inner")

    # **The coordinate map's only live guard.** `homolog_cells` takes
    # `parent_aa` from the clade founder; DNMS's key spells its own. A map that
    # is off by one lands on a different residue, and here that is a hard stop
    # rather than a quietly worse correlation.
    _disagreements = merged[merged["parent_aa"] != merged["dnms_parent_aa"]]
    if len(_disagreements):
        raise ValueError(f"{len(_disagreements)} cells disagree on the parent residue")
    if len(merged) != len(cells):
        raise ValueError(f"joined {len(merged)} of {len(cells)} measured cells")
    return cells, founder, merged


@app.cell
def _(cells, founder, gapped, merged, mo, scores):
    mo.md(
        f"**The join.** DNMS scores {len(scores):,} substitutions on this parent "
        f"({scores['aa_site'].nunique():,} positions x 19, from a "
        f"{len(gapped):,}-column construct with {gapped.count('-')} deletions). "
        f"All {len(merged):,} of `homolog_cells('BA.1')`'s {len(cells):,} "
        f"measured cells join, with **0** parent-residue disagreements — which "
        f"is the gap-stripped-to-WH1 map verifying itself. Founder length "
        f"{len(founder):,}."
    )


@app.cell
def _(dms, merged, pd, spearmanr):
    #: The four series, in the order the panels read top to bottom: the
    #: combination first, then the three terms it is built from.
    SERIES = {
        "DNMS rank-sum": "dnms",
        "grammaticality, log p": "log_grammaticality",
        "semantic change": "semantic_change",
        "attention change": "attention_change",
    }

    _records = []
    for _label, _column in SERIES.items():
        for _stratum in (dms.ALL, dms.SINGLE_NT, dms.MULTI_NT):
            _part = (
                merged if _stratum == dms.ALL else merged[merged["stratum"] == _stratum]
            )
            _records.append(
                {
                    "series": _label,
                    "stratum": _stratum,
                    "n": len(_part),
                    "pearson_r": _part["dms_cell_entry"].corr(_part[_column]),
                    "spearman_r": spearmanr(
                        _part["dms_cell_entry"], _part[_column]
                    ).statistic,
                }
            )
    correlations = pd.DataFrame(_records)
    return SERIES, correlations


@app.cell
def _(correlations, mo):
    mo.md(
        "## Correlation with BA.1 cell entry\n\n"
        + correlations.to_markdown(
            index=False, floatfmt=("", "", ".0f", "+.4f", "+.4f")
        )
        + "\n\n**The two negative signs are the expected direction**: a larger "
        "embedding or attention disruption goes with lower fitness. The "
        "rank-sum flips both internally, so all four series point the same way."
    )


@app.cell
def _(REPO_ROOT, mo, pd):
    _committed = pd.read_csv(
        REPO_ROOT / "results/panels/dms/predictor_correlations.csv"
    )
    _ba1 = _committed[_committed["homolog"] == "BA.1"]
    committed_ba1 = _ba1.pivot(index="predictor", columns="stratum", values="pearson_r")
    mo.md(
        "## For scale: what the pipeline already reports on BA.1\n\n"
        "Pearson *r*, from `results/panels/dms/predictor_correlations.csv` as "
        "committed. `dasm_prob` and `neutral_rate` are defined on the "
        "single-nucleotide stratum alone, so their `multi_nt` cells are empty "
        "by declaration.\n\n" + committed_ba1.to_markdown(floatfmt="+.4f")
    )
    return (committed_ba1,)


@app.cell
def _(SERIES, dms, merged, pd):
    # **Three columns and nothing else, so every point can be drawn.** Altair
    # inlines its data into the Vega spec, and the grid draws every measured
    # cell four times over and again for the union column -- so a per-row label
    # string or a tooltip field nobody reads is tens of thousands of copies of
    # itself. The first version carried eight columns and produced a 13 MB
    # output that marimo refused. Nothing is subsampled: the coefficients live
    # in the table above instead, which is where a reader looks for a number
    # anyway.
    _long = merged.melt(
        id_vars=["dms_cell_entry", "stratum"],
        value_vars=list(SERIES.values()),
        var_name="column",
        value_name="prediction",
    )
    _long["series"] = _long["column"].map({v: k for k, v in SERIES.items()})
    _long = _long.drop(columns=["column"])
    _long["prediction"] = _long["prediction"].round(4)
    _long["dms_cell_entry"] = _long["dms_cell_entry"].round(4)

    # `all` is the union rather than a third stratum, so it is drawn by
    # duplicating every row into its own column of the grid.
    faceted = pd.concat([_long, _long.assign(stratum=dms.ALL)], ignore_index=True)
    return (faceted,)


@app.cell
def _(alt, dms, faceted):
    grid = (
        alt.Chart(faceted)
        .mark_circle(size=6, opacity=0.25, color="#0072B2")
        .encode(
            # The measurement on x, the model on y: the prediction is what is
            # being judged, and a reader scans a row of panels asking how each
            # model quantity responds to the same assay.
            x=alt.X(
                "dms_cell_entry:Q", title="DMS cell entry", scale=alt.Scale(zero=False)
            ),
            y=alt.Y(
                "prediction:Q", title=None, scale=alt.Scale(zero=False, nice=False)
            ),
        )
        .properties(width=200, height=175)
        .facet(
            column=alt.Column(
                "stratum:N",
                title=None,
                sort=[dms.ALL, dms.SINGLE_NT, dms.MULTI_NT],
                header=alt.Header(labelFontSize=13, labelFontWeight="bold"),
            ),
            row=alt.Row(
                "series:N",
                title=None,
                sort=[
                    "DNMS rank-sum",
                    "grammaticality, log p",
                    "semantic change",
                    "attention change",
                ],
                header=alt.Header(labelFontSize=12),
            ),
        )
        # The four series carry four different units, and they are now on y --
        # so it is y that varies by row. Every panel shares one DMS axis, which
        # is what makes a column comparable down its length.
        .resolve_scale(y="independent")
    )
    grid
    return (grid,)


@app.cell
def _(SERIES, dms, mo):
    series_pick = mo.ui.dropdown(
        options=list(SERIES), value="DNMS rank-sum", label="Series"
    )
    stratum_pick = mo.ui.dropdown(
        options=[dms.ALL, dms.SINGLE_NT, dms.MULTI_NT], value=dms.ALL, label="Stratum"
    )
    mo.md(
        "## One panel, larger\n\nThe same data, for reading a single cloud "
        "rather than comparing twelve."
    )
    return series_pick, stratum_pick


@app.cell
def _(mo, series_pick, stratum_pick):
    mo.hstack([series_pick, stratum_pick], justify="start", gap=2)


@app.cell
def _(SERIES, alt, dms, merged, series_pick, stratum_pick):
    # Built from `merged` rather than from `faceted`, so the site and the
    # mutant residue are available for tooltips here without riding along in
    # the grid's inlined data. One stratum of one series is at most 5,413 rows.
    _column = SERIES[series_pick.value]
    _one = (
        merged
        if stratum_pick.value == dms.ALL
        else merged[merged["stratum"] == stratum_pick.value]
    )
    _one = _one[
        ["aa_site", "parent_aa", "mutant_aa", "dms_cell_entry", _column]
    ].rename(columns={_column: "prediction"})
    zoom = (
        alt.Chart(_one)
        .mark_circle(size=14, opacity=0.3, color="#0072B2")
        .encode(
            x=alt.X(
                "dms_cell_entry:Q", title="DMS cell entry", scale=alt.Scale(zero=False)
            ),
            y=alt.Y(
                "prediction:Q",
                title=series_pick.value,
                scale=alt.Scale(zero=False, nice=False),
            ),
            tooltip=[
                "aa_site:Q",
                "parent_aa:N",
                "mutant_aa:N",
                "prediction:Q",
                "dms_cell_entry:Q",
            ],
        )
        .properties(width=520, height=420)
    )
    zoom
    return (zoom,)


@app.cell
def _(dms, lib, scores):
    # **Where does codon accessibility sit in DNMS's own ranking?** The rank is
    # defined over this parent's whole candidate set, so the question is asked
    # there rather than on the measured subset. `single_nt_indicator` is
    # codon-aware, read from the founder's actual nucleotides rather than from
    # which mutations a counts table happens to carry.
    _indicator = dms.single_nt_indicator(lib.dms_dataset("BA.1").clade)
    candidates = scores.copy()
    candidates["stratum"] = [
        dms.SINGLE_NT if bool(_indicator.loc[mutant, site - 1]) else dms.MULTI_NT
        for site, mutant in zip(candidates["aa_site"], candidates["mutant_aa"])
    ]
    return (candidates,)


@app.cell
def _(candidates, dms, merged, pd, rankdata):
    RANK_SERIES = {
        "DNMS rank-sum": "dnms",
        "grammaticality": "grammaticality",
        "semantic change": "semantic_change",
        "attention change": "attention_change",
    }

    def rank_split(frame):
        """Each stratum's mean position in the ranking, over this population."""
        out = []
        is_single = (frame["stratum"] == dms.SINGLE_NT).to_numpy()
        for label, column in RANK_SERIES.items():
            ranks = rankdata(frame[column])
            single, multi = ranks[is_single], ranks[~is_single]
            # `u_single` counts the (single, multi) pairs where the
            # single-nucleotide mutation ranks ABOVE the multi-nucleotide one,
            # so dividing by the pair count gives P(multi below single)
            # directly. Getting this the wrong way round reverses the finding,
            # so it is written out rather than negated.
            u_single = single.sum() - len(single) * (len(single) + 1) / 2
            out.append(
                {
                    "series": label,
                    "single-nt mean percentile": 100 * single.mean() / len(ranks),
                    "multi-nt mean percentile": 100 * multi.mean() / len(ranks),
                    "P(multi below single)": u_single / (len(single) * len(multi)),
                }
            )
        return pd.DataFrame(out)

    whole_set = rank_split(candidates)
    measured_set = rank_split(merged)
    return RANK_SERIES, measured_set, rank_split, whole_set


@app.cell
def _(candidates, cells, dms, measured_set, merged, mo, whole_set):
    _single_candidates = int((candidates["stratum"] == dms.SINGLE_NT).sum())
    _single_cells = int((cells["stratum"] == dms.SINGLE_NT).sum())
    _dms_single = cells[cells["stratum"] == dms.SINGLE_NT]["dms_cell_entry"].mean()
    _dms_multi = cells[cells["stratum"] == dms.MULTI_NT]["dms_cell_entry"].mean()
    mo.md(
        f"""
        ## Does codon accessibility sit anywhere in DNMS's ranking?

        It does, strongly — and nothing in DNMS was told about codons.

        **Over the whole candidate set** ({_single_candidates:,}
        single-nucleotide, {len(candidates) - _single_candidates:,}
        multi-nucleotide):

        {whole_set.to_markdown(index=False, floatfmt=("", ".1f", ".1f", ".3f"))}

        **Over the measured cells only** ({_single_cells:,} and
        {len(merged) - _single_cells:,}):

        {measured_set.to_markdown(index=False, floatfmt=("", ".1f", ".1f", ".3f"))}

        **And the assay itself barely separates them**: mean cell entry
        {_dms_single:+.4f} on the single-nucleotide stratum against
        {_dms_multi:+.4f} on the multi-nucleotide one. So the gap above is a
        property of the model, not of what the measurements say about these two
        groups of mutations.
        """
    )


@app.cell
def _(committed_ba1, correlations, mo):
    _rank_sum = correlations[
        (correlations["series"] == "DNMS rank-sum") & (correlations["stratum"] == "all")
    ]["pearson_r"].item()
    _blosum = committed_ba1.loc["blosum62", "all"]
    mo.md(
        f"""
        ## What this says

        **On the one background DNMS was trained on, its rank-sum lands on
        BLOSUM62.** {_rank_sum:+.4f} against {_blosum:+.4f} over all measured
        cells, and on the single-nucleotide stratum it is the lower of the two.
        No component rescues it — grammaticality alone is the weakest of the
        four series.

        **DNMS has almost no single-nucleotide advantage in its *correlation*,
        although it strongly prefers single-nucleotide mutations in its
        *ranking*.** Those are different statements and the section above is
        the first: the model has learned which substitutions are one codon
        change away, because that is what the sequences it read are made of.
        What it has not learned is which of them a cell tolerates.

        **The rank-sum panels are a wedge rather than a cloud**, and that is an
        artifact of the restriction, not a signal: the ranking is taken over
        all 24,073 substitutions and only the ~5,400 measured ones are kept,
        and a DMS library is built toward mutations that are tolerated. The
        shape says something about the library, not about the model.
        """
    )


if __name__ == "__main__":
    app.run()
