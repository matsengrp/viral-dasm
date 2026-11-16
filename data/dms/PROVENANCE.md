# The three deep mutational scans `fig:dms` reads

Each file here is a copy of what its study published, unmodified. The
conversion to this pipeline's `wildtype,site,mutant,functional_effect` columns
happens in `scripts/dms_published.py` when the file is read, so what is
carried is the study's own release and what is applied to it is visible and
tested rather than inherited from a prepared CSV built elsewhere.

`config.yaml`'s `dms` block pins each file by md5 and names the columns and
thresholds below. These are published datasets, not GISAID-derived, so they
and anything derived from them may be committed.

| homolog | clade | file | source | released |
|---|---|---|---|---|
| BA.1 | 21K | `dadonaite2023_BA1_muteffects.csv` | [dms-vep/SARS-CoV-2_Omicron_BA.1_spike_DMS_mAbs](https://github.com/dms-vep/SARS-CoV-2_Omicron_BA.1_spike_DMS_mAbs), `results/muteffects_functional/muteffects_observed.csv` at commit `fc551fe332ac8c50c3ffeacc4c6a8dac5fe43369` | MIT |
| XBB.1.5 | 23A | `dadonaite2024_XBB15_summary.csv` | [dms-vep/SARS-CoV-2_XBB.1.5_spike_DMS](https://github.com/dms-vep/SARS-CoV-2_XBB.1.5_spike_DMS), `results/summaries/summary.csv` | MIT, granting "this software and associated data" |
| KP.3.1.1 | 24E | `dadonaite2025_KP311_func_effects.csv` | [dms-vep/SARS-CoV-2_KP.3.1.1_spike_DMS](https://github.com/dms-vep/SARS-CoV-2_KP.3.1.1_spike_DMS), `results/func_effects/averages/293T_medium_ACE2_entry_func_effects.csv` | MIT, as above |

Citations: BA.1, Dadonaite et al., [10.1016/j.cell.2023.02.001](https://doi.org/10.1016/j.cell.2023.02.001); XBB.1.5, Dadonaite et al., [10.1038/s41586-024-07636-1](https://doi.org/10.1038/s41586-024-07636-1); KP.3.1.1, Dadonaite et al., [10.1128/jvi.01423-25](https://doi.org/10.1128/jvi.01423-25).

## Filtering: each study's own, not a house rule

These are three separate experiments with their own thresholds, and the
measurements used here are the ones their authors would hand you. One of the
three releases a file that is already filtered; two do not, and what each is
missing is read from that study rather than borrowed from the other two.

### BA.1 — not filtered by its study, so its threshold is applied here

`results/muteffects_functional/muteffects_observed.csv` is the per-mutation
effect on spike-mediated entry that experiment publishes, on the observed
phenotype scale: the median over its four functional selections (Lib-1 in two
replicates, plus Lib-2 and Lib-3), per `muteffects_avg_method: median`. It is
released over every mutation the pipeline fit — it still carries `times_seen`,
`effect_std` and `n_libraries` — so this pipeline applies the threshold that
repository declares:

- `config.yaml`, `muteffects_plot_kwargs`: `addtl_slider_stats: {times_seen: 3}`.

**That is the threshold its own analysis uses for this job.** The repository's
`notebooks/compare_muteffects.ipynb` correlates these effects against four
other datasets — natural-sequence enrichment, and the Starr, Ouyang and Tan
scans — which is exactly what `fig:dms` does, and it reads:

```python
times_seen = config["muteffects_plot_kwargs"]["addtl_slider_stats"]["times_seen"]
muteffects = (
    pd.read_csv(config["muteffects_observed"])
    .query("wildtype != mutant")
    .query("(wildtype not in ['-', '*']) and (mutant not in ['-', '*'])")
    .query("times_seen >= @times_seen")
)
```

The first two queries are what `scripts/dms_published.py` and `scripts/dms.py`
already do for all three homologs; the third is `min_times_seen: 3`.

**`times_seen` here is a mean, not a count.** `dms-vep-pipeline`'s
`avg_muteffects.ipynb` aggregates it as `s.sum() / n_selections` over all four
selections, so a mutation seen in one selection and not the other three is
reported at a quarter of its count — which is why values such as 0.25 appear.
Threshold 3 therefore means "seen in three variants per selection on
average", and that is how the manuscript states it.

**No `effect_std` bound is set, because that study declares none.** Its
largest value on this file is 3.54, so borrowing KP.3.1.1's 2.5 would drop
rows these authors keep. `tests/test_dms_published.py` pins both directions of
the predicate.

**`reference_site`, not `sequential_site`.** The file carries both; the first
is Wuhan-Hu-1 numbering, which is what every table in this pipeline is keyed
on, and the second indexes the construct. They agree on 6% of rows, so reading
the wrong one would not fail loudly.

#### This replaces the multidms refit, which this figure read before

Until this change, BA.1's effects came from `matsengrp/SARS-CoV-2_spike_multidms`'s
`results/mutations_df.csv` — the `avg_predicted_func_score` column of a
multidms fit over Delta, BA.1 and BA.2, at commit
`66b9833d39308b853a9531634cc7e9043d6a11f8`. Two reasons it is gone:

- **It is a variant's predicted phenotype, not a mutation's effect.** The
  wildtype does not sit at zero on that scale: the 1,117 mutations the fit
  gives `|avg_beta| < 0.05` land at −1.68 ± 0.17, and 99.98% of the column is
  below zero. BA.1 was the one homolog of the three whose axis did not mean
  "effect relative to this clade's wildtype", and nothing on the figure said
  so.
- **It covers less.** That table is written with
  `combine_replicate_muts(..., how="inner")`, keeping only mutations seen in
  all three homolog libraries; the experiment's own table keeps every mutation
  it fit. Retained substitutions rise from 5,431 to 6,828, and measured cells
  from 5,413 to 6,808.

The two quantities correlate at r = 0.911, and the switch moves the figure's
correlations by at most 0.015 — the DASM's, over the cells both carry, goes
+0.585 → +0.571 overall and +0.642 → +0.627 on the single-nucleotide stratum.
It was not made to change a number.

### XBB.1.5 — already filtered by its study

`results/summaries/summary.csv` is that experiment's filtered summary across
its assays; this figure reads the `spike mediated entry` column, beside the
`human sera escape` and `ACE2 binding` columns in the same file. Nothing is
filtered here. Converting it reproduces the prepared CSV this pipeline used
before, byte for byte, which is what licenses the switch for this homolog.

### KP.3.1.1 — not filtered by its study, so its thresholds are applied here

That repository publishes the averaged per-phenotype table unfiltered — it
still carries `times_seen` and `n_selections`, left to the consumer — so this
pipeline applies the two thresholds that repository declares:

- `data/func_effects_config.yml`, `avg_func_effects_default`:
  `addtl_slider_stats: {times_seen: 2, effect_std: 2.5}` with
  `addtl_slider_stats_as_max: [effect_std]`.
- `data/summaries_config.yml`, `summary_default`: `min_times_seen: 2`.

**Verified against their own published filtered column.** Applying exactly
`times_seen >= 2` and `effect_std <= 2.5` to
`func_effects/averages/293T_medium_ACE2_entry_func_effects.csv` reproduces the
`spike mediated entry` column of `results/summaries/sera_group_averages.csv`:
the same **8,263** substitutions, with `max |difference| = 0.0` on every one.
(The summary carries 9,576 rows; the difference is 1,235 synonymous rows and
82 insertion-site rows, which this pipeline drops for all three homologs.)

Two notes for whoever revisits this:

- **The `effect_std` bound is inert on today's file**, whose largest value is
  1.879. It is applied because it is the authors' stated threshold and a later
  release may reach it.
- **Not every summary file agrees.** The sera-group summaries
  (`sera_group_averages.csv`, `all_adult_sera_escape.csv`,
  `post_vaccination.csv` and the rest) carry an identical 9,576-row entry
  column, but `antibody_escape.csv` carries a different one, 9,205 rows, from
  its own summary block. Taking "whichever summary is to hand" would silently
  pick a different population.
