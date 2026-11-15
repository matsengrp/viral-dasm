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
| BA.1 | 21K | `haddox2023_BA1_mutations_df.csv` | [matsengrp/SARS-CoV-2_spike_multidms](https://github.com/matsengrp/SARS-CoV-2_spike_multidms), `results/mutations_df.csv` at commit `66b9833d39308b853a9531634cc7e9043d6a11f8` | no licence declared; this project's own paper |
| XBB.1.5 | 23A | `dadonaite2024_XBB15_summary.csv` | [dms-vep/SARS-CoV-2_XBB.1.5_spike_DMS](https://github.com/dms-vep/SARS-CoV-2_XBB.1.5_spike_DMS), `results/summaries/summary.csv` | MIT, granting "this software and associated data" |
| KP.3.1.1 | 24E | `dadonaite2025_KP311_func_effects.csv` | [dms-vep/SARS-CoV-2_KP.3.1.1_spike_DMS](https://github.com/dms-vep/SARS-CoV-2_KP.3.1.1_spike_DMS), `results/func_effects/averages/293T_medium_ACE2_entry_func_effects.csv` | MIT, as above |

Citations: BA.1, Haddox et al., [10.1101/2023.07.31.551037](https://doi.org/10.1101/2023.07.31.551037); XBB.1.5, Dadonaite et al., [10.1038/s41586-024-07636-1](https://doi.org/10.1038/s41586-024-07636-1); KP.3.1.1, Dadonaite et al., [10.1128/jvi.01423-25](https://doi.org/10.1128/jvi.01423-25).

## Filtering: each study's own, not a house rule

These are three separate experiments with their own thresholds, and the
measurements used here are the ones their authors would hand you. Two of the
three release a file that is already filtered; one does not, and what it is
missing is read from that study rather than borrowed from the other two.

### BA.1 — already filtered by its study

`results/mutations_df.csv` is written by that repository's own
`spike-analysis.ipynb`:

```python
# the number of times a mutation must be seen in each condition to be included in downstream analysis
times_seen_threshold = 1
# The chosen lasso strength for our final spike model results
chosen_lasso_strength = 5e-5
...
mut_df_replicates = multidms.utils.combine_replicate_muts(
    {row.replicate: row[model_name] for idx, row in chosen_replicate_models.iterrows()},
    times_seen_threshold=times_seen_threshold,
    how="inner",
)
mut_df_replicates.to_csv(f"{OUTDIR}/mutations_df.csv")
```

So every mutation it carries was seen at least once in **each** of the three
homolog conditions and was fit in **both** replicates, at the lasso strength
that notebook calls the chosen one. Nothing further is filtered here.

**The effect column is `avg_predicted_func_score`**, the two replicates'
average. It carries no homolog suffix because BA.1 is the model's *reference*
condition — the file has `shift_Delta` and `shift_Omicron_BA2` columns and no
BA.1 shift, which is what identifies the reference.

**This is a fitted artifact, not a raw measurement table.** It is the output
of a multidms fit over that repository's own functional scores, so reproducing
it means rerunning that pipeline rather than downloading a file. Unlike the
other two, which are released measurements, this one is a model's parameters —
which is also why the manuscript says the BA.1 effects it analyzes "are not
that experiment's own estimates, but effects re-estimated from its data".

**The commit is the one that accompanies the preprint**, Haddox et al.,
"Jointly modeling deep mutational scans identifies shifted mutational effects
among SARS-CoV-2 spike homologs", bioRxiv
[2023.07.31.551037v1](https://doi.org/10.1101/2023.07.31.551037) — `main.bib`'s
`haddox2023jointly`, whose PDF is in the manuscript repository's `papers/`.
That is what fixes the fit this file holds: the `times_seen_threshold` and the
lasso strength above are the ones that version's analysis notebook sets, and a
later revision of the paper may move both.

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
