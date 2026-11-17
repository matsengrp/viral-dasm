# The four deep mutational scans `fig:dms` reads

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
| Delta | 21J | `dadonaite2023_Delta_muteffects.csv` | [dms-vep/SARS-CoV-2_Delta_spike_DMS_REGN10933](https://github.com/dms-vep/SARS-CoV-2_Delta_spike_DMS_REGN10933), `results/muteffects_functional/muteffects_observed.csv` at commit `ac9c29b0d9aea6b2ef484b63c3a634ea0aa26adb` | MIT, `Copyright (c) 2022, Bernadeta Dadonaite and Jesse Bloom` |
| BA.1 | 21K | `dadonaite2023_BA1_muteffects.csv` | [dms-vep/SARS-CoV-2_Omicron_BA.1_spike_DMS_mAbs](https://github.com/dms-vep/SARS-CoV-2_Omicron_BA.1_spike_DMS_mAbs), `results/muteffects_functional/muteffects_observed.csv` at commit `fc551fe332ac8c50c3ffeacc4c6a8dac5fe43369` | MIT |
| XBB.1.5 | 23A | `dadonaite2024_XBB15_summary.csv` | [dms-vep/SARS-CoV-2_XBB.1.5_spike_DMS](https://github.com/dms-vep/SARS-CoV-2_XBB.1.5_spike_DMS), `results/summaries/summary.csv` | MIT, granting "this software and associated data" |
| KP.3.1.1 | 24E | `dadonaite2025_KP311_func_effects.csv` | [dms-vep/SARS-CoV-2_KP.3.1.1_spike_DMS](https://github.com/dms-vep/SARS-CoV-2_KP.3.1.1_spike_DMS), `results/func_effects/averages/293T_medium_ACE2_entry_func_effects.csv` | MIT, as above |

Citations: Delta and BA.1 are the same paper, Dadonaite et al., [10.1016/j.cell.2023.02.001](https://doi.org/10.1016/j.cell.2023.02.001) (`dadonaite2023pseudovirus`); XBB.1.5, Dadonaite et al., [10.1038/s41586-024-07636-1](https://doi.org/10.1038/s41586-024-07636-1); KP.3.1.1, Dadonaite et al., [10.1128/jvi.01423-25](https://doi.org/10.1128/jvi.01423-25).

## Filtering: each study's own, not a house rule

These are four separate experiments with their own thresholds, and the
measurements used here are the ones their authors would hand you. One of the
four releases a file that is already filtered; three do not, and what each is
missing is read from that study rather than borrowed from the others.

### Delta — not filtered by its study, so its threshold is applied here

Delta and BA.1 come from the same paper and the same pipeline, so this file is
BA.1's in every structural respect: the per-mutation effect on spike-mediated
entry on the observed phenotype, released over every mutation the pipeline fit
and still carrying `times_seen`, `effect_std` and `n_libraries`. The threshold
is that repository's own `muteffects_plot_kwargs.addtl_slider_stats:
{times_seen: 3}`, read from the same config key as BA.1's and applied by the
same `notebooks/compare_muteffects.ipynb`. It declares no `effect_std` bound,
so none is set here. `reference_site` is Wuhan-Hu-1 numbering, as for BA.1.

Retained: 6,143 cells — 4,599 one nucleotide change from the founder's codon,
1,544 needing two or three.

**What `effect` is.** That repository's `config.yaml` names this file as
`muteffects_observed` and sets `muteffects_avg_method: median`.
`dms-vep-pipeline` (its submodule, pinned at `48bf2f21`)
`notebooks/avg_muteffects.ipynb` concatenates the per-*selection* global
epistasis fits for all eight Delta selections — four libraries, two replicates
each — groups by site and mutation, and takes that median. So it is the median
across replicates **and** libraries, which is what the paper says it reports.
The `Lib-N effect` columns beside it are a separate merge, the median within
each library, and are not the inputs to `effect`; `effect` therefore does not
equal their median.

**One way it is not BA.1's twin: the observed scale is saturated on this
background.** Delta's wildtype sits at the top of the global-epistasis link's
range in five of the eight selections (Lib-1 ×2, Lib-2 replicate 1, Lib-4 ×2),
whose per-selection fits max out at exactly `-0.0000` with no positive value
at all; the other three behave normally. With the median taken over eight
selections of which five are capped, the published `effect` is exactly 0.0 for
43.9% of rows and 43.5% of retained cells, against no exact zeros at all in
the other three homologs. The latent phenotype is unaffected in all eight
selections, and so are the raw variant-level `func_scores`, so the ceiling is
the link function rather than the libraries — and it is a property of the
published data, not of this copy. It bounds every predictor equally, which is
why Delta's correlations are not comparable at face value with the other
three homologs'.

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
already do for every homolog; the third is `min_times_seen: 3`.

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

#### Superseded: the multidms refit this figure read before

Earlier versions of this pipeline took BA.1's effects from
`matsengrp/SARS-CoV-2_spike_multidms`'s `results/mutations_df.csv` — the
`avg_predicted_func_score` column of a multidms fit over Delta, BA.1 and BA.2,
at commit `66b9833d39308b853a9531634cc7e9043d6a11f8`. The commit is kept here
so that source stays recoverable. Two reasons it is not what this figure
wants:

- **It is a variant's predicted phenotype, not a mutation's effect.** It is
  `g(C + β)`: the global-epistasis link applied to the wildtype's own latent
  baseline plus the mutation's. The wildtype therefore does not sit at zero —
  the 1,117 mutations that fit gives `|avg_beta| < 0.05` land at −1.68 ± 0.17,
  and 99.98% of the column is below zero. Pearson *r* is shift-invariant, so
  the displaced zero biases nothing by itself; what it costs is that BA.1 was
  the one homolog of three whose axis did not mean "effect relative to this
  clade's wildtype", and nothing on the figure said so. The nonlinearity is
  the part that does move a number: `r(avg_beta, avg_predicted_func_score)` is
  0.90, so it is a squashed monotone transform of the effect, not the effect.
- **Its mutation set is biased toward the single-nucleotide stratum.** That
  table is written with `combine_replicate_muts(..., how="inner")`, keeping
  only mutations sampled in all three homolog libraries — which preferentially
  keeps well-sampled ones, and a mutation one nucleotide change from the
  founder's codon is better sampled than one needing two or three. Measured
  cells go from 5,413 to 6,808, and the multi-nucleotide share of the
  population from 24.7% to 33.3%.

**How to read the before/after correlations, because the pooled number
misleads.** Each figure below names the set it is computed over.

| | old set (5,413) | new set (6,808) |
|---|---|---|
| DASM, all | +0.5852 | +0.5379 |
| DASM, single-nt | +0.6415 (n=4,074) | +0.6114 (n=4,538) |
| DASM, multi-nt | +0.4127 (n=1,339) | **+0.4153** (n=2,270) |

The pooled fall of 0.047 is **larger than either stratum's change**, and
multi-nt's agreement *rises*. It is a re-weighting: the new population is a
third multi-nucleotide instead of a quarter, and that is the harder stratum.
The 1,617 added cells are 60.9% multi-nucleotide against 24.7% in the old set,
and correlate at +0.469; 222 previously measured cells fall below
`times_seen >= 3` and are dropped.

Over the 5,191 cells both populations carry — the quantity changing and the
population held fixed — the DASM goes +0.5747 → +0.5596 overall, +0.6347 →
+0.6190 on the single-nucleotide stratum and +0.3899 → +0.3848 on the
multi-nucleotide one, and the two assay quantities correlate at +0.9195. That
is the whole effect of the change in quantity: about 0.015. Everything else is
the comparator no longer being depleted of the mutations the model finds
hardest.

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
82 insertion-site rows, which this pipeline drops for every homolog.)

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
