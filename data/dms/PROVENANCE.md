# Where these three files came from

Each is one study's own file, carried here so that `fig:dms`'s DMS inputs need
nothing fetched. `config.yaml`'s `dms` block pins each one's md5 and declares
the conversion `scripts/dms_published.py` applies; this file records where the
bytes came from and under what terms, which `config.yaml` records only in
passing.

## `dadonaite2024_XBB15_summary.csv`

| | |
|---|---|
| source repository | [`dms-vep/SARS-CoV-2_XBB.1.5_spike_DMS`](https://github.com/dms-vep/SARS-CoV-2_XBB.1.5_spike_DMS) |
| path there | `results/summaries/summary.csv` |
| source commit | `ef1e4a9b6cb816aa5137e181324159f7771fafb5` |
| commit date | 2024-09-22 |
| taken on | 2026-11-12 |
| licence | MIT, `LICENSE.txt` |
| paper | Dadonaite et al., *Nature* 2024, [10.1038/s41586-024-07636-1](https://doi.org/10.1038/s41586-024-07636-1) |
| byte-identical to that commit | yes |

Every phenotype in one table. `fig:dms` reads the `spike mediated entry`
column; the sera-escape and ACE2-binding columns beside it are untouched.

## `dadonaite2025_KP311_func_effects.csv`

| | |
|---|---|
| source repository | [`dms-vep/SARS-CoV-2_KP.3.1.1_spike_DMS`](https://github.com/dms-vep/SARS-CoV-2_KP.3.1.1_spike_DMS) |
| path there | `results/func_effects/averages/293T_medium_ACE2_entry_func_effects.csv` |
| source commit | `73200f4bc0b7203c0c79efe8d4f05e4ad93aacd1` |
| commit date | 2025-10-14 |
| taken on | 2026-11-12 |
| licence | MIT, `LICENSE.txt` |
| paper | Dadonaite et al., *J Virol* 2025, [10.1128/jvi.01423-25](https://doi.org/10.1128/jvi.01423-25) |
| byte-identical to that commit | yes |

This repository publishes no file named `results/summaries/summary.csv`, so
the phenotype is read from its own averaged table and this homolog carries the
**unfiltered** measurements. Its `results/summaries/*.csv` do each carry a
`spike mediated entry` column, with values identical to this file's `effect`
on the rows they keep, so they are the repoint target for EPIC #1's M5c;
filtering this file's own `times_seen` column reaches the same numbers.

**Measured at medium ACE2**, where XBB.1.5's column above is high ACE2. Each
is its study's designated primary phenotype, so the two homologs' effects are
not on one scale.

Both MIT grants read "this software **and associated data** and documentation
files", and both include the right to distribute, which is what permits the
two files above to be carried here. `gh repo view --json licenseInfo` reports
`other` rather than `MIT` for each, partly because of a trailing note that the
`dms-vep-pipeline` subdirectory is licensed separately. The same licence
requires its copyright and permission notice to travel with a copy, and this
directory does not yet carry either — see the pull request's DEFERRED section.
Both are `Copyright (c) 2022, Bernadeta Dadonaite and Jesse Bloom`.

## `haddox_spike_multidms.csv.gz`

| | |
|---|---|
| source repository | [`matsengrp/SARS-CoV-2_spike_multidms`](https://github.com/matsengrp/SARS-CoV-2_spike_multidms) |
| path there | `results/spike_analysis/mutations_df.csv`, on the `phase-0-foundation` and `5-update-notebooks` branches only |
| source commit | `6c98b7b607d7387b508cdaa192d659ee9fca7367` (blob `bc2842f526c57428574bbcda14b1c0a5f61cea99`) |
| commit date | 2024-03-13 |
| taken on | 2026-11-12 |
| licence | none declared |
| paper | Haddox et al. 2023, [10.1101/2023.07.31.551037](https://doi.org/10.1101/2023.07.31.551037) |
| byte-identical to that commit | yes, uncompressed |
| compression | `gzip -n9`, so the archive carries no timestamp and its digest is reproducible |

**This one is not a copy of a maintained published table, and that is worth
stating plainly.** `results/spike_analysis/` is in that repository's
`.gitignore` and `mutations_df.csv` is an output of its own multidms fitting
run, absent from `main`; a copy survives on the two feature branches named
above, which is where these bytes match. What `main` publishes is the roughly
230 MB of per-variant functional scores the fit consumes, across all three
homologs. So the bytes here are a fit, not a transcription, and regenerating
them means rerunning that pipeline.

The fit is **joint over Delta, Omicron_BA1 and Omicron_BA2** — that is what
the paper is about, and the file's own columns show it, carrying
`predicted_func_score_` and `times_seen_` for all three. BA.1 is the
reference condition.

Uncompressed it is 16,307,300 bytes: nine regularization strengths over 10,123
mutations in two replicates. `fig:dms` reads one strength. Carrying the whole
sweep rather than the slice is what puts the three choices the paper made —
which regularization, how thin a measurement to keep, and that the replicates
are averaged — in `config.yaml` where they can be read, instead of leaving
them in a prep script in another repository. That visibility is the reason #30
exists.
