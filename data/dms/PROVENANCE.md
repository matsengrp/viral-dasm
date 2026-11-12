# Where these three files came from

Each is one study's own file, carried here so `fig:dms` can be rebuilt from
this checkout. `config.yaml`'s `dms` block pins each one's md5 and declares
the conversion `scripts/dms_published.py` applies; this file records where the
bytes came from, which `config.yaml` deliberately does not, so that it stays
paths and pins.

## `dadonaite2024_XBB15_summary.csv`

| | |
|---|---|
| source repository | [`dms-vep/SARS-CoV-2_XBB.1.5_spike_DMS`](https://github.com/dms-vep/SARS-CoV-2_XBB.1.5_spike_DMS) |
| path there | `results/summaries/summary.csv` |
| commit | `ef1e4a9b6cb816aa5137e181324159f7771fafb5` |
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
| commit | `73200f4bc0b7203c0c79efe8d4f05e4ad93aacd1` |
| taken on | 2026-11-12 |
| licence | MIT, `LICENSE.txt` |
| paper | Dadonaite et al., *J Virol* 2025, [10.1128/jvi.01423-25](https://doi.org/10.1128/jvi.01423-25) |
| byte-identical to that commit | yes |

This repository publishes no `results/summaries/summary.csv`, so unlike
XBB.1.5 the phenotype is read from its own averaged table. Its `times_seen`
and `n_selections` columns are the filtering EPIC #1's M5c would apply, and
they are in this file rather than in a second one.

Both MIT grants read "this software **and associated data** and documentation
files", and both include the right to distribute, which is what permits the
two files above to be carried here. `gh repo view --json licenseInfo` reports
`other` rather than `MIT` for each, because of a trailing note that the
`dms-vep-pipeline` subdirectory is licensed separately.

## `haddox_spike_multidms.csv.gz`

| | |
|---|---|
| source repository | [`matsengrp/SARS-CoV-2_spike_multidms`](https://github.com/matsengrp/SARS-CoV-2_spike_multidms) |
| path there | `results/spike_analysis/mutations_df.csv`, **untracked** |
| produced by | that repository's `workflow/rules/spike.smk`, over its tracked `data/Omicron_BA1/*_func_scores.csv` |
| taken on | 2026-11-12 |
| licence | none declared; this project's own paper |
| paper | Haddox et al. 2023, [10.1101/2023.08.01.551488](https://doi.org/10.1101/2023.08.01.551488) |
| byte-identical to that commit | not applicable -- see below |
| compression | `gzip -n9`, so the archive carries no timestamp and its digest is reproducible |

**This one is not a copy of a published file, and that is worth stating
plainly.** `mutations_df.csv` appears nowhere in that repository's tree:
`results/spike_analysis/` is in its `.gitignore`, and the file is an output of
its own multidms fitting run. What is published is the ~70 MB of per-variant
functional scores the fit consumes. So a reader cannot download this file;
they would have to rerun that pipeline, and the bytes here are a fit, not a
transcription.

It is carried anyway, and uncompressed it is 16,307,300 bytes holding nine
regularization strengths over 10,123 mutations in two replicates. `fig:dms`
reads one strength. Carrying the whole sweep rather than the slice is what
puts the three choices the paper made -- which regularization, how thin a
measurement to keep, which replicates to average -- in `config.yaml` where
they can be read, instead of leaving them in a prep script in another
repository. That visibility is the reason #30 exists.
