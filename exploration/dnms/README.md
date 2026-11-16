# DNMS against the BA.1 spike DMS

**Exploration, not pipeline.** Nothing here is reached by a Snakemake rule, no
test covers it, and `snakemake` neither builds nor needs any of it. It is the
evidence behind a scope decision on EPIC #1's **M5d**: whether DNMS should
become a seventh predictor in `results/panels/dms/predictor_correlations.csv`.

`exploration/` exists so that evidence lives somewhere a reader can find it
rather than in a session transcript. Code here does not meet the pipeline's
standards — it carries a cluster path, it has no rule and no test — and it must
not be imported by anything under `scripts/` or `notebooks/`.

## The question, and the answer

DNMS is Elkin & Zhu, *Commun Biol* 8 (2025),
[10.1038/s42003-024-07262-7](https://doi.org/10.1038/s42003-024-07262-7) — a
ProtBERT fine-tuned on NCBI spike proteins, scoring every substitution on a
supplied parent with three quantities and one combination of them:

```
dnms = rank( 1.5 * rank(-semantic_change) + 3.0 * rank(grammaticality) + 1.0 * rank(-attention_change) )
```

The weights are the authors' CLI defaults (`bin/ParentChildMutate.py`), not
`bin/MutationRankingResults.py`'s, which default to 1/1/1. **Every term is a
rank within one parent's candidate set**, so the score has no units and no zero.

**On BA.1 its rank-sum is +0.213 Pearson against cell entry, where BLOSUM62 is
+0.217.** On the single-nucleotide stratum it is the lower of the two (+0.225
against +0.260), and it is far below `leaf_logfreq` (+0.290), `antigen`
(+0.328) and `dasm` (+0.585). `dnms_ba1_correlations.csv` carries all twelve
numbers; the notebook draws them.

**One further finding, which is the interesting one.** DNMS ranks
single-nucleotide mutations far above multi-nucleotide ones — 66.1 against 42.7
mean percentile over the whole 24,073-substitution candidate set, so a randomly
chosen multi-nucleotide mutation sits below a randomly chosen single-nucleotide
one 73% of the time — while the assay barely separates the two strata at all
(mean cell entry −2.117 against −2.124). Nothing told DNMS about codons; it
learned mutational accessibility from what surveillance sequences are made of.
What it did not learn is which of those mutations a cell tolerates. Its
rank-sum is closer to the DASM's `lambda` than to its `f`.

## Why only BA.1

DNMS's language model is trained on sequences collected **before 1 January
2022**. The corpus runs to January 2023, but those later sequences supply test
mutations, not training: of the 6,256 sequences in `train_seq.pkl`, the latest
collection date is 2021-12-31 and none is on or after the cutoff.

Of this paper's three DMS parents, checked by string membership against
`train_seq.pkl` with deletion columns removed as the authors remove them:

| homolog | in the training set | in the corpus |
|---|---|---|
| BA.1 | **yes**, the exact construct protein | 4,075 clade-21K sequences, 233 of them in training |
| XBB.1.5 | no | one sequence in the whole corpus, Dec 2022, not in training |
| KP.3.1.1 | no | **none** — no KP.3\*, JN.1\* or BA.2.86\* at all |

So BA.1 is DNMS's best case, and the other two would measure the absence of a
lineage rather than the quality of a method. Even BA.1 is thin: its 233
training sequences are 3.7% of a set that is 43% Delta and 19% Alpha, all
collected in the variant's first seven weeks.

## The archive, and why it is mirrored

The paper's `README` hosts its data and weights at
`https://www.cse.fau.edu/~xqzhu/dnms/`. **Both links return HTTP 404** as of
2026-11-15, while the host itself answers — the files are gone, not the site.
The Zenodo record the paper cites,
[10.5281/zenodo.14015344](https://doi.org/10.5281/zenodo.14015344), is code
only: one 199 KB zip of `maggieelkin/CovMutation` at `v1.0.0`, no data and no
weights. There is no later release; the repository has one branch, one tag and
one tree version, last pushed 2024-10-31.

Both files were recovered from the Internet Archive and mirrored to
`/fh/fast/matsen_e/shared/elkin-dnms/`:

| file | bytes | sha256 | Wayback snapshot |
|---|---:|---|---|
| `archive/ncbi_tree_v1.zip` | 383,441,007 | `71d9f0a535015298106099a2e7b9f509b3f8aa76d076712f46c9f557c1736c25` | `20250122141914` |
| `archive/protbert_ft_v1.zip` | 4,361,144,453 | `931dfcddac3f48b04fd4f0cee40386dd2befa281d3836a9fcce502879bc34542` | `20250123203457` |

`ncbi_tree_v1/` beside them holds the five members this analysis reads,
unzipped. **The checkpoint is not read by anything here** — BA.1's three value
dictionaries are already published, so no forward pass is run. It is mirrored
because it is otherwise unobtainable.

## Running it

```bash
uv run python exploration/dnms/prep_dnms_ba1.py          # ~70 s, writes dnms_ba1_scores.parquet
uv run marimo run exploration/dnms/dnms_ba1_panels.py --host 0.0.0.0 --port 2719 --headless
```

Set `DNMS_ROOT` if the archive is somewhere other than the shared mirror.

`dnms_ba1_scores.parquet` (1 MB) is **not committed** — it is derived, and the
prep script rebuilds it. The static scatter PNG is not committed either; the
notebook draws it.

**The prep script gates the pickles before unpickling.** They came from a third
party's web host by way of the Internet Archive, and unpickling is arbitrary
code execution, so `pickletools.genops` walks the opcode stream first and the
load is refused unless the only globals are numpy's. That walk is most of the
70 seconds, which is why it happens once here rather than on every page load.

## Status

Filed as **matsengrp/viral-dasm#52**, which carries the measurements above, the
design and traps an implementation would need, and the three open decisions —
chiefly whether to build the arm at all. The pre-measurement draft that
preceded it has been dropped from this directory; the issue supersedes it.

On the evidence here the recommendation is **not** to build the arm, and to
spend the finding instead: DNMS ranks by mutational accessibility rather than
by tolerance, which is the distinction `fig:probabilities` draws.
