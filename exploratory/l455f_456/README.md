# Does the DASM recover the L455F / F456L epistasis in XBB?

[viral-dasm#64](https://github.com/matsengrp/viral-dasm/issues/64). Exploratory.
Nothing here is in the manuscript, and nothing here is a pipeline rule.

## The question

L455F and F456L are individually costly to the XBB spike and jointly beneficial
-- the "FLip" pairing. The model-side form of that claim is a conditional one:
the DASM's selection factor for L455F should be much higher on a parent that
already carries 456L than on one that carries the founder's 456F.

The counts approach asks the same question of the data without the model. Split
a clade's branches by the parent's residue at 456 and compare
`log(actual / expected)` for L455F between the two pools. Both pools require the
founder residue at 455, so `actual` and `expected` describe the same branches on
both sides and the only declared difference is the residue at 456.

## Why this is a script and not a rule

`rule clade_backgrounds` picks each clade's marker unaided. For 23A it picks
F456L on its own -- and then pulls in L455F (59%) and A475V (53%) as
co-travellers, which leaves the very cell we want to measure unscoreable. These
scripts fix the marker at site 456 alone, by hand, with no co-travellers. That
is a deliberate departure from the figure's machinery, so it stays out of it.

Everything downstream of the pool definition *is* the pipeline's machinery:
`shift_per_clade.py` calls `pool_counts.clade_counts`, which is what
`rule pool_counts` calls. The counts estimate and the DASM estimate here are the
two `fig:within_clade_shifts` reports for every other candidate, computed the
same way on different pools.

None of these scripts invokes snakemake, writes under `artifact_root`, or
touches tracked pipeline output. They read the published checkpoint to score;
they never train.

## Running them

From a checkout with its virtual environment, at the repository root:

```bash
python exploratory/l455f_456/shift_per_clade.py --out _ignore/issue64
python exploratory/l455f_456/shift_pooled.py    --out _ignore/issue64_pooled
python exploratory/l455f_456/cotravellers.py  > _ignore/issue64_cotravellers.txt
```

`--out` takes the intermediates too -- the scored haplotypes and the per-pool
cell tables, which are parquet and stay out of the tree. `results/` below holds
the small tables copied out of those runs. `shift_pooled.py` reuses
`<out>/haplotype_scores.parquet` if it is already there, so a rerun skips the
scoring pass.

Scoring is CPU-only and reads one published checkpoint. The counts side streams
the full PCP corpus once per clade to tally the pools, and `shift_per_clade.py`
streams it a second time per clade for the randomization test, so that script is
the slow one. `cotravellers.py` is a single pass over the whole corpus.

## What is here

| file | what it is |
| --- | --- |
| `shift_per_clade.py` | within-clade contrast, one clade at a time, with a 10,000-draw permutation p-value |
| `shift_pooled.py` | the same contrast with every XBB subclade's branches pooled into two 456 pools |
| `cotravellers.py` | what else differs between the two pools, by branch mass |
| `results/per_clade_shifts.json` | `shift_per_clade.py` on 23D, 23A, 23B, 22F and 23C |
| `results/pooled_per_clade.csv` | the twelve clade-pools `shift_pooled.py` aggregates |
| `results/pooled_shift.json` | the pooled result |
| `results/cotravellers.txt` | `cotravellers.py` output, verbatim |

## What they say

**Within clade**, where elapsed time is nearly fixed, the shift is large and the
two approaches agree. Counts and DASM are the 456L pool's effect minus the 456F
pool's:

| clade | lineage | counts | DASM | p |
| --- | --- | ---: | ---: | ---: |
| 22F | XBB | +2.91 | +2.49 | <0.0001 |
| 23A | XBB.1.5 | +2.12 | +2.36 | <0.0001 |
| 23B | XBB.1.16 | +2.13 | +2.34 | 0.0006 |
| 23D | XBB.1.9 | +1.75 | +2.44 | <0.0001 |
| 23C | CH.1.1, **not XBB** | −0.58 | +0.48 | 0.99 |

23C is the negative control: not an XBB descendant, and the contrast vanishes.

**Pooled across the six XBB subclades** (22F, 23A, 23B, 23D, 23E, 23F):

| pool | branches | actual | expected | counts | DASM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 456F | 107,337 | 85 | 58.77 | +0.37 | +0.36 |
| 456L | 36,133 | 476 | 19.99 | +3.17 | +3.47 |
| **shift** | | | | **+2.80** | **+3.11** |

The counts side is a ratio of sums: `expected` is a total evolutionary time
times a rate, and total time is additive over disjoint branch sets. The DASM
side matches that operation at both levels -- haplotypes to clade-pool, then
clade-pools to pooled -- by averaging selection factors in the linear domain
through `pooled_log_sf`, not in the log domain. A log-domain mean would be the
geometric mean, which sits about σ²/2 below; on the 456L pool that is 0.10 nats.

## Three things to hold onto

**The amplitude is about 1.1, not 0.35.** `matsengrp/viral-dasm-tex#3` puts the
median amplitude `|DASM| / |counts|` at 0.35 over the cells the systematic scan
calls in `fig:within_clade_shifts`E -- the DASM recovers the direction of a
within-clade shift broadly but attenuates the median called cell to about a
third. Here it does not attenuate at all: +3.11 against the counts' +2.80.
Whatever drives the general attenuation is not acting on this contrast, and
that is worth saying out loud rather than letting it pass as agreement.

**The co-traveller objection does not survive the per-clade DASM values.** The
pools are observational, and `results/cotravellers.txt` shows real differences
besides 456. Per clade: A701V (4%→62%) and K478R (9%→56%) in 23D, S486P
(49%→100%) in 22F, T547I (21%→0%) and P521T (11%→0%) in 23B, and in 23F a long
list led by L452R (5%→31%) and F157L (19%→35%). Pooled, the three largest gaps
are F157L (+22.8%), L452R (+20.1%) and A701V (+10.0%) -- the first two carried
almost entirely by 23F. 23A is the clean one: nothing reaches 5%.

But the DASM's 456L-pool estimate is +2.76, +2.75, +2.71, +2.79 and +2.78
across the five clades whose founder carries 456F, whose co-traveller profiles
have almost nothing in common, and its 456F-pool estimate across those same
five is +0.26 to +0.38. Confounders that differ that much between clades do not
produce estimates that flat. 23F sits apart on both sides (+3.72 and +2.93)
because its founder already carries 456L, so its haplotypes are a different
set; its 456F pool is 98 branches and its percentages above are noise.

**The pooled number leans on 23F.** EG.5.1 is the one XBB subclade whose
*founder* carries 456L, and it brings 12.78 of the 456L pool's 19.99 expected
counts at an effect of +3.40. It is also the clade with the most time separating
it from the others, so it is the weakest link in the time-controlled argument.
The per-clade rows above are the claim that controls for elapsed time; the
pooled row is the claim that has the events. They agree, which is the point.
Two per-clade cells are too thin to read on their own and are in the table only
for the pooled sums: 23E's 456L pool is one event, and 23F's 456F pool is five
events on 98 branches.
