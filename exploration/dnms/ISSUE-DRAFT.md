# M5d: score the spike DMS with DNMS, which has seen one of the three backgrounds

🤖 **Background.** EPIC #1's **M5d** predictor table lists one unfiled arm: **DNMS** — Elkin & Zhu, *Commun Biol* 8 (2025), [10.1038/s42003-024-07262-7](https://doi.org/10.1038/s42003-024-07262-7), a ProtBERT fine-tuned on NCBI spike proteins. dasm2-experiments#1966 recommended it as "the one cheap addition: full spike, parent-conditioned, free (`maggieelkin/CovMutation`), DMS-free, and already in the output form the predictor join wants", with its January 2023 corpus end "a stated limitation for two of the three backgrounds, not a disqualification". The predictor contract it would join is #28's and the table is `results/panels/dms/predictor_correlations.csv`, which already carries five arms: `blosum62`, `leaf_logfreq`, `antigen`, `dasm`, `dasm_prob`, `neutral_rate` (`scripts/panel_spec.py:563-570`).

**This issue is written after reading the released artifacts rather than the abstract, and two of #1966's premises do not survive that reading.** Both are stated below with the measurements behind them. Neither is fatal, but together they change what this arm can be: a one-homolog comparison that costs almost nothing, or a three-homolog comparison that is mostly a statement about what DNMS never saw.

**Goal.** `snakemake dms` writes a `dnms_{homolog}.parquet` through `dms_models.write_scores` for each homolog in scope, and `predictor_correlations.csv` gains its rows, with every existing row unchanged.

---

## What was verified on 2026-11-15, before this issue was written

Each was run against the real files. An implementation that contradicts one of them is wrong.

### 1. The cutoff is 1 January 2022, and it is a real cut

The paper states it: *"We create a SARS-CoV-2 sequence database from the NCBI database from December 2019 up to January 2023. Using a cutoff date of January 1st, 2022, we create training and test sequences. The cutoff date represents a date in time where we can utilize information learned from all sequences collected to predict future mutations."* The code agrees — `ParentChildMutate.py:158,216` default `cut_off='2022-1-1'`, and `:303` selects test mutations as `mutation_data['first_date'] >= self.cut_off`.

**Read from the data, not inferred from the prose.** `treebuild/seq_metadata.csv` holds 35,943 sequences, collected **2019-12-23 to 2023-01-03** (release dates to 2023-01-10) and carrying an `in_training` flag. Of the **6,256** flagged `in_training` — exactly the length of `exp_settings/train_seq.pkl` — the latest collection date is **2021-12-31**, and **zero** were collected on or after 2022-01-01. So the two dates in circulation mean different things and both are right: the **corpus** ends January 2023, the **language model's** training set ends 1 January 2022, and the post-cutoff two thirds of the corpus exist only to supply test mutations.

### 2. One of the three DMS parents is in the training set; the other two are not in the corpus at all

Checked by string membership against `train_seq.pkl`, using `data/antigen/dms_parents.fasta` with deletion columns removed as `FitnessExp.py:150` removes them (`seq.replace('-','')`).

| homolog | residues after removing deletions | in `train_seq.pkl` | released LM values |
|---|---:|---|---|
| BA.1 | 1,267 | **yes** | **yes** — all three terms |
| XBB.1.5 | 1,269 | no | no |
| KP.3.1.1 | 1,264 | no | no |

And the lineages behind the two misses are thin or absent: **80** `XBB*` sequences in the whole corpus, of which exactly **one** is called `XBB.1.5`, all collected Sep–Dec 2022 and none in training; **zero** `KP.3*`, `JN.1*` or `BA.2.86*` sequences — KP.3.1.1 postdates the corpus by about eighteen months, and nothing in its ancestry past BA.2 is represented.

**This is the mirror image of arm D.** `data/antigen/PROVENANCE.md` records that antiGen's cutoff was chosen *because* it is the only rung whose training split contains all three parents. DNMS has no such rung to choose: there is only one released model.

### 3. BA.1 costs nothing to score — the values are already published

`tree_v1_seq_forward_prob_ft.pkl`, `tree_v1_seq_l2_change_ft.pkl` and `tree_v1_seq_attn_ft.pkl` are dictionaries keyed by parent sequence over **360** sequences, and the BA.1 DMS parent is one of them. For it they carry grammaticality over all 1,267 positions and semantic change and attention change over **24,073** substitutions — 1,267 × 19, every single amino-acid substitution on that parent. **So the BA.1 column needs no checkpoint, no GPU and no forward pass: it is a lookup in a published pickle.** The other two homologs need the checkpoint and a full inference run.

### 4. The artifacts are offline; only Internet Archive copies exist

The repo's `README` hosts both at `https://www.cse.fau.edu/~xqzhu/dnms/`. **Both return HTTP 404** as of 2026-11-15, while the host itself answers — the files are gone, not the site. The Zenodo record the paper cites, [10.5281/zenodo.14015344](https://doi.org/10.5281/zenodo.14015344), is **code only**: one 199 KB zip of the repo at `v1.0.0`, no data and no weights.

Both are in the Internet Archive and have been **mirrored to `/fh/fast/matsen_e/shared/elkin-dnms/archive/`**, in the pattern `/fh/fast/matsen_e/shared/specht-antigen/` set for arm D:

| file | bytes | sha256 | Wayback snapshot |
|---|---:|---|---|
| `ncbi_tree_v1.zip` | 383,441,007 | `71d9f0a535015298106099a2e7b9f509b3f8aa76d076712f46c9f557c1736c25` | `20250122141914` |
| `protbert_ft_v1.zip` | ~4,360,882,930 | recorded on completion | `20250123203457` |

**That directory still needs its `PROVENANCE.md`**, in `data/antigen/PROVENANCE.md`'s form; writing it is the first task of whoever takes this issue. Nothing in this repository may cite the FAU URLs as a live source.

### 5. There is no later model, and retraining is not a port

`maggieelkin/CovMutation` has one branch, one tag and one release (`v1.0.0`), last pushed 2024-10-31. The data layout is versioned (`v{tree-version}`) but only `ncbi_tree_v1` was ever published. Bringing DNMS past 1 Jan 2022 would mean rebuilding their NCBI download, Nextclade QC and Nextstrain tree and re-fine-tuning ProtBERT — a reimplementation of the paper, not an adapter. **Out of scope, and it should not be proposed as a cheap follow-up the way arm D's cap-lifted retrain is.**

### 6. The output is a within-parent weighted rank-sum, not an effect size

`MutationRankingResults.py:dnms` computes, over one parent's candidate mutations:

```
acquisition = alpha * rank(-semantic_change) + beta * rank(grammaticality) + gamma * rank(-attention_change)
dnms        = rank(acquisition)
```

with `alpha=1.5, beta=3.0, gamma=1.0` in the paper's experiments. **Every term is a rank within that parent's own candidate set**, so the score has no units, no zero and no meaning across parents. dasm2-experiments#1966's body called DNMS "already in the output form #1909's adapter contract wants"; the fact-checked comment on the same issue corrected that to "a weighted rank-sum (Eq. 18) within a parent, not an effect size", and the code is with the correction.

**The authors' own DMS comparison is Spearman, per strain, and on the three terms separately rather than on the rank-sum.** `FitnessExp.py:38-63` correlates `prob`, `change` and `attn` against Cao et al. RBD binding and expression for WT, BA.1 and BA.2 with `scipy.stats.spearmanr`. There is no published DNMS-versus-DMS number to check an implementation against.

### 7. An arbitrary parent can be scored, and the authors already do it

`FitnessExp.BioTransFitness` constructs with `load_tree=False` and scores supplied strain sequences, which is exactly this repository's protocol. The reusable pieces are `BioTransLanguageModel.compute_probabilities` (forward mode), the semantic-change and attention-change batch calls, and `get_seq_mutation_dict(seq, probabilities, changes, significant_mutations, attn)`, which assembles the per-mutation table. `ParentChildMutate.mutate_single_parent` is not reusable as written — it reaches into `self.seq_parents` and `self.parent_child` — but it is thin over those pieces.

### 8. The environment is python 3.7.9 and torch 1.8, and cannot be this repository's

`requirements.txt`: `python=3.7.9`, `torch~=1.8.0+cu111`, `ray~=1.8.0`, `bio-transformers`, `pandas~=1.3.4`. This repository runs torch 2.6.0 on a far newer python. Arm D's checkout-not-dependency decision was taken over a `torch>=2.7` pin that the code did not need; here the gap is six major torch releases in the other direction and a wrapper library (`bio-transformers`) that is itself unmaintained. **A second environment is unavoidable for any homolog that needs a forward pass — and avoidable entirely for BA.1, which needs none.**

---

## Decisions for the user

**These are the issue's open questions, and it should not be implemented until they are answered.**

### 1. Which homologs this arm covers — and whether the arm is worth having at all

| option | what it costs | what it says |
|---|---|---|
| **A. BA.1 only** *(recommended)* | one reader over a published pickle; no checkpoint, no GPU, no second environment | "here is a published spike language model, scored on the one background it was trained on" — a fair comparison, and the only fair one available |
| **B. All three** | A, plus a 4 GB checkpoint, a python-3.7 environment and two full inference runs | adds two columns that measure DNMS on lineages absent from its corpus; a reviewer reads them as a straw man unless the caption carries the cutoff |
| **C. Drop the arm** | nothing | #1966 recorded DNMS as considered; EPIC #1 could cite that and say why, as it does for Evo 2, EVE and PyR0 |

Option B is the EPIC's current wording, written before any of §2 was measured. **The recommendation is A**, because it is the version of this comparison that answers a question — and because the out-of-sample columns B would add cannot distinguish "DNMS is worse" from "DNMS was never shown this lineage", which is the distinction `fig:dms` exists to keep straight. If A is chosen, the one-homolog shape needs a deliberate home: `predictor_correlations.csv` is per `(predictor, homolog, stratum)` so a predictor with one homolog is representable, but `scripts/dms_summary.py` and every chart that groups by homolog must be checked for an assumption that each predictor fills every group.

### 2. Which metric DNMS's row reports

Pearson *r* is the manuscript's coefficient, and `predictor_correlations.csv` carries `pearson_r`. **A within-parent rank-sum has no scale, so a Pearson against a measured effect is a correlation of a rank against a number** — computable, and not what either side means. The candidates:

- **Report Spearman for DNMS**, as its authors do — but then one bar in a chart of Pearson bars is a different quantity, which is worse than the problem it fixes.
- **Report Pearson of the rank-sum** and say in the caption that the predictor is a ranking — defensible, since `fig:dms`'s question is ordering agreement and Pearson-on-ranks is a monotone-ish summary, but it is not a quantity either paper defines.
- **Score DNMS's grammaticality term alone**, which is a log-probability over the amino acid at a position and so is on the same footing as the DASM's `log p` and antiGen's — and is one of the three series the authors themselves correlate with DMS. This is the only option that puts a comparable quantity in a comparable column. **But it is not DNMS**, and calling it DNMS in the paper would be wrong; it would be "Elkin & Zhu's ProtBERT grammaticality".

**No recommendation.** This is a question about what the figure claims, which is the user's. Whatever is chosen is named in the predictor id, the label and the caption, and the other two options are recorded in the PR body as considered.

### 3. Whether `/fh/fast/matsen_e/shared/elkin-dnms/` is where this lives

It follows arm D's precedent and is the only copy of these files we know of outside the Internet Archive. Confirm the location before the `PROVENANCE.md` is written against it.

---

## Design, once the decisions are made

One rule per homolog in scope, with a `{homolog}` wildcard, in `dms_blosum_scores`' shape (`workflow/figures.smk:587-...`), running a new `scripts/dms_dnms.py`.

**The predictor does not build its own long frame and does not filter.** `dms_models.homolog_cells(homolog)` returns the cells and the founder; the predictor is a per-cell lookup, as `scripts/dms_blosum.py:62-76` and `scripts/dms_leaf_scores.py:85-110` are. It hands its frame to `dms_models.write_scores` with a `Manifest` declaring the sequence it read.

**Under option A** the whole predictor is: read the three pickles, index by the gap-stripped BA.1 parent string, assemble the per-mutation table, apply the DNMS weighting (or take the single term decision 2 chooses), map DNMS's gap-stripped coordinates back to WH1 sites, look each measured cell up, write. No model, no torch, no second environment — and the `config.yaml` entry is a `root`, three `file`s and three `md5`s in the `dms:` block's spelling.

**Coordinate mapping is the one piece of real work.** DNMS removes deletion tokens from every sequence, so the BA.1 parent is 1,267 residues and its position *i* is not WH1 site *i+1*. The repository already holds both frames: `data/antigen/dms_parents.fasta` is the same sequence *with* its 6 deletion columns as `-`, so the map is a single cumulative count over that string. `data/antigen/PROVENANCE.md` also records the one trap in it — the DMS site map places BA.1's three-residue NTD deletion at WH1 143–145 and keeps `D` at 142, while Nextclade and the models' training data place it at 142–144 and keep `D` at 145. The protein is identical, so nothing raises; a join that trusts site 142's wild-type is wrong.

---

## Traps

Each is a silent wrong answer rather than an error.

1. **Position *i* in a DNMS dictionary is not WH1 site *i+1*.** See above. The guard is a value assertion: pick a site where the BA.1 parent's residue is distinctive, and assert that the mapped entry's `parent_aa` matches `homolog_cells`' own.
2. **`train_seq.pkl` and the three value dictionaries are pickles from a third-party archive.** Inspect with `pickletools.genops` before loading, and refuse anything that is not a plain list or a dict whose only globals are numpy's. This was done for all four on 2026-11-15: `train_seq.pkl` has **no** `GLOBAL`, `REDUCE` or `INST` opcode at all, and the three value dictionaries reference only `numpy.core.multiarray scalar` and `numpy dtype`. Keep the check in the code, not only in this issue — the archive is not ours.
3. **The predictor must score every measured cell in its declared strata, or the rule fails.** `dms_summary.summarize` raises on a shortfall rather than reporting a smaller `n`. DNMS's dictionaries cover all 19 substitutions at all 1,267 positions, so the only way to come up short is a mapping that drops the deleted columns' neighbours.
4. **A predictor with no label or no colour fails a test before it fails a chart** — `tests/test_dms_models.py:214-220`, and the measured CIELAB separation docstring in `scripts/panel_charts.py` that a new colour falsifies.
5. **`DMS_PREDICTORS` is pinned** at `tests/test_dms_models.py:201-211`; adding an id without updating it fails the committed suite. `tests/test_rule_inputs.py`'s `ENTRY_POINTS` likewise.
6. **Under option A, nothing currently asserts that a predictor may cover one homolog.** Check `dms_summary.py` and `panel_charts.py` for a completeness assumption before writing the rule, not after.
7. **Do not re-run any training.** This arm trains nothing. Run `snakemake -n` before any snakemake invocation and confirm the `train` job count is unchanged.

---

## Test plan

Bug-catching only; the contract's own refusals are tested in `tests/test_dms_models.py:103-150`.

1. **The pickle gate**: a fixture pickle carrying a `REDUCE` opcode is refused by the loader.
2. **A value test on the coordinate map**: a known `(site, mutant)` on the BA.1 parent resolves to the dictionary entry whose key spells that same substitution in DNMS's gap-stripped frame, and the founder residues agree.
3. **The parent read is the parent declared**: the gap-stripped BA.1 string's sha256 matches the pinned value, and it is a key of all three dictionaries.
4. **The weighting is the paper's**: a hand-built three-column frame reproduces `alpha * rank(-change) + beta * rank(prob) + gamma * rank(-attn)` at `1.5 / 3.0 / 1.0`, including the sign flips, which are the easy thing to get backwards.
5. **The archive digests gate**, through `lib.file_md5` as `scripts/dms.py:222` already does.

---

## Success criteria

1. `snakemake dms` writes `dnms_{homolog}.parquet` for each homolog in scope and `predictor_correlations.csv` gains its rows, with every pre-existing row byte-identical.
2. The manifest beside each Parquet names the archive, its digest, the model's training cutoff (**2022-01-01**) and the sha256 of the sequence scored.
3. A dry run before and after shows the `train` job count unchanged.
4. `/fh/fast/matsen_e/shared/elkin-dnms/PROVENANCE.md` exists and records both archives, their Wayback snapshots, their digests and the 404s that made the mirror necessary.
5. The PR body states, in one sentence a reader of the paper could use, that DNMS's language model was trained on sequences collected before 1 January 2022, that the BA.1 DMS parent is in that training set, and that KP.3.1.1's lineage is absent from its corpus entirely.

---

## Scope boundaries

**In scope:** DNMS as a predictor row, its rule, its tests, its provenance, and the `PROVENANCE.md` for the mirror.

**Out of scope:** retraining or extending DNMS to a later cutoff (§5 — a reimplementation, not a follow-up); CoVFit, evoPLM-Tree and SARITA, which dasm2-experiments#1966 defers or excludes; any change to the existing five predictors' rows; and any second correlation coefficient added to `predictor_correlations.csv` for predictors other than this one — decision 2 is about **this** row.

---

## References

- EPIC #1, **M5d** — the predictor table that lists this arm as unfiled.
- dasm2-experiments#1966 — the survey that recommended DNMS (body, "Finding 3") and the fact-checked comment that qualified it ("a weighted rank-sum … not an effect size"; "Data are NCBI GenBank through 11 Jan 2023, not GISAID"). **The body and the comment disagree about whether DNMS belongs in the panel**; this issue is written against the comment, which is the later and checked one.
- #32 (arm D, antiGen) — the shape this issue follows, and `data/antigen/PROVENANCE.md`, which records the in-sample verification this one mirrors.
- #28 — the predictor contract and the join. #26, #30 — the DMS loader and inputs.
- Elkin & Zhu, *Commun Biol* 8 (2025), [10.1038/s42003-024-07262-7](https://doi.org/10.1038/s42003-024-07262-7); code `maggieelkin/CovMutation` at `v1.0.0`, [10.5281/zenodo.14015344](https://doi.org/10.5281/zenodo.14015344) (code only). PDF: `viral-dasm-tex/papers/s42003-024-07262-7.pdf`.
