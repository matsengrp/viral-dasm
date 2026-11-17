# Where these two files came from

`fig:dms`'s predictor comparison sets the DASM against antiGen, so this directory carries the antiGen checkpoint that row is scored by and the sequences it is fed. `config.yaml`'s `antigen` block pins each one's md5 and declares how `scripts/dms_antigen.py` uses them; this file records where the bytes came from, under what terms, and the one fact about them that no digest can state — that each scored sequence is in the model's own training split.

Separate from `data/dms/PROVENANCE.md`, which documents the four published DMS datasets. These are a model and its inputs, not measurements.

## `fitness_simple_d128_n2_lr0.0001_20261112_104225.pt`

| | |
|---|---|
| model | antiGen `fitness_predictor`, trained `seq_len` 1273, fixed coordinate frame |
| source | `/fh/fast/matsen_e/shared/specht-antigen/checkpoints/sc2_grid_cutoff5.939/` |
| reproduction write-up | `/fh/fast/matsen_e/shared/specht-antigen/reports/2026-11-12-sc2-cutoff-grid.md` |
| trained with | [`evo-design/antiGen`](https://github.com/evo-design/antiGen) at `de963f685a575157b802b087bc1b6022f19ef490` (2026-09-05) |
| licence | MIT, `LICENSE`, `Copyright (c) 2026 Ivan Specht` |
| taken on | 2026-11-12 |
| size | 6,823,503 bytes |
| md5 | `cd560fda4bd0e151a60d21238ab0514b` |
| sha256 | `c06bfd707a0928779ad141bb66da260dbe909a5b59bdd65c74b45302bc2d8f20` |
| byte-identical to the source | yes |
| recipe | the grid's own: 16 epochs, batch 64, lr 1e-4, capped at 10,000 batches; seed 42; trained on an L40S |

**This is the grid's latest cutoff, and that is the reason it is the one.** antiGen's paper sweeps a time cutoff and trains one model per rung on the branches whose parent node predates it; the sc2 grid is 18 rungs at `cutoff_years` 0.352, 0.681, 1.010 … 5.610, **5.939**, and this is the last of them — the model trained on the most data. The goal is not recency for its own sake: the comparison asks what antiGen predicts about a scan that has **already been performed**, so every DMS parent sequence must be in its training split, and the earlier rungs predate the later lineages. Only the last rung has them all, so only it scores every cell of the comparison with a model that saw that cell's background. The only other sc2 checkpoints in that tree, `checkpoints/sc2_cutoff1_*`, split at `cutoff_years` 1.0 and are out of sample for all of them. Were a later rung trained, it would be preferred for the same reason, and `config.yaml`'s md5 gate makes that swap deliberate. The in-sample property is verified by reading the split rather than inferred from dates: that cutoff's dumped examples, `<checkout>/../examples/output_sc2_grid/cutoff_5.939/examples_cutoff_5.93893.txt` — beside the checkout `config.yaml` names, not inside it — hold 145,326 records, 143,066 labelled `TRAINING` and 2,260 `TESTING`. BA.1, XBB.1.5 and KP.3.1.1 each occur exactly once, as example `65080` / `91851` / `105443`. Delta occurs **six** times, as examples `10474`, `15933`, `16798`, `17066`, `17266` and `19211`: its spike is carried by six nodes that differ elsewhere in the genome, which is unsurprising for a background that circulated widely before the cutoff. Every one of those nine records is labelled `TRAINING` and carries a non-empty `MUTATIONS_TO_CHILDREN` block, so none is dropped by antiGen's `ignore_dead_ends`. `tests/test_dms_antigen.py` re-runs that check wherever the examples file is readable; because antiGen scores a sequence rather than a node, it requires at least one occurrence and holds both conditions on every occurrence.

Two caveats a reader of the antiGen row needs, neither of which this file can fix:

- **One seed per cutoff**, seed 42, which the reproduction's own §5 calls "one draw, not a centre". The antiGen row is a single-seed point estimate and carries no error bar.
- **This is not the paper's headline run**, which is at `checkpoints/sc2_cutoff1_run2/` under different training settings. This is the grid, reproduced here, and the grid's PSSM baseline differs from the authors' published one by 0.02–0.07.

The code is reached through `config.yaml`'s `antigen.checkout` rather than committed or taken as a dependency; that entry records why.

## `dms_parents.fasta`

| | |
|---|---|
| source repository | [`matsengrp/dasm2-experiments`](https://github.com/matsengrp/dasm2-experiments), issue #1962, branch `1939-dms-parent-sequences` |
| path there | `experiments/2026-10-30-external-dms-benchmarks/dms_parents.json`, each homolog's `model_input.sequence` |
| source commit | `4972e582` |
| taken on | 2026-11-12 |
| md5 | `ed52ae34f4178f34e5acdb237d893cf4` |
| format | four records, wrapped at 60 columns, named by homolog as `config.yaml` names them |

Delta's record was added later than the other three and not taken from #1962, which had no 1,273-column projection for it. It was built by the recipe that issue records, and the implementation was validated by reproducing BA.1's committed record byte for byte, including its sha256, before being run on Delta.

Each record is the **WH1 projection of the construct that assay expressed**, 1,273 columns, carrying the lineage's real deletions as `-`:

| homolog | clade | deletion columns | sha256 |
|---|---|---:|---|
| Delta | 21J | 2 | `bbe63422d31d251ec9e4646dea5c99ab1b6a761d7d79a1992665de47c0537634` |
| BA.1 | 21K | 6 | `5f3398d44f945d0b62bed6f8bb602243007a02f0e666696f2aa2ed5cba053140` |
| XBB.1.5 | 23A | 4 | `92622b76bd6b5fb70b81e26e0d0778b48e9ce8e7bd173cf7c9757f20f3d134ae` |
| KP.3.1.1 | 24E | 9 | `de144a5e5047addeb4d788aae2dc3fffc07cf66a767106d61c12b7dc2c01818d` |

**Not the clade founder**, which is what BLOSUM62, the leaf frequency predictor and the DASM all score against. The founders carry no gaps at all: BA.1's presents `H69 V70`, `G142 V143 Y144` and `N211` to a model whose training data carries those as deletions. The two declarations differ at 3 / 8 / 6 / 11 columns (Delta / BA.1 / XBB.1.5 / KP.3.1.1) and `scripts/dms.py`'s masked-site and founder-disagreement filters have already removed every one of them, so the choice moves no measured cell — it moves only the 1,273-residue context the transformer conditions on. Nothing in `scripts/dms_models.py`'s contract can distinguish the two, which is why the gap counts and hashes above are asserted in `tests/test_dms_antigen.py`; they are the only live guard.

**BA.1's deletion convention is not the DMS site map's.** #1962's `training_frame_adjustment` records it: the site map places the three-residue NTD deletion at WH1 143–145 and keeps `D` at 142, while Nextclade and antiGen's own reconstruction place it at 142–144 and keep `D` at 145. The protein is identical either way and nothing raises; these sequences use the training convention, which is what the model was fitted in.

**Delta needs the same reconciliation, and the training data settles it the same way.** Its `E156G` with `del157/158` can be written `G--` at WH1 156–158, which is the frame the DMS site map uses, or `--G`, which is the training convention. The examples dump decides it: the `--G` spelling occurs 6 times and `G--` occurs 0 times, so that is what is committed. The ungapped protein is the same either way. The DMS measures site 156 with wildtype `G` and has no rows at 157 or 158, and `load_dms` drops all eleven site-156 cells, so `cell_scores`'s `rows = sites - 1` never indexes a gap column. The choice is still worth recording, because scoring the `G--` spelling instead moves Delta's antiGen correlation by about 0.02 (single-nt 0.411 against the committed 0.388).

#1962 also stores an `antigen_training_residue` counterfactual for XBB.1.5 — the same string with every column that departs from its nearest antiGen training example set to that example's residue. It differs at exactly one column, site 490 (`F` against `S`), and site 490 is dropped by `load_dms`'s founder-mismatch filter, so it cannot move a single measured cell. It is not carried here for that reason.
