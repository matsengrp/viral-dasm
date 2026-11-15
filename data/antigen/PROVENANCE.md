# Where these two files came from

`fig:dms`'s predictor comparison sets the DASM against antiGen, so this directory carries the antiGen checkpoint that row is scored by and the sequences it is fed. `config.yaml`'s `antigen` block pins each one's md5 and declares how `scripts/dms_antigen.py` uses them; this file records where the bytes came from, under what terms, and the one fact about them that no digest can state — that each scored sequence is in the model's own training split.

Separate from `data/dms/PROVENANCE.md`, which documents the three published DMS datasets. These are a model and its inputs, not measurements.

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

**This is the grid's last cutoff, 5.93893, and that is the reason it is the one.** antiGen's paper sweeps a time cutoff and trains one model per rung on the branches whose parent node predates it. Only the last rung's training split contains all three DMS parent sequences, so only it scores every cell of the comparison with a model that saw that cell's background. The in-sample property is verified rather than inferred from dates: each of the three sequences in `dms_parents.fasta` occurs exactly once in the authors' dumped examples for this cutoff, `examples/output_sc2_grid/cutoff_5.939/examples_cutoff_5.93893.txt`, as example `65080` / `91851` / `105443`, each labelled `TRAINING` and each carrying a non-empty `MUTATIONS_TO_CHILDREN` block, so none is dropped by antiGen's `ignore_dead_ends`. `tests/test_dms_antigen.py` re-runs that check wherever the examples file is readable.

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
| md5 | `87903175ad401a2ff567bbec6221336d` |
| format | three records, wrapped at 60 columns, named by homolog as `config.yaml` names them |

Each record is the **WH1 projection of the construct that assay expressed**, 1,273 columns, carrying the lineage's real deletions as `-`:

| homolog | clade | deletion columns | sha256 |
|---|---|---:|---|
| BA.1 | 21K | 6 | `5f3398d44f945d0b62bed6f8bb602243007a02f0e666696f2aa2ed5cba053140` |
| XBB.1.5 | 23A | 4 | `92622b76bd6b5fb70b81e26e0d0778b48e9ce8e7bd173cf7c9757f20f3d134ae` |
| KP.3.1.1 | 24E | 9 | `de144a5e5047addeb4d788aae2dc3fffc07cf66a767106d61c12b7dc2c01818d` |

**Not the clade founder**, which is what BLOSUM62, the leaf frequency predictor and the DASM all score against. The founders carry no gaps at all: BA.1's presents `H69 V70`, `G142 V143 Y144` and `N211` to a model whose training data carries those as deletions. The two declarations differ at 8 / 6 / 11 columns and `scripts/dms.py`'s masked-site and founder-disagreement filters have already removed every one of them, so the choice moves no measured cell — it moves only the 1,273-residue context the transformer conditions on. Nothing in `scripts/dms_models.py`'s contract can distinguish the two, which is why the gap counts and hashes above are asserted in `tests/test_dms_antigen.py`; they are the only live guard.

**BA.1's deletion convention is not the DMS site map's.** #1962's `training_frame_adjustment` records it: the site map places the three-residue NTD deletion at WH1 143–145 and keeps `D` at 142, while Nextclade and antiGen's own reconstruction place it at 142–144 and keep `D` at 145. The protein is identical either way and nothing raises; these sequences use the training convention, which is what the model was fitted in.

#1962 also stores an `antigen_training_residue` counterfactual for XBB.1.5 — the same string with every column that departs from its nearest antiGen training example set to that example's residue. It differs at exactly one column, site 490 (`F` against `S`), and site 490 is dropped by `load_dms`'s founder-mismatch filter, so it cannot move a single measured cell. It is not carried here for that reason.
