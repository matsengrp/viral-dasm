"""Reduce DNMS's three released dictionaries to one small table for BA.1.

**Run once, offline.** The three value dictionaries are 772 MB between them and
hold roughly 8.7 million floats across 360 parent sequences; the opcode gate
below walks every byte of them in pure Python before anything is unpickled.
That is minutes of work, and `marimo run` re-executes its notebook for every
browser session -- so the notebook reads this script's output rather than the
archive, and nobody waits.

What comes out is every substitution DNMS scores on the BA.1 DMS parent:
1,267 positions x 19 mutant residues = 24,073 rows, already mapped from DNMS's
gap-stripped frame onto WH1 sites.

    uv run python exploration/dnms/prep_dnms_ba1.py
"""

from __future__ import annotations

import io
import os
import pickle
import pickletools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

#: The mirror of the authors' archive, because their own host 404s both of its
#: downloads -- see `README.md`. This is a cluster path, which tracked
#: *pipeline* code may not carry (`dasm2/PROVENANCE.md` treats a leftover one
#: as a defect); this directory is exploration rather than pipeline, and the
#: environment variable means a reader holding the archive somewhere else needs
#: no edit.
DNMS_ROOT = Path(
    os.environ.get("DNMS_ROOT", "/fh/fast/matsen_e/shared/elkin-dnms/ncbi_tree_v1")
)
OUTPUT = HERE / "dnms_ba1_scores.parquet"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import lib
from lib import assert_dasm2_is_this_repo

import dasm2  # noqa: F401

assert_dasm2_is_this_repo()

import dms_antigen
import numpy as np
import pandas as pd
from scipy.stats import rankdata

#: The authors' own CLI defaults (`bin/ParentChildMutate.py` argparse, and the
#: values `examples/run_dnms.ipynb` passes). NOT
#: `bin/MutationRankingResults.py`'s, which default to 1/1/1.
ALPHA, BETA, GAMMA = 1.5, 3.0, 1.0

#: The globals a numpy-bearing pickle legitimately needs.
ALLOWED_GLOBALS = {
    "numpy",
    "numpy.core.multiarray",
    "numpy._core.multiarray",
    "_codecs",
}


def load_guarded(path: Path):
    """Unpickle a third-party archive member, after reading its opcodes.

    These files came from someone else's web host by way of the Internet
    Archive. Unpickling is arbitrary code execution, so the opcode stream is
    walked first and anything that would construct an object we did not expect
    refuses the load. This is the expensive step, and the reason this script
    exists separately from the notebook.
    """
    raw = path.read_bytes()
    seen = set()
    for op, arg, _pos in pickletools.genops(io.BytesIO(raw)):
        if op.name in ("GLOBAL", "INST", "STACK_GLOBAL"):
            seen.add(arg if isinstance(arg, str) else "<stack_global>")
    unexpected = [g for g in seen if g.split(" ")[0] not in ALLOWED_GLOBALS]
    if unexpected:
        raise ValueError(f"{path.name} references unexpected globals {unexpected}")
    print(
        f"  {path.name}: gate passed ({len(raw) / 1e6:.0f} MB, globals {sorted(seen) or 'none'})"
    )
    return pickle.load(open(path, "rb"))


def main() -> None:
    # **Two coordinate frames, and the map between them.** DNMS strips deletion
    # tokens from every sequence (`bin/FitnessExp.py`: `seq.replace('-','')`),
    # so its position `i` is not WH1 site `i`. `data/antigen/dms_parents.fasta`
    # carries the same BA.1 construct *with* its six deletion columns, so the
    # map is one cumulative count over that record.
    gapped = dms_antigen.read_parents(lib.antigen_model().parents)["BA.1"]
    stripped = gapped.replace("-", "")
    site_of = [column + 1 for column, residue in enumerate(gapped) if residue != "-"]
    if len(site_of) != len(stripped):
        raise ValueError(
            f"{len(site_of)} mapped sites against {len(stripped)} residues"
        )
    print(
        f"BA.1 construct: {len(gapped)} columns, {gapped.count('-')} deletions, "
        f"{len(stripped)} residues in DNMS's frame"
    )

    print("loading the released dictionaries (the slow part):")
    grammaticality = load_guarded(DNMS_ROOT / "tree_v1_seq_forward_prob_ft.pkl")[
        stripped
    ]
    semantic = load_guarded(DNMS_ROOT / "tree_v1_seq_l2_change_ft.pkl")[stripped]
    attention = load_guarded(DNMS_ROOT / "tree_v1_seq_attn_ft.pkl")[stripped]

    rows = []
    for key, change in semantic.items():
        position = int(key[1:-1])  # 1-based, in DNMS's gap-stripped frame
        parent_aa, mutant_aa = key[0], key[-1]
        if stripped[position - 1] != parent_aa:
            raise ValueError(
                f"{key}: parent residue disagrees with the stripped sequence"
            )
        rows.append(
            {
                "aa_site": site_of[position - 1],
                "dnms_parent_aa": parent_aa,
                "mutant_aa": mutant_aa,
                "grammaticality": float(grammaticality[position - 1][mutant_aa]),
                "semantic_change": float(change),
                "attention_change": float(attention[key]),
            }
        )
    scores = pd.DataFrame(rows)

    # **Ranked over the whole candidate set, which is what "within a parent"
    # means.** The ranking is taken over all of this parent's substitutions,
    # before any restriction to the cells an assay happened to measure. Ranking
    # after the restriction would be a different quantity -- which is why it is
    # done here, where the full set is in hand, rather than in the notebook.
    acquisition = (
        ALPHA * rankdata(-scores["semantic_change"])
        + BETA * rankdata(scores["grammaticality"])
        + GAMMA * rankdata(-scores["attention_change"])
    )
    scores["dnms"] = rankdata(acquisition)
    scores["log_grammaticality"] = np.log(scores["grammaticality"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(OUTPUT, index=False)
    print(
        f"\nwrote {OUTPUT}: {len(scores):,} substitutions over "
        f"{scores['aa_site'].nunique():,} sites, {OUTPUT.stat().st_size / 1e6:.2f} MB"
    )


if __name__ == "__main__":
    main()
