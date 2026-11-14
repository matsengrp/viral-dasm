"""Which substitutions define a clade's two genetic backgrounds.

`fig:within_clade_shifts`' panels B and D ask what a mutation's effect is on
two backgrounds within one clade. This module decides what those two
backgrounds *are*, from the tree alone.

**Frequency, not the model.** The earlier pipeline chose a background marker by
scoring every parent haplotype under the DASM, splitting at the median, and
taking the site that best separated the halves. That works, but the answer
moves with the checkpoint: on the landed model, site 356 shifts `D574V` by only
0.31 nats and misses the 0.4-nat bar, so 22D's background is named by two sites
rather than three and the measured shift changes with it. A background defined
by how much of the clade carries what is a property of the tree, so every
checkpoint is measured against the same contrast, and it is a sentence long in
the Methods rather than a paragraph.

**It gives the same answer.** On the two clades the figure draws, marker plus
majority co-travellers reproduces the DASM-led selection exactly: 22D's
``R346T + K356T + F490S`` and 20B's ``P681H + T478K + T732A``.

**The three gates, and why each is needed.** A substitution carried by nearly
the whole clade is not a difference between two backgrounds -- in 22D every
observed haplotype carries ``D142G``, so splitting on it would put nothing on
one side. Those are *fixed* and belong to both backgrounds. Of what is left,
the most abundant substitution is the **marker**. Anything carried by most of
the marker's own branches is a **co-traveller**: no split of this clade
separates it from the marker, so leaving it out would make the two backgrounds
differ at it by accident rather than by design.

GISAID: reads GISAID-derived parent sequences, and writes only site, residue
and branch-mass fractions.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from dasm2.sequence.conversion import translate_sequence

from panels import PCP_CHUNK_ROWS, _founder_nt_sequence

#: Above this share of the clade's branch mass, a substitution is *fixed*: it
#: belongs to both constructed backgrounds rather than distinguishing them.
#: Inherited from `dasm2.analysis.within_clade_pools.CLADE_FIXED_FRACTION`,
#: which the DASM-led pipeline used for the same purpose, so switching how the
#: marker is chosen does not quietly also move this.
CLADE_FIXED_FRACTION = 0.95

#: Share of the marker's own branch mass a substitution must reach to count as
#: travelling with it rather than being separable from it. Inherited from
#: `within_clade_pools.MAJORITY_COTRAVELLER_FRACTION` for the same reason.
MAJORITY_COTRAVELLER_FRACTION = 0.5

#: Branch-mass share the marker itself must reach for the clade to *have* two
#: backgrounds worth contrasting. Below it the clade has no sweep, so the
#: "background with" side is a minority of a minority and the contrast is
#: between the clade and a corner of it. Not inherited: the DASM-led search had
#: no equivalent, because it chose a marker by effect rather than by abundance
#: and so could pick a site that almost nothing carries.
MIN_MARKER_FRACTION = 0.10

#: The unknown-residue code. A parent with an ambiguous codon at a site is not
#: evidence either way about that site and is counted at neither residue.
UNKNOWN_AA = "X"


def substitution_mass(
    clades: list[str], pcp_path: str | Path, *, marker: dict[str, tuple[int, str]] | None = None
) -> tuple[dict[str, dict[tuple[int, str], float]], dict[str, float]]:
    """Branch mass per non-founder residue, per clade.

    One pass over the corpus. Each parent is compared to its clade founder at
    the nucleotide level first and only the differing codons are translated:
    a parent differs from the founder at a handful of sites, so translating
    whole sequences would spend 1,273 lookups per row to find three.

    Args:
        clades: The clades to tally.
        pcp_path: The corpus the branches come from.
        marker: When given, restricts the tally to branches whose parent
            carries ``{clade: (1-indexed site, residue)}`` -- which is what
            turns this from a frequency into a co-travelling frequency. The
            denominator returned is then the marker's mass, not the clade's.

    Returns:
        ``({clade: {(site, residue): mass}}, {clade: denominator})``.
    """
    founder_nt = {clade: _founder_nt_sequence(clade) for clade in clades}
    founder_bytes = {
        clade: np.frombuffer(sequence.encode(), dtype="S1")
        for clade, sequence in founder_nt.items()
    }
    codon_count = len(next(iter(founder_nt.values()))) // 3
    founder_aa = {
        clade: [
            translate_sequence(sequence[3 * index : 3 * index + 3])
            for index in range(codon_count)
        ]
        for clade, sequence in founder_nt.items()
    }

    mass: dict[str, dict[tuple[int, str], float]] = {
        clade: defaultdict(float) for clade in clades
    }
    denominator = dict.fromkeys(clades, 0.0)
    for chunk in pd.read_csv(
        pcp_path, usecols=["clade", "parent", "branch_length"], chunksize=PCP_CHUNK_ROWS
    ):
        chunk = chunk[chunk["clade"].isin(clades)]
        if chunk.empty:
            continue
        for clade, part in chunk.groupby("clade"):
            lengths = part["branch_length"].to_numpy(float)
            parents = part["parent"].tolist()
            if marker is not None:
                site, residue = marker[clade]
                start = 3 * (site - 1)
                carried = np.fromiter(
                    (
                        translate_sequence(parent[start : start + 3]) == residue
                        for parent in parents
                    ),
                    dtype=bool,
                    count=len(parents),
                )
                keep = np.nonzero(carried)[0]
                if not len(keep):
                    continue
                parents = [parents[index] for index in keep]
                lengths = lengths[keep]
            denominator[clade] += lengths.sum()
            block = np.frombuffer("".join(parents).encode(), dtype="S1").reshape(
                len(parents), -1
            )
            differs = (
                (block != founder_bytes[clade])
                .reshape(len(parents), codon_count, 3)
                .any(axis=2)
            )
            rows, codons = np.nonzero(differs)
            for row, codon in zip(rows.tolist(), codons.tolist()):
                residue = translate_sequence(
                    parents[row][3 * codon : 3 * codon + 3]
                )
                if residue == founder_aa[clade][codon] or residue == UNKNOWN_AA:
                    continue
                mass[clade][(codon + 1, residue)] += lengths[row]
    return mass, denominator


def clade_backgrounds(clades: list[str], pcp_path: str | Path) -> pd.DataFrame:
    """The substitutions that define each clade's two backgrounds.

    Two passes over the corpus: one to rank substitutions by branch mass and
    pick the marker, one to ask what travels with it. The second cannot be
    folded into the first because the marker is not known until the first ends.

    Raises:
        ValueError: If a clade's most abundant substitution does not reach
            :data:`MIN_MARKER_FRACTION`. A clade with no sweep has no two
            backgrounds to contrast, and silently contrasting its largest
            minority against everything else would draw a panel whose caption
            is wrong about what it shows.

    Returns:
        One row per substitution, with ``role`` one of ``marker``,
        ``cotraveller`` or ``fixed``, ``clade_fraction`` its share of the
        clade's branch mass, and ``marker_fraction`` its share of the marker's.
        The fixed rows are carried rather than dropped: they say which
        substitutions both backgrounds hold, which is what makes the two
        constructed sequences ones that occur.
    """
    mass, total = substitution_mass(clades, pcp_path)
    marker: dict[str, tuple[int, str]] = {}
    fixed: dict[str, set[tuple[int, str]]] = {}
    for clade in clades:
        shares = {key: value / total[clade] for key, value in mass[clade].items()}
        fixed[clade] = {
            key for key, share in shares.items() if share > CLADE_FIXED_FRACTION
        }
        free = {key: share for key, share in shares.items() if key not in fixed[clade]}
        top, share = max(free.items(), key=lambda item: item[1])
        if share < MIN_MARKER_FRACTION:
            raise ValueError(
                f"{clade}'s most abundant substitution reaches only "
                f"{share:.1%} of its branch mass, under the "
                f"{MIN_MARKER_FRACTION:.0%} a background contrast needs. The "
                f"clade has no sweep to split on."
            )
        marker[clade] = top

    carried, marker_total = substitution_mass(clades, pcp_path, marker=marker)
    rows = []
    for clade in clades:
        founder = _founder_aa_at(clade)
        shares = {key: value / total[clade] for key, value in mass[clade].items()}
        with_marker = {
            key: value / marker_total[clade] for key, value in carried[clade].items()
        }
        for key in sorted(fixed[clade] | {marker[clade]} | set(with_marker)):
            site, residue = key
            if key in fixed[clade]:
                role = "fixed"
            elif key == marker[clade]:
                role = "marker"
            elif with_marker.get(key, 0.0) > MAJORITY_COTRAVELLER_FRACTION:
                role = "cotraveller"
            else:
                continue
            rows.append(
                {
                    "clade": clade,
                    "mutation": f"{founder[site - 1]}{site}{residue}",
                    "aa_site": site,
                    "mutant_aa": residue,
                    "role": role,
                    "clade_fraction": shares.get(key, 0.0),
                    "marker_fraction": with_marker.get(key, 0.0),
                }
            )
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["clade", "role", "clade_fraction"], ascending=[True, True, False]
    ).reset_index(drop=True)


def _founder_aa_at(clade: str) -> list[str]:
    """One clade's founder residues, 0-indexed by codon."""
    sequence = _founder_nt_sequence(clade)
    return [
        translate_sequence(sequence[3 * index : 3 * index + 3])
        for index in range(len(sequence) // 3)
    ]


def main() -> None:
    """Write the background definition for the figure's clades."""
    import argparse

    from panel_spec import BACKGROUND_CSV, CANDIDATE_CLADES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcp", required=True, help="the PCP corpus to read")
    parser.add_argument("--out", default=str(BACKGROUND_CSV), help="where to write")
    args = parser.parse_args()

    frame = clade_backgrounds(list(CANDIDATE_CLADES), args.pcp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
