"""Prove the three mat-pcp clade rosters name the same clades.

Scoring joins three of mat-pcp's outputs per clade: the founder the model is
scored on, the fitness effects it is scored against, and the censor table that
says which of those mutations training never saw. Each names its clades in a
different way -- a FASTA per clade, a CSV per clade, and one line per clade in
a text file -- and nothing in the mat-pcp run makes them agree.

A mismatch does not error downstream, which is the reason for this guard. A
clade with a founder and no fitness file scores to an all-null join and reads
as a clade the model simply got wrong; a clade with fitness and no founder is
dropped from the grid and never appears at all. Both produce a full-looking
table with fewer clades in it than the run has, and neither says so.

So the roster is read from the files rather than restated in ``config.yaml``,
and this guard requires the three readings to be identical. Adding a clade to
the mat-pcp run then either shows up in all three or fails here.

Two directory contents are deliberately *not* clades:

* ``aamut_fitness/all_aamut_fitness.csv`` is the clade-pooled table. It has no
  ``cluster`` column and its columns are in a different order, so a glob that
  swept it in with the per-clade files would read its second field --
  ``clade_founder_aa`` -- where the per-clade files carry ``gene``, and a
  single-letter residue would pass for a gene name.
* ``aamut_fitness/cap_*_0/`` are the per-arm refits of the same quantity.
  Scoring reads full depth for every arm, so these are a different comparison
  (see ``config.yaml``'s ``aamut_fitness_dir``).

GISAID: clade names are aggregate labels, not sequence or per-sample data, so
the report may be read off the cluster. Nothing it writes contains a sequence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lib

#: mat-pcp writes this beside the founders when the clade set is complete. It
#: is not a clade and must not be read as one.
FOUNDER_SENTINEL = "founders.done"

#: The pooled table's basename inside ``aamut_fitness_dir``.
POOLED_BASENAME = "all_aamut_fitness.csv"

#: Suffix mat-pcp gives each per-clade fitness table.
FITNESS_SUFFIX = "_aamut_fitness.csv"


def founder_clades(founders_dir: Path) -> set[str]:
    """Clades with a founder FASTA, from ``{clade}.fa`` basenames."""
    return {
        path.stem
        for path in sorted(founders_dir.glob("*.fa"))
        if path.name != FOUNDER_SENTINEL
    }


def fitness_clades(fitness_dir: Path) -> set[str]:
    """Clades with a per-clade fitness table, excluding the pooled one.

    Globs the directory's own level rather than walking it, so the per-arm
    refit subdirectories are not reached.
    """
    return {
        path.name[: -len(FITNESS_SUFFIX)]
        for path in sorted(fitness_dir.glob(f"*{FITNESS_SUFFIX}"))
        if path.name != POOLED_BASENAME
    }


def censor_clades(clades_file: Path) -> set[str]:
    """Clades listed in the censor table's companion clade list."""
    return {
        line.strip() for line in clades_file.read_text().splitlines() if line.strip()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, required=True, help="Where to write the JSON report."
    )
    args = parser.parse_args()

    founders_dir = lib.mat_pcp_path("clade_founders_dir")
    fitness_dir = lib.mat_pcp_path("aamut_fitness_dir")
    pooled_file = lib.mat_pcp_path("aamut_fitness_pooled_file")
    clades_file = lib.mat_pcp_path("censored_clades_file")

    rosters = {
        "clade_founders": founder_clades(founders_dir),
        "aamut_fitness": fitness_clades(fitness_dir),
        "censored_clades": censor_clades(clades_file),
    }

    failures: list[str] = []
    for name, roster in rosters.items():
        if not roster:
            failures.append(f"{name} names no clades at all")

    reference = rosters["clade_founders"]
    for name, roster in rosters.items():
        if name == "clade_founders" or roster == reference:
            continue
        missing = sorted(reference - roster)
        extra = sorted(roster - reference)
        failures.append(
            f"{name} does not match clade_founders: "
            f"{len(missing)} missing ({missing[:5]}), "
            f"{len(extra)} extra ({extra[:5]})"
        )

    if not pooled_file.is_file():
        failures.append(f"the pooled fitness table is missing at {pooled_file}")

    report = {
        "gene": lib.GENE,
        "clade_founders_dir": str(founders_dir),
        "aamut_fitness_dir": str(fitness_dir),
        "aamut_fitness_pooled_file": str(pooled_file),
        "censored_clades_file": str(clades_file),
        "n_clades": len(reference),
        "clades": sorted(reference),
        "roster_sizes": {name: len(roster) for name, roster in rosters.items()},
        "passed": not failures,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))

    if failures:
        raise SystemExit(
            "clade guard failed:\n" + "\n".join(f"  - {f}" for f in failures)
        )
    print(f"clade guard passed: {len(reference)} clades agree across three rosters")


if __name__ == "__main__":
    main()
