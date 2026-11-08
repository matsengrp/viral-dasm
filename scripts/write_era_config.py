"""Regenerate the era -> clade map from the mat-pcp run's own era_fit_summary.csv.

The clade partition is not a constant of this pipeline; it is a property of the
mat-pcp run whose rate tables it consumes. ``glm/era_fit_summary.csv`` is the
only artifact recording which clades each era's GLM was actually fitted over,
so this reads it rather than restating the split. Regenerating after an
upstream re-run is how the training corpus and the rate tables are kept from
drifting apart.

The emitted file is a *template*: ``{gene}`` and ``{artifact_root}`` survive
into it, and :func:`render_era_config` fills them. The placeholders are what
make the committed template installation-independent -- a reader with their own
artifact root renders their own archive paths from the same committed file.

Usage::

    python scripts/write_era_config.py --output data/eras-pre-omicron-omicron.yaml

GISAID: the partition is a list of clade names, which is aggregate metadata.
The template is committed; the rendered map is not, because it holds absolute
paths.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

# `archive_name` is the single authority on how an archive is spelled. Imported
# rather than restated: a second hand-spelled stem here could name a file the
# builder never writes, and the failure would be a missing archive at render
# time rather than a mismatch anyone could see.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_era_neutral_models import archive_name
from lib import CUTOFF, VERSION, mat_pcp_path

REQUIRED_SUMMARY_COLUMNS = frozenset({"era", "fitted", "n_clades", "clades"})

# `fitted` spellings accepted. Both sets are enumerated so an unrecognized value
# is an error rather than silently falsey -- see `read_era_partition`.
TRUE_VALUES = frozenset({"true", "1", "t", "yes"})
FALSE_VALUES = frozenset({"false", "0", "f", "no"})

GENE_PLACEHOLDER = "{gene}"
ARTIFACT_ROOT_PLACEHOLDER = "{artifact_root}"
ERA_PLACEHOLDER = "{era}"

# Archives live directly under this pipeline's own artifact root. Upstream's
# template named another experiment's directory under a shared root, which would
# render here to a path this pipeline never writes -- and `render_era_map`'s
# existence check would then pass against that other project's archives and
# nothing would notice.
#
# Every substitution is by `str.replace`, never `str.format`: the template
# deliberately contains braces that must survive into the written file, and
# `.format` would try to fill them and raise.
ARCHIVE_TEMPLATE = (
    f"{ARTIFACT_ROOT_PLACEHOLDER}/neutral_models/"
    f"{archive_name(GENE_PLACEHOLDER, ERA_PLACEHOLDER)}.zip"
)

# `{scheme}` names the lambda scheme -- `per-era` or `pooled`. It lands in the
# Parquet's `neutral_model_name` metadata, which is the only thing in a finished
# file that says which scheme produced it, so two schemes must not share a value.
SCHEME_PLACEHOLDER = "{scheme}"
MODEL_NAME_TEMPLATE = (
    f"{SCHEME_PLACEHOLDER}-TabularRates-SCoV2-{GENE_PLACEHOLDER}-{CUTOFF}-{VERSION}"
)

HEADER = """\
# Era -> clade map for the mat-pcp pre-Omicron / Omicron split.
#
# GENERATED -- do not hand-edit. Regenerate with:
#   python scripts/write_era_config.py --output <this file>
#
# The partition is read from the mat-pcp run's own `glm/era_fit_summary.csv`,
# the only artifact recording which clades each era's GLM was actually fitted
# over. Read rather than restated so the training corpus and the rate tables
# cannot drift apart.
#
# This is a TEMPLATE: `{gene}` and `{artifact_root}` are substituted by the
# Snakefile. The placeholders are what make this committed file
# installation-independent, so a reader with their own mat-pcp run and their own
# artifact root renders their own archive paths from it.
#
# Every clade in the corpus must fall in exactly one era. `prepare_viral_parquet`
# raises on an unmapped clade and on an era matching zero rows, and never
# defaults. Clades that sit in the tree but have no clade founder are absent,
# because mat-pcp emits no PCPs for them.
#
# One map serves every corpus. mat-pcp does not refit the GLM per subsampling
# arm, so the era partition and both eras' rate tables are shared, and the
# archives built from them are the same for every rung of the ladder.
"""


def read_era_partition(summary: Path) -> dict[str, list[str]]:
    """Fitted era -> its clades, from the run's ``glm/era_fit_summary.csv``.

    Raises:
        ValueError: If the summary is missing, lists no fitted era, or gives an
            era no clades. An era that was requested but not fitted has absent
            or stale rate tables, so including it would point the map at a file
            from an earlier run -- which validates, and is therefore worse than
            a missing file.
    """
    if not summary.exists():
        raise ValueError(
            f"{summary} not found. Only a per-era mat-pcp run writes it; a "
            "pooled-only run has no partition to read."
        )
    with summary.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        rows = list(reader)
    # Validate the header rather than letting a renamed column surface as a bare
    # KeyError, which this function's docstring promises will be a ValueError.
    if missing_columns := REQUIRED_SUMMARY_COLUMNS - fieldnames:
        raise ValueError(
            f"{summary} is missing column(s) {sorted(missing_columns)}; it has "
            f"{sorted(fieldnames)}."
        )

    partition: dict[str, list[str]] = {}
    for row in rows:
        fitted = row["fitted"].strip().lower()
        # Reject rather than default. Treating an unrecognized value as "not
        # fitted" would silently drop an era that mat-pcp did fit, yielding a
        # partition that covers less of the corpus than it appears to.
        if fitted not in TRUE_VALUES | FALSE_VALUES:
            raise ValueError(
                f"{summary}: era {row['era']!r} has fitted={row['fitted']!r}, "
                f"which is neither true ({sorted(TRUE_VALUES)}) nor false "
                f"({sorted(FALSE_VALUES)}). Refusing to guess."
            )
        if fitted in FALSE_VALUES:
            continue
        clades = [c for c in row["clades"].split(";") if c]
        if not clades:
            raise ValueError(f"{summary}: era {row['era']!r} lists no clades.")
        # The summary states its own count; use it as a free cross-check on the
        # `;`-split, which a quoting mishap or a truncated field would corrupt.
        declared = int(row["n_clades"])
        if declared != len(clades):
            raise ValueError(
                f"{summary}: era {row['era']!r} declares n_clades={declared} but "
                f"its `clades` field splits into {len(clades)}: {clades}."
            )
        partition[row["era"]] = clades
    if not partition:
        raise ValueError(f"{summary} lists no fitted era.")
    overlap = _overlapping_clades(partition)
    if overlap:
        raise ValueError(
            f"{summary}: clade(s) {sorted(overlap)} appear in more than one era, "
            "so a row's neutral model would depend on iteration order."
        )
    return partition


def _overlapping_clades(partition: dict[str, list[str]]) -> set[str]:
    """Clades claimed by more than one era."""
    seen: set[str] = set()
    overlap: set[str] = set()
    for clades in partition.values():
        overlap |= seen & set(clades)
        seen |= set(clades)
    return overlap


def build_config(partition: dict[str, list[str]], header: str = HEADER) -> str:
    """Render the template YAML text for ``partition``, under ``header``."""
    # Quoted for the same reason the archive path is: the unrendered value starts
    # with `{`, which YAML reads as the opening of a flow mapping rather than as
    # a string. `tests/test_era_map.py` is what catches this.
    lines = [header, f'neutral_model_name: "{MODEL_NAME_TEMPLATE}"', "eras:"]
    for era, clades in partition.items():
        lines.append(f"  {era}:")
        lines.append(f"    # {len(clades)} clades")
        archive = ARCHIVE_TEMPLATE.replace(ERA_PLACEHOLDER, era)
        lines.append(f'    archive: "{archive}"')
        lines.append("    clades:")
        lines.extend(f"      - {clade}" for clade in clades)
    return "\n".join(lines) + "\n"


def render_era_config(template_text: str, gene: str, artifact_root: Path | str) -> str:
    """Substitute ``{gene}`` and ``{artifact_root}`` in a template config.

    Kept next to the writer so the template's placeholders and the code that
    fills them cannot drift. ``str.replace`` rather than ``str.format``, because
    the YAML body is not a format string -- a stray brace in a comment would
    raise.
    """
    return template_text.replace(GENE_PLACEHOLDER, gene).replace(
        ARTIFACT_ROOT_PLACEHOLDER, str(artifact_root)
    )


def read_template_partition(template_text: str) -> dict[str, list[str]]:
    """The era -> clades partition back out of a rendered or unrendered template."""
    spec = yaml.safe_load(template_text)
    return {era: list(body["clades"]) for era, body in spec["eras"].items()}


def build_era_map(template_text: str) -> str:
    """The per-era map YAML, from the committed template.

    Read back from the template rather than re-derived from the run, so the map
    and the committed template cannot disagree: a map built from a second read
    of ``era_fit_summary.csv`` would silently diverge if the template had been
    regenerated against a different run.

    Args:
        template_text: The committed template, still holding ``{gene}`` and
            ``{artifact_root}``.

    Those two placeholders survive into the result for
    :func:`render_era_config` to fill. ``{scheme}`` does not: it is filled here,
    because this function is what decides the scheme is per-era.
    """
    partition = read_template_partition(template_text)
    header = (
        "# Era map generated from the committed era template.\n"
        "#\n"
        "# GENERATED -- do not hand-edit. Regenerate by re-running the Snakefile's\n"
        "# `render_era_map` rule; the generator is scripts/write_era_config.py.\n"
        "#\n"
        "# This is the PER-ERA partition the mat-pcp run's own\n"
        "# `glm/era_fit_summary.csv` records as actually fitted. Every corpus\n"
        "# uses it; lambda is not an axis of this pipeline.\n"
        "#\n"
        "# Still a TEMPLATE: `{gene}` and `{artifact_root}` are substituted at\n"
        "# render time, since one clade partition serves every gene while each\n"
        "# gene has its own archives.\n"
    )
    return build_config(partition, header=header).replace(SCHEME_PLACEHOLDER, "per-era")


def main() -> None:
    """Write the era template from the mat-pcp run ``config.yaml`` names."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    partition = read_era_partition(mat_pcp_path("era_fit_summary_file"))
    text = build_config(partition)
    # Parse before writing: a template that is not valid YAML would otherwise
    # only fail once the Snakefile rendered it, one stage downstream.
    parsed = yaml.safe_load(text)
    if set(parsed["eras"]) != set(partition):
        raise ValueError(
            f"rendered YAML has eras {sorted(parsed['eras'])} but the "
            f"partition has {sorted(partition)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    counts = {era: len(clades) for era, clades in partition.items()}
    print(f"Wrote {args.output}: {counts} ({sum(counts.values())} clades)")


if __name__ == "__main__":
    main()
