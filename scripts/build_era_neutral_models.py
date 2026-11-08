# ruff: noqa: I001 -- import order in this file is load-bearing: `lib` puts this
# repository's root on `sys.path` before anything imports `dasm2`, so the
# vendored fork wins over an editable install pointing at another checkout.
"""Build one gene's per-era TabularRates neutral models from the mat-pcp run.

Thin CLI over :mod:`dasm2.neutral.scov2_rates`, which needs no era-aware code
path: an era is only ever a different ``master_table``. That is what lets the
per-era rates be a drop-in substitute for the pooled ones, so a comparison
between the two measures one change rather than two.

**Built once and shared by every corpus.** mat-pcp does not refit the GLM per
subsampling arm -- an arm rebuilds what counts mutations and reuses what was
fitted once -- so every arm's archives would come out identical anyway, and
building them once is what pins lambda across the ladder. A per-arm rebuild
would let the neutral model move with the corpus, which is the one thing a
depth contrast cannot afford.

Three archives come out: both eras, and the pooled fit. The pooled archive is
not optional -- it is what a pooled-lambda comparison would read, and
``verify_era_neutral_models.py`` checks all three.

Every path comes from ``config.yaml``. Upstream resolved these through
data-central manifests; see ``dasm2/PROVENANCE.md`` for what that cost and what
replaces the guard it carried.

GISAID: reads the mat-pcp run's GLM tables and writes archives of fitted rates
to the artifact root. Both stay on the cluster.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import (
    CUTOFF,
    GENE,
    VERSION,
    assert_dasm2_is_this_repo,
    mat_pcp_path,
    master_table_path,
)

from dasm2.neutral.scov2_rates import (
    GENE_CDS_BLOCKS,
    NeutralBuildInputs,
    build_neutral_model,
)

assert_dasm2_is_this_repo()

logger = logging.getLogger(__name__)

#: ``None`` is the pooled fit. Eras first, so a run that dies partway has still
#: produced the archives the pipeline actually trains against.
BUILD_VARIANTS: tuple[str | None, ...] = ("pre_omicron", "omicron", None)


def archive_name(gene: str, era: str | None) -> str:
    """Archive stem, which is also the model name recorded in its metadata."""
    return (
        f"TabularRates-SCoV2-gisaid-public-{gene}-{CUTOFF}-{VERSION}-{era or 'pooled'}"
    )


def build_inputs(gene: str, era: str | None) -> NeutralBuildInputs:
    """One (gene, era) build's three inputs, all from ``config.yaml``.

    The era changes exactly one of them. ``site_annotations`` is genome-wide and
    ``founder_fasta`` is a property of a clade rather than of a GLM fit, so
    neither is era-stratified -- which is the whole reason an era can be "a
    different ``master_table``" and nothing else.
    """
    return NeutralBuildInputs(
        gene=gene,
        master_table=master_table_path(era),
        site_annotations=mat_pcp_path("site_annotations_file"),
        founder_fasta=mat_pcp_path("reference_sequence_file"),
    )


def main() -> None:
    """Build the pooled and per-era archives for one gene."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gene", default=GENE, choices=sorted(GENE_CDS_BLOCKS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the archives into",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite archives that already exist",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.gene != GENE:
        raise SystemExit(
            f"ERROR: --gene is {args.gene!r} but config.yaml's mat_pcp.gene is "
            f"{GENE!r}. Every path this script reads comes from that one block, "
            f"so building another gene from it would pair this gene's founder "
            f"and site annotations with that gene's name."
        )

    outputs = {
        era: args.output_dir / f"{archive_name(args.gene, era)}.zip"
        for era in BUILD_VARIANTS
    }
    # Refuse-and-exit rather than skip-if-exists. Silently skipping would defeat
    # the staleness gate: `build_neutral` declares `scov2_rates.py` as an input
    # precisely so a fix to `select_gene_rates` forces a rebuild, and a script
    # that skipped existing outputs would leave the stale archive in place.
    if not args.force:
        existing = sorted(str(p) for p in outputs.values() if p.exists())
        if existing:
            raise SystemExit(
                f"ERROR: {len(existing)} archive(s) already exist: {existing}. "
                "Pass --force to overwrite, having checked that the existing "
                "ones are not the inputs of a run in progress."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for era, output in outputs.items():
        build_neutral_model(
            build_inputs(args.gene, era), output, archive_name(args.gene, era)
        )


if __name__ == "__main__":
    main()
