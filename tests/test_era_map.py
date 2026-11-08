"""The era template renders to real archive paths, and to this pipeline's.

The template is committed with `{gene}` and `{artifact_root}` unfilled, which
is what makes it installation-independent. Two things can go wrong with that
and neither shows up as an error: a placeholder survives into an archive path,
and nothing notices until the archive is opened; or the path renders cleanly
but points somewhere this pipeline never writes, in which case the Snakefile's
existence check passes against a file it did not build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib import ARTIFACT_ROOT, CUTOFF, GENE, VERSION
from write_era_config import (
    build_era_map,
    read_template_partition,
    render_era_config,
)

TEMPLATE = REPO_ROOT / "data" / "eras-pre-omicron-omicron.yaml"
PLACEHOLDERS = ("{gene}", "{artifact_root}", "{era}", "{scheme}")


def rendered() -> dict:
    return yaml.safe_load(
        render_era_config(build_era_map(TEMPLATE.read_text()), GENE, ARTIFACT_ROOT)
    )


def test_the_unrendered_template_is_valid_yaml():
    """A value starting with `{` is a flow mapping unless it is quoted."""
    spec = yaml.safe_load(TEMPLATE.read_text())
    assert isinstance(spec["neutral_model_name"], str)
    for body in spec["eras"].values():
        assert isinstance(body["archive"], str)


def test_no_placeholder_survives_rendering():
    text = render_era_config(build_era_map(TEMPLATE.read_text()), GENE, ARTIFACT_ROOT)
    survivors = [token for token in PLACEHOLDERS if token in text]
    assert not survivors, (
        f"{survivors} survived rendering, so an archive path is not a real path "
        f"and the failure would surface only when the archive was opened."
    )


def test_archives_render_under_this_pipelines_artifact_root():
    """Not under another project's directory, which would read as a pass.

    The template this was ported from named the directory of the experiment
    that built the archives. Rendered here, that path exists and holds real
    archives -- so `render_era_map`'s existence check would pass while every
    corpus was built against a neutral model this pipeline never produced.
    """
    for era, body in rendered()["eras"].items():
        archive = Path(body["archive"])
        assert archive.is_relative_to(ARTIFACT_ROOT), (
            f"era {era!r} renders to {archive}, outside this pipeline's "
            f"artifact root {ARTIFACT_ROOT}."
        )
        assert archive.parent.name == "neutral_models"


def test_archive_stems_are_the_ones_the_builder_writes():
    """One spelling, so the map cannot name a file `build_neutral` never writes."""
    from build_era_neutral_models import BUILD_VARIANTS, archive_name

    want = {f"{archive_name(GENE, era)}.zip" for era in BUILD_VARIANTS}
    got = {Path(body["archive"]).name for body in rendered()["eras"].values()}
    assert got <= want, f"the map names {sorted(got - want)}, which nothing builds"


def test_the_model_name_records_the_scheme_and_the_run():
    name = rendered()["neutral_model_name"]
    assert name == f"per-era-TabularRates-SCoV2-{GENE}-{CUTOFF}-{VERSION}"


def test_every_clade_falls_in_exactly_one_era():
    partition = read_template_partition(TEMPLATE.read_text())
    clades = [clade for era_clades in partition.values() for clade in era_clades]
    assert len(clades) == len(set(clades))
    assert len(clades) == 56
