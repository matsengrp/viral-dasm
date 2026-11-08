"""The modules the corpus build needs are vendored, and are still usable.

#4 pruned the tree against a trace of *training*, which reads a Parquet that
already carries `nt_rates` and `nt_csps`. Building that Parquet is a different
code path, so three modules came back for it (`dasm2/PROVENANCE.md`). Two of
them had to be forked on the way in, and each fork is a way the module could
be present but broken -- which is what these check.

None of this reads a corpus. The expensive checks that a rebuilt corpus
reproduces the paper's fold live in `workflow/corpus.smk`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

VENDORED_FOR_THE_BUILD = (
    "dasm2.neutral.tabular_rates",
    "dasm2.neutral.scov2_rates",
    "dasm2.data.parquet_writer",
)


@pytest.mark.parametrize("module_name", VENDORED_FOR_THE_BUILD)
def test_the_build_modules_import_from_this_repository(module_name):
    """Each imports, and lands here rather than in an editable install."""
    import importlib

    from lib import assert_dasm2_is_this_repo

    module = importlib.import_module(module_name)
    assert Path(module.__file__).resolve().is_relative_to(REPO_ROOT)
    assert_dasm2_is_this_repo()


def test_the_neutral_model_is_still_concrete():
    """`TabularRatesNeutralModel` can be constructed.

    `AbstractNeutralModel` declares `to_config` abstract. The fork cannot
    return what upstream's returned -- #4 deleted the neutral-model config
    classes -- but deleting the override instead of replacing it leaves an
    abstract class, and `TypeError: Can't instantiate abstract class` is then
    the first thing a corpus build hits, hours of input reading later.
    """
    from dasm2.neutral.tabular_rates import TabularRatesNeutralModel

    assert not TabularRatesNeutralModel.__abstractmethods__, (
        f"TabularRatesNeutralModel is abstract on "
        f"{sorted(TabularRatesNeutralModel.__abstractmethods__)}, so it cannot "
        f"be constructed and no corpus can be built."
    )


def test_to_config_refuses_rather_than_raising_an_import_error():
    """The override says why it cannot work, instead of failing on an import."""
    from dasm2.neutral.tabular_rates import TabularRatesNeutralModel

    model = TabularRatesNeutralModel.__new__(TabularRatesNeutralModel)
    model._archive_path = None
    with pytest.raises(NotImplementedError, match="no TabularRatesNeutralModelConfig"):
        model.to_config()


def test_scov2_rates_resolves_no_manifests():
    """The registry import is gone, and with it the dependency #4 removed.

    `viral_fitness_registry` pulls in `discovery`, `manifests` and
    `registry_schema`. If this module reaches for any of them again, the fork
    has quietly reacquired ~2,200 lines and the README's claim that every path
    is in `config.yaml` has stopped being true.
    """
    source = (REPO_ROOT / "dasm2" / "neutral" / "scov2_rates.py").read_text()
    for forbidden in (
        "viral_fitness_registry",
        "dasm2.data.discovery",
        "dasm2.data.manifests",
        "dasm2.data.registry_schema",
    ):
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source


def test_the_build_inputs_are_three_explicit_paths():
    """`NeutralBuildInputs` is the only way in, and it carries paths."""
    from dasm2.neutral.scov2_rates import NeutralBuildInputs

    fields = NeutralBuildInputs.__dataclass_fields__
    assert set(fields) == {"gene", "master_table", "site_annotations", "founder_fasta"}


def _rates_frame(values):
    """A minimal rate table: the key columns `RATE_KEY` uses, plus `rate`."""
    import pandas as pd

    return pd.DataFrame(
        {
            "mut_type": ["AC", "AC", "CT"],
            "motif": ["AAA", "AAC", "ACA"],
            "unpaired": [0, 1, 0],
            "rate": list(values),
        }
    )


def test_the_rates_digest_ignores_last_bit_noise_but_not_the_model():
    """The archive pin must answer "did the rates change", not "which NumPy".

    No rate is recomputed between the two builds -- `select_gene_rates` masks
    rows out of the GLM master table and sorts them -- but the archives the pin
    was taken from rendered their rates at about 16 significant digits, where
    this environment writes the shortest round-trip repr. Parsed back, the two
    differ by up to 4.43e-14 relative with the model unchanged, and a digest
    over the member's bytes called that a changed model. So the two
    perturbations below have to land on opposite sides of the check, and a test
    that only asserted the tolerant half would pass just as well against a
    digest of a constant.
    """
    from verify_era_neutral_models import canonical_rates_digest

    base = [0.1336109211125319, 2.2867444902552037, 0.0021947363675641]
    pinned = canonical_rates_digest(_rates_frame(base))

    noise = [v * (1 + 4.43e-14) for v in base]
    assert noise != base
    assert canonical_rates_digest(_rates_frame(noise)) == pinned

    # 1e-6 relative is still far smaller than any rate-selection change, which
    # takes a whole different row of the GLM table and moves rates by percent.
    real = [base[0] * (1 + 1e-6), *base[1:]]
    assert canonical_rates_digest(_rates_frame(real)) != pinned


def test_the_rates_digest_is_blind_to_row_order():
    """Row order belongs to whoever wrote the archive, not to the model."""
    from verify_era_neutral_models import canonical_rates_digest

    frame = _rates_frame([0.5, 1.5, 2.5])
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    assert canonical_rates_digest(shuffled) == canonical_rates_digest(frame)
