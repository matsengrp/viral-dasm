"""The stopping rule every run inherits, and the configs that inherit it.

EPIC #1 settles M3 on one rule for all seven runs: a model trains until
``ReduceLROnPlateau`` has driven the learning rate below ``min_learning_rate``,
not to a step budget. Three values decide where that lands --
``min_learning_rate`` 1e-6, plateau ``factor`` 0.5, plateau ``eval_patience``
10 -- and none is *set* as a key in any config file in this repository. They
are defaults in the vendored ``dasm2``.

**The floor is not the only gate, and it is not the first one.**
``Trainer._should_stop_early`` returns on ``evals_without_improvement >=
early_stopping_patience`` before it looks at the learning rate, and its own
docstring says so. The templates set patience 30 against an ``eval_patience``
of 10, so a run that stops improving outright stops after about three
reductions rather than ten. The floor is the rule only while improvements
keep arriving at least once per 30 evaluations, which is what the reference
run did. ``max_train_steps`` bounds each cycle too, and torch's own
``ReduceLROnPlateau`` defaults -- ``threshold`` 1e-4 relative, ``cooldown`` 0
-- decide when each halving fires, since the trainer passes only ``mode``,
``factor`` and ``patience``. Those are not pinned here.

A change to any of them raises no error. Runs stop somewhere else, and the
only evidence is the final learning rate in a finished log. At roughly five
days wall per full-depth arm, that is the most expensive silent failure M3
can have.

``min_learning_rate`` is the one not reachable from a config object at all,
and it fails in two independent ways:

- ``dasm2/train_command.py`` writes the literal ``1e-6`` into every run's
  ``logger_metadata``, not read from the trainer. Lower the trainer's default
  and each run still *records* 1e-6 while stopping elsewhere -- the wrong
  number and the run's own provenance agreeing with each other.
- No ``Trainer(...)`` call sets it, which is what makes "every run takes the
  default" true. That is a fact about call sites rather than about defaults,
  so asserting the default alone would leave this file's own premise
  uncovered: a future ``dasm2`` passing ``min_learning_rate=1e-7`` at any
  construction would keep both values at 1e-6 and move every run.

**Every helper takes source text**, never a path it reads itself. That is
what lets the mutation tests below be committed tests rather than a manual
edit-and-revert of a vendored file -- which would also touch an mtime that a
revert does not restore.

It is also why ``min_learning_rate`` is read from ``trainer.py``'s source
rather than from ``inspect.signature(Trainer.__init__)``. Torch is already paid --
``dasm2.config`` pulls it in below, and ``tests/test_fork.py`` imports it
too -- but the trainer's own module tree costs tens of seconds warm and minutes
cold on this filesystem (41 s and 263 s marginal, measured, though the figure
moves with cache and contention) -- inside ``validate_tests``, which gates
every corpus build. The source is the record; what makes reading it sound is asserted
rather than assumed, in ``test_trainer_init_is_a_plain_function`` and
``test_only_two_modules_mention_the_floor``, and the gap those leave is
stated there.

The plateau pair is asserted on the *loaded config* rather than on
``ReduceOnPlateauSchedulerConfig()``'s class defaults, because a template that
grew its own ``scheduler:`` block would satisfy the class check and still
train under different numbers. The template is what trains.

**What is pinned is written values and one wiring constant, not the stopping
logic.** ``trainer.py`` could keep every number and change
``self.min_learning_rate = min_learning_rate`` to scale it, or invert the
comparison, and nothing here would fail -- both names stay inside
``FLOOR_MODULES``, which checks which files mention the parameter, not what
they do with it. Pinning the behaviour would mean constructing a Trainer,
which is the import this file exists to avoid.

Adapted from ``scripts/check_rendered_config.py`` at ``4cc5ddb^``,
which made the same three assertions as a gate script before the port
scaffolding was removed.
"""

from __future__ import annotations

import ast
import sys
from math import inf
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import render_config
from lib import NUM_CYCLES, assert_dasm2_is_this_repo

from dasm2.config import TrainingConfig

assert_dasm2_is_this_repo()

#: The floor every run in this repository stops on. EPIC #1 makes this a
#: deliberate choice for all seven runs rather than an inherited accident: it
#: is the one place `dasm-1077k` departs from the upstream arm it follows,
#: which lowers it to 1e-7.
EXPECTED_MIN_LEARNING_RATE = 1e-6

#: The plateau pair. At factor 0.5, lr 1e-3 needs ten halvings to cross the
#: floor and 5e-4 needs nine, which is the whole derivation of the reference
#: run's 436,000-step stop and of the EPIC's prediction that the 5e-4 arms
#: stop one halving's worth of steps earlier.
EXPECTED_PLATEAU_FACTOR = 0.5
EXPECTED_PLATEAU_EVAL_PATIENCE = 10

#: Where the value lives, for the two reads that cannot go through an import.
TRAINER_SOURCE = REPO_ROOT / "dasm2" / "training" / "trainer.py"
TRAIN_COMMAND_SOURCE = REPO_ROOT / "dasm2" / "train_command.py"

#: ``max_train_steps`` for the render. Any value below the templates'
#: ``eval_every_steps`` is rejected by ``TrainingConfig`` -- "a cycle must
#: contain at least one evaluation" -- so this is a real budget, not a token.
RENDER_STEPS = 500_000

TEMPLATES = sorted((REPO_ROOT / "configs").glob("*.template.yaml"))
assert TEMPLATES, (
    "no configs/*.template.yaml; the parametrized test below would collect nothing and pass"
)


def _trainer_class(source: str) -> ast.ClassDef:
    """The ``Trainer`` class definition node."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == "Trainer":
            return node
    raise AssertionError("no `class Trainer` in the given source")


def _trainer_init(source: str) -> ast.FunctionDef:
    """``Trainer.__init__``'s definition node.

    Returns the last one defined, which is the one Python keeps.
    """
    inits = [
        item
        for item in _trainer_class(source).body
        if isinstance(item, ast.FunctionDef) and item.name == "__init__"
    ]
    assert inits, "no Trainer.__init__ in the given source"
    return inits[-1]


def trainer_default(source: str) -> float:
    """``Trainer.__init__``'s ``min_learning_rate`` default, read from text."""
    args = _trainer_init(source).args
    # Defaults align to the tail of posonly+args; kwonly carry their own.
    positional = args.posonlyargs + args.args
    paired = list(
        zip(positional[len(positional) - len(args.defaults) :], args.defaults)
    )
    paired += [
        (a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None
    ]
    for arg, default in paired:
        if arg.arg == "min_learning_rate":
            return ast.literal_eval(default)
    raise AssertionError(
        "no `min_learning_rate` default found on Trainer.__init__; the "
        "signature moved and this file no longer reads what it claims to."
    )


def min_lr_position(source: str) -> float:
    """How many positional arguments a ``Trainer(...)`` call needs to reach it.

    Derived rather than pinned: the parameter's index among
    ``posonly + args`` is exactly that count, because dropping ``self``
    offsets the zero-based index by one. A keyword-only parameter cannot be
    reached positionally at all, which is ``inf``.
    """
    args = _trainer_init(source).args
    names = [a.arg for a in args.posonlyargs + args.args]
    return names.index("min_learning_rate") if "min_learning_rate" in names else inf


#: Eight for this tree: model, neutral_model, tasks, backend, batch_size,
#: val_batch_size, optimizer, then this.
MIN_LR_POSITION = min_lr_position(TRAINER_SOURCE.read_text())


def metadata_literal(source: str) -> float:
    """The ``min_learning_rate`` written into a run's recorded metadata.

    Only reachable with ``ast``: the dict is built inside a long function
    body, so it cannot be imported, and a regex on the spelling would break
    under ``ruff format`` or a value written ``0.000001``.
    """
    found = [
        ast.literal_eval(value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "min_learning_rate"
    ]
    assert len(found) == 1, (
        f"expected exactly one `min_learning_rate` metadata entry, found "
        f"{len(found)}: {found}. Two of them means a run's recorded floor "
        f"depends on which branch wrote it, and this check would be reading "
        f"whichever came first."
    )
    return found[0]


def trainer_calls(source: str) -> list[tuple[int, bool]]:
    """Every ``Trainer(...)`` call, and whether it could set the floor.

    "Could" is wider than a ``min_learning_rate=`` keyword, because the
    keyword is only the obvious way. A call that passes eight or more
    positional arguments reaches the parameter without naming it, and a
    ``**kwargs`` expansion can carry it invisibly.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "Trainer")
                or (
                    isinstance(node.func, ast.Attribute) and node.func.attr == "Trainer"
                )
            )
        ):
            continue
        reaches = (
            any(k.arg == "min_learning_rate" for k in node.keywords)
            or len(node.args) >= MIN_LR_POSITION
            or any(k.arg is None for k in node.keywords)
        )
        found.append((node.lineno, reaches))
    return found


def test_trainer_default_is_the_floor():
    """The value every run takes, read from the source of record."""
    default = trainer_default(TRAINER_SOURCE.read_text())
    assert default == EXPECTED_MIN_LEARNING_RATE, (
        f"Trainer.min_learning_rate defaults to {default!r}, not "
        f"{EXPECTED_MIN_LEARNING_RATE!r}. No config file sets this, so every "
        f"run in this repository takes the default and would now stop at a "
        f"different learning rate with nothing in its loss curve to say why."
    )


def test_trainer_init_is_a_plain_function():
    """Licenses reading the default from source instead of from the import.

    Reading source is only as good as the claim that what is written is what
    a call sees. Four ways that could stop being true are checked here: a
    decorator returning a wrapper, a metaclass or base class rewriting the
    signature, and a second ``__init__`` shadowing the one read.

    What remains uncovered is code that rewrites ``__defaults__`` generically,
    without naming the parameter -- which exists nowhere in this tree and
    which ``test_only_two_modules_mention_the_floor`` would catch in every
    form that does name it. That is a stated gap, not a proof. If any
    assertion here starts failing, read the default through
    ``inspect.signature(Trainer.__init__)`` instead and pay the import: tens
    of seconds warm, minutes cold, inside the job that gates every corpus
    build, which is why it is not paid by default.
    """
    source = TRAINER_SOURCE.read_text()
    klass = _trainer_class(source)
    inits = [
        item
        for item in klass.body
        if isinstance(item, ast.FunctionDef) and item.name == "__init__"
    ]
    assert not klass.bases and not klass.keywords, (
        f"Trainer now has bases/keywords "
        f"{[ast.unparse(b) for b in klass.bases + [k.value for k in klass.keywords]]}, "
        f"so a base class or metaclass could rewrite its signature."
    )
    assert len(inits) == 1, (
        f"Trainer defines {len(inits)} __init__ methods; the written defaults "
        f"are no longer unambiguous."
    )
    assert not inits[0].decorator_list, (
        f"Trainer.__init__ is now decorated with "
        f"{[ast.unparse(d) for d in inits[0].decorator_list]}, so its written "
        f"defaults may not be the ones a call sees."
    )


def test_recorded_metadata_matches_the_trainer_default():
    """A run's provenance must not be able to disagree with its behaviour."""
    recorded = metadata_literal(TRAIN_COMMAND_SOURCE.read_text())
    assert recorded == EXPECTED_MIN_LEARNING_RATE, (
        f"train_command.py records min_learning_rate as {recorded!r} in every "
        f"run's metadata, but the floor is {EXPECTED_MIN_LEARNING_RATE!r}. "
        f"This literal is not read from the trainer, so the two can drift and "
        f"the run's own record would agree with the wrong number."
    )


#: Every `Trainer(...)` construction in the vendored package. Pinned so the
#: scan below cannot pass against nothing: if the matcher broke outright it
#: would find zero calls and still report no override. It does *not* catch a
#: call site added under an alias -- `T(...)` after `import Trainer as T`
#: leaves the count at 3 -- which is what the module-mention test below is
#: for. An unrelated re-vendor adding a fourth construction fails this; in a
#: frozen, pruned tree that is a change worth a human look.
KNOWN_TRAINER_CALLS = 3

#: The only two modules in `dasm2` that mention the floor at all. Pinned for
#: the same reason, and because it closes the paths the call scan cannot see:
#: a post-construction `trainer.min_learning_rate = x`, a subclass, or a
#: factory growing its own parameter would all have to name it somewhere.
FLOOR_MODULES = {"dasm2/training/trainer.py", "dasm2/train_command.py"}


def test_no_call_site_can_set_the_floor():
    """The premise: every run takes the default because no call sets it."""
    calls = {
        str(path.relative_to(REPO_ROOT)): found
        for path in sorted((REPO_ROOT / "dasm2").rglob("*.py"))
        if (found := trainer_calls(path.read_text()))
    }
    total = sum(len(v) for v in calls.values())
    assert total == KNOWN_TRAINER_CALLS, (
        f"found {total} Trainer(...) constructions in dasm2, expected "
        f"{KNOWN_TRAINER_CALLS}: {calls}. If this dropped to zero the scan is "
        f"matching nothing -- an aliased import or a rename -- and the "
        f"assertion below would pass while checking nothing."
    )
    offenders = {f: [ln for ln, reaches in v if reaches] for f, v in calls.items()}
    offenders = {f: v for f, v in offenders.items() if v}
    assert not offenders, (
        f"a Trainer(...) call could set min_learning_rate at {offenders} -- by "
        f"keyword, by {MIN_LR_POSITION} or more positional arguments, or "
        f"through a ** expansion. Every assertion in this file reads a "
        f"default, so a call-site value would move every run while leaving "
        f"both the written default and the recorded metadata at "
        f"{EXPECTED_MIN_LEARNING_RATE!r}. Assert the call site's value "
        f"instead."
    )


def test_one_cycle_per_run():
    """The fourth input to the stopping rule, and the one most easily missed.

    ``dasm2.train_command.train_model`` defaults ``cycles`` to 4. Above one
    cycle, crossing the floor ends a *cycle* rather than the run:
    ``Trainer._reset_optimizer_with_adjusted_lr`` rebuilds the optimizer at a
    bumped learning rate and rebuilds the scheduler, so the ten-halvings
    derivation runs once per cycle from a different starting rate each time
    and "every run stops on the floor" stops being true.

    This repository overrides that to 1 in ``scripts/lib.py``, which
    ``scripts/train.py`` passes through as ``--cycles``. Asserted because the
    override is a bare constant nothing else checks, and because it is the one
    input to the stopping rule that is this repository's own choice rather
    than the library's.
    """
    assert NUM_CYCLES == 1, (
        f"NUM_CYCLES is {NUM_CYCLES}, not 1. Above one cycle the learning-rate "
        f"floor is a cycle boundary rather than a stop -- the optimizer is "
        f"rebuilt at a bumped rate and the schedule restarts -- so EPIC #1's "
        f"'every run stops on the floor' no longer holds and steps-to-stop is "
        f"no longer comparable across the seven runs."
    )


def test_only_two_modules_mention_the_floor():
    """Closes the paths the call scan cannot see.

    Setting the attribute after construction, subclassing ``Trainer``, or a
    factory growing its own ``min_learning_rate`` parameter would all be
    invisible above -- but each has to name the thing somewhere.
    """
    mentions = {
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / "dasm2").rglob("*.py"))
        if "min_learning_rate" in path.read_text()
    }
    assert mentions == FLOOR_MODULES, (
        f"dasm2 modules naming min_learning_rate are {sorted(mentions)}, not "
        f"{sorted(FLOOR_MODULES)}. A new one means the floor is being read, "
        f"set or passed somewhere this file does not check."
    )


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_template_renders_and_carries_the_plateau(template: Path):
    """Render, load, and pin the two values the loaded config does carry.

    The render-and-load is the smoke: it is what this does on its way to the
    assertions, so a template that renders but will not validate fails here
    rather than at the start of a multi-day run.
    """
    rendered = render_config.render(
        template.read_text(), render_config.substitutions(RENDER_STEPS)
    )
    config = TrainingConfig(**yaml.safe_load(rendered))
    assert (
        config.scheduler.factor,
        config.scheduler.eval_patience,
    ) == (EXPECTED_PLATEAU_FACTOR, EXPECTED_PLATEAU_EVAL_PATIENCE), (
        f"{template.name} loads with plateau factor "
        f"{config.scheduler.factor!r} and eval_patience "
        f"{config.scheduler.eval_patience!r}, not "
        f"{EXPECTED_PLATEAU_FACTOR!r} / {EXPECTED_PLATEAU_EVAL_PATIENCE!r}. "
        f"Those two decide how many halvings reach the floor, and so where "
        f"this run stops."
    )


def test_a_configured_neutral_model_is_refused():
    """A neutral model in a training config would be loaded and never used.

    These corpora carry ``nt_rates`` and ``nt_csps`` precomputed, so a
    configured neutral model is a setting that looks like it has an effect
    and has none. ``configs/dasm-85k.template.yaml`` says so in a comment,
    and ``TrainingConfig.neutral_model`` is annotated ``None``, so the
    vendored config refuses one outright.

    Pinned because that refusal is load-bearing and invisible: a re-vendor
    widening the annotation would start accepting the block silently, and M3
    writes three more templates against this contract. The deleted
    ``check_rendered_config.py`` refused the key in the raw YAML; asserting
    the vendored refusal is the stronger form of the same check.
    """
    rendered = render_config.render(
        TEMPLATES[0].read_text() + "\nneutral_model:\n    type: TabularRates\n",
        render_config.substitutions(RENDER_STEPS),
    )
    with pytest.raises(ValidationError, match="neutral_model"):
        TrainingConfig(**yaml.safe_load(rendered))


def test_a_template_overriding_the_plateau_is_caught():
    """The plateau assertion bites: it reads the config, not class defaults."""
    template = (
        TEMPLATES[0].read_text()
        + "\nscheduler:\n"
        + "    scheduler_type: reduce_on_plateau\n"
        + "    factor: 0.1\n"
    )
    config = TrainingConfig(
        **yaml.safe_load(
            render_config.render(template, render_config.substitutions(RENDER_STEPS))
        )
    )
    assert config.scheduler.factor == 0.1, (
        "a template-level `scheduler:` block did not reach the loaded "
        "config, so the assertion above would pass on class defaults while "
        "the run trained under the template's numbers."
    )


def test_trainer_default_read_bites():
    """Mutation: the text helper reports a changed default."""
    mutated = TRAINER_SOURCE.read_text().replace(
        "min_learning_rate: float = 1e-6", "min_learning_rate: float = 1e-7", 1
    )
    assert trainer_default(mutated) == 1e-7


def test_metadata_literal_read_bites():
    """Mutation: the metadata helper reports a changed literal."""
    mutated = TRAIN_COMMAND_SOURCE.read_text().replace(
        '"min_learning_rate": 1e-6', '"min_learning_rate": 1e-7', 1
    )
    assert metadata_literal(mutated) == 1e-7


def test_call_site_scan_bites():
    """Mutation: the call-site scan reports an explicit override."""
    assert trainer_calls("Trainer(model, tasks, min_learning_rate=1e-7)") == [(1, True)]
    assert trainer_calls("Trainer(a, b, c, d, e, f, g, h)") == [(1, True)]
    assert trainer_calls("Trainer(**kw)") == [(1, True)]
    assert trainer_calls("Trainer(model, tasks)") == [(1, False)]
