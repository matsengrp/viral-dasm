"""Fill a training-config template from ``config.yaml``.

The five censoring and reference-sequence paths are ``TrainingConfig`` fields,
read out of the training YAML by ``dasm2`` itself. viral-dasm's ``config.yaml``
therefore cannot replace them -- it renders them in, which is what keeps one
file the complete list of paths the pipeline reads.

Substitution is a literal ``{KEY}`` replace rather than ``str.format`` or a
template engine, because the templates are YAML full of braces in flow
sequences (``betas: [0.9, 0.999]``) and comments, and a formatter would either
choke on them or quietly rewrite them.

Every placeholder in the template must be supplied and every supplied value
must be used; both directions raise. A template that grew a placeholder nobody
fills would otherwise render a config with a literal ``{...}`` in a path, and
training would fail several minutes in with a confusing error.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import CONFIG, mat_pcp_path

PLACEHOLDER = re.compile(r"\{([A-Z0-9_]+)\}")


def substitutions(
    max_train_steps: int,
    eval_every_steps: int | None = None,
    max_aa_seq_len: int | None = None,
) -> dict[str, str]:
    """Every value a training template may ask for.

    The two optional values are included **only when supplied**, because
    :func:`render` raises on a value the template does not use. Every template
    carries ``{EVAL_EVERY_STEPS}``; only ``configs/ci-484k.template.yaml``
    carries ``{MAX_AA_SEQ_LEN}``, since ``PerSiteParentChildHead`` is the one
    head that sizes a parameter from it. Passing it to a transformer template
    is the mistake that silently truncates every sequence in other trees, and
    here it is a ``KeyError`` instead.
    """
    values = {
        "MAX_TRAIN_STEPS": str(max_train_steps),
        "CENSORED_MUTATIONS_FILE": str(mat_pcp_path("censored_mutations_file")),
        "CENSORED_CLADES_FILE": str(mat_pcp_path("censored_clades_file")),
        "MASKED_NT_SITES_FILE": str(mat_pcp_path("masked_nt_sites_file")),
        "REFERENCE_SEQUENCE_FILE": str(mat_pcp_path("reference_sequence_file")),
        "CENSOR_MODE": str(CONFIG["mat_pcp"]["censor_mode"]),
    }
    if eval_every_steps is not None:
        values["EVAL_EVERY_STEPS"] = str(eval_every_steps)
    if max_aa_seq_len is not None:
        values["MAX_AA_SEQ_LEN"] = str(max_aa_seq_len)
    return values


def render(template: str, values: dict[str, str]) -> str:
    """``template`` with every ``{KEY}`` replaced.

    Raises:
        KeyError: If the template names a placeholder ``values`` has no entry
            for, or if ``values`` carries one the template never uses.
    """
    wanted = set(PLACEHOLDER.findall(template))
    missing = wanted - values.keys()
    if missing:
        raise KeyError(
            f"the template asks for {sorted(missing)}, which config.yaml does "
            f"not supply. A rendered config would carry the literal braces "
            f"into a path and fail minutes into training."
        )
    unused = values.keys() - wanted
    if unused:
        raise KeyError(
            f"{sorted(unused)} would be rendered into nothing. Either the "
            f"template dropped a field it still needs -- a censor path that "
            f"stops reaching the transform trains perfectly well and reports "
            f"a number for a condition it did not run -- or this list is stale."
        )
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-train-steps",
        required=True,
        type=int,
        help="the step budget this run gets",
    )
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=None,
        help=(
            "evaluations cadence, in optimizer steps. Every template asks for "
            "it; the training rules supply runs.EVAL_EVERY_STEPS and the "
            "smoke supplies a value small enough that a four-step cycle "
            "contains an evaluation."
        ),
    )
    parser.add_argument(
        "--max-aa-seq-len",
        type=int,
        default=None,
        help=(
            "the corpus's longest amino-acid sequence, read from its own "
            "Parquet metadata. Only ci-484k asks for it; supplying it to a "
            "template that does not raises rather than rendering an unused "
            "value."
        ),
    )
    args = parser.parse_args()

    values = substitutions(
        args.max_train_steps,
        eval_every_steps=args.eval_every_steps,
        max_aa_seq_len=args.max_aa_seq_len,
    )
    for key, value in values.items():
        if key.endswith("_FILE") and not Path(value).exists():
            raise SystemExit(
                f"{key} points at {value}, which does not exist. Censoring "
                f"reads it at training start, and a missing reversion "
                f"reference is the single most damaging thing to lose here -- "
                f"that source is roughly 40x the other two on Spike."
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(args.template.read_text(), values))
    print(f"rendered {args.template} -> {args.output}")


if __name__ == "__main__":
    main()
