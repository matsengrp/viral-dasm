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

from lib import CANARY_STEPS, CONFIG, mat_pcp_path

PLACEHOLDER = re.compile(r"\{([A-Z0-9_]+)\}")


def substitutions(max_train_steps: int) -> dict[str, str]:
    """Every value a training template may ask for."""
    return {
        "MAX_TRAIN_STEPS": str(max_train_steps),
        "CENSORED_MUTATIONS_FILE": str(mat_pcp_path("censored_mutations_file")),
        "CENSORED_CLADES_FILE": str(mat_pcp_path("censored_clades_file")),
        "MASKED_NT_SITES_FILE": str(mat_pcp_path("masked_nt_sites_file")),
        "REFERENCE_SEQUENCE_FILE": str(mat_pcp_path("reference_sequence_file")),
        "CENSOR_MODE": str(CONFIG["mat_pcp"]["censor_mode"]),
    }


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
        type=int,
        default=CANARY_STEPS,
        help="the step budget this run gets; the canary's by default",
    )
    args = parser.parse_args()

    values = substitutions(args.max_train_steps)
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
