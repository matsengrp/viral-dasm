"""Collect the vendored comments that need a human decision before release.

`dasm2`'s comments are unusually detailed, and most of that detail is about
work that is not this paper. Carrying them into a public repository leaks two
different things:

- **Dead links.** References to private `dasm2-experiments` issue numbers, of
  the form `#1234`. A reader cannot follow them. This is the small problem.
- **Other projects' results.** Measurements, benchmark numbers and design
  arguments from antibody, BCR, ProteinGym, Tsuboyama, Compara, QDASM and
  ESM2 work. The comment body is the part that leaks, not the link.

Generic explanation stays -- it is the reason the vendored code is readable at
all, and deleting it to be safe would make this repository worse than the one
it came from. So this script decides nothing. It collects, with enough context
to judge each one, into a Markdown file for Hugh to go through.

Flagging happens while porting, with the file already open; the decision
happens once, before the repository goes public.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

#: A private `dasm2-experiments` issue reference. Bounded to 3-4 digits so it
#: does not match `#1` in prose, a colour literal, or a fragment in a URL.
ISSUE_REF = re.compile(r"(?<![\w/])#(\d{3,4})(?!\d)")

#: Topics that are not this paper. A comment naming one is discussing work a
#: reader of viral-dasm has no access to and no reason to care about.
OTHER_PROJECT_TERMS = (
    "antibody",
    "antibodies",
    "BCR",
    "ProteinGym",
    "Tsuboyama",
    "megascale",
    "Compara",
    "superfamily",
    "QDASM",
    "ESM2",
    "thrifty",
    "Olmsted",
    "paired chain",
    "paired-chain",
    "ANARCI",
    "IMGT",
    "germline",
    "naive B",
    "SHM",
    "S5F",
    "shmoof",
    "MPNN",
    "ThermoMPNN",
    "contrastive",
    "binding affinity",
    "delta affinity",
    "3Di",
    "foldseek",
    "influenza",
)

#: Language that marks a comment as reporting a result rather than explaining
#: a mechanism. A measured number is the thing that cannot be re-derived from
#: the code in front of you.
RESULT_TERMS = (
    "measured",
    "we found",
    "turned out",
    "in practice this",
    "benchmark",
    "x faster",
    "x slower",
    "% faster",
    "% slower",
    "regressed",
    "improved by",
    "nats",
    "AUC",
    "Spearman",
    "Pearson",
    "wall clock",
    "GPU-hour",
)

#: How many lines of context each flag carries.
CONTEXT_LINES = 2


@dataclass(frozen=True)
class Flag:
    path: str
    line: int
    reason: str
    text: str
    context: str


def comment_blocks(lines: list[str]) -> list[tuple[int, str]]:
    """Every comment line and docstring line, as (1-indexed line no, text).

    Crude on purpose: a `#` inside a string literal is included. A false
    positive costs a human one glance; a false negative ships a leak.
    """
    out = []
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#") or '"""' in line or "'''" in line or "#" in line:
            out.append((i, line.rstrip()))
    return out


def reasons_for(text: str) -> list[str]:
    """Why this line needs a decision, if it does."""
    reasons = []
    issues = ISSUE_REF.findall(text)
    if issues:
        reasons.append("private issue ref " + ", ".join(f"#{n}" for n in issues))
    lowered = text.lower()
    hits = [t for t in OTHER_PROJECT_TERMS if t.lower() in lowered]
    if hits:
        reasons.append("other project: " + ", ".join(sorted(set(hits))))
    results = [t for t in RESULT_TERMS if t.lower() in lowered]
    if results:
        reasons.append("reports a result: " + ", ".join(sorted(set(results))))
    return reasons


def scan(root: Path) -> list[Flag]:
    flags = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        lines = path.read_text().splitlines()
        for line_no, text in comment_blocks(lines):
            reasons = reasons_for(text)
            if not reasons:
                continue
            lo = max(0, line_no - 1 - CONTEXT_LINES)
            hi = min(len(lines), line_no + CONTEXT_LINES)
            flags.append(
                Flag(
                    path=str(path.relative_to(root.parent)),
                    line=line_no,
                    reason="; ".join(reasons),
                    text=text.strip(),
                    context="\n".join(lines[lo:hi]),
                )
            )
    return flags


def render(flags: list[Flag], root: Path) -> str:
    by_file: dict[str, list[Flag]] = {}
    for flag in flags:
        by_file.setdefault(flag.path, []).append(flag)

    issue_numbers = Counter()
    for flag in flags:
        issue_numbers.update(ISSUE_REF.findall(flag.text))

    n_py = sum(1 for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    out = [
        "# Comments to decide on before this repository goes public",
        "",
        (
            "Generated by `scripts/collect_comment_flags.py`. **Nothing here has "
            "been changed** -- this is a review list, not a diff."
        ),
        "",
        (
            "`dasm2`'s comments are unusually detailed, and a lot of that detail "
            "is about work that is not this paper. Three rules for working "
            "through the list:"
        ),
        "",
        (
            "- **Generic explanation stays.** An explanation of why the code does "
            "what it does is the reason this tree is readable. Deleting it to be "
            "safe would make this repository worse than the one it came from."
        ),
        (
            "- **A private issue reference goes or gets inlined.** `#1234` is a "
            "dead link for every reader of this repository. If the comment is "
            "only a pointer, cut it; if the issue holds the reason, write the "
            "reason into the comment."
        ),
        (
            "- **Another project's result goes.** A measurement, a benchmark or a "
            "design argument from antibody, BCR, ProteinGym, Tsuboyama, Compara, "
            "QDASM or ESM2 work is the part that actually leaks. The dead link is "
            "the small problem."
        ),
        "",
        "## Totals",
        "",
        f"- {len(flags)} flagged lines across {len(by_file)} of {n_py} files",
        (
            f"- {sum(issue_numbers.values())} private issue references, "
            f"{len(issue_numbers)} distinct issues"
        ),
        "",
    ]
    if issue_numbers:
        top = ", ".join(f"#{n} ({c})" for n, c in issue_numbers.most_common(10))
        out += [f"Most-cited issues: {top}", ""]

    out += ["## By file", "", "| file | flagged lines |", "|---|---:|"]
    for path, items in sorted(by_file.items(), key=lambda kv: -len(kv[1])):
        out.append(f"| `{path}` | {len(items)} |")
    out.append("")

    out += ["## The flags", ""]
    for path, items in sorted(by_file.items(), key=lambda kv: -len(kv[1])):
        out += [f"### `{path}`", ""]
        for flag in items:
            out += [
                f"**line {flag.line}** -- {flag.reason}",
                "",
                "```python",
                flag.context,
                "```",
                "",
            ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dasm2"))
    parser.add_argument("--output", type=Path, default=Path("dasm2/COMMENT-FLAGS.md"))
    args = parser.parse_args()

    flags = scan(args.root)
    args.output.write_text(render(flags, args.root))
    print(f"{len(flags)} flags written to {args.output}")


if __name__ == "__main__":
    main()
