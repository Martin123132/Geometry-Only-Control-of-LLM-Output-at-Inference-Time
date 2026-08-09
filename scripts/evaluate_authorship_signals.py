"""Run the research-only ManifoldGuard authorship-signal evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mbt_ai_tools.authorship import (  # noqa: E402
    AuthorshipCorpusError,
    evaluate_authorship_corpus,
    format_authorship_report,
)

DEFAULT_CORPUS = PROJECT_ROOT / "examples" / "ai_authorship_fixture_corpus.jsonl"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ManifoldGuard's research-only structural authorship-signal "
            "evaluator. This tool does not prove authorship."
        )
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="JSONL corpus with explicit source labels and provenance status.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output file. Keep generated evidence outside the repository.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Console and optional output format.",
    )
    parser.add_argument(
        "--min-separation",
        type=float,
        default=0.15,
        help="Absolute geometry signal below which the evaluator abstains.",
    )
    parser.add_argument(
        "--require-verified",
        action="store_true",
        help="Reject fixture or mixed-provenance corpora.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = evaluate_authorship_corpus(
            args.corpus,
            min_separation=args.min_separation,
            require_verified=args.require_verified,
        )
    except (AuthorshipCorpusError, OSError, ValueError) as exc:
        print(f"Authorship-signal evaluation failed: {exc}", file=sys.stderr)
        return 2

    rendered = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else format_authorship_report(report)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote report: {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
