import argparse

from .mbt.regulator import regulate_candidates
from .mbt.stability import classify_entropy, confidence_score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MBT-5 geometry-only confidence probe and v11 candidate regulator."
    )
    parser.add_argument(
        "text",
        nargs="?",
        help='Text or blank-line separated responses to score, e.g. `mbt-check "answer a\n\nanswer b"`',
    )
    parser.add_argument(
        "--reference",
        "-r",
        action="append",
        default=[],
        help="Reference statement. Repeat to build a reference manifold for candidate regulation.",
    )
    parser.add_argument(
        "--candidate",
        "-c",
        action="append",
        default=[],
        help="Candidate output to regulate. Repeat for candidate selection; emits best safe candidate or BLOCK.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Run only literal/relation/negation clamps without semantic embeddings.",
    )
    args = parser.parse_args()

    if args.reference or args.candidate:
        candidates = args.candidate or ([args.text] if args.text else [])
        if not args.reference:
            parser.error("--reference is required when using --candidate regulation mode")
        if not candidates:
            parser.error("provide at least one --candidate or positional text in regulation mode")

        result = regulate_candidates(
            candidates,
            args.reference,
            use_embeddings=not args.no_embeddings,
        )
        if result.action == "block":
            print("BLOCK | no safe candidate")
        else:
            assert result.emitted is not None
            print(f"EMIT | {result.emitted_text} | score={result.emitted.regulator_score:.4f}")
        for i, evaluation in enumerate(result.evaluations):
            label = "safe" if evaluation.safe_to_emit else "blocked"
            clamps = "|".join(evaluation.clamp_summary)
            print(f"[{i}] {label} | score={evaluation.regulator_score:.4f} | clamps={clamps}")
        return

    if args.text is None:
        parser.error("text is required unless --candidate is supplied")

    score = confidence_score(args.text)
    label, _ = classify_entropy(score)

    print(f"{label} | Internal Entropy: {score:.4f}")


if __name__ == "__main__":
    main()
