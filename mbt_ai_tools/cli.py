import argparse

from .mbt.stability import classify_entropy, confidence_score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MBT-5 geometry-only confidence probe (inference-time)."
    )
    parser.add_argument(
        "text",
        help='Text or blank-line separated responses to score, e.g. `mbt-check "answer a\\n\\nanswer b"`',
    )
    args = parser.parse_args()

    score = confidence_score(args.text)
    label, _ = classify_entropy(score)

    print(f"{label} | Internal Entropy: {score:.4f}")


if __name__ == "__main__":
    main()
