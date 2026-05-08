import json
from pathlib import Path

from mbt_ai_tools import regulate_candidates


CORPUS_PATH = Path(__file__).resolve().parents[1] / "examples" / "regression_corpus.jsonl"


def _load_cases():
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def test_offline_regression_corpus_matches_expected_actions():
    cases = list(_load_cases())

    assert len(cases) >= 100, "public regression corpus should keep at least 100 cases"

    for case in cases:
        result = regulate_candidates(
            case["candidates"],
            case["references"],
            use_embeddings=False,
        )

        assert result.action == case["expected_action"], case["id"]
        assert result.emitted_text == case["expected_emitted_text"], case["id"]

        if "expected_candidate_safe" in case:
            observed = [evaluation.safe_to_emit for evaluation in result.evaluations]
            assert observed == case["expected_candidate_safe"], case["id"]

        if "expected_clamp_contains" in case:
            for index, expected_clamps in case["expected_clamp_contains"].items():
                observed = set(result.evaluations[int(index)].clamp_summary)
                assert set(expected_clamps).issubset(observed), case["id"]
