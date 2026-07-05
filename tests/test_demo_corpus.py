from pathlib import Path

from mbt_ai_tools.cli import build_batch_reports


def test_demo_corpus_covers_public_walkthrough_cases():
    project_root = Path(__file__).resolve().parent.parent
    demo_corpus = project_root / "examples" / "demo_corpus.jsonl"

    reports = list(
        build_batch_reports(
            demo_corpus,
            use_embeddings=False,
            include_explanations=True,
        )
    )

    assert [report["id"] for report in reports] == [
        "safe_emit_reference_match",
        "blocked_negation",
        "blocked_numeric_drift",
    ]
    assert [report["action"] for report in reports] == ["emit", "block", "block"]
    assert reports[0]["emitted_text"] == "The capital of France is Paris."
    assert reports[1]["evaluations"][0]["safe_to_emit"] is False
    assert reports[2]["evaluations"][0]["safe_to_emit"] is False
