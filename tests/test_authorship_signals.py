import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from mbt_ai_tools.authorship import (
    FEATURE_NAMES,
    AuthorshipCorpusError,
    evaluate_authorship_corpus,
    text_profile,
)


def write_fixture_corpus(path: Path, *, provenance_status: str = "fixture") -> None:
    records = [
        {
            "id": "human_train_one",
            "text": "I missed the bus, walked home, and found the keys in my coat.",
            "label": "human",
            "split": "train",
            "provenance_status": provenance_status,
            "source_notes": "Test record.",
        },
        {
            "id": "human_train_two",
            "text": "The kettle clicked off early, so I boiled it again and made tea.",
            "label": "human",
            "split": "train",
            "provenance_status": provenance_status,
            "source_notes": "Test record.",
        },
        {
            "id": "ai_train_one",
            "text": "Clear planning improves delivery, strengthens coordination, and supports efficient review.",
            "label": "ai_raw",
            "split": "train",
            "provenance_status": provenance_status,
            "source_notes": "Test record.",
        },
        {
            "id": "ai_train_two",
            "text": "Consistent communication helps teams identify priorities and respond to change.",
            "label": "ai_raw",
            "split": "train",
            "provenance_status": provenance_status,
            "source_notes": "Test record.",
        },
        {
            "id": "human_eval",
            "text": "The parcel arrived late, but the neighbour had kept it dry for me.",
            "label": "human",
            "split": "eval",
            "provenance_status": provenance_status,
            "source_notes": "Test record.",
        },
        {
            "id": "ai_eval",
            "text": "Regular review makes decisions easier to explain and simpler to revisit.",
            "label": "ai_edited",
            "split": "eval",
            "provenance_status": provenance_status,
            "source_notes": "Test record.",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_authorship_signals.py"
    spec = importlib.util.spec_from_file_location("evaluate_authorship_signals", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_text_profile_is_deterministic_and_finite():
    first = text_profile("Short sentence. A second sentence has different length!")
    second = text_profile("Short sentence. A second sentence has different length!")

    assert tuple(first) == FEATURE_NAMES
    assert first == second
    assert all(np.isfinite(value) for value in first.values())


def test_fixture_corpus_reports_research_only_boundary(tmp_path: Path):
    corpus = tmp_path / "fixture.jsonl"
    write_fixture_corpus(corpus)

    report = evaluate_authorship_corpus(corpus)

    assert report["mode"] == "research_only"
    assert report["evidence_status"] == "fixture_or_mixed"
    assert report["summary"]["evaluation_records"] == 2
    assert report["summary"]["non_abstained_agreement"] is None
    assert {
        case["prediction"]["classification"] for case in report["cases"]
    } <= {"human_like", "ai_assistance_like", "uncertain"}


def test_require_verified_rejects_fixture_records(tmp_path: Path):
    corpus = tmp_path / "fixture.jsonl"
    write_fixture_corpus(corpus)

    with pytest.raises(AuthorshipCorpusError, match="verified corpus"):
        evaluate_authorship_corpus(corpus, require_verified=True)


def test_verified_research_corpus_can_report_non_abstained_agreement(tmp_path: Path):
    corpus = tmp_path / "verified.jsonl"
    write_fixture_corpus(corpus, provenance_status="verified")

    report = evaluate_authorship_corpus(corpus)

    assert report["evidence_status"] == "verified_research_corpus"
    assert "non_abstained_agreement" in report["summary"]


def test_script_writes_json_report(tmp_path: Path):
    corpus = tmp_path / "fixture.jsonl"
    output = tmp_path / "report.json"
    write_fixture_corpus(corpus)
    script = load_script_module()

    assert script.main(["--corpus", str(corpus), "--format", "json", "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["evidence_status"] == "fixture_or_mixed"
    assert "does not determine" in payload["claim_boundary"]


def test_committed_fixture_remains_non_evidential():
    corpus = Path(__file__).resolve().parents[1] / "examples" / "ai_authorship_fixture_corpus.jsonl"

    report = evaluate_authorship_corpus(corpus)

    assert report["evidence_status"] == "fixture_or_mixed"
    assert report["summary"]["total_records"] == 16
