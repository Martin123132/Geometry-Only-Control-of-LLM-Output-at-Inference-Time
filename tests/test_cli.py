import csv
import json
import sys
from io import StringIO

import pytest

from mbt_ai_tools import __version__
from mbt_ai_tools import regulate_candidates
from mbt_ai_tools.cli import (
    build_batch_summary,
    build_decision_explanation,
    build_regulation_report,
    format_csv_audit,
    format_demo_text,
    format_markdown_audit,
    format_markdown_report,
    format_regulation_text,
    main,
)


def test_cli_version_reports_package_version(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["manifold-check", "--version"])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"manifold-check {__version__}"


def test_legacy_cli_version_alias_reports_package_version(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mbt-check", "--version"])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"mbt-check {__version__}"


def test_cli_regulation_json_report_without_embeddings(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--reference",
            "The capital of France is Paris.",
            "--candidate",
            "The capital of France is London.",
            "--candidate",
            "The capital of France is Paris.",
            "--no-embeddings",
            "--format",
            "json",
        ],
    )

    assert main() == 0

    report = json.loads(capsys.readouterr().out)
    assert report["action"] == "emit"
    assert report["emitted_text"] == "The capital of France is Paris."
    assert report["emitted_index"] == 1
    assert report["evaluations"][0]["status"] == "blocked"
    assert report["evaluations"][0]["pool_group_key"] == "the capital of france is london"
    assert report["evaluations"][0]["duplicate_of"] is None
    assert report["evaluations"][0]["selection_status"] == "blocked_before_ranking"
    assert report["evaluations"][0]["selection_rank"] is None
    assert "known_participant_unsupported_relation_clamp" in report["evaluations"][0]["clamps"]
    assert report["evaluations"][1]["status"] == "safe"
    assert report["evaluations"][1]["pool_group_key"] == "the capital of france is paris"
    assert report["evaluations"][1]["duplicate_of"] is None
    assert report["evaluations"][1]["selection_status"] == "emitted"
    assert report["evaluations"][1]["selection_rank"] == 1


def test_cli_regulation_json_report_can_include_explanations(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--reference",
            "The capital of France is Paris.",
            "--candidate",
            "The capital of France is London.",
            "--candidate",
            "The capital of France is Paris.",
            "--no-embeddings",
            "--format",
            "json",
            "--explain",
        ],
    )

    assert main() == 0

    report = json.loads(capsys.readouterr().out)
    blocked = report["evaluations"][0]["explanation"]
    safe = report["evaluations"][1]["explanation"]
    assert blocked["decision"] == "blocked"
    assert "known_participant_unsupported_relation_clamp" in {
        reason["code"] for reason in blocked["reasons"]
    }
    assert safe["decision"] == "safe"
    assert "exactly matches a supplied reference" in safe["summary"]


def test_cli_why_alias_includes_text_explanations(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--reference",
            "Water is liquid at room temperature.",
            "--candidate",
            "Water is not liquid at room temperature.",
            "--no-embeddings",
            "--why",
        ],
    )

    assert main() == 0

    output = capsys.readouterr().out
    assert output.startswith("BLOCK | no safe candidate")
    assert "explain | Blocked because these guards fired:" in output
    assert "reason | negated_positive_support_clamp" in output


def test_cli_regulation_json_report_can_write_output(monkeypatch, capsys, tmp_path):
    output_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--reference",
            "The capital of France is Paris.",
            "--candidate",
            "The capital of France is Paris.",
            "--no-embeddings",
            "--format",
            "json",
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0

    assert capsys.readouterr().out == ""
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["action"] == "emit"
    assert report["emitted_text"] == "The capital of France is Paris."


def test_cli_batch_jsonl_report(monkeypatch, capsys, tmp_path):
    input_path = tmp_path / "batch.jsonl"
    output_path = tmp_path / "batch-report.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "france-capital",
                        "references": ["The capital of France is Paris."],
                        "candidates": [
                            "The capital of France is London.",
                            "The capital of France is Paris.",
                        ],
                    }
                ),
                json.dumps(
                    {
                        "id": "negation",
                        "reference": "Water is liquid at room temperature.",
                        "candidate": "Water is not liquid at room temperature.",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--input-jsonl",
            str(input_path),
            "--no-embeddings",
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0

    assert capsys.readouterr().out == ""
    reports = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [report["id"] for report in reports] == ["france-capital", "negation"]
    assert reports[0]["action"] == "emit"
    assert reports[0]["emitted_index"] == 1
    assert reports[1]["action"] == "block"
    assert "negated_positive_support_clamp" in reports[1]["evaluations"][0]["clamps"]


def test_cli_batch_summary_and_fail_on_block(monkeypatch, capsys, tmp_path):
    input_path = tmp_path / "batch.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "negation",
                "reference": "Water is liquid at room temperature.",
                "candidate": "Water is not liquid at room temperature.",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--input-jsonl",
            str(input_path),
            "--no-embeddings",
            "--summary",
            "--fail-on-block",
        ],
    )

    assert main() == 2

    reports = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
    ]
    assert reports[0]["action"] == "block"
    assert reports[1] == {
        "blocked": 1,
        "blocked_candidates": 1,
        "emitted": 0,
        "record_type": "summary",
        "safe_candidates": 0,
        "total": 1,
    }


def test_cli_batch_markdown_audit(monkeypatch, capsys, tmp_path):
    input_path = tmp_path / "batch.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "france-capital",
                        "references": ["The capital of France is Paris."],
                        "candidates": [
                            "The capital of France is London.",
                            "The capital of France is Paris.",
                        ],
                    }
                ),
                json.dumps(
                    {
                        "id": "negation",
                        "reference": "Water is liquid at room temperature.",
                        "candidate": "Water is not liquid at room temperature.",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--input-jsonl",
            str(input_path),
            "--no-embeddings",
            "--format",
            "markdown",
            "--fail-on-block",
        ],
    )

    assert main() == 2
    output = capsys.readouterr().out
    assert output.startswith("# ManifoldGuard Audit Report\n")
    assert "- Total cases: 2" in output
    assert "- Blocked: 1" in output
    assert "## Case: france-capital" in output
    assert "## Case: negation" in output
    assert "negated_positive_support_clamp" in output


def test_cli_batch_csv_audit(monkeypatch, capsys, tmp_path):
    input_path = tmp_path / "batch.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "france-capital",
                        "references": ["The capital of France is Paris."],
                        "candidates": [
                            "The capital of France is London.",
                            "The capital of France is Paris.",
                        ],
                    }
                ),
                json.dumps(
                    {
                        "id": "negation",
                        "reference": "Water is liquid at room temperature.",
                        "candidate": "Water is not liquid at room temperature.",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--input-jsonl",
            str(input_path),
            "--no-embeddings",
            "--format",
            "csv",
            "--fail-on-block",
        ],
    )

    assert main() == 2
    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert len(rows) == 3
    assert rows[0]["case_id"] == "france-capital"
    assert rows[0]["candidate_index"] == "0"
    assert rows[0]["status"] == "blocked"
    assert rows[1]["emitted_index"] == "1"
    assert rows[1]["status"] == "safe"
    assert rows[2]["case_id"] == "negation"
    assert rows[2]["action"] == "block"
    assert "negated_positive_support_clamp" in rows[2]["clamps"]


def test_cli_demo_text_runs_built_in_offline_demo(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--demo",
            "--why",
        ],
    )

    assert main() == 0

    output = capsys.readouterr().out
    assert output.startswith("ManifoldGuard demo (offline)\n")
    assert "safe_emit_reference_match -> emit" in output
    assert "blocked_negation -> block" in output
    assert "blocked_numeric_drift -> block" in output
    assert "why | Blocked because these guards fired:" in output


def test_cli_demo_markdown_and_fail_on_block(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--demo",
            "--format",
            "markdown",
            "--why",
            "--fail-on-block",
        ],
    )

    assert main() == 2

    output = capsys.readouterr().out
    assert output.startswith("# ManifoldGuard Audit Report\n")
    assert "## Case: safe_emit_reference_match" in output
    assert "## Case: blocked_negation" in output
    assert "## Case: blocked_numeric_drift" in output
    assert "- Action: block" in output


def test_cli_demo_json_summary(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--demo",
            "--format",
            "json",
            "--summary",
        ],
    )

    assert main() == 0

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["id"] for record in records[:3]] == [
        "safe_emit_reference_match",
        "blocked_negation",
        "blocked_numeric_drift",
    ]
    assert records[-1] == {
        "blocked": 2,
        "blocked_candidates": 2,
        "emitted": 1,
        "record_type": "summary",
        "safe_candidates": 1,
        "total": 3,
    }


def test_cli_demo_disallows_conflicting_inputs(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--demo",
            "--candidate",
            "The capital of France is Paris.",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--demo cannot be combined" in captured.err


def test_cli_disallows_token_shock_with_no_embeddings_in_single_mode(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--reference",
            "The capital of France is Paris.",
            "--candidate",
            "The capital of France is Paris.",
            "--no-embeddings",
            "--token-shock",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--token-shock requires sentence-transformers" in captured.err
    assert "Remove --no-embeddings or install with .[embeddings]." in captured.err


def test_cli_single_regulation_fail_on_block(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--reference",
            "Water is liquid at room temperature.",
            "--candidate",
            "Water is not liquid at room temperature.",
            "--no-embeddings",
            "--fail-on-block",
        ],
    )

    assert main() == 2
    assert capsys.readouterr().out.startswith("BLOCK | no safe candidate")


def test_cli_single_markdown_report(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--reference",
            "The capital of France is Paris.",
            "--candidate",
            "The capital of France is London.",
            "--candidate",
            "The capital of France is Paris.",
            "--no-embeddings",
            "--format",
            "markdown",
        ],
    )

    assert main() == 0
    output = capsys.readouterr().out
    assert output.startswith("# ManifoldGuard Regulation Report\n")
    assert "- Action: emit" in output
    assert "#### Candidate 0 - blocked" in output
    assert "#### Candidate 1 - safe" in output


def test_cli_disallows_token_shock_with_no_embeddings_in_batch_mode(monkeypatch, capsys, tmp_path):
    input_path = tmp_path / "batch.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "france-capital",
                "references": ["The capital of France is Paris."],
                "candidates": [
                    "The capital of France is London.",
                    "The capital of France is Paris.",
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifold-check",
            "--input-jsonl",
            str(input_path),
            "--no-embeddings",
            "--token-shock",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--token-shock requires sentence-transformers" in captured.err
    assert "Remove --no-embeddings or install with .[embeddings]." in captured.err


def test_build_regulation_report_can_include_token_shock(monkeypatch):
    calls = []

    def fake_token_shock_map(text, *, max_samples=None, top_k=None, order="token"):
        calls.append((text, max_samples, top_k, order))
        return [("London", 12.5)]

    monkeypatch.setattr("mbt_ai_tools.cli.token_shock_map", fake_token_shock_map)
    result = regulate_candidates(
        [
            "The capital of France is London.",
            "The capital of France is Paris.",
        ],
        ["The capital of France is Paris."],
        use_embeddings=False,
    )

    report = build_regulation_report(
        result,
        include_token_shock=True,
        token_shock_max_samples=4,
        token_shock_top_k=1,
        token_shock_order="score",
    )

    assert calls == [
        ("The capital of France is London.", 4, 1, "score"),
        ("The capital of France is Paris.", 4, 1, "score"),
    ]
    assert report["evaluations"][0]["token_shock"] == [
        {"token": "London", "shock": 12.5}
    ]


def test_report_formats_include_selection_diagnostics():
    result = regulate_candidates(
        [
            "The capital of France is London.",
            "The capital of France is Paris.",
        ],
        ["The capital of France is Paris."],
        use_embeddings=False,
    )
    report = build_regulation_report(result)

    assert report["evaluations"][0]["selection_status"] == "blocked_before_ranking"
    assert report["evaluations"][1]["selection_status"] == "emitted"
    assert "lowest-score safe candidate" in report["evaluations"][1]["selection_reason"]

    csv_rows = list(csv.DictReader(StringIO(format_csv_audit([report]))))
    assert csv_rows[0]["selection_status"] == "blocked_before_ranking"
    assert csv_rows[0]["pool_group_key"] == "the capital of france is london"
    assert csv_rows[0]["duplicate_of"] == ""
    assert csv_rows[0]["selection_rank"] == ""
    assert csv_rows[1]["selection_status"] == "emitted"
    assert csv_rows[1]["pool_group_key"] == "the capital of france is paris"
    assert csv_rows[1]["duplicate_of"] == ""
    assert csv_rows[1]["selection_rank"] == "1"

    markdown = format_markdown_report(report)
    assert "- Pool group: the capital of france is london" in markdown
    assert "- Duplicate of: `null`" in markdown
    assert "- Selection: blocked_before_ranking" in markdown
    assert "- Selection: emitted" in markdown
    assert "- Selection rank: 1" in markdown


def test_build_batch_summary_counts_reports_and_candidates():
    reports = [
        {
            "action": "emit",
            "evaluations": [
                {"safe_to_emit": False},
                {"safe_to_emit": True},
            ],
        },
        {
            "action": "block",
            "evaluations": [
                {"safe_to_emit": False},
            ],
        },
    ]

    assert build_batch_summary(reports) == {
        "blocked": 1,
        "blocked_candidates": 2,
        "emitted": 1,
        "record_type": "summary",
        "safe_candidates": 1,
        "total": 2,
    }


def test_format_markdown_helpers_render_expected_sections():
    result = regulate_candidates(
        [
            "The capital of France is London.",
            "The capital of France is Paris.",
        ],
        ["The capital of France is Paris."],
        use_embeddings=False,
    )
    report = build_regulation_report(result)

    single = format_markdown_report(report)
    audit = format_markdown_audit([{**report, "id": "france-capital", "line": 1}])

    assert "# ManifoldGuard Regulation Report" in single
    assert "known_participant_unsupported_relation_clamp" in single
    assert "# ManifoldGuard Audit Report" in audit
    assert "## Case: france-capital" in audit


def test_format_csv_audit_renders_candidate_rows():
    result = regulate_candidates(
        [
            "The capital of France is London.",
            "The capital of France is Paris.",
        ],
        ["The capital of France is Paris."],
        use_embeddings=False,
    )
    report = build_regulation_report(result)

    output = format_csv_audit(
        [
            {
                **report,
                "id": "france-capital",
                "line": 1,
                "references": ["The capital of France is Paris."],
            }
        ]
    )

    rows = list(csv.DictReader(StringIO(output)))
    assert rows[0]["case_id"] == "france-capital"
    assert rows[0]["references"] == "The capital of France is Paris."
    assert rows[0]["candidate_text"] == "The capital of France is London."
    assert rows[0]["safe_to_emit"] == "false"
    assert rows[1]["candidate_text"] == "The capital of France is Paris."
    assert rows[1]["safe_to_emit"] == "true"


def test_format_regulation_text_matches_existing_cli_shape():
    result = regulate_candidates(
        [
            "The capital of France is London.",
            "The capital of France is Paris.",
        ],
        ["The capital of France is Paris."],
        use_embeddings=False,
    )

    output = format_regulation_text(build_regulation_report(result))

    assert output.startswith("EMIT | The capital of France is Paris. | score=0.0000\n")
    assert "[0] blocked | score=" in output
    assert "[1] safe | score=0.0000 | clamps=exact_reference_member" in output


def test_format_demo_text_summarizes_actions():
    reports = [
        {
            "id": "safe_emit_reference_match",
            "action": "emit",
            "emitted_text": "The capital of France is Paris.",
            "evaluations": [{"safe_to_emit": True, "clamps": ["exact_reference_member"]}],
        },
        {
            "id": "blocked_negation",
            "action": "block",
            "emitted_text": None,
            "evaluations": [
                {
                    "safe_to_emit": False,
                    "clamps": ["negated_positive_support_clamp"],
                    "explanation": {
                        "summary": "Blocked because these guards fired: negated_positive_support_clamp."
                    },
                }
            ],
        },
    ]

    output = format_demo_text(reports)

    assert "safe_emit_reference_match -> emit" in output
    assert "blocked_negation -> block" in output
    assert "block | negated_positive_support_clamp" in output
    assert "Use --format markdown for a full audit report." in output


def test_explain_text_and_markdown_render_decision_reasons():
    result = regulate_candidates(
        [
            "The capital of France is London.",
            "The capital of France is Paris.",
        ],
        ["The capital of France is Paris."],
        use_embeddings=False,
    )
    report = build_regulation_report(result, include_explanations=True)

    text = format_regulation_text(report)
    markdown = format_markdown_report(report)

    assert "explain | Blocked because these guards fired:" in text
    assert "reason | known_participant_unsupported_relation_clamp" in text
    assert "Explanation:" in markdown
    assert "`known_participant_unsupported_relation_clamp`" in markdown


def test_build_decision_explanation_describes_safe_reference_member():
    result = regulate_candidates(
        ["The capital of France is Paris."],
        ["The capital of France is Paris."],
        use_embeddings=False,
    )

    explanation = build_decision_explanation(result.evaluations[0])

    assert explanation["decision"] == "safe"
    assert explanation["reasons"][0]["code"] == "exact_reference_member"
    assert "exactly matches a supplied reference" in explanation["summary"]
