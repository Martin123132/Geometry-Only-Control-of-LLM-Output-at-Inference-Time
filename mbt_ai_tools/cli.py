import argparse
import csv
import json
from io import StringIO
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import __version__
from .mbt.regulator import regulate_candidates
from .mbt.stability import classify_entropy, confidence_score
from .mbt.tokens import token_shock_map


DEMO_CASES = [
    {
        "id": "safe_emit_reference_match",
        "references": ["The capital of France is Paris."],
        "candidates": ["The capital of France is Paris."],
    },
    {
        "id": "blocked_negation",
        "references": ["Water is liquid at room temperature."],
        "candidates": ["Water is not liquid at room temperature."],
    },
    {
        "id": "blocked_numeric_drift",
        "references": ["The medication dose is 10 mg."],
        "candidates": ["The medication dose is 20 mg."],
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ManifoldGuard geometry-only confidence probe and v11 candidate regulator."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help='Text or blank-line separated responses to score, e.g. `manifold-check "answer a\n\nanswer b"`',
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
        "--input-jsonl",
        type=Path,
        help="Batch regulation input JSONL. Each line needs references and candidates fields.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the built-in offline demo corpus without needing a local examples checkout.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write command output to a file instead of stdout.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Append a batch summary JSON object when using --input-jsonl.",
    )
    parser.add_argument(
        "--fail-on-block",
        action="store_true",
        help="Exit with status 2 when a regulation run blocks.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Run only literal/relation/negation clamps without semantic embeddings.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "markdown", "csv"),
        default="text",
        help="Output format for regulation reports. Batch mode defaults to JSONL unless markdown or csv is selected.",
    )
    parser.add_argument(
        "--explain",
        "--why",
        dest="explain",
        action="store_true",
        help="Include per-candidate decision explanations in text, JSON, and Markdown reports. --why is a friendly alias.",
    )
    parser.add_argument(
        "--token-shock",
        action="store_true",
        help="Include token-level shock scores for each candidate in regulation mode.",
    )
    parser.add_argument(
        "--token-shock-max-samples",
        type=int,
        default=None,
        help="Maximum number of token-removal variants to score when --token-shock is used.",
    )
    parser.add_argument(
        "--token-shock-top-k",
        type=int,
        default=None,
        help="Return only the highest-shock tokens when --token-shock is used.",
    )
    parser.add_argument(
        "--token-shock-order",
        choices=("token", "score"),
        default="score",
        help="Order token shock output by original token position or descending score.",
    )
    args = parser.parse_args()

    if args.token_shock and args.no_embeddings:
        parser.error(
            "--token-shock requires sentence-transformers. "
            "Remove --no-embeddings or install with .[embeddings]."
        )

    if args.demo:
        if args.input_jsonl or args.reference or args.candidate or args.text:
            parser.error("--demo cannot be combined with positional text, --input-jsonl, --reference, or --candidate")
        if args.token_shock:
            parser.error("--demo runs offline and cannot be combined with --token-shock")
        reports = list(build_demo_reports(include_explanations=args.explain))
        if args.format == "markdown":
            content = format_markdown_audit(reports)
        elif args.format == "csv":
            content = format_csv_audit(reports)
        elif args.format == "json":
            output_items = reports[:]
            if args.summary:
                output_items.append(build_batch_summary(reports))
            content = "".join(f"{json.dumps(report, sort_keys=True)}\n" for report in output_items)
        else:
            content = format_demo_text(reports)
        _emit_output(content, args.output)
        return 2 if args.fail_on_block and any(report["action"] == "block" for report in reports) else 0

    if args.input_jsonl:
        if args.reference or args.candidate or args.text:
            parser.error("--input-jsonl cannot be combined with positional text, --reference, or --candidate")
        try:
            reports = list(
                build_batch_reports(
                    args.input_jsonl,
                    use_embeddings=not args.no_embeddings,
                    include_explanations=args.explain,
                    include_token_shock=args.token_shock,
                    token_shock_max_samples=args.token_shock_max_samples,
                    token_shock_top_k=args.token_shock_top_k,
                    token_shock_order=args.token_shock_order,
                )
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if args.format == "markdown":
            content = format_markdown_audit(reports)
        elif args.format == "csv":
            content = format_csv_audit(reports)
        else:
            output_items = reports[:]
            if args.summary:
                output_items.append(build_batch_summary(reports))
            content = "".join(f"{json.dumps(report, sort_keys=True)}\n" for report in output_items)
        _emit_output(content, args.output)
        return 2 if args.fail_on_block and any(report["action"] == "block" for report in reports) else 0

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
        report = build_regulation_report(
            result,
            include_explanations=args.explain,
            include_token_shock=args.token_shock,
            token_shock_max_samples=args.token_shock_max_samples,
            token_shock_top_k=args.token_shock_top_k,
            token_shock_order=args.token_shock_order,
        )
        if args.format == "json":
            content = f"{json.dumps(report, indent=2, sort_keys=True)}\n"
        elif args.format == "markdown":
            content = format_markdown_report(report)
        elif args.format == "csv":
            content = format_csv_report(report)
        else:
            content = format_regulation_text(report)
        _emit_output(content, args.output)
        return 2 if args.fail_on_block and report["action"] == "block" else 0

    if args.text is None:
        parser.error("text is required unless --candidate is supplied")

    score = confidence_score(args.text)
    label, _ = classify_entropy(score)

    _emit_output(f"{label} | Internal Entropy: {score:.4f}\n", args.output)
    return 0


def build_regulation_report(
    result,
    *,
    include_explanations: bool = False,
    include_token_shock: bool = False,
    token_shock_max_samples: Optional[int] = None,
    token_shock_top_k: Optional[int] = None,
    token_shock_order: str = "score",
) -> Dict[str, Any]:
    emitted_index = None
    if result.emitted is not None:
        emitted_index = next(
            (
                index
                for index, evaluation in enumerate(result.evaluations)
                if evaluation is result.emitted
            ),
            None,
        )

    evaluations = []
    for index, evaluation in enumerate(result.evaluations):
        item: Dict[str, Any] = {
            "index": index,
            "text": evaluation.text,
            "pool_group_key": evaluation.pool_group_key,
            "duplicate_of": evaluation.duplicate_of,
            "status": "safe" if evaluation.safe_to_emit else "blocked",
            "selection_status": evaluation.selection_status,
            "selection_rank": evaluation.selection_rank,
            "selection_reason": evaluation.selection_reason,
            "safe_to_emit": evaluation.safe_to_emit,
            "pred_hallucinated": evaluation.pred_hallucinated,
            "regulator_score": evaluation.regulator_score,
            "mbt5_shock": evaluation.mbt5_shock,
            "threshold": evaluation.threshold,
            "literal_score": evaluation.literal_score,
            "clamps": list(evaluation.clamp_summary),
            "relations": list(evaluation.relations),
            "negated_relations": list(evaluation.negated_relations),
            "exact_reference_member": evaluation.exact_reference_member,
        }
        if include_explanations:
            item["explanation"] = build_decision_explanation(evaluation)
        if include_token_shock:
            item["token_shock"] = [
                {"token": token, "shock": score}
                for token, score in token_shock_map(
                    evaluation.text,
                    max_samples=token_shock_max_samples,
                    top_k=token_shock_top_k,
                    order=token_shock_order,
                )
            ]
        evaluations.append(item)

    candidate_pool = {
        "total_candidates": len(evaluations),
        "unique_pool_groups": len(
            {evaluation["pool_group_key"] for evaluation in evaluations}
        ),
        "duplicate_candidates": sum(
            1 for evaluation in evaluations if evaluation["duplicate_of"] is not None
        ),
        "safe_candidates": sum(
            1 for evaluation in evaluations if evaluation["safe_to_emit"]
        ),
        "blocked_candidates": sum(
            1 for evaluation in evaluations if not evaluation["safe_to_emit"]
        ),
    }

    return {
        "action": result.action,
        "candidate_pool": candidate_pool,
        "emitted_text": result.emitted_text,
        "emitted_index": emitted_index,
        "emitted_score": None
        if emitted_index is None
        else evaluations[emitted_index]["regulator_score"],
        "evaluations": evaluations,
    }


CLAMP_EXPLANATIONS = {
    "protected_number": "Candidate introduces a number that is not supported by the supplied references.",
    "protected_unit": "Candidate introduces a unit that is not supported by the supplied references.",
    "protected_entity": "Candidate introduces a named entity that is not supported by the supplied references.",
    "protected_literal_drift": "Candidate changes protected literal content such as numbers, units, or entities.",
    "final_literal_block": "Literal drift exceeded the offline blocking threshold.",
    "content_clamp_flag": "Candidate introduces unsupported content tokens beyond ordinary paraphrase.",
    "overclaim_flag": "Candidate makes a stronger claim than the supplied references support.",
    "negated_positive_support_clamp": "Candidate negates a relation that is positively supported by the references.",
    "role_swapped_relation_clamp": "Candidate reverses a supported relation between known participants.",
    "known_participant_unsupported_relation_clamp": "Candidate uses known reference participants in an unsupported relation.",
    "missed_predicate_relation_clamp": "Candidate changes the supported predicate for a known reference participant.",
    "historical_date_relation_clamp": "Candidate changes a supported historical/date relation.",
    "guarded_known_participant_unsupported_relation_clamp": "Guarded relation recall confirmed unsupported known-participant drift.",
    "exp19b_guarded_patch_clamp": "Guarded relation patch lineage marked this candidate as unsupported.",
    "final_geometric_block": "Embedding-backed semantic shock exceeded the active manifold threshold.",
    "exact_reference_member": "Candidate exactly matches a supplied reference after normalization.",
    "reference_member_geometry_override": "Exact reference membership overrode a geometry-only block.",
}


def build_decision_explanation(evaluation) -> Dict[str, Any]:
    """Build a compact, user-facing explanation for one candidate decision."""

    clamps = [clamp for clamp in evaluation.clamp_summary if clamp != "none"]
    reasons = [
        {
            "code": clamp,
            "message": CLAMP_EXPLANATIONS.get(
                clamp,
                "Candidate triggered this regulator guard.",
            ),
        }
        for clamp in clamps
    ]

    if evaluation.safe_to_emit:
        if evaluation.exact_reference_member:
            summary = "Safe because the candidate exactly matches a supplied reference."
        elif reasons:
            summary = "Safe after non-blocking guards because no hard block remained active."
        else:
            summary = "Safe because no blocking literal, relation, negation, or geometry guard fired."
    elif reasons:
        codes = ", ".join(reason["code"] for reason in reasons)
        summary = f"Blocked because these guards fired: {codes}."
    else:
        summary = "Blocked by the regulator, but no named clamp was recorded."

    return {
        "decision": "safe" if evaluation.safe_to_emit else "blocked",
        "summary": summary,
        "reasons": reasons,
    }


def build_batch_reports(
    input_jsonl: Path,
    *,
    use_embeddings: bool = True,
    include_explanations: bool = False,
    include_token_shock: bool = False,
    token_shock_max_samples: Optional[int] = None,
    token_shock_top_k: Optional[int] = None,
    token_shock_order: str = "score",
) -> Iterable[Dict[str, Any]]:
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                item = json.loads(raw_line)
            except JSONDecodeError as exc:
                raise ValueError(
                    f"{input_jsonl}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(f"{input_jsonl}:{line_number}: expected a JSON object")

            references = _text_list(item, "references", "reference", input_jsonl, line_number)
            candidates = _text_list(item, "candidates", "candidate", input_jsonl, line_number)
            result = regulate_candidates(
                candidates,
                references,
                use_embeddings=use_embeddings,
            )
            report = build_regulation_report(
                result,
                include_explanations=include_explanations,
                include_token_shock=include_token_shock,
                token_shock_max_samples=token_shock_max_samples,
                token_shock_top_k=token_shock_top_k,
                token_shock_order=token_shock_order,
            )
            report["line"] = line_number
            report["references"] = references
            if "id" in item:
                report["id"] = item["id"]
            yield report


def build_demo_reports(*, include_explanations: bool = False) -> Iterable[Dict[str, Any]]:
    for line_number, item in enumerate(DEMO_CASES, start=1):
        result = regulate_candidates(
            item["candidates"],
            item["references"],
            use_embeddings=False,
        )
        report = build_regulation_report(
            result,
            include_explanations=include_explanations,
        )
        report["line"] = line_number
        report["references"] = item["references"]
        report["id"] = item["id"]
        yield report


def build_batch_summary(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    safe_candidates = sum(
        1
        for report in reports
        for evaluation in report["evaluations"]
        if evaluation["safe_to_emit"]
    )
    blocked_candidates = sum(
        1
        for report in reports
        for evaluation in report["evaluations"]
        if not evaluation["safe_to_emit"]
    )
    return {
        "record_type": "summary",
        "total": len(reports),
        "emitted": sum(1 for report in reports if report["action"] == "emit"),
        "blocked": sum(1 for report in reports if report["action"] == "block"),
        "safe_candidates": safe_candidates,
        "blocked_candidates": blocked_candidates,
    }


CSV_FIELDNAMES = [
    "case_id",
    "line",
    "references",
    "action",
    "emitted_index",
    "emitted_text",
    "pool_total_candidates",
    "pool_unique_groups",
    "pool_duplicate_candidates",
    "pool_safe_candidates",
    "pool_blocked_candidates",
    "candidate_index",
    "candidate_text",
    "pool_group_key",
    "duplicate_of",
    "status",
    "selection_status",
    "selection_rank",
    "selection_reason",
    "safe_to_emit",
    "pred_hallucinated",
    "regulator_score",
    "mbt5_shock",
    "threshold",
    "literal_score",
    "clamps",
    "relations",
    "negated_relations",
    "exact_reference_member",
    "token_shock",
]


def format_csv_report(report: Dict[str, Any]) -> str:
    return format_csv_audit([report])


def format_csv_audit(reports: List[Dict[str, Any]]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=CSV_FIELDNAMES,
        lineterminator="\n",
    )
    writer.writeheader()
    for report in reports:
        for evaluation in report["evaluations"]:
            writer.writerow(
                {
                    "case_id": _csv_value(report.get("id")),
                    "line": _csv_value(report.get("line")),
                    "references": " || ".join(report.get("references", [])),
                    "action": report["action"],
                    "emitted_index": _csv_value(report["emitted_index"]),
                    "emitted_text": _csv_value(report["emitted_text"]),
                    "pool_total_candidates": report["candidate_pool"]["total_candidates"],
                    "pool_unique_groups": report["candidate_pool"]["unique_pool_groups"],
                    "pool_duplicate_candidates": report["candidate_pool"]["duplicate_candidates"],
                    "pool_safe_candidates": report["candidate_pool"]["safe_candidates"],
                    "pool_blocked_candidates": report["candidate_pool"]["blocked_candidates"],
                    "candidate_index": evaluation["index"],
                    "candidate_text": evaluation["text"],
                    "pool_group_key": evaluation["pool_group_key"],
                    "duplicate_of": _csv_value(evaluation["duplicate_of"]),
                    "status": evaluation["status"],
                    "selection_status": evaluation["selection_status"],
                    "selection_rank": _csv_value(evaluation["selection_rank"]),
                    "selection_reason": evaluation["selection_reason"],
                    "safe_to_emit": _csv_bool(evaluation["safe_to_emit"]),
                    "pred_hallucinated": _csv_bool(evaluation["pred_hallucinated"]),
                    "regulator_score": _csv_float(evaluation["regulator_score"]),
                    "mbt5_shock": _csv_float(evaluation["mbt5_shock"]),
                    "threshold": _csv_float(evaluation["threshold"]),
                    "literal_score": _csv_float(evaluation["literal_score"]),
                    "clamps": "; ".join(evaluation["clamps"]),
                    "relations": _csv_relations(evaluation["relations"]),
                    "negated_relations": _csv_relations(evaluation["negated_relations"]),
                    "exact_reference_member": _csv_bool(evaluation["exact_reference_member"]),
                    "token_shock": _csv_token_shock(evaluation.get("token_shock", [])),
                }
            )
    return buffer.getvalue()


def format_markdown_audit(reports: List[Dict[str, Any]]) -> str:
    summary = build_batch_summary(reports)
    lines = [
        "# ManifoldGuard Audit Report",
        "",
        "## Summary",
        "",
        f"- Total cases: {summary['total']}",
        f"- Emitted: {summary['emitted']}",
        f"- Blocked: {summary['blocked']}",
        f"- Safe candidate evaluations: {summary['safe_candidates']}",
        f"- Blocked candidate evaluations: {summary['blocked_candidates']}",
        "",
    ]

    for report in reports:
        label = report.get("id") or f"line {report.get('line', '?')}"
        lines.extend(
            [
                f"## Case: {label}",
                "",
                f"- Action: {report['action']}",
                f"- Emitted index: {_markdown_value(report['emitted_index'])}",
                f"- Emitted text: {_markdown_value(report['emitted_text'])}",
            ]
        )
        if "line" in report:
            lines.append(f"- Input line: {report['line']}")
        lines.extend(
            ["", *_markdown_candidate_pool(report), "", *_markdown_evaluations(report), ""]
        )

    return "\n".join(lines).rstrip() + "\n"


def format_markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# ManifoldGuard Regulation Report",
        "",
        f"- Action: {report['action']}",
        f"- Emitted index: {_markdown_value(report['emitted_index'])}",
        f"- Emitted text: {_markdown_value(report['emitted_text'])}",
        "",
        *_markdown_candidate_pool(report),
        "",
        *_markdown_evaluations(report),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _markdown_candidate_pool(report: Dict[str, Any]) -> List[str]:
    summary = report["candidate_pool"]
    return [
        "### Candidate Pool",
        "",
        f"- Total candidates: {summary['total_candidates']}",
        f"- Unique pool groups: {summary['unique_pool_groups']}",
        f"- Duplicate candidates: {summary['duplicate_candidates']}",
        f"- Safe candidates: {summary['safe_candidates']}",
        f"- Blocked candidates: {summary['blocked_candidates']}",
    ]


def _markdown_evaluations(report: Dict[str, Any]) -> List[str]:
    lines = ["### Candidate Evaluations", ""]
    for evaluation in report["evaluations"]:
        lines.extend(
            [
                f"#### Candidate {evaluation['index']} - {evaluation['status']}",
                "",
                f"- Text: {_markdown_value(evaluation['text'])}",
                f"- Pool group: {_markdown_value(evaluation['pool_group_key'])}",
                f"- Duplicate of: {_markdown_value(evaluation['duplicate_of'])}",
                f"- Score: {evaluation['regulator_score']:.4f}",
                f"- Shock: {evaluation['mbt5_shock']:.4f}",
                f"- Literal score: {evaluation['literal_score']:.4f}",
                f"- Clamps: {', '.join(evaluation['clamps'])}",
                f"- Selection: {evaluation['selection_status']}",
                f"- Selection rank: {_markdown_value(evaluation['selection_rank'])}",
                f"- Selection reason: {evaluation['selection_reason']}",
                f"- Relations: {_markdown_relations(evaluation['relations'])}",
                f"- Negated relations: {_markdown_relations(evaluation['negated_relations'])}",
                "",
            ]
        )
        explanation = evaluation.get("explanation")
        if explanation:
            lines.extend(
                [
                    "Explanation:",
                    "",
                    f"- Decision: {explanation['decision']}",
                    f"- Summary: {explanation['summary']}",
                ]
            )
            if explanation["reasons"]:
                lines.append("- Reasons:")
                lines.extend(
                    f"  - `{reason['code']}`: {reason['message']}"
                    for reason in explanation["reasons"]
                )
            lines.append("")
        token_shock = evaluation.get("token_shock", [])
        if token_shock:
            lines.extend(["Token shock:", ""])
            lines.extend(
                f"- `{entry['token']}`: {entry['shock']:.4f}"
                for entry in token_shock
            )
            lines.append("")
    return lines


def format_regulation_text(report: Dict[str, Any]) -> str:
    if report["action"] == "block":
        lines = ["BLOCK | no safe candidate"]
    else:
        lines = [
            f"EMIT | {report['emitted_text']} | score={report['emitted_score']:.4f}"
        ]

    for evaluation in report["evaluations"]:
        clamps = "|".join(evaluation["clamps"])
        lines.append(
            f"[{evaluation['index']}] {evaluation['status']} | "
            f"score={evaluation['regulator_score']:.4f} | clamps={clamps}"
        )
        for token in evaluation.get("token_shock", []):
            lines.append(f"    token_shock | {token['token']} | shock={token['shock']:.4f}")
        explanation = evaluation.get("explanation")
        if explanation:
            lines.append(f"    explain | {explanation['summary']}")
            for reason in explanation["reasons"]:
                lines.append(f"    reason | {reason['code']} | {reason['message']}")
    return "\n".join(lines) + "\n"


def format_demo_text(reports: List[Dict[str, Any]]) -> str:
    lines = [
        "ManifoldGuard demo (offline)",
        "",
    ]
    for report in reports:
        lines.append(f"{report['id']} -> {report['action']}")
        if report["action"] == "emit":
            lines.append(f"    emit | {report['emitted_text']}")
        else:
            blocked = next(
                (
                    evaluation
                    for evaluation in report["evaluations"]
                    if not evaluation["safe_to_emit"]
                ),
                None,
            )
            if blocked is not None:
                primary_guard = next(
                    (clamp for clamp in blocked["clamps"] if clamp != "none"),
                    "regulator_block",
                )
                lines.append(f"    block | {primary_guard}")
                explanation = blocked.get("explanation")
                if explanation:
                    lines.append(f"    why | {explanation['summary']}")
        lines.append("")
    lines.append("Use --format markdown for a full audit report.")
    return "\n".join(lines).rstrip() + "\n"


def _markdown_relations(relations: List[List[str]]) -> str:
    if not relations:
        return "none"
    return "; ".join(" / ".join(str(part) for part in relation) for relation in relations)


def _csv_relations(relations: List[List[str]]) -> str:
    return "; ".join(" / ".join(str(part) for part in relation) for relation in relations)


def _csv_token_shock(token_shock: List[Dict[str, Any]]) -> str:
    return "; ".join(
        f"{entry['token']}:{_csv_float(entry['shock'])}" for entry in token_shock
    )


def _csv_bool(value: bool) -> str:
    return "true" if value else "false"


def _csv_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _markdown_value(value: Any) -> str:
    if value is None:
        return "`null`"
    return str(value)


def _text_list(
    item: Dict[str, Any],
    plural_key: str,
    singular_key: str,
    input_jsonl: Path,
    line_number: int,
) -> List[str]:
    value = item.get(plural_key, item.get(singular_key))
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list) and all(isinstance(entry, str) for entry in value):
        values = value
    else:
        raise ValueError(
            f"{input_jsonl}:{line_number}: {plural_key} must be a string or list of strings"
        )
    if not values:
        raise ValueError(f"{input_jsonl}:{line_number}: {plural_key} cannot be empty")
    return values


def _emit_output(content: str, output_path: Optional[Path]) -> None:
    if output_path is None:
        print(content, end="")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
