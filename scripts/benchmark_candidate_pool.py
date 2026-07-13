#!/usr/bin/env python
"""Benchmark offline regulation and reporting on the frozen 257-row EXP20 ledger."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Iterable

from mbt_ai_tools.cli import (
    build_regulation_report,
    format_csv_report,
    format_markdown_report,
)
from mbt_ai_tools.mbt.regulator import regulate_candidates


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER = (
    PROJECT_ROOT
    / "data"
    / "csv_exports"
    / "mbt5_exp20_master_candidate_ledger.csv"
)
DEFAULT_EXPECTED_CANDIDATES = 257
RegulatorCallable = Callable[..., Any]


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def load_ledger(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {"case_id", "text", "label_hallucinated"}
    if not rows:
        raise ValueError(f"{path}: ledger is empty")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}")

    candidates = [row["text"].strip() for row in rows]
    references = unique_in_order(
        row["text"].strip()
        for row in rows
        if row["label_hallucinated"].strip() == "0"
    )
    cases = unique_in_order(row["case_id"].strip() for row in rows)
    if any(not candidate for candidate in candidates):
        raise ValueError(f"{path}: candidate text must not be blank")
    if not references:
        raise ValueError(f"{path}: no valid-labelled reference rows found")

    return {
        "candidates": candidates,
        "references": references,
        "cases": cases,
    }


def timing_summary(samples_seconds: list[float]) -> dict[str, Any]:
    samples_ms = [sample * 1000.0 for sample in samples_seconds]
    ordered = sorted(samples_ms)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "samples": len(samples_ms),
        "mean_ms": round(statistics.fmean(samples_ms), 3),
        "median_ms": round(statistics.median(samples_ms), 3),
        "min_ms": round(min(samples_ms), 3),
        "max_ms": round(max(samples_ms), 3),
        "p95_ms": round(ordered[p95_index], 3),
    }


def encoded_size(text: str) -> int:
    return len(text.encode("utf-8"))


def load_callable(specification: str) -> RegulatorCallable:
    """Load a benchmark callable from ``module:attribute`` notation."""

    module_name, separator, attribute_path = specification.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError("callable must use module:attribute notation")
    value: Any = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    if not callable(value):
        raise TypeError(f"{specification}: resolved value is not callable")
    return value


def run_paired_comparison(
    candidates: list[str],
    references: list[str],
    *,
    baseline_regulator: RegulatorCallable,
    candidate_regulator: RegulatorCallable,
    pairs: int = 10,
    warmup: int = 2,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    max_regression_percent: float = 3.0,
    cases: int = 0,
    source: str = "synthetic",
) -> dict[str, Any]:
    """Compare two offline regulators with alternating order and exact parity."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    if not references:
        raise ValueError("at least one reference is required")
    if pairs < 1:
        raise ValueError("pairs must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be zero or greater")
    if max_regression_percent < 0:
        raise ValueError("max regression percent must be zero or greater")
    if not baseline_label or not candidate_label or baseline_label == candidate_label:
        raise ValueError("baseline and candidate labels must be non-empty and distinct")
    if baseline_regulator is candidate_regulator:
        raise ValueError("baseline and candidate callables must be distinct")

    implementations = (
        ("baseline", baseline_regulator),
        ("candidate", candidate_regulator),
    )

    for pair_index in range(warmup):
        order = implementations if pair_index % 2 == 0 else implementations[::-1]
        for _, implementation in order:
            implementation(candidates, references, use_embeddings=False)

    samples: dict[str, list[float]] = {"baseline": [], "candidate": []}
    mismatch_pairs: list[int] = []
    final_report: dict[str, Any] | None = None

    for pair_index in range(pairs):
        order = implementations if pair_index % 2 == 0 else implementations[::-1]
        reports: dict[str, dict[str, Any]] = {}
        for role, implementation in order:
            started = time.perf_counter()
            result = implementation(candidates, references, use_embeddings=False)
            samples[role].append(time.perf_counter() - started)
            reports[role] = build_regulation_report(result)
        if reports["baseline"] != reports["candidate"]:
            mismatch_pairs.append(pair_index)
        final_report = reports["candidate"]

    if final_report is None:  # pragma: no cover - guarded by pairs validation
        raise RuntimeError("paired benchmark produced no result")

    baseline_timing = timing_summary(samples["baseline"])
    candidate_timing = timing_summary(samples["candidate"])
    deltas_ms = [
        round((baseline - candidate) * 1000.0, 3)
        for baseline, candidate in zip(samples["baseline"], samples["candidate"])
    ]
    baseline_median = baseline_timing["median_ms"]
    candidate_median = candidate_timing["median_ms"]
    speedup_percent = round(
        (1.0 - candidate_median / baseline_median) * 100.0,
        3,
    )
    performance_pass = candidate_median <= baseline_median * (
        1.0 + max_regression_percent / 100.0
    )
    behavior_parity = not mismatch_pairs

    return {
        "schema_version": "1.0",
        "benchmark": "offline_candidate_pool_paired_comparison",
        "evidence_tier": "development_performance",
        "source": {
            "path": source,
            "cases": cases,
            "candidate_rows": len(candidates),
            "reference_rows": len(references),
        },
        "configuration": {
            "use_embeddings": False,
            "warmup_pairs": warmup,
            "measured_pairs": pairs,
            "alternating_order": True,
            "max_regression_percent": max_regression_percent,
        },
        "implementations": {
            "baseline": baseline_label,
            "candidate": candidate_label,
        },
        "behavior": {
            "exact_report_parity": behavior_parity,
            "mismatch_pairs": mismatch_pairs,
        },
        "pool": {
            **final_report["candidate_pool"],
            "action": final_report["action"],
        },
        "timings": {
            "baseline": baseline_timing,
            "candidate": candidate_timing,
        },
        "paired": {
            "delta_definition": "baseline_ms_minus_candidate_ms",
            "deltas_ms": deltas_ms,
            "median_delta_ms": round(statistics.median(deltas_ms), 3),
            "median_speedup_percent": speedup_percent,
            "candidate_faster_or_equal_pairs": sum(delta >= 0 for delta in deltas_ms),
        },
        "acceptance": {
            "behavior_parity_pass": behavior_parity,
            "performance_threshold_pass": performance_pass,
            "accepted": behavior_parity and performance_pass,
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
    }


def run_benchmark(
    candidates: list[str],
    references: list[str],
    *,
    cases: int = 0,
    iterations: int = 10,
    warmup: int = 3,
    source: str = "synthetic",
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    if not references:
        raise ValueError("at least one reference is required")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be zero or greater")

    for _ in range(warmup):
        warmup_result = regulate_candidates(
            candidates,
            references,
            use_embeddings=False,
        )
        json.dumps(
            build_regulation_report(warmup_result),
            separators=(",", ":"),
            sort_keys=True,
        )

    regulation_samples: list[float] = []
    report_samples: list[float] = []
    serialization_samples: list[float] = []
    end_to_end_samples: list[float] = []
    final_result = None

    for _ in range(iterations):
        total_started = time.perf_counter()
        regulation_started = time.perf_counter()
        result = regulate_candidates(
            candidates,
            references,
            use_embeddings=False,
        )
        regulation_samples.append(time.perf_counter() - regulation_started)

        report_started = time.perf_counter()
        report = build_regulation_report(result)
        report_samples.append(time.perf_counter() - report_started)

        serialization_started = time.perf_counter()
        json.dumps(report, separators=(",", ":"), sort_keys=True)
        serialization_samples.append(time.perf_counter() - serialization_started)
        end_to_end_samples.append(time.perf_counter() - total_started)
        final_result = result

    tracemalloc.start()
    memory_result = regulate_candidates(
        candidates,
        references,
        use_embeddings=False,
    )
    memory_report = build_regulation_report(memory_result)
    json.dumps(memory_report, separators=(",", ":"), sort_keys=True)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if final_result is None:  # pragma: no cover - guarded by iterations validation
        raise RuntimeError("benchmark produced no result")

    report = build_regulation_report(final_result)
    blocked_report = build_regulation_report(
        final_result,
        candidate_filter="blocked",
        candidate_order="score",
    )
    duplicate_report = build_regulation_report(
        final_result,
        candidate_filter="duplicates",
        candidate_order="input",
    )
    compact_json = json.dumps(report, separators=(",", ":"), sort_keys=True)
    pool = report["candidate_pool"]
    submitted = pool["total_candidates"]
    duplicates = pool["duplicate_candidates"]

    timings = {
        "regulation": timing_summary(regulation_samples),
        "report_build": timing_summary(report_samples),
        "json_serialization": timing_summary(serialization_samples),
        "end_to_end": timing_summary(end_to_end_samples),
    }
    timings["regulation"]["median_per_submitted_candidate_ms"] = round(
        timings["regulation"]["median_ms"] / submitted,
        6,
    )
    timings["regulation"]["median_per_unique_pool_group_ms"] = round(
        timings["regulation"]["median_ms"] / pool["unique_pool_groups"],
        6,
    )
    median_regulation_seconds = timings["regulation"]["median_ms"] / 1000.0
    timings["regulation"]["median_submitted_candidates_per_second"] = round(
        submitted / median_regulation_seconds,
        3,
    )
    timings["regulation"]["median_unique_pool_groups_per_second"] = round(
        pool["unique_pool_groups"] / median_regulation_seconds,
        3,
    )

    return {
        "schema_version": "1.0",
        "benchmark": "offline_candidate_pool_scale",
        "evidence_tier": "development_performance",
        "source": {
            "path": source,
            "cases": cases,
            "candidate_rows": len(candidates),
            "reference_rows": len(references),
        },
        "configuration": {
            "use_embeddings": False,
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "pool": {
            **pool,
            "evaluations_avoided_by_duplicate_grouping": duplicates,
            "duplicate_compression_ratio": round(duplicates / submitted, 6),
            "action": report["action"],
        },
        "timings": timings,
        "memory": {
            "measurement": "tracemalloc_peak_python_allocations",
            "peak_bytes": peak_bytes,
            "peak_mib": round(peak_bytes / (1024 * 1024), 3),
        },
        "report_sizes": {
            "json_bytes": encoded_size(compact_json),
            "markdown_bytes": encoded_size(format_markdown_report(report)),
            "csv_bytes": encoded_size(format_csv_report(report)),
            "blocked_filter_json_bytes": encoded_size(
                json.dumps(blocked_report, separators=(",", ":"), sort_keys=True)
            ),
            "duplicate_filter_json_bytes": encoded_size(
                json.dumps(duplicate_report, separators=(",", ":"), sort_keys=True)
            ),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark offline candidate-pool regulation on the frozen EXP20 ledger."
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--expected-candidates",
        type=int,
        default=DEFAULT_EXPECTED_CANDIDATES,
        help="Required candidate-row count; use 0 to disable the check.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline-callable",
        help="Baseline regulator in module:attribute notation for paired comparison.",
    )
    parser.add_argument(
        "--candidate-callable",
        help="Candidate regulator in module:attribute notation for paired comparison.",
    )
    parser.add_argument("--compare-pairs", type=int, default=10)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--max-regression-percent", type=float, default=3.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ledger = load_ledger(args.ledger)
    if args.expected_candidates and len(ledger["candidates"]) != args.expected_candidates:
        raise ValueError(
            f"{args.ledger}: expected {args.expected_candidates} candidates, "
            f"found {len(ledger['candidates'])}"
        )

    try:
        source = str(args.ledger.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        source = str(args.ledger.resolve())
    comparison_requested = bool(args.baseline_callable or args.candidate_callable)
    if comparison_requested and not (
        args.baseline_callable and args.candidate_callable
    ):
        raise ValueError(
            "paired comparison requires both --baseline-callable and "
            "--candidate-callable"
        )

    if comparison_requested:
        report = run_paired_comparison(
            ledger["candidates"],
            ledger["references"],
            baseline_regulator=load_callable(args.baseline_callable),
            candidate_regulator=load_callable(args.candidate_callable),
            pairs=args.compare_pairs,
            warmup=args.warmup,
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
            max_regression_percent=args.max_regression_percent,
            cases=len(ledger["cases"]),
            source=source,
        )
        exit_code = 0 if report["acceptance"]["accepted"] else 1
    else:
        report = run_benchmark(
            ledger["candidates"],
            ledger["references"],
            cases=len(ledger["cases"]),
            iterations=args.iterations,
            warmup=args.warmup,
            source=source,
        )
        exit_code = 0
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote candidate-pool benchmark to {args.output}")
    else:
        print(payload, end="")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
