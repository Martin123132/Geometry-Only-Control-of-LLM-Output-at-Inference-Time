#!/usr/bin/env python
"""Benchmark offline regulation and reporting on the frozen 257-row EXP20 ledger."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Iterable

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
    report = run_benchmark(
        ledger["candidates"],
        ledger["references"],
        cases=len(ledger["cases"]),
        iterations=args.iterations,
        warmup=args.warmup,
        source=source,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote candidate-pool benchmark to {args.output}")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
